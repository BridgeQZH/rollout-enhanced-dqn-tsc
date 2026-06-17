"""Pydantic-validated configuration for the rollout-enhanced TLCS project.

A single :class:`Settings` model drives every entry point: DQN training, the five
benchmark control modes, and the verification/calibration tooling. Validation is
strict — malformed configs fail loudly at load time rather than midway through a
multi-hour run.

Locked design decisions encoded here:

* reward alignment   -> the DQN is trained on ``reward = g`` (negative queue
  cost), so the tail value ``H`` is commensurate with the rollout stage cost;
* rollout weight      -> ``beta = 1.0`` by default;
* canonical cost      -> ``cost_type = "linear"`` (unserved-queue delay);
* main rollout depth  -> ``lookahead_depth = 1``;
* causal sensing      -> upstream virtual detector with travel-time lag ``tau``;
* benchmark size      -> ``n_eval_seeds = 30`` paired traffic seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from .constants import (
    DEFAULT_DETECTOR_LAG_TAU,
    DEFAULT_DETECTOR_WINDOW_S,
    DEFAULT_N_EVAL_SEEDS,
    DEFAULT_SATURATION_FLOW,
    DEFAULT_STARTUP_LOST_TIME,
)

# Control mode selected for an evaluation run (and used to gate which artifacts
# a run requires, e.g. a trained model or the arrival-rate estimator).
ControlMode = Literal["dqn", "fixed_time", "greedy", "rollout_1s", "rollout_ms"]

# Canonical (primary) cost is the linear unserved-queue delay; quadratic is the
# fairness-oriented ablation.
CostType = Literal["linear", "quadratic"]

# Modes whose decision rule queries a trained DQN (directly or as a tail value).
_MODES_NEEDING_MODEL: frozenset[str] = frozenset({"dqn", "rollout_1s", "rollout_ms"})

# Modes whose decision rule rolls the transition model f forward.
_ROLLOUT_MODES: frozenset[str] = frozenset({"rollout_1s", "rollout_ms"})


class Settings(BaseModel):
    """Validated configuration for training, evaluation and rollout control.

    Core simulation / network / memory fields are required; the research-specific
    cost, rollout, detector and evaluation fields carry sensible defaults so that
    a minimal YAML still produces a fully specified, runnable configuration.
    """

    # --- control mode -----------------------------------------------------
    mode: ControlMode = "dqn"

    # --- simulation -------------------------------------------------------
    gui: bool = False
    total_episodes: PositiveInt
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    green_duration: PositiveInt
    yellow_duration: PositiveInt
    turn_chance: Annotated[float, Field(ge=0, le=1)]

    # --- model (MLP) ------------------------------------------------------
    num_layers: PositiveInt
    width_layers: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    training_epochs: PositiveInt

    # --- replay memory ----------------------------------------------------
    memory_size_min: NonNegativeInt
    memory_size_max: PositiveInt

    # --- agent ------------------------------------------------------------
    gamma: Annotated[float, Field(ge=0, le=1)]

    # --- stage cost g -----------------------------------------------------
    cost_type: CostType = "linear"
    # Per-second cost normalizer for the quadratic variant (ignored when linear).
    cost_quadratic_kappa: PositiveFloat = 10.0
    # Fixed additive penalty applied once whenever the phase changes.
    switch_penalty: NonNegativeFloat = 0.0

    # --- rollout ----------------------------------------------------------
    # Look-ahead depth for the multi-step controller; the one-step controller
    # always uses depth 1 regardless of this value.
    lookahead_depth: PositiveInt = 1
    # Weight on the discounted DQN tail value H in the rollout score q-tilde.
    beta: NonNegativeFloat = 1.0
    # Horizon (s) used to truncate the greedy base-policy tail in multi-step.
    rollout_truncation: PositiveInt = 60

    # --- transition model f (physics; calibrated in Phase 2) --------------
    saturation_flow: PositiveFloat = DEFAULT_SATURATION_FLOW
    startup_lost_time: NonNegativeFloat = DEFAULT_STARTUP_LOST_TIME

    # --- causal upstream detector ----------------------------------------
    # Sliding-window length (s) for the arrival-rate average.
    detector_window: PositiveInt = DEFAULT_DETECTOR_WINDOW_S
    # Detector -> stop-line travel-time lag (s); τ = d / v_free by default.
    detector_lag_tau: NonNegativeFloat = DEFAULT_DETECTOR_LAG_TAU

    # --- evaluation harness ----------------------------------------------
    n_eval_seeds: PositiveInt = DEFAULT_N_EVAL_SEEDS
    eval_seed_start: NonNegativeInt = 0

    # --- paths ------------------------------------------------------------
    sumocfg_file: Path

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _check_memory_bounds(self) -> Self:
        """Ensure the replay warmup threshold is below the capacity cap."""
        if self.memory_size_min >= self.memory_size_max:
            msg = (
                f"memory_size_min ({self.memory_size_min}) must be smaller than "
                f"memory_size_max ({self.memory_size_max})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_multistep_depth(self) -> Self:
        """A multi-step rollout must look ahead at least two action steps."""
        if self.mode == "rollout_ms" and self.lookahead_depth < 2:  # noqa: PLR2004
            msg = (
                "mode 'rollout_ms' requires lookahead_depth >= 2 "
                f"(got {self.lookahead_depth}); use mode 'rollout_1s' for depth 1"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_eval_window(self) -> Self:
        """The detector window must fit inside the episode horizon."""
        if self.detector_window > self.max_steps:
            msg = (
                f"detector_window ({self.detector_window}) cannot exceed "
                f"max_steps ({self.max_steps})"
            )
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------ #
    # Convenience accessors used to gate which artifacts a run requires
    # ------------------------------------------------------------------ #
    @property
    def needs_trained_model(self) -> bool:
        """Whether the selected mode queries a trained DQN."""
        return self.mode in _MODES_NEEDING_MODEL

    @property
    def needs_arrival_model(self) -> bool:
        """Whether the selected mode rolls the transition model f forward."""
        return self.mode in _ROLLOUT_MODES

    @property
    def is_rollout(self) -> bool:
        """Whether the selected mode is one of the rollout controllers."""
        return self.mode in _ROLLOUT_MODES

    @property
    def effective_lookahead_depth(self) -> int:
        """Look-ahead depth actually used by the selected control mode."""
        if self.mode == "rollout_1s":
            return 1
        if self.mode == "rollout_ms":
            return self.lookahead_depth
        return 0

    def eval_seeds(self) -> list[int]:
        """Return the deterministic list of evaluation traffic seeds.

        Every benchmark mode is replayed across this exact list so that all five
        controllers face identical traffic, enabling paired statistical tests.

        Returns:
            A list of ``n_eval_seeds`` consecutive integer seeds.
        """
        return list(range(self.eval_seed_start, self.eval_seed_start + self.n_eval_seeds))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the top-level YAML node is not a mapping.
    """
    if not path.exists():
        msg = f"Settings file not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        msg = f"Invalid YAML format in {path}; expected a mapping at the top level"
        raise TypeError(msg)

    return data


def load_settings(settings_file: Path) -> Settings:
    """Load and validate a :class:`Settings` instance from a YAML file.

    Args:
        settings_file: Path to the settings YAML file.

    Returns:
        A validated :class:`Settings` instance.
    """
    return Settings.model_validate(load_yaml(settings_file))
