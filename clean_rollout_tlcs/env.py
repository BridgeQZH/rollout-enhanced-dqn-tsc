"""Single-agent environment facade (:class:`Environment`).

Phase 2.1 split the old monolith into two collaborators:

* :class:`~clean_rollout_tlcs.session.SumoSession` — the TraCI lifecycle, the
  global clock and route generation (the *one global thing*);
* :class:`~clean_rollout_tlcs.junction_env.JunctionEnv` — one junction's
  observation, stage cost ``g``, arrival-rate estimator and phase actuation (the
  *many local things*).

:class:`Environment` is now a thin **facade** that composes exactly one session
and one junction and exposes the original public API unchanged, so the
single-agent ``train.py`` / ``eval.py`` / GUI demo and the Phase 2.0 regression
oracle keep working byte-for-byte. The multi-agent runner (Phase 2.3) will instead
compose one session with *many* junctions, bypassing this facade.

``EnvStats`` is re-exported here for backward compatibility with importers that
still do ``from .env import Environment, EnvStats``.
"""

from __future__ import annotations

from pathlib import Path

from .intersection_spec import IntersectionSpec, build_single_intersection_spec
from .junction_env import EnvStats, JunctionEnv
from .session import SumoSession
from .settings import CostType, Settings

__all__ = ["EnvStats", "Environment"]


class Environment:
    """Single-agent facade composing one :class:`SumoSession` + one junction."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_cars_generated: int,
        max_steps: int,
        green_duration: int,
        yellow_duration: int,
        turn_chance: float,
        sumocfg_file: Path,
        gui: bool,
        gamma: float,
        cost_type: CostType,
        switch_penalty: float,
        cost_quadratic_kappa: float,
        detector_window: int,
        detector_lag_tau: float,
        spec: IntersectionSpec | None = None,
    ) -> None:
        """Initialize the facade and its session + junction collaborators.

        Args:
            n_cars_generated: Number of vehicles to spawn per episode.
            max_steps: Maximum number of simulation steps (seconds) per episode.
            green_duration: Green phase hold time (s).
            yellow_duration: Yellow phase hold time (s) inserted on phase change.
            turn_chance: Probability a generated vehicle takes a turning route.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to launch the SUMO GUI binary.
            gamma: Discount factor used in the discounted stage-cost integral.
            cost_type: Canonical stage cost, ``"linear"`` or ``"quadratic"``.
            switch_penalty: Additive penalty applied once on a phase change.
            cost_quadratic_kappa: Normalizer for the quadratic cost variant.
            detector_window: Sliding-window length (s) for the arrival estimator.
            detector_lag_tau: Detector->stop-line travel-time lag (s).
            spec: Structural description of the controlled junction; defaults to
                the canonical single intersection.
        """
        spec = spec if spec is not None else build_single_intersection_spec()

        # Mirror construction-time scalars for read-compatibility with callers.
        self.n_cars_generated = n_cars_generated
        self.turn_chance = turn_chance
        self.sumocfg_file = sumocfg_file
        self.gui = gui
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.gamma = gamma
        self.cost_type: CostType = cost_type
        self.switch_penalty = switch_penalty
        self.cost_quadratic_kappa = cost_quadratic_kappa
        self.detector_window = detector_window
        self.detector_lag_tau = detector_lag_tau
        self.spec = spec

        self.session = SumoSession(
            sumocfg_file=sumocfg_file,
            gui=gui,
            max_steps=max_steps,
            n_cars_generated=n_cars_generated,
            turn_chance=turn_chance,
        )
        self.junction = JunctionEnv(
            spec=spec,
            session=self.session,
            green_duration=green_duration,
            yellow_duration=yellow_duration,
            gamma=gamma,
            cost_type=cost_type,
            switch_penalty=switch_penalty,
            cost_quadratic_kappa=cost_quadratic_kappa,
            detector_window=detector_window,
            detector_lag_tau=detector_lag_tau,
        )

    @classmethod
    def from_settings(cls, settings: Settings, spec: IntersectionSpec | None = None) -> Environment:
        """Build an :class:`Environment` from a validated :class:`Settings`.

        Args:
            settings: Validated configuration.
            spec: Junction structure; defaults to the canonical single intersection.

        Returns:
            A configured environment instance.
        """
        return cls(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
            turn_chance=settings.turn_chance,
            sumocfg_file=settings.sumocfg_file,
            gui=settings.gui,
            gamma=settings.gamma,
            cost_type=settings.cost_type,
            switch_penalty=settings.switch_penalty,
            cost_quadratic_kappa=settings.cost_quadratic_kappa,
            detector_window=settings.detector_window,
            detector_lag_tau=settings.detector_lag_tau,
            spec=spec,
        )

    # ------------------------------------------------------------------ #
    # Live clock views (owned by the session)
    # ------------------------------------------------------------------ #
    @property
    def step(self) -> int:
        """Current global simulation step (owned by the session)."""
        return self.session.step

    @property
    def max_steps(self) -> int:
        """Episode horizon in steps (owned by the session)."""
        return self.session.max_steps

    # ------------------------------------------------------------------ #
    # Episode lifecycle (delegated to the session)
    # ------------------------------------------------------------------ #
    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line (delegated to the session)."""
        return self.session.build_sumo_cmd()

    def reset(self) -> None:
        """Reset the global clock and the junction's detector buffers."""
        self.session.reset()
        self.junction.reset()

    def activate(self) -> None:
        """Reset episode state and start the SUMO simulation."""
        self.reset()
        self.session.start(self.build_sumo_cmd())

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        self.session.close()

    def is_over(self) -> bool:
        """Return whether the episode has reached ``max_steps``."""
        return self.session.is_over()

    def generate_routefile(self, seed: int) -> None:
        """Generate the per-episode route file (delegated to the session).

        Args:
            seed: Traffic seed.
        """
        self.session.generate_routefile(seed)

    # ------------------------------------------------------------------ #
    # Local observation / cost / detector / metrics (delegated to junction)
    # ------------------------------------------------------------------ #
    def get_state(self):  # noqa: ANN201 - returns NDArray, kept light for the facade
        """Return the junction's lane-count state."""
        return self.junction.get_state()

    def stage_cost(self, state, action: int, prev_action: int) -> float:  # noqa: ANN001
        """Compute the junction's canonical stage cost ``g``."""
        return self.junction.stage_cost(state, action, prev_action)

    def get_arrival_rates(self):  # noqa: ANN201 - returns NDArray
        """Return the junction's lag-corrected arrival-rate estimate."""
        return self.junction.get_arrival_rates()

    def get_queue_length(self) -> int:
        """Return the junction's halted-vehicle count on incoming edges."""
        return self.junction.get_queue_length()

    def get_cumulated_waiting_time(self) -> float:
        """Return the junction's summed accumulated waiting time."""
        return self.junction.get_cumulated_waiting_time()

    # ------------------------------------------------------------------ #
    # Action execution (junction-local logic; stepping injected from here)
    # ------------------------------------------------------------------ #
    def execute(self, action: int) -> list[EnvStats]:
        """Apply an action, advancing the world via this facade's stepping loop.

        Args:
            action: Discrete action index mapped to a traffic-light phase.

        Returns:
            Per-step statistics gathered while the phases were held.
        """
        return self.junction.execute(action, self._simulate, self.session.is_over)

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the world by up to ``duration`` steps, sensing each step.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            One :class:`EnvStats` per executed simulation step.
        """
        stats: list[EnvStats] = []
        for _ in range(self.session.steps_remaining(duration)):
            self.session.tick()
            stats.append(self.junction.observe_step())
        return stats
