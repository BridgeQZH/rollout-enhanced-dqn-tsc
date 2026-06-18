"""Cross-policy evaluation harness (Phase 3).

Loads a trained DQN checkpoint and benchmarks all five control modes
(``dqn``, ``fixed_time``, ``greedy``, ``rollout_1s``, ``rollout_ms``) over the
exact same set of evaluation traffic seeds, producing a tidy paired-metrics CSV
for downstream confidence intervals and figures.

Paired-comparison guarantee: for each seed the route file is generated **once**
and all five controllers replay that identical traffic; the route-file hash is
asserted unchanged across the five runs. Because every mode faces the same
traffic per seed, the rows can be analysed with paired statistics.

Per (seed, mode) metrics:

* ``total_wait_vehsec`` — cumulative delay (sum of queue length over all steps);
* ``avg_queue`` — mean halted vehicles per step;
* ``throughput`` — vehicles that completed their trip during the episode;
* ``residual_vehicles`` — vehicles left unfinished at episode end (still running or
  not yet inserted); reported alongside delay to guard against a "win" that is
  really demand-shedding under heavy congestion;
* ``total_fuel`` / ``total_co2`` — summed edge emissions (if SUMO provides them);
* ``total_stage_cost_g`` — summed canonical stage cost ``g`` over decisions.

For the two rollout modes it additionally reports the agreement between the
transition model ``f`` and the realized next state, pooled over all decision
steps and lanes: ``f_mae`` (mean absolute error) and ``f_spearman`` (Spearman
rank correlation).

Run as a module from a directory containing the SUMO ``intersection/`` assets::

    python -m clean_rollout_tlcs.eval --model-dir models/dqn_100ep/seed_0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import traci
from numpy.typing import NDArray

from .agent import DQNAgent
from .constants import MODEL_FILE, NUM_ACTIONS, ROUTES_FILE, STATE_SIZE
from .env import Environment, EnvStats
from .model import Model
from .settings import ControlMode, Settings, load_settings
from .transition import TransitionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_rollout_tlcs.eval")

# Order in which the five control modes are evaluated for each seed.
EVAL_MODES: tuple[ControlMode, ...] = ("dqn", "fixed_time", "greedy", "rollout_1s", "rollout_ms")

# Modes that roll the transition model f forward (and thus report f residuals).
ROLLOUT_MODES: frozenset[str] = frozenset({"rollout_1s", "rollout_ms"})

# Fallback multi-step depth when settings.lookahead_depth is the one-step value.
DEFAULT_MULTISTEP_DEPTH = 3

# Default output location requested for the paired benchmark file.
DEFAULT_RESULTS_CSV = Path("clean_rollout_tlcs/eval_results/paired_benchmarks.csv")

CSV_COLUMNS: tuple[str, ...] = (
    "model_run",
    "seed",
    "mode",
    "route_hash",
    "n_decisions",
    "total_wait_vehsec",
    "avg_queue",
    "throughput",
    "residual_vehicles",
    "total_stage_cost_g",
    "total_fuel",
    "total_co2",
    "emissions_available",
    "f_mae",
    "f_spearman",
)


# --------------------------------------------------------------------------- #
# Emission-aware environment (keeps env.py untouched)
# --------------------------------------------------------------------------- #
class EvalEnvironment(Environment):
    """Environment that also accumulates per-step fuel/CO2, throughput and residual.

    Overrides :meth:`Environment._simulate` to read edge-level emissions each
    simulation step; if the SUMO build does not expose them, it degrades
    gracefully and flags ``emissions_available = False``. It additionally tracks
    cumulative throughput (vehicles that completed their trip) and the residual
    count of unfinished vehicles at episode end.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base environment and the per-episode accumulators."""
        super().__init__(**kwargs)
        self.fuel_total: float = 0.0
        self.co2_total: float = 0.0
        self.emissions_available: bool = True
        # Throughput (vehicles arrived at destination) and the residual count of
        # unfinished vehicles; used to detect a delay "win" bought by shedding.
        self.arrived_total: int = 0
        self.residual_vehicles: int = 0

    def reset(self) -> None:
        """Reset base episode state and the per-episode accumulators."""
        super().reset()
        self.fuel_total = 0.0
        self.co2_total = 0.0
        self.emissions_available = True
        self.arrived_total = 0
        self.residual_vehicles = 0

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance SUMO, sensing queues, detector arrivals and emissions.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            One :class:`EnvStats` per executed simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            self._record_detector()
            stats.append(EnvStats(queue_length=self.get_queue_length()))
            self._accumulate_emissions()
            # Throughput accrues as vehicles reach their destination; the residual
            # (still running + waiting to be inserted) is overwritten each step so
            # the final value reflects what was left unfinished at episode end.
            self.arrived_total += int(traci.simulation.getArrivedNumber())
            self.residual_vehicles = int(traci.simulation.getMinExpectedNumber())

        return stats

    def _accumulate_emissions(self) -> None:
        """Add this step's edge-level fuel and CO2 to the running totals."""
        if not self.emissions_available:
            return
        try:
            for edge in self.spec.incoming_edges:
                self.fuel_total += float(traci.edge.getFuelConsumption(edge))
                self.co2_total += float(traci.edge.getCO2Emission(edge))
        except (traci.TraCIException, AttributeError):  # pragma: no cover - build-dependent
            self.emissions_available = False


# --------------------------------------------------------------------------- #
# Rank-correlation helpers (self-contained; no scipy dependency)
# --------------------------------------------------------------------------- #
def _rankdata(values: NDArray) -> NDArray:
    """Assign average ranks to data, handling ties (scipy-compatible).

    Args:
        values: 1-D array of values.

    Returns:
        Array of average ranks (1-based).
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(n, dtype=int)
    inv[sorter] = np.arange(n)
    arr_sorted = arr[sorter]
    is_new = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = is_new.cumsum()[inv]
    boundaries = np.r_[np.nonzero(is_new)[0], n]
    return 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the Spearman rank correlation between two sequences.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        The Spearman correlation, or ``nan`` if undefined (too few points or a
        constant input).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size != x.size:  # noqa: PLR2004
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# Single (seed, mode) evaluation episode
# --------------------------------------------------------------------------- #
def run_eval_episode(
    env: EvalEnvironment,
    agent: DQNAgent,
    mode: ControlMode,
    multistep_depth: int,
) -> dict[str, Any]:
    """Run one evaluation episode under a single control mode.

    The route file must already be generated; this function does not regenerate
    it, preserving the paired-comparison guarantee.

    Args:
        env: Emission-aware environment (route file already generated).
        agent: Agent holding the trained model, transition model and cost fn.
        mode: Control mode to evaluate.
        multistep_depth: Look-ahead depth used when ``mode == "rollout_ms"``.

    Returns:
        A dictionary of per-episode metrics for this mode.
    """
    is_rollout = mode in ROLLOUT_MODES

    env.activate()
    agent.reset_episode_state()

    prev_action = -1
    queue_sum = 0
    step_count = 0
    total_g = 0.0
    n_decisions = 0
    pred_pool: list[float] = []
    actual_pool: list[float] = []

    while not env.is_over():
        state = env.get_state()
        arrival_rates = env.get_arrival_rates() if is_rollout else None

        action = _select_action(agent, mode, state, prev_action, arrival_rates, multistep_depth)

        total_g += env.stage_cost(state, action, prev_action)
        n_decisions += 1

        stats = env.execute(action)
        next_state = env.get_state()

        if is_rollout:
            signed = agent.log_residual(next_state)
            if signed is not None:
                predicted = signed + next_state  # pred = (pred - actual) + actual
                pred_pool.extend(predicted.tolist())
                actual_pool.extend(next_state.tolist())

        for stat in stats:
            queue_sum += stat.queue_length
            step_count += 1
        prev_action = action

    env.deactivate()

    result: dict[str, Any] = {
        "n_decisions": n_decisions,
        "total_wait_vehsec": queue_sum,
        "avg_queue": queue_sum / max(step_count, 1),
        "throughput": env.arrived_total,
        "residual_vehicles": env.residual_vehicles,
        "total_stage_cost_g": total_g,
        "total_fuel": env.fuel_total,
        "total_co2": env.co2_total,
        "emissions_available": env.emissions_available,
        "f_mae": None,
        "f_spearman": None,
    }

    if is_rollout and pred_pool:
        result["f_mae"] = float(agent.residual_summary()["mae"])
        result["f_spearman"] = spearman(pred_pool, actual_pool)

    return result


def _select_action(
    agent: DQNAgent,
    mode: ControlMode,
    state: NDArray,
    prev_action: int,
    arrival_rates: NDArray | None,
    multistep_depth: int,
) -> int:
    """Select an action for the given mode by calling the matching control path.

    Args:
        agent: The agent providing the control paths.
        mode: Control mode.
        state: Current state.
        prev_action: Previously applied action (``-1`` if none).
        arrival_rates: Arrival-rate estimate for rollout modes (else ``None``).
        multistep_depth: Depth for the multi-step rollout.

    Returns:
        The selected action index.

    Raises:
        ValueError: If a rollout mode is requested without arrival rates.
        RuntimeError: For an unknown mode (unreachable given the Literal type).
    """
    if mode == "dqn":
        return agent.choose_action(state)
    if mode == "fixed_time":
        return agent.choose_action_fixed_time()
    if mode == "greedy":
        return agent.greedy_base(state, prev_action)
    if arrival_rates is None:
        msg = f"mode '{mode}' requires arrival_rates"
        raise ValueError(msg)
    if mode == "rollout_1s":
        return agent.select_rollout_1s(state, prev_action, arrival_rates)
    if mode == "rollout_ms":
        return agent.select_rollout_ms(state, prev_action, arrival_rates, depth=multistep_depth)
    msg = f"Unknown control mode: {mode}"  # pragma: no cover
    raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def _route_hash() -> str:
    """Return a short hash of the current route file (paired-traffic check)."""
    return hashlib.md5(ROUTES_FILE.read_bytes()).hexdigest()[:12]  # noqa: S324 - not security


def run_benchmark(
    settings: Settings,
    model: Model,
    model_run: str,
    out_csv: Path,
) -> list[dict[str, Any]]:
    """Evaluate all five modes over all eval seeds and write the paired CSV.

    Args:
        settings: Validated configuration (provides eval seeds and parameters).
        model: The trained Q-network (shared as the rollout tail value).
        model_run: Identifier for the model run (stored in each row).
        out_csv: Destination CSV path.

    Returns:
        The list of result rows (one per seed-mode pair).
    """
    seeds = settings.eval_seeds()
    multistep_depth = (
        settings.lookahead_depth
        if settings.lookahead_depth >= 2  # noqa: PLR2004
        else DEFAULT_MULTISTEP_DEPTH
    )
    logger.info(
        "Benchmarking %d modes x %d seeds (seeds %d..%d); rollout_ms depth=%d",
        len(EVAL_MODES),
        len(seeds),
        seeds[0],
        seeds[-1],
        multistep_depth,
    )

    env = EvalEnvironment.from_settings(settings)
    transition = TransitionModel.from_settings(settings)

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        env.generate_routefile(seed)
        route_hash = _route_hash()

        for mode in EVAL_MODES:
            # Fresh agent per (seed, mode): clean residual/fixed-time state,
            # shared trained model so the tail value H is identical everywhere.
            agent = DQNAgent(
                settings=settings,
                model=model,
                transition=transition,
                cost_fn=env.stage_cost,
                epsilon=0.0,
            )
            metrics = run_eval_episode(env, agent, mode, multistep_depth)

            if _route_hash() != route_hash:  # pragma: no cover - safety assertion
                msg = f"Route file changed during seed {seed} (mode {mode}); pairing broken."
                raise RuntimeError(msg)

            row = {"model_run": model_run, "seed": seed, "mode": mode, "route_hash": route_hash}
            row.update(metrics)
            rows.append(row)

            logger.info(
                "seed=%d mode=%-11s wait=%-8d avg_q=%6.2f g=%10.1f%s",
                seed,
                mode,
                int(metrics["total_wait_vehsec"]),
                metrics["avg_queue"],
                metrics["total_stage_cost_g"],
                ""
                if metrics["f_mae"] is None
                else f"  f_MAE={metrics['f_mae']:.3f} f_rho={metrics['f_spearman']:.3f}",
            )

    _write_csv(rows, out_csv)
    _write_manifest(settings, model_run, seeds, multistep_depth, out_csv)
    _log_summary(rows)
    return rows


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    """Write result rows to a tidy CSV with a fixed column order.

    Args:
        rows: Result rows.
        out_csv: Destination path (its parent is created if needed).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})
    logger.info("Wrote %d rows to %s", len(rows), out_csv)


def _write_manifest(
    settings: Settings,
    model_run: str,
    seeds: list[int],
    multistep_depth: int,
    out_csv: Path,
) -> None:
    """Write a JSON manifest describing the evaluation run for reproducibility.

    Args:
        settings: Configuration used.
        model_run: Model run identifier.
        seeds: Evaluation seeds used.
        multistep_depth: Depth used for the multi-step rollout.
        out_csv: The CSV path (the manifest is written alongside it).
    """
    manifest = {
        "model_run": model_run,
        "modes": list(EVAL_MODES),
        "seeds": seeds,
        "n_eval_seeds": settings.n_eval_seeds,
        "eval_seed_start": settings.eval_seed_start,
        "multistep_depth": multistep_depth,
        "cost_type": settings.cost_type,
        "gamma": settings.gamma,
        "beta": settings.beta,
        "detector_lag_tau": settings.detector_lag_tau,
        "saturation_flow": settings.saturation_flow,
        "startup_lost_time": settings.startup_lost_time,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    manifest_path = out_csv.with_name("eval_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote run manifest to %s", manifest_path)


def _log_summary(rows: list[dict[str, Any]]) -> None:
    """Log the per-mode mean of the headline metrics across seeds.

    Args:
        rows: Result rows.
    """
    logger.info("--- Per-mode means across seeds ---")
    for mode in EVAL_MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue
        mean_wait = float(np.mean([r["total_wait_vehsec"] for r in mode_rows]))
        mean_q = float(np.mean([r["avg_queue"] for r in mode_rows]))
        mean_g = float(np.mean([r["total_stage_cost_g"] for r in mode_rows]))
        logger.info(
            "%-11s | mean wait=%9.1f | mean avg_q=%6.2f | mean g=%10.1f",
            mode,
            mean_wait,
            mean_q,
            mean_g,
        )


# --------------------------------------------------------------------------- #
# Checkpoint loading + CLI
# --------------------------------------------------------------------------- #
def load_model_from_dir(model_dir: Path, settings: Settings) -> tuple[Model, str]:
    """Load the trained checkpoint from a run directory.

    Args:
        model_dir: Directory containing ``trained_model.pt``.
        settings: Configuration (provides the learning rate for the optimizer).

    Returns:
        A tuple ``(model, model_run)`` where ``model_run`` identifies the run.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    checkpoint = model_dir / MODEL_FILE
    if not checkpoint.exists():
        msg = f"No trained checkpoint at {checkpoint}. Train first (python -m clean_rollout_tlcs.train)."
        raise FileNotFoundError(msg)

    model = Model.load_checkpoint(checkpoint, learning_rate=settings.learning_rate)
    if model.input_dim != STATE_SIZE or model.output_dim != NUM_ACTIONS:
        msg = (
            f"Checkpoint dims ({model.input_dim}->{model.output_dim}) do not match "
            f"the expected {STATE_SIZE}->{NUM_ACTIONS}."
        )
        raise ValueError(msg)
    logger.info("Loaded model from %s on device %s", checkpoint, model.device)
    return model, model_dir.name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Cross-policy evaluation harness for rollout-TLCS.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("clean_rollout_tlcs") / "settings" / "training_settings.yaml",
        help="Path to the settings YAML.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models") / "dqn_100ep" / "seed_0",
        help="Run directory containing trained_model.pt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Destination CSV for paired benchmark metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load the model and benchmark all five control modes."""
    args = parse_args()
    settings = load_settings(args.settings)
    model, model_run = load_model_from_dir(args.model_dir, settings)
    run_benchmark(settings, model, model_run, args.out)


if __name__ == "__main__":
    main()
