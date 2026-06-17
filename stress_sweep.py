"""Phase 2.1 demand-sweep orchestrator for the high-congestion stress test.

This script is an **orchestration architect**, not a runner. It never starts
SUMO and never invokes the evaluation loop itself. Instead it:

1. derives one validated settings YAML per demand level ``D`` from a frozen base
   config, overriding only ``n_cars_generated`` (and optionally ``max_steps``);
2. lays out the structured output tree under ``eval_results_stress/D_<D>/``;
3. prints the EXACT, copy-pasteable ``python -m clean_rollout_tlcs.eval``
   commands you run yourself, one per (demand level x model run); and
4. writes a ``sweep_plan.json`` manifest so the whole sweep is reproducible.

Why per-demand YAMLs?  The frozen eval CLI exposes only
``--settings / --model-dir / --out`` -- demand lives inside the settings file as
``n_cars_generated`` (consumed by ``Environment`` route generation). Sweeping
demand therefore means handing the harness a different, fully-valid settings
file per level; this script generates those files from your base config so the
physics / detector / seed parameters stay byte-for-byte identical and only the
demand changes.

Pairing guarantee preserved: every demand level reuses the SAME eval seed list
(``eval_seed_start`` / ``n_eval_seeds`` inherited from the base), so within a
level all five controllers face one route file per seed (the harness already
asserts the route hash is unchanged). Across levels the route files differ -- by
design, because demand differs -- so all paired statistics are computed
*within* a demand level by :mod:`analyze_stress_results`.

Run from the repository root (the directory that contains ``intersection/``)::

    python stress_sweep.py                       # default sweep, prints commands
    python stress_sweep.py --demands 1000 1250 1500 1750 2000
    python stress_sweep.py --print-only          # don't write any prep files

Truncation-bias note (see blueprint_2.0.html, Gate 2.1): by default ``max_steps``
is inherited unchanged so the "exam duration" is constant across levels and only
vehicle *density* rises. Pass ``--max-steps`` to extend the horizon at high
demand if you prefer to measure full delay rather than delay-before-the-bell.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Defaults (grounded in the frozen modules)
# ---------------------------------------------------------------------------

# Demand grid swept for the congestion stress test (vehicles per episode).
DEFAULT_DEMANDS: tuple[int, ...] = (1000, 1500, 2000)

# Frozen base config to derive each demand variant from. The long-horizon
# (500-episode) run snapshot matches the converged models in models/dqn_500ep/.
DEFAULT_BASE_SETTINGS = Path("models") / "dqn_500ep" / "seed_0" / "training_settings.yaml"

# Root holding the converged checkpoints to benchmark (one subdir per seed).
DEFAULT_MODEL_ROOT = Path("models") / "dqn_500ep"

# Output root for the whole sweep.
DEFAULT_OUT_ROOT = Path("eval_results_stress")

# The frozen evaluation entry point (invoked by YOU via the printed commands).
EVAL_MODULE = "clean_rollout_tlcs.eval"

# Filename the eval harness expects inside each run directory.
MODEL_FILE = "trained_model.pt"

# Columns the frozen eval harness (clean_rollout_tlcs/eval.py) actually emits.
# NOTE: throughput and residual_vehicles are intentionally listed as MISSING --
# see the banner printed by main(); analyze_stress_results.py parses them only
# if a (future) harness extension adds them.
FROZEN_CSV_METRICS: tuple[str, ...] = (
    "total_wait_vehsec",
    "avg_queue",
    "total_stage_cost_g",
    "total_fuel",
    "total_co2",
    "f_mae",
    "f_spearman",
)
METRICS_MISSING_FROM_FROZEN_SCHEMA: tuple[str, ...] = (
    "throughput",
    "residual_vehicles",
)


# ---------------------------------------------------------------------------
# Discovery and config derivation
# ---------------------------------------------------------------------------
def discover_model_runs(model_root: Path) -> list[Path]:
    """Find run directories under ``model_root`` that hold a trained checkpoint.

    Args:
        model_root: Directory containing per-seed run subdirectories.

    Returns:
        Sorted list of run directories each containing ``trained_model.pt``.

    Raises:
        FileNotFoundError: If ``model_root`` does not exist or holds no runs.
    """
    if not model_root.exists():
        msg = f"Model root not found: {model_root}"
        raise FileNotFoundError(msg)

    runs = sorted(
        child for child in model_root.iterdir() if (child / MODEL_FILE).is_file()
    )
    if not runs:
        msg = (
            f"No '{MODEL_FILE}' found in any subdirectory of {model_root}. "
            "Train the long-horizon models first, or point --model-root elsewhere."
        )
        raise FileNotFoundError(msg)
    return runs


def load_base_config(path: Path) -> dict[str, Any]:
    """Load the base settings YAML as a plain dictionary.

    Args:
        path: Path to the base settings YAML.

    Returns:
        Parsed mapping of settings.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the top-level YAML node is not a mapping.
    """
    if not path.exists():
        msg = f"Base settings file not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Invalid YAML in {path}: expected a mapping at the top level"
        raise TypeError(msg)
    return data


def derive_demand_config(
    base_cfg: dict[str, Any],
    demand: int,
    max_steps_override: int | None,
) -> dict[str, Any]:
    """Return a copy of ``base_cfg`` with the demand (and maybe horizon) changed.

    Only ``n_cars_generated`` is altered by default; every other parameter --
    physics, detector, gamma, eval seeds -- is inherited verbatim so results stay
    directly comparable to the 1000-car baseline.

    Args:
        base_cfg: Parsed base settings mapping.
        demand: Vehicles per episode for this level.
        max_steps_override: Optional new episode horizon (s); ``None`` keeps base.

    Returns:
        A new settings mapping for this demand level.
    """
    cfg = dict(base_cfg)
    cfg["n_cars_generated"] = int(demand)
    if max_steps_override is not None:
        cfg["max_steps"] = int(max_steps_override)
    return cfg


def validate_config(cfg: dict[str, Any]) -> str | None:
    """Best-effort validation of a derived config against the frozen schema.

    Imports :class:`clean_rollout_tlcs.settings.Settings` if available and runs
    pydantic validation. The package ``__init__`` is traci-free, so this import
    does not require SUMO. Validation is advisory: if the package cannot be
    imported (e.g. run from outside the repo root), the harness will still
    validate the YAML when you run it.

    Args:
        cfg: Derived settings mapping.

    Returns:
        ``None`` if valid or validation was skipped; an error string otherwise.
    """
    try:
        from clean_rollout_tlcs.settings import Settings  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - validation is advisory only
        return None
    try:
        Settings.model_validate(cfg)
    except Exception as exc:  # noqa: BLE001 - surface pydantic message verbatim
        return str(exc)
    return None


def write_demand_settings(cfg: dict[str, Any], demand: int, level_dir: Path) -> Path:
    """Write the derived settings YAML for a demand level.

    Args:
        cfg: Derived settings mapping.
        demand: Vehicles per episode (used in the filename).
        level_dir: ``eval_results_stress/D_<demand>`` directory.

    Returns:
        Path to the written settings YAML.
    """
    level_dir.mkdir(parents=True, exist_ok=True)
    settings_path = level_dir / f"settings_D{demand}.yaml"
    header = (
        f"# AUTO-GENERATED by stress_sweep.py for demand level D={demand}.\n"
        f"# Derived from the frozen base config; only n_cars_generated"
        f"{' and max_steps' if 'max_steps' in cfg else ''} changed.\n"
        f"# Do not hand-edit -- regenerate from the base config instead.\n"
    )
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    settings_path.write_text(header + body, encoding="utf-8")
    return settings_path


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def build_eval_command(
    python: str,
    settings_path: Path,
    model_dir: Path,
    out_csv: Path,
) -> str:
    """Build one copy-pasteable eval command string.

    Forward slashes are used for cross-shell portability (PowerShell and POSIX
    both accept them).

    Args:
        python: Python executable token (e.g. ``"python"`` or a venv path).
        settings_path: Derived per-demand settings YAML.
        model_dir: Run directory containing ``trained_model.pt``.
        out_csv: Destination CSV for this (demand, model run).

    Returns:
        A single shell command string.
    """
    return (
        f"{python} -m {EVAL_MODULE} "
        f"--settings {settings_path.as_posix()} "
        f"--model-dir {model_dir.as_posix()} "
        f"--out {out_csv.as_posix()}"
    )


def out_csv_for(out_root: Path, demand: int, model_run: str) -> Path:
    """Return the destination CSV path for a (demand, model run) cell.

    Each model run gets its own subdirectory so the per-run ``eval_manifest.json``
    written by the harness (alongside the CSV) does not collide.

    Args:
        out_root: Sweep output root.
        demand: Vehicles per episode.
        model_run: Model run identifier (the run directory name).

    Returns:
        Path to ``eval_results_stress/D_<demand>/<model_run>/paired_benchmarks.csv``.
    """
    return out_root / f"D_{demand}" / model_run / "paired_benchmarks.csv"


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------
def print_schema_notice() -> None:
    """Print the throughput / residual-vehicle schema gap prominently."""
    bar = "=" * 78
    print(bar)
    print("METRIC SCHEMA NOTICE  (blueprint_2.0.html, Gate 2.1)")
    print(bar)
    print(
        "The FROZEN eval harness (clean_rollout_tlcs/eval.py) emits these per-run\n"
        "metrics:\n"
        f"    {', '.join(FROZEN_CSV_METRICS)}\n\n"
        "It does NOT currently emit the anti-truncation-bias metrics:\n"
        f"    {', '.join(METRICS_MISSING_FROM_FROZEN_SCHEMA)}\n\n"
        "Throughput (vehicles cleared) and residual/undeparted vehicles require\n"
        "traci calls (simulation.getArrivedNumber / getMinExpectedNumber) inside\n"
        "the episode loop, so they cannot be recovered post-hoc from the CSV.\n"
        "Until the harness is extended to log them, the throughput/residual panels\n"
        "in analyze_stress_results.py stay empty (it degrades gracefully). The\n"
        "delay trend analysis (Delta(D), Wilcoxon, Jonckheere-Terpstra) is fully\n"
        "functional on the existing schema."
    )
    print(bar)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plan (do NOT run) the Phase 2.1 congestion demand sweep: derive "
            "per-demand settings, lay out the output tree, and print the exact "
            "eval commands to run by hand."
        ),
    )
    parser.add_argument(
        "--base-settings",
        type=Path,
        default=DEFAULT_BASE_SETTINGS,
        help=f"Base settings YAML to derive demand variants from (default: {DEFAULT_BASE_SETTINGS}).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"Root holding per-seed run dirs with {MODEL_FILE} (default: {DEFAULT_MODEL_ROOT}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        default=None,
        help="Explicit run directory (repeatable). Overrides --model-root discovery.",
    )
    parser.add_argument(
        "--demands",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEMANDS),
        help=f"Demand levels (vehicles/episode) to sweep (default: {' '.join(map(str, DEFAULT_DEMANDS))}).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output root for the sweep (default: {DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional episode-horizon override applied to every level (default: inherit base).",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable token used in the printed commands (default: 'python').",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print commands; do not write settings YAMLs, directories, or the plan manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Plan the demand sweep and print the eval commands. Never runs SUMO.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)

    base_cfg = load_base_config(args.base_settings)
    model_runs = (
        list(args.model_dir)
        if args.model_dir
        else discover_model_runs(args.model_root)
    )

    print_schema_notice()

    plan: dict[str, Any] = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "base_settings": args.base_settings.as_posix(),
        "demands": list(args.demands),
        "max_steps_override": args.max_steps,
        "eval_seed_start": base_cfg.get("eval_seed_start"),
        "n_eval_seeds": base_cfg.get("n_eval_seeds"),
        "model_runs": [m.as_posix() for m in model_runs],
        "out_root": args.out_root.as_posix(),
        "frozen_csv_metrics": list(FROZEN_CSV_METRICS),
        "metrics_missing_from_frozen_schema": list(METRICS_MISSING_FROM_FROZEN_SCHEMA),
        "levels": [],
        "commands": [],
    }

    print(f"Base config      : {args.base_settings.as_posix()}")
    print(f"Demand levels    : {', '.join(str(d) for d in args.demands)}")
    print(f"Model runs       : {', '.join(m.name for m in model_runs)}")
    print(f"Eval seeds       : {base_cfg.get('n_eval_seeds')} starting at {base_cfg.get('eval_seed_start')}")
    print(f"Output root      : {args.out_root.as_posix()}")
    if args.max_steps is not None:
        print(f"max_steps        : OVERRIDDEN -> {args.max_steps}")
    else:
        print(f"max_steps        : inherited ({base_cfg.get('max_steps')}; density-only sweep)")
    print(f"Write prep files : {'NO (--print-only)' if args.print_only else 'YES'}")
    print()

    for demand in args.demands:
        level_dir = args.out_root / f"D_{demand}"
        cfg = derive_demand_config(base_cfg, demand, args.max_steps)

        problem = validate_config(cfg)
        if problem is not None:
            print(f"[validation] D={demand}: derived config FAILED validation:\n  {problem}", file=sys.stderr)
            return 2

        if args.print_only:
            settings_path = level_dir / f"settings_D{demand}.yaml"
        else:
            settings_path = write_demand_settings(cfg, demand, level_dir)

        level_record: dict[str, Any] = {
            "demand": demand,
            "settings_path": settings_path.as_posix(),
            "commands": [],
        }

        print(f"# ---- D = {demand} cars " + "-" * 50)
        for model_dir in model_runs:
            out_csv = out_csv_for(args.out_root, demand, model_dir.name)
            cmd = build_eval_command(args.python, settings_path, model_dir, out_csv)
            print(cmd)
            level_record["commands"].append(cmd)
            plan["commands"].append(cmd)
        print()

        plan["levels"].append(level_record)

    if not args.print_only:
        args.out_root.mkdir(parents=True, exist_ok=True)
        plan_path = args.out_root / "sweep_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Wrote plan manifest -> {plan_path.as_posix()}")
        print(
            "Wrote per-demand settings under "
            f"{args.out_root.as_posix()}/D_<demand>/settings_D<demand>.yaml"
        )

    print()
    print("Next steps:")
    print("  1. Run the commands above from the repo root (the dir containing intersection/).")
    print("  2. Then analyze:  python analyze_stress_results.py --results-root "
          f"{args.out_root.as_posix()}")
    print()
    print("This script started no SUMO instances and ran no evaluations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
