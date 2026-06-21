"""Run one 2x2-grid episode under a model-free controller (Phase 2.3 demo).

Drives all four junctions from a single shared SUMO session via the
:class:`MultiAgentRunner` event-driven loop, then prints per-junction and
network metrics. No trained model is required — the ``greedy`` and ``fixed_time``
controllers are model-free, so this is the end-to-end smoke test for the grid
pipeline (net parsing -> OD route generation -> multi-agent control).

Run from the repository root::

    python scripts/run_grid_demo.py --mode greedy --demand 600 --max-steps 1200
    python scripts/run_grid_demo.py --mode greedy --gui          # watch it in SUMO-GUI

A SUMO-GUI window appears only with ``--gui``; otherwise the run is headless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clean_rollout_tlcs.multi_agent import build_grid_runner
from clean_rollout_tlcs.settings import load_settings

DEFAULT_BASE_SETTINGS = Path("models") / "dqn_100ep" / "seed_0" / "training_settings.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run one 2x2-grid episode (model-free controller).")
    p.add_argument("--mode", default="greedy", choices=["greedy", "fixed_time"],
                   help="Model-free controller to drive every junction (default: greedy).")
    p.add_argument("--base-settings", type=Path, default=DEFAULT_BASE_SETTINGS,
                   help="Base settings YAML to derive durations / cost / detector params from.")
    p.add_argument("--demand", type=int, default=600, help="n_cars_generated (default: 600).")
    p.add_argument("--max-steps", type=int, default=1200, help="Episode horizon in steps (default: 1200).")
    p.add_argument("--seed", type=int, default=100000, help="Traffic seed (default: 100000).")
    p.add_argument("--gui", action="store_true", help="Launch SUMO-GUI (otherwise headless).")
    p.add_argument("--delay", type=int, default=80, help="SUMO-GUI step delay in ms (default: 80).")
    return p.parse_args()


def main() -> int:
    """Build the grid runner and run one episode, printing metrics."""
    args = parse_args()
    settings = load_settings(args.base_settings).model_copy(
        update={"max_steps": args.max_steps, "n_cars_generated": args.demand, "gui": args.gui},
    )

    extra = ["--start", "true", "--delay", str(args.delay), "--quit-on-end", "true"] if args.gui else None
    runner = build_grid_runner(settings, mode=args.mode, gui=args.gui, extra_sumo_args=extra)

    print(f"[grid-demo] mode={args.mode} demand={args.demand} max_steps={args.max_steps} "
          f"seed={args.seed} gui={'on' if args.gui else 'off'}")
    metrics = runner.run_episode(args.seed)

    print(f"[grid-demo] DONE  steps={metrics['n_steps']}  junctions={metrics['n_junctions']}")
    print(f"           network total_wait = {metrics['network_total_wait_vehsec']:,} veh-s")
    print(f"           network avg_queue  = {metrics['network_avg_queue']:.2f} veh")
    print(f"           throughput         = {metrics['throughput']}  residual = {metrics['residual_vehicles']}")
    for tl in sorted(metrics["per_junction"]):
        pj = metrics["per_junction"][tl]
        print(f"             {tl}: decisions={pj['n_decisions']:>3}  avg_queue={pj['avg_queue']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
