"""Capture the pre-refactor golden snapshot of single-agent eval metrics.

Run this ONCE against known-good code (here: immediately before the Phase 2.0
spec refactor) to freeze the behavioural baseline::

    python -m tests.capture_golden

It runs the small deterministic benchmark (5 modes x 2 seeds) through the public
eval harness and writes the pinned metrics to ``tests/golden/``. The companion
``test_eval_regression.py`` then asserts the refactored code reproduces it
byte-for-byte. Re-capture only when an intended behavioural change is approved.
"""

from __future__ import annotations

import json

from tests._eval_oracle import run_small_benchmark
from tests._fixtures import GOLDEN_EVAL_FILE


def main() -> int:
    """Capture and persist the golden eval snapshot."""
    print("[capture] running small deterministic benchmark (this launches SUMO headless)...")
    pinned = run_small_benchmark()

    GOLDEN_EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_EVAL_FILE.write_text(json.dumps(pinned, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[capture] wrote {len(pinned)} (seed,mode) rows to {GOLDEN_EVAL_FILE}")
    for key in sorted(pinned):
        row = pinned[key]
        print(f"  {key:<22} wait={row['total_wait_vehsec']:<8} avg_q={row['avg_queue']:.4f} g={row['total_stage_cost_g']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
