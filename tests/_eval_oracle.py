"""Shared machinery for the end-to-end single-agent eval regression oracle.

Both the golden *capture* and the *verify* path run the very same small benchmark
through the public evaluation harness and extract the same deterministic columns,
so a refactor that preserves behaviour yields a byte-identical comparison.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from clean_rollout_tlcs.eval import load_model_from_dir, run_benchmark

from tests._fixtures import GOLDEN_MODEL_DIR, small_eval_settings

# Columns whose values are fully determined by (model, route file, policy) and
# therefore must match exactly across an equivalence-preserving refactor.
_PINNED_COLUMNS: tuple[str, ...] = (
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


def run_small_benchmark() -> dict[str, dict[str, Any]]:
    """Run the small deterministic benchmark and return pinned metrics.

    Returns:
        A mapping ``"<seed>|<mode>" -> {pinned column: value}`` over all five
        control modes and both eval seeds.
    """
    settings = small_eval_settings()
    model, model_run = load_model_from_dir(GOLDEN_MODEL_DIR, settings)

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "oracle.csv"
        rows = run_benchmark(settings, model, model_run, out_csv)

    pinned: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['seed']}|{row['mode']}"
        pinned[key] = {col: row.get(col) for col in _PINNED_COLUMNS}
    return pinned


def diff_against_golden(
    golden: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    *,
    float_atol: float = 1e-9,
) -> list[str]:
    """Compare current pinned metrics to the golden snapshot.

    Integers, strings and booleans are compared exactly; floats are compared with
    a tight absolute tolerance to absorb only benign last-bit reassociation.

    Args:
        golden: The committed golden metrics.
        current: Freshly computed metrics.
        float_atol: Absolute tolerance for float comparisons.

    Returns:
        A list of human-readable mismatch descriptions; empty means identical.
    """
    diffs: list[str] = []

    missing = set(golden) - set(current)
    extra = set(current) - set(golden)
    for key in sorted(missing):
        diffs.append(f"[{key}] present in golden but missing from current run")
    for key in sorted(extra):
        diffs.append(f"[{key}] present in current run but missing from golden")

    for key in sorted(set(golden) & set(current)):
        g_row, c_row = golden[key], current[key]
        for col in _PINNED_COLUMNS:
            g_val, c_val = g_row.get(col), c_row.get(col)
            if isinstance(g_val, float) and isinstance(c_val, (int, float)):
                if g_val != g_val and c_val != c_val:  # both NaN -> equal
                    continue
                if abs(float(c_val) - g_val) > float_atol:
                    diffs.append(f"[{key}] {col}: golden={g_val!r} current={c_val!r} (atol={float_atol})")
            elif g_val != c_val:
                diffs.append(f"[{key}] {col}: golden={g_val!r} current={c_val!r}")

    return diffs
