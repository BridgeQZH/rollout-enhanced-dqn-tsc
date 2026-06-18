"""End-to-end byte-identical eval regression vs the captured golden (SUMO).

Re-runs the same small deterministic benchmark the golden was captured from and
asserts every pinned metric matches the committed snapshot. This is the strongest
Phase 2.0 guarantee: it exercises the full closed loop (SUMO + state observation +
cost + detector + rollout) and proves the spec refactor changed no behaviour.

Requires SUMO on PATH (``sumo``) and the golden file produced by
``python -m tests.capture_golden``. If the golden is missing, the test reports a
clear skip rather than a failure.
"""

from __future__ import annotations

import json

from tests._eval_oracle import diff_against_golden, run_small_benchmark
from tests._fixtures import GOLDEN_EVAL_FILE


def test_eval_metrics_match_golden() -> None:
    """Refactored eval must reproduce the golden metrics byte-for-byte."""
    if not GOLDEN_EVAL_FILE.exists():
        msg = (
            f"golden snapshot not found at {GOLDEN_EVAL_FILE}; "
            "run `python -m tests.capture_golden` against known-good code first"
        )
        raise FileNotFoundError(msg)

    golden = json.loads(GOLDEN_EVAL_FILE.read_text(encoding="utf-8"))
    current = run_small_benchmark()

    diffs = diff_against_golden(golden, current)
    if diffs:
        joined = "\n    ".join(diffs)
        msg = f"{len(diffs)} metric(s) diverged from golden:\n    {joined}"
        raise AssertionError(msg)


_TESTS = (test_eval_metrics_match_golden,)


def main() -> int:
    """Run the end-to-end regression test; return non-zero on failure."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - oracle reports, does not raise
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"[test_eval_regression] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
