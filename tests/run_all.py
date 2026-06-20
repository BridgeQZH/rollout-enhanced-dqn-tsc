"""Run the full Phase 2.0 regression oracle.

Usage (from the repository root)::

    python -m tests.run_all              # fast checks + end-to-end SUMO regression
    python -m tests.run_all --fast       # skip the SUMO end-to-end regression

Exit code is non-zero if any test fails, so this doubles as a CI gate.
"""

from __future__ import annotations

import argparse

from tests import test_intersection_spec, test_net_parser, test_single_agent_equivalence

# Fast, SUMO-free modules always run.
_FAST_MODULES = (test_intersection_spec, test_single_agent_equivalence, test_net_parser)


def main() -> int:
    """Run the selected oracle modules and aggregate their exit codes."""
    parser = argparse.ArgumentParser(description="Phase 2.0 single-agent regression oracle.")
    parser.add_argument("--fast", action="store_true", help="Skip the end-to-end SUMO regression.")
    args = parser.parse_args()

    modules = list(_FAST_MODULES)
    if not args.fast:
        from tests import test_eval_regression  # imported lazily so --fast needs no SUMO

        modules.append(test_eval_regression)

    rc = 0
    for module in modules:
        print(f"\n=== {module.__name__} ===")
        rc |= module.main()

    print("\n" + ("ALL TESTS PASSED" if rc == 0 else "SOME TESTS FAILED"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
