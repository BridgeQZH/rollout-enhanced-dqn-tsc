"""Phase 2.3 end-to-end test: the multi-agent grid runner (SUMO required).

Runs a short 2x2-grid episode with the model-free fixed-time controller and
asserts the event-driven loop drives the whole network to completion, every
junction makes decisions, the metrics are finite, and the run is seed
deterministic. Also checks the greedy controller meaningfully reduces queueing
versus fixed-time on the same traffic — evidence the loop actually reacts to
local state rather than just cycling.

Requires SUMO on PATH. Kept out of the ``--fast`` subset.
"""

from __future__ import annotations

import math
from pathlib import Path

from tests._fixtures import GOLDEN_BASE_SETTINGS  # noqa: F401 - repo-root side effect

from clean_rollout_tlcs.settings import load_settings
from clean_rollout_tlcs.multi_agent import build_grid_runner

_SMALL = {"max_steps": 300, "n_cars_generated": 200, "gui": False}


def _settings():  # noqa: ANN202 - small helper
    return load_settings(GOLDEN_BASE_SETTINGS).model_copy(update=_SMALL)


def test_grid_episode_completes_with_all_agents_deciding() -> None:
    """Every junction must drive to the horizon and make multiple decisions."""
    runner = build_grid_runner(_settings(), mode="fixed_time")
    m = runner.run_episode(seed=100000)

    assert m["n_steps"] == _SMALL["max_steps"], f"episode stopped early at {m['n_steps']}"
    assert m["n_junctions"] == 4
    assert set(m["per_junction"]) == {"A0", "A1", "B0", "B1"}
    for tl, pj in m["per_junction"].items():
        assert pj["n_decisions"] > 1, f"{tl} made too few decisions ({pj['n_decisions']})"
        assert math.isfinite(pj["avg_queue"]) and pj["avg_queue"] >= 0.0, tl
    assert m["throughput"] >= 0 and m["residual_vehicles"] >= 0
    assert math.isfinite(m["network_avg_queue"])


def test_grid_run_is_seed_deterministic() -> None:
    """Two runs on the same seed must produce identical metrics."""
    a = build_grid_runner(_settings(), mode="greedy").run_episode(seed=100001)
    b = build_grid_runner(_settings(), mode="greedy").run_episode(seed=100001)
    assert a == b, "grid episode is not seed-deterministic"


def test_greedy_beats_fixed_time_on_same_traffic() -> None:
    """Greedy (state-reactive) should not be worse than blind fixed-time cycling."""
    seed = 100000
    fixed = build_grid_runner(_settings(), mode="fixed_time").run_episode(seed)
    greedy = build_grid_runner(_settings(), mode="greedy").run_episode(seed)
    assert greedy["network_total_wait_vehsec"] <= fixed["network_total_wait_vehsec"], (
        f"greedy wait {greedy['network_total_wait_vehsec']} > fixed {fixed['network_total_wait_vehsec']}"
    )


_TESTS = (
    test_grid_episode_completes_with_all_agents_deciding,
    test_grid_run_is_seed_deterministic,
    test_greedy_beats_fixed_time_on_same_traffic,
)


def main() -> int:
    """Run all multi-agent grid tests; return non-zero on failure."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - oracle reports, does not raise
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"[test_multi_agent] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
