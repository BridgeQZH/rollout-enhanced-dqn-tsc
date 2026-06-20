"""Phase 2.2 tests: net-driven spec construction + graceful short-link geometry.

Validates that :func:`build_specs_from_net` turns the generated 2x2 grid into four
internally-consistent, shared-parameter-compatible specs, and that the detector
geometry degrades gracefully when the nominal 636 m distance exceeds the link.
Reads the committed ``grid2x2.net.xml`` via ``sumolib`` only — no SUMO run.
"""

from __future__ import annotations

from pathlib import Path

from tests._fixtures import _REPO_ROOT  # noqa: F401 - ensures repo root on sys.path

from clean_rollout_tlcs.constants import DETECTOR_DISTANCE_M, V_FREE_FLOW_MS
from clean_rollout_tlcs.intersection_spec import IntersectionSpec
from clean_rollout_tlcs.net_parser import DETECTOR_LANE_FRACTION, build_specs_from_net
from clean_rollout_tlcs.session import SumoSession
from clean_rollout_tlcs.junction_env import JunctionEnv

GRID_NET = Path("clean_rollout_tlcs") / "networks" / "grid2x2" / "grid2x2.net.xml"


def _specs() -> dict[str, IntersectionSpec]:
    return build_specs_from_net(GRID_NET)


def test_discovers_four_homogeneous_junctions() -> None:
    """All four grid TLs must be discovered with identical (state, action) dims."""
    specs = _specs()
    assert sorted(specs) == ["A0", "A1", "B0", "B1"]
    dims = {(s.state_size, s.num_actions) for s in specs.values()}
    assert dims == {(16, 4)}, f"non-uniform tensor dims across grid: {dims}"


def test_each_spec_is_valid_and_fully_served() -> None:
    """Every spec self-validates; every lane is served by at least one action."""
    for tid, s in _specs().items():
        # action_to_phase covers all actions; greens map to distinct yellows.
        assert set(s.action_to_phase) == set(range(s.num_actions)), tid
        assert len(set(s.green_to_yellow.values())) == s.num_actions, tid
        # No lane is perpetually unserved (its queue can always discharge).
        union = set().union(*s.served_lanes)
        assert union == set(range(s.state_size)), f"{tid}: unserved lanes {set(range(s.state_size)) - union}"


def test_compass_ordering_groups_lanes_by_approach() -> None:
    """State indices must be contiguous per approach edge (compass-consistent)."""
    s = _specs()["A0"]
    index_to_edge: dict[int, str] = {}
    for lane_id, idx in s.lane_id_to_state_index.items():
        index_to_edge[idx] = lane_id.rsplit("_", 1)[0]  # strip lane suffix -> edge id
    # Walking indices in order, the edge changes in contiguous blocks (no interleave).
    edges_in_order = [index_to_edge[i] for i in range(s.state_size)]
    blocks = [e for i, e in enumerate(edges_in_order) if i == 0 or e != edges_in_order[i - 1]]
    assert len(blocks) == len(set(blocks)), f"approach lanes interleaved: {edges_in_order}"


def test_detector_distance_clamped_to_fit_short_links() -> None:
    """The detector must sit inside the shortest link, with a finite derived tau."""
    for tid, s in _specs().items():
        shortest = min(s.lane_length.values())
        assert s.detector_distance_m < shortest, f"{tid}: detector {s.detector_distance_m} >= link {shortest}"
        assert s.detector_distance_m <= DETECTOR_DISTANCE_M, tid
        assert abs(s.detector_distance_m - shortest * DETECTOR_LANE_FRACTION) < 1e-9, tid
        tau = s.detector_lag_tau(V_FREE_FLOW_MS)
        assert 0.0 < tau < DETECTOR_DISTANCE_M / V_FREE_FLOW_MS, f"{tid}: tau {tau} not gracefully reduced"


def test_arrival_estimator_runs_gracefully_on_short_links() -> None:
    """A JunctionEnv on a grid spec yields finite arrival rates with the short-link tau.

    No SUMO is launched: we inject a synthetic detector history and a clock value,
    then check the lag-windowed estimate is finite, correctly shaped and uses the
    spec-derived (small) tau rather than the nominal 45.8 s that would overshoot.
    """
    spec = _specs()["A0"]
    session = SumoSession(
        sumocfg_file=Path("unused.sumocfg"), gui=False,
        max_steps=600, n_cars_generated=100, turn_chance=0.25,
    )
    junction = JunctionEnv(
        spec=spec, session=session,
        green_duration=10, yellow_duration=4, gamma=0.98, cost_type="linear",
        switch_penalty=0.0, cost_quadratic_kappa=10.0, detector_window=100,
        detector_lag_tau=None,  # force derivation from the clamped detector distance
    )
    # tau derived from the clamped distance, not the nominal single-junction lag.
    assert abs(junction.detector_lag_tau - spec.detector_lag_tau(V_FREE_FLOW_MS)) < 1e-9
    assert junction.detector_lag_tau < 45.0

    # Inject one arrival per step on lane group 0 for the whole window.
    for g in range(spec.state_size):
        junction._arrival_history[g] = [1 if g == 0 else 0] * 200
    session.step = 150

    rates = junction.get_arrival_rates()
    assert rates.shape == (spec.state_size,)
    assert all(r == r for r in rates)  # finite (no NaN)
    assert rates[0] > 0.0, "expected a positive arrival rate on the loaded lane group"


_TESTS = (
    test_discovers_four_homogeneous_junctions,
    test_each_spec_is_valid_and_fully_served,
    test_compass_ordering_groups_lanes_by_approach,
    test_detector_distance_clamped_to_fit_short_links,
    test_arrival_estimator_runs_gracefully_on_short_links,
)


def main() -> int:
    """Run all net-parser tests; return non-zero on failure."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - oracle reports, does not raise
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"[test_net_parser] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
