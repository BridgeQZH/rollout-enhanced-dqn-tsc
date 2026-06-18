"""Structural-identity tests for :class:`IntersectionSpec` (no SUMO required).

These assert that the canonical single-intersection spec packages the legacy
``constants.py`` literals *exactly*, and that the spec's self-validation rejects
malformed structures — the guard rail the Phase 2.2 net-derived path will lean on.
"""

from __future__ import annotations

import numpy as np

from tests._fixtures import _REPO_ROOT  # noqa: F401 - ensures repo root on sys.path

from clean_rollout_tlcs import constants as C
from clean_rollout_tlcs.intersection_spec import IntersectionSpec, build_single_intersection_spec


def test_spec_reproduces_legacy_constants() -> None:
    """The single-intersection spec must equal the legacy constants field by field."""
    spec = build_single_intersection_spec()

    assert spec.tl_id == C.TRAFFIC_LIGHT_ID
    assert spec.num_actions == C.NUM_ACTIONS
    assert spec.state_size == C.STATE_SIZE
    assert spec.lane_id_to_state_index == C.LANE_ID_TO_STATE_INDEX
    assert all(spec.served_lanes[u] == C.SERVED_LANES[u] for u in range(C.NUM_ACTIONS))
    assert tuple(spec.lanes_per_group) == C.LANES_PER_GROUP
    assert np.allclose(spec.lane_queue_capacity, C.LANE_QUEUE_CAPACITY)
    assert tuple(spec.incoming_edges) == C.INCOMING_EDGES
    assert spec.action_to_phase == C.ACTION_TO_TL_PHASE
    assert spec.green_to_yellow == C.TL_GREEN_TO_YELLOW
    assert spec.road_max_length == C.ROAD_MAX_LENGTH
    assert spec.stopline_zone_m == C.STOPLINE_ZONE_M
    assert spec.detector_distance_m == C.DETECTOR_DISTANCE_M


def test_service_matrix_matches_legacy_indicator() -> None:
    """The derived 0/1 service matrix must equal the legacy SERVICE_INDICATOR."""
    spec = build_single_intersection_spec()
    assert spec.service_indicator == C.SERVICE_INDICATOR
    assert spec.service_matrix().shape == (C.NUM_ACTIONS, C.STATE_SIZE)
    # Every served index, and only those, is marked in the matrix.
    matrix = spec.service_matrix()
    for u in range(C.NUM_ACTIONS):
        served = {i for i in range(C.STATE_SIZE) if matrix[u, i] == 1.0}
        assert served == set(C.SERVED_LANES[u])


def test_spec_rejects_malformed_structure() -> None:
    """__post_init__ must raise on inconsistent shapes / out-of-range indices."""
    good = build_single_intersection_spec()

    def rebuild(**overrides: object) -> None:
        kwargs = {
            "tl_id": good.tl_id,
            "num_actions": good.num_actions,
            "state_size": good.state_size,
            "lane_id_to_state_index": dict(good.lane_id_to_state_index),
            "served_lanes": good.served_lanes,
            "lanes_per_group": good.lanes_per_group,
            "lane_queue_capacity": good.lane_queue_capacity,
            "incoming_edges": good.incoming_edges,
            "action_to_phase": dict(good.action_to_phase),
            "green_to_yellow": dict(good.green_to_yellow),
            "road_max_length": good.road_max_length,
            "stopline_zone_m": good.stopline_zone_m,
            "detector_distance_m": good.detector_distance_m,
        }
        kwargs.update(overrides)
        IntersectionSpec(**kwargs)  # type: ignore[arg-type]

    # Wrong number of served-lane entries (one too few for num_actions).
    _assert_raises(ValueError, lambda: rebuild(served_lanes=good.served_lanes[:-1]))
    # lanes_per_group shorter than state_size.
    _assert_raises(ValueError, lambda: rebuild(lanes_per_group=good.lanes_per_group[:-1]))
    # A served index out of range.
    _assert_raises(ValueError, lambda: rebuild(served_lanes=(frozenset({999}),) + good.served_lanes[1:]))
    # A lane mapping out of range.
    _assert_raises(ValueError, lambda: rebuild(lane_id_to_state_index={"N2TL_0": 99}))
    # Missing an action->phase entry.
    _assert_raises(ValueError, lambda: rebuild(action_to_phase={0: 0}))


def _assert_raises(exc: type[BaseException], fn) -> None:  # noqa: ANN001
    """Assert that ``fn()`` raises ``exc``."""
    try:
        fn()
    except exc:
        return
    msg = f"expected {exc.__name__} but no exception was raised"
    raise AssertionError(msg)


_TESTS = (
    test_spec_reproduces_legacy_constants,
    test_service_matrix_matches_legacy_indicator,
    test_spec_rejects_malformed_structure,
)


def main() -> int:
    """Run all structural-identity tests; return non-zero on failure."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - oracle reports, does not raise
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"[test_intersection_spec] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
