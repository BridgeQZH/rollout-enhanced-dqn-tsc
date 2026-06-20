"""Per-junction structural description (:class:`IntersectionSpec`).

Phase 2.0 of the multi-agent refactor moves the *structure* of a controlled
intersection — its lane→state mapping, its service indicators, its action→phase
table and its arm geometry — out of module-level constants and into a runtime
value object. The single-agent system is recovered exactly by
:func:`build_single_intersection_spec`, which packages the legacy
``constants.py`` literals into one spec; the multi-agent system (Phase 2.2) will
instead build one spec per traffic light by parsing the ``.net.xml``.

Nothing here touches SUMO/``traci``: a spec is pure data plus the small amount of
derived structure (the 0/1 service matrix, the lag-aware state counter) that the
transition model ``f``, the stage cost ``g`` and the state observation consume.
Keeping it simulator-free is what lets the Phase 2.0 regression oracle prove
byte-identical single-agent behaviour without launching SUMO.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .constants import (
    ACTION_TO_TL_PHASE,
    DETECTOR_DISTANCE_M,
    INCOMING_EDGES,
    LANE_ID_TO_STATE_INDEX,
    LANE_QUEUE_CAPACITY,
    LANES_PER_GROUP,
    NUM_ACTIONS,
    ROAD_MAX_LENGTH,
    SERVED_LANES,
    STATE_SIZE,
    STOPLINE_ZONE_M,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)


@dataclass(frozen=True, eq=False)
class IntersectionSpec:
    """Immutable structural description of a single signalised junction.

    A spec is the complete answer to "what is this intersection?", independent of
    the controller driving it and of the simulator. The transition model ``f``,
    the stage cost ``g`` and the environment's state/detector logic are all
    parameterised by one of these, so swapping the spec (single junction today,
    one-per-traffic-light grid tomorrow) is the only change needed to retarget the
    physics and observation onto a different intersection.

    Attributes:
        tl_id: SUMO traffic-light id this spec actuates.
        num_actions: Number of discrete green-phase actions.
        state_size: Dimension of the lane-count state vector.
        lane_id_to_state_index: Map from incoming SUMO lane id to its state index.
        served_lanes: Per-action frozenset of state indices that receive green.
        lanes_per_group: Physical lane count aggregated into each state index.
        lane_queue_capacity: Per-index storage cap ``C_i`` (veh) used to clip ``f``.
        incoming_edges: Incoming edge ids (queue / waiting-time bookkeeping).
        action_to_phase: Map from action index to SUMO green-phase code.
        green_to_yellow: Map from green-phase code to its matching yellow code.
        road_max_length: Reference incoming-arm length (m); used as the fallback
            stop-line distance reference for any lane absent from ``lane_length``.
        stopline_zone_m: Distance-to-junction (m) within which a vehicle counts
            into the state vector.
        detector_distance_m: Distance-to-junction (m) of the virtual upstream
            loop detector that feeds the arrival-rate estimator. On a grid this is
            clamped by the parser so it always fits inside the shortest link.
        lane_length: Per-incoming-lane length (m). On a uniform single junction
            every entry equals ``road_max_length``; on a grid the lengths differ
            per approach, which is why stop-line distance is computed per lane.
    """

    tl_id: str
    num_actions: int
    state_size: int
    lane_id_to_state_index: Mapping[str, int]
    served_lanes: tuple[frozenset[int], ...]
    lanes_per_group: tuple[int, ...]
    lane_queue_capacity: tuple[float, ...]
    incoming_edges: tuple[str, ...]
    action_to_phase: Mapping[int, int]
    green_to_yellow: Mapping[int, int]
    road_max_length: float
    stopline_zone_m: float
    detector_distance_m: float
    lane_length: Mapping[str, float] = field(default_factory=dict)

    # Cached derived structure (populated in __post_init__; not init args).
    _service_matrix: NDArray = field(default=None, init=False, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate internal consistency and precompute the service matrix.

        Raises:
            ValueError: If any structural invariant is violated (wrong lengths,
                out-of-range indices, missing action→phase entries, etc.).
        """
        if self.num_actions <= 0 or self.state_size <= 0:
            msg = f"num_actions and state_size must be positive; got {self.num_actions}, {self.state_size}"
            raise ValueError(msg)
        if len(self.served_lanes) != self.num_actions:
            msg = f"served_lanes must have num_actions={self.num_actions} entries; got {len(self.served_lanes)}"
            raise ValueError(msg)
        if len(self.lanes_per_group) != self.state_size:
            msg = f"lanes_per_group must have state_size={self.state_size} entries; got {len(self.lanes_per_group)}"
            raise ValueError(msg)
        if len(self.lane_queue_capacity) != self.state_size:
            msg = (
                f"lane_queue_capacity must have state_size={self.state_size} entries; "
                f"got {len(self.lane_queue_capacity)}"
            )
            raise ValueError(msg)

        for u, served in enumerate(self.served_lanes):
            for idx in served:
                if not 0 <= idx < self.state_size:
                    msg = f"served_lanes[{u}] index {idx} out of range [0, {self.state_size})"
                    raise ValueError(msg)
        for lane_id, idx in self.lane_id_to_state_index.items():
            if not 0 <= idx < self.state_size:
                msg = f"lane '{lane_id}' maps to index {idx} out of range [0, {self.state_size})"
                raise ValueError(msg)
        for u in range(self.num_actions):
            if u not in self.action_to_phase:
                msg = f"action_to_phase is missing an entry for action {u}"
                raise ValueError(msg)

        matrix = np.zeros((self.num_actions, self.state_size), dtype=float)
        for u, served in enumerate(self.served_lanes):
            for idx in served:
                matrix[u, idx] = 1.0
        # frozen dataclass: bypass the frozen setattr guard to cache the matrix.
        object.__setattr__(self, "_service_matrix", matrix)

    # ------------------------------------------------------------------ #
    # Derived structure consumed by f, g and the state observation
    # ------------------------------------------------------------------ #
    def service_matrix(self) -> NDArray:
        """Return the ``(num_actions, state_size)`` 0/1 service-indicator matrix.

        Returns:
            A float array where entry ``[u, i] == 1`` iff action ``u`` serves
            (discharges) lane group ``i``.
        """
        return self._service_matrix

    @property
    def service_indicator(self) -> tuple[tuple[int, ...], ...]:
        """Service matrix as a nested int tuple (legacy-compatible view)."""
        return tuple(tuple(int(v) for v in row) for row in self._service_matrix)

    def lane_distance_to_stopline(self, lane_id: str, lane_position: float) -> float:
        """Distance (m) from a vehicle to the stop line on its incoming lane.

        Uses the per-lane length so short grid links and long single-junction arms
        are handled by the same formula.

        Args:
            lane_id: Incoming lane id the vehicle is on.
            lane_position: Vehicle position along the lane from its start (m).

        Returns:
            Metres remaining to the stop line at the junction.
        """
        length = self.lane_length.get(lane_id, self.road_max_length)
        return length - lane_position

    def count_state(self, vehicles: Iterable[tuple[str, float]]) -> NDArray:
        """Count vehicles into the lane-count state vector (pure, SUMO-free).

        Args:
            vehicles: Iterable of ``(lane_id, lane_position)`` pairs, where
                ``lane_position`` is the vehicle's position along its lane (m).

        Returns:
            A float array of shape ``(state_size,)`` counting vehicles that lie on
            a tracked incoming lane within ``stopline_zone_m`` of the junction.
        """
        state = np.zeros(self.state_size, dtype=float)
        for lane_id, lane_position in vehicles:
            group = self.lane_id_to_state_index.get(lane_id)
            if group is None:
                continue
            if self.lane_distance_to_stopline(lane_id, lane_position) <= self.stopline_zone_m:
                state[group] += 1.0
        return state

    def detector_lag_tau(self, v_free: float) -> float:
        """Travel-time lag ``tau = detector_distance / v_free`` for this junction.

        Args:
            v_free: Free-flow speed on the incoming arms (m/s).

        Returns:
            The detector->stop-line travel-time lag (s). Because the parser clamps
            ``detector_distance_m`` to fit the shortest link, this stays finite and
            small on short grid links instead of exceeding the link length.
        """
        return self.detector_distance_m / v_free


def build_single_intersection_spec() -> IntersectionSpec:
    """Build the canonical single-junction spec from the legacy constants.

    This reproduces the exact structure the single-agent system has always used —
    the Vidali 4-arm intersection controlled by traffic light ``TL`` — so that
    threading a spec through ``f``, ``g`` and the state observation is a pure
    decoupling with no behavioural change.

    Returns:
        The :class:`IntersectionSpec` describing the ``TL`` junction.
    """
    return IntersectionSpec(
        tl_id=TRAFFIC_LIGHT_ID,
        num_actions=NUM_ACTIONS,
        state_size=STATE_SIZE,
        lane_id_to_state_index=dict(LANE_ID_TO_STATE_INDEX),
        served_lanes=tuple(SERVED_LANES[u] for u in range(NUM_ACTIONS)),
        lanes_per_group=tuple(LANES_PER_GROUP),
        lane_queue_capacity=tuple(LANE_QUEUE_CAPACITY),
        incoming_edges=tuple(INCOMING_EDGES),
        action_to_phase=dict(ACTION_TO_TL_PHASE),
        green_to_yellow=dict(TL_GREEN_TO_YELLOW),
        road_max_length=ROAD_MAX_LENGTH,
        stopline_zone_m=STOPLINE_ZONE_M,
        detector_distance_m=DETECTOR_DISTANCE_M,
        # Every incoming arm of the canonical junction is the same length, so the
        # per-lane map is uniform and reproduces the legacy scalar geometry.
        lane_length={lane: ROAD_MAX_LENGTH for lane in LANE_ID_TO_STATE_INDEX},
    )
