"""Build :class:`IntersectionSpec` objects from a SUMO ``.net.xml`` (Phase 2.2).

This is the net-driven replacement for the hand-written single-intersection
constants: point it at any network and it discovers every traffic light and
constructs one spec per junction by reading the actual controlled lanes,
geometry and signal program via ``sumolib``. The single intersection still uses
:func:`~clean_rollout_tlcs.intersection_spec.build_single_intersection_spec`
(its legacy constants are the regression oracle); this parser is for the new
multi-junction networks.

Three design choices make the result usable for *shared-parameter* MARL on a
homogeneous grid:

* **Compass-consistent lane ordering.** Incoming lanes are ordered by the compass
  bearing of their approach (N, E, S, W) and then by lane index. State index ``k``
  therefore means the same movement type at every junction, which is what lets one
  shared policy read every junction's state coherently.
* **Actions from protected greens.** Each signal phase that has a protected green
  (``G``) and no yellow (``y``) becomes one action; ``served`` lanes are exactly
  those whose controlled link shows ``G`` in that phase.
* **Geometry that fits the link.** Per-lane lengths are recorded, and the upstream
  detector distance is clamped to sit inside the *shortest* incoming lane so the
  arrival estimator degrades gracefully when the nominal 636 m exceeds the link.
"""

from __future__ import annotations

import math
from pathlib import Path

import sumolib

from .constants import (
    DETECTOR_DISTANCE_M,
    STOPLINE_ZONE_M,
    VEHICLE_FOOTPRINT_M,
)
from .intersection_spec import IntersectionSpec

# Fraction of the shortest incoming lane at which to place the virtual detector
# when the nominal distance does not fit. 0.8 keeps it upstream of the stop-line
# zone while leaving a margin before the lane's upstream end.
DETECTOR_LANE_FRACTION = 0.8

# Compass buckets in canonical order; an approach is assigned to the nearest one.
_COMPASS_ORDER: tuple[tuple[str, float], ...] = (
    ("N", 90.0),
    ("E", 0.0),
    ("S", 270.0),
    ("W", 180.0),
)


def _approach_compass_key(edge: "sumolib.net.edge.Edge") -> int:
    """Return the canonical compass order index (N=0,E=1,S=2,W=3) of an approach.

    The approach direction is the bearing *from the junction toward the upstream
    end* of the incoming edge, i.e. the side traffic arrives from.

    Args:
        edge: The incoming edge.

    Returns:
        Index into :data:`_COMPASS_ORDER` of the nearest compass direction.
    """
    fx, fy = edge.getFromNode().getCoord()
    tx, ty = edge.getToNode().getCoord()
    bearing = math.degrees(math.atan2(fy - ty, fx - tx)) % 360.0
    # Nearest compass direction by circular distance.
    best_idx, best_dist = 0, 360.0
    for idx, (_name, ref) in enumerate(_COMPASS_ORDER):
        dist = abs((bearing - ref + 180.0) % 360.0 - 180.0)
        if dist < best_dist:
            best_idx, best_dist = idx, dist
    return best_idx


def _ordered_incoming_lanes(tls: "sumolib.net.TLS") -> list["sumolib.net.lane.Lane"]:
    """Return the TL's incoming lanes in compass-consistent order.

    Args:
        tls: The traffic-light logic object.

    Returns:
        Unique incoming lanes sorted by (approach compass, lane index).
    """
    lanes: dict[str, "sumolib.net.lane.Lane"] = {}
    for in_lane, _out_lane, _link_index in tls.getConnections():
        lanes[in_lane.getID()] = in_lane
    return sorted(
        lanes.values(),
        key=lambda ln: (_approach_compass_key(ln.getEdge()), ln.getIndex()),
    )


def _green_phase_indices(phases: list) -> list[int]:  # noqa: ANN001 - sumolib Phase
    """Return indices of protected-green action phases (have ``G``, no ``y``).

    Args:
        phases: The signal program's phase list.

    Returns:
        Phase indices that represent controllable green actions, in order.
    """
    return [i for i, ph in enumerate(phases) if "G" in ph.state and "y" not in ph.state]


def _matching_yellow_index(phases: list, green_index: int) -> int:  # noqa: ANN001
    """Return the yellow phase that follows a given green phase.

    Args:
        phases: The signal program's phase list.
        green_index: Index of the green phase.

    Returns:
        Index of the following phase if it contains yellow, else the green index
        itself (no transition phase defined).
    """
    nxt = (green_index + 1) % len(phases)
    return nxt if "y" in phases[nxt].state else green_index


def build_spec_for_tls(
    net: "sumolib.net.Net",
    tls: "sumolib.net.TLS",
    *,
    stopline_zone_m: float = STOPLINE_ZONE_M,
    nominal_detector_m: float = DETECTOR_DISTANCE_M,
) -> IntersectionSpec:
    """Build one :class:`IntersectionSpec` for a single traffic light.

    Args:
        net: The parsed network (unused directly but kept for symmetry / future
            cross-link lookups).
        tls: The traffic-light logic to describe.
        stopline_zone_m: Stop-line counting zone (m).
        nominal_detector_m: Nominal upstream detector distance (m) before clamping.

    Returns:
        A fully populated, self-validated spec for this junction.
    """
    ordered_lanes = _ordered_incoming_lanes(tls)
    lane_ids = [ln.getID() for ln in ordered_lanes]
    lane_index = {lid: i for i, lid in enumerate(lane_ids)}
    state_size = len(lane_ids)

    lane_length = {ln.getID(): float(ln.getLength()) for ln in ordered_lanes}
    incoming_edges = tuple(dict.fromkeys(ln.getEdge().getID() for ln in ordered_lanes))

    # Map each incoming lane to the link indices it controls (for served lookup).
    lane_links: dict[str, list[int]] = {lid: [] for lid in lane_ids}
    for in_lane, _out_lane, link_index in tls.getConnections():
        lane_links[in_lane.getID()].append(link_index)

    phases = list(next(iter(tls.getPrograms().values())).getPhases())
    green_indices = _green_phase_indices(phases)
    num_actions = len(green_indices)

    action_to_phase = {action: phase_idx for action, phase_idx in enumerate(green_indices)}
    green_to_yellow = {
        phase_idx: _matching_yellow_index(phases, phase_idx) for phase_idx in green_indices
    }

    served_lanes: list[frozenset[int]] = []
    for phase_idx in green_indices:
        state = phases[phase_idx].state
        served = {
            lane_index[lid]
            for lid in lane_ids
            if any(state[li] == "G" for li in lane_links[lid])
        }
        served_lanes.append(frozenset(served))

    # Per-lane capacity uses the effective zone (a short lane caps its own queue).
    lane_queue_capacity = tuple(
        min(stopline_zone_m, lane_length[lid]) / VEHICLE_FOOTPRINT_M for lid in lane_ids
    )

    # Clamp the detector so it always fits inside the shortest incoming lane.
    shortest = min(lane_length.values())
    detector_distance_m = min(nominal_detector_m, shortest * DETECTOR_LANE_FRACTION)

    return IntersectionSpec(
        tl_id=tls.getID(),
        num_actions=num_actions,
        state_size=state_size,
        lane_id_to_state_index=lane_index,
        served_lanes=tuple(served_lanes),
        lanes_per_group=tuple(1 for _ in lane_ids),
        lane_queue_capacity=lane_queue_capacity,
        incoming_edges=incoming_edges,
        action_to_phase=action_to_phase,
        green_to_yellow=green_to_yellow,
        road_max_length=max(lane_length.values()),
        stopline_zone_m=stopline_zone_m,
        detector_distance_m=detector_distance_m,
        lane_length=lane_length,
    )


def build_specs_from_net(
    netfile: Path | str,
    *,
    stopline_zone_m: float = STOPLINE_ZONE_M,
    nominal_detector_m: float = DETECTOR_DISTANCE_M,
) -> dict[str, IntersectionSpec]:
    """Discover every traffic light in a network and build one spec per junction.

    Args:
        netfile: Path to the SUMO ``.net.xml`` file.
        stopline_zone_m: Stop-line counting zone (m).
        nominal_detector_m: Nominal upstream detector distance (m) before clamping.

    Returns:
        A mapping ``tl_id -> IntersectionSpec`` for all traffic lights, ordered by
        traffic-light id for determinism.

    Raises:
        FileNotFoundError: If the network file does not exist.
        ValueError: If the network contains no traffic lights.
    """
    netfile = Path(netfile)
    if not netfile.exists():
        msg = f"Network file not found: {netfile}"
        raise FileNotFoundError(msg)

    net = sumolib.net.readNet(str(netfile), withPrograms=True)
    tls_list = sorted(net.getTrafficLights(), key=lambda t: t.getID())
    if not tls_list:
        msg = f"No traffic lights found in {netfile}"
        raise ValueError(msg)

    return {
        tls.getID(): build_spec_for_tls(
            net,
            tls,
            stopline_zone_m=stopline_zone_m,
            nominal_detector_m=nominal_detector_m,
        )
        for tls in tls_list
    }
