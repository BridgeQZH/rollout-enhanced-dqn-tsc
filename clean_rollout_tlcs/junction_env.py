"""A single traffic-light junction (:class:`JunctionEnv`).

Phase 2.1 extracts everything *local to one intersection* out of the monolithic
``Environment`` into this class: the lane-count observation, the canonical stage
cost ``g``, the causal upstream arrival-rate estimator, and the phase actuation.
A :class:`JunctionEnv` is parameterised entirely by an
:class:`~clean_rollout_tlcs.intersection_spec.IntersectionSpec`, so the same code
drives the canonical single junction and any junction discovered in a grid.

It does not own the simulation clock: it holds a back-reference to the shared
:class:`~clean_rollout_tlcs.session.SumoSession` to read the global ``step`` (for
the detector lag window) and delegates world-stepping to an injected
``simulate_fn`` during :meth:`execute`. That injection is what lets the
single-agent facade reuse its (emission-aware) stepping loop unchanged while the
Phase 2.3 multi-agent runner supplies a different one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import traci
from numpy.typing import NDArray

from .constants import V_FREE_FLOW_MS
from .intersection_spec import IntersectionSpec
from .settings import CostType

if TYPE_CHECKING:
    from .session import SumoSession

# Steps the world by ``duration`` and returns one EnvStats per executed step.
SimulateFn = Callable[[int], list["EnvStats"]]


@dataclass
class EnvStats:
    """Per-step environment statistics gathered while simulating a phase.

    Attributes:
        queue_length: Number of halted vehicles on the junction's incoming edges.
    """

    queue_length: int


class JunctionEnv:
    """Local observation, cost, detector and actuation for one traffic light."""

    def __init__(
        self,
        *,
        spec: IntersectionSpec,
        session: SumoSession,
        green_duration: int,
        yellow_duration: int,
        gamma: float,
        cost_type: CostType,
        switch_penalty: float,
        cost_quadratic_kappa: float,
        detector_window: int,
        detector_lag_tau: float | None = None,
        v_free: float = V_FREE_FLOW_MS,
    ) -> None:
        """Initialize the junction wrapper.

        Args:
            spec: Structural description of this junction.
            session: Shared SUMO session (read-only clock access).
            green_duration: Green phase hold time (s).
            yellow_duration: Yellow phase hold time (s) inserted on phase change.
            gamma: Per-second discount used in the discounted stage-cost integral.
            cost_type: Canonical stage cost, ``"linear"`` or ``"quadratic"``.
            switch_penalty: Additive penalty applied once on a phase change.
            cost_quadratic_kappa: Normalizer for the quadratic cost variant.
            detector_window: Sliding-window length (s) for the arrival estimator.
            detector_lag_tau: Detector->stop-line travel-time lag (s). If ``None``
                it is derived from the spec's (possibly clamped) detector distance
                as ``tau = detector_distance / v_free`` — the graceful path for the
                short grid links where the nominal lag would overshoot the link.
            v_free: Free-flow speed (m/s) used to derive ``tau`` when not given.
        """
        self.spec = spec
        self.session = session
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.gamma = gamma
        self.cost_type: CostType = cost_type
        self.switch_penalty = switch_penalty
        self.cost_quadratic_kappa = cost_quadratic_kappa

        self.detector_window = detector_window
        self.detector_lag_tau = (
            detector_lag_tau if detector_lag_tau is not None else spec.detector_lag_tau(v_free)
        )

        # Per-step arrival counts registered by the virtual upstream detector,
        # one list per state index; index t holds detections during step t.
        self._arrival_history: dict[int, list[int]] = {i: [] for i in range(spec.state_size)}
        # Vehicle ids already registered by the detector (count each vehicle once).
        self._detected_ids: set[str] = set()

    def reset(self) -> None:
        """Reset the per-episode detector history buffers."""
        self._arrival_history = {i: [] for i in range(self.spec.state_size)}
        self._detected_ids = set()

    # ------------------------------------------------------------------ #
    # State observation
    # ------------------------------------------------------------------ #
    def get_state(self) -> NDArray:
        """Return the true lane-count state for this junction.

        Each component counts vehicles within ``stopline_zone_m`` of the junction
        on the corresponding lane group.

        Returns:
            A float array of shape ``(spec.state_size,)`` of non-negative counts.
        """
        vehicles = (
            (traci.vehicle.getLaneID(car_id), float(traci.vehicle.getLanePosition(car_id)))
            for car_id in traci.vehicle.getIDList()
        )
        return self.spec.count_state(vehicles)

    # ------------------------------------------------------------------ #
    # Stage cost  g(x, u, u_prev)
    # ------------------------------------------------------------------ #
    def stage_cost(self, state: NDArray, action: int, prev_action: int) -> float:
        """Compute the canonical stage cost ``g(x, u, u_prev)``.

        The cost integrates a constant per-second penalty on the *unserved* lane
        groups over the effective action window, discounted by ``gamma``, plus a
        one-off switching penalty when the phase changes::

            Delta = green + yellow * 1[u != u_prev]
            c     = sum_{i not served by u} x_i           (linear, canonical)
                    (1/kappa) * sum_{i not served} x_i^2   (quadratic, ablation)
            g     = - c * (1 - gamma^Delta) / (1 - gamma)  - P * 1[u != u_prev]

        Args:
            state: Current state ``x``.
            action: Candidate action ``u`` in ``[0, spec.num_actions)``.
            prev_action: Previously applied action ``u_prev``; use ``-1`` when no
                phase has been applied yet (no switch penalty is charged).

        Returns:
            The stage cost ``g`` as a non-positive float.

        Raises:
            ValueError: If ``action`` is outside the valid action range.
        """
        if not 0 <= action < self.spec.num_actions:
            msg = f"action must be in [0, {self.spec.num_actions}); got {action}"
            raise ValueError(msg)

        switched = prev_action != -1 and action != prev_action
        delta = self.green_duration + (self.yellow_duration if switched else 0)

        state_size = self.spec.state_size
        served = self.spec.served_lanes[action]
        if self.cost_type == "linear":
            per_second_cost = float(
                sum(state[i] for i in range(state_size) if i not in served),
            )
        else:  # quadratic fairness variant
            per_second_cost = (
                float(sum(state[i] ** 2 for i in range(state_size) if i not in served))
                / self.cost_quadratic_kappa
            )

        discount_sum = self._discounted_window(delta)
        switch_cost = self.switch_penalty if switched else 0.0

        return -(per_second_cost * discount_sum) - switch_cost

    def _discounted_window(self, delta: int) -> float:
        """Closed-form discounted sum ``sum_{t=0}^{delta-1} gamma^t``.

        Args:
            delta: Number of seconds in the effective action window.

        Returns:
            The geometric sum, or ``delta`` exactly when ``gamma == 1``.
        """
        if self.gamma >= 1.0:
            return float(delta)
        return (1.0 - self.gamma**delta) / (1.0 - self.gamma)

    # ------------------------------------------------------------------ #
    # Causal upstream arrival-rate estimator (virtual loop detector)
    # ------------------------------------------------------------------ #
    def _record_detector(self) -> None:
        """Register new vehicles at the upstream virtual detector for this step.

        A vehicle is counted exactly once, the first time it is observed farther
        than ``spec.detector_distance_m`` from the stop line. The per-group counts
        are appended to the rolling history so :meth:`get_arrival_rates` can apply
        the travel-time lag.
        """
        counts = [0] * self.spec.state_size
        lane_to_index = self.spec.lane_id_to_state_index
        detector_distance = self.spec.detector_distance_m

        for car_id in traci.vehicle.getIDList():
            if car_id in self._detected_ids:
                continue
            lane_id = traci.vehicle.getLaneID(car_id)
            group = lane_to_index.get(lane_id)
            if group is None:
                continue
            dist_to_tl = self.spec.lane_distance_to_stopline(
                lane_id, float(traci.vehicle.getLanePosition(car_id)),
            )
            if dist_to_tl >= detector_distance:
                counts[group] += 1
                self._detected_ids.add(car_id)

        for group in range(self.spec.state_size):
            self._arrival_history[group].append(counts[group])

    def get_arrival_rates(self) -> NDArray:
        """Estimate the lag-corrected stop-line arrival rate per lane group.

        The detector senses vehicles upstream; a vehicle sensed now reaches the
        stop line ``tau`` seconds later. The arrival rate relevant to the stop line
        *now* is therefore the detector rate measured ``tau`` seconds ago, so the
        estimator averages the per-step counts over the lagged window
        ``[step - tau - window, step - tau)``.

        Returns:
            A float array of shape ``(spec.state_size,)`` of arrival rates (veh/s).
            Lane groups with insufficient lagged history return ``0.0``.
        """
        rates = np.zeros(self.spec.state_size, dtype=float)

        lag_steps = int(round(self.detector_lag_tau))
        window_end = self.session.step - lag_steps
        window_start = max(0, window_end - self.detector_window)

        span = window_end - window_start
        if span <= 0:
            return rates  # not enough history behind the lag yet

        for group in range(self.spec.state_size):
            history = self._arrival_history[group]
            # Guard against window_end exceeding the recorded history length.
            end = min(window_end, len(history))
            start = min(window_start, end)
            effective_span = end - start
            if effective_span <= 0:
                continue
            rates[group] = sum(history[start:end]) / effective_span

        return rates

    # ------------------------------------------------------------------ #
    # Per-step observation hook (used by the stepping loop)
    # ------------------------------------------------------------------ #
    def observe_step(self) -> EnvStats:
        """Record the detector and snapshot this junction's queue for one step.

        Returns:
            The per-step statistics for this junction.
        """
        self._record_detector()
        return EnvStats(queue_length=self.get_queue_length())

    # ------------------------------------------------------------------ #
    # Action execution (local; world-stepping is injected)
    # ------------------------------------------------------------------ #
    def execute(
        self,
        action: int,
        simulate_fn: SimulateFn,
        is_over_fn: Callable[[], bool],
    ) -> list[EnvStats]:
        """Apply an action: optional yellow transition, then the green phase.

        If the requested green phase differs from the current one, a yellow phase
        is inserted first. World-stepping is delegated to ``simulate_fn`` so the
        caller controls how the clock advances (and what extra per-step bookkeeping
        happens), while the phase logic stays local to the junction.

        Args:
            action: Discrete action index mapped to a traffic-light phase.
            simulate_fn: Callable advancing the world by N steps and returning the
                per-step stats (capped at the episode horizon by the caller).
            is_over_fn: Predicate reporting whether the episode horizon is reached.

        Returns:
            Per-step statistics gathered while the phases were held.
        """
        next_green_phase = self.spec.action_to_phase[action]
        current_green_phase = traci.trafficlight.getPhase(self.spec.tl_id)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            self.set_yellow_phase(current_green_phase)
            stats.extend(simulate_fn(self.yellow_duration))

        if is_over_fn():
            return stats

        self.set_green_phase(next_green_phase)
        stats.extend(simulate_fn(self.green_duration))

        return stats

    def set_yellow_phase(self, current_green_phase: int) -> None:
        """Switch the traffic light to the yellow matching a green phase.

        Args:
            current_green_phase: Code of the currently active green phase.
        """
        traci.trafficlight.setPhase(self.spec.tl_id, self.spec.green_to_yellow[current_green_phase])

    def set_green_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the given green phase.

        Args:
            green_phase_code: Code of the green phase to activate.
        """
        traci.trafficlight.setPhase(self.spec.tl_id, green_phase_code)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def get_queue_length(self) -> int:
        """Return the number of halted vehicles on this junction's incoming edges.

        Returns:
            Total count of halted vehicles across the incoming edges.
        """
        return int(
            sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.spec.incoming_edges),
        )

    def get_cumulated_waiting_time(self) -> float:
        """Return the summed accumulated waiting time over incoming vehicles.

        Returns:
            Total accumulated waiting time (s) of all vehicles on incoming edges.
        """
        total = 0.0
        incoming_edges = self.spec.incoming_edges
        for car_id in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(car_id) in incoming_edges:
                total += float(traci.vehicle.getAccumulatedWaitingTime(car_id))
        return total
