"""SUMO/traci environment wrapper for the rollout-enhanced TLCS agent.

The :class:`Environment` owns everything that touches the simulator:

* episode lifecycle (route generation, ``traci`` start/stop, stepping);
* the **12-dimensional lane-count state** ``get_state`` used by every controller;
* the canonical **stage cost** ``stage_cost`` (== ``g(x, u, u_prev)``), which is
  also the per-step reward signal the DQN is trained on (``reward = g``);
* the **causal upstream arrival-rate estimator** — a virtual loop detector placed
  ~636 m from the stop line, with a travel-time lag correction ``tau = d / v_free``
  so the rate that drives the stop-line prediction reflects what was sensed
  ``tau`` seconds ago.

The transition model ``f`` and the rollout controllers consume the state and the
arrival rates exposed here, but live in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from .constants import (
    DEFAULT_DEPART_SPEED,
    ROUTES_FILE,
    ROUTES_FILE_HEADER,
    STRAIGHT_ROUTES,
    TURN_ROUTES,
    WEIBULL_SHAPE,
)
from .intersection_spec import IntersectionSpec, build_single_intersection_spec
from .settings import CostType, Settings


@dataclass
class EnvStats:
    """Per-step environment statistics gathered while simulating a phase.

    Attributes:
        queue_length: Number of halted vehicles on all incoming edges at this step.
    """

    queue_length: int


class Environment:
    """Reinforcement-learning environment around a SUMO traffic simulation."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_cars_generated: int,
        max_steps: int,
        green_duration: int,
        yellow_duration: int,
        turn_chance: float,
        sumocfg_file: Path,
        gui: bool,
        gamma: float,
        cost_type: CostType,
        switch_penalty: float,
        cost_quadratic_kappa: float,
        detector_window: int,
        detector_lag_tau: float,
        spec: IntersectionSpec | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            n_cars_generated: Number of vehicles to spawn per episode.
            max_steps: Maximum number of simulation steps (seconds) per episode.
            green_duration: Green phase hold time (s).
            yellow_duration: Yellow phase hold time (s) inserted on phase change.
            turn_chance: Probability a generated vehicle takes a turning route.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to launch the SUMO GUI binary.
            gamma: Discount factor used in the discounted stage-cost integral.
            cost_type: Canonical stage cost, ``"linear"`` or ``"quadratic"``.
            switch_penalty: Additive penalty applied once on a phase change.
            cost_quadratic_kappa: Normalizer for the quadratic cost variant.
            detector_window: Sliding-window length (s) for the arrival estimator.
            detector_lag_tau: Detector->stop-line travel-time lag (s).
            spec: Structural description of the controlled junction; defaults to
                the canonical single intersection.
        """
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.turn_chance = turn_chance
        self.sumocfg_file = sumocfg_file
        self.gui = gui

        self.gamma = gamma
        self.cost_type: CostType = cost_type
        self.switch_penalty = switch_penalty
        self.cost_quadratic_kappa = cost_quadratic_kappa

        self.detector_window = detector_window
        self.detector_lag_tau = detector_lag_tau

        self.spec = spec if spec is not None else build_single_intersection_spec()

        self.step = 0

        # Per-step arrival counts registered by the virtual upstream detector,
        # one list per state index; index t holds detections during step t.
        self._arrival_history: dict[int, list[int]] = {i: [] for i in range(self.spec.state_size)}
        # Vehicle ids already registered by the detector (count each vehicle once).
        self._detected_ids: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings, spec: IntersectionSpec | None = None) -> Environment:
        """Build an :class:`Environment` from a validated :class:`Settings`.

        Args:
            settings: Validated configuration.
            spec: Junction structure; defaults to the canonical single intersection.

        Returns:
            A configured environment instance.
        """
        return cls(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
            turn_chance=settings.turn_chance,
            sumocfg_file=settings.sumocfg_file,
            gui=settings.gui,
            gamma=settings.gamma,
            cost_type=settings.cost_type,
            switch_penalty=settings.switch_penalty,
            cost_quadratic_kappa=settings.cost_quadratic_kappa,
            detector_window=settings.detector_window,
            detector_lag_tau=settings.detector_lag_tau,
            spec=spec,
        )

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line from the configuration.

        Returns:
            List of command-line arguments to start SUMO.

        Raises:
            FileNotFoundError: If the SUMO config file does not exist.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        return [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]

    def reset(self) -> None:
        """Reset per-episode counters and the detector history buffers."""
        self.step = 0
        self._arrival_history = {i: [] for i in range(self.spec.state_size)}
        self._detected_ids = set()

    def activate(self) -> None:
        """Reset episode state and start the SUMO simulation."""
        self.reset()
        traci.start(self.build_sumo_cmd())

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Return whether the episode has reached ``max_steps``."""
        return self.step >= self.max_steps

    # ------------------------------------------------------------------ #
    # Route generation (causal "exam paper": fixed per seed, never read back)
    # ------------------------------------------------------------------ #
    def generate_routefile(self, seed: int) -> None:
        """Generate the per-episode SUMO route file for a given traffic seed.

        Vehicle departure times follow a Weibull(2) profile rescaled onto the
        episode horizon; each vehicle takes a straight route with probability
        ``1 - turn_chance`` and a turning route otherwise.

        Args:
            seed: Random seed; identical seeds yield identical traffic, which is
                what makes the five-mode benchmark a paired comparison.
        """
        rng = np.random.default_rng(seed)

        timings = np.sort(rng.weibull(WEIBULL_SHAPE, self.n_cars_generated))
        # Rescale the raw Weibull samples onto [0, max_steps].
        depart_steps = np.rint(
            np.interp(timings, (timings.min(), timings.max()), (0, self.max_steps)),
        ).astype(int)

        lines: list[str] = [ROUTES_FILE_HEADER]
        for car_counter, depart in enumerate(depart_steps):
            if rng.uniform() < (1.0 - self.turn_chance):
                route = str(rng.choice(STRAIGHT_ROUTES))
            else:
                route = str(rng.choice(TURN_ROUTES))
            lines.append(
                f'    <vehicle id="{route}_{car_counter}" type="standard_car" '
                f'route="{route}" depart="{depart}" departLane="random" '
                f'departSpeed="{DEFAULT_DEPART_SPEED}" />',
            )
        lines.append("</routes>")

        ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROUTES_FILE.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # State observation (12-dimensional lane-count vector)
    # ------------------------------------------------------------------ #
    def get_state(self) -> NDArray:
        """Return the true 12-dimensional vehicle-count state.

        Each component counts the vehicles currently within ``STOPLINE_ZONE_M``
        metres of the junction on the corresponding lane group, in the order given
        by ``LANE_GROUP_LABELS``.

        Returns:
            A float array of shape ``(STATE_SIZE,)`` of non-negative counts.
        """
        road_max_length = self.spec.road_max_length
        vehicles = (
            (
                traci.vehicle.getLaneID(car_id),
                road_max_length - float(traci.vehicle.getLanePosition(car_id)),
            )
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
        one-off switching penalty when the phase changes:

            Delta = green + yellow * 1[u != u_prev]
            c     = sum_{i not served by u} x_i           (linear, canonical)
                    (1/kappa) * sum_{i not served} x_i^2   (quadratic, ablation)
            g     = - c * (1 - gamma^Delta) / (1 - gamma)  - P * 1[u != u_prev]

        With ``reward = g`` the DQN tail value ``H`` is on the same scale as the
        rollout stage cost (``beta = 1`` is principled).

        Args:
            state: Current 12-dimensional state ``x``.
            action: Candidate action ``u`` in ``[0, NUM_ACTIONS)``.
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
        than ``DETECTOR_DISTANCE_M`` from the stop line (i.e. just after entering
        the arm). The per-group counts are appended to the rolling history so that
        :meth:`get_arrival_rates` can apply the travel-time lag.
        """
        counts = [0] * self.spec.state_size
        lane_to_index = self.spec.lane_id_to_state_index
        road_max_length = self.spec.road_max_length
        detector_distance = self.spec.detector_distance_m

        for car_id in traci.vehicle.getIDList():
            if car_id in self._detected_ids:
                continue
            group = lane_to_index.get(traci.vehicle.getLaneID(car_id))
            if group is None:
                continue
            dist_to_tl = road_max_length - float(traci.vehicle.getLanePosition(car_id))
            if dist_to_tl >= detector_distance:
                counts[group] += 1
                self._detected_ids.add(car_id)

        for group in range(self.spec.state_size):
            self._arrival_history[group].append(counts[group])

    def get_arrival_rates(self) -> NDArray:
        """Estimate the lag-corrected stop-line arrival rate per lane group.

        The detector senses vehicles ~636 m upstream; a vehicle sensed now reaches
        the stop line ``tau`` seconds later. The arrival rate relevant to the stop
        line *now* is therefore the detector rate measured ``tau`` seconds ago, so
        the estimator averages the per-step counts over the lagged window
        ``[step - tau - window, step - tau)``.

        Returns:
            A float array of shape ``(STATE_SIZE,)`` of arrival rates (veh/s).
            Lane groups with insufficient lagged history return ``0.0``.
        """
        rates = np.zeros(self.spec.state_size, dtype=float)

        lag_steps = int(round(self.detector_lag_tau))
        window_end = self.step - lag_steps
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
    # Action execution
    # ------------------------------------------------------------------ #
    def execute(self, action: int) -> list[EnvStats]:
        """Apply an action: optional yellow transition, then the green phase.

        If the requested green phase differs from the current one, a yellow phase
        is inserted first. Stepping is capped so the episode never exceeds
        ``max_steps``.

        Args:
            action: Discrete action index mapped to a traffic-light phase.

        Returns:
            Per-step statistics gathered while the phases were held.
        """
        next_green_phase = self.spec.action_to_phase[action]
        current_green_phase = traci.trafficlight.getPhase(self.spec.tl_id)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats.extend(self._simulate(self.yellow_duration))

        if self.is_over():
            return stats

        self._set_green_phase(next_green_phase)
        stats.extend(self._simulate(self.green_duration))

        return stats

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance SUMO by up to ``duration`` steps, sensing and recording.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            One :class:`EnvStats` per executed simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            self._record_detector()
            stats.append(EnvStats(queue_length=self.get_queue_length()))

        return stats

    def _set_yellow_phase(self, current_green_phase: int) -> None:
        """Switch the traffic light to the yellow matching a green phase.

        Args:
            current_green_phase: Code of the currently active green phase.
        """
        traci.trafficlight.setPhase(self.spec.tl_id, self.spec.green_to_yellow[current_green_phase])

    def _set_green_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the given green phase.

        Args:
            green_phase_code: Code of the green phase to activate.
        """
        traci.trafficlight.setPhase(self.spec.tl_id, green_phase_code)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def get_queue_length(self) -> int:
        """Return the number of halted vehicles on all incoming edges.

        Returns:
            Total count of vehicles with speed 0 across the four incoming edges.
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
