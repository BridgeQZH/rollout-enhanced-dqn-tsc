"""Multi-agent grid runner (Phase 2.3).

Drives an N-junction network from a single shared
:class:`~clean_rollout_tlcs.session.SumoSession` using a **1-second, event-driven
stepping loop**: the world advances one second at a time and each junction runs
its own green/yellow state machine, deciding only when its current phase window
expires. This preserves every agent's variable effective window ``Delta`` (and
therefore the ``gamma**Delta`` value alignment) rather than forcing a single
global decision cadence.

Phase 2.3 ships the runner plus two model-free controllers (fixed-time, greedy) so
the whole grid pipeline is runnable and testable end-to-end today; the DQN /
rollout controllers plug in at Phase 2.4 once a grid-shaped (16->4) value function
has been trained.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import traci
from numpy.typing import NDArray

from .constants import GRID2X2_NET, GRID2X2_ROUTES, GRID2X2_SUMOCFG
from .intersection_spec import IntersectionSpec
from .junction_env import JunctionEnv
from .net_parser import build_specs_from_net
from .route_gen import GridODRoutes
from .session import SumoSession
from .settings import Settings

# Per-agent phase-machine states.
_DECIDE = "DECIDE"
_YELLOW = "YELLOW"
_GREEN = "GREEN"


class Controller(Protocol):
    """Selects a discrete action for one junction at each decision point."""

    needs_arrival_rates: bool

    def select(self, state: NDArray, prev_action: int, arrival_rates: NDArray | None) -> int:
        """Return the action index for the given local observation."""
        ...

    def reset(self) -> None:
        """Reset any per-episode internal state."""
        ...


class FixedTimeController:
    """Deterministic fixed-time cycler: 0 -> 1 -> ... -> n-1 -> 0 (state-blind)."""

    needs_arrival_rates = False

    def __init__(self, num_actions: int) -> None:
        """Initialize the cycler.

        Args:
            num_actions: Number of discrete phases to cycle through.
        """
        self.num_actions = num_actions
        self._counter = 0

    def select(self, state: NDArray, prev_action: int, arrival_rates: NDArray | None) -> int:
        """Return the next phase in the fixed cycle."""
        action = self._counter % self.num_actions
        self._counter += 1
        return action

    def reset(self) -> None:
        """Reset the cycle counter."""
        self._counter = 0


class GreedyController:
    """Greedy base policy: pick the action with the highest stage cost ``g``."""

    needs_arrival_rates = False

    def __init__(self, cost_fn: Callable[[NDArray, int, int], float], num_actions: int) -> None:
        """Initialize the greedy controller.

        Args:
            cost_fn: The junction's stage cost ``g(state, action, prev_action)``.
            num_actions: Number of discrete actions.
        """
        self.cost_fn = cost_fn
        self.num_actions = num_actions

    def select(self, state: NDArray, prev_action: int, arrival_rates: NDArray | None) -> int:
        """Return the least-cost (highest ``g``) action."""
        scores = np.array(
            [self.cost_fn(state, u, prev_action) for u in range(self.num_actions)],
            dtype=float,
        )
        return int(np.argmax(scores))

    def reset(self) -> None:
        """No per-episode state to reset."""


@dataclass
class _AgentSchedule:
    """Per-junction phase state machine + per-episode metric accumulators."""

    state: str = _DECIDE
    timer: int = 0
    prev_action: int = -1
    pending_action: int = -1
    queue_sum: int = 0
    n_decisions: int = 0

    def reset(self) -> None:
        """Reset the machine and accumulators for a new episode."""
        self.state = _DECIDE
        self.timer = 0
        self.prev_action = -1
        self.pending_action = -1
        self.queue_sum = 0
        self.n_decisions = 0


class MultiAgentRunner:
    """Runs one shared SUMO session with one controller per junction."""

    def __init__(
        self,
        *,
        session: SumoSession,
        junctions: dict[str, JunctionEnv],
        controllers: dict[str, Controller],
        green_duration: int,
        yellow_duration: int,
        extra_sumo_args: list[str] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            session: The shared SUMO session driving the whole network.
            junctions: One :class:`JunctionEnv` per traffic light, keyed by tl id.
            controllers: One controller per traffic light, keyed by tl id.
            green_duration: Green hold time (s) applied per decision.
            yellow_duration: Yellow hold time (s) inserted on a phase change.
            extra_sumo_args: Extra SUMO CLI flags appended at start (e.g. GUI play).
        """
        self.session = session
        self.junctions = junctions
        self.controllers = controllers
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.extra_sumo_args = extra_sumo_args or []

        self.order = sorted(junctions)
        self.schedules: dict[str, _AgentSchedule] = {tl: _AgentSchedule() for tl in self.order}
        self.arrived_total = 0
        self.residual_vehicles = 0

    # ------------------------------------------------------------------ #
    # Episode driver
    # ------------------------------------------------------------------ #
    def run_episode(self, seed: int) -> dict:
        """Run one full grid episode under the configured controllers.

        Args:
            seed: Traffic seed (also fixes the generated grid OD trips).

        Returns:
            A metrics dict with per-junction and network-level aggregates.
        """
        self.session.generate_routefile(seed)
        self._reset_episode()
        self.session.start(self.session.build_sumo_cmd() + self.extra_sumo_args)

        while not self.session.is_over():
            # 1) Resolve every agent currently at a phase boundary (timer == 0).
            for tl in self.order:
                if self.schedules[tl].timer == 0:
                    self._advance(tl)
            # 2) Advance the shared world by exactly one second.
            self.session.tick()
            # 3) Each junction senses this step; tick down its phase timer.
            for tl in self.order:
                stats = self.junctions[tl].observe_step()
                self.schedules[tl].queue_sum += stats.queue_length
                self.schedules[tl].timer -= 1
            # 4) Network-level throughput / residual bookkeeping.
            self.arrived_total += int(traci.simulation.getArrivedNumber())
            self.residual_vehicles = int(traci.simulation.getMinExpectedNumber())

        self.session.close()
        return self._metrics()

    def _advance(self, tl: str) -> None:
        """Advance one agent's phase state machine at a boundary.

        A junction in yellow proceeds to its pending green; otherwise it makes a
        new decision (its first, or after a green window expired) and either
        inserts a yellow (on a phase change) or commits the green directly.

        Args:
            tl: Traffic-light id of the agent to advance.
        """
        sched = self.schedules[tl]
        junction = self.junctions[tl]
        controller = self.controllers[tl]
        spec = junction.spec

        if sched.state == _YELLOW:
            junction.set_green_phase(spec.action_to_phase[sched.pending_action])
            sched.prev_action = sched.pending_action
            sched.state = _GREEN
            sched.timer = self.green_duration
            return

        # Decision point (initial, or a green window just expired).
        state = junction.get_state()
        rates = junction.get_arrival_rates() if controller.needs_arrival_rates else None
        action = controller.select(state, sched.prev_action, rates)
        sched.n_decisions += 1

        if sched.prev_action != -1 and action != sched.prev_action:
            junction.set_yellow_phase(spec.action_to_phase[sched.prev_action])
            sched.pending_action = action
            sched.state = _YELLOW
            sched.timer = self.yellow_duration
        else:
            junction.set_green_phase(spec.action_to_phase[action])
            sched.prev_action = action
            sched.state = _GREEN
            sched.timer = self.green_duration

    def _reset_episode(self) -> None:
        """Reset the session clock, junctions, schedules and controllers."""
        self.session.reset()
        for tl in self.order:
            self.junctions[tl].reset()
            self.schedules[tl].reset()
            self.controllers[tl].reset()
        self.arrived_total = 0
        self.residual_vehicles = 0

    def _metrics(self) -> dict:
        """Aggregate per-junction and network metrics after an episode."""
        n_steps = max(self.session.step, 1)
        per_junction = {
            tl: {
                "queue_sum": sched.queue_sum,
                "avg_queue": sched.queue_sum / n_steps,
                "n_decisions": sched.n_decisions,
            }
            for tl, sched in self.schedules.items()
        }
        total_wait = sum(s["queue_sum"] for s in per_junction.values())
        return {
            "n_steps": self.session.step,
            "n_junctions": len(self.order),
            "per_junction": per_junction,
            "network_total_wait_vehsec": total_wait,
            "network_avg_queue": total_wait / n_steps,
            "throughput": self.arrived_total,
            "residual_vehicles": self.residual_vehicles,
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_grid_junctions(
    settings: Settings,
    specs: dict[str, IntersectionSpec],
    session: SumoSession,
) -> dict[str, JunctionEnv]:
    """Build one :class:`JunctionEnv` per spec sharing one session.

    Args:
        settings: Validated configuration (durations, cost, detector params).
        specs: Per-junction specs (e.g. from :func:`build_specs_from_net`).
        session: The shared session the junctions read the clock from.

    Returns:
        Mapping ``tl_id -> JunctionEnv``. Detector lag ``tau`` is derived per
        junction from the clamped detector distance (passed ``None``).
    """
    return {
        tl: JunctionEnv(
            spec=spec,
            session=session,
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
            gamma=settings.gamma,
            cost_type=settings.cost_type,
            switch_penalty=settings.switch_penalty,
            cost_quadratic_kappa=settings.cost_quadratic_kappa,
            detector_window=settings.detector_window,
            detector_lag_tau=None,
        )
        for tl, spec in specs.items()
    }


def build_grid_runner(
    settings: Settings,
    *,
    mode: str = "greedy",
    gui: bool = False,
    extra_sumo_args: list[str] | None = None,
) -> MultiAgentRunner:
    """Assemble a :class:`MultiAgentRunner` for the committed 2x2 grid.

    Args:
        settings: Validated configuration (provides demand, horizon, durations).
        mode: Controller mode; ``"fixed_time"`` or ``"greedy"`` (model-free). The
            ``"dqn"`` / ``"rollout_*"`` modes require a trained 16->4 grid value
            function and are wired in Phase 2.4.
        gui: Whether to launch the SUMO GUI binary.
        extra_sumo_args: Extra SUMO CLI flags appended at start (GUI playback).

    Returns:
        A ready-to-run multi-agent runner over the grid.

    Raises:
        ValueError: If ``mode`` is not a model-free controller.
    """
    specs = build_specs_from_net(GRID2X2_NET)
    route_generator = GridODRoutes(
        netfile=GRID2X2_NET,
        out_file=GRID2X2_ROUTES,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
    )
    session = SumoSession(
        sumocfg_file=GRID2X2_SUMOCFG,
        gui=gui,
        max_steps=settings.max_steps,
        n_cars_generated=settings.n_cars_generated,
        turn_chance=settings.turn_chance,
        route_generator=route_generator,
    )
    junctions = build_grid_junctions(settings, specs, session)

    controllers: dict[str, Controller] = {}
    for tl, junction in junctions.items():
        if mode == "fixed_time":
            controllers[tl] = FixedTimeController(junction.spec.num_actions)
        elif mode == "greedy":
            controllers[tl] = GreedyController(junction.stage_cost, junction.spec.num_actions)
        else:
            msg = (
                f"mode '{mode}' needs a trained grid model (Phase 2.4); "
                "use 'fixed_time' or 'greedy' for now"
            )
            raise ValueError(msg)

    return MultiAgentRunner(
        session=session,
        junctions=junctions,
        controllers=controllers,
        green_duration=settings.green_duration,
        yellow_duration=settings.yellow_duration,
        extra_sumo_args=extra_sumo_args,
    )
