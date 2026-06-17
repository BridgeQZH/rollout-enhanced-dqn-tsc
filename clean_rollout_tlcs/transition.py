"""Analytical traffic transition model ``f(x, u, u_prev)``.

This module is deliberately free of any SUMO/``traci`` dependency: it operates
purely on the 12-dimensional queue vector and an externally supplied arrival-rate
estimate (produced by the environment's causal upstream detector). Keeping the
physics here, decoupled from the simulator, lets the rollout controllers roll the
model forward over imagined states and lets us unit-test and calibrate ``f`` in
isolation.

Model (point-queue / store-and-forward):

    Delta       = green + yellow * 1[u != u_prev]          (arrivals window)
    green_eff   = max(green - startup * 1[u != u_prev], 0) (discharge window)
    arrivals_i  = lambda_i * Delta
    discharge_i = A(i,u) * sat_flow * lanes_i * green_eff
    x'_i        = clip( x_i + arrivals_i - discharge_i , 0 , C_i )

Arrivals accrue over the whole window (vehicles keep arriving during yellow),
while discharge happens only during the green portion, reduced by startup lost
time when the phase has just changed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import (
    LANE_QUEUE_CAPACITY,
    LANES_PER_GROUP,
    NUM_ACTIONS,
    SERVICE_INDICATOR,
    STATE_SIZE,
)
from .settings import Settings


def is_switch(prev_action: int, action: int) -> bool:
    """Return whether applying ``action`` changes the active phase.

    Args:
        prev_action: Previously applied action, or ``-1`` if none yet.
        action: Candidate action.

    Returns:
        ``True`` if a yellow transition would be inserted, else ``False``.
    """
    return prev_action != -1 and action != prev_action


def effective_window(green_duration: int, yellow_duration: int, prev_action: int, action: int) -> int:
    """Effective action window ``Delta`` over which arrivals accrue.

    Args:
        green_duration: Green hold time (s).
        yellow_duration: Yellow hold time (s) inserted on a phase change.
        prev_action: Previously applied action, or ``-1`` if none yet.
        action: Candidate action.

    Returns:
        ``green + yellow`` on a phase change, otherwise ``green``.
    """
    return green_duration + (yellow_duration if is_switch(prev_action, action) else 0)


def effective_green(
    green_duration: int,
    startup_lost_time: float,
    prev_action: int,
    action: int,
) -> float:
    """Effective green time available for discharge, net of startup lost time.

    Args:
        green_duration: Green hold time (s).
        startup_lost_time: Lost time at green onset (s), charged only on a switch.
        prev_action: Previously applied action, or ``-1`` if none yet.
        action: Candidate action.

    Returns:
        ``max(green - startup, 0)`` on a phase change, otherwise ``green``.
    """
    loss = startup_lost_time if is_switch(prev_action, action) else 0.0
    return max(float(green_duration) - loss, 0.0)


class TransitionModel:
    """Analytical one-step transition model for the 12-D queue state."""

    def __init__(
        self,
        *,
        green_duration: int,
        yellow_duration: int,
        saturation_flow: float,
        startup_lost_time: float,
        capacity: NDArray | None = None,
    ) -> None:
        """Initialize the transition model with its physical constants.

        Args:
            green_duration: Green hold time (s).
            yellow_duration: Yellow hold time (s) inserted on a phase change.
            saturation_flow: Saturation discharge rate per physical lane (veh/s).
            startup_lost_time: Lost time at green onset (s) charged on a switch.
            capacity: Optional per-state-index queue capacity ``C_i``; defaults to
                ``LANE_QUEUE_CAPACITY`` from the network geometry.
        """
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.saturation_flow = saturation_flow
        self.startup_lost_time = startup_lost_time

        self._lanes_per_group = np.asarray(LANES_PER_GROUP, dtype=float)
        self._capacity = (
            np.asarray(capacity, dtype=float)
            if capacity is not None
            else np.asarray(LANE_QUEUE_CAPACITY, dtype=float)
        )
        # Precompute the 4x12 service-indicator matrix once.
        self._service = np.asarray(SERVICE_INDICATOR, dtype=float)

    @classmethod
    def from_settings(cls, settings: Settings) -> TransitionModel:
        """Build a :class:`TransitionModel` from a validated :class:`Settings`.

        Args:
            settings: Validated configuration.

        Returns:
            A configured transition model.
        """
        return cls(
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
            saturation_flow=settings.saturation_flow,
            startup_lost_time=settings.startup_lost_time,
        )

    def f(
        self,
        state: NDArray,
        action: int,
        prev_action: int,
        arrival_rates: NDArray,
    ) -> NDArray:
        """Predict the next queue state ``x'`` under ``action``.

        Args:
            state: Current 12-D queue vector ``x``.
            action: Candidate action ``u`` in ``[0, NUM_ACTIONS)``.
            prev_action: Previously applied action ``u_prev`` (``-1`` if none).
            arrival_rates: Per-group arrival rates ``lambda`` (veh/s), typically
                the lag-corrected estimate from the environment.

        Returns:
            The predicted next queue vector of shape ``(STATE_SIZE,)``, clipped to
            ``[0, C_i]``.

        Raises:
            ValueError: If ``action`` is invalid or the input shapes are wrong.
        """
        if not 0 <= action < NUM_ACTIONS:
            msg = f"action must be in [0, {NUM_ACTIONS}); got {action}"
            raise ValueError(msg)

        x = np.asarray(state, dtype=float)
        rates = np.asarray(arrival_rates, dtype=float)
        if x.shape != (STATE_SIZE,) or rates.shape != (STATE_SIZE,):
            msg = (
                f"state and arrival_rates must both have shape ({STATE_SIZE},); "
                f"got {x.shape} and {rates.shape}"
            )
            raise ValueError(msg)

        window = effective_window(self.green_duration, self.yellow_duration, prev_action, action)
        green_eff = effective_green(self.green_duration, self.startup_lost_time, prev_action, action)

        arrivals = rates * window
        discharge = self._service[action] * self.saturation_flow * self._lanes_per_group * green_eff

        next_state = x + arrivals - discharge
        return np.clip(next_state, 0.0, self._capacity)
