"""Analytic equivalence: spec-threaded f / g / state-count vs an independent ref.

This is the fast, SUMO-free heart of the Phase 2.0 oracle. It re-derives the
transition model ``f``, the stage cost ``g`` and the state counter directly from
the math (using the legacy ``constants.py`` literals as ground truth), then checks
that the refactored spec-parameterised code reproduces them bit-for-bit over a
deterministic random battery of states, actions and arrival rates.

Because the reference here is an *independent* implementation — not a recording of
the old code — it also guards against a latent bug that happened to exist before.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tests._fixtures import small_eval_settings

from clean_rollout_tlcs import constants as C
from clean_rollout_tlcs.env import Environment
from clean_rollout_tlcs.intersection_spec import build_single_intersection_spec
from clean_rollout_tlcs.transition import TransitionModel

_RNG = np.random.default_rng(20260618)
_N_TRIALS = 400


# --------------------------------------------------------------------------- #
# Independent reference implementations (from the model equations + constants)
# --------------------------------------------------------------------------- #
def _ref_effective_window(green: int, yellow: int, prev_action: int, action: int) -> int:
    switched = prev_action != -1 and action != prev_action
    return green + (yellow if switched else 0)


def _ref_f(
    state: NDArray,
    action: int,
    prev_action: int,
    rates: NDArray,
    *,
    green: int,
    yellow: int,
    sat: float,
    startup: float,
) -> NDArray:
    switched = prev_action != -1 and action != prev_action
    window = green + (yellow if switched else 0)
    green_eff = max(float(green) - (startup if switched else 0.0), 0.0)
    arrivals = rates * window
    service = np.asarray(C.SERVICE_INDICATOR, dtype=float)[action]
    discharge = service * sat * np.asarray(C.LANES_PER_GROUP, dtype=float) * green_eff
    nxt = state + arrivals - discharge
    return np.clip(nxt, 0.0, np.asarray(C.LANE_QUEUE_CAPACITY, dtype=float))


def _ref_stage_cost(
    state: NDArray,
    action: int,
    prev_action: int,
    *,
    green: int,
    yellow: int,
    gamma: float,
    cost_type: str,
    kappa: float,
    switch_penalty: float,
) -> float:
    switched = prev_action != -1 and action != prev_action
    delta = green + (yellow if switched else 0)
    served = C.SERVED_LANES[action]
    if cost_type == "linear":
        per_sec = float(sum(state[i] for i in range(C.STATE_SIZE) if i not in served))
    else:
        per_sec = float(sum(state[i] ** 2 for i in range(C.STATE_SIZE) if i not in served)) / kappa
    discount_sum = float(delta) if gamma >= 1.0 else (1.0 - gamma**delta) / (1.0 - gamma)
    switch_cost = switch_penalty if switched else 0.0
    return -(per_sec * discount_sum) - switch_cost


def _ref_count_state(vehicles: list[tuple[str, float]]) -> NDArray:
    state = np.zeros(C.STATE_SIZE, dtype=float)
    for lane_id, position in vehicles:
        group = C.LANE_ID_TO_STATE_INDEX.get(lane_id)
        if group is None:
            continue
        if (C.ROAD_MAX_LENGTH - position) <= C.STOPLINE_ZONE_M:  # uniform 750 m arms
            state[group] += 1.0
    return state


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_transition_f_matches_reference() -> None:
    """Spec-threaded TransitionModel.f must equal the independent reference."""
    settings = small_eval_settings()
    model = TransitionModel.from_settings(settings)  # spec defaults to single intersection

    max_abs = 0.0
    for _ in range(_N_TRIALS):
        state = _RNG.uniform(0, 15, size=C.STATE_SIZE)
        rates = _RNG.uniform(0, 0.4, size=C.STATE_SIZE)
        action = int(_RNG.integers(0, C.NUM_ACTIONS))
        prev_action = int(_RNG.integers(-1, C.NUM_ACTIONS))

        got = model.f(state, action, prev_action, rates)
        ref = _ref_f(
            state, action, prev_action, rates,
            green=settings.green_duration, yellow=settings.yellow_duration,
            sat=settings.saturation_flow, startup=settings.startup_lost_time,
        )
        max_abs = max(max_abs, float(np.max(np.abs(got - ref))))

    assert max_abs == 0.0, f"f deviates from reference (max abs diff {max_abs})"


def test_stage_cost_matches_reference_both_cost_types() -> None:
    """Spec-threaded Environment.stage_cost must equal the reference (linear+quad)."""
    for cost_type in ("linear", "quadratic"):
        settings = small_eval_settings().model_copy(update={"cost_type": cost_type})
        env = Environment.from_settings(settings)  # no SUMO started; stage_cost is pure

        worst = 0.0
        for _ in range(_N_TRIALS):
            state = _RNG.uniform(0, 15, size=C.STATE_SIZE)
            action = int(_RNG.integers(0, C.NUM_ACTIONS))
            prev_action = int(_RNG.integers(-1, C.NUM_ACTIONS))

            got = env.stage_cost(state, action, prev_action)
            ref = _ref_stage_cost(
                state, action, prev_action,
                green=settings.green_duration, yellow=settings.yellow_duration,
                gamma=settings.gamma, cost_type=cost_type,
                kappa=settings.cost_quadratic_kappa, switch_penalty=settings.switch_penalty,
            )
            worst = max(worst, abs(got - ref))

        assert worst == 0.0, f"stage_cost[{cost_type}] deviates from reference (max abs diff {worst})"


def test_count_state_matches_reference() -> None:
    """Spec.count_state must equal the legacy distance-threshold counting logic."""
    spec = build_single_intersection_spec()
    lane_ids = list(C.LANE_ID_TO_STATE_INDEX.keys()) + ["TL2N_0", ":TL_3_0"]  # last two: untracked

    for _ in range(_N_TRIALS):
        n = int(_RNG.integers(0, 40))
        vehicles = [
            (str(_RNG.choice(lane_ids)), float(_RNG.uniform(0, C.ROAD_MAX_LENGTH)))
            for _ in range(n)
        ]
        got = spec.count_state(vehicles)  # second element is now lane position
        ref = _ref_count_state(vehicles)
        assert np.array_equal(got, ref), "count_state deviates from reference"


_TESTS = (
    test_transition_f_matches_reference,
    test_stage_cost_matches_reference_both_cost_types,
    test_count_state_matches_reference,
)


def main() -> int:
    """Run all analytic-equivalence tests; return non-zero on failure."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - oracle reports, does not raise
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"[test_single_agent_equivalence] {len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
