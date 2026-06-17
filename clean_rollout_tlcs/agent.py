"""DQN agent with experience replay and rollout-based action selection.

This module provides:

* :class:`Memory` — a capacity-bounded experience replay buffer.
* :class:`DQNAgent` — an epsilon-greedy DQN trainer plus the four model-based
  control paths used at evaluation time (greedy, one-step rollout, multi-step
  rollout, and a fixed-time cycler), with f-prediction residual logging hooks.

Sign convention (locked): the injected ``cost_fn`` returns the canonical stage
value ``g(x, u, u_prev) <= 0`` (already the *negated* positive queue cost). The
training reward is exactly this value, i.e. ``R = -(positive queue cost) = g``,
so the DQN tail value ``H = max_a Q`` is commensurate with the rollout stage cost
and ``beta = 1`` is principled. No second negation is ever applied.

Time-discount convention: each action advances the world by an effective window
``Delta`` (green, plus yellow on a switch). The bootstrap therefore uses
``gamma**Delta`` (stored per transition), matching the rollout tail discount
``gamma**Delta`` exactly.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import NUM_ACTIONS, STATE_SIZE
from .model import Model
from .settings import Settings
from .transition import TransitionModel, effective_window

# Stage-cost signature: g(state, action, prev_action) -> float (<= 0).
CostFn = Callable[[NDArray, int, int], float]


@dataclass
class Sample:
    """A single replay transition.

    Attributes:
        state: State observed before the action.
        action: Action taken.
        reward: Aligned stage reward ``R = g`` (<= 0).
        next_state: State observed after the effective action window.
        discount: Effective bootstrap discount ``gamma**Delta`` for this step.
    """

    state: NDArray
    action: int
    reward: float
    next_state: NDArray
    discount: float


class Memory:
    """Capacity-bounded experience replay buffer."""

    def __init__(self, size_max: int, size_min: int) -> None:
        """Initialize the buffer.

        Args:
            size_max: Maximum number of samples retained (FIFO eviction).
            size_min: Minimum number of samples required before sampling returns
                a non-empty batch (warmup threshold).
        """
        self._samples: deque[Sample] = deque(maxlen=size_max)
        self._size_min = size_min

    def add_sample(self, sample: Sample) -> None:
        """Append a transition to the buffer.

        Args:
            sample: The transition to store.
        """
        self._samples.append(sample)

    def get_samples(self, n: int) -> list[Sample]:
        """Draw a random batch, or an empty list before warmup completes.

        Args:
            n: Desired batch size.

        Returns:
            A list of up to ``n`` randomly sampled transitions, or an empty list
            if the buffer has not yet reached the warmup threshold.
        """
        if len(self._samples) < self._size_min:
            return []
        n = min(n, len(self._samples))
        return random.sample(self._samples, n)

    @property
    def is_ready(self) -> bool:
        """Whether the warmup threshold has been reached."""
        return len(self._samples) >= self._size_min

    def __len__(self) -> int:
        """Number of stored transitions."""
        return len(self._samples)


class DQNAgent:
    """Epsilon-greedy DQN agent with model-based rollout control paths."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: Model,
        memory: Memory | None = None,
        transition: TransitionModel | None = None,
        cost_fn: CostFn | None = None,
        epsilon: float = 1.0,
    ) -> None:
        """Initialize the agent.

        Args:
            settings: Validated configuration (hyperparameters and weights).
            model: The Q-network wrapper.
            memory: Replay buffer; a fresh one is created from ``settings`` if None.
            transition: Transition model ``f``; required for rollout control paths.
            cost_fn: Stage-cost callable ``g``; required for greedy/rollout paths.
            epsilon: Initial exploration probability for the DQN path.
        """
        self.settings = settings
        self.model = model
        self.memory = memory or Memory(settings.memory_size_max, settings.memory_size_min)
        self.transition = transition
        self.cost_fn = cost_fn

        self.gamma = settings.gamma
        self.beta = settings.beta
        self.green_duration = settings.green_duration
        self.yellow_duration = settings.yellow_duration
        self.num_actions = NUM_ACTIONS
        self.epsilon = epsilon

        # Fixed-time cycler counter (reset per episode).
        self._fixed_time_counter = 0

        # f-prediction verification state.
        self._pending_prediction: NDArray | None = None
        self._abs_residuals: list[NDArray] = []
        self._signed_residuals: list[NDArray] = []

    # ------------------------------------------------------------------ #
    # Exploration / episode control
    # ------------------------------------------------------------------ #
    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration probability.

        Args:
            epsilon: Value in ``[0, 1]``.

        Raises:
            ValueError: If ``epsilon`` is outside ``[0, 1]``.
        """
        if not 0.0 <= epsilon <= 1.0:
            msg = f"epsilon must be in [0, 1]; got {epsilon}"
            raise ValueError(msg)
        self.epsilon = epsilon

    def reset_episode_state(self) -> None:
        """Reset per-episode controller state (fixed-time counter, pending pred)."""
        self._fixed_time_counter = 0
        self._pending_prediction = None

    # ------------------------------------------------------------------ #
    # Discount / value helpers
    # ------------------------------------------------------------------ #
    def transition_discount(self, prev_action: int, action: int) -> float:
        """Effective bootstrap discount ``gamma**Delta`` for one transition.

        Args:
            prev_action: Previously applied action (``-1`` if none).
            action: Action taken.

        Returns:
            ``gamma`` raised to the effective action window length.
        """
        window = effective_window(self.green_duration, self.yellow_duration, prev_action, action)
        return float(self.gamma**window)

    def tail_value(self, state: NDArray) -> float:
        """DQN tail value ``H(x) = max_a Q(x, a)``.

        Args:
            state: State at which to evaluate the tail value.

        Returns:
            The maximum Q-value over actions.
        """
        return float(np.max(self.model.predict_one(state)))

    # ------------------------------------------------------------------ #
    # Control path: DQN (epsilon-greedy)
    # ------------------------------------------------------------------ #
    def choose_action(self, state: NDArray) -> int:
        """Select an action with the epsilon-greedy DQN policy.

        Args:
            state: Current state.

        Returns:
            Action index. With probability ``epsilon`` a uniform random action is
            returned; otherwise the greedy ``argmax_a Q(state, a)``.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        return int(np.argmax(self.model.predict_one(state)))

    # ------------------------------------------------------------------ #
    # Control path: pure greedy base policy (argmax over stage cost g)
    # ------------------------------------------------------------------ #
    def greedy_base(self, state: NDArray, prev_action: int) -> int:
        """Greedy base policy: pick the action with the highest stage cost ``g``.

        Because ``g`` is the (negative) cost, the highest ``g`` is the least-cost
        action. Scores are computed into a float array and compared with
        ``np.argmax`` — there are no tuples, so no lexicographic tie-breaking bug.

        Args:
            state: Current state.
            prev_action: Previously applied action (``-1`` if none).

        Returns:
            The greedy action index.

        Raises:
            RuntimeError: If no ``cost_fn`` was injected.
        """
        cost_fn = self._require_cost_fn()
        scores = np.array(
            [cost_fn(state, u, prev_action) for u in range(self.num_actions)],
            dtype=float,
        )
        return int(np.argmax(scores))

    # ------------------------------------------------------------------ #
    # Control path: one-step rollout
    # ------------------------------------------------------------------ #
    def select_rollout_1s(
        self,
        state: NDArray,
        prev_action: int,
        arrival_rates: NDArray,
    ) -> int:
        """One-step look-ahead rollout: ``q~(u) = g(x,u) + beta * gamma^Delta * H(f(x,u))``.

        Scores are accumulated into a float array and compared with ``np.argmax``
        (no tuples). The predicted next state for the chosen action is cached for
        residual verification against the realized next state.

        Args:
            state: Current state ``x``.
            prev_action: Previously applied action (``-1`` if none).
            arrival_rates: Per-group arrival-rate estimate (veh/s) fed to ``f``.

        Returns:
            The selected action index.
        """
        cost_fn = self._require_cost_fn()
        transition = self._require_transition()

        q_tilde = np.empty(self.num_actions, dtype=float)
        predictions: list[NDArray] = []

        for u in range(self.num_actions):
            g_u = cost_fn(state, u, prev_action)
            next_state = transition.f(state, u, prev_action, arrival_rates)
            discount = self.transition_discount(prev_action, u)
            q_tilde[u] = g_u + self.beta * discount * self.tail_value(next_state)
            predictions.append(next_state)

        best = int(np.argmax(q_tilde))
        self._pending_prediction = predictions[best]
        return best

    # ------------------------------------------------------------------ #
    # Control path: multi-step rollout (greedy base policy + DQN tail)
    # ------------------------------------------------------------------ #
    def select_rollout_ms(
        self,
        state: NDArray,
        prev_action: int,
        arrival_rates: NDArray,
        depth: int | None = None,
    ) -> int:
        """Multi-step look-ahead rollout with a greedy base policy and DQN tail.

        For each first candidate action, the model is rolled forward ``depth``
        steps: the first action is the candidate, subsequent actions follow the
        greedy base policy. Discounted stage costs are accumulated and a discounted
        DQN tail value is added at the final predicted state. The first-step
        prediction for the chosen action is cached for residual verification.

        Args:
            state: Current state ``x``.
            prev_action: Previously applied action (``-1`` if none).
            arrival_rates: Per-group arrival-rate estimate (veh/s) fed to ``f``.
            depth: Look-ahead depth; defaults to ``settings.lookahead_depth``.

        Returns:
            The selected first action index.
        """
        cost_fn = self._require_cost_fn()
        transition = self._require_transition()
        horizon = depth if depth is not None else self.settings.lookahead_depth

        q_tilde = np.empty(self.num_actions, dtype=float)
        first_predictions: list[NDArray] = []

        for first_action in range(self.num_actions):
            total = 0.0
            cumulative_window = 0  # sum of Delta_j so far -> gamma exponent
            x_t = np.asarray(state, dtype=float).copy()
            u_prev_t = prev_action
            u_t = first_action
            first_pred: NDArray | None = None

            for t in range(horizon):
                g_t = cost_fn(x_t, u_t, u_prev_t)
                total += (self.gamma**cumulative_window) * g_t

                window = effective_window(self.green_duration, self.yellow_duration, u_prev_t, u_t)
                x_next = transition.f(x_t, u_t, u_prev_t, arrival_rates)
                if t == 0:
                    first_pred = x_next

                cumulative_window += window
                u_prev_t = u_t
                x_t = x_next
                if t < horizon - 1:
                    u_t = self.greedy_base(x_t, u_prev_t)

            tail = self.beta * (self.gamma**cumulative_window) * self.tail_value(x_t)
            q_tilde[first_action] = total + tail
            # first_pred is always set because horizon >= 1.
            first_predictions.append(first_pred if first_pred is not None else x_t)

        best = int(np.argmax(q_tilde))
        self._pending_prediction = first_predictions[best]
        return best

    # ------------------------------------------------------------------ #
    # Control path: fixed-time cycler
    # ------------------------------------------------------------------ #
    def choose_action_fixed_time(self) -> int:
        """Fixed-time control: cycle phases 0 -> 1 -> 2 -> 3 deterministically.

        Returns:
            The next action in the fixed cycle (independent of state).
        """
        action = self._fixed_time_counter % self.num_actions
        self._fixed_time_counter += 1
        return action

    # ------------------------------------------------------------------ #
    # Unified dispatcher (used by the evaluation harness)
    # ------------------------------------------------------------------ #
    def select_action(
        self,
        state: NDArray,
        prev_action: int = -1,
        arrival_rates: NDArray | None = None,
    ) -> int:
        """Select an action according to ``settings.mode``.

        Args:
            state: Current state.
            prev_action: Previously applied action (``-1`` if none).
            arrival_rates: Required for rollout modes; ignored otherwise.

        Returns:
            The selected action index.

        Raises:
            ValueError: If a rollout mode is requested without ``arrival_rates``.
            RuntimeError: For an unknown mode (should be unreachable given the
                validated ``Literal`` type).
        """
        mode = self.settings.mode
        if mode == "dqn":
            return self.choose_action(state)
        if mode == "greedy":
            return self.greedy_base(state, prev_action)
        if mode == "fixed_time":
            return self.choose_action_fixed_time()
        if mode in {"rollout_1s", "rollout_ms"}:
            if arrival_rates is None:
                msg = f"mode '{mode}' requires arrival_rates"
                raise ValueError(msg)
            if mode == "rollout_1s":
                return self.select_rollout_1s(state, prev_action, arrival_rates)
            return self.select_rollout_ms(state, prev_action, arrival_rates)
        msg = f"Unknown control mode: {mode}"  # pragma: no cover
        raise RuntimeError(msg)

    # ------------------------------------------------------------------ #
    # f-prediction residual verification hooks
    # ------------------------------------------------------------------ #
    def log_residual(self, actual_next_state: NDArray) -> NDArray | None:
        """Record the residual between the cached f-prediction and reality.

        Call this once per decision step *after* the environment has executed the
        chosen action and the true next state has been observed. Does nothing if no
        prediction is pending (e.g. for non-rollout modes).

        Args:
            actual_next_state: The true next state observed from the simulator.

        Returns:
            The signed per-lane residual ``predicted - actual``, or ``None`` if no
            prediction was pending.
        """
        if self._pending_prediction is None:
            return None
        predicted = self._pending_prediction
        actual = np.asarray(actual_next_state, dtype=float)
        signed = predicted - actual
        self._signed_residuals.append(signed)
        self._abs_residuals.append(np.abs(signed))
        self._pending_prediction = None
        return signed

    def residual_summary(self) -> dict[str, NDArray | float | int]:
        """Aggregate the logged f-prediction residuals.

        Returns:
            A dictionary with per-lane and overall error metrics: ``n`` (count),
            ``mae_per_lane``, ``rmse_per_lane``, ``bias_per_lane``, ``mae`` and
            ``rmse``. Per-lane arrays are zeros when no residuals were logged.
        """
        if not self._abs_residuals:
            zeros = np.zeros(STATE_SIZE, dtype=float)
            return {
                "n": 0,
                "mae_per_lane": zeros,
                "rmse_per_lane": zeros.copy(),
                "bias_per_lane": zeros.copy(),
                "mae": 0.0,
                "rmse": 0.0,
            }

        abs_arr = np.asarray(self._abs_residuals, dtype=float)
        signed_arr = np.asarray(self._signed_residuals, dtype=float)
        return {
            "n": int(abs_arr.shape[0]),
            "mae_per_lane": abs_arr.mean(axis=0),
            "rmse_per_lane": np.sqrt((signed_arr**2).mean(axis=0)),
            "bias_per_lane": signed_arr.mean(axis=0),
            "mae": float(abs_arr.mean()),
            "rmse": float(np.sqrt((signed_arr**2).mean())),
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def remember(self, sample: Sample) -> None:
        """Store a transition in replay memory.

        Args:
            sample: The transition to store.
        """
        self.memory.add_sample(sample)

    def replay(self, batch_size: int) -> float | None:
        """Sample a batch and perform one Q-learning update.

        The bootstrap target uses the per-transition effective discount:
        ``target = R + gamma**Delta * max_a' Q(s', a')``.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            The training loss, or ``None`` if the buffer is not yet warmed up.
        """
        batch = self.memory.get_samples(batch_size)
        if not batch:
            return None

        states = np.array([s.state for s in batch], dtype=np.float32)
        next_states = np.array([s.next_state for s in batch], dtype=np.float32)

        q_values = self.model.predict_batch(states)
        next_q_values = self.model.predict_batch(next_states)
        max_next = next_q_values.max(axis=1)

        targets = q_values.copy()
        for i, sample in enumerate(batch):
            targets[i, sample.action] = sample.reward + sample.discount * max_next[i]

        return self.model.train_batch(states, targets)

    def save_model(self, path) -> None:  # noqa: ANN001 - pathlib.Path
        """Save the underlying model checkpoint.

        Args:
            path: Destination checkpoint path.
        """
        self.model.save_checkpoint(path)

    # ------------------------------------------------------------------ #
    # Internal guards
    # ------------------------------------------------------------------ #
    def _require_cost_fn(self) -> CostFn:
        """Return the injected cost function or raise if missing."""
        if self.cost_fn is None:
            msg = "This control path requires a cost_fn (g) to be injected."
            raise RuntimeError(msg)
        return self.cost_fn

    def _require_transition(self) -> TransitionModel:
        """Return the injected transition model or raise if missing."""
        if self.transition is None:
            msg = "This control path requires a TransitionModel (f) to be injected."
            raise RuntimeError(msg)
        return self.transition
