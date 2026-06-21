"""Shared-parameter DQN training on the 2x2 grid (Phase 2.4).

A single Q-network (and a single replay buffer) is shared by all four junctions:
every junction selects actions with the same network and contributes its
transitions to one pooled buffer, so the learned value generalises across the
homogeneous grid. The network is trained on the canonical reward ``R = g`` exactly
as in the single-intersection trainer, so the resulting values double as the tail
value ``H`` for the Multi-Agent Physics Shield (rollout) at evaluation time.

Transitions are assembled from the runner's per-decision hook: for each junction a
decision's ``(state, action, reward=g, discount=gamma**Delta)`` is paired with that
junction's *next* decision state to form a replay sample.

Run from the repository root::

    python -m clean_rollout_tlcs.train_grid --episodes 500 --demand 1000 --max-steps 3600
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from .agent import DQNAgent, Sample
from .constants import CONFIG_SNAPSHOT_FILE, GRID2X2_NET, MODEL_FILE, RUN_MANIFEST_FILE
from .junction_env import JunctionEnv
from .model import Model
from .multi_agent import AgentController, MultiAgentRunner, build_grid_junctions, build_grid_session
from .net_parser import build_specs_from_net
from .settings import Settings, load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_rollout_tlcs.train_grid")

# Route seed for an episode = training_seed * STRIDE + episode_index.
ROUTE_SEED_STRIDE = 10_000


def set_global_seeds(seed: int) -> None:
    """Seed all relevant RNGs for a reproducible training run.

    Args:
        seed: The training seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TransitionRecorder:
    """Assembles per-junction decisions into pooled shared-buffer transitions.

    Each junction's decision is held pending until that junction's *next* decision,
    whose state becomes the ``next_state``. The dangling final decision per junction
    (no successor before the episode ends) is dropped — at most one sample per
    junction per episode.
    """

    def __init__(self, agent: DQNAgent, junctions: dict[str, JunctionEnv]) -> None:
        """Initialize the recorder.

        Args:
            agent: The shared DQN agent owning the pooled replay buffer.
            junctions: The per-junction environments (for the stage cost ``g``).
        """
        self.agent = agent
        self.junctions = junctions
        self._pending: dict[str, tuple] = {}

    def reset(self) -> None:
        """Drop any pending half-transitions at the start of a new episode."""
        self._pending = {}

    def __call__(self, tl: str, state: np.ndarray, action: int, prev_action: int) -> None:
        """Record a decision, completing the previous one's transition.

        Args:
            tl: Traffic-light id that just decided.
            state: Local state observed at this decision.
            action: Action chosen.
            prev_action: Action applied before this decision.
        """
        junction = self.junctions[tl]
        reward = junction.stage_cost(state, action, prev_action)
        discount = self.agent.transition_discount(prev_action, action)

        prev = self._pending.get(tl)
        if prev is not None:
            prev_state, prev_act, prev_reward, prev_discount = prev
            self.agent.remember(
                Sample(
                    state=prev_state,
                    action=prev_act,
                    reward=prev_reward,
                    next_state=np.asarray(state, dtype=float).copy(),
                    discount=prev_discount,
                ),
            )
        self._pending[tl] = (np.asarray(state, dtype=float).copy(), action, reward, discount)


def train_one_run(
    settings: Settings,
    training_seed: int,
    episodes: int,
    out_root: Path,
    settings_path: Path,
) -> Path:
    """Train a single shared-parameter DQN run on the grid and save artifacts.

    Args:
        settings: Validated configuration.
        training_seed: Seed for global RNGs and the per-episode route schedule.
        episodes: Number of training episodes.
        out_root: Root output directory; the run is saved under ``seed_<seed>``.
        settings_path: Settings YAML, copied into the run directory.

    Returns:
        The run output directory.
    """
    set_global_seeds(training_seed)
    run_dir = out_root / f"seed_{training_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs_from_net(GRID2X2_NET)
    any_spec = next(iter(specs.values()))
    logger.info(
        "=== Grid training | seed=%d | %d junctions | state=%d action=%d -> %s ===",
        training_seed,
        len(specs),
        any_spec.state_size,
        any_spec.num_actions,
        run_dir,
    )

    model = Model(
        input_dim=any_spec.state_size,
        output_dim=any_spec.num_actions,
        num_layers=settings.num_layers,
        width=settings.width_layers,
        learning_rate=settings.learning_rate,
    )
    # One shared agent (shared weights + one pooled replay buffer) for all junctions.
    shared_agent = DQNAgent(settings=settings, model=model)
    shared_agent.num_actions = any_spec.num_actions

    session = build_grid_session(settings)
    junctions = build_grid_junctions(settings, specs, session)
    recorder = _TransitionRecorder(shared_agent, junctions)
    controllers = {tl: AgentController(shared_agent, "dqn") for tl in junctions}
    runner = MultiAgentRunner(
        session=session,
        junctions=junctions,
        controllers=controllers,
        green_duration=settings.green_duration,
        yellow_duration=settings.yellow_duration,
        on_decision=recorder,
    )

    avg_queue_store: list[float] = []
    for episode in range(episodes):
        epsilon = max(0.0, 1.0 - episode / episodes)
        shared_agent.set_epsilon(epsilon)
        recorder.reset()

        route_seed = training_seed * ROUTE_SEED_STRIDE + episode
        metrics = runner.run_episode(route_seed)

        for _ in range(settings.training_epochs):
            shared_agent.replay(settings.batch_size)

        avg_queue_store.append(metrics["network_avg_queue"])
        logger.info(
            "seed=%d | ep %3d/%d | eps=%.2f | net_avg_q=%6.2f | throughput=%d | residual=%d | buffer=%d",
            training_seed,
            episode + 1,
            episodes,
            epsilon,
            metrics["network_avg_queue"],
            metrics["throughput"],
            metrics["residual_vehicles"],
            len(shared_agent.memory),
        )

    shared_agent.save_model(run_dir / MODEL_FILE)
    copyfile(src=settings_path, dst=run_dir / CONFIG_SNAPSHOT_FILE)

    manifest = {
        "training_seed": training_seed,
        "network": "grid2x2",
        "n_junctions": len(specs),
        "state_size": any_spec.state_size,
        "num_actions": any_spec.num_actions,
        "episodes": episodes,
        "n_cars_generated": settings.n_cars_generated,
        "max_steps": settings.max_steps,
        "gamma": settings.gamma,
        "cost_type": settings.cost_type,
        "shared_parameters": True,
        "device": str(model.device),
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "final_net_avg_queue": avg_queue_store[-1] if avg_queue_store else None,
    }
    (run_dir / RUN_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved shared-parameter grid checkpoint + artifacts to %s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the shared-parameter grid DQN.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("clean_rollout_tlcs") / "settings" / "training_settings.yaml",
        help="Base settings YAML (durations, cost, model, replay).",
    )
    parser.add_argument("--out", type=Path, default=Path("models") / "grid2x2",
                        help="Root output directory for trained runs.")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated training seeds.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: settings.total_episodes).")
    parser.add_argument("--demand", type=int, default=None,
                        help="Override n_cars_generated for grid demand.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override episode horizon.")
    return parser.parse_args()


def main() -> None:
    """Entry point: train one shared-parameter grid run per requested seed."""
    args = parse_args()
    settings = load_settings(args.settings)

    update: dict = {}
    if args.demand is not None:
        update["n_cars_generated"] = args.demand
    if args.max_steps is not None:
        update["max_steps"] = args.max_steps
    if update:
        settings = settings.model_copy(update=update)

    episodes = args.episodes if args.episodes is not None else settings.total_episodes
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    logger.info("Training %d grid run(s) | episodes=%d | demand=%d | max_steps=%d",
                len(seeds), episodes, settings.n_cars_generated, settings.max_steps)
    for seed in seeds:
        train_one_run(settings, seed, episodes, args.out, args.settings)


if __name__ == "__main__":
    main()
