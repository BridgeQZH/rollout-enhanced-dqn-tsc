"""Top-level training runner for the rollout-enhanced TLCS DQN.

Trains the Q-network used both as the standalone ``dqn`` controller and as the
tail value ``H`` for the rollout controllers. Supports several independent
training seeds (default three) for robustness, each producing its own neatly
namespaced checkpoint, settings snapshot, training curves and run manifest.

Run as a module from a directory that contains the ``intersection/`` SUMO assets::

    python -m clean_rollout_tlcs.train --settings settings/training_settings.yaml
    python -m clean_rollout_tlcs.train --seeds 0,1,2 --out models/dqn_100ep

Reward alignment: the per-step reward is ``R = env.stage_cost(...) = g`` (the
already-negated queue cost), so the learned values are commensurate with the
rollout stage cost.
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

from .constants import (
    CONFIG_SNAPSHOT_FILE,
    DEFAULT_MODEL_PATH,
    MODEL_FILE,
    NUM_ACTIONS,
    RUN_MANIFEST_FILE,
    STATE_SIZE,
)
from .env import Environment
from .model import Model
from .agent import DQNAgent, Sample
from .settings import Settings, load_settings
from .transition import TransitionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_rollout_tlcs.train")

# Route seed for an episode = training_seed * ROUTE_SEED_STRIDE + episode_index.
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


def _save_series(values: list[float], out_dir: Path, name: str, ylabel: str) -> None:
    """Persist a metric series to a text file and (best-effort) a PNG plot.

    Args:
        values: One value per episode.
        out_dir: Output directory.
        name: Base filename (without extension).
        ylabel: Y-axis label for the plot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"plot_{name}_data.txt").write_text(
        "\n".join(str(v) for v in values),
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(values) + 1), values, marker=".")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_{name}.png", dpi=96)
        plt.close()
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.warning("Skipped plot '%s' (%s)", name, exc)


def train_one_run(
    settings: Settings,
    training_seed: int,
    out_root: Path,
    settings_path: Path,
) -> Path:
    """Train a single DQN run for one seed and save all artifacts.

    Args:
        settings: Validated configuration.
        training_seed: Seed for global RNGs and the per-episode route-seed schedule.
        out_root: Root output directory; the run is saved under ``seed_<seed>``.
        settings_path: Path to the settings YAML, copied into the run directory.

    Returns:
        The run output directory.
    """
    set_global_seeds(training_seed)
    run_dir = out_root / f"seed_{training_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training run | seed=%d -> %s ===", training_seed, run_dir)

    model = Model(
        input_dim=STATE_SIZE,
        output_dim=NUM_ACTIONS,
        num_layers=settings.num_layers,
        width=settings.width_layers,
        learning_rate=settings.learning_rate,
    )
    env = Environment.from_settings(settings)
    agent = DQNAgent(
        settings=settings,
        model=model,
        transition=TransitionModel.from_settings(settings),
        cost_fn=env.stage_cost,
    )

    reward_store: list[float] = []
    cumulative_wait_store: list[float] = []
    avg_queue_store: list[float] = []

    for episode in range(settings.total_episodes):
        epsilon = max(0.0, 1.0 - episode / settings.total_episodes)
        agent.set_epsilon(epsilon)
        agent.reset_episode_state()

        route_seed = training_seed * ROUTE_SEED_STRIDE + episode
        env.generate_routefile(route_seed)
        env.activate()

        prev_action = -1
        episode_reward = 0.0
        queue_sum = 0
        step_count = 0

        while not env.is_over():
            state = env.get_state()
            action = agent.choose_action(state)

            # R = g (already the negated queue cost) -> aligned with the tail value.
            reward = env.stage_cost(state, action, prev_action)
            discount = agent.transition_discount(prev_action, action)

            stats = env.execute(action)
            next_state = env.get_state()

            agent.remember(
                Sample(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    discount=discount,
                ),
            )

            episode_reward += reward
            for stat in stats:
                queue_sum += stat.queue_length
                step_count += 1
            prev_action = action

        env.deactivate()

        for _ in range(settings.training_epochs):
            agent.replay(settings.batch_size)

        avg_queue = queue_sum / max(step_count, 1)
        reward_store.append(episode_reward)
        cumulative_wait_store.append(float(queue_sum))
        avg_queue_store.append(avg_queue)

        logger.info(
            "seed=%d | ep %3d/%d | eps=%.2f | reward=%.1f | cum_wait=%d | avg_queue=%.2f",
            training_seed,
            episode + 1,
            settings.total_episodes,
            epsilon,
            episode_reward,
            queue_sum,
            avg_queue,
        )

    # --- persist artifacts ---------------------------------------------
    agent.save_model(run_dir / MODEL_FILE)
    copyfile(src=settings_path, dst=run_dir / CONFIG_SNAPSHOT_FILE)

    _save_series(reward_store, run_dir, "reward", "Cumulative stage reward (g)")
    _save_series(cumulative_wait_store, run_dir, "delay", "Cumulative delay (veh-s)")
    _save_series(avg_queue_store, run_dir, "queue", "Average queue length (veh)")

    manifest = {
        "training_seed": training_seed,
        "route_seed_formula": f"training_seed * {ROUTE_SEED_STRIDE} + episode",
        "mode_at_train_time": settings.mode,
        "total_episodes": settings.total_episodes,
        "n_cars_generated": settings.n_cars_generated,
        "max_steps": settings.max_steps,
        "gamma": settings.gamma,
        "cost_type": settings.cost_type,
        "device": str(model.device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "final_reward": reward_store[-1] if reward_store else None,
        "final_avg_queue": avg_queue_store[-1] if avg_queue_store else None,
    }
    (run_dir / RUN_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved checkpoint + artifacts to %s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train the rollout-TLCS DQN.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings") / "training_settings.yaml",
        help="Path to the training settings YAML.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Root output directory for trained runs.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated list of training seeds (default: three runs).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: train one run per requested seed."""
    args = parse_args()
    settings = load_settings(args.settings)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        msg = "No training seeds provided."
        raise SystemExit(msg)

    logger.info("Training %d run(s) with seeds %s", len(seeds), seeds)
    if settings.mode != "dqn":
        logger.info(
            "settings.mode='%s'; training the DQN value function regardless "
            "(it serves as the rollout tail value H).",
            settings.mode,
        )

    run_dirs = [train_one_run(settings, seed, args.out, args.settings) for seed in seeds]
    logger.info("All runs complete: %s", [str(d) for d in run_dirs])


if __name__ == "__main__":
    main()
