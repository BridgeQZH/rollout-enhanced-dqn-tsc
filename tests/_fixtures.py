"""Shared fixtures for the Phase 2.0 regression oracle.

Centralises the small, fast, fully-deterministic evaluation configuration used by
both the golden *capture* and the golden *verify* so the two can never drift.
The configuration deliberately shrinks the episode horizon and demand: the oracle
needs a reproducible trajectory to pin behaviour, not a representative one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/<script>.py` as well as `python -m tests.<script>`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clean_rollout_tlcs.settings import Settings, load_settings  # noqa: E402

# Existing trained checkpoint reused as the (frozen) DQN + rollout tail value.
GOLDEN_MODEL_DIR = Path("models") / "dqn_100ep" / "seed_0"
GOLDEN_BASE_SETTINGS = GOLDEN_MODEL_DIR / "training_settings.yaml"

# Golden artifact written by capture_golden.py and read by test_eval_regression.py.
GOLDEN_DIR = Path("tests") / "golden"
GOLDEN_EVAL_FILE = GOLDEN_DIR / "single_agent_eval_golden.json"

# Small, deterministic eval overrides (fast enough to run in the oracle loop).
_SMALL_EVAL_OVERRIDES = {
    "gui": False,
    "max_steps": 600,
    "n_cars_generated": 120,
    "n_eval_seeds": 2,
    "eval_seed_start": 100000,
    "detector_window": 100,
}


def small_eval_settings() -> Settings:
    """Return the shared, deterministic small-eval :class:`Settings`.

    Returns:
        The model-run base settings with the small-horizon overrides applied.
    """
    base = load_settings(GOLDEN_BASE_SETTINGS)
    return base.model_copy(update=_SMALL_EVAL_OVERRIDES)
