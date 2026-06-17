"""Rollout-enhanced Traffic Light Control System (clean_rollout_tlcs).

A clean, strongly-typed PyTorch reimplementation that adopts the AndreaVidali
object-oriented architecture as its chassis and injects the research engine:

* a 12-dimensional lane-count state space (``env.get_state``);
* an analytical transition model ``f`` (:mod:`clean_rollout_tlcs.transition`);
* a canonical stage cost ``g`` (:meth:`clean_rollout_tlcs.env.Environment.stage_cost`);
* a causal upstream arrival-rate estimator with travel-time lag correction;
* DQN training plus one-step and multi-step rollout control paths
  (:mod:`clean_rollout_tlcs.agent`).

Run training as a module from a directory containing the SUMO ``intersection/``
assets::

    python -m clean_rollout_tlcs.train
"""

from __future__ import annotations

__version__ = "0.1.0"
