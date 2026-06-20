"""The global SUMO session (:class:`SumoSession`).

Phase 2.1 of the multi-agent refactor separates the *one global thing* — the
TraCI connection and the simulation clock that every junction shares — from the
*many local things* (the per-junction observation/cost/actuation, which live in
:mod:`clean_rollout_tlcs.junction_env`). A :class:`SumoSession` owns:

* the TraCI lifecycle (``start`` / ``close``) and the SUMO command line;
* the global clock (``step``) and the atomic world tick (``tick``);
* the Weibull-based per-episode route-generation pipeline.

It is deliberately junction-agnostic: it knows how to advance the world, not what
is being controlled. One session drives a single isolated intersection today and
an N-junction grid in Phase 2.3 — the only difference is how many
:class:`JunctionEnv` instances observe each tick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import traci
from sumolib import checkBinary

from .constants import (
    DEFAULT_DEPART_SPEED,
    ROUTES_FILE,
    ROUTES_FILE_HEADER,
    STRAIGHT_ROUTES,
    TURN_ROUTES,
    WEIBULL_SHAPE,
)


class SumoSession:
    """Owns the TraCI connection, the global clock and route generation."""

    def __init__(
        self,
        *,
        sumocfg_file: Path,
        gui: bool,
        max_steps: int,
        n_cars_generated: int,
        turn_chance: float,
    ) -> None:
        """Initialize the session.

        Args:
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to launch the SUMO GUI binary.
            max_steps: Maximum number of simulation steps (seconds) per episode.
            n_cars_generated: Number of vehicles to spawn per episode.
            turn_chance: Probability a generated vehicle takes a turning route.
        """
        self.sumocfg_file = sumocfg_file
        self.gui = gui
        self.max_steps = max_steps
        self.n_cars_generated = n_cars_generated
        self.turn_chance = turn_chance

        self.step = 0

    # ------------------------------------------------------------------ #
    # TraCI lifecycle + global clock
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
        """Reset the global clock to the start of an episode."""
        self.step = 0

    def start(self, cmd: list[str]) -> None:
        """Start the SUMO simulation with the given command line.

        Args:
            cmd: The fully built SUMO command (possibly extended with GUI flags by
                a caller, which is why the command is injected rather than rebuilt).
        """
        traci.start(cmd)

    def close(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def tick(self) -> int:
        """Advance the world by exactly one simulation step.

        Returns:
            The new global step count.
        """
        traci.simulationStep()
        self.step += 1
        return self.step

    def is_over(self) -> bool:
        """Return whether the episode has reached ``max_steps``."""
        return self.step >= self.max_steps

    def steps_remaining(self, duration: int) -> int:
        """Number of steps actually executable without exceeding the horizon.

        Args:
            duration: Desired number of steps.

        Returns:
            ``min(duration, max_steps - step)`` clamped at zero.
        """
        return max(0, min(duration, self.max_steps - self.step))

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
