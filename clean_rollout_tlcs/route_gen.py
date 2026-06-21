"""Per-episode route generators (Phase 2.3).

Route generation is the one part of the simulation pipeline that is intrinsically
network-shaped, so it lives behind a small :class:`RouteGenerator` protocol that
:class:`~clean_rollout_tlcs.session.SumoSession` can be handed:

* the single intersection keeps its original Weibull OD generator (inline in the
  session, used when no generator is injected) so its traffic stays byte-identical;
* the 2x2 grid uses :class:`GridODRoutes`, which emits **long-range origin/
  destination trips** that enter at one boundary and leave at a *different*
  junction's boundary, so vehicles actually traverse the corridor and load the
  internal links the multi-agent controllers coordinate over.

Both keep the paired-seed property: a given seed yields identical traffic, which
is what makes cross-controller comparisons fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import sumolib

from .constants import DEFAULT_DEPART_SPEED, WEIBULL_SHAPE

# vType shared by the grid trips; mirrors the single-intersection standard_car.
_GRID_VTYPE = (
    '    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" '
    'minGap="2.5" maxSpeed="25" sigma="0.5" />'
)


class RouteGenerator(Protocol):
    """Writes a per-episode SUMO route file for a given traffic seed."""

    def generate(self, seed: int) -> None:
        """Generate and write the route file for ``seed``."""
        ...


class GridODRoutes:
    """Long-range boundary OD trip generator for a grid network.

    Vehicles are inserted on a fringe *entry* edge and routed to a fringe *exit*
    edge belonging to a different junction, so every trip crosses at least one
    internal link. Departure times follow the same rescaled Weibull(2) profile as
    the single-intersection generator, keeping demand shape and seed-pairing
    consistent across the two networks.
    """

    def __init__(
        self,
        *,
        netfile: Path | str,
        out_file: Path | str,
        n_cars_generated: int,
        max_steps: int,
        depart_speed: float = DEFAULT_DEPART_SPEED,
        weibull_shape: float = WEIBULL_SHAPE,
    ) -> None:
        """Initialize the generator and classify the network's boundary edges.

        Args:
            netfile: Path to the grid ``.net.xml``.
            out_file: Destination route file (referenced by the grid ``.sumocfg``).
            n_cars_generated: Number of trips to emit per episode.
            max_steps: Episode horizon (s); departures are rescaled onto it.
            depart_speed: Initial speed for inserted vehicles (m/s).
            weibull_shape: Shape parameter of the departure-time profile.

        Raises:
            ValueError: If the network exposes no usable boundary entry/exit edges.
        """
        self.out_file = Path(out_file)
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.depart_speed = depart_speed
        self.weibull_shape = weibull_shape

        net = sumolib.net.readNet(str(netfile))
        tl_nodes = {t.getID() for t in net.getTrafficLights()}

        # Entry: border-node -> junction. Exit: junction -> border-node.
        self._entry_edges: list[str] = []
        self._exit_edges: list[str] = []
        self._edge_junction: dict[str, str] = {}
        for edge in net.getEdges():
            if edge.isSpecial():
                continue
            from_id, to_id = edge.getFromNode().getID(), edge.getToNode().getID()
            if from_id not in tl_nodes and to_id in tl_nodes:
                self._entry_edges.append(edge.getID())
                self._edge_junction[edge.getID()] = to_id
            elif from_id in tl_nodes and to_id not in tl_nodes:
                self._exit_edges.append(edge.getID())
                self._edge_junction[edge.getID()] = from_id

        if not self._entry_edges or not self._exit_edges:
            msg = f"No boundary entry/exit edges found in {netfile}"
            raise ValueError(msg)
        self._entry_edges.sort()
        self._exit_edges.sort()

    def generate(self, seed: int) -> None:
        """Write the per-episode grid route file for ``seed``.

        Args:
            seed: Traffic seed; identical seeds yield identical trips.
        """
        rng = np.random.default_rng(seed)

        timings = np.sort(rng.weibull(self.weibull_shape, self.n_cars_generated))
        depart_steps = np.rint(
            np.interp(timings, (timings.min(), timings.max()), (0, self.max_steps)),
        ).astype(int)

        lines: list[str] = ["<routes>", _GRID_VTYPE, ""]
        for car_counter, depart in enumerate(depart_steps):
            entry = str(rng.choice(self._entry_edges))
            exit_edge = self._pick_exit(rng, entry)
            lines.append(
                f'    <trip id="{entry}_{exit_edge}_{car_counter}" type="standard_car" '
                f'depart="{depart}" from="{entry}" to="{exit_edge}" '
                f'departLane="random" departSpeed="{self.depart_speed}" />',
            )
        lines.append("</routes>")

        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.out_file.write_text("\n".join(lines), encoding="utf-8")

    def _pick_exit(self, rng: np.random.Generator, entry: str) -> str:
        """Pick an exit edge at a different junction than the entry edge.

        Args:
            rng: The episode RNG.
            entry: The chosen entry edge id.

        Returns:
            An exit edge id whose junction differs from the entry's (falling back
            to any exit edge if the entry's junction is the only one available).
        """
        entry_junction = self._edge_junction[entry]
        candidates = [e for e in self._exit_edges if self._edge_junction[e] != entry_junction]
        if not candidates:  # degenerate single-junction network: any exit will do
            candidates = self._exit_edges
        return str(rng.choice(candidates))
