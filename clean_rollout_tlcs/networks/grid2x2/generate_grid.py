"""Reproducibly (re)generate the 2x2 grid network via ``netgenerate``.

Run from the repository root (requires ``SUMO_HOME`` set)::

    python clean_rollout_tlcs/networks/grid2x2/generate_grid.py

Design of the generated network:

* ``--grid.number=2`` with ``--grid.attach-length`` makes four *symmetric* 4-way
  junctions (two internal approaches to neighbours + two external stubs each), so
  every junction has identical (16-lane, 4-action) tensor dims — the homogeneity
  the shared-parameter DQN needs.
* ``--turn-lanes=1`` spanning the full edge yields a dedicated left-turn lane,
  which makes ``netgenerate`` emit **protected left phases**: an 8-phase program
  whose four green phases (NS-through, NS-left, EW-through, EW-left) map directly
  onto our four discrete actions.
* Links are ~166-183 m — deliberately far shorter than the 636 m single-junction
  detector distance, exercising the parser's graceful detector clamp.

The committed ``grid2x2.net.xml`` is the artifact the net parser and tests read;
this script just documents and reproduces exactly how it was built.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_OUT = Path(__file__).resolve().parent / "grid2x2.net.xml"

_ARGS = [
    "--grid",
    "--grid.number=2",
    "--grid.length=200",
    "--grid.attach-length=200",
    "--default.lanenumber=3",
    "--turn-lanes=1",
    "--turn-lanes.length=200",
    "--default.speed=13.89",
    "--no-turnarounds=true",
    "--tls.guess=true",
    "--tls.guess.threshold=0",
    "--tls.left-green.time=6",
    "--tls.green.time=31",
    "--tls.yellow.time=4",
]


def main() -> int:
    """Invoke netgenerate with the pinned arguments."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        print("SUMO_HOME is not set; cannot locate netgenerate.", file=sys.stderr)
        return 1

    netgenerate = Path(sumo_home) / "bin" / "netgenerate"
    cmd = [str(netgenerate), *_ARGS, f"--output-file={_OUT}"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)  # noqa: S603 - pinned args, trusted binary
    if result.returncode == 0:
        print(f"Wrote {_OUT}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
