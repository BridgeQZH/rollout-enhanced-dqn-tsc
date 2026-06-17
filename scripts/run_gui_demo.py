"""Launch ONE controller in SUMO-GUI for the interview demo, with LIVE telemetry.

Run directly from a terminal (no shell/batch wrappers), e.g.::

    python scripts/run_gui_demo.py --mode dqn         --demand 2000 --seed 100000 --delay 120
    python scripts/run_gui_demo.py --mode rollout_ms  --demand 2000 --seed 100000 --delay 120

Two windows appear simultaneously:

* the **SUMO-GUI** animating micro-traffic at the junction, and
* an independent, non-blocking **live telemetry window** (matplotlib) that draws
  the instantaneous total queue length (the unserved-queue stage signal) and a
  moving average on a rolling timeline — the closed-loop MDP dynamics, live.

Fairness: the route file is generated from (seed, demand), so running ``--mode dqn``
and ``--mode rollout_ms`` with the SAME ``--seed`` gives byte-identical traffic.

This reuses the frozen evaluation modules verbatim (no edits). It re-implements the
control loop locally only so it can push per-step telemetry; action selection is
delegated to the frozen ``clean_rollout_tlcs.eval._select_action`` so the on-screen
policy is identical to the headless benchmark.

-----------------------------------------------------------------------------
SUMO-GUI VISUAL TUNING (toggle these once the GUI opens, for sharp video)
-----------------------------------------------------------------------------
Open  View Settings  (the ▾/gear icon on the toolbar), then:
  * Vehicles  > Color by: "waiting time"  (cars redden as they wait -> the DQN
      run shows far more sustained red under saturation).
  * Vehicles  > Size: enable "Show as ... " and raise the exaggeration / min-size
      so cars stay legible after video compression (try exaggerate ≈ 2-3).
  * Junctions > check "Draw junction shape" and "Show link junction index" off;
      keep "Draw crossings/walkingareas" off to reduce clutter.
  * Streets   > Color by: "occupancy" (or "edge data") to shade congested lanes;
      enable "Show lane borders" for crisp lane separation.
  * Background: set to solid (Background > uncheck "Show grid", set decals off)
      so compression artifacts don't dance on the backdrop.
  * Zoom so all four ~750 m approaches are in frame; keep the SAME zoom/viewport
      for the DQN and rollout runs (gui_settings.xml pins a starting viewport).
You can Save these as a scheme from the dialog and reload for the second run.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

# Running `python scripts/run_gui_demo.py` puts scripts/ (not the repo root) on
# sys.path, so the clean_rollout_tlcs package would not import. Add the repo root.
# (Data paths like models/ and intersection/ still assume cwd = repo root.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Frozen modules — imported, never modified.
from clean_rollout_tlcs.agent import DQNAgent
from clean_rollout_tlcs.eval import (
    DEFAULT_MULTISTEP_DEPTH,
    ROLLOUT_MODES,
    EvalEnvironment,
    _select_action,
    load_model_from_dir,
)
from clean_rollout_tlcs.settings import ControlMode, load_settings
from clean_rollout_tlcs.transition import TransitionModel

DEFAULT_BASE_SETTINGS = Path("models") / "dqn_500ep" / "seed_0" / "training_settings.yaml"
DEFAULT_GUI_SETTINGS = Path("intersection") / "gui_settings.xml"

# Telemetry line colour per controller (matches the deck theme).
_MODE_COLOR = {
    "dqn": "#f59e0b",         # amber
    "rollout_ms": "#34d399",  # emerald
    "rollout_1s": "#34d399",
    "greedy": "#64748b",      # slate
    "fixed_time": "#475569",
}
_MODE_LABEL = {
    "dqn": "Frozen DQN",
    "rollout_ms": "Rollout-DQN (multi-step)",
    "rollout_1s": "Rollout-DQN (1-step)",
    "greedy": "Greedy (g only)",
    "fixed_time": "Fixed-time",
}


class GuiEvalEnvironment(EvalEnvironment):
    """EvalEnvironment that adds GUI-friendly flags to the SUMO command line."""

    gui_delay_ms: int = 120
    gui_settings_path: Path | None = None
    auto_start: bool = True
    quit_on_end: bool = False

    def build_sumo_cmd(self) -> list[str]:
        """Extend the frozen command with GUI playback options."""
        cmd = super().build_sumo_cmd()
        cmd += ["--delay", str(self.gui_delay_ms)]
        if self.auto_start:
            cmd += ["--start", "true"]
        if self.quit_on_end:
            cmd += ["--quit-on-end", "true"]
        if self.gui_settings_path and self.gui_settings_path.exists():
            cmd += ["--gui-settings-file", str(self.gui_settings_path)]
        return cmd


class LiveTelemetry:
    """Non-blocking live plot of total queue length + moving average.

    Driven manually from the TraCI loop (push per step, redraw per decision).
    Uses an interactive matplotlib backend; degrades to a no-op with a warning
    if no GUI backend is available, so the SUMO run still proceeds.
    """

    def __init__(self, mode: str, ma_window: int = 60, rolling: int = 0, hold: bool = True) -> None:
        """Set up the dark-theme live figure.

        Args:
            mode: Controller name (selects the line colour / title).
            ma_window: Moving-average window in simulation steps.
            rolling: If > 0, show only the last ``rolling`` steps (scrolling x-axis);
                0 shows the full episode timeline.
            hold: If True, block at the end so the final curve stays on screen.
        """
        self.enabled = True
        self.mode = mode
        self.window = ma_window
        self.rolling = rolling
        self.hold = hold
        self.xs: list[int] = []
        self.ys: list[float] = []
        self.ma: list[float] = []
        self._dq: deque[float] = deque(maxlen=ma_window)
        self._ysum = 0.0

        try:
            import matplotlib

            if matplotlib.get_backend().lower() == "agg":
                for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
                    try:
                        matplotlib.use(backend, force=True)
                        break
                    except Exception:  # noqa: BLE001 - try the next backend
                        continue
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001 - telemetry is optional
            print(f"[telemetry] disabled (no interactive matplotlib backend: {exc})")
            self.enabled = False
            return

        # Figure creation can still fail even after a backend is selected (e.g. Tk
        # importable but no display / no $DISPLAY) -- disable gracefully if so.
        try:
            self.plt = plt
            color = _MODE_COLOR.get(mode, "#34d399")
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7.2, 4.2))
            self.fig.patch.set_facecolor("#0e1217")
            self.ax.set_facecolor("#0e1217")
            for spine in self.ax.spines.values():
                spine.set_color("#334155")
            self.ax.tick_params(colors="#e5e7eb")
            self.ax.grid(True, color="#1f2933", alpha=0.8)
            self.ax.set_xlabel("Simulation step", color="#e5e7eb")
            self.ax.set_ylabel("Total queue length  (veh, unserved)", color="#e5e7eb")
            self.ax.set_title(f"Live MDP telemetry — {_MODE_LABEL.get(mode, mode)}",
                              color="#f1f5f9", fontweight="bold")
            (self.l_raw,) = self.ax.plot([], [], color=color, lw=1.0, alpha=0.40, label="step queue")
            (self.l_ma,) = self.ax.plot([], [], color=color, lw=2.6, label=f"moving avg ({ma_window} steps)")
            self.txt = self.ax.text(0.985, 0.95, "", transform=self.ax.transAxes, ha="right", va="top",
                                    color="#e5e7eb", fontsize=11, family="monospace",
                                    bbox={"facecolor": "#11161d", "edgecolor": "#243040", "boxstyle": "round,pad=0.4"})
            leg = self.ax.legend(loc="upper left", frameon=False, fontsize=10)
            for t in leg.get_texts():
                t.set_color("#cdd8e6")
            try:
                self.fig.canvas.manager.set_window_title(f"Telemetry — {_MODE_LABEL.get(mode, mode)}")
            except Exception:  # noqa: BLE001 - window title is cosmetic
                pass
            self.fig.tight_layout()
            plt.show(block=False)
            self._flush()
        except Exception as exc:  # noqa: BLE001 - telemetry is optional
            print(f"[telemetry] disabled (window could not be created: {exc})")
            self.enabled = False

    def push(self, step: int, queue: float) -> None:
        """Record one simulation step's instantaneous total queue."""
        if not self.enabled:
            return
        self.xs.append(step)
        self.ys.append(queue)
        self._ysum += queue
        self._dq.append(queue)
        self.ma.append(sum(self._dq) / len(self._dq))

    def redraw(self) -> None:
        """Refresh the figure (call once per decision, after pushing its steps)."""
        if not self.enabled or not self.xs:
            return
        self.l_raw.set_data(self.xs, self.ys)
        self.l_ma.set_data(self.xs, self.ma)
        last = self.xs[-1]
        if self.rolling > 0 and last > self.rolling:
            self.ax.set_xlim(last - self.rolling, last)
            ywin = self.ys[-self.rolling:]
        else:
            self.ax.set_xlim(0, max(50, last))
            ywin = self.ys
        self.ax.set_ylim(0, max(ywin) * 1.15 + 1)
        self.txt.set_text(
            f"step  {last:>4}\nqueue {self.ys[-1]:>5.0f}\nMA{self.window:<3} {self.ma[-1]:6.1f}\n"
            f"mean  {self._ysum / len(self.ys):6.1f}",
        )
        self._flush()

    def _flush(self) -> None:
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:  # noqa: BLE001 - keep the sim running even if draw fails
            pass

    def finish(self) -> None:
        """Mark the run done; optionally block so the final curve persists."""
        if not self.enabled:
            return
        self.ax.set_title(self.ax.get_title() + "  · DONE", color="#f1f5f9")
        self._flush()
        if self.hold:
            try:
                self.plt.ioff()
                print("[telemetry] episode complete — close the telemetry window to exit.")
                self.plt.show()
            except Exception:  # noqa: BLE001
                pass


def run_gui_episode(
    env: GuiEvalEnvironment,
    agent: DQNAgent,
    mode: ControlMode,
    depth: int,
    telemetry: LiveTelemetry,
    seed: int,
) -> dict[str, float]:
    """Run one GUI episode, pushing per-step telemetry, return final metrics.

    Mirrors ``clean_rollout_tlcs.eval.run_eval_episode`` but updates the live plot
    after every decision. Action selection is delegated to the frozen
    ``_select_action`` so the on-screen policy matches the benchmark exactly.
    """
    is_rollout = mode in ROLLOUT_MODES
    env.generate_routefile(seed)
    env.activate()
    agent.reset_episode_state()

    prev_action = -1
    queue_sum = 0
    step_count = 0
    total_g = 0.0
    n_decisions = 0

    while not env.is_over():
        state = env.get_state()
        arrival_rates = env.get_arrival_rates() if is_rollout else None
        action = _select_action(agent, mode, state, prev_action, arrival_rates, depth)
        total_g += env.stage_cost(state, action, prev_action)
        n_decisions += 1

        stats = env.execute(action)
        for stat in stats:
            queue_sum += stat.queue_length
            step_count += 1
            telemetry.push(step_count, stat.queue_length)
        telemetry.redraw()
        prev_action = action

    env.deactivate()
    return {
        "n_decisions": n_decisions,
        "total_wait_vehsec": queue_sum,
        "avg_queue": queue_sum / max(step_count, 1),
        "throughput": env.arrived_total,
        "residual_vehicles": env.residual_vehicles,
        "total_stage_cost_g": total_g,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run one controller in SUMO-GUI with live MDP telemetry.")
    p.add_argument("--mode", default="rollout_ms",
                   choices=["dqn", "fixed_time", "greedy", "rollout_1s", "rollout_ms"],
                   help="Controller to visualize (default: rollout_ms).")
    p.add_argument("--model-dir", type=Path, default=Path("models") / "dqn_500ep" / "seed_0",
                   help="Run directory with trained_model.pt (default: models/dqn_500ep/seed_0).")
    p.add_argument("--base-settings", type=Path, default=DEFAULT_BASE_SETTINGS,
                   help="Base settings YAML to derive from.")
    p.add_argument("--demand", type=int, default=2000, help="n_cars_generated (default: 2000).")
    p.add_argument("--seed", type=int, default=100000, help="Traffic seed; use the SAME for both runs.")
    p.add_argument("--delay", type=int, default=120, help="SUMO-GUI step delay in ms (default: 120).")
    p.add_argument("--max-steps", type=int, default=None, help="Optional episode-horizon override.")
    p.add_argument("--gui-settings", type=Path, default=DEFAULT_GUI_SETTINGS,
                   help="SUMO view-settings file (default: intersection/gui_settings.xml).")
    p.add_argument("--quit-on-end", action="store_true", help="Close the SUMO-GUI automatically when finished.")
    # telemetry options
    p.add_argument("--no-telemetry", action="store_true", help="Disable the live telemetry window.")
    p.add_argument("--ma-window", type=int, default=60, help="Moving-average window in steps (default: 60).")
    p.add_argument("--telemetry-rolling", type=int, default=0,
                   help="Show only the last N steps (scrolling x-axis); 0 = full timeline (default).")
    p.add_argument("--no-hold", action="store_true",
                   help="Do not block on the telemetry window at the end (it will close with the script).")
    p.add_argument("--telemetry-selftest", action="store_true",
                   help="Render ~200 synthetic points to verify the matplotlib backend / dark theme, "
                        "then exit. No SUMO, no model, no repo-root needed.")
    return p.parse_args()


def _run_selftest(args: argparse.Namespace) -> int:
    """Exercise the telemetry window with synthetic data — backend smoke test.

    Builds a :class:`LiveTelemetry` for the chosen ``--mode`` and animates ~200
    synthetic congestion points so you can confirm the interactive backend works
    and preview the dark-theme layout before launching the full simulation.

    Args:
        args: Parsed CLI arguments (uses ``mode``, ``ma_window``,
            ``telemetry_rolling`` and ``no_hold``).

    Returns:
        0 if the window rendered; 1 if no interactive backend was available.
    """
    import math
    import random
    import time

    print(f"[selftest] telemetry backend check — mode={args.mode}, ~200 synthetic points (no SUMO).")
    tel = LiveTelemetry(
        mode=args.mode,
        ma_window=args.ma_window,
        rolling=args.telemetry_rolling,
        hold=not args.no_hold,
    )
    if not tel.enabled:
        print("[selftest] FAILED: no interactive matplotlib backend available.")
        print("           Test Tk with:  python -m tkinter   (or install PyQt), then retry.")
        return 1

    rng = random.Random(0)
    n = 200
    for k in range(1, n + 1):
        # Synthetic congestion: rising ramp + periodic waves + noise (D=2000-ish).
        ramp = 8.0 + 38.0 * (k / n)
        wave = 9.0 * math.sin(k / 11.0)
        q = max(0.0, ramp + wave + rng.uniform(-3.0, 3.0))
        tel.push(k, q)
        if k % 3 == 0 or k == n:
            tel.redraw()
            time.sleep(0.02)

    print("[selftest] OK — telemetry rendered 200 points. Close the window to exit.")
    tel.finish()
    return 0


def main() -> int:
    """Build env/agent/telemetry and run one GUI episode for the chosen controller."""
    args = parse_args()

    if args.telemetry_selftest:
        return _run_selftest(args)

    settings = load_settings(args.base_settings)
    update = {"n_cars_generated": args.demand, "gui": True}
    if args.max_steps is not None:
        update["max_steps"] = args.max_steps
    settings = settings.model_copy(update=update)

    model, _run = load_model_from_dir(args.model_dir, settings)
    transition = TransitionModel.from_settings(settings)

    env = GuiEvalEnvironment.from_settings(settings)
    env.gui_delay_ms = args.delay
    env.gui_settings_path = args.gui_settings
    env.auto_start = True
    env.quit_on_end = args.quit_on_end

    agent = DQNAgent(
        settings=settings,
        model=model,
        transition=transition,
        cost_fn=env.stage_cost,
        epsilon=0.0,
    )

    mode: ControlMode = args.mode  # type: ignore[assignment]
    depth = settings.lookahead_depth if settings.lookahead_depth >= 2 else DEFAULT_MULTISTEP_DEPTH

    telemetry = LiveTelemetry(
        mode=mode,
        ma_window=args.ma_window,
        rolling=args.telemetry_rolling,
        hold=not args.no_hold,
    )

    print(f"[gui-demo] mode={mode}  demand={args.demand}  seed={args.seed}  delay={args.delay}ms")
    print(f"[gui-demo] model={args.model_dir.as_posix()}  telemetry={'off' if args.no_telemetry else 'on'}")
    if args.no_telemetry:
        telemetry.enabled = False

    metrics = run_gui_episode(env, agent, mode, depth, telemetry, args.seed)

    print(f"[gui-demo] DONE  mode={mode}")
    print(f"           total_wait_vehsec = {metrics['total_wait_vehsec']:,}")
    print(f"           avg_queue         = {metrics['avg_queue']:.2f}")
    print(f"           throughput        = {metrics['throughput']}  residual = {metrics['residual_vehicles']}")

    telemetry.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
