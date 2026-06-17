"""Reproducible figure generation for the PhD interview deck.

Reads the validated source artifacts (paired-benchmark CSVs and the JSON
summaries) and renders the five *data-driven* charts as dark-theme SVG into
``deck/assets/figures/``. Conceptual diagrams (architecture, MDP, frontier,
self-challenge) are authored as inline SVG in ``deck/index.html`` and are not
produced here.

Run from the repository root (after extract_evidence.py)::

    python scripts/generate_figures.py

Figures
-------
1. fig_sample_efficiency.svg  -- rollout@100ep recovers converged DQN@500ep
2. fig_demand_stress.svg      -- Delta(D) re-opens under saturation (+ JT trend)
3. fig_paired_seed.svg        -- 60-pair slopegraph at D=2000 (robustness)
4. fig_tail_variance.svg      -- delay distribution + CV growth (tail truncation)
5. fig_controller_baseline.svg-- five-controller reduction vs fixed-time
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "deck" / "assets" / "figures"
STRESS = ROOT / "eval_results_stress"
EVIDENCE = json.loads((ROOT / "scripts" / "evidence.json").read_text(encoding="utf-8"))
STRESS_SUMMARY = json.loads((STRESS / "stress_analysis_summary.json").read_text(encoding="utf-8"))

# ---- palette (matches the deck theme) -------------------------------------
BG = "#0e1217"
INK = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#1f2933"
EDGE = "#334155"
EMERALD = "#34d399"
EMERALD_D = "#10b981"
SLATE_BLUE = "#60a5fa"
AMBER = "#f59e0b"
DQN_GRAY = "#94a3b8"
GREEDY = "#64748b"
FIXED = "#475569"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": EDGE, "grid.color": GRID, "font.size": 12.5,
    "font.family": "DejaVu Sans", "axes.titlecolor": INK, "axes.titleweight": "bold",
    "svg.fonttype": "none",
})


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stress_rows(demand: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in (0, 1):
        p = STRESS / f"D_{demand}" / f"seed_{s}" / "paired_benchmarks.csv"
        if p.exists():
            rows += _read(p)
    return rows


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- #
def fig_sample_efficiency() -> None:
    """Rollout at 100ep recovers fully-converged DQN-at-500ep performance."""
    h100 = EVIDENCE["nominal_100ep"]["per_mode_mean_wait"]
    h500 = EVIDENCE["nominal_500ep"]["per_mode_mean_wait"]
    dqn = [h100["dqn"], h500["dqn"]]
    roll = [h100["rollout_ms"], h500["rollout_ms"]]
    converged = h500["dqn"]

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x = [0, 1]
    w = 0.34
    ax.bar([xi - w / 2 for xi in x], dqn, w, color=DQN_GRAY, label="DQN (amortized only)")
    ax.bar([xi + w / 2 for xi in x], roll, w, color=EMERALD, label="Rollout-DQN (search)")
    ax.axhline(converged, color=SLATE_BLUE, ls="--", lw=1.4)
    ax.text(1.46, converged, "converged DQN\n(500ep) target", color=SLATE_BLUE,
            fontsize=9, va="center", ha="left")

    for xi, v in zip([xi - w / 2 for xi in x], dqn):
        ax.text(xi, v + 120, f"{v:,.0f}", ha="center", color=DQN_GRAY, fontsize=9)
    for xi, v in zip([xi + w / 2 for xi in x], roll):
        ax.text(xi, v + 120, f"{v:,.0f}", ha="center", color=EMERALD, fontsize=9)

    # under-training recovery annotation at 100ep
    recov = (dqn[0] - roll[0]) / (dqn[0] - converged) * 100
    ax.annotate("", xy=(0.17, roll[0]), xytext=(0.17, dqn[0]),
                arrowprops={"arrowstyle": "<->", "color": EMERALD, "lw": 1.6})
    ax.text(0.22, (dqn[0] + roll[0]) / 2, f"recovers {recov:.0f}% of\nunder-training gap",
            color=EMERALD, fontsize=8.5, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(["100 episodes\n(under-trained)", "500 episodes\n(converged)"])
    ax.set_ylabel("Mean waiting time  (veh·s)   ↓ better")
    ax.set_ylim(9500, max(dqn) + 700)
    ax.set_title("Sample efficiency: search compensates for under-training")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.text(0.0, 9560, "paired p=0.019", color=EMERALD, fontsize=8.5, ha="center")
    ax.text(1.0, 9560, "paired p=0.553 (n.s.)", color=MUTED, fontsize=8.5, ha="center")
    _save(fig, "fig_sample_efficiency.svg")


def fig_demand_stress() -> None:
    """Delta(D): the rollout advantage re-opens monotonically under demand."""
    demands = [1000, 1500, 2000]
    pd = STRESS_SUMMARY["per_demand"]
    means = [pd[str(d)]["mean_reduction_pct"] for d in demands]
    ci = [pd[str(d)]["mean_reduction_pct_ci95"] for d in demands]
    lo = [m - c[0] for m, c in zip(means, ci)]
    hi = [c[1] - m for m, c in zip(means, ci)]
    norm = [pd[str(d)].get("mean_normalized_reduction_pct") for d in demands]

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.axhline(0, color=MUTED, ls="--", lw=1)
    ax.errorbar(demands, means, yerr=[lo, hi], marker="o", ms=8, capsize=6, lw=2.4,
                color=EMERALD, label="raw paired reduction (95% CI)")
    ax.plot(demands, norm, marker="x", ls=":", lw=1.6, color=SLATE_BLUE,
            label="per-vehicle normalized")
    for d, m in zip(demands, means):
        ax.annotate(f"{m:+.2f}%", (d, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", color=EMERALD, fontsize=11, fontweight="bold")
    ax.axvspan(1500, 2000, color=AMBER, alpha=0.08)
    ax.text(2000, 1.5, "saturation\nregime", color=AMBER, fontsize=9, ha="right")
    jt = STRESS_SUMMARY["trend_jonckheere_terpstra"]["z"]
    ax.set_title(f"Robustness: gap re-opens under load   (JT trend z = +{jt:.2f})")
    ax.set_xlabel("Demand  (vehicles / episode)")
    ax.set_ylabel("Rollout reduction vs DQN  Δ(D)  [%]")
    ax.set_xticks(demands)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    _save(fig, "fig_demand_stress.svg")


def fig_paired_seed() -> None:
    """60-pair slopegraph at D=2000: robustness across seeds, not just means."""
    rows = _stress_rows(2000)
    keyed: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        keyed[(r["model_run"], r["seed"])][r["mode"]] = float(r["total_wait_vehsec"])
    pairs = [(v["dqn"], v["rollout_ms"]) for v in keyed.values() if "dqn" in v and "rollout_ms" in v]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    wins = 0
    for d, rr in pairs:
        win = rr < d
        wins += win
        ax.plot([0, 1], [d, rr], color=EMERALD if win else AMBER,
                alpha=0.55, lw=1.1, marker="o", ms=3)
    ax.plot([0, 0], [0, 0], color=EMERALD, label=f"rollout wins ({wins}/{len(pairs)})")
    ax.plot([0, 0], [0, 0], color=AMBER, label=f"DQN wins ({len(pairs) - wins}/{len(pairs)})")
    # mean markers
    md = sum(d for d, _ in pairs) / len(pairs)
    mr = sum(r for _, r in pairs) / len(pairs)
    ax.plot([0, 1], [md, mr], color=INK, lw=3, marker="D", ms=8, zorder=5, label="mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DQN", "Rollout-DQN"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Episode waiting time  (veh·s)   ↓ better")
    ax.set_title("Paired robustness at D=2000\n(every line = one shared-seed episode)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _save(fig, "fig_paired_seed.svg")


def fig_tail_variance() -> None:
    """Distribution of delay + CV growth: rollout truncates the bad tail."""
    rows = _stress_rows(2000)
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["mode"]].append(float(r["total_wait_vehsec"]))
    dqn_v, roll_v = by["dqn"], by["rollout_ms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    # left: box + strip
    bp = ax1.boxplot([dqn_v, roll_v], positions=[0, 1], widths=0.5, patch_artist=True,
                     showfliers=False, medianprops={"color": INK, "lw": 2})
    for patch, col in zip(bp["boxes"], [DQN_GRAY, EMERALD]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set_color(EDGE)
    rng = __import__("random")
    rng.seed(0)
    for i, (vals, col) in enumerate([(dqn_v, DQN_GRAY), (roll_v, EMERALD)]):
        xs = [i + (rng.random() - 0.5) * 0.28 for _ in vals]
        ax1.scatter(xs, vals, s=12, color=col, alpha=0.6, zorder=3)
    ax1.scatter([0], [max(dqn_v)], s=70, color=AMBER, zorder=4, marker="v")
    ax1.annotate(f"DQN worst: {max(dqn_v):,.0f}", (0, max(dqn_v)), xytext=(0.18, max(dqn_v)),
                 color=AMBER, fontsize=9, va="center")
    ax1.annotate(f"rollout worst: {max(roll_v):,.0f}", (1, max(roll_v)), xytext=(0.55, max(roll_v) + 6000),
                 color=EMERALD, fontsize=9, va="center")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["DQN", "Rollout-DQN"])
    ax1.set_ylabel("Episode waiting time  (veh·s)")
    ax1.set_title("Delay distribution at D=2000")
    ax1.grid(True, axis="y", alpha=0.3)

    # right: CV vs demand
    demands = [1000, 1500, 2000]
    cv_d = [EVIDENCE["stress"]["per_demand"][str(d)]["dispersion"]["dqn"]["cv"] for d in demands]
    cv_r = [EVIDENCE["stress"]["per_demand"][str(d)]["dispersion"]["rollout_ms"]["cv"] for d in demands]
    ax2.plot(demands, cv_d, marker="o", lw=2.2, color=DQN_GRAY, label="DQN")
    ax2.plot(demands, cv_r, marker="s", lw=2.2, color=EMERALD, label="Rollout-DQN")
    for d, c in zip(demands, cv_d):
        ax2.annotate(f"{c:.3f}", (d, c), textcoords="offset points", xytext=(0, 9), ha="center",
                     color=DQN_GRAY, fontsize=9)
    for d, c in zip(demands, cv_r):
        ax2.annotate(f"{c:.3f}", (d, c), textcoords="offset points", xytext=(0, -15), ha="center",
                     color=EMERALD, fontsize=9)
    ax2.set_xlabel("Demand (vehicles / episode)")
    ax2.set_ylabel("Coefficient of variation  (σ/μ)")
    ax2.set_title("Variance suppression")
    ax2.set_xticks(demands)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, loc="upper left", fontsize=9.5)
    _save(fig, "fig_tail_variance.svg")


def fig_controller_baseline() -> None:
    """Five-controller reduction vs fixed-time (500ep, D=1000)."""
    red = EVIDENCE["nominal_500ep"]["controller_reduction_vs_fixed"]
    order = ["greedy", "dqn", "rollout_1s", "rollout_ms"]
    labels = ["Greedy (g only)", "DQN", "Rollout 1-step", "Rollout multi-step"]
    vals = [red[m] for m in order]
    cols = [GREEDY, DQN_GRAY, EMERALD_D, EMERALD]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.barh(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v - 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="right",
                color=BG, fontweight="bold", fontsize=10)
    ax.set_xlabel("Waiting-time reduction vs fixed-time  [%]   → better")
    ax.set_title("Controller ladder (converged, nominal demand)")
    ax.set_xlim(0, 50)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    _save(fig, "fig_controller_baseline.svg")


def main() -> int:
    print("Generating deck figures ->", FIG_DIR.as_posix())
    fig_sample_efficiency()
    fig_demand_stress()
    fig_paired_seed()
    fig_tail_variance()
    fig_controller_baseline()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
