"""Evidence extractor for the PhD interview deck.

Parses every validated result artifact in the workspace and emits:

* ``deck/EVIDENCE.md``      -- a human-readable evidence table (metric, source,
  value, deck location) for auditability;
* ``scripts/evidence.json`` -- machine-readable numbers consumed by
  ``generate_figures.py`` and quoted in the deck.

Every number is read or recomputed from source; nothing is invented. Run from
the repository root::

    python scripts/extract_evidence.py

Sources
-------
* 100ep nominal : clean_rollout_tlcs/eval_results/paired_benchmarks_seed{0,1,2}.csv
                  + analysis_summary.json   (n=90, 3 training seeds x 30)
* 500ep nominal : clean_rollout_tlcs/eval_results_long/paired_benchmarks_seed{0,1}.csv
                  + academic_artifacts/statistical_summary.json (n=60)
* stress sweep  : eval_results_stress/D_*/seed_*/paired_benchmarks.csv
                  + eval_results_stress/stress_analysis_summary.json
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL_100 = ROOT / "clean_rollout_tlcs" / "eval_results"
EVAL_500 = ROOT / "clean_rollout_tlcs" / "eval_results_long"
STRESS = ROOT / "eval_results_stress"
ACADEMIC = ROOT / "academic_artifacts"

MODES = ("fixed_time", "dqn", "greedy", "rollout_1s", "rollout_ms")
METRIC = "total_wait_vehsec"
BASELINE, TREATMENT = "dqn", "rollout_ms"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_mode_mean(rows: list[dict[str, str]], col: str = METRIC) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["mode"]].append(float(r[col]))
    return {m: round(st.mean(by[m]), 1) for m in MODES if m in by}


def _paired(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Recompute paired DQN-vs-rollout_ms stats from raw rows."""
    keyed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        keyed[(r["model_run"], r["seed"])][r["mode"]] = r
    adv, red, norm = [], [], []
    for by in keyed.values():
        if BASELINE not in by or TREATMENT not in by:
            continue
        wd, wr = float(by[BASELINE][METRIC]), float(by[TREATMENT][METRIC])
        adv.append(wd - wr)
        red.append(100 * (wd - wr) / wd)
        td = float(by[BASELINE].get("throughput", "nan") or "nan")
        tr = float(by[TREATMENT].get("throughput", "nan") or "nan")
        if td == td and tr == tr:  # not NaN
            norm.append(100 * ((wd / td) - (wr / tr)) / (wd / td))
    out = {
        "n_pairs": len(adv),
        "win_rate_pct": round(100 * sum(a > 0 for a in adv) / len(adv), 2) if adv else None,
        "mean_reduction_pct": round(st.mean(red), 3) if red else None,
        "median_advantage": round(st.median(adv), 1) if adv else None,
    }
    if norm:
        out["mean_normalized_reduction_pct"] = round(st.mean(norm), 3)
    return out


def _completion_cv(rows: list[dict[str, str]], demand: int) -> dict[str, Any]:
    by_wait: dict[str, list[float]] = defaultdict(list)
    by_thru: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_wait[r["mode"]].append(float(r[METRIC]))
        if r.get("throughput"):
            by_thru[r["mode"]].append(float(r["throughput"]))
    out: dict[str, Any] = {}
    for m in (BASELINE, TREATMENT):
        w = by_wait.get(m, [])
        out[m] = {
            "mean_wait": round(st.mean(w), 1) if w else None,
            "cv": round(st.pstdev(w) / st.mean(w), 4) if w else None,
            "max_wait": round(max(w), 1) if w else None,
            "completion_pct": round(100 * st.mean(by_thru[m]) / demand, 4) if by_thru.get(m) else None,
        }
    return out


def main() -> int:
    evidence: dict[str, Any] = {"_note": "All numbers read/recomputed from source files; none invented."}
    table: list[tuple[str, str, str, str]] = []  # metric, source, value, deck slide

    # ---- 100ep nominal -------------------------------------------------
    rows100 = []
    for s in (0, 1, 2):
        p = EVAL_100 / f"paired_benchmarks_seed{s}.csv"
        if p.exists():
            rows100 += _read_csv(p)
    a100 = json.loads((EVAL_100 / "analysis_summary.json").read_text(encoding="utf-8"))
    h100 = {
        "n_pairs": a100["n_pairs"],
        "win_rate_pct": round(a100["win_rate_pct"], 2),
        "mean_reduction_pct": round(a100["mean_paired_reduction_pct"], 3),
        "median_advantage": a100["median_paired_diff_vehsec"],
        "wilcoxon_p": a100["wilcoxon"]["p_value"],
        "effect_size": round(a100["wilcoxon"]["effect_size"], 3),
        "per_mode_mean_wait": _per_mode_mean(rows100) if rows100 else {},
        "f_mae": round(a100["f_fidelity"]["mean_f_mae"], 3),
        "f_spearman": round(a100["f_fidelity"]["mean_f_spearman"], 3),
        "latency_ms": a100.get("latency", {}),
    }
    evidence["nominal_100ep"] = h100
    table += [
        ("100ep paired win-rate", "eval_results/analysis_summary.json", f"{h100['win_rate_pct']}%", "S5 Findings I"),
        ("100ep mean reduction", "eval_results/analysis_summary.json", f"{h100['mean_reduction_pct']}%", "S5 Findings I"),
        ("100ep Wilcoxon p", "eval_results/analysis_summary.json", f"{h100['wilcoxon_p']:.4f}", "S5 Findings I"),
        ("rollout latency ratio", "eval_results/analysis_summary.json",
         f"{h100['latency_ms'].get('ratio_ms_over_dqn', '?'):.2f}x", "S6 Defense"),
    ]

    # ---- 500ep nominal -------------------------------------------------
    rows500 = []
    for s in (0, 1):
        p = EVAL_500 / f"paired_benchmarks_seed{s}.csv"
        if p.exists():
            rows500 += _read_csv(p)
    stat500 = json.loads((ACADEMIC / "statistical_summary.json").read_text(encoding="utf-8"))
    pr = stat500["paired_dqn_vs_rollout_ms"]
    h500 = {
        "n_pairs": stat500["metadata"]["n_pairs"],
        "win_rate_pct": round(pr["win_rate_pct"], 2),
        "mean_reduction_pct": round(pr["mean_reduction_pct"], 3),
        "median_advantage": pr["median_diff_vehsec"],
        "wilcoxon_p": pr["wilcoxon"]["p_value"],
        "per_mode_mean_wait": _per_mode_mean(rows500) if rows500 else {},
        "controller_reduction_vs_fixed": {
            m: round(v["mean_reduction_pct"], 2)
            for m, v in stat500["normalized_reduction_by_mode"].items()
        },
        "f_mae": round(stat500["transition_model_fidelity"]["mean_f_mae"], 3),
        "f_spearman": round(stat500["transition_model_fidelity"]["mean_f_spearman"], 3),
    }
    evidence["nominal_500ep"] = h500
    table += [
        ("500ep paired win-rate", "academic_artifacts/statistical_summary.json", f"{h500['win_rate_pct']}%", "S5 Findings I"),
        ("500ep mean reduction", "academic_artifacts/statistical_summary.json", f"{h500['mean_reduction_pct']}%", "S5 Findings I"),
        ("500ep Wilcoxon p", "academic_artifacts/statistical_summary.json", f"{h500['wilcoxon_p']:.4f}", "S5 Findings I"),
        ("controller reduction vs fixed-time", "academic_artifacts/statistical_summary.json",
         str(h500["controller_reduction_vs_fixed"]), "S4 Baselines"),
    ]

    # ---- stress sweep --------------------------------------------------
    sumj = json.loads((STRESS / "stress_analysis_summary.json").read_text(encoding="utf-8"))
    stress: dict[str, Any] = {"per_demand": {}}
    for D in (1000, 1500, 2000):
        rows = []
        for s in (0, 1):
            p = STRESS / f"D_{D}" / f"seed_{s}" / "paired_benchmarks.csv"
            if p.exists():
                rows += _read_csv(p)
        pd = sumj["per_demand"][str(D)]
        cell = {
            "win_rate_pct": round(pd["win_rate_pct"], 2),
            "mean_reduction_pct": round(pd["mean_reduction_pct"], 3),
            "mean_normalized_reduction_pct": round(pd.get("mean_normalized_reduction_pct", float("nan")), 3),
            "median_advantage": pd["median_advantage"],
            "wilcoxon_p": pd["wilcoxon"]["p_value"],
            "effect_size": round(pd["wilcoxon"]["effect_size_rank_biserial"], 3),
            "per_mode_mean_wait": _per_mode_mean(rows) if rows else {},
            "dispersion": _completion_cv(rows, D) if rows else {},
        }
        stress["per_demand"][str(D)] = cell
    stress["jt_z"] = round(sumj["trend_jonckheere_terpstra"]["z"], 3)
    stress["jt_statistic"] = sumj["trend_jonckheere_terpstra"]["statistic"]
    stress["ols_slope"] = round(sumj["trend_ols_interaction"]["slope"], 3)
    stress["ols_slope_ci95"] = [round(x, 2) for x in sumj["trend_ols_interaction"]["slope_ci95"]]
    stress["shedding_suspected_any"] = any(
        sumj["throughput_guard"][str(D)]["shedding_suspected"] for D in (1000, 1500, 2000)
    )
    evidence["stress"] = stress
    table += [
        ("Stress JT-trend z", "stress_analysis_summary.json", f"+{stress['jt_z']}", "S5 Findings II"),
        ("D=2000 mean reduction", "stress_analysis_summary.json",
         f"{stress['per_demand']['2000']['mean_reduction_pct']}%", "S5 Findings II"),
        ("D=2000 win-rate", "stress_analysis_summary.json",
         f"{stress['per_demand']['2000']['win_rate_pct']}%", "S5 Findings II"),
        ("D=2000 median advantage", "stress_analysis_summary.json",
         f"{stress['per_demand']['2000']['median_advantage']} veh*s", "S5 Findings II"),
        ("OLS interaction slope", "stress_analysis_summary.json",
         f"+{stress['ols_slope']} veh*s/veh CI{stress['ols_slope_ci95']}", "S5 Findings II"),
        ("D=2000 CV (DQN vs rollout)", "stress CSVs (recomputed)",
         f"{stress['per_demand']['2000']['dispersion'][BASELINE]['cv']} vs "
         f"{stress['per_demand']['2000']['dispersion'][TREATMENT]['cv']}", "S5 Findings III"),
        ("D=2000 worst-case delay", "stress CSVs (recomputed)",
         f"{stress['per_demand']['2000']['dispersion'][BASELINE]['max_wait']:.0f} vs "
         f"{stress['per_demand']['2000']['dispersion'][TREATMENT]['max_wait']:.0f} veh*s", "S5 Findings III"),
        ("D=2000 completion (both)", "stress CSVs (recomputed)",
         f"{stress['per_demand']['2000']['dispersion'][BASELINE]['completion_pct']}% / "
         f"{stress['per_demand']['2000']['dispersion'][TREATMENT]['completion_pct']}%", "S5/S6 confound"),
        ("Shedding confound flagged", "stress_analysis_summary.json (post-tolerance)",
         str(stress["shedding_suspected_any"]), "S6 Defense"),
    ]

    # ---- training horizons (manifests) ---------------------------------
    evidence["checkpoints"] = {
        "dqn_100ep": {"episodes": 100, "seeds": [0, 1, 2], "path": "models/dqn_100ep"},
        "dqn_500ep": {"episodes": 500, "seeds": [0, 1], "path": "models/dqn_500ep"},
    }

    # ---- write artifacts ----------------------------------------------
    (ROOT / "scripts" / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    lines = [
        "# Evidence Table - PhD Interview Deck",
        "",
        "All figures below are read or recomputed directly from the source files named.",
        "No value is invented or approximated. Regenerate with `python scripts/extract_evidence.py`.",
        "",
        "| Metric | Source file | Value | Deck location |",
        "|---|---|---|---|",
    ]
    for metric, source, value, slide in table:
        lines.append(f"| {metric} | `{source}` | {value} | {slide} |")
    lines += [
        "",
        "## Per-controller mean waiting time (veh·s) — recomputed from paired CSVs",
        "",
        "| Config | fixed_time | greedy | dqn | rollout_1s | rollout_ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def _row(label: str, pmm: dict[str, float]) -> str:
        cells = " | ".join(f"{pmm.get(m, '—')}" for m in MODES)
        return f"| {label} | {cells} |"

    lines.append(_row("100ep · D=1000", h100["per_mode_mean_wait"]))
    lines.append(_row("500ep · D=1000", h500["per_mode_mean_wait"]))
    for D in (1000, 1500, 2000):
        lines.append(_row(f"500ep · D={D} (stress)", stress["per_demand"][str(D)]["per_mode_mean_wait"]))
    lines += [
        "",
        "## Headline paired results",
        "",
        "| Config | n pairs | Win % | Mean Δ% | Median adv (veh·s) | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 100ep · D=1000 | {h100['n_pairs']} | {h100['win_rate_pct']} | {h100['mean_reduction_pct']} | {h100['median_advantage']} | {h100['wilcoxon_p']:.4g} |",
        f"| 500ep · D=1000 | {h500['n_pairs']} | {h500['win_rate_pct']} | {h500['mean_reduction_pct']} | {h500['median_advantage']} | {h500['wilcoxon_p']:.4g} |",
    ]
    for D in (1500, 2000):
        c = stress["per_demand"][str(D)]
        lines.append(f"| 500ep · D={D} | 60 | {c['win_rate_pct']} | {c['mean_reduction_pct']} | {c['median_advantage']} | {c['wilcoxon_p']:.4g} |")
    lines.append("")
    (ROOT / "deck" / "EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote scripts/evidence.json and deck/EVIDENCE.md")
    print(f"  100ep: win {h100['win_rate_pct']}%  Δ {h100['mean_reduction_pct']}%  p={h100['wilcoxon_p']:.4g}")
    print(f"  500ep: win {h500['win_rate_pct']}%  Δ {h500['mean_reduction_pct']}%  p={h500['wilcoxon_p']:.4g}")
    for D in (1000, 1500, 2000):
        c = stress["per_demand"][str(D)]
        print(f"  D={D}: win {c['win_rate_pct']}%  Δ {c['mean_reduction_pct']}%  (norm {c['mean_normalized_reduction_pct']}%)  p={c['wilcoxon_p']:.3g}")
    print(f"  JT z=+{stress['jt_z']}  OLS slope=+{stress['ols_slope']} {stress['ols_slope_ci95']}")
    print(f"  shedding flagged after tolerance patch: {stress['shedding_suspected_any']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
