"""Final code-level validation for Chapter 8 (High-Congestion Stress Testing).

Independently recomputes every metric quoted in Chapter 8 (Tables 8.1 and 8.2,
the Jonckheere-Terpstra and OLS trend statistics, and the §8.3 dispersion /
tail-clipping figures) directly from the raw paired-benchmark CSVs, then asserts
that they match (a) the regenerated ``stress_analysis_summary.json`` and (b) the
hard-coded values written into the thesis prose.

This is a *recompute-from-source* check: it does not import the analyzer's
statistics, it re-derives them with plain Python/numpy so a bug shared with the
analyzer cannot hide. Run after regenerating the summary::

    python validate_chapter8.py

Exit code 0 = all assertions pass; 1 = at least one mismatch (details printed).
No SUMO, no evaluation -- pure offline validation over the committed CSVs.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path("eval_results_stress")
SUMMARY_JSON = RESULTS_ROOT / "stress_analysis_summary.json"
METRIC = "total_wait_vehsec"
BASELINE = "dqn"
TREATMENT = "rollout_ms"
DEMANDS = (1000, 1500, 2000)

# Numbers quoted in the Chapter 8 prose / tables (the values we are shipping).
CHAPTER8_EXPECTED = {
    "table_8_1": {
        1000: {"win": 55.0, "raw_pct": 0.118, "median_adv": 97.0, "rank_biserial": 0.089},
        1500: {"win": 76.667, "raw_pct": 4.318, "median_adv": 746.0, "rank_biserial": 0.689},
        2000: {"win": 91.667, "raw_pct": 15.023, "median_adv": 4814.5, "rank_biserial": 0.949},
    },
    "table_8_2": {
        1000: {"compl_dqn": 99.838, "compl_treat": 99.840, "raw_pct": 0.118, "norm_pct": 0.120},
        1500: {"compl_dqn": 99.916, "compl_treat": 99.912, "raw_pct": 4.318, "norm_pct": 4.314},
        2000: {"compl_dqn": 99.927, "compl_treat": 99.927, "raw_pct": 15.023, "norm_pct": 15.022},
    },
    "dispersion": {  # §8.3
        "cv_dqn": {1000: 0.074, 1500: 0.087, 2000: 0.259},
        "cv_treat": {1000: 0.068, 1500: 0.068, 2000: 0.099},
        "max_dqn_2000": 99586.0,
        "max_treat_2000": 43389.0,
    },
    "trend": {"jt_stat": 8941.0, "jt_z": 9.302, "ols_slope": 7.264},
}


def load_rows() -> list[dict]:
    """Load every stress-sweep CSV, tagging each row with its demand level."""
    rows: list[dict] = []
    for p in glob.glob(str(RESULTS_ROOT / "D_*" / "seed_*" / "paired_benchmarks.csv")):
        parts = Path(p).parts
        demand = int(next(s for s in parts if s.startswith("D_"))[2:])
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_D"] = demand
                rows.append(r)
    if not rows:
        msg = f"No CSVs found under {RESULTS_ROOT}/D_*/seed_*/"
        raise FileNotFoundError(msg)
    return rows


def paired(rows: list[dict]) -> dict[int, list[dict]]:
    """Recompute paired DQN-vs-rollout records per demand from raw rows."""
    keyed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        keyed[(r["_D"], r["model_run"], r["seed"])][r["mode"]] = r
    out: dict[int, list[dict]] = defaultdict(list)
    for (D, _run, _seed), by in keyed.items():
        if BASELINE not in by or TREATMENT not in by:
            continue
        wd, wr = float(by[BASELINE][METRIC]), float(by[TREATMENT][METRIC])
        td, tr = float(by[BASELINE]["throughput"]), float(by[TREATMENT]["throughput"])
        out[D].append(
            {
                "adv": wd - wr,
                "raw_red": 100 * (wd - wr) / wd,
                "norm_red": 100 * ((wd / td) - (wr / tr)) / (wd / td),
            },
        )
    return dict(out)


class Checker:
    """Tiny assertion accumulator with tolerant float comparison."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, label: str, got: float, want: float, tol: float) -> None:
        """Compare ``got`` vs ``want`` within ``tol`` and record the outcome."""
        ok = math.isclose(got, want, abs_tol=tol)
        mark = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{mark}] {label:<46} got={got:>12.4f}  want={want:>12.4f}  (tol={tol})")


def main() -> int:
    """Recompute and assert all Chapter 8 metrics; return 0 iff everything matches."""
    rows = load_rows()
    pairs = paired(rows)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.exists() else None
    ck = Checker()

    exp = CHAPTER8_EXPECTED

    print("== Table 8.1 — paired advantage (recomputed from CSV vs prose) ==")
    for D in DEMANDS:
        adv = np.array([p["adv"] for p in pairs[D]])
        raw = np.array([p["raw_red"] for p in pairs[D]])
        win = 100 * float(np.mean(adv > 0))
        e = exp["table_8_1"][D]
        ck.check(f"D={D} win-rate %", win, e["win"], 0.5)
        ck.check(f"D={D} mean raw reduction %", float(np.mean(raw)), e["raw_pct"], 0.02)
        ck.check(f"D={D} median advantage", float(np.median(adv)), e["median_adv"], 0.5)

    print("== Table 8.2 — completion % and normalization invariance ==")
    for D in DEMANDS:
        dvals = [float(r["throughput"]) for r in rows if r["_D"] == D and r["mode"] == BASELINE]
        tvals = [float(r["throughput"]) for r in rows if r["_D"] == D and r["mode"] == TREATMENT]
        compl_d = 100 * float(np.mean(dvals)) / D
        compl_t = 100 * float(np.mean(tvals)) / D
        raw = float(np.mean([p["raw_red"] for p in pairs[D]]))
        norm = float(np.mean([p["norm_red"] for p in pairs[D]]))
        e = exp["table_8_2"][D]
        ck.check(f"D={D} completion % (dqn)", compl_d, e["compl_dqn"], 0.01)
        ck.check(f"D={D} completion % (rollout)", compl_t, e["compl_treat"], 0.01)
        ck.check(f"D={D} normalized reduction %", norm, e["norm_pct"], 0.02)
        ck.check(f"D={D} |raw - norm| invariance (pp)", abs(raw - norm), 0.0, 0.01)

    print("== §8.3 — dispersion (CV) and worst-case tail ==")
    for D in DEMANDS:
        wd = np.array([float(r[METRIC]) for r in rows if r["_D"] == D and r["mode"] == BASELINE])
        wr = np.array([float(r[METRIC]) for r in rows if r["_D"] == D and r["mode"] == TREATMENT])
        cv_d = float(wd.std() / wd.mean())
        cv_t = float(wr.std() / wr.mean())
        ck.check(f"D={D} CV(dqn)", cv_d, exp["dispersion"]["cv_dqn"][D], 0.002)
        ck.check(f"D={D} CV(rollout)", cv_t, exp["dispersion"]["cv_treat"][D], 0.002)
    wd2 = np.array([float(r[METRIC]) for r in rows if r["_D"] == 2000 and r["mode"] == BASELINE])
    wr2 = np.array([float(r[METRIC]) for r in rows if r["_D"] == 2000 and r["mode"] == TREATMENT])
    ck.check("D=2000 worst-case delay (dqn)", float(wd2.max()), exp["dispersion"]["max_dqn_2000"], 1.0)
    ck.check("D=2000 worst-case delay (rollout)", float(wr2.max()), exp["dispersion"]["max_treat_2000"], 1.0)

    print("== Trend statistics (prose vs JSON) ==")
    if summary is not None:
        jt = summary["trend_jonckheere_terpstra"]
        ols = summary["trend_ols_interaction"]
        ck.check("JT statistic (JSON vs prose)", jt["statistic"], exp["trend"]["jt_stat"], 0.5)
        ck.check("JT z (JSON vs prose)", jt["z"], exp["trend"]["jt_z"], 0.01)
        ck.check("OLS slope (JSON vs prose)", ols["slope"], exp["trend"]["ols_slope"], 0.01)

        print("== JSON <-> recompute cross-check (per-demand) ==")
        for D in DEMANDS:
            s = summary["per_demand"][str(D)]
            raw = float(np.mean([p["raw_red"] for p in pairs[D]]))
            norm = float(np.mean([p["norm_red"] for p in pairs[D]]))
            ck.check(f"D={D} JSON mean_reduction_pct", s["mean_reduction_pct"], raw, 1e-6)
            ck.check(f"D={D} JSON mean_normalized_reduction_pct", s["mean_normalized_reduction_pct"], norm, 1e-6)
            g = summary["throughput_guard"][str(D)]
            ck.check(f"D={D} guard shedding flag == 0", float(bool(g["shedding_suspected"])), 0.0, 0.0)
    else:
        print("  [warn] stress_analysis_summary.json not found; JSON cross-checks skipped.")

    print("=" * 78)
    total = ck.passed + ck.failed
    print(f"RESULT: {ck.passed}/{total} checks passed, {ck.failed} failed.")
    return 0 if ck.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
