"""Phase 2.1 analysis: does rollout re-open a gap as demand rises?

Consumes the paired benchmark CSVs produced by the demand sweep (laid out by
``stress_sweep.py`` under ``eval_results_stress/D_<D>/<model_run>/``), and tests
the central Phase 2.1 hypothesis:

    H0: the paired rollout advantage over DQN is flat across demand.
    H1: the advantage *increases* with demand (ordered alternative).

What it computes
----------------
* Per demand level: win rate, mean paired reduction %, median paired difference,
  a self-contained tie-corrected Wilcoxon signed-rank test, and 95% bootstrap
  confidence intervals.
* Across levels: a **Jonckheere-Terpstra** trend test (the canonical ordered-
  alternative test) on the per-pair advantage, plus an OLS slope of advantage vs
  demand as a robustness companion (the "demand x method interaction").
* Anti-truncation-bias guards: aggregate **throughput** and **residual_vehicles**
  per (demand, mode) IF those columns are present, and a flag warning if the
  treatment's delay win coincides with *lower* throughput (i.e. a win bought by
  shedding demand rather than serving it).
* A Delta(D) figure: mean paired reduction % vs demand with 95% bootstrap CI
  error bars, saved as a vector PDF.

No SciPy dependency: Wilcoxon, Jonckheere-Terpstra and the normal CDF are
implemented locally (matching the repo's self-contained style in eval.py).

Usage::

    python analyze_stress_results.py --results-root eval_results_stress
    python analyze_stress_results.py --metric total_wait_vehsec --treatment rollout_ms
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Defaults / schema
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_ROOT = Path("eval_results_stress")
DEFAULT_METRIC = "total_wait_vehsec"  # lower is better -> advantage = baseline - treatment
DEFAULT_BASELINE = "dqn"
DEFAULT_TREATMENT = "rollout_ms"

# Optional anti-truncation-bias columns (present only if the eval harness was
# extended to log them; absent in the frozen schema -- handled gracefully).
THROUGHPUT_COL = "throughput"
RESIDUAL_COL = "residual_vehicles"

# Demand level encoded in the directory name, e.g. ".../D_1500/...".
_DEMAND_DIR_RE = re.compile(r"D_(\d+)")

# Shedding-guard tolerance: a delay "win" is only flagged as possibly bought by
# demand-shedding when the treatment clears *meaningfully* fewer vehicles than
# the baseline -- more than max(0.5% of baseline throughput, 1 vehicle). Without
# this window a sub-vehicle rounding difference (e.g. -0.017 veh at D=2000) trips
# a false positive. See Chapter 8.3: throughput is near-identical across modes.
SHEDDING_TOLERANCE_FRAC = 0.005
SHEDDING_TOLERANCE_VEH = 1.0

# Reproducible bootstrap stream.
_BOOTSTRAP_SEED = 12345
_DEFAULT_BOOTSTRAP = 5000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _coerce_float(value: Any) -> float | None:
    """Parse a CSV cell to float, mapping blanks/None to ``None``.

    Args:
        value: Raw CSV cell.

    Returns:
        The float value, or ``None`` if empty/unparseable.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _demand_from_path(path: Path) -> int | None:
    """Extract the demand level from a ``D_<D>`` component of a path.

    Args:
        path: A CSV path somewhere under ``.../D_<D>/...``.

    Returns:
        The integer demand, or ``None`` if no ``D_<D>`` component is found.
    """
    for part in path.parts:
        match = _DEMAND_DIR_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def load_rows(results_root: Path, metric: str) -> tuple[list[dict[str, Any]], bool, bool]:
    """Load every paired-benchmark CSV under the results root.

    Args:
        results_root: Root containing ``D_<D>/<model_run>/paired_benchmarks*.csv``.
        metric: Metric column to extract as the comparison value.

    Returns:
        A tuple ``(rows, has_throughput, has_residual)`` where ``rows`` is a list
        of dicts with keys ``demand, model_run, seed, mode, value`` (plus
        ``throughput`` / ``residual`` when available).

    Raises:
        FileNotFoundError: If no CSVs are found.
        KeyError: If the requested metric column is missing from a CSV.
    """
    csv_paths = sorted(results_root.glob("D_*/**/paired_benchmarks*.csv"))
    if not csv_paths:
        msg = (
            f"No 'paired_benchmarks*.csv' found under {results_root}/D_*/. "
            "Run the stress_sweep.py commands first."
        )
        raise FileNotFoundError(msg)

    rows: list[dict[str, Any]] = []
    has_throughput = False
    has_residual = False

    for path in csv_paths:
        demand = _demand_from_path(path)
        if demand is None:
            print(f"[warn] skipping (no D_<D> in path): {path}", file=sys.stderr)
            continue

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or metric not in reader.fieldnames:
                msg = f"Metric column '{metric}' not found in {path} (columns: {reader.fieldnames})"
                raise KeyError(msg)
            file_has_tp = THROUGHPUT_COL in reader.fieldnames
            file_has_res = RESIDUAL_COL in reader.fieldnames
            has_throughput = has_throughput or file_has_tp
            has_residual = has_residual or file_has_res

            for raw in reader:
                value = _coerce_float(raw.get(metric))
                if value is None:
                    continue
                rows.append(
                    {
                        "demand": demand,
                        "model_run": raw.get("model_run", path.parent.name),
                        "seed": raw.get("seed"),
                        "mode": raw.get("mode"),
                        "value": value,
                        "throughput": _coerce_float(raw.get(THROUGHPUT_COL)) if file_has_tp else None,
                        "residual": _coerce_float(raw.get(RESIDUAL_COL)) if file_has_res else None,
                    },
                )

    return rows, has_throughput, has_residual


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def build_pairs(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
) -> dict[int, list[dict[str, float]]]:
    """Pair baseline vs treatment within each (demand, model_run, seed).

    The advantage is ``baseline_value - treatment_value`` so that, for a
    lower-is-better metric (delay), a positive advantage means the treatment
    (rollout) is better.

    Args:
        rows: Loaded rows from :func:`load_rows`.
        baseline: Baseline mode name (e.g. ``"dqn"``).
        treatment: Treatment mode name (e.g. ``"rollout_ms"``).

    Returns:
        Mapping ``demand -> list of pair dicts`` with keys ``advantage``,
        ``reduction_pct``, ``baseline``, ``treatment``.
    """
    # key -> {mode: row}
    keyed: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["demand"], str(row["model_run"]), str(row["seed"]))
        keyed[key][str(row["mode"])] = row

    pairs_by_demand: dict[int, list[dict[str, float]]] = defaultdict(list)
    for (demand, _run, _seed), by_mode in keyed.items():
        if baseline not in by_mode or treatment not in by_mode:
            continue
        b_row = by_mode[baseline]
        t_row = by_mode[treatment]
        b = b_row["value"]
        t = t_row["value"]
        advantage = b - t
        reduction_pct = (advantage / b * 100.0) if b != 0 else float("nan")
        pair: dict[str, float] = {
            "advantage": advantage,
            "reduction_pct": reduction_pct,
            "baseline": b,
            "treatment": t,
        }
        # Throughput-normalized advantage (delay per completed vehicle): proves
        # the gap is not an artifact of differential throughput (Table 8.2).
        b_tp = b_row.get("throughput")
        t_tp = t_row.get("throughput")
        if b_tp and t_tp:
            b_norm = b / b_tp
            t_norm = t / t_tp
            pair["norm_advantage"] = b_norm - t_norm
            pair["norm_reduction_pct"] = (
                (b_norm - t_norm) / b_norm * 100.0 if b_norm != 0 else float("nan")
            )
        pairs_by_demand[demand].append(pair)
    return dict(sorted(pairs_by_demand.items()))


# ---------------------------------------------------------------------------
# Statistics (self-contained; no SciPy)
# ---------------------------------------------------------------------------
def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function.

    Args:
        x: Quantile.

    Returns:
        ``P(Z <= x)``.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _average_ranks(values: NDArray) -> NDArray:
    """Average (tie-corrected) ranks of ``values``, 1-based.

    Args:
        values: 1-D array.

    Returns:
        Array of average ranks.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: NDArray) -> dict[str, Any]:
    """Tie-corrected Wilcoxon signed-rank test (two-sided normal approximation).

    Mirrors the method recorded in the repo's analysis artifacts
    (``normal_approx_tie_corrected``).

    Args:
        diffs: Paired differences (zeros are dropped, per the standard test).

    Returns:
        Dict with ``p_value``, ``z``, ``w_plus``, ``w_minus``, ``n_nonzero`` and
        the rank-biserial effect size.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return {
            "p_value": float("nan"),
            "z": float("nan"),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "n_nonzero": 0,
            "effect_size_rank_biserial": float("nan"),
            "method": "normal_approx_tie_corrected",
        }

    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0

    # Tie correction on |d|.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    var_w -= tie_term / 48.0

    if var_w <= 0:
        z = 0.0
        p = 1.0
    else:
        # Continuity-corrected z on W+.
        diff = w_plus - mean_w
        cc = math.copysign(0.5, diff) if diff != 0 else 0.0
        z = (diff - cc) / math.sqrt(var_w)
        p = 2.0 * (1.0 - normal_cdf(abs(z)))

    total = w_plus + w_minus
    effect = (w_plus - w_minus) / total if total > 0 else float("nan")
    return {
        "p_value": p,
        "z": z,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "n_nonzero": n,
        "effect_size_rank_biserial": effect,
        "method": "normal_approx_tie_corrected",
    }


def jonckheere_terpstra(groups: list[NDArray]) -> dict[str, Any]:
    """Jonckheere-Terpstra trend test for an increasing ordered alternative.

    Tests H0 (all group distributions equal) against H1 (stochastically
    increasing across groups in the given order). Groups must be supplied in the
    hypothesised ascending order (here: ascending demand). Uses the standard
    normal approximation; the variance term assumes no/limited ties (documented
    approximation).

    Args:
        groups: List of 1-D arrays, one per ordered group.

    Returns:
        Dict with the JT statistic, its null mean/variance, ``z`` and the
        one-sided p-value for an increasing trend.
    """
    k = len(groups)
    sizes = [g.size for g in groups]
    n_total = int(sum(sizes))
    if k < 2 or n_total < 2:
        return {"statistic": float("nan"), "z": float("nan"), "p_value_increasing": float("nan"), "n_groups": k}

    # JT statistic: for every ordered pair of groups (i < j), count how often a
    # value in group j exceeds one in group i (ties count 0.5).
    jt = 0.0
    for i in range(k):
        gi = groups[i]
        for j in range(i + 1, k):
            gj = groups[j]
            # Broadcast compare: U_ij = #{(x in gi, y in gj): y > x} + 0.5 ties.
            cmp = gj[:, None] - gi[None, :]
            jt += float(np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0))

    sum_sq = sum(s * s for s in sizes)
    mean_jt = (n_total**2 - sum_sq) / 4.0
    var_jt = (
        n_total**2 * (2 * n_total + 3) - sum(s**2 * (2 * s + 3) for s in sizes)
    ) / 72.0

    if var_jt <= 0:
        z = 0.0
        p_inc = 1.0
    else:
        z = (jt - mean_jt) / math.sqrt(var_jt)
        p_inc = 1.0 - normal_cdf(z)  # one-sided: increasing trend

    return {
        "statistic": jt,
        "null_mean": mean_jt,
        "null_var": var_jt,
        "z": z,
        "p_value_increasing": p_inc,
        "n_groups": k,
        "group_sizes": sizes,
        "note": "normal approximation; variance assumes negligible ties",
    }


def ols_trend(demands: NDArray, advantages: NDArray) -> dict[str, Any]:
    """OLS slope of per-pair advantage on demand (demand x method interaction).

    Args:
        demands: Per-pair demand level.
        advantages: Per-pair advantage.

    Returns:
        Dict with slope, intercept, slope SE and a 95% normal-approx CI.
    """
    x = np.asarray(demands, dtype=float)
    y = np.asarray(advantages, dtype=float)
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return {"slope": float("nan"), "intercept": float("nan"), "slope_ci95": [float("nan"), float("nan")]}

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sse = float(np.sum(resid**2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_slope = math.sqrt(sse / (n - 2) / sxx) if sxx > 0 else float("nan")
    half = 1.96 * se_slope
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": se_slope,
        "slope_ci95": [float(slope - half), float(slope + half)],
        "interpretation": "advantage gained per +1 vehicle of demand (positive => gap widens with demand)",
    }


def bootstrap_ci(
    values: NDArray,
    statistic: str = "mean",
    n_boot: int = _DEFAULT_BOOTSTRAP,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean or median of ``values``.

    Args:
        values: 1-D sample (NaNs are dropped).
        statistic: ``"mean"`` or ``"median"``.
        n_boot: Number of bootstrap resamples.

    Returns:
        ``(low, high)`` 2.5/97.5 percentile bounds; ``(nan, nan)`` if empty.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    samples = v[idx]
    stat = np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


# ---------------------------------------------------------------------------
# Per-demand summaries and throughput guard
# ---------------------------------------------------------------------------
def summarize_by_demand(
    pairs_by_demand: dict[int, list[dict[str, float]]],
    n_boot: int,
) -> dict[int, dict[str, Any]]:
    """Compute per-demand paired statistics.

    Args:
        pairs_by_demand: Output of :func:`build_pairs`.
        n_boot: Bootstrap resample count.

    Returns:
        Mapping ``demand -> summary dict``.
    """
    summary: dict[int, dict[str, Any]] = {}
    for demand, pairs in pairs_by_demand.items():
        adv = np.array([p["advantage"] for p in pairs], dtype=float)
        red = np.array([p["reduction_pct"] for p in pairs], dtype=float)
        red_valid = red[~np.isnan(red)]

        red_lo, red_hi = bootstrap_ci(red_valid, "mean", n_boot)
        med_lo, med_hi = bootstrap_ci(adv, "median", n_boot)

        entry: dict[str, Any] = {
            "n_pairs": int(adv.size),
            "win_rate_pct": float(np.mean(adv > 0) * 100.0) if adv.size else float("nan"),
            "mean_reduction_pct": float(np.mean(red_valid)) if red_valid.size else float("nan"),
            "mean_reduction_pct_ci95": [red_lo, red_hi],
            "median_advantage": float(np.median(adv)) if adv.size else float("nan"),
            "median_advantage_ci95": [med_lo, med_hi],
            "mean_advantage": float(np.mean(adv)) if adv.size else float("nan"),
            "wilcoxon": wilcoxon_signed_rank(adv),
        }

        # Throughput-normalized reduction (Table 8.2): present iff throughput was
        # logged for both modes. Should track the raw reduction to <0.01 points.
        norm_red = np.array(
            [p["norm_reduction_pct"] for p in pairs if "norm_reduction_pct" in p],
            dtype=float,
        )
        norm_red_valid = norm_red[~np.isnan(norm_red)]
        if norm_red_valid.size:
            nlo, nhi = bootstrap_ci(norm_red_valid, "mean", n_boot)
            norm_adv = np.array(
                [p["norm_advantage"] for p in pairs if "norm_advantage" in p],
                dtype=float,
            )
            entry["mean_normalized_reduction_pct"] = float(np.mean(norm_red_valid))
            entry["mean_normalized_reduction_pct_ci95"] = [nlo, nhi]
            entry["median_normalized_advantage"] = float(np.median(norm_adv)) if norm_adv.size else float("nan")

        summary[demand] = entry
    return summary


def throughput_guard(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    """Aggregate throughput/residual and flag delay wins bought by shedding.

    Args:
        rows: Loaded rows (must contain non-null ``throughput``/``residual``).
        baseline: Baseline mode.
        treatment: Treatment mode.

    Returns:
        Dict keyed by demand with per-mode mean throughput/residual and a
        ``shedding_suspected`` flag. The flag is raised only when the treatment's
        throughput deficit exceeds ``max(SHEDDING_TOLERANCE_FRAC * baseline,
        SHEDDING_TOLERANCE_VEH)``, so sub-vehicle rounding noise does not trip it.
    """
    agg: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("throughput") is None and row.get("residual") is None:
            continue
        agg[row["demand"]][f"{row['mode']}::tp"].append(row.get("throughput") or float("nan"))
        agg[row["demand"]][f"{row['mode']}::res"].append(row.get("residual") or float("nan"))

    out: dict[str, Any] = {}
    for demand in sorted(agg):
        modes = agg[demand]

        def _mean(key: str) -> float:
            vals = np.array(modes.get(key, [np.nan]), dtype=float)
            vals = vals[~np.isnan(vals)]
            return float(np.mean(vals)) if vals.size else float("nan")

        base_tp = _mean(f"{baseline}::tp")
        treat_tp = _mean(f"{treatment}::tp")
        if math.isnan(base_tp) or math.isnan(treat_tp):
            deficit = float("nan")
            tolerance = float("nan")
            shedding: bool | None = None
        else:
            deficit = base_tp - treat_tp  # >0 => treatment cleared fewer vehicles
            tolerance = max(SHEDDING_TOLERANCE_FRAC * base_tp, SHEDDING_TOLERANCE_VEH)
            shedding = bool(deficit > tolerance)
        out[str(demand)] = {
            "per_mode_mean_throughput": {
                m.split("::")[0]: _mean(m) for m in modes if m.endswith("::tp")
            },
            "per_mode_mean_residual": {
                m.split("::")[0]: _mean(m) for m in modes if m.endswith("::res")
            },
            "baseline_throughput": base_tp,
            "treatment_throughput": treat_tp,
            "throughput_deficit": deficit,
            "shedding_tolerance": tolerance,
            "shedding_suspected": shedding,
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_delta_curve(
    summary: dict[int, dict[str, Any]],
    out_path: Path,
    baseline: str,
    treatment: str,
) -> bool:
    """Plot Delta(D): mean paired reduction % vs demand with bootstrap CI bars.

    Args:
        summary: Per-demand summary from :func:`summarize_by_demand`.
        out_path: Destination PDF path.
        baseline: Baseline mode name (for the title).
        treatment: Treatment mode name (for the title).

    Returns:
        ``True`` if the figure was written, ``False`` if matplotlib is missing.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable ({exc}); skipping Delta(D) plot.", file=sys.stderr)
        return False

    demands = sorted(summary)
    means = [summary[d]["mean_reduction_pct"] for d in demands]
    lo = [summary[d]["mean_reduction_pct"] - summary[d]["mean_reduction_pct_ci95"][0] for d in demands]
    hi = [summary[d]["mean_reduction_pct_ci95"][1] - summary[d]["mean_reduction_pct"] for d in demands]
    norm = [summary[d].get("mean_normalized_reduction_pct") for d in demands]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axhline(0.0, color="#a83a3a", lw=1.0, ls="--", label="no advantage")
    ax.errorbar(
        demands, means, yerr=[lo, hi],
        marker="o", capsize=5, lw=2, color="#245f9f", label=f"{treatment} vs {baseline} (raw)",
    )
    # Overlay throughput-normalized reduction to show the curves coincide.
    if all(n is not None for n in norm):
        ax.plot(demands, norm, marker="x", ls=":", lw=1.5, color="#087b78",
                label="per-vehicle normalized")
    # Value labels (raw %) at each point.
    for d, m in zip(demands, means):
        ax.annotate(f"{m:+.2f}%", (d, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#245f9f")
    ax.set_xlabel("Demand  (n_cars_generated, vehicles/episode)")
    ax.set_ylabel("Mean paired waiting-time reduction  Δ(D)  [%]")
    ax.set_title("Δ(D): predictive rollout re-opens the gap under congestion")
    ax.set_xticks(demands)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    if all(n is not None for n in norm):
        ax.text(0.99, 0.02,
                "raw and per-vehicle-normalized curves coincide (<0.01 pp):\n"
                "the advantage is not a throughput artifact",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#5e6875")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return True


def compute_dispersion(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[int, dict[str, Any]]:
    """Compute per-(demand, mode) dispersion of the metric across episodes.

    Args:
        rows: Loaded rows from :func:`load_rows`.
        modes: Modes to summarize (e.g. baseline and treatment).

    Returns:
        Mapping ``demand -> mode -> {mean, std, cv, max, min, n}`` using the
        population standard deviation (ddof=0).
    """
    by: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["mode"] in modes:
            by[row["demand"]][str(row["mode"])].append(row["value"])

    out: dict[int, dict[str, Any]] = {}
    for demand in sorted(by):
        out[demand] = {}
        for mode in modes:
            v = np.array(by[demand].get(mode, []), dtype=float)
            if v.size == 0:
                continue
            mean = float(v.mean())
            std = float(v.std())  # population std, matches the Chapter 8 CV figures
            out[demand][mode] = {
                "mean": mean,
                "std": std,
                "cv": std / mean if mean else float("nan"),
                "max": float(v.max()),
                "min": float(v.min()),
                "n": int(v.size),
            }
    return out


def plot_robustness_panel(
    disp: dict[int, dict[str, Any]],
    out_path: Path,
    baseline: str,
    treatment: str,
) -> bool:
    """Plot the robustness/dispersion panel: CV growth and worst-case tail.

    Left: coefficient of variation of delay vs demand (DQN explodes, rollout
    stays tight). Right: worst-case (max) episode delay vs demand, grouped bars,
    illustrating tail clipping.

    Args:
        disp: Output of :func:`compute_dispersion`.
        out_path: Destination PDF path.
        baseline: Baseline mode name.
        treatment: Treatment mode name.

    Returns:
        ``True`` if the figure was written, ``False`` if matplotlib is missing.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable ({exc}); skipping robustness panel.", file=sys.stderr)
        return False

    demands = sorted(disp)
    c_base, c_treat = "#a83a3a", "#245f9f"

    def series(mode: str, key: str) -> list[float]:
        return [disp[d][mode][key] for d in demands]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # --- Left: coefficient of variation vs demand ---
    cv_base, cv_treat = series(baseline, "cv"), series(treatment, "cv")
    ax1.plot(demands, cv_base, marker="o", lw=2, color=c_base, label=baseline)
    ax1.plot(demands, cv_treat, marker="s", lw=2, color=c_treat, label=treatment)
    for d, c in zip(demands, cv_base):
        ax1.annotate(f"{c:.3f}", (d, c), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color=c_base)
    for d, c in zip(demands, cv_treat):
        ax1.annotate(f"{c:.3f}", (d, c), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color=c_treat)
    ax1.set_xlabel("Demand (vehicles/episode)")
    ax1.set_ylabel("Coefficient of variation of delay  (σ/μ)")
    ax1.set_title("Delay dispersion: myopic policy destabilizes under load")
    ax1.set_xticks(demands)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="upper left")

    # --- Right: worst-case (max) episode delay, grouped bars ---
    max_base, max_treat = series(baseline, "max"), series(treatment, "max")
    x = np.arange(len(demands))
    width = 0.38
    ax2.bar(x - width / 2, max_base, width, color=c_base, label=baseline)
    ax2.bar(x + width / 2, max_treat, width, color=c_treat, label=treatment)
    for xi, val in zip(x - width / 2, max_base):
        ax2.annotate(f"{val:,.0f}", (xi, val), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, color=c_base)
    for xi, val in zip(x + width / 2, max_treat):
        ax2.annotate(f"{val:,.0f}", (xi, val), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, color=c_treat)
    ax2.set_xlabel("Demand (vehicles/episode)")
    ax2.set_ylabel("Worst-case episode delay  (veh·s)")
    ax2.set_title("Tail clipping: rollout truncates catastrophic blow-ups")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in demands])
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze the Phase 2.1 demand-sweep CSVs (trend test + Δ(D) plot).",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help=f"Root with D_<D>/<model_run>/paired_benchmarks*.csv (default: {DEFAULT_RESULTS_ROOT}).")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help=f"Metric column to compare; lower-is-better assumed (default: {DEFAULT_METRIC}).")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help=f"Baseline mode (default: {DEFAULT_BASELINE}).")
    parser.add_argument("--treatment", default=DEFAULT_TREATMENT,
                        help=f"Treatment mode (default: {DEFAULT_TREATMENT}).")
    parser.add_argument("--n-boot", type=int, default=_DEFAULT_BOOTSTRAP,
                        help=f"Bootstrap resamples for CIs (default: {_DEFAULT_BOOTSTRAP}).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir for summary JSON + figures (default: <results-root>).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full Phase 2.1 analysis and write artifacts.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    out_dir = args.out_dir or args.results_root

    rows, has_tp, has_res = load_rows(args.results_root, args.metric)
    pairs_by_demand = build_pairs(rows, args.baseline, args.treatment)
    if not pairs_by_demand:
        print(
            f"[error] no '{args.baseline}'/'{args.treatment}' pairs found for metric "
            f"'{args.metric}'. Check mode names and that the sweep completed.",
            file=sys.stderr,
        )
        return 2

    summary = summarize_by_demand(pairs_by_demand, args.n_boot)

    # Cross-level trend tests on per-pair advantage, groups ordered by demand.
    ordered_demands = sorted(pairs_by_demand)
    groups = [np.array([p["advantage"] for p in pairs_by_demand[d]], dtype=float) for d in ordered_demands]
    jt = jonckheere_terpstra(groups)
    flat_demands = np.array([d for d in ordered_demands for _ in pairs_by_demand[d]], dtype=float)
    flat_adv = np.concatenate(groups) if groups else np.array([])
    ols = ols_trend(flat_demands, flat_adv)

    # Anti-truncation-bias guard (only if columns exist).
    guard: dict[str, Any] = {}
    if has_tp or has_res:
        guard = throughput_guard(rows, args.baseline, args.treatment)

    # Dispersion / robustness (CV growth and worst-case tail).
    dispersion = compute_dispersion(rows, (args.baseline, args.treatment))

    # ---- console report -------------------------------------------------
    print(f"Phase 2.1 stress analysis  |  metric={args.metric}  treatment={args.treatment} vs {args.baseline}")
    print("=" * 78)
    for d in ordered_demands:
        s = summary[d]
        w = s["wilcoxon"]
        print(
            f"D={d:>5} | n={s['n_pairs']:>3} | win={s['win_rate_pct']:5.1f}% | "
            f"meanΔ%={s['mean_reduction_pct']:+6.3f} "
            f"CI[{s['mean_reduction_pct_ci95'][0]:+.3f},{s['mean_reduction_pct_ci95'][1]:+.3f}] | "
            f"median_adv={s['median_advantage']:+8.1f} | "
            f"Wilcoxon p={w['p_value']:.4f} (z={w['z']:+.2f})"
        )
        if "mean_normalized_reduction_pct" in s:
            print(
                f"        normalized meanΔ%={s['mean_normalized_reduction_pct']:+6.3f} "
                f"(raw {s['mean_reduction_pct']:+6.3f}; "
                f"|Δ|={abs(s['mean_normalized_reduction_pct'] - s['mean_reduction_pct']):.4f} pp) "
                f"-> normalization-invariant"
            )
    print("-" * 78)
    print(
        f"Jonckheere-Terpstra (increasing trend): JT={jt['statistic']:.1f}, "
        f"z={jt['z']:+.3f}, one-sided p={jt['p_value_increasing']:.4f}"
    )
    print(
        f"OLS interaction slope: {ols['slope']:+.5f} %adv/veh "
        f"CI95[{ols['slope_ci95'][0]:+.5f},{ols['slope_ci95'][1]:+.5f}]"
    )
    if guard:
        print("-" * 78)
        for d, g in guard.items():
            flag = {True: "YES (delay win may be demand-shedding!)", False: "no", None: "n/a"}[g["shedding_suspected"]]
            print(f"D={d:>5} throughput {args.treatment}={g['treatment_throughput']:.2f} "
                  f"vs {args.baseline}={g['baseline_throughput']:.2f} | "
                  f"deficit={g['throughput_deficit']:+.3f} (tol={g['shedding_tolerance']:.2f}) | "
                  f"shedding suspected: {flag}")
    else:
        print("-" * 78)
        print(f"[note] columns '{THROUGHPUT_COL}'/'{RESIDUAL_COL}' absent from the CSVs "
              "(frozen eval schema). Throughput/residual guard skipped; extend the "
              "harness to enable it.")
    print("=" * 78)

    # ---- artifacts ------------------------------------------------------
    plot_path = out_dir / "delta_of_demand.pdf"
    plotted = plot_delta_curve(summary, plot_path, args.baseline, args.treatment)
    robustness_path = out_dir / "robustness_dispersion.pdf"
    robustness_plotted = plot_robustness_panel(dispersion, robustness_path, args.baseline, args.treatment)

    result = {
        "metadata": {
            "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
            "results_root": args.results_root.as_posix(),
            "metric": args.metric,
            "baseline": args.baseline,
            "treatment": args.treatment,
            "advantage_definition": "baseline - treatment (positive => treatment better for lower-is-better metric)",
            "n_bootstrap": args.n_boot,
            "throughput_available": has_tp,
            "residual_available": has_res,
        },
        "per_demand": {str(d): summary[d] for d in ordered_demands},
        "trend_jonckheere_terpstra": jt,
        "trend_ols_interaction": ols,
        "throughput_guard": guard,
        "dispersion": {str(d): dispersion[d] for d in sorted(dispersion)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "stress_analysis_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote summary -> {summary_path.as_posix()}")
    if plotted:
        print(f"Wrote figure  -> {plot_path.as_posix()}")
    if robustness_plotted:
        print(f"Wrote figure  -> {robustness_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
