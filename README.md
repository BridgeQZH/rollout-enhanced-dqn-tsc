# Rollout-Enhanced DQN for Adaptive Traffic Signal Control

**Amortized learning + online search for sample-efficient, robust signal control.**

> A frozen Deep-Q-Network supplies an *amortized* value prior; a short analytic
> rollout supplies *test-time search*. Together they recover performance the
> network alone leaves on the table — most when the network is **under-trained**
> or pushed **off-distribution** under heavy demand.

---

## Abstract

Deep reinforcement learning for traffic signal control is held back by two
practical barriers: **sample efficiency** (training is expensive and unsafe
online) and **robustness** (real demand drifts away from the training
distribution). This project studies a single mechanism that attacks both —
**rollout**: a one/multi-step lookahead that grounds the first decisions in an
analytic transition model `f` and stage cost `g`, then bootstraps from a frozen
DQN's value `H` as a discounted tail.

The central trade-off is **search amortization ⇄ test-time compute**. A trained
value network amortizes planning into one forward pass; rollout *de-amortizes*,
spending inference-time computation to correct that network's residual
value-approximation error. The empirical story has two faces of the *same* error:

- **Sample efficiency.** With only **100 training episodes**, rollout recovers
  ≈85% of the under-training penalty — matching a fully **converged 500-episode**
  DQN at one-fifth the data (paired Wilcoxon *p* = 0.019). At convergence the
  advantage correctly washes out (*p* = 0.553), confirming rollout is a
  *compensation* mechanism, not a free asymptotic gain.
- **Robustness under distribution shift.** Run on the converged model and stressed
  from D = 1000 → 2000 vehicles/episode, the rollout advantage **re-opens
  monotonically**, reaching a **≈15% waiting-time reduction at D = 2000**
  (*p* ≈ 2×10⁻¹⁰; Jonckheere–Terpstra trend *z* = +9.30).

The result is mechanistically explained by **variance / tail truncation**: rollout
clips the catastrophic worst-case episodes that a purely myopic value function
suffers off-distribution.

---

## Live Demonstration

D = 2000 saturation · identical traffic seed · **Frozen DQN vs Rollout-DQN**, with
a live MDP telemetry overlay (total queue length + moving average).

<video src="media/final_comparison.mp4" controls width="100%" muted playsinline></video>

▶️ If the inline player does not load, open **[media/final_comparison.mp4](media/final_comparison.mp4)** directly.

---

## Key Empirical Findings

| Configuration | n pairs | Win rate | Mean Δ waiting-time | Wilcoxon *p* | Verdict |
|---|---:|---:|---:|---:|---|
| **100-ep** · D=1000 | 90 | 64.4% | **+1.12%** | 0.019 | rollout compensates under-training |
| **500-ep** · D=1000 | 60 | 55.0% | +0.12% | 0.553 | washout (converged prior is enough) |
| **500-ep** · D=1500 | 60 | 76.7% | **+4.32%** | 3.6×10⁻⁶ | gap re-opens under load |
| **500-ep** · D=2000 | 60 | 91.7% | **+15.02%** | 1.7×10⁻¹⁰ | massive divergence |

- **Trend:** Jonckheere–Terpstra *z* = **+9.30** (*p* < 10⁻¹⁶); OLS interaction slope
  +7.26 veh·s per vehicle of demand (95% CI [5.15, 9.38]).
- **Tail truncation @ D=2000:** worst-case episode delay **99,586 → 43,389 veh·s**;
  coefficient of variation **0.074 → 0.259 (DQN)** vs **0.068 → 0.099 (rollout)**.
- **No demand-shedding confound:** both controllers clear **≈99.93%** of vehicles;
  the gap is **normalization-invariant** (raw vs per-vehicle differ < 0.01 pp).
- **Cost of search:** rollout ≈ **9.5×** the per-decision latency of bare DQN
  (4.07 ms vs 0.43 ms, CPU) — the explicit price of test-time compute.

All numbers are reproduced from source by `python validate_chapter8.py` (41/41 checks)
and traced in [`deck/EVIDENCE.md`](deck/EVIDENCE.md).

---

## Method (one screen)

For each candidate phase `u`, score a one-step lookahead and pick the best:

```
u*  =  argmax_u  [  g(x, u)            # analytic stage cost over the window
                  + β · γ^Δ · H(f(x,u)) ]   # discounted frozen-DQN tail value
```

- **State** `x ∈ ℝ¹²` — per-lane-group queue counts (a deliberately minimal,
  Markov-sufficient operational statistic; cf. max-pressure control).
- **`g`** — linear unserved-queue delay = the objective itself (∫ queue dt), not a
  shaped surrogate. Reward ≡ `g`, so `H` and `g` are commensurate and **β = 1** is
  principled, not tuned.
- **`f`** — store-and-forward "physics shield": it need only *rank* candidate
  actions, and stays valid off-distribution where the learned `H` degrades.
- Each green+yellow phase is a temporally-extended (**SMDP**) action; the tail uses
  `γ^Δ` discounting.

---

## Repository Layout

```
clean_rollout_tlcs/      research engine: Settings, transition f, stage cost g, DQN agent, eval harness
  settings/              training/eval YAML configs
  eval_results*/         paired benchmark CSVs + analysis (100ep & 500ep)
models/
  dqn_100ep/             frozen checkpoints — 100 episodes, seeds 0,1,2  (under-trained baseline)
  dqn_500ep/             frozen checkpoints — 500 episodes, seeds 0,1    (converged baseline)
eval_results_stress/     demand-sweep results D∈{1000,1500,2000} + stress_analysis_summary.json
academic_artifacts/      consolidated statistical summaries
intersection/            SUMO network, config, GUI view-settings (route file is regenerated)
scripts/                 deck pipeline (extract_evidence, generate_figures) + run_gui_demo (live telemetry)
stress_sweep.py          demand-sweep orchestrator (prints exact eval commands)
analyze_stress_results.py   trend tests (Wilcoxon, Jonckheere–Terpstra, bootstrap) + figures
validate_chapter8.py     independent recompute-from-source validation harness
deck/                    self-contained PhD-interview HTML slide deck (open deck/index.html)
docs/                    narrative HTML reports + the reference paper PDF
media/                   curated demo video (<50MB)
```

---

## Installation & Setup

**Prerequisites:** Python ≥ 3.11 and **SUMO** (system install). Set `SUMO_HOME`
and put `%SUMO_HOME%\bin` on your PATH (`sumo-gui --version` should work). See
[`SUMO_GUI_RECORDING_GUIDE.md`](SUMO_GUI_RECORDING_GUIDE.md) §1.

**Activate the environment from the project root** (no more `cd` juggling):

```powershell
# Windows PowerShell
. .\activate.ps1
```
```bash
# bash / WSL / Git Bash
source activate.sh
```

These wrappers prefer a root `.venv\`, falling back to the existing environment.
For a **fresh clone**, create the env at the root once:

```bash
python -m venv .venv
. .\activate.ps1                 # or: source activate.sh
pip install -r requirements.txt
pip install -e .                 # makes `clean_rollout_tlcs` importable anywhere
```

---

## Usage

Run everything from the repo root with the environment activated.

**Train** the DQN value prior (used standalone and as the rollout tail `H`):
```bash
python -m clean_rollout_tlcs.train --settings clean_rollout_tlcs/settings/training_settings.yaml --seeds 0,1,2 --out models/dqn_100ep
```

**Evaluate** all five controllers under identical paired traffic:
```bash
python -m clean_rollout_tlcs.eval --model-dir models/dqn_500ep/seed_0 --out clean_rollout_tlcs/eval_results_long/paired_benchmarks_seed0.csv
```

**High-congestion stress sweep** (prints exact per-demand eval commands, then analyze):
```bash
python stress_sweep.py
# ...run the printed `python -m clean_rollout_tlcs.eval ...` commands...
python analyze_stress_results.py --results-root eval_results_stress
python validate_chapter8.py          # independent recompute-from-source check
```

**Live terminal telemetry GUI demo** (SUMO-GUI + real-time queue plot):
```bash
python scripts/run_gui_demo.py --telemetry-selftest --mode rollout_ms        # backend smoke-test, no SUMO
python scripts/run_gui_demo.py --mode dqn        --demand 2000 --seed 100000 --delay 120
python scripts/run_gui_demo.py --mode rollout_ms --demand 2000 --seed 100000 --delay 120
```
Same `--seed` ⇒ identical traffic ⇒ a fair paired comparison. Full recording
workflow in [`SUMO_GUI_RECORDING_GUIDE.md`](SUMO_GUI_RECORDING_GUIDE.md).

**Interview slide deck:** open [`deck/index.html`](deck/index.html) in a browser
(← / → to navigate, `F` fullscreen, Ctrl+P to export PDF). Rebuild its figures
with `python scripts/extract_evidence.py && python scripts/generate_figures.py`
(see [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md)).

---

## Reproducibility

- **Paired design:** one route file per seed, replayed across all five
  controllers (route-file hash asserted unchanged) → valid paired statistics.
- **Leak-free seeds:** training route seeds `s·10⁴+e` are provably disjoint from
  evaluation seeds `100000–100029`.
- **No SciPy:** Wilcoxon (tie-corrected), Jonckheere–Terpstra, and bootstrap CIs
  are self-contained and cross-checked by `validate_chapter8.py`.
- **Pinned environment:** see `requirements.txt`; results were produced with
  torch 2.9.0 (CPU), numpy 2.3.4, SUMO/traci 1.24.0.

---

## Roadmap → MARL

This single-intersection study isolates the rollout mechanism cleanly. Next:
learned-residual `f` (to lift the transition rank-correlation past its ≈0.54
ceiling), uncertainty-aware rollout budgeting, and **multi-intersection
coordination** (MARL) under the test-time-compute constraint.

## Lineage & Acknowledgements

The engineering chassis derives from AndreaVidali's *Deep Q-Learning Agent for
Traffic Signal Control*; the reference paper is included at
[`docs/reference_paper_Vidali_DRL_TLCS.pdf`](docs/reference_paper_Vidali_DRL_TLCS.pdf).
The 12-D state, analytic `f`/`g`/`H` rollout engine, and paired benchmark
protocol are this project's contribution.

## License

MIT (see `pyproject.toml`). Research/educational use.
