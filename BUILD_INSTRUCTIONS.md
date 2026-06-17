# Build Instructions

How to regenerate the evidence, figures, and deck, and how to run the SUMO demo.
Run all commands from the repository root (`D:\rl_cc`).


## 1. Regenerate the evidence table
Parses all result JSON/CSV and writes the auditable evidence files. **No figure or
deck number is hand-typed** — they flow from here.
```bash
python scripts/extract_evidence.py
#  -> scripts/evidence.json        (machine-readable, consumed by figures)
#  -> deck/EVIDENCE.md             (human evidence table: metric/source/value/slide)
```

## 2. Regenerate the figures
Reads `scripts/evidence.json` + the raw CSVs + `eval_results_stress/stress_analysis_summary.json`
and renders the five dark-theme SVG charts.
```bash
python scripts/generate_figures.py
#  -> deck/assets/figures/fig_sample_efficiency.svg
#  -> deck/assets/figures/fig_demand_stress.svg
#  -> deck/assets/figures/fig_paired_seed.svg
#  -> deck/assets/figures/fig_tail_variance.svg
#  -> deck/assets/figures/fig_controller_baseline.svg
```

## 3. Open the deck
Open `deck/index.html` in a browser. Print to PDF with Ctrl+P (all slides render,
one per page).

## 4. Update numbers if experiment results change
The pipeline is data-driven, so:
1. Re-run the experiments (training / `clean_rollout_tlcs.eval` / the stress sweep
   via `stress_sweep.py`), refreshing the CSVs and `stress_analysis_summary.json`.
   Re-validate with `python validate_chapter8.py` (recompute-from-source check).
2. `python scripts/extract_evidence.py` then `python scripts/generate_figures.py`.
3. Refresh `deck/index.html`.
4. **Headline numbers embedded as text** in `deck/index.html` (e.g. the stat cards
   and the appendix manifest table) are not auto-injected — cross-check them against
   the freshly printed `deck/EVIDENCE.md` and edit if the data moved. The figures
   update automatically; the inline text is the only manual touch-point.

## 5. Run the SUMO GUI demo (D=2000, DQN vs Rollout)
Direct Python, no wrappers. Same seed → identical traffic → fair paired comparison.
A live telemetry window (total queue length + moving average) opens alongside SUMO-GUI.
```bash
python scripts/run_gui_demo.py --mode dqn        --demand 2000 --seed 100000 --delay 120
python scripts/run_gui_demo.py --mode rollout_ms --demand 2000 --seed 100000 --delay 120
```
Composite the two windows in OBS, export to `deck/videos/final_comparison.mp4`, and the
deck's "Live demonstration" slide picks it up automatically. Full recording workflow,
telemetry flags, queue colouring, OBS/ffmpeg, and troubleshooting:
see `SUMO_GUI_RECORDING_GUIDE.md` and `recording_commands.md`.

## File map (new artifacts; nothing existing was overwritten)
```
deck/
  index.html                     # the slide deck (self-contained)
  README.md                      # deck overview + navigation
  EVIDENCE.md                    # generated evidence table
  assets/styles/deck.css         # theme
  assets/figures/*.svg           # generated data charts
  videos/                        # drop final_comparison.mp4 here (demo slide)
scripts/
  extract_evidence.py            # data -> evidence.json + EVIDENCE.md
  generate_figures.py            # data -> SVG charts
  evidence.json                  # generated, machine-readable numbers
  run_gui_demo.py                # single-controller SUMO-GUI runner + live telemetry
SUMO_GUI_RECORDING_GUIDE.md      # GUI + recording guide (direct-Python, no wrappers)
recording_commands.md            # ffmpeg recipes
intersection/gui_settings.xml    # SUMO view settings (delay/viewport)
BUILD_INSTRUCTIONS.md            # this file
```
