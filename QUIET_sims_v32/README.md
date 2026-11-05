# QUIET v32 simulations (realistic loss-budget parameters)

Report: [REPORT.md](REPORT.md). One command: `python run_all.py` (all ensembles, ≈ 2 h on 4 cores) or
`python run_all.py --analysis` (analyses, figures, tables, tests from the stored `results/`, ≈ 40 s); `make`, `make analysis`,
`make test` do the same. Node ≥ 18, Python 3 with numpy, matplotlib, pytest; HAPI 1.3 for `scripts/t7_lines.py` (offline,
line files in `../HITRAN/hitran_data`).

| path | content |
|---|---|
| `engine/quiet_engine.js` | ray / speckle engine (copy of `quiet_sims/engine/quiet_engine.js`, the Fig. 2 artifact engine with membrane loss) |
| `scripts/ensemble.js`, `cells.js` | ensemble runner (30 inclusion layouts + regular cell; options for R, loss, n_eff, Γ, wavelength, dithers, inclusions; `--stat --paths --curves`) |
| `scripts/chaos_metrics.js`, `poincare_dump.js` | launch-beam Poincaré coverage; Fig. 2e sections |
| `scripts/run_t1.sh`, `run_t2.sh`, `run_t5_t6_t8.sh` | the simulation runs of tasks 1, 2 (+ mid-IR), 5, 6, 8 |
| `scripts/t*_*.py`, `fig2ef.py`, `report_tables.py` | analyses, figures and the REPORT tables |
| `qv/` | analysis formulas (`analysis.py`), TM0 slab solver, HAPI wrapper, figure style |
| `configs/setA.json`, `setB.json` | the two parameter sets |
| `data/baseline/` | the stored ensembles of the published figures (regression reference) and the Fig. 3 time grid / HITRAN strips |
| `results/` | raw results (JSON / CSV) per task; `figs/` vector PDFs (+ PNG previews) |
| `scripts/build_fig23_v32.py`, `artifacts/` | Fig. 2 / Fig. 3 claude.ai artifacts: live sources as last edited (`fig*_src_live.html`) patched to `fig*_v32.html` with a data-set selector (set B default; step 10 and published set A for comparison) and, in Fig. 3, the T1.3 detector noise model |
| `tests/test_t1_regression.py` | task-1 regression, including a bit-for-bit rerun against the published ensembles |

Figures for the proposal: `figs/fig2ef_setB_R0.95.pdf` (Fig. 2 e, f at R = 95 %; R0.97 / R0.995 variants),
`figs/fig3_setB.pdf` (Fig. 3 a, b); supplementary: `fig2f_setB_0.3dB.pdf`, `fig2f_setB_0.5dB.pdf`, `eps_sweep.pdf`,
`angle_dither.pdf`, `layout_spread.pdf`.
