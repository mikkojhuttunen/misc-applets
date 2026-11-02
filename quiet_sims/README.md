# QUIET simulations

Ray-tracing and speckle engine for chaotic multipass cells (PlanarEngine, GasEngine, DitherLab; JavaScript, Node ≥ 18)
with membrane loss, extended for the QUIET proposal: free-standing SiNx membrane cells, TM0 only. Python (numpy,
scipy, pytest) for the slab-mode and trench-DBR calculators, the metrics and the noise budget. Every script runs from
the command line and writes JSON to `results/`; `python -m pytest -q tests` checks the baselines and the extensions.

| Path | Content |
|---|---|
| `engine/quiet_engine.js` | working engine (Fig. 2 copy) with the QUIET extensions; `engine/original/` the two engines as received |
| `scripts/ens.js`, `scripts/ens2.js` | Fig. 3 / Fig. 2 ensembles (same CLI as the originals in `scripts/original/`; `--engine quiet|fig2|lab`) |
| `scripts/cells.js` | shared cell settings (Fig. 2 settings, n_eff, loss, R or R(sin χ), launch m) |
| `scripts/angle_hist.js` | distribution of \|sin χ\| over the bounces of the detected light |
| `scripts/step*.py` | the steps of the QUIET plan (below) |
| `qs/` | Python helpers: metrics, mirror reflectivity, gas absorption, noise budget |
| `design_calcs/slab.py`, `design_calcs/tmm.py` | free-standing slab TM0/TE0 n_eff and Γ; trench DBR transfer matrix |
| `results/baseline/` | the ensembles as received; `results/stepN*` the outputs of each step |

Settings of the baseline: circle 4 mm, 24 facets, 16 inclusions (Δn 0.01, radius 50 µm), launch 7.5° from the wall
normal, eps 1e-6, ±2 mrad angle dither, ±20 pm wavelength dither, 1 mW, Γ 0.2, R 0.99, 0.1 dB/cm.

## Step 0: baselines

`node scripts/ens2.js 0.99 0.1 circle 30 results/step0/f2_circle_0.99.json`,
`node scripts/ens.js 0.99 0.1 21 results/step0/ens_tgt.json`, then `python scripts/step0_baseline.py`
(→ `results/step0_baseline.json`). The repo engine reproduces the stored layouts bit for bit:

| Quantity | Quoted | Stored ensembles | Rerun |
|---|---|---|---|
| static fringe, both dithers, chaotic median (30 layouts) | 4.2e-8 | 4.16e-8 | 4.16e-8 |
| same, regular cell | 1.7e-7 | 1.66e-7 | 1.66e-7 |
| Γ⟨L⟩ | 2.5 cm | 2.53 cm | 2.53 cm |
| NH3 limit at 1 s (21 layouts) | 110 ppb | 110 ppb | 110 ppb |
| Si 250 nm TM0 at 1.55 µm: n_eff, Γ | 2.0, 0.64 | — | 2.015, 0.639 |

Definitions (`qs/metrics.py`): static fringe V_det = 2√eps · V / √grains (V the static floor of the dither mode, grains
the M² speckle grains of the detected light); NH3 limit = NEA / (α Γ⟨L⟩) with NEA = √(q/(P T_cell)/τ + r(τ)²), r(τ)
the fringe residual after averaging τ (drift 0.1 rad/s), medians over the layouts, and α = 2.27e-7 /cm/ppm, the
breath-broadened NH3 peak of the Fig. 3 HITRAN strips.
