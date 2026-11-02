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

## Step 1: angle-dependent mirror reflectivity

The engine takes an optional `R_table: {s, R}` (R against |sin χ|, χ from the wall normal); every bounce before the exit
multiplies the ray's power by R(|sin χ|) (ensemble, fringe spectrum, speckle beamlets); without a table it uses the
constant `R_mirror` with unchanged arithmetic (a flat table reproduces it to 1e-15, `scripts/check_rtable.js`).

`qs/mirror.py`: effective-index trench DBR (vectorised `design_calcs/tmm.py`, s-polarisation for the TM0 slab mode,
gaps of index 1, teeth of index n_eff, air behind), total internal reflection above the critical angle included
(evanescent gaps; frustrated TIR if the membrane continues behind), optional sidewall roughness R·exp(−(4π n_eff σ
cos χ/λ)²) and an optional vertical-diffraction estimate. `node scripts/angle_hist.js --cell circle|herriott` gives the
|sin χ| distribution of the bounces of the detected light (8 layouts); `python scripts/step1_mirrors.py` optimises the
design (N ≤ 20 periods, odd quarter-wave gap and tooth orders 1/3/5, design angle, linear chirp ±0.2, two-band) for
the largest bounce-averaged R over that distribution and writes `results/step1_mirrors.json` and the R tables
`results/step1/Rtable_<cell>_neff<n>.json` (and `_rough5nm` versions). `python scripts/report.py step1` prints:

circle: bounce-averaged R vs N (optimised design per N)

| n_eff (film) | above crit. | N=1 | N=2 | N=4 | N=6 | N=8 | N=12 | N=20 | best design (N=20) | σ=5 nm | σ=10 nm | diffraction bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.36 (400 nm @ 1.55 µm) | 12.6 % | 0.3503 | 0.5333 | 0.7641 | 0.8652 | 0.9109 | 0.9514 | 0.9932 | m 1/1, gap 505 nm, tooth 321 nm, 2-band, chirp +0.2 | 0.9907 | 0.9835 | 0.50 |
| 1.53 (500 nm @ 1.55 µm) | 16.3 % | 0.5085 | 0.7380 | 0.9158 | 0.9701 | 0.9854 | 0.9954 | 0.9998 | m 1/1, gap 466 nm, tooth 271 nm, 2-band, chirp +0.2 | 0.9967 | 0.9874 | 0.61 |
| 1.66 (600 nm @ 1.55 µm) | 20.2 % | 0.6072 | 0.8317 | 0.9737 | 0.9940 | 0.9985 | 0.9998 | 1.0000 | m 1/1, gap 428 nm, tooth 241 nm, 2-band, chirp +0.2 | 0.9963 | 0.9855 | 0.65 |
| 1.74 (700 nm @ 1.55 µm) | 22.8 % | 0.6571 | 0.8701 | 0.9853 | 0.9983 | 0.9997 | 1.0000 | 1.0000 | m 1/1, gap 456 nm, tooth 234 nm | 0.9960 | 0.9841 | 0.69 |
| 1.46 (1000 nm @ 3.30 µm) | 14.3 % | 0.4456 | 0.6644 | 0.8675 | 0.9374 | 0.9630 | 0.9884 | 0.9987 | m 1/1, gap 1075 nm, tooth 624 nm, 2-band, chirp +0.2 | 0.9981 | 0.9962 | 0.58 |
| 1.59 (1200 nm @ 3.30 µm) | 18.1 % | 0.5571 | 0.7873 | 0.9509 | 0.9842 | 0.9953 | 0.9991 | 0.9999 | m 1/1, gap 902 nm, tooth 535 nm, 2-band, chirp +0.2 | 0.9992 | 0.9969 | 0.64 |
| 1.71 (1500 nm @ 3.30 µm) | 21.8 % | 0.6393 | 0.8570 | 0.9820 | 0.9972 | 0.9995 | 1.0000 | 1.0000 | m 1/1, gap 971 nm, tooth 507 nm | 0.9991 | 0.9966 | 0.68 |

herriott: bounce-averaged R vs N (optimised design per N)

| n_eff (film) | above crit. | N=1 | N=2 | N=4 | N=6 | N=8 | N=12 | N=20 | best design (N=20) | σ=5 nm | σ=10 nm | diffraction bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.36 (400 nm @ 1.55 µm) | 33.2 % | 0.4643 | 0.6174 | 0.8502 | 0.9453 | 0.9763 | 0.9906 | 0.9945 | m 1/1, gap 505 nm, tooth 321 nm, 2-band, chirp +0.2 | 0.9925 | 0.9864 | 0.66 |
| 1.53 (500 nm @ 1.55 µm) | 33.7 % | 0.5587 | 0.7575 | 0.9454 | 0.9871 | 0.9958 | 0.9992 | 0.9998 | m 1/1, gap 428 nm, tooth 263 nm, 2-band, chirp +0.2 | 0.9972 | 0.9895 | 0.73 |
| 1.66 (600 nm @ 1.55 µm) | 33.9 % | 0.6238 | 0.8299 | 0.9740 | 0.9959 | 0.9991 | 0.9999 | 1.0000 | m 1/1, gap 423 nm, tooth 240 nm, 2-band, chirp +0.1 | 0.9969 | 0.9878 | 0.74 |
| 1.74 (700 nm @ 1.55 µm) | 34.0 % | 0.6599 | 0.8628 | 0.9837 | 0.9980 | 0.9997 | 1.0000 | 1.0000 | m 1/1, gap 427 nm, tooth 230 nm, chirp +0.2 | 0.9966 | 0.9866 | 0.76 |
| 1.46 (1000 nm @ 3.30 µm) | 33.5 % | 0.5207 | 0.7062 | 0.9177 | 0.9764 | 0.9910 | 0.9971 | 0.9991 | m 1/1, gap 973 nm, tooth 604 nm, 2-band, chirp +0.2 | 0.9986 | 0.9970 | 0.72 |
| 1.59 (1200 nm @ 3.30 µm) | 33.8 % | 0.5897 | 0.7941 | 0.9612 | 0.9925 | 0.9978 | 0.9997 | 1.0000 | m 1/1, gap 923 nm, tooth 540 nm, 2-band, chirp +0.2 | 0.9993 | 0.9975 | 0.74 |
| 1.71 (1500 nm @ 3.30 µm) | 34.0 % | 0.6468 | 0.8514 | 0.9807 | 0.9973 | 0.9995 | 1.0000 | 1.0000 | m 1/1, gap 971 nm, tooth 507 nm | 0.9993 | 0.9971 | 0.76 |

Findings: the first-order design (gap and tooth both λ/4 at the design angle) wins everywhere, usually with two bands
and the largest chirp tried; R grows fast with N for n_eff ≥ 1.53 (≥ 0.999 from N ≈ 8–12) while n_eff 1.36 stays at
0.993 (circle) / 0.995 (Herriott) at N = 20, its low index contrast cannot cover all sub-critical angles. 13–23 % of
the circle's bounces (34 % in the Herriott, its side walls) are above the critical angle and reflect totally.
Sidewall roughness of 5 nm rms caps the bounce average at 0.996–0.997 at 1.55 µm (0.998–0.999 at 3.3 µm), 10 nm at
≈ 0.985.

Vertical diffraction (limitation, estimated): the membrane TM0 modes are only ≈ λ/5 wide (rms of the power flux), so
a free-standing λ/4 air gap couples just 77–81 % of the power back into the next tooth (exact scalar angular-spectrum
propagation, `qs/mirror.gap_coupling`). Charging every gap traversal independently would cut the bounce average to
0.5–0.76 (column "diffraction bound"); that bound is far too pessimistic for a first-order mirror, whose period
(≈ 0.4 λ < λ/2) puts the stop-band Bloch mode below the light line so it does not radiate inside the mirror; the real
loss sits at the slab-to-mirror transition and needs a vertical 2D (FDTD / mode-matching) calculation, ideally with an
adiabatic taper of the first periods. The ensembles below use the in-plane R tables (no diffraction term).

## Step 2: ensembles with R(χ) and 0.1 dB/cm

`bash scripts/run_step2.sh` (→ `results/step2/`, `node scripts/ensemble.js`, 30 layouts + the regular cell per case,
R tables of step 1, SiNx n_eff at 1.55 µm, 0.1 dB/cm, circle CMPC and Herriott cell with the Fig. 2 settings) and
`python scripts/step2_ensembles.py` (→ `results/step2_summary.json`); `python scripts/report.py step2`:


| cell | film (n_eff, Γ) | R(χ) | static fringe: none / spectral / angle / both (chaotic median) | both, regular | layouts < 5e-7 | Γ⟨L⟩ | T | ⟨R⟩ bounce | NH3 1 s |
|---|---|---|---|---|---|---|---|---|---|
| circle (baseline) | Si 250 nm (2.0, 0.20) | R 0.99 | 1.2e-04 / 4.5e-06 / 4.5e-07 / 4.2e-08 | 1.7e-07 | 97 % | 2.53 cm | 0.23 | 0.99 | 110 ppb |
| circle | SiNx 400 nm (1.36, 0.67) | R(χ) | 1.4e-04 / 5.3e-06 / 2.5e-07 / 5.5e-08 | 4.3e-08 | 100 % | 8.34 cm | 0.28 | 0.9931 | 41 ppb |
| circle | SiNx 500 nm (1.53, 0.46) | R(χ) | 1.3e-04 / 5.4e-06 / 2.9e-07 / 4.6e-08 | 7.7e-08 | 93 % | 6.63 cm | 0.35 | 0.9998 | 36 ppb |
| circle | SiNx 600 nm (1.66, 0.31) | R(χ) | 1.3e-04 / 5.4e-06 / 2.6e-07 / 3.4e-08 | 4.0e-08 | 100 % | 4.67 cm | 0.34 | 1.0000 | 34 ppb |
| circle | SiNx 600 nm (1.66, 0.31) | R(χ), σ 5 nm | 1.3e-04 / 5.4e-06 / 2.8e-07 / 3.7e-08 | 4.4e-08 | 97 % | 4.36 cm | 0.30 | 0.9963 | 40 ppb |
| circle | SiNx 700 nm (1.74, 0.21) | R(χ) | 1.2e-04 / 3.4e-06 / 2.5e-07 / 5.5e-08 | 1.7e-07 | 100 % | 3.13 cm | 0.34 | 1.0000 | 71 ppb |
| herriott | SiNx 400 nm (1.36, 0.67) | R(χ) | 2.2e-04 / 1.0e-05 / 3.1e-06 / 3.8e-07 | 6.8e-05 | 67 % | 10.23 cm | 0.56 | 0.9946 | 179 ppb |
| herriott | SiNx 500 nm (1.53, 0.46) | R(χ) | 2.2e-04 / 7.9e-06 / 3.2e-06 / 3.8e-07 | 8.3e-05 | 60 % | 7.79 cm | 0.63 | 0.9999 | 180 ppb |
| herriott | SiNx 600 nm (1.66, 0.31) | R(χ) | 2.2e-04 / 6.0e-06 / 3.0e-06 / 2.1e-07 | 7.7e-05 | 67 % | 5.64 cm | 0.61 | 1.0000 | 148 ppb |
| herriott | SiNx 600 nm (1.66, 0.31) | R(χ), σ 5 nm | 2.2e-04 / 6.5e-06 / 3.3e-06 / 2.3e-07 | 7.7e-05 | 70 % | 5.34 cm | 0.56 | 0.9969 | 182 ppb |
| herriott | SiNx 700 nm (1.74, 0.21) | R(χ) | 2.1e-04 / 5.2e-06 / 3.2e-06 / 2.2e-07 | 1.9e-04 | 70 % | 4.10 cm | 0.59 | 1.0000 | 179 ppb |

Static fringe: V_det = 2√eps·V/√grains for each dither mode (V = 1 without dither); medians over 30 layouts. Γ⟨L⟩ with
the Γ of each film (slab solver: 0.67, 0.46, 0.31, 0.22). NH3 limit at 1 s from the medians of the first 21 layouts
as in the baseline; the noise-equivalent absorbance does not depend on Γ, so ppb scales as 1/(Γ⟨L⟩).

Findings: with the step-1 mirrors the circular SiNx cells beat the Si baseline by about 3× (34–41 ppb against 110):
higher Γ, near-unity mirrors (T 0.28–0.35 against 0.23) and a static fringe that stays at 3–6e-8 with both dithers,
all or nearly all layouts below 5e-7. 600 nm (n_eff 1.66) is the best compromise; 400 nm has the largest Γ⟨L⟩ but its
mirror (⟨R⟩ 0.993) costs transmission; 700 nm loses on Γ. 5 nm sidewall roughness costs ≈ 15 % (34 → 40 ppb). The
Herriott cell keeps more light (T ≈ 0.6) and a long path but its fringe is 5–10× worse (2–4e-7 with both dithers,
regular 1e-4) and only 60–70 % of the layouts reach 5e-7: 150–180 ppb.

## Step 3: launch-angle study

`bash scripts/run_step3.sh` (per launch and n_eff: bounce-angle histogram → N = 20 mirror optimised for it with
`scripts/make_rtable.py` → 21 layouts + regular) and `python scripts/step3_launch.py` (→ `results/step3_summary.json`,
with bootstrap 68 % intervals of the NH3 limit over the layouts); `python scripts/report.py step3`:

| n_eff | m | χ | passes to exit (regular) | bounces above critical | ⟨R⟩ bounce (N=20 mirror) | Γ⟨L⟩ | T | static fringe both (chaotic / regular) | NH3 1 s [68 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1.66 | 11 | 7.5° | 23 | 19 % | 1.0000 | 4.70 cm | 0.35 | 4.6e-08 / 1.2e-07 | 48 ppb [38–69] |
| 1.66 | 8 | 30.0° | 2 | 24 % | 1.0000 | 0.47 cm | 0.90 | 2.4e-05 / 3.2e-05 | 221862 ppb [193765–343485] |
| 1.66 | 5 | 52.5° | 23 | 65 % | 1.0000 | 3.21 cm | 0.39 | 4.4e-08 / 8.1e-08 | 66 ppb [51–85] |
| 1.66 | 2 | 75.0° | 11 | 89 % | 1.0000 | 1.82 cm | 0.60 | 4.7e-07 / 1.0e-06 | 1127 ppb [988–1210] |
| 1.66 | 1 | 82.5° | 23 | 93 % | 1.0000 | 1.80 cm | 0.54 | 8.9e-08 / 6.6e-07 | 221 ppb [211–286] |
| 1.36 | 11 | 7.5° | 23 | 12 % | 0.9931 | 8.33 cm | 0.29 | 5.8e-08 / 5.3e-08 | 33 ppb [31–53] |
| 1.36 | 8 | 30.0° | 2 | 7 % | 0.9919 | 0.58 cm | 0.83 | 4.1e-05 / 1.6e-04 | 310785 ppb [272457–384756] |
| 1.36 | 5 | 52.5° | 23 | 55 % | 0.9974 | 6.35 cm | 0.33 | 2.0e-07 / 1.6e-07 | 138 ppb [90–163] |
| 1.36 | 2 | 75.0° | 11 | 92 % | 0.9988 | 2.56 cm | 0.53 | 5.3e-07 / 6.6e-07 | 911 ppb [820–943] |
| 1.36 | 1 | 82.5° | 23 | 91 % | 0.9994 | 3.82 cm | 0.50 | 1.7e-07 / 8.4e-07 | 199 ppb [182–272] |

Star orbit with step 24 − m facets: the exit facet m is reached after k passes with (24 − m)·k ≡ m (mod 24). Every
odd m (11, 5, 1) needs all 23 passes, but m = 8 closes a triangle and exits after 2 chords (Γ⟨L⟩ 0.5 cm, the fringe is
not scrambled) and m = 2 a 12-gon that exits after 11. For m = 2 and 1 the rays live in an annulus between the caustic
a·sin χ and the wall thinner than a 50 µm inclusion, so the inclusions are placed anywhere in the cell (inc_ring 0)
and the rays rarely meet them.

Decision for SiNx: keep the 7.5° launch (m = 11). It gives the longest Γ⟨L⟩ and the lowest fringe (NH3 33–48 ppb).
Grazing launches buy total internal reflection (89–93 % of the bounces at 75–82.5°, R ≈ 1 without a mirror) but
shorten the path to 2–4 cm Γ⟨L⟩ and leave the inclusions behind (fringe 1–5e-7): 200 ppb and worse. 52.5° (m = 5) is
the fall-back if the trench mirrors underperform: 55–65 % of its bounces reflect totally, so a sub-critical mirror
reflectivity R_s enters as ⟨R⟩ ≈ 0.35 R_s + 0.65 (m = 5, n_eff 1.66) against 0.81 R_s + 0.19 (m = 11): with R_s =
0.98, 0.993 against 0.984 per bounce. It costs ≈ 1.4× in NH3 (66 ppb) with ideal mirrors.
