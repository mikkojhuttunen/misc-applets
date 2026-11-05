# QUIET v32: CMPC speckle and detection-limit simulations at realistic parameters

Proposal: `RCF2027_CMPC_gas_sensors_multibeam_v32_review_round1.tex` (Sec. 3.2 "Preliminary results", Figs. 2f, 3, T1.1–T1.3,
M1–M8). Code, configs, raw results and figures: this folder (`QUIET_sims_v32/`). One command reproduces everything:
`python run_all.py` (ensembles ≈ 2 h on 4 cores) or `python run_all.py --analysis` (analyses, figures, tests from the stored
results). Every number below is a median over the inclusion layouts with the interquartile range [Q1–Q3] unless it is a
regular-cell value (single). Nothing was tuned after seeing results; failing cases are kept.

## 0. Summary (PASS / FAIL against the proposal targets)

| target | set A (old) | set B (realistic) | verdict |
|---|---|---|---|
| T1 regression of the published numbers | reproduced (Herriott gain 55×, not 80×) | – | PASS |
| M3: residual fringe < 5e-7, both dithers, near-IR C-CMPC | 3.2e-8 | R 95 %: 1.8e-7 [5.3e-8–4.0e-7]; 80 % of layouts | PASS (×2.8 margin, was ×15) |
| M3 at the frequency excursion of the 3.3 µm dither (±0.55 GHz) | – | near-IR 7.5e-7; mid-IR (1000 nm, 3.3 µm) 1.1–1.6e-6 | FAIL |
| M3 at 3.3 µm with a ±2.2 GHz (±81 pm) dither (sensitivity, same runs) | – | 2.4e-7 (R 97 %), 3.3e-7 (R 95 %); 60–76 % of layouts; chaotic gain 1.7–3.5× | PASS (needs the wider dither) |
| M5: chaotic gain ≥ 2–5× (both dithers), R = 95 % | 3.1× | 2.06× (0.3 dB/cm), 1.93× (0.5 dB/cm) | marginal: at/below the lower edge |
| ε thresholds (5e-7 up to ε ≈ 2e-4; 8e-8 up to 6e-6) | 2.3e-4 / 5.9e-6 | near-IR R 95 %: 7.6e-6 / 1.9e-7; mid-IR: < 1e-6 | FAIL (25–30× tighter) |
| angle dither gives K ≈ many independent patterns (K ≲ M ≈ 300) | – | K_eff ≈ 1.7–2.3 in ±2 mrad (see Task 5) | FAIL (risk R1.4 materialises) |
| angle dither suppresses the fringe (engine V_θ ≈ 5e-3) | – | wave-optics bound V_θ ≥ 0.98 at ±2 mrad, 40 µm waist (Task 5b): both-dither fringe 3.4e-6 at R 97 % | FAIL (engine optimistic) |
| NH3 ~50 ppb at 1 s (Sec. 3.2) | 56 ppb | 14 dB / 1 mW, 2 cm: 590–690 ppb; simulated Γ⟨L⟩: 270–480 ppb | FAIL |
| M7: CH4 / C2H6 10 ppb at ≤ 15 s (2 cm, 14 dB, 1 mW) | – | R 95 %: 13 / 15 ppb; R 97 %: 9.8 / 11 ppb; R 99.5 %: 6.8 / 7.7 ppb | FAIL at 95 %, marginal at 97 %, PASS at 99.5 % |
| ethane stretch 2 ppb | – | not reached in 100 s for any R at 2 cm (fringe floor 11–21 ppb) | FAIL |
| T1.3 detector arithmetic (1 s at 14 dB, 11 s at 20 dB, 3 mW / 15 s at 25 dB) | – | 0.7 s, 11.0 s, 12.2 s | PASS (detector only) |
| 3.3 µm lines 150–230× stronger than NH3 1.53 µm | – | 147× (C2H6 Q), 167× (CH4) | partly (230× is CH4 3.31 vs CH4 1.64 µm) |
| inclusions: 1–7 nm step → Δn_eff 0.001–0.01, R < 2e-5 per interface | – | 0.0015–0.0105, ≤ 1.2e-5 (500 nm, 1.53 µm) | PASS (near-IR); 1000 nm film needs ≈ 14 nm for 0.01 |

The near-IR circular CMPC still meets the M3 fringe target with ±20 pm (±2.6 GHz) and ±2 mrad dithers at R = 95 %, but
with a third of the old margin, and the chaotic advantage shrinks to 2× with both dithers (it stays 3.6–4.5× with the angle
dither alone). The mid-infrared cell, where the same ±20 pm is only ±0.55 GHz, does not meet M3 with the present dithers,
and its fringe, not the detector, sets the 3.3 µm detection limits. The NH3 limit is dominated by the uncooled detector
behind the 14–25 dB cell loss (old model: shot noise only).

## 1. Model, formulas and assumptions

Ray engine: `engine/quiet_engine.js` (copy of the Fig. 2 artifact engine PlanarEngine / GasEngine / DitherLab with membrane
loss and optional R(|sin χ|) tables; Task 1 shows it reproduces the published ensembles bit for bit). 2D in-plane ray
tracing, TM0 slab mode only, effective index n_eff in the membrane, specular DBR facets with a constant power reflectance R
per bounce, membrane power loss exp(−α_m L), α_m = dB/cm · 100 / 4.343 m⁻¹. Cells: 24-facet circle of 4 mm diameter, launch
7.5° from the wall normal, output port at 0.458 of the perimeter, 150 µm ports; Herriott 10 × 1.5 mm, N = 14, M = 3,
side walls, re-aimed launch (`retune`). Chaotic = 16 index discs, radius 50 µm, Δn_eff = ±0.01 (random sign), placed in
the ring crossed by the launch rays (circle) or anywhere (Herriott), layouts = mulberry32 seeds 1–30.

Static residual fringe of dither mode m (exit field against a stray copy of power fraction ε that follows the
wavelength but not the angle):

    ΔP/P_m = 2 √ε · V_m / √M,   V_none = 1, V_m = |time average of the fringe phasor over the dither|, M = speckle grains (M²)

with sine angle dither ±2 mrad at 1 kHz and triangle wavelength dither ±20 pm at 1.297 kHz; the static (0 Hz) component is
V. After averaging T the residual is r_m(T) = 2√ε/√M · [Σ_k P_k H(f_k T)²]^½ (boxcar, fringe drift 0.1 rad/s).
Noise-equivalent absorbance and limit:

    NEA(T) = √( r_both(T)² + δ_det(T)² ),   δ_det = √(NEP² + 2hνP_d) / (P_d √(2T)),   P_d = P_in 10^(−IL/10)
    c_min = NEA / (α_1ppm · Γ⟨L⟩)

(set A regression: δ_det = √(q / (P_in T_cell T)), shot noise only, as the published Fig. 3b). Spectral decorrelation
from the exit-path distribution p(L): C(Δν) = |Σ W_k e^{2πiΔν n L_k/c}|² / (Σ W_k)², N_λ = 1 / ⟨C(ν₁−ν₂)⟩ over a uniform
dither. Speckle numbers of the engine: N = (tr G)² / ‖G‖²_F of the exit-field coherence matrix over the dither states.

### Assumptions (conservative where a choice was needed)

1. Constant R per bounce (no angle dependence, no total internal reflection above the critical angle). The designed trench
   DBR reflects totally above ≈ 40° (n_eff 1.54) and the earlier step-1 study showed this raises the effective R; ignoring
   it is pessimistic.
2. Set B near-IR: 500 nm SiNx at the NH3 line 1531.7 nm, n_eff = 1.5391, Γ_E = 0.4532 (TM0 solver, Si3N4 Sellmeier of
   Luke et al.); mid-IR: 1000 nm at 3315.1 nm, n_eff = 1.4253, Γ_E = 0.5526 (C2H6 at 3.348 µm: 0.562). The proposal's
   Γ ≈ 0.45 is the E-field fraction Γ_E; the n_g-weighted Γ_abs of Fig. 1d is 0.392 (500 nm) — the limits use Γ_E.
3. The engine converts a wavelength change into phase with n_eff; the slab mode's group index is n_g = 2.28 (near-IR) /
   2.15 (mid-IR), so the engine's spectral decorrelation is conservative by ≈ 1.5×. Task 4 gives N_λ with both.
4. Dithers as Fig. 2f: ±20 pm at every wavelength (= ±2.56 GHz at 1.53 µm, ±0.55 GHz at 3.3 µm); the near-IR fringe at the
   ±0.55 GHz excursion is read off the same runs (stored fringe-vs-depth curve, grid point 4.2 pm).
5. ε = 1e-6 (placeholder of the proposal); ε only scales the fringe (Task 3).
6. The fringe residual after T assumes a fringe drift of 0.1 rad/s (as in the published Fig. 3b); a no-drift variant
   (residual = static floor at all T) is given in Task 7.
7. The cell loss IL (14 / 20 / 25 dB) of T1.3 replaces the simulated ray transmission in the detection limits; the
   simulated transmission is reported separately (it is low, 5–17 % for the circle: port geometry and mirror loss).
8. NEP 14 pW/√Hz, responsivity-free form of T1.3; breath at 35 °C, 1 atm, 5.5 % H2O, 4.5 % CO2 (HAPI 1.3, Voigt, air
   broadening); peak absorption of the analyte only (no spectral fitting of the background).
9. Folder `QUIET_sims_v32/` instead of `QUIET_sims/`: the repo already has `quiet_sims/`, and two names differing only in
   case break checkouts on macOS / Windows. Branch `quiet-v32-realistic`.
10. Not found in the repo: `slab_membrane_gamma.py` (the repo's `qs/slabmodes.py` TM0 solver is used, copied to `qv/`), the
    figure settings JSONs (the figures use the artifact defaults those JSONs were exported from: Liberation Sans 10 pt,
    16.8 cm, house colours), and no MATLAB pipeline.

## 2. Task 1: regression with set A

Set A: n_eff 2, Γ 0.2, R 0.995, ε 1e-6, no membrane loss, CH4 wavelength 1653.7 nm, 30 layouts. `results/t1/`,
`tests/test_t1_regression.py` (5 tests, including a bit-for-bit rerun of two layouts against the stored ensembles of the
published figures: PASS).

**Task 1: set A regression (30 layouts; median [IQR])**

| quantity | proposal v32 | rerun | PASS/FAIL |
|---|---|---|---|
| C-CMPC both dithers | 3.3e-8 | 3.2e-08 [1.1e-08–6.5e-08] | PASS |
| C-CMPC angle only | 3.2e-7 | 2.9e-07 [8.4e-08–8e-07] | PASS |
| R-CMPC both | – | 1e-07 |  |
| gain R/C, CMPC (both) | ~3× | 3.12× | PASS |
| gain R/C, Herriott (both) | ~80× | 55× (ratio over the C IQR: 20–166×) | PASS within IQR; median 55×, not 80× |
| NEA no dither, 1 s | 1.1e-4 | 0.00011 [0.00011–0.00012] | PASS |
| NEA both, 1 s | ~4e-8 | 3.7e-08 [2.4e-08–6.8e-08] | PASS |
| NH3 limit 1 s (median; min–max) | ~50 / 57 ppb; 28 ppb–2 ppm | 56.3 ppb; 27.1–2.13e+03 ppb | PASS |
| Γ⟨L⟩ | 3.2 cm | 3.28 cm | PASS |

The Herriott "≈ 80×" of Sec. 3.2 is not the median (55×); 80× lies inside the layout spread of the gain (20–166×). The
NH3 numbers use the published shot-noise-only model.

## 3. Task 2: realistic Fig. 2f (set B)

`results/t2/`, `figs/fig2f_setB_0.3dB.pdf`, `figs/fig2f_setB_0.5dB.pdf`, `figs/fig2ef_setB_R{0.95,0.97,0.995}.pdf`.

**Task 2: set B static residual fringe (ε = 1e-6), medians [IQR] over 30 layouts**

| cell | band | R | loss | none | spectral | angle | both, chaotic | both, regular | gain R/C | layouts <5e-7 | both @ ±0.55 GHz | gain @0.55 GHz | 5e-7 (median) | gain ≥2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| circle | nir | 0.95 | 0.3dB | 0.00014 | 5.4e-06 | 9.1e-07 | 1.8e-07 [5.3e-08–4e-07] | 3.7e-07 | 2.06 | 80 % | 7.5e-07 | 2.94 | PASS | PASS |
| circle | nir | 0.95 | 0.5dB | 0.00014 | 7.4e-06 | 1.1e-06 | 2.1e-07 [5.9e-08–4.6e-07] | 4.1e-07 | 1.93 | 77 % | 8.9e-07 | 2.62 | PASS | FAIL |
| circle | nir | 0.97 | 0.3dB | 0.00013 | 3.4e-06 | 7.2e-07 | 1.3e-07 [2.6e-08–2.8e-07] | 3e-07 | 2.27 | 83 % | 6.3e-07 | 3.10 | PASS | PASS |
| circle | nir | 0.97 | 0.5dB | 0.00014 | 3.6e-06 | 8.9e-07 | 1.7e-07 [3.2e-08–3.4e-07] | 3.6e-07 | 2.12 | 80 % | 7.7e-07 | 2.80 | PASS | PASS |
| circle | nir | 0.995 | 0.3dB | 0.00013 | 3e-06 | 4.7e-07 | 8.2e-08 [1.8e-08–2e-07] | 1.9e-07 | 2.27 | 90 % | 3.8e-07 | 3.83 | PASS | PASS |
| circle | nir | 0.995 | 0.5dB | 0.00013 | 3.1e-06 | 6e-07 | 1.1e-07 [2.4e-08–2.5e-07] | 2.7e-07 | 2.52 | 87 % | 5e-07 | 3.63 | PASS | PASS |
| herriott | nir | 0.95 | 0.3dB | 0.00029 | 1.4e-05 | 4.1e-05 | 1.7e-06 [7.6e-07–3.9e-06] | 5.8e-05 | 33.57 | 13 % | 3.7e-05 | 5.49 | FAIL | PASS |
| herriott | nir | 0.95 | 0.5dB | 0.00029 | 1.3e-05 | 2.6e-05 | 1.3e-06 [5.5e-07–3.4e-06] | 4e-06 | 3.09 | 20 % | 2.4e-05 | 3.29 | FAIL | PASS |
| herriott | nir | 0.97 | 0.3dB | 0.00027 | 1.2e-05 | 3.1e-05 | 1.8e-06 [5.7e-07–3.4e-06] | 5.9e-05 | 33.66 | 20 % | 2.8e-05 | 7.36 | FAIL | PASS |
| herriott | nir | 0.97 | 0.5dB | 0.00029 | 1.3e-05 | 2.5e-05 | 1.1e-06 [5.1e-07–3.1e-06] | 3.9e-06 | 3.52 | 27 % | 2.3e-05 | 3.27 | FAIL | PASS |
| herriott | nir | 0.995 | 0.3dB | 0.00024 | 9.8e-06 | 8.8e-06 | 5.3e-07 [2.5e-07–8.3e-07] | 6.1e-05 | 115.09 | 50 % | 5.6e-06 | 38.57 | FAIL | PASS |
| herriott | nir | 0.995 | 0.5dB | 0.00025 | 1.2e-05 | 3.5e-05 | 1.7e-06 [4.7e-07–4.1e-06] | 0.00015 | 86.43 | 27 % | 3.2e-05 | 17.74 | FAIL | PASS |
| circle | mir | 0.95 | 0.3dB | 0.00021 | 4.3e-05 | 1.6e-06 | 1.4e-06 [6.6e-07–3.3e-06] | 4.7e-06 | 3.29 | 20 % | 1.4e-06 (=) | 3.25 | FAIL | PASS |
| circle | mir | 0.95 | 0.5dB | 0.00022 | 4.4e-05 | 1.9e-06 | 1.6e-06 [7.5e-07–3.7e-06] | 5.2e-06 | 3.21 | 13 % | 1.6e-06 (=) | 3.17 | FAIL | PASS |
| circle | mir | 0.97 | 0.3dB | 0.00021 | 2.8e-05 | 1.2e-06 | 1.1e-06 [6.7e-07–2.7e-06] | 3.9e-06 | 3.59 | 20 % | 1.1e-06 (=) | 3.55 | FAIL | PASS |
| circle | mir | 0.97 | 0.5dB | 0.00021 | 2.8e-05 | 1.5e-06 | 1.3e-06 [7.8e-07–3.2e-06] | 4.6e-06 | 3.40 | 17 % | 1.3e-06 (=) | 3.35 | FAIL | PASS |
| circle | mir | 0.995 | 0.3dB | 0.0002 | 2.4e-05 | 8.6e-07 | 7.4e-07 [3.4e-07–1.7e-06] | 2.4e-06 | 3.29 | 40 % | 7.4e-07 (=) | 3.29 | FAIL | PASS |
| circle | mir | 0.995 | 0.5dB | 0.0002 | 2.5e-05 | 1.1e-06 | 9.2e-07 [4.6e-07–2.2e-06] | 3.5e-06 | 3.78 | 27 % | 9e-07 (=) | 3.78 | FAIL | PASS |

**Task 2: path, bounces, transmission, Γ⟨L⟩, speckle numbers (chaotic median; regular in brackets)**

| case | ⟨L⟩ cm | bounces | T_cell | Γ⟨L⟩ cm | N_λ (±20 pm) | N_θ (±2 mrad) | M (grains) |
|---|---|---|---|---|---|---|---|
| circle_nir_R0.95_0.3dB | 7.0 (10.0) | 17.6 (24.6) | 0.063 (0.047) | 3.16 (4.52) | 1.93 (1.18) | 1.74 (1.65) | 209 (53) |
| circle_nir_R0.95_0.5dB | 6.4 (9.5) | 16.1 (23.3) | 0.047 (0.030) | 2.89 (4.30) | 1.69 (1.17) | 1.66 (1.65) | 205 (52) |
| circle_nir_R0.97_0.3dB | 8.0 (11.0) | 20.8 (27.2) | 0.095 (0.080) | 3.61 (4.97) | 2.28 (1.23) | 1.81 (1.65) | 221 (55) |
| circle_nir_R0.97_0.5dB | 7.2 (10.1) | 18.3 (24.9) | 0.067 (0.049) | 3.24 (4.57) | 1.99 (1.19) | 1.74 (1.65) | 214 (53) |
| circle_nir_R0.995_0.3dB | 10.4 (13.7) | 27.8 (34.2) | 0.175 (0.172) | 4.69 (6.19) | 2.87 (1.42) | 1.89 (1.67) | 239 (63) |
| circle_nir_R0.995_0.5dB | 8.7 (11.5) | 22.9 (28.7) | 0.113 (0.097) | 3.95 (5.23) | 2.45 (1.27) | 1.81 (1.66) | 228 (57) |
| herriott_nir_R0.95_0.3dB | 5.5 (11.7) | 5.9 (10.7) | 0.304 (0.233) | 2.51 (5.29) | 1.25 (1.02) | 2.03 (1.68) | 49 (7) |
| herriott_nir_R0.95_0.5dB | 5.0 (4.2) | 5.0 (3.2) | 0.256 (0.504) | 2.25 (1.89) | 1.12 (1.96) | 2.07 (1.74) | 47 (2) |
| herriott_nir_R0.97_0.3dB | 6.4 (12.0) | 7.3 (11.0) | 0.316 (0.292) | 2.90 (5.45) | 1.56 (1.01) | 2.10 (1.67) | 54 (7) |
| herriott_nir_R0.97_0.5dB | 5.3 (4.2) | 5.6 (3.2) | 0.285 (0.539) | 2.39 (1.91) | 1.17 (1.93) | 2.02 (1.55) | 49 (2) |
| herriott_nir_R0.995_0.3dB | 10.9 (12.4) | 13.7 (11.4) | 0.311 (0.389) | 4.92 (5.62) | 2.82 (1.01) | 2.25 (1.67) | 70 (6) |
| herriott_nir_R0.995_0.5dB | 5.9 (14.0) | 7.0 (13.0) | 0.330 (0.187) | 2.67 (6.34) | 1.29 (1.00) | 2.04 (1.82) | 62 (1) |
| circle_mir_R0.95_0.3dB | 6.1 (9.5) | 15.1 (24.2) | 0.078 (0.051) | 3.36 (5.27) | 1.09 (1.04) | 1.48 (1.34) | 88 (72) |
| circle_mir_R0.95_0.5dB | 5.4 (9.2) | 13.5 (23.3) | 0.060 (0.033) | 3.00 (5.09) | 1.06 (1.04) | 1.46 (1.34) | 84 (71) |
| circle_mir_R0.97_0.3dB | 7.1 (10.2) | 18.0 (25.9) | 0.109 (0.086) | 3.90 (5.62) | 1.14 (1.05) | 1.50 (1.36) | 93 (74) |
| circle_mir_R0.97_0.5dB | 6.2 (9.6) | 15.7 (24.4) | 0.081 (0.054) | 3.45 (5.32) | 1.10 (1.04) | 1.48 (1.34) | 89 (73) |
| circle_mir_R0.995_0.3dB | 9.2 (11.9) | 24.9 (30.6) | 0.186 (0.174) | 5.10 (6.58) | 1.24 (1.17) | 1.52 (1.40) | 101 (76) |
| circle_mir_R0.995_0.5dB | 7.7 (10.5) | 19.9 (26.9) | 0.126 (0.104) | 4.25 (5.82) | 1.17 (1.07) | 1.51 (1.37) | 97 (75) |

Notes. (i) Near-IR C-CMPC: the M3 target holds at every R and loss with ±20 pm (median 0.8–2.1e-7, 77–90 % of layouts),
but at ±0.55 GHz (the 3.3 µm excursion) it does not (3.9–8.9e-7). (ii) The chaotic gain with both dithers is 1.9–2.5×
(circle) at all R; with the angle dither alone it is 3.6–4.5×. (iii) Herriott: the regular cell's re-aimed launch at
0.5 dB/cm picks a short orbit (3 bounces, 4.2 cm), which makes its fringe low and the gain small; at 0.3 dB/cm the gain
is 34× (R 95 %) to 115× (99.5 %), but the chaotic Herriott misses 5e-7 at R ≤ 97 % (1.1–1.8e-6). (iv) Mid-IR C-CMPC:
1.1–1.6e-6, FAIL at all R; N_λ ≈ 1.1 (the ±0.55 GHz dither hardly decorrelates the speckle). (v) Sensitivity (same runs, stored fringe-vs-depth curve; `results/t2/mir_dither_sensitivity.json`): a wider
wavelength dither at 3.3 µm, ±81 pm = ±2.2 GHz, gives 3.3e-7 (R 95 %), 2.4e-7 (97 %), 1.4e-7 (99.5 %) at 0.3 dB/cm, with
60–76 % of the layouts below 5e-7, but the chaotic gain falls to 1.7–3.5× (the regular cell also benefits). This is a
design option for T1.2, not part of set B.

## 4. Task 3: ε sweep

`results/t3/`, `figs/eps_sweep.pdf`. ΔP/P ∝ √ε exactly (log-log slope 0.500): in the model ε is only the stray amplitude,
while V and M are properties of the exit field. V and M do depend on R, loss, n_eff, wavelength and dither, so the
thresholds move with them; the static offset ε itself is not a fringe (only its drift would matter).

**Task 3: ε thresholds (both dithers; best chaotic cell circle_nir_R0.995_0.3dB; log-log slope 0.500)**

| case | ΔP/P at ε=1e-6 | ε_max for 5e-7, chaotic median [IQR] | regular | ε_max for 8e-8, chaotic [IQR] | regular |
|---|---|---|---|---|---|
| circle_nir_R0.95_0.3dB | 1.8e-07 | 7.6e-06 [1.6e-06–8.8e-05] | 1.8e-06 | 1.9e-07 [4.1e-08–2.3e-06] | 4.6e-08 |
| circle_nir_R0.95_0.5dB | 2.1e-07 | 5.4e-06 [1.2e-06–7.2e-05] | 1.5e-06 | 1.4e-07 [3e-08–1.8e-06] | 3.7e-08 |
| circle_nir_R0.97_0.3dB | 1.3e-07 | 1.4e-05 [3.2e-06–0.00037] | 2.7e-06 | 3.6e-07 [8.2e-08–9.4e-06] | 6.9e-08 |
| circle_nir_R0.97_0.5dB | 1.7e-07 | 8.5e-06 [2.2e-06–0.00024] | 1.9e-06 | 2.2e-07 [5.6e-08–6.1e-06] | 4.8e-08 |
| circle_nir_R0.995_0.3dB | 8.2e-08 | 3.7e-05 [6.6e-06–0.0008] | 7.2e-06 | 9.5e-07 [1.7e-07–2.1e-05] | 1.8e-07 |
| circle_nir_R0.995_0.5dB | 1.1e-07 | 2.2e-05 [3.9e-06–0.00043] | 3.4e-06 | 5.5e-07 [9.9e-08–1.1e-05] | 8.7e-08 |
| herriott_nir_R0.95_0.3dB | 1.7e-06 | 8.4e-08 [1.7e-08–4.3e-07] | 7.5e-11 | 2.2e-09 [4.3e-10–1.1e-08] | 1.9e-12 |
| herriott_nir_R0.95_0.5dB | 1.3e-06 | 1.5e-07 [2.2e-08–8.1e-07] | 1.5e-08 | 3.7e-09 [5.7e-10–2.1e-08] | 3.9e-10 |
| herriott_nir_R0.97_0.3dB | 1.8e-06 | 8.1e-08 [2.1e-08–7.8e-07] | 7.2e-11 | 2.1e-09 [5.5e-10–2e-08] | 1.8e-12 |
| herriott_nir_R0.97_0.5dB | 1.1e-06 | 2.1e-07 [2.6e-08–9.6e-07] | 1.7e-08 | 5.3e-09 [6.6e-10–2.5e-08] | 4.3e-10 |
| herriott_nir_R0.995_0.3dB | 5.3e-07 | 9e-07 [3.6e-07–3.9e-06] | 6.8e-11 | 2.3e-08 [9.3e-09–1e-07] | 1.7e-12 |
| herriott_nir_R0.995_0.5dB | 1.7e-06 | 8.2e-08 [1.5e-08–1.1e-06] | 1.1e-11 | 2.1e-09 [3.7e-10–2.9e-08] | 2.8e-13 |
| circle_mir_R0.95_0.3dB | 1.4e-06 | 1.2e-07 [2.3e-08–5.8e-07] | 1.1e-08 | 3.1e-09 [5.9e-10–1.5e-08] | 2.9e-10 |
| circle_mir_R0.95_0.5dB | 1.6e-06 | 9.5e-08 [1.8e-08–4.5e-07] | 9.2e-09 | 2.4e-09 [4.6e-10–1.1e-08] | 2.4e-10 |
| circle_mir_R0.97_0.3dB | 1.1e-06 | 2.2e-07 [3.4e-08–5.5e-07] | 1.7e-08 | 5.5e-09 [8.7e-10–1.4e-08] | 4.3e-10 |
| circle_mir_R0.97_0.5dB | 1.3e-06 | 1.4e-07 [2.5e-08–4.1e-07] | 1.2e-08 | 3.5e-09 [6.3e-10–1e-08] | 3.1e-10 |
| circle_mir_R0.995_0.3dB | 7.4e-07 | 4.6e-07 [8.4e-08–2.1e-06] | 4.2e-08 | 1.2e-08 [2.1e-09–5.4e-08] | 1.1e-09 |
| circle_mir_R0.995_0.5dB | 9.2e-07 | 3e-07 [5.2e-08–1.2e-06] | 2.1e-08 | 7.6e-09 [1.3e-09–3.1e-08] | 5.3e-10 |

The proposal's thresholds (5e-7 up to ε ≈ 2e-4, 8e-8 up to 6e-6) follow from set A (2.3e-4, 5.9e-6). At set B they are
25–30× tighter in the near-IR (R 95 %, 0.3 dB/cm: 7.6e-6 and 1.9e-7 for the median layout) and the mid-IR cell already
fails 5e-7 at ε = 1e-6 (ε_max 1.2–4.6e-7).

## 5. Task 4: path, bounces, N_λ for the 2 cm milestone

`results/t4/`. Γ = 0.45 → L_phys = 2 cm / 0.45 = 4.44 cm, i.e. 11–12 bounces in the circle (chord 0.37–0.40 cm) or ≈ 5 in
the Herriott cell (chord 0.8–1.0 cm); mirror + membrane loss 1.3–4.7 dB for R 99.5–95 %, 0.3–0.5 dB/cm.

**Task 4: 2 cm gas-equivalent milestone (Γ = 0.45 → L = 4.44 cm) and N_λ from p(L)**

| case | chord cm | bounces for 4.44 cm | mirror+membrane loss dB | simulated ⟨L⟩ cm | simulated Γ⟨L⟩ cm | N_λ ±0.55 GHz (n_eff / n_g) | N_λ ±20 pm (n_eff / n_g) | δν_c GHz (n_eff / n_g) |
|---|---|---|---|---|---|---|---|---|
| circle_nir_R0.95_0.3dB | 0.39 | 11.2 | 3.8 | 7.0 | 3.16 | 1.24 / 1.47 | 3.03 / 4.01 | 0.97 / 0.66 |
| circle_nir_R0.95_0.5dB | 0.40 | 11.1 | 4.7 | 6.4 | 2.89 | 1.20 / 1.38 | 2.71 / 3.53 | 1.13 / 0.77 |
| circle_nir_R0.97_0.3dB | 0.39 | 11.4 | 2.8 | 8.0 | 3.61 | 1.33 / 1.60 | 3.65 / 4.81 | 0.82 / 0.56 |
| circle_nir_R0.97_0.5dB | 0.39 | 11.3 | 3.7 | 7.2 | 3.24 | 1.26 / 1.50 | 3.15 / 4.15 | 0.92 / 0.63 |
| circle_nir_R0.995_0.3dB | 0.37 | 11.8 | 1.6 | 10.4 | 4.69 | 1.57 / 1.96 | 4.67 / 6.29 | 0.59 / 0.39 |
| circle_nir_R0.995_0.5dB | 0.38 | 11.6 | 2.5 | 8.7 | 3.95 | 1.40 / 1.72 | 3.93 / 5.27 | 0.73 / 0.49 |
| herriott_nir_R0.95_0.3dB | 0.96 | 4.6 | 2.4 | 5.5 | 2.51 | 1.18 / 1.26 | 1.48 / 1.50 | nan / nan |
| herriott_nir_R0.95_0.5dB | 0.99 | 4.4 | 3.2 | 5.0 | 2.25 | 1.10 / 1.17 | 1.35 / 1.35 | nan / nan |
| herriott_nir_R0.97_0.3dB | 0.90 | 4.9 | 2.0 | 6.4 | 2.90 | 1.29 / 1.42 | 1.75 / 1.76 | nan / nan |
| herriott_nir_R0.97_0.5dB | 0.94 | 4.7 | 2.8 | 5.3 | 2.39 | 1.14 / 1.22 | 1.44 / 1.43 | nan / nan |
| herriott_nir_R0.995_0.3dB | 0.81 | 5.4 | 1.4 | 10.9 | 4.92 | 1.63 / 2.09 | 4.50 / 5.20 | 0.56 / 0.37 |
| herriott_nir_R0.995_0.5dB | 0.85 | 5.2 | 2.3 | 5.9 | 2.67 | 1.22 / 1.33 | 1.64 / 1.66 | nan / nan |
| circle_mir_R0.95_0.3dB | 0.40 | 9.1 | 3.1 | 6.1 | 3.36 | 1.21 / 1.41 | 1.21 / 1.40 | 1.08 / 0.72 |
| circle_mir_R0.95_0.5dB | 0.40 | 9.0 | 3.8 | 5.4 | 3.00 | 1.17 / 1.33 | 1.16 / 1.32 | 1.21 / 0.81 |
| circle_mir_R0.97_0.3dB | 0.39 | 9.3 | 2.3 | 7.1 | 3.90 | 1.30 / 1.57 | 1.29 / 1.56 | 0.89 / 0.59 |
| circle_mir_R0.97_0.5dB | 0.40 | 9.1 | 3.0 | 6.2 | 3.45 | 1.23 / 1.44 | 1.23 / 1.43 | 1.05 / 0.68 |
| circle_mir_R0.995_0.3dB | 0.37 | 9.7 | 1.3 | 9.2 | 5.10 | 1.50 / 1.89 | 1.49 / 1.88 | 0.64 / 0.42 |
| circle_mir_R0.995_0.5dB | 0.38 | 9.5 | 2.0 | 7.7 | 4.25 | 1.35 / 1.65 | 1.35 / 1.65 | 0.80 / 0.52 |
| exponential p(L) reference | N_λ at ±0.55 GHz | δν_c (GHz) |
|---|---|---|
| exponential_pL_L10cm_n2.00 | 2.25 | 0.32 |
| exponential_pL_L4.44cm_n1.54 | 1.29 | 0.93 |
| exponential_pL_L4.44cm_n2.28 | 1.51 | 0.64 |

The old "∼10 cm physical, N_λ ≈ 2–2.7" is reproduced by an exponential p(L) with ⟨L⟩ = 10 cm at n = 2 (N_λ = 2.25 at
±0.55 GHz). At the milestone path (4.44 cm) N_λ falls to 1.29 (n_eff) / 1.51 (n_g); the simulated chaotic CMPC at 3.3 µm
gives 1.2–1.5 (n_eff) / 1.3–1.9 (n_g). With ±20 pm at 1.53 µm (±2.56 GHz) N_λ = 2.7–4.7 (n_eff) / 3.5–6.3 (n_g).

Detection-limit scaling at Γ⟨L⟩ = 2 cm (set-B noise; 14 dB, 1 mW, 1 s, both dithers): NH3 590–690 ppb (proposal: 80 ppb);
CH4 3.31 µm 10–19 ppb, C2H6 12–22 ppb (proposal: 0.6 ppb). The 150–230× line ratio does not carry over at equal noise: the
mid-IR fringe is 8–9× larger than the near-IR one (N_λ ≈ 1.1 at ±0.55 GHz) and dominates the 3.3 µm noise.

## 6. Task 5: angle-dither validity

`results/t5/`, `figs/angle_dither.pdf`. Set-B near-IR working point (R 97 %, 0.3 dB/cm), 30 layouts per range, speckle
correlation against the launch-angle change and N_θ for angle dithers of ±2 to ±50 mrad. Port mode count
M = 2 n_eff w / λ = 2·1.539·150 µm / 1.532 µm = 301 (the simulated exit field has 213–264 phase-space channels, consistent).

| cell, ±A | K_eff = N_θ chaotic [IQR] | regular | Δθ_c (1/e) chaotic | N_λ (±20 pm) | N_eff (both) | V_θ·√N_θ | V_joint·√(N_θN_λ) | fringe angle-only C / R | fringe both C / R |
|---|---|---|---|---|---|---|---|---|---|
| circle, ±2 mrad | 1.88 [1.64–1.98] | 1.46 | 1.16 mrad | 2.24 | 3.19 | 0.007 | 0.0022 | 7.2e-07 / 3e-06 | 1.3e-07 / 3e-07 |
| circle, ±5 mrad | 2.53 [2.41–2.82] | 2.38 | 1.28 mrad | 2.31 | 4.37 | 0.007 | 0.0024 | 5.5e-07 / 2.1e-06 | 1.4e-07 / 1.1e-07 |
| circle, ±10 mrad | 3.68 [3.24–4.20] | 3.43 | 1.30 mrad | 2.00 | 5.90 | 0.010 | 0.0048 | 7.3e-07 / 7.3e-06 | 2.3e-07 / 8.6e-07 |
| circle, ±20 mrad | 4.29 [4.09–4.48] | 2.71 | 1.49 mrad | 2.12 | 6.20 | 0.013 | 0.0055 | 8.9e-07 / 8.6e-06 | 2.4e-07 / 4.5e-06 |
| circle, ±50 mrad | 4.89 [4.30–5.97] | 4.66 | 2.67 mrad | 1.76 | 6.71 | 0.030 | 0.0080 | 1.7e-06 / 3.5e-06 | 3.2e-07 / 3.3e-06 |
| herriott, ±2 mrad | 2.15 [1.97–2.25] | 1.69 | 1.30 mrad | 1.51 | 3.29 | 0.177 | 0.0104 | 3.1e-05 / 0.00078 | 1.8e-06 / 5.9e-05 |
| herriott, ±5 mrad | 3.61 [3.31–3.90] | 2.57 | 1.31 mrad | 1.59 | 4.99 | 0.129 | 0.0084 | 2.3e-05 / 0.00055 | 1e-06 / 4.3e-05 |
| herriott, ±10 mrad | 5.39 [4.68–5.77] | 3.68 | 1.28 mrad | 1.47 | 6.63 | 0.114 | 0.0060 | 1.4e-05 / 0.00029 | 6.1e-07 / 2.5e-05 |
| herriott, ±20 mrad | 6.69 [6.18–7.20] | 3.43 | 1.37 mrad | 1.38 | 8.91 | 0.096 | 0.0051 | 1e-05 / 0.00032 | 4.7e-07 / 2.5e-05 |
| herriott, ±50 mrad | 8.30 [7.76–8.90] | 4.88 | 1.70 mrad | 1.48 | 11.46 | 0.081 | 0.0054 | 7.7e-06 / 0.00023 | 3.8e-07 / 2e-05 |

Findings (plainly): **the angle dither gives far fewer independent speckle patterns than assumed.** The exit speckle
decorrelates within Δθ_c ≈ 1.2–1.5 mrad (1/e), but K_eff = N_θ is only 1.9 (C-CMPC) and 2.2 (C-Herriott) for ±2 mrad and
saturates at 4.9 (CMPC) / 8.3 (Herriott) for ±50 mrad, two orders of magnitude below M ≈ 300 and well below the naive
1 + 2A/Δθ_c (≈ 4 for ±2 mrad, ≈ 70 for ±50 mrad). The correlation revives at Δθ ≈ 3.2 mrad (C ≈ 0.6 in both CMPCs, Fig. a):
the 24-facet geometry maps nearby launch angles onto the same exit channels, which caps K_eff. Chaos raises K_eff only
modestly (×1.0–1.7 over the regular cell).

**The (K N_λ)^(−1/2) scaling of T1.2 does not describe the simulated fringe.** The fringe visibility with the angle dither
is V_θ ≈ 5×10⁻³ in the C-CMPC (V_θ √N_θ ≈ 0.01, not 1), and log-log fits of V_θ against N_θ over the layouts give
slopes from −1.2 to +0.9, i.e. no power law. The angle dither suppresses the fringe by averaging the interference phase
between the exit light and the stray path (each mrad changes the exit path by many wavelengths), not by producing
independent speckle patterns; speckle-pattern counting (K, N_λ) measures a different quantity (the intensity-pattern
diversity at the detector). The two must be kept apart in T1.2: the fringe suppression is real and simulated, the
"K ≲ M independent patterns" picture is not supported. A wider angle range does not help the fringe monotonically
(C-CMPC angle-only 7.2e-7 at ±2 mrad, 5.5e-7 at ±5, 1.7e-6 at ±50 mrad: wider dithers move the beam onto different
orbits), while the Herriott cell improves steadily (3.1e-5 → 7.7e-6).

## 6b. Task 5b: wave-optics limit of the angle dither (model limitation found after the runs)

`scripts/t5b_wave_bound.py` → `results/t5/wave_bound.json`. Question: why does the angle dither give so few independent
speckle patterns (K_eff ≈ 2), although the chaotic cell is strongly chaotic? It is: the finite-time Lyapunov exponent of
the C-CMPC is 0.53 per bounce (regular 0.018), and with ≈ 21 bounces a ray perturbation is stretched by ≈ 6×10⁴. Chaos is
not the bottleneck; diffraction at the input is.

The cell is linear and time-invariant, so the exit field for a tilted input is E(θ) = U E_in(θ), with
E_in(θ) = E_in(x) e^{i k n θ x}. The time-averaged (static) fringe of a sine angle dither of amplitude A is therefore the
fringe of the averaged input field E_in(x) J0(k n A x), whatever U does. For the 40 µm launch waist the beam's own
divergence is λ / (π n w) = 7.9 mrad; a ±2 mrad tilt stays inside it and hardly changes the input field:

| input waist w | divergence | V_wave ±2 mrad | ±5 | ±10 | ±20 | ±50 mrad |
|---|---|---|---|---|---|---|
| 40 µm (as simulated) | 7.9 mrad | 0.98 | 0.92 | 0.77 | 0.61 | 0.43 |
| 75 µm | 4.2 mrad | 0.95 | 0.79 | 0.62 | 0.48 | 0.33 |
| 150 µm (port width) | 2.1 mrad | 0.84 | 0.62 | 0.48 | 0.37 | 0.25 |

(V_wave = √⟨J0(k n A x)²⟩ over the input intensity: rms static fringe factor, coherent part plus a random part from the
orthogonal remainder.) The engine's V_θ ≈ 5×10⁻³ follows one centre ray per dither angle, whose exit path changes by many
wavelengths per mrad; a ray has no width, so the engine resolves tilts far below the diffraction limit. Its speckle
count K_eff ≈ 1.9 (from the full beam bundle) is consistent with the wave picture (≈ 1 + 2A / divergence); its fringe
factor V_θ is not. The classical chaos amplifies sub-diffraction perturbations only for rays; a wave field cannot resolve
them (the optical Ehrenfest-time argument).

Consequence (circle, set B, 0.3 dB/cm, ε = 1e-6, both dithers, static): engine 1.8e-7 / 1.3e-7 / 8.2e-8 at R 95 / 97 /
99.5 %, wave-bounded 5.3e-6 / 3.4e-6 / 2.9e-6, i.e. essentially the wavelength dither alone (±20 pm), 6–11× above the M3
target 5e-7. The "angle" and "both" results of Tasks 2, 3, 5, 7, 8 and of the Fig. 2 / Fig. 3 artifacts are optimistic
by this factor for ±2 mrad; the wavelength-dither results stand (a fixed path difference sweeps its fringe phase under
a wavelength change, a genuine wave effect). Not a coding bug: the single-ray phase model is a physics limitation of the
engine. Results are kept as computed and flagged here.

Ways to make the decorrelation real (to be simulated with a wave-consistent model before relying on them):
1. Sweep more input modes: dither range ≫ divergence, i.e. a larger input waist (w = 150 µm: ±10 mrad gives V ≈ 0.5),
   or a combined angle and position dither covering N ≈ (Δx Δθ) n / λ input modes; V ∝ N^(−1/2) at best.
2. Modulate inside the cell, not at the input: a non-uniform internal perturbation (local thermo-optic heaters over a
   few inclusions, a vibrating or heated facet) changes the relative phases of the many paths; each independent setting is
   a new speckle realisation (up to M ≈ 300), and chaos then does help. A uniform index or temperature change is only a
   wavelength dither in disguise.
3. Deeper wavelength dither: V_λ falls with the dither depth (Task 2 mid-IR: ±2.2 GHz gives 4–5× over ±0.55 GHz).
4. Lower ε (fringe ∝ √ε): ε = 5×10⁻⁸ brings 3.4e-6 to 7.6e-7 at R 97 %.

## 7. Task 6: inclusions

`results/t6/`. One parameter at a time around 16 discs × Δn_eff 0.01 × r 50 µm (circle, set-B near-IR, R 97 %,
0.3 dB/cm, 30 layouts; regular cell 3.05e-7 with both dithers). Slab check (`results/t6/slab_step.json`):
dn_eff/dt = 1.48×10⁻³ per nm (TM0, 500 nm, 1.532 µm) — the proposal's 1.5×10⁻³ is confirmed; 1–7 nm steps give
Δn_eff = 0.0015–0.0105 and a normal-incidence reflection (Δn/2n)² ≤ 1.2×10⁻⁵ per interface. For the 1000 nm mid-IR film
dn_eff/dt = 7.2×10⁻⁴ per nm: Δn_eff = 0.01 needs ≈ 14 nm.

| inclusions | step (nm, 500 nm film) | both, chaotic [IQR] | gain R/C | angle only | FTLE /hit | p(L) width | coverage | N_λ | Γ⟨L⟩ cm | T_cell | Fresnel R: normal / mean over impact | redirected per path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dn=0.001 | 0.7 | 5.1e-07 [3.2e-07–8.9e-07] | 0.60 | 5.2e-06 | 0.21 | 0.40 | 0.97 | 1.37 | 4.86 | 0.070 | 1.1e-07 / 9.4e-05 | 1.3e-03 |
| dn=0.002 | 1.4 | 4.3e-07 [1.4e-07–6.6e-07] | 0.70 | 3.7e-06 | 0.27 | 0.41 | 1.00 | 1.79 | 4.66 | 0.072 | 4.2e-07 / 2.0e-04 | 2.7e-03 |
| dn=0.005 | 3.4 | 1e-07 [5.9e-08–1.8e-07] | 2.96 | 7.5e-07 | 0.39 | 0.52 | 1.00 | 2.17 | 4.23 | 0.075 | 2.6e-06 / 5.3e-04 | 6.5e-03 |
| default=16 x 0.01 x 50 um | 6.8 | 1.3e-07 [2.6e-08–2.8e-07] | 2.27 | 7.2e-07 | 0.53 | 0.60 | 1.00 | 2.28 | 3.61 | 0.095 | 1.1e-05 / 1.1e-03 | 1.1e-02 |
| n=4 | 6.8 | 1.3e-07 [4.9e-08–2.3e-07] | 2.27 | 9.3e-07 | 0.27 | 0.48 | 1.00 | 2.04 | 4.25 | 0.080 | 1.1e-05 / 1.1e-03 | 3.3e-03 |
| n=8 | 6.8 | 9.6e-08 [3.7e-08–1.6e-07] | 3.16 | 6.3e-07 | 0.39 | 0.52 | 1.00 | 2.44 | 4.17 | 0.073 | 1.1e-05 / 1.1e-03 | 6.5e-03 |
| n=32 | 6.8 | 9.5e-08 [3.4e-08–2e-07] | 3.20 | 4.7e-07 | 0.77 | 0.65 | 1.00 | 1.98 | 3.31 | 0.104 | 1.1e-05 / 1.1e-03 | 2.1e-02 |
| radius_um=25 | 6.8 | 4.3e-08 [1.7e-08–1.3e-07] | 7.16 | 3.3e-07 | 0.51 | 0.55 | 1.00 | 2.06 | 3.99 | 0.085 | 1.1e-05 / 1.1e-03 | 6.2e-03 |
| radius_um=100 | 6.8 | 1.7e-07 [6e-08–7.6e-07] | 1.78 | 6.5e-07 | 0.57 | 0.68 | 1.00 | 2.29 | 3.18 | 0.102 | 1.1e-05 / 1.1e-03 | 2.0e-02 |

Launch-beam Poincaré coverage (5 rays, 2000 hits, 40 × 20 bins): regular CMPC 0.50, chaotic CMPC 0.97–1.00 for every
inclusion set; regular Herriott 0.015, chaotic Herriott 0.94.

Findings. (i) **Weak inclusions do not help:** at Δn_eff = 0.001–0.002 (≈ 0.7–1.4 nm steps) the chaotic cell is *worse*
than the regular one (5.1e-7 and 4.3e-7 vs 3.05e-7), although its phase space is already fully covered and its FTLE is
positive (0.21–0.27 per hit). The fringe gain needs Δn_eff ≥ 0.005 (≥ 3.4 nm step): 1.0e-7 (gain 3.0×) at 0.005, 1.3e-7
(2.3×) at 0.01. (ii) The number of discs (4–32) hardly matters (0.95–1.3e-7). (iii) Smaller discs are better:
r = 25 µm gives 4.3e-8 (gain 7.2×), 100 µm 1.7e-7 with a wide spread. (iv) Chaos indicators do not rank the fringe: the
FTLE grows monotonically with Δn and number (0.21 → 0.77 per hit) and the coverage saturates at 1, while the fringe does
not follow. (v) Fresnel reflection at the disc edges: normal incidence 1e-7–1e-5, but averaged over the impact parameter
(grazing hits dominate, total internal reflection when leaving) 1e-4–1e-3 per interface; with 3–19 interfaces per path
0.3–2 % of the light is redirected per path. This light stays in the cell (it is what makes the cell chaotic), so it is not
a loss in the 2D model; its indirect cost is a shorter dwell: Γ⟨L⟩ falls from 4.9 cm (Δn 0.001) to 3.2–3.3 cm (32 discs or
r 100 µm). Out-of-plane scattering at the etched step edge is not modelled and must be measured (T2.3).

## 8. Task 7: detection limits (Fig. 3 for set B)

`results/t7/`, `figs/fig3_setB.pdf` (circle, R 97 %, 0.3 dB/cm, 14 dB, 1 mW; (a) spectra with noise realisations from the
exit paths of the representative layout, (b) NEA vs T at the NH3 line). HITRAN peaks of 1 ppm in breath (35 °C, 1 atm):
NH3 6528.77 cm⁻¹ 2.28e-7 cm⁻¹; CH4 3016.49 cm⁻¹ 3.79e-5; C2H6 Q-branch 2996.85 cm⁻¹ 3.34e-5 (strongest C2H6 line in the
window, 2986.68 cm⁻¹: 3.69e-5); CH4 1.64 µm (6114.65 cm⁻¹) 1.54e-7. Ratios to NH3: 167× (CH4), 147× (C2H6 Q).

Verification of the T1.3 / M7 statements (2 cm gas-equivalent):

| statement | check | verdict |
|---|---|---|
| 10 ppb needs δ ≈ 6e-7 | CH4 7.6e-7, C2H6 (Q) 6.7e-7 | PASS (≈) |
| detector share 3e-7 leaves the rest to the fringe | fringe residual at 1 s, 3.3 µm: 7.4e-7 (R 99.5 %), 1.07e-6 (97 %), 1.43e-6 (95 %) — alone above the whole budget | FAIL |
| T ≈ 1 s at 14 dB (1 mW) | δ_det(1 s) = 2.5e-7 → T(3e-7) = 0.7 s | PASS |
| T ≈ 11 s at 20 dB | 11.0 s | PASS |
| 25 dB needs ≈ 3 mW for 15 s | 12.2 s at 3 mW (109 s at 1 mW) | PASS |
| ethane 2 ppb: ~8 s at 14 dB/1 mW or 1 s at 20 dB/10 mW | detector alone: 3.5 s / 1 s; with the simulated fringe: not within 100 s (fringe floor 11–21 ppb at 2 cm) | FAIL |
| 3.3 µm lines 150–230× stronger than NH3 | 147–167× | partly |

**Task 7: detection limits at T = 1 s and 15 s (circle, chaotic median [IQR], 0.3 dB/cm, both dithers, drift 0.1 rad/s)**

| line | R | IL dB | P_in mW | 1 s, simulated Γ⟨L⟩ | 15 s, simulated Γ⟨L⟩ | 1 s, Γ⟨L⟩ = 2 cm | 15 s, Γ⟨L⟩ = 2 cm |
|---|---|---|---|---|---|---|---|
| NH3_1.53 | 0.95 | 14 | 1.0 | 479 [3.8e+02–6e+02] | 203 [1.1e+02–3.5e+02] | 691 [5.7e+02–1e+03] | 311 [1.7e+02–6.2e+02] |
| NH3_1.53 | 0.95 | 20 | 1.0 | 1.48e+03 [1.4e+03–1.7e+03] | 442 [3.8e+02–4.9e+02] | 2.23e+03 [2.2e+03–2.4e+03] | 631 [5.7e+02–8.2e+02] |
| NH3_1.53 | 0.95 | 20 | 10.0 | 303 [1.8e+02–5.1e+02] | 183 [55–3.4e+02] | 463 [2.6e+02–9e+02] | 280 [1e+02–6e+02] |
| NH3_1.53 | 0.95 | 25 | 1.0 | 4.5e+03 [4.1e+03–5.1e+03] | 1.19e+03 [1.1e+03–1.4e+03] | 6.9e+03 [6.9e+03–6.9e+03] | 1.8e+03 [1.8e+03–1.9e+03] |
| NH3_1.53 | 0.95 | 25 | 3.0 | 1.55e+03 [1.4e+03–1.8e+03] | 457 [4e+02–5.1e+02] | 2.34e+03 [2.3e+03–2.5e+03] | 658 [6e+02–8.4e+02] |
| NH3_1.53 | 0.97 | 14 | 1.0 | 377 [3.3e+02–4.4e+02] | 145 [94–2.3e+02] | 635 [5.6e+02–8.3e+02] | 249 [1.5e+02–4.5e+02] |
| NH3_1.53 | 0.97 | 20 | 1.0 | 1.25e+03 [1.1e+03–1.4e+03] | 366 [3.2e+02–3.9e+02] | 2.21e+03 [2.2e+03–2.3e+03] | 601 [5.7e+02–7.1e+02] |
| NH3_1.53 | 0.97 | 20 | 10.0 | 218 [1.5e+02–3.3e+02] | 122 [45–2.2e+02] | 376 [2.4e+02–6.6e+02] | 211 [72–4.3e+02] |
| NH3_1.53 | 0.97 | 25 | 1.0 | 3.88e+03 [3.6e+03–4.3e+03] | 1.01e+03 [9.2e+02–1.2e+03] | 6.9e+03 [6.9e+03–6.9e+03] | 1.79e+03 [1.8e+03–1.8e+03] |
| NH3_1.53 | 0.97 | 25 | 3.0 | 1.31e+03 [1.2e+03–1.5e+03] | 381 [3.3e+02–4.1e+02] | 2.33e+03 [2.3e+03–2.4e+03] | 630 [6e+02–7.3e+02] |
| NH3_1.53 | 0.995 | 14 | 1.0 | 272 [2.4e+02–2.9e+02] | 83.4 [67–1.3e+02] | 589 [5.6e+02–7.1e+02] | 191 [1.5e+02–3.3e+02] |
| NH3_1.53 | 0.995 | 20 | 1.0 | 981 [8.9e+02–1.1e+03] | 266 [2.4e+02–2.8e+02] | 2.2e+03 [2.2e+03–2.2e+03] | 579 [5.7e+02–6.4e+02] |
| NH3_1.53 | 0.995 | 20 | 10.0 | 129 [1.1e+02–1.9e+02] | 58.9 [31–1.2e+02] | 294 [2.3e+02–4.9e+02] | 138 [66–3e+02] |
| NH3_1.53 | 0.995 | 25 | 1.0 | 2.97e+03 [2.8e+03–3.3e+03] | 794 [7.2e+02–8.5e+02] | 6.89e+03 [6.9e+03–6.9e+03] | 1.78e+03 [1.8e+03–1.8e+03] |
| NH3_1.53 | 0.995 | 25 | 3.0 | 1.03e+03 [9.4e+02–1.1e+03] | 279 [2.5e+02–3e+02] | 2.31e+03 [2.3e+03–2.3e+03] | 609 [6e+02–6.6e+02] |
| CH4_3.31 | 0.95 | 14 | 1.0 | 12 [5.5–27] | 8.11 [3.6–18] | 19.1 [9.3–43] | 13 [6–30] |
| CH4_3.31 | 0.95 | 20 | 1.0 | 14.4 [10–28] | 8.38 [4.1–18] | 22.9 [16–45] | 13.4 [6.8–30] |
| CH4_3.31 | 0.95 | 20 | 10.0 | 11.8 [5.2–26] | 8.1 [3.6–18] | 18.9 [8.8–43] | 12.9 [6–30] |
| CH4_3.31 | 0.95 | 25 | 1.0 | 30 [27–37] | 10.5 [7.9–19] | 45.4 [42–60] | 16.8 [12–32] |
| CH4_3.31 | 0.95 | 25 | 3.0 | 14.6 [11–28] | 8.41 [4.1–18] | 23.3 [16–46] | 13.4 [6.9–30] |
| CH4_3.31 | 0.97 | 14 | 1.0 | 7.15 [4.5–18] | 4.78 [3–13] | 14.5 [9.5–36] | 9.76 [6.2–24] |
| CH4_3.31 | 0.97 | 20 | 1.0 | 9.8 [8.4–20] | 5.09 [3.4–13] | 19.3 [16–38] | 10.3 [7–25] |
| CH4_3.31 | 0.97 | 20 | 10.0 | 6.96 [4.3–18] | 4.76 [2.9–13] | 14.2 [9–36] | 9.73 [6.1–24] |
| CH4_3.31 | 0.97 | 25 | 1.0 | 24.1 [22–28] | 7.45 [6.5–14] | 43.6 [42–55] | 14.4 [12–27] |
| CH4_3.31 | 0.97 | 25 | 3.0 | 10.1 [8.8–20] | 5.12 [3.4–13] | 19.8 [16–38] | 10.4 [7.1–25] |
| CH4_3.31 | 0.995 | 14 | 1.0 | 4.05 [2.3–9.1] | 2.67 [1.4–6.2] | 10.3 [5.6–23] | 6.75 [3.2–16] |
| CH4_3.31 | 0.995 | 20 | 1.0 | 6.42 [5.5–10] | 2.94 [1.9–6.3] | 16.3 [14–26] | 7.5 [4.6–16] |
| CH4_3.31 | 0.995 | 20 | 10.0 | 3.89 [2–9] | 2.66 [1.3–6.2] | 9.83 [4.7–23] | 6.7 [3.1–16] |
| CH4_3.31 | 0.995 | 25 | 1.0 | 18 [16–20] | 5.03 [4.4–7.6] | 42.4 [42–47] | 12.6 [11–19] |
| CH4_3.31 | 0.995 | 25 | 3.0 | 6.68 [5.7–11] | 2.97 [2–6.4] | 16.9 [15–27] | 7.58 [4.7–16] |
| C2H6_3.34Q | 0.95 | 14 | 1.0 | 13.3 [6.2–30] | 9.05 [4–20] | 21.7 [11–49] | 14.7 [6.8–34] |
| C2H6_3.34Q | 0.95 | 20 | 1.0 | 16.1 [12–31] | 9.35 [4.5–20] | 26 [18–51] | 15.2 [7.8–34] |
| C2H6_3.34Q | 0.95 | 20 | 10.0 | 13.2 [5.9–30] | 9.03 [4–20] | 21.4 [9.9–49] | 14.7 [6.8–34] |
| C2H6_3.34Q | 0.95 | 25 | 1.0 | 33.5 [30–41] | 11.7 [8.8–22] | 51.5 [48–68] | 19 [14–36] |
| C2H6_3.34Q | 0.95 | 25 | 3.0 | 16.3 [12–31] | 9.38 [4.6–20] | 26.5 [18–52] | 15.2 [7.9–34] |
| C2H6_3.34Q | 0.97 | 14 | 1.0 | 7.97 [5–21] | 5.33 [3.3–14] | 16.5 [11–41] | 11.1 [7–28] |
| C2H6_3.34Q | 0.97 | 20 | 1.0 | 10.9 [9.4–22] | 5.67 [3.7–14] | 21.9 [18–43] | 11.7 [7.9–28] |
| C2H6_3.34Q | 0.97 | 20 | 10.0 | 7.77 [4.8–21] | 5.31 [3.3–14] | 16.1 [10–40] | 11 [6.9–28] |
| C2H6_3.34Q | 0.97 | 25 | 1.0 | 26.8 [24–32] | 8.3 [7.3–15] | 49.5 [48–62] | 16.4 [14–30] |
| C2H6_3.34Q | 0.97 | 25 | 3.0 | 11.2 [9.8–22] | 5.71 [3.8–14] | 22.4 [19–43] | 11.8 [8–28] |
| C2H6_3.34Q | 0.995 | 14 | 1.0 | 4.52 [2.6–10] | 2.98 [1.5–6.9] | 11.7 [6.4–26] | 7.66 [3.7–18] |
| C2H6_3.34Q | 0.995 | 20 | 1.0 | 7.16 [6.1–12] | 3.28 [2.1–7.1] | 18.5 [16–30] | 8.51 [5.2–18] |
| C2H6_3.34Q | 0.995 | 20 | 10.0 | 4.34 [2.2–10] | 2.97 [1.5–6.9] | 11.2 [5.4–26] | 7.61 [3.5–18] |
| C2H6_3.34Q | 0.995 | 25 | 1.0 | 20.1 [18–22] | 5.61 [4.9–8.4] | 48.1 [47–54] | 14.3 [13–22] |
| C2H6_3.34Q | 0.995 | 25 | 3.0 | 7.45 [6.4–12] | 3.31 [2.2–7.1] | 19.2 [16–30] | 8.6 [5.4–18] |

**Task 7 (conservative): no fringe drift, residual = static floor at all T; Γ⟨L⟩ = 2 cm; median [IQR], ppb**

| line | R | 14 dB, 1 mW, 1 s | 14 dB, 1 mW, 15 s | 20 dB, 10 mW, 1 s | 25 dB, 3 mW, 15 s | fringe floor alone |
|---|---|---|---|---|---|---|
| NH3_1.53 | 0.95 | 692 [5.7e+02–1e+03] | 426 [1.9e+02–8.8e+02] | 464 [2.6e+02–9e+02] | 721 [6.1e+02–1.1e+03] | 399 |
| NH3_1.53 | 0.97 | 635 [5.6e+02–8.3e+02] | 329 [1.6e+02–6.3e+02] | 376 [2.4e+02–6.6e+02] | 666 [6e+02–8.6e+02] | 295 |
| NH3_1.53 | 0.995 | 589 [5.6e+02–7.1e+02] | 232 [1.5e+02–4.5e+02] | 294 [2.3e+02–4.9e+02] | 623 [6e+02–7.3e+02] | 181 |
| CH4_3.31 | 0.95 | 19.1 [9.3–44] | 18.9 [8.7–43] | 18.9 [8.8–43] | 19.2 [9.4–44] | 18.9 |
| CH4_3.31 | 0.97 | 14.6 [9.5–36] | 14.2 [8.9–36] | 14.2 [9–36] | 14.6 [9.6–36] | 14.2 |
| CH4_3.31 | 0.995 | 10.3 [5.6–23] | 9.8 [4.6–23] | 9.85 [4.7–23] | 10.4 [5.8–23] | 9.76 |
| C2H6_3.34Q | 0.95 | 21.7 [11–49] | 21.4 [9.9–49] | 21.5 [10–49] | 21.8 [11–49] | 21.4 |
| C2H6_3.34Q | 0.97 | 16.5 [11–41] | 16.1 [10–40] | 16.2 [10–40] | 16.6 [11–41] | 16.1 |
| C2H6_3.34Q | 0.995 | 11.7 [6.4–26] | 11.1 [5.2–26] | 11.2 [5.4–26] | 11.8 [6.5–26] | 11.1 |

**Task 7: averaging time to the M7 targets (median layout; s; '–' = not within 100 s); 0.3 dB/cm**

| R | IL dB | P_in mW | Γ⟨L⟩ basis | CH4 10 ppb | C2H6 10 ppb | C2H6 2 ppb | NH3 100 ppb |
|---|---|---|---|---|---|---|---|
| 0.95 | 14 | 1 | 2cm | 24.4 | 27 | – | 73.9 |
| 0.95 | 14 | 1 | sim | 9.86 | 13.3 | 100 | 44.7 |
| 0.95 | 20 | 1 | 2cm | 24.4 | 29.9 | – | – |
| 0.95 | 20 | 1 | sim | 10.9 | 13.3 | – | – |
| 0.95 | 20 | 10 | 2cm | 24.4 | 27 | – | 60.4 |
| 0.95 | 20 | 10 | sim | 8.91 | 13.3 | 100 | 36.5 |
| 0.95 | 25 | 1 | 2cm | 36.5 | 44.7 | – | – |
| 0.95 | 25 | 1 | sim | 18 | 19.9 | – | – |
| 0.95 | 25 | 3 | 2cm | 24.4 | 29.9 | – | – |
| 0.95 | 25 | 3 | sim | 10.9 | 13.3 | – | – |
| 0.97 | 14 | 1 | 2cm | 14.8 | 18 | – | 60.4 |
| 0.97 | 14 | 1 | sim | 0.064 | 0.106 | 54.6 | 29.9 |
| 0.97 | 20 | 1 | 2cm | 16.3 | 19.9 | – | – |
| 0.97 | 20 | 1 | sim | 0.972 | 1.61 | 66.8 | – |
| 0.97 | 20 | 10 | 2cm | 14.8 | 18 | – | 44.7 |
| 0.97 | 20 | 10 | sim | 0.0104 | 0.0173 | 54.6 | 22.1 |
| 0.97 | 25 | 1 | 2cm | 29.9 | 36.5 | – | – |
| 0.97 | 25 | 1 | sim | 8.06 | 10.9 | – | – |
| 0.97 | 25 | 3 | 2cm | 16.3 | 19.9 | – | – |
| 0.97 | 25 | 3 | sim | 1.07 | 1.78 | 66.8 | – |
| 0.995 | 14 | 1 | 2cm | 1.78 | 7.29 | 81.8 | 49.4 |
| 0.995 | 14 | 1 | sim | 0.0234 | 0.0286 | 29.9 | 10.9 |
| 0.995 | 20 | 1 | 2cm | 7.29 | 10.9 | – | – |
| 0.995 | 20 | 1 | sim | 0.355 | 0.434 | 33 | 100 |
| 0.995 | 20 | 10 | 2cm | 0.392 | 7.29 | 81.8 | 27 |
| 0.995 | 20 | 10 | sim | 0.0104 | 0.0104 | 29.9 | 2.66 |
| 0.995 | 25 | 1 | 2cm | 24.4 | 29.9 | – | – |
| 0.995 | 25 | 1 | sim | 3.6 | 4.4 | – | – |
| 0.995 | 25 | 3 | 2cm | 8.06 | 10.9 | – | – |
| 0.995 | 25 | 3 | sim | 0.392 | 0.48 | 33 | – |

## 9. Task 8: layout spread

`results/t8/summary.json` (all 100 layouts with every metric), `figs/layout_spread.pdf`. Set-B near-IR working point
(circle, R 97 %, 0.3 dB/cm), layouts 1–100, NH3 limit at 1 s with the T1.3 noise model (14 dB, 1 mW):

| quantity (100 layouts) | median [IQR] | min – max |
|---|---|---|
| NH3 limit, simulated Γ⟨L⟩ | 399 ppb [332–583] | 259 ppb – 3.5 ppm |
| NH3 limit, Γ⟨L⟩ = 2 cm | 659 ppb [571–924] | 561 ppb – 5.9 ppm |
| fringe, both dithers | 1.6e-7 [4.9e-8–3.4e-7] | 3.3e-9 – 2.7e-6 |
| layouts below 5e-7 | 79 % | |

Spearman rank correlation with the fringe (both dithers) / with the limit: mean number of passes of the detected light
n_pass −0.47 / −0.55; Γ⟨L⟩ +0.07 / −0.39; relative p(L) width −0.04 / +0.37; FTLE −0.21 / −0.17; transmission −0.12 / +0.26;
N_λ +0.10 / +0.14; N_θ +0.06 / +0.02; launch-beam Poincaré coverage −0.01 / −0.01 and entropy −0.03 / +0.04.

**Design rule.** The only useful single predictor is n_pass, the power-weighted mean number of wall passes of the light that
reaches the output port (computable by ray tracing a candidate layout in seconds): keeping layouts with n_pass ≥ 36 (the
median) selects the better half with a 70 % hit rate and halves the fringe (7.2e-8 vs 2.6e-7 median). The chaos indicators
(coverage, entropy, FTLE) do not predict good layouts: every inclusion layout is fully chaotic by them, while the fringe
still spreads over three decades. With the detector-limited NH3 noise, the spread of the limit is smaller than that of the
fringe (IQR ×1.8), except for a tail of ≈ 15 % of layouts above 1 ppm (fringe-limited).

## 10. Discrepancies with the proposal (v32)

1. **Herriott chaotic gain "≈ 80×" (Sec. 3.2)**: the median is 55× (set A; the stored ensembles of the published figure
   give the same). 80× lies inside the layout spread (20–166×). Say "≈ 55× (median)".
2. **Spectral dither ±20 pm is not ±0.55 GHz in the near-IR.** Fig. 2f and Fig. 3b were simulated at 1.65 µm, where ±20 pm
   is ±2.2 GHz (±2.6 GHz at 1.53 µm). Only at 3.3 µm is ±20 pm equal to ±0.55 GHz. At ±0.55 GHz the near-IR C-CMPC is at
   7.5e-7 (R 95 %), above the M3 target.
3. **n_eff = 2 → 1.54** (500 nm, TM0, 1.532 µm) as the proposal already notes; **Γ = 0.45 is Γ_E** (E-field fraction); the
   Fig. 1d definition (n_g-weighted Γ_abs) gives 0.39. Use one definition (the limits here use Γ_E).
4. **Phase index vs group index.** The engine (and the "decorrelation width ≈ ±0.55 GHz for 10 cm" estimate of T1.2) uses
   n_eff; a frequency change acts through n_g = 2.28 (near-IR) / 2.15 (mid-IR), 1.5× faster decorrelation. The simulated
   numbers are therefore conservative in N_λ.
5. **"N_λ ≈ 2 at ±0.55 GHz for ~10 cm physical paths"** holds for an exponential p(L) with ⟨L⟩ = 10 cm at n = 2 (2.25), but
   the 2 cm milestone at Γ = 0.45 needs only 4.4 cm: N_λ = 1.3 (n_eff) – 1.5 (n_g); the simulated 3.3 µm CMPC gives 1.1–1.5
   (engine) / 1.3–1.9 (p(L), n_g).
6. **ε thresholds** "5e-7 up to ε ≈ 2e-4; 8e-8 up to 6e-6" are set-A numbers. Set B near-IR R 95 %: 7.6e-6 and 1.9e-7
   (median layout); mid-IR: 5e-7 is not met even at ε = 1e-6 (ε_max 1.2–4.6e-7). The √ε scaling itself is exact.
7. **Angle dither and the (K N_λ)^(−1/2) law (T1.2).** K_eff = 1.9–2.2 in ±2 mrad, ≤ 5–8 up to ±50 mrad, ≪ M ≈ 300. The
   fringe suppression by the angle dither is real (V_θ ≈ 5e-3), but it does not follow (K N_λ)^(−1/2): it comes from phase
   averaging, not from independent patterns. Risk R1.4 is confirmed in its speckle-counting form.
8. **Chaotic gain at realistic R**: 2.06× (R 95 %, 0.3 dB/cm) and 1.93× (0.5 dB/cm) with both dithers — at or below the
   lower edge of M5's 2–5×; 3.6–4.5× with the angle dither alone. The decision rule of T1.1 ("if the chaotic gain at
   R = 95 % drops below the 2–5× of M5, regular cells are adopted") is triggered at 0.5 dB/cm.
9. **Inclusion strength (T1.1)**: Δn ≈ 0.001–0.002 makes the cell chaotic (full coverage, positive FTLE) but gives no fringe
   gain (chaotic worse than regular); Δn_eff ≥ 0.005 (≥ 3.4 nm step in 500 nm SiNx, ≈ 7 nm in 1000 nm) is needed, and
   small discs (25 µm) are best.
10. **NH3 "≈ 50 ppb at 1 s" and "≈ 80 ppb at 2 cm"** assume shot noise only at the ray transmission (no detector noise, no
    coupling loss). With the T1.3 budget (NEP 14 pW/√Hz, 14 dB, 1 mW) and set-B fringe: 270–480 ppb (simulated Γ⟨L⟩
    3.2–4.7 cm), 590–690 ppb at 2 cm, layout spread over 100 layouts in Task 8.
11. **"0.4 ppb at 3.3 µm (0.6 ppb at 2 cm)"**: at set B the 3.3 µm limits are 10–19 ppb (CH4) and 12–22 ppb (C2H6) at 1 s and
    2 cm (14 dB, 1 mW), 6.8–13 / 7.7–15 ppb at 15 s. The line ratio does not transfer at equal noise because the mid-IR
    fringe (N_λ ≈ 1.1 at ±0.55 GHz) is 8–9× the near-IR one.
12. **"150–230× stronger lines"**: 167× (CH4 3.31 µm) and 147× (C2H6 Q-branch) relative to NH3 6528.77 cm⁻¹ (HITRAN, breath,
    35 °C). 230–250× is CH4 3.31 µm vs CH4 1.64 µm.
13. **T1.3 "detector may use 3e-7 of the 6e-7"**: the arithmetic of the detector times is right (0.7 s at 14 dB, 11 s at
    20 dB, 12 s at 25 dB / 3 mW), but the remaining 5e-7 for the fringe is not available at 3.3 µm: the simulated fringe
    residual at 1 s is 0.74–1.4e-6 (R 99.5–95 %); M7 is fringe-limited.
14. **Ethane stretch 2 ppb**: δ = 1.34e-7 at 2 cm; the detector alone needs 3.5 s (14 dB, 1 mW), but with the simulated
    fringe 2 ppb is not reached within 100 s for any R (fringe floor 11–21 ppb at 2 cm with ±20 pm at 3.3 µm).
15. **Transmission.** The simulated circular cells pass only 5–17 % of the injected rays to the output port (R 95–99.5 %),
    i.e. 8–13 dB from the cell geometry and mirrors alone, before couplers, fan-in and fabrication excess; the 10–14 dB
    aim of T1.3 leaves little for the couplers. The Herriott cells pass 26–39 %.
16. **Fig. 3 caption** "curves showing medians over 30 simulations" — the published Fig. 3b used the first 21 layouts (the
    30-layout median differs by < 2 %).

## 11. Paste-ready text

### Sec. 3.2, "Preliminary results" (replacement)

```latex
\textbf{Preliminary results.} We ray-traced circular cells of 4 mm diameter and Herriott cells of 10$\times$1.5 mm at the
loss-budget values of T1.3: a 500 nm SiN$_x$ membrane (TM$_0$, $n_\mathrm{eff}=1.54$, $\Gamma=0.45$ at 1.53\,\textmu{}m),
mirror reflectivity $R=95$--99.5\,\%, membrane loss 0.3--0.5 dB/cm and parasitic power fraction $\varepsilon=10^{-6}$
(medians and interquartile ranges over 30 inclusion layouts). With angle ($\pm2$ mrad) and wavelength ($\pm20$ pm) dither
the chaotic circular cell reaches a residual fringe of $1.8\times10^{-7}$ (IQR $0.5$--$4\times10^{-7}$) at $R=95\,\%$ and
$8\times10^{-8}$ at $R=99.5\,\%$, below the target of $5\times10^{-7}$ for 80--90\,\% of the layouts (Fig.~2f). The chaotic
cell is better than the regular one by a factor of about 2 with both dithers and 4 with the angle dither alone; the chaotic
Herriott cell misses the target at every $R$ ($5\times10^{-7}$ at $R=99.5\,\%$). The residual fringe scales as $\sqrt{\varepsilon}$, so the target holds up
to $\varepsilon\approx8\times10^{-6}$ at $R=95\,\%$ ($4\times10^{-5}$ at $R=99.5\,\%$); $\varepsilon$ is measured in T2.3.
The angle dither suppresses the fringe by averaging the interference phase, but it yields only about two independent speckle
patterns within $\pm2$ mrad, so its contribution must be verified on the first cells (T2.3, R1.4). Inclusions need
$\Delta n_\mathrm{eff}\geq0.005$ (etch steps $\geq3.5$ nm) to give a chaotic advantage, and the mean number of passes of the
detected light is the best predictor of a good layout (T1.1). At 3.3\,\textmu{}m, where $\pm20$ pm is only $\pm0.55$ GHz, the
same cell (1000 nm membrane) reaches $1.1$--$1.4\times10^{-6}$ ($R=97$--95\,\%); a wavelength dither of $\pm2.2$ GHz ($\pm80$ pm) brings it to
$2.4$--$3.3\times10^{-7}$ ($R=97$--95\,\%), which T1.2 adopts for the mid-infrared. With the T1.3 noise budget (14 dB cell
loss, 1 mW, uncooled detector) the detection limit at $\Gamma\langle L\rangle=2$ cm and 1 s is about 0.6 ppm of NH$_3$
(1.53\,\textmu{}m) and 10--19 ppb of CH$_4$ and 12--22 ppb of C$_2$H$_6$ at 3.3\,\textmu{}m ($R=99.5$--95\,\%), and 7--13 ppb
of CH$_4$ at 15 s: the 10 ppb milestone (M7) is met within 15 s for CH$_4$ at $R\geq97\,\%$ and for C$_2$H$_6$ only at
$R\approx99.5\,\%$ (or with the wider mid-infrared dither), and it is set by the fringe, not the detector.
```

### Fig. 2 caption (replacement)

```latex
\textbf{Figure 2.} Ray tracing simulation (Sec.~3.2) of regular and chaotic cells (RO1, RO2) on a 500 nm SiN$_x$ membrane
(TM$_0$, $n_\mathrm{eff}=1.54$, $\lambda=1.53$\,\textmu{}m). (a,b) Segmented circular multipass cells, regular (R-CMPC) and
chaotic (C-CMPC). (c,d) Herriott cells, regular (R-Herriott) and chaotic (C-Herriott, 16 index inclusions,
$\Delta n_\mathrm{eff}=0.01$, orange dots). (e) Poincar\'e sections (angle of incidence $\chi$, position along the wall $s$).
(f) Static residual fringe amplitude $\Delta P/P=2\sqrt{\varepsilon}\,V/\sqrt{M}$ ($\varepsilon=10^{-6}$) at mirror
reflectivity $R=95\,\%$ and membrane loss 0.3 dB/cm, with no dither, spectral dither ($\pm20$ pm), angle dither ($\pm2$ mrad)
and both. Target $5\times10^{-7}$ (dashed). Chaotic bars: medians over 30 inclusion layouts, whiskers the interquartile range;
regular cells (no inclusions) give single values.
```

### Fig. 3 caption (replacement)

```latex
\textbf{Figure 3.} Simulated spectroscopy in breath background (RO1, RO3; 35\,$^\circ$C, 1 atm, 5.5\,\% H$_2$O, 4.5\,\% CO$_2$).
(a) Absorption coefficient of 1 ppm of NH$_3$ (1.53\,\textmu{}m), CH$_4$ (1.64 and 3.31\,\textmu{}m) and C$_2$H$_6$
(3.34\,\textmu{}m) in a breath background (grey), retrieved after averaging for 1 s without dither (grey line), with angle
dither (green), wavelength dither (pink) and both (black), for the chaotic CMPC with 14 dB cell loss, 1 mW and an uncooled
detector (NEP 14 pW/$\sqrt{\mathrm{Hz}}$). (b) Noise-equivalent absorbance at the NH$_3$ line versus averaging time $T$ (500 nm
SiN$_x$, $R=97\,\%$, 0.3 dB/cm, $\Gamma\langle L\rangle=3.6$ cm), medians and interquartile bands over 30 layouts, with the
detector noise at 14, 20 and 25 dB cell loss; at 1 s the cell is detector-limited (about 0.4 ppm NH$_3$).
```

### T1.3 loss-budget text (replacement of the sentences after "Detector noise sets the averaging time")

```latex
Detector noise sets the averaging time $T$: $\delta=\sqrt{\mathrm{NEP}^2+2h\nu P_d}/(P_d\sqrt{2T})$, with $P_d$ the power at
the detector. 10 ppb at 2 cm gas-equivalent path needs $\delta\approx7\times10^{-7}$ (CH$_4$ $7.6\times10^{-7}$, C$_2$H$_6$
$6.7\times10^{-7}$). The detector reaches $3\times10^{-7}$ in 0.7 s at 14 dB and 11 s at 20 dB with 1 mW, and in 12 s at 25 dB
with 3 mW. The simulated residual fringe at 3.3\,\textmu{}m with $\pm0.55$ GHz and $\pm2$ mrad dither is
$0.7$--$1.4\times10^{-6}$ at 1 s ($R=99.5$--95\,\%), so the fringe, not the detector, sets the limit; a wavelength
dither of $\pm2.2$ GHz lowers it to $1.4$--$3.3\times10^{-7}$. We estimate the cell loss from the idler to the detector per element ($\Gamma\approx0.45$, ${\sim}$11 mirror bounces
for a 2 cm gas-equivalent path in the circular cell): couplers in and out 2--4 dB each, DBR mirrors 1.5--2.5 dB ($R=97$--95\,\%), membrane
1.3--2.2 dB (0.3--0.5 dB/cm), and fan-in, fabrication excess and optics 3--6 dB, ${\approx}$10--20 dB in total. The simulated circular
cell delivers only 5--17\,\% of the injected light to its output port, so the port geometry has to be optimised alongside the mirrors.
We take $T=1$ s as the baseline, accept up to 15 s, and state $T$ with every detection limit.
```

### T1.2 (amendment)

```latex
The angle dither averages the phase of the interference between the cell output and stray paths. In simulations it yields
only about two independent speckle patterns within $\pm2$ mrad (decorrelation angle 1.2--1.5 mrad, at most 5--8 up to
$\pm50$ mrad) rather than $K\lesssim M\approx300$, so the fringe suppression cannot be counted as $(KN_\lambda)^{-1/2}$; it is
simulated directly and measured on the first cells (T2.3). In the mid-infrared the wavelength dither is $\pm2.2$ GHz
($\pm80$ pm at 3.3\,\textmu{}m, $\pm18$ pm on the telecom signal laser at 1.57\,\textmu{}m).
```
