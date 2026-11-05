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
| M5: chaotic gain ≥ 2–5× (both dithers), R = 95 % | 3.1× | 2.06× (0.3 dB/cm), 1.93× (0.5 dB/cm) | marginal: at/below the lower edge |
| ε thresholds (5e-7 up to ε ≈ 2e-4; 8e-8 up to 6e-6) | 2.3e-4 / 5.9e-6 | near-IR R 95 %: 7.6e-6 / 1.9e-7; mid-IR: < 1e-6 | FAIL (25–30× tighter) |
| angle dither gives K ≈ many independent patterns (K ≲ M ≈ 300) | – | K_eff ≈ 1.7–2.3 in ±2 mrad (see Task 5) | FAIL (risk R1.4 materialises) |
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

<<T1>>

The Herriott "≈ 80×" of Sec. 3.2 is not the median (55×); 80× lies inside the layout spread of the gain (20–166×). The
NH3 numbers use the published shot-noise-only model.

## 3. Task 2: realistic Fig. 2f (set B)

`results/t2/`, `figs/fig2f_setB_0.3dB.pdf`, `figs/fig2f_setB_0.5dB.pdf`, `figs/fig2ef_setB_R{0.95,0.97,0.995}.pdf`.

<<T2>>

Notes. (i) Near-IR C-CMPC: the M3 target holds at every R and loss with ±20 pm (median 0.8–2.1e-7, 77–90 % of layouts),
but at ±0.55 GHz (the 3.3 µm excursion) it does not (3.9–8.9e-7). (ii) The chaotic gain with both dithers is 1.9–2.5×
(circle) at all R; with the angle dither alone it is 3.6–4.5×. (iii) Herriott: the regular cell's re-aimed launch at
0.5 dB/cm picks a short orbit (3 bounces, 4.2 cm), which makes its fringe low and the gain small; at 0.3 dB/cm the gain
is 34× (R 95 %) to 115× (99.5 %), but the chaotic Herriott misses 5e-7 at R ≤ 97 % (1.1–1.8e-6). (iv) Mid-IR C-CMPC:
1.1–1.6e-6, FAIL at all R; N_λ ≈ 1.1 (the ±0.55 GHz dither hardly decorrelates the speckle).

## 4. Task 3: ε sweep

`results/t3/`, `figs/eps_sweep.pdf`. ΔP/P ∝ √ε exactly (log-log slope 0.500): in the model ε is only the stray amplitude,
while V and M are properties of the exit field. V and M do depend on R, loss, n_eff, wavelength and dither, so the
thresholds move with them; the static offset ε itself is not a fringe (only its drift would matter).

<<T3>>

The proposal's thresholds (5e-7 up to ε ≈ 2e-4, 8e-8 up to 6e-6) follow from set A (2.3e-4, 5.9e-6). At set B they are
25–30× tighter in the near-IR (R 95 %, 0.3 dB/cm: 7.6e-6 and 1.9e-7 for the median layout) and the mid-IR cell already
fails 5e-7 at ε = 1e-6 (ε_max 1.2–4.6e-7).

## 5. Task 4: path, bounces, N_λ for the 2 cm milestone

`results/t4/`. Γ = 0.45 → L_phys = 2 cm / 0.45 = 4.44 cm, i.e. 11–12 bounces in the circle (chord 0.37–0.40 cm) or ≈ 5 in
the Herriott cell (chord 0.8–1.0 cm); mirror + membrane loss 1.3–4.7 dB for R 99.5–95 %, 0.3–0.5 dB/cm.

<<T4>>

The old "∼10 cm physical, N_λ ≈ 2–2.7" is reproduced by an exponential p(L) with ⟨L⟩ = 10 cm at n = 2 (N_λ = 2.25 at
±0.55 GHz). At the milestone path (4.44 cm) N_λ falls to 1.29 (n_eff) / 1.51 (n_g); the simulated chaotic CMPC at 3.3 µm
gives 1.2–1.5 (n_eff) / 1.3–1.9 (n_g). With ±20 pm at 1.53 µm (±2.56 GHz) N_λ = 2.7–4.7 (n_eff) / 3.5–6.3 (n_g).

Detection-limit scaling at Γ⟨L⟩ = 2 cm (set-B noise; 14 dB, 1 mW, 1 s, both dithers): NH3 590–690 ppb (proposal: 80 ppb);
CH4 3.31 µm 10–19 ppb, C2H6 12–22 ppb (proposal: 0.6 ppb). The 150–230× line ratio does not carry over at equal noise: the
mid-IR fringe is 5–9× larger than the near-IR one (N_λ ≈ 1.1 at ±0.55 GHz) and dominates the 3.3 µm noise.

## 6. Task 5: angle-dither validity

<<T5>>

## 7. Task 6: inclusions

<<T6>>

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

<<T7>>

## 9. Task 8: layout spread

<<T8>>

## 10. Discrepancies with the proposal (v32)

<<DISC>>

## 11. Paste-ready text

<<TEXT>>
