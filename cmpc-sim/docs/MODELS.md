# Models, equations and assumptions

All quantities SI unless a name ends in `_um`, `_nm`, `_cm`. Code references are `module.function`.
Version of this document: v0.4 (see `CHANGELOG.md`).

## 0. Conventions
* Membrane in the *x–y* plane, propagation in-plane; *z* is vertical. Cladding = air (`n_clad=1`) on both sides unless set.
* χ = angle of incidence of a ray on a boundary facet, measured from the outward normal. `sinchi` is **signed**
  (`d × n`): in a circle it is conserved *with sign* (angular momentum).
* Slab **TE** = E in the membrane plane; slab **TM** = dominant E_z. For a vertical trench (plane of incidence = membrane
  plane): **TE → p-polarised, TM → s-polarised** (`cmpc_sim.LATERAL_POL`).
* Γ = ratio of modal power absorption to the bulk gas absorption coefficient: α_modal = Γ α_gas.

## 1. Materials (`cmpc_sim.n_material`)
Si, Si₃N₄ ("SiNx", stoichiometric), sapphire from `../math-engines` `engines.materials` (Sellmeier; Si fit valid above ~1.2 µm,
real part only). Added here: **Ge** — Barnes–Piltch Sellmeier, n² = 9.28156 + 6.72880 λ²/(λ²−0.44105) + 0.21307 λ²/(λ²−3870.1)
(λ in µm, 2–14 µm; *coefficients entered from memory — verify before design use*); **Al₂O₃** — constant n = 1.65 (assumed;
the engines only have crystalline sapphire). `n_const` overrides any material.

## 2. Membrane modes (`membrane_mode`, `evanescent_volume`, `field_profile`)
* n_eff from `engines.slab_waveguide.neff_three_layer` (symmetric 3-layer slab, TE/TM, any order). Group index
  n_g = n_eff − λ dn_eff/dλ by central finite difference (h = 10⁻³ λ) including material dispersion.
* Core transverse wavenumber κ = k√(n_f²−n_eff²), decay constant γ = k√(n_eff²−n_c²), β = k n_eff.
  Field profile f(x) = cos κx (even) or sin κx (odd) in the core, matched exponential outside.
* **TE:** Γ = (n_c/n_eff) · ∫_clad f² / ∫ f².
  **TM** (f = H_y): Γ = n_c ∫_clad (f′² + β² f²)/n_c⁴ / ( kβ ∫ f²/n² ).
  Both follow from (dissipation in the gas)/(Poynting flux); validated against numerical quadrature (<0.3 %), Γ→1 as d→0.
  TM can exceed 1 near cut-off (field discontinuity n²) but the mode is then weakly confined and the mirrors fail.
* Penetration depth (1/e intensity, per side) z_p = 1/(2γ). Evanescent gas volume V_ev = sides · A · z_p (default 2 faces).
* Surface-field weight `E_edge2` (|E|² on the core side of the surface per unit power, common factors dropped): crude proxy for
  roughness scattering. `roughness_scaled_alpha`: α ∝ E_edge2. Scripts additionally scale by ((n²−1)/(n_ref²−1))² and
  (k/k_ref)² when changing material/wavelength, **assuming the same rms roughness**; reference = 0.1 dB/cm for Si 220 nm TE at 1.55 µm
  (assumed, not measured).

## 3. In-plane mirrors
### 3.1 Transfer matrix (`stack_R_oblique`)
Characteristic-matrix method with tilted admittances q = n cosθ (s) or n/cosθ (p), cosθ = √(1−(n_in sinχ/n)²) with the branch
Im ≥ 0, so total and frustrated total internal reflection are included. `sin_in` is clamped to 1−10⁻⁹. Complex index allowed.
Equals `engines.bragg_grating.stack_reflectance` at normal incidence (tested to 10⁻¹⁵).
### 3.2 DBR (`DBR`, `double_resonant_orders`, `cascaded_R`)
Period = [air gap, membrane tooth]; thicknesses are odd multiples of λ/4 (gap) and λ/(4 n_eff) (tooth); tooth index = slab n_eff of
the *same* mode; `bounce_loss` multiplies R (lumps slot radiation, roughness, finite etch depth: **assumed 3×10⁻⁴**).
For a TIR-capable facet sin χ > 1/n_eff the first gap already reflects; the DBR only matters below that angle.
### 3.3 Brewster window and the three-index mirror (`mirror_design`)
At the Brewster angle every interface of a stack of **two** materials has r_p = 0, so the stack is transparent *for any thicknesses*
(tested: R < 10⁻¹² for random stacks). A chaotic cell visits all angles, so TE (p-pol) loses ⟨1−R⟩ ≈ 7–10 % per bounce. A **third index**
(partly etched, thinned membrane region with its own n_eff) breaks the coincidence. `design` minimises ⟨1−R⟩ + 0.1·max(1−R) over
sin χ ∈ [0.002, 0.998] for the period [air g | thinned c | tooth a] × N with `scipy.optimize.differential_evolution`
(bounds 0.25–3 µm, seed fixed). `tolerance` = Monte-Carlo with Gaussian layer-thickness errors. Stored result
(`data/mirror_designs.json["130_16"]`, Ge 300 nm TE at 5.263 µm): g = 250 nm, c = 720 nm (Ge 130 nm, n_eff 1.49), a = 725 nm, N = 16,
⟨1−R⟩ = 0.0083 (two-material N = 20: 0.102); 20 nm errors → median ≈ 0.014. 1D effective-index treatment of a thickness step is rough.

## 4. Cells and ray tracing
* `SegmentedCell(r, n_facets, tilt_rms, curvature, offset_rms, seed)`: regular N-gon of circumradius r, each facet an in-plane mirror.
  Optional designed perturbations: random facet **tilt** (rad, about the facet midpoint), radial **offset**, facet **curvature** 1/ρ
  (>0 convex into the cell = dispersing → hyperbolic dynamics). Facets are extended by 15 % so rays never leak through corner gaps.
  Boundary coordinate *s* runs along the facets (2h each). The unperturbed polygon is pseudo-integrable (zero Lyapunov exponent).
  `Stadium` (legacy, a = 0 is a circle) is kept for comparison.
* `trace_rays(cell, port_w, n_rays, n_bounce, theta0, theta_c, s_in, s_out, extra_ports, launch_w, launch_off, max_path, ...)`:
  specular reflection, normals and directions renormalised each bounce. Launch: positions uniform in the input window
  (`launch_w` narrows the footprint, i.e. a divergent beam), angle uniform in θ_c ± θ0 from the inward normal. **A single-mode port of width w launches
  θ0 ≈ λ/(n_eff w)**, not a wide cone. Rays end at the input window (lost), the output window (detected), any `extra_ports`
  (other beams' ports, lost), at `max_path`, or after `n_bounce`. The table stores per-hit chord and signed sinχ, exit index/port and a
  64-bit hash of the facet sequence (**itinerary** = path identity).
* **Re-weighting** (`evaluate`): for detected rays W = exp(Σ ln R(|sinχ_j|) − α_bg L) (R interpolated on 513 points). Outputs
  T_det = ΣW/N_launched, ⟨L⟩ = ΣWL/ΣW, Γ⟨L⟩ (gas-equivalent path), S1 = Γ T ⟨L⟩ (signal yield), p(L), bounce count, and a
  weighted standard error. One geometric table serves any mirror, loss, Γ, wavelength with the same geometry/launch.
* `mean_field_estimate`: ergodic geometric-series cross-check (agrees with the Monte-Carlo within ~7 % in the stadium test).
* `occupancy_map`, `poincare` (uniform initial conditions in (s, sinχ), the invariant measure), `lyapunov`
  (finite-time Benettin-type pair separation; **flat tilted facets give ≈ 0 although they mix; only curved facets are formally chaotic**).
* Ray model = no diffraction; use `rayleigh_check` / launch spread to judge validity.

## 5. Path-resolved coherence model (`coherence_model`)
* **Mode cell:** output exit directions are binned in sin χ with width Δ = λ/(n_eff w) (one mode of the port of width w).
* **Path:** distinct (mode cell, itinerary). Path j has power P_j (sum of ray weights) and length L_j (power-weighted mean).
  M_spat = (ΣP_c)²/ΣP_c² is the effective number of independently summed output modes; N_p = (ΣP)²/ΣP² the paths per mode.
* **Intensity** in cell c: I_c(ν) = |Σ_j √P_j e^{iφ_j(ν)}|², φ_j = 2π ν n_g L_j/c + const_j (random-phase statistics).
* **Contrast:** one cell, monochromatic: C² = 1 − ΣP²/(ΣP)² (one path → 0). With a Lorentzian source (FWHM γ) every pair term is scaled:
  Var_c = Σ_{j≠k} P_j P_k exp(−2π γ n_g|L_j−L_k|/c) · sinc²(S n_g|L_j−L_k|/c) (second factor = uniform sweep/phase-dither of equivalent span S).
  A detector summing all cells: C² = ΣVar_c/(ΣP_c)² (`contrast(spatial=True)`); one typical cell: ΣVar_c/ΣP_c² (`spatial=False`).
  Scaling law (checked): C² ≈ (1−1/N_p)/(K M).
* **K beams:** `merge_incoherent` (mutually incoherent *or* dithered/time-multiplexed: cells stay separate, variances add) vs
  `merge_coherent` (one laser, simultaneous: cells with the same exit direction are united and interfere). Result: **coherent merging gives no gain**;
  a delay about the path-length spread (~1 m per beam in the long-path design) is needed to approach the incoherent limit.
  **Independence requires distinct input modes**; two incoherent beams in the *same* mode give identical speckle patterns. The model cannot detect this
  (rays in a chaotic cell mostly have distinct itineraries even if they belong to the same classical beam).
* **Autocovariance** g(Δν) = Σ_{j≠k}P_jP_k cos(2πΔν n_g ΔL/c)/Σ_{j≠k}P_jP_k; correlation width = FWHM of g.
* **Drift:** Δn_eff = (dn_eff/dT) ΔT shifts all phases by k Δn_eff L, equivalent to Δν_eq = ν (dn_eff/dT) ΔT/n_g, so the pattern decorrelates as g(Δν_eq).
  `speckle_noise_A`: σ_A = C(γ, S) · √(2(1−ρ(ΔT))) · √min(1, FWHM_g/FWHM_line) (the last factor: a fitted line averages ~FWHM_line/FWHM_g grains).
  dn_eff/dT = 1.5×10⁻⁴ K⁻¹ (Si) and 3×10⁻⁴ K⁻¹ (Ge) are **assumed**.
* `simulate_spectrum`: one random-phase realisation of I(ν) (optionally with a gas line through exp(−Γ α L/2) per path).
  Reflection phases and diffraction are neglected. This is a *prediction* model, not a wave simulation.

## 6. Gas absorption (`gas_spectra`)
Voigt profile (`scipy.special.wofz`), Doppler HWHM = 3.5812×10⁻⁷ ν √(T/M[amu]), Lorentz HWHM = γ_air (p/p₀)(296/T)^n (self-broadening ignored).
Line strength S(T) = S₀ [Q(296)/Q(T)] exp(−c₂E″(1/T−1/296)) [1−e^{−c₂ν/T}]/[1−e^{−c₂ν/296}], Q from HITRAN TIPS-2021 via HAPI when installed
(power-law fallback otherwise). Mixtures: α(ν) = Σ_s N x_s Σ_l S φ. Through the cell: T(ν) = Σ_i W_i exp(−Γ α(ν) L_i)/ΣW_i (path-length distribution, not exp(−αΓ⟨L⟩)).
`shot_noise_A`: √(2eΔf/(R P)) (idealised; mid-IR detectors are usually not shot-limited).
**Data status: `ILLUSTRATIVE_LINES` are hand-entered order-of-magnitude placeholders** (CH₄, NH₃, CO₂, H₂O near 1.5–1.7 µm; NO, CO, N₂O, H₂O near 4.5–5.3 µm).
`scripts/fetch_hitran.py nir|mir` writes `data/hitran_lines*.json`, which `load_lines()` then uses automatically.

## 7. Where each result comes from
| Result | Script → functions |
|---|---|
| Γ, n_eff, V_ev vs thickness | `run_figures.py` fig 1–2, `run_midir.py` fig 1 → `membrane_mode`, `evanescent_volume` |
| TE Brewster hole, TM mirror | `run_figures.py` fig 3, `make_prelim_figs.py` fig 3 → `stack_R_oblique`, `DBR` |
| Γ⟨L⟩, 50 cm milestone | `make_prelim_figs.py` fig 3, `run_figures.py` fig 4/6 → `trace_rays`, `evaluate` |
| Chaos vs regular, Poincaré | `run_figures.py` fig 5, `make_prelim_figs2.py` fig 1 → `SegmentedCell`, `poincare`, `lyapunov` |
| Speckle vs perturbation, K beams | `chaos_scan.py`, `run_chaos_figs.py`, `multibeam_nir.py`, `make_prelim_figs2.py` → `coherence_model` |
| Single port vs K ports | `single_port_scan*.py`, `single_port_figs.py` → `trace_rays(extra_ports, launch_w, theta_c)` |
| Three-index mirror | `design_mirror.py`, `chaos_scan.py` → `mirror_design` |
| Detection-limit budget | `budget_scan.py`, `run_chaos_figs.py` fig D, `make_prelim_figs2.py` fig 4 → `speckle_noise_A` |
| Mid-IR (Ge, 5.26 µm) | `run_midir.py` |

## 8. Validation status
Tested (`tests/`, `cmpc_sim.selftest`): TMM vs engine at normal incidence; s = p at normal incidence; Brewster transparency; Γ vs quadrature and thin limit;
circle conserves signed sinχ; polygon geometry; MC vs mean-field ⟨L⟩; contrast formulas (single path, two paths, 1/√K); Voigt normalisation; weak-absorption limit.
**Not validated:** coherence model vs wave simulation or experiment; mirror model vs 3D FDTD; any absolute loss; any gas-line parameter.

## 9. Assumed inputs that move the results
| Input | Value used | Where |
|---|---|---|
| membrane loss | 0.1 dB/cm (Si 220 nm TE ref.), 0.04 dB/cm for the "long-path" design, 0.013–0.03 dB/cm mid-IR | scripts' `ALPHA`/`CFG` |
| extra per-bounce mirror loss | 3×10⁻⁴ | `bounce_loss` |
| dn_eff/dT | 1.5×10⁻⁴ (Si), 3×10⁻⁴ (Ge) K⁻¹ | `speckle_noise_A` callers |
| Al₂O₃ index | 1.65 | `MATERIALS` |
| Ge Sellmeier | from memory | `n_material` |
| ports | ideal windows, no taper/coupler loss, no crosstalk, no fan-in/switch loss | tracer |
| gas lines | illustrative | `gas_spectra.ILLUSTRATIVE_LINES` |

## 10. Correction history (important)
1. v0.1 paired the TE mode's Γ with an s-polarised mirror. For a vertical trench TE is p-polarised (Brewster window) and TM s-polarised. Fixed in v0.2; all v0.1 path numbers are void.
2. Normal vectors on curved facets drifted from unit length over hundreds of bounces (|sinχ| > 1); now normalised and tested.
3. Lyapunov estimator: a fixed window mistook linear growth of regular billiards for exponential growth; now fits before saturation and falls back to a late window.
4. Launch model: a ±20° cone was replaced by λ/(n_eff w) single-mode launch; contrast results changed substantially.
5. Beams from one laser were first assumed incoherent beyond the coherence length; the pair analysis shows they must be delayed by about the path-length spread (or dithered).
