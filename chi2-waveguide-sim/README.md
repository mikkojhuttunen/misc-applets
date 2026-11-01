# chi2-waveguide-sim

χ² frequency conversion (SHG now; cascaded THG and DFG/OPA next) in TFLN and TFLT waveguides with ordered, random, quasi-random, chirped/adiabatic and hybrid poling: batched solvers, ensemble statistics and inverse design for a centre wavelength and FWHM bandwidth.

The coupled-wave physics lives in `math-engines/engines/cwe_shg` (spec'd, tested, with test vectors). This package adds poling generators, a batched undepleted solver (numpy or JAX), closed-form and Monte-Carlo ensemble statistics, and synthesis. It must sit inside a misc-applets checkout, next to `math-engines/` (or set `MISC_APPLETS`). SI units.

```
cd chi2-waveguide-sim
pip install -e ".[test,jax,figures]"
python -m pytest -q
python examples/band_designs.py --backend jax --fig figs/band_designs.png
```

## Layout

```
chi2sim/poling.py      Poling (boundaries + g per domain); ordered, jittered, random_domains (LengthDistribution:
                       gaussian, uniform, gamma, exponential, lognormal; l_min; AR(1) neighbour correlation),
                       quasi_random (PRBS-15, Fibonacci, Thue-Morse, Rudin-Shapiro, Barker-13), pwm, chirped
                       (linear, tanh), window, duty_from_amplitude, fabricate (grid snap + l_min), FAB_PRESETS
chi2sim/fast.py        G(Δk) = ∫ g e^{iΔk z} dz as a sum over walls, batched over structures and Δk, with loss;
                       numpy (chunked) or JAX (jit, float64, jax_spectrum differentiable in the wall positions)
chi2sim/ensemble.py    monte_carlo; closed forms: renewal_mean (i.i.d. lengths), jitter_mean (Gaussian wall jitter)
chi2sim/metrics.py     fwhm, band_stats (mean, min, ripple dB, contrast), ensemble_stats (mean, σ/mean, quantiles)
chi2sim/materials.py   LN (congruent, MgO) and LT index tensors per cut, type-0 d coefficients (d33, d22) with Miller scaling
chi2sim/waveguide.py   Ridge (film, etch, top width, sidewall angle, cladding, cut, material, optional deposited strip
                       load = sinx | si3n4 | sic | sio2 | al2o3 or load_index); with_thickness; eim; 2D semi-vectorial
                       FD solver (port of waveguide-core.js; sparse shift-invert, sub-pixel ε on a graded grid);
                       a_eff_shg; SHGModel (n_p, n_SH, n_g, A_eff tables, splined; Δk(λ) callable, κ(λ), period,
                       save/load/cached); width_scan (period, Δk and A_eff vs width, dΔk/dw);
                       OPAModel (fixed pump, signal table, idler by energy conservation, three-mode A_eff,
                       Miller-scaled d, gain_spectrum via cwe_dfg, mid-IR absorption flags);
                       THGModel (modes at λ, λ/2 and λ/3 with a selectable 3ω order, Δk_a, Δk_b, both overlaps,
                       Miller-scaled d per process); mode_of_order
chi2sim/thg.py         cascaded THG designs on THGModel: two_section, superlattice, blend (SHG weight sweeping along z),
                       commensurate higher-order single period; thg_spectrum, optimisers, references,
                       thg_devices (thickness spread), double_matching_shift
chi2sim/optimize.py    Target (mean | flat | yield | min | geo over devices × wavelengths), device_nodes (thickness
                       quadrature), band_target, robust_target; envelope IFTA + PWM binarisation; WallParams and
                       gradient_optimise (JAX, Adam, l_min and length kept by construction); evolve_chirp (DE, optional
                       wall-jitter variable with the exact jitter mean); THG: thg_mask, THGTarget, evolve_thg
chi2sim/wafer.py       ThicknessSpread (±5 nm uniform default, film and deposited layer independent), ThicknessFamily
                       (mode models on a thickness stencil, Lagrange interpolation, analytic sensitivity and grating
                       tolerance), shg_devices / opa_devices (one mask on many devices), yield_fraction
chi2sim/dispersion.py  simple Δk(λ) models: LinearIndexSHG, MaterialSHG (bulk Sellmeier)
chi2sim/synthesis.py   band_to_dk, lambda_grid, efficiency_ceiling, chirp_from_spectrum, poling_from_chirp,
                       parasitic_orders, random_band_design, design(mode="chirp" | "random" | "hybrid") -> Design
examples/band_designs.py   ordered baseline vs qpm_shg; chirp / random / hybrid for 10, 50, 100 nm FWHM vs the ceiling
examples/ridge_shg.py      TFLN vs TFLT x-cut ridges from the FD solver: period, A_eff, η, width tolerance, band designs
examples/optimize_designs.py  flat-top bands (chirp -> IFTA -> gradient), robust SHG (TFLN, TFLT) and THG over ±5 nm film
examples/thg_designs.py    cascaded THG 1550 -> 517 nm: designs, spectra, TE10 option, depletion, thickness spread
examples/wafer_spread.py   ±5 nm thickness spread over 200 devices: SHG yield of ordered vs chirped masks (TFLN, TFLT,
                           SiNx-loaded TFLN), OPA with 775 / 532 / 1064 nm pumps
tests/                 generators, fast solver vs cwe_shg and JAX gradients, ensemble statistics, synthesis
```

## Statistical design

- **Random (i.i.d. domain lengths).** The ensemble mean is closed form: ⟨|G|²⟩ ≈ (4N/q²) Re[(1−φ)/(1+φ)] with φ(q) = E e^{iqℓ}. Mean length ≈ π/K0 centres the band; the length spread sets a Lorentzian-like width (σ² ≈ πB_k/K0³ to start, then bisection on the exact form). Each realization is fully developed speckle: σ/mean → 1, linear in N.
- **Wall jitter on any mask.** Exact mean: the coherent part falls by e^{−q²σ²} and the rest becomes an incoherent background. `design(mode="hybrid", sigma_wall=...)` gives the fabrication-tolerance ensemble of a chirped mask.
- **Ceiling.** Parseval ∫|G|² dq = 2πL. A first-order binary grating puts at most (2/π)² of that into the useful band, so the in-band mean of |G|² is at most (2/π)² 2πL/B_k. Every design reports the fraction it reaches.

Example output (L = 5 mm, 1550 → 775 nm, `LinearIndexSHG`, d = 25 pm/V, A_eff = 1 µm² placeholders; ripple over the central 80 % of the FWHM band):

| FWHM | design | fraction of ceiling | ripple | speckle σ/mean | ⟨η⟩ (%/W) |
|---|---|---|---|---|---|
| 50 nm | chirp, Hann 25 % apodisation | 0.61 | 0.7 dB | (deterministic) | 37 |
| 50 nm | random, Gaussian lengths (ensemble mean) | 0.50 | 2.2 dB | 0.96 | 30 |
| 50 nm | chirp + 100 nm wall jitter (mean) | 0.60 | 0.7 dB | 0.02 | 36 |
| 100 nm | chirp | 0.64 | 0.5 dB | | 19 |
| 100 nm | random | 0.50 | 2.2 dB | 0.98 | 15 |

The random ensemble lands at 0.50 for every bandwidth. That is a Lorentzian line shape holding the whole first-order sideband: half of its area lies outside its FWHM. Each single realization still has 100 % speckle. The apodised chirp gets more power into the band, and the 100 nm wall jitter barely changes it. For a 10 nm band at 5 mm (≈3× the ordered bandwidth) no window gives a flat top (≈2.5 dB). That case needs stage 4 (IFTA / wall-level optimisation) or a longer device.

## Waveguide layer

`W.SHGModel(W.Ridge(...), lambdas)` solves the TE00 (or TM00) modes at λ and λ/2 on one FD grid for every
tabulated λ. It is a Δk(λ) callable, so it drops straight into `synthesis.design`, and `coupling(λ)` gives κ
with the Miller-scaled d and the overlap area. Validation (tests/test_waveguide.py):
- unetched film vs the exact anisotropic slab (math-engines `anisotropic_slab`), after removing the lateral box
  mode: 4e-4 at h = 20 nm, second order in h;
- ridge: FD within 0.01 of the EIM; h = 40/20/10 nm converge (Δn_eff < 5e-4 between the two finest);
- A_eff against the Gaussian hand value 9πab/8; spline tables reproduce direct solves to 2e-6.

x-cut ridge, 600 nm film, 300 nm etch, 1.4 µm top width, 70° walls, air clad (`examples/ridge_shg.py`):

| | period | A_eff | d33(1550) | first-order η | ordered FWHM (4 mm) | width tolerance |
|---|---|---|---|---|---|---|
| TFLN | 4.04 µm | 0.90 µm² | 22.7 pm/V | 3700 %/(W cm²) | 6.1 nm | ±2.4 nm |
| TFLT | 4.11 µm | 0.91 µm² | 12.5 pm/V | 1150 %/(W cm²) | 7.9 nm | ±2.4 nm |

For comparison, Wang et al., Optica 5, 1438 (2018) report a 4 µm first-order period and 2600 %/(W cm²)
measured in x-cut TFLN; their exact film and etch values were not checked here, so this is a plausibility check,
not a calibrated match. The d33 values are Shoji et al. (1997) at 1064 nm, Miller-scaled. d22 values are
approximate. The model is semi-vectorial: TE-TM mode hybridisation, which appears at some widths in x-cut films,
is not captured.

## Wafer thickness spread (devices differ, each device uniform)

Thin LN/LT films and deposited layers (SiNx, SiC, ...) are taken to vary by ±5 nm (uniform, independent) across the
wafer and to be constant within one device. `ThicknessFamily` solves the mode model on a 5-point thickness stencil
(5 × 3 with a deposited strip) and interpolates Δk and κ to every device; interpolation error < 0.2 nm of
phase-matching wavelength with 3 nodes, about 0.01 nm with 5 (tests). The etch depth is kept (timed etch) unless
`etch="slab"`. `examples/wafer_spread.py`, L = 4 mm, 200 devices:

| guide | dλ_pm/dt | ordered-grating thickness tolerance |
|---|---|---|
| TFLN ridge 600/300 nm, 1.4 µm, SHG | -9.7 nm/nm | ±0.31 nm |
| TFLT ridge, same geometry, SHG | -12.5 nm/nm | ±0.32 nm |
| SiNx-loaded TFLN (300 nm LN + 300 nm SiNx strip), SHG | -4.4 (LN), -2.4 (SiNx) nm/nm | ±0.34 nm (LN), ±0.63 nm (SiNx) |
| TFLN ridge, OPA 532 nm pump, signal 1550 nm | -23.8 nm/nm | ±0.32 nm |
| TFLN ridge, OPA 775 nm pump (degenerate) | no first-order shift: the gain band collapses or splits | ±0.31 nm |
| TFLN 900/450 nm, 2.4 µm, OPA 1064 nm pump (idler 3.39 µm) | -10.3 nm/nm | ±0.80 nm |

TFLN SHG at 1550 nm, one mask on all devices (η at 1550 nm in %/W; yield = share of devices with η ≥ ½ nominal):

| design | η nominal | median | worst | yield |
|---|---|---|---|---|
| ordered | 595 | 2.4 | 0.0 | 0.08 |
| chirp 30 nm | 83 | 13 | 0.2 | 0.33 |
| chirp 60 nm | 38 | 36 | 2.3 | 0.71 |
| chirp 100 nm | 25 | 25 | 10.8 | 0.98 |

A ±5 nm film spread moves the SHG phase matching by about ±48 nm, eight times the 6 nm acceptance of a 4 mm
grating, so an ordered grating alone yields 8 %. The options are per-device tuning (temperature, or a ladder of
about 16 periods per die to cover ±48 nm with 6 nm steps) or a broadband mask that trades peak efficiency for
yield (100 nm chirp: 98 % of devices within a factor of 2, at 4 % of the ordered peak). OPA behaves the same way.
The degenerate 775 nm-pumped OPA is the most fragile: its gain band either disappears or splits into
non-degenerate lobes outside 1450-1650 nm. The 1064 nm pump (900 nm film, 2.4 µm) is the least sensitive in
thickness, but its 3.39 µm idler sits in the SiO2 absorption range (flagged, not modelled).

## Cascaded THG (1550 -> 517 nm)

`math-engines/engines/cwe_cascade` solves ω+ω→2ω and ω+2ω→3ω on one mask (exact nested undepleted integral,
depleted RK4). `examples/thg_designs.py`, x-cut TFLN ridge 600/300 nm, 1.4 µm, TE00 at all three wavelengths,
L = 4 mm: Λ_a = 4.039 µm, Λ_b = 3.134 µm (Δk_b/Δk_a = 1.289), A_eff 0.90 / 0.74 µm², d 22.7 / 25.5 pm/V.

| design | η_THG at 1550 nm | FWHM |
|---|---|---|
| both processes matched at full strength (uniform, ideal reference) | 64900 %/W² | |
| two-section (SHG then SFG, split 1/2) | 2670 %/W² (= closed form (2/π)⁴ κ3²κa² L⁴/16) | 1.75 nm |
| superlattice sign(w cos K_a z + (1-w) cos(K_b z + ψ)), optimised | 1760 %/W² | 1.0 nm |
| blend: superlattice with w falling 0.9 -> 0.1 along z, optimised | 3610 %/W² | 1.3 nm |
| single period with orders (7, 9) | ≈ 0 (strength 0.006) | |

The blend wins because 2ω is made early and converted late, while both gratings stay present everywhere.
No small-integer ratio fits 1.289, so a single period is useless here. Moving 3ω to TE10 (fully etched 1.2 µm
ridge) lengthens Λ_b to 6.7 µm but its overlap with the even ω and 2ω modes nearly cancels (A_eff 371 µm²,
SFG efficiency × 1.6e-3). Depletion (blend, full RK4): 0.034 mW at 10 mW, 18 mW at 0.1 W, 0.17 W at 0.3 W,
0.94 W at 1 W (lossless CW idealisation; the undepleted value fails above ~10 mW).

Thickness: a film change re-matches the SHG step at -9.7 nm/nm but the SFG step at only -1.4 nm/nm, so tuning
the fundamental wavelength cannot restore both. Over ±5 nm only about 20 % of devices reach half the nominal
peak anywhere in 1500-1600 nm (100 devices), for both two-section and blend masks. Robust THG needs
a different lever (temperature acts on both steps with yet another ratio, or masks optimised over the
thickness ensemble, milestone 6).

## Optimised designs (milestone 6)

`examples/optimize_designs.py`, x-cut ridges 600/300 nm, 1.4 µm, L = 4 mm. All masks fabricated (10 nm grid,
l_min 300 nm) before evaluation.

**Flat-top SHG bands** (core = central 80 % of the FWHM band; ceiling = (2/π)² 2πL/B_k):

| band | Hann chirp | envelope IFTA | IFTA + wall gradient (flat, α = 3) |
|---|---|---|---|
| 20 nm | 0.51 of ceiling, 2.56 dB | 0.62, 2.01 dB | 0.63, 0.90 dB |
| 40 nm | 0.58, 1.73 dB | 0.73, 1.23 dB | 0.76, 0.85 dB |

IFTA iterates on the first-order envelope e(z) = a e^{-iψ} with |e| ≤ 2/π and realises it by PWM; a GS loop on the
±1 mask itself stalls because a narrow band spans few FFT bins and never moves a wall. Bands below about 3× the
ordered FWHM (6 nm here, L·B_k < 10) cannot be made flat by any mask. Wall gradients need small steps (a few nm per
wall per step).

**Robust SHG at 1550 nm over the ±5 nm film spread** (η at 1550 nm on 200 random devices, %/W):

| | TFLN mean / p10 / worst | TFLT mean / p10 / worst |
|---|---|---|
| ordered (nominal peak 595 / 184) | 47 / 0.15 / 0.00 | 15 / 0.06 / 0.00 |
| 100 nm chirp (milestone 4) | 25 / 20 / 11 | 8.3 / 4.0 / 1.8 |
| DE on the geometric mean over devices (chirp span ≈ 1.5 %, no apodisation) | 38 / 20 / 13 | 12 / 8.4 / 4.4 |
| DE + wall gradient | 41 / 26 / 9.6 | 12 / 8.4 / 2.4 |

The expected efficiency alone is a degenerate objective: by Parseval an ordered grating already reaches the cap of
the mean |G|² over the devices' Δk range (that is why the ordered mean is the highest); what a mask can change is how
evenly the devices share it. The geometric mean punishes dead devices and selects a chirp matched to the Δk spread
of ±5 nm (span ≈ 1.5 % of K). The gradient polish raises mean and p10 but lowers the single worst device (it tunes
to the 21 quadrature nodes); DE alone is the better worst-case design. Wall jitter as a design variable only lowers
the objective (tests): randomness moves power from the band into a broad weak floor. TFLT follows TFLN at about a
third of the efficiency (d33 13.8 vs 25.2 pm/V).

**Robust THG** (η_THG at 1550 nm, 100 devices, %/W²):

| | nominal | mean / p10 / worst |
|---|---|---|
| two-section | 2666 | 180 / 0.00 / 0.00 |
| blend 0.9 -> 0.1 | 3609 | 183 / 0.02 / 0.01 |
| DE geometric mean: blended chirped superlattice, w 0.56 -> 0.44, spans 1.7 % (K_a) / 1.0 % (K_b) | 10 | 8.3 / 2.2 / 0.86 |

The optimiser matches both chirps to the ±5 nm spreads of Δk_a (1.4 %) and Δk_b (1.0 %): every device then works,
at about 1/300 of the nominal double-matched peak (the THG analogue of the bandwidth-efficiency ceiling, applied
twice).

## Fabrication presets (`poling.FAB_PRESETS`)

| preset | l_min | grid | basis |
|---|---|---|---|
| `reproducible` (default) | 300 nm | 10 nm | 600 nm periods in x-cut TFLN (Zhao et al., APL 2020) |
| `frontier` | 110 nm | 5 nm | 215 nm periods in x-cut TFLN (Sabatti et al., 2025 preprint; check the published value) |
| `conventional` | 1 µm | 20 nm | few-µm periods of 1550 nm SHG/DFG devices |

These are demonstrated periodic gratings. Short isolated domains in aperiodic masks may merge or fail to nucleate, so check them against your own process. `fabricate` snaps walls to the grid and absorbs shorter domains into their neighbours. Every design is evaluated after this step.

## Status (plan milestones)

1. `cwe_shg` engine: done (math-engines).
2. Poling generators, fast undepleted solver, ensemble statistics, synthesis stages 1–3 (+ statistical random design): done here.
3. Waveguide layer (EIM, Python 2D semi-vectorial FD solver, overlap A_eff, Miller-scaled d, cached tables): done.
4. `cwe_dfg` engine (math-engines; transfer matrix, depleted RK4, JS cross-check), OPAModel, loading strips,
   wafer thickness spread: done.
5. `cwe_cascade` engine (math-engines), THGModel, double-matching designs, THG thickness spread: done.
6. Optimisation: envelope IFTA, JAX wall gradients, DE over chirp / apodisation / jitter / blended THG masks,
   band and thickness-ensemble objectives (mean, flat, yield, soft minimum, geometric mean), TFLN vs TFLT: done.
7. Optional: JS port and applet, pulsed / dispersive extension: open.

Not yet: pump depletion inside `design` (use `cwe_shg.shg_coupled` on the mask), Δk(z) from width variation in the batched solver (use `cwe_shg`), adiabatic-conversion robustness (needs the depleted/OPA solvers).
