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
chi2sim/waveguide.py   Ridge (film, etch, top width, sidewall angle, cladding, cut, material); eim; 2D semi-vectorial
                       FD solver (port of waveguide-core.js; sparse shift-invert, sub-pixel ε on a graded grid);
                       a_eff_shg; SHGModel (n_p, n_SH, n_g, A_eff tables, splined; Δk(λ) callable, κ(λ), period,
                       save/load/cached); width_scan (period, Δk and A_eff vs width, dΔk/dw)
chi2sim/dispersion.py  simple Δk(λ) models: LinearIndexSHG, MaterialSHG (bulk Sellmeier)
chi2sim/synthesis.py   band_to_dk, lambda_grid, efficiency_ceiling, chirp_from_spectrum, poling_from_chirp,
                       parasitic_orders, random_band_design, design(mode="chirp" | "random" | "hybrid") -> Design
examples/band_designs.py   ordered baseline vs qpm_shg; chirp / random / hybrid for 10, 50, 100 nm FWHM vs the ceiling
examples/ridge_shg.py      TFLN vs TFLT x-cut ridges from the FD solver: period, A_eff, η, width tolerance, band designs
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
4. `cwe_dfg` (OPA), 5. `cwe_cascade` (THG), 6. hybrid optimisation, IFTA and JAX gradient design: open.

Not yet: pump depletion inside `design` (use `cwe_shg.shg_coupled` on the mask), Δk(z) from width variation in the batched solver (use `cwe_shg`), adiabatic-conversion robustness (needs the depleted/OPA solvers).
