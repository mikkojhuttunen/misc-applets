# cmpc-ngrc task list

The tasks are in dependency order. Each one is a module or an experiment, and has a definition of
done (DoD) that is a test or a figure. IDs (T1.2, …) are for commits and cross-references.

## 0. Decisions to fix first (defaults proposed)

| # | Question | Proposed default |
|---|---|---|
| D1 | Cell | Smooth stadium (chaotic, λ ≈ 1/reflection) as main case; segmented circle CMPC (pseudo-integrable) as contrast. Walls from `gmpc.planar`. |
| D2 | Platform / λ | Si₃N₄ membrane slab, TE0, λ = 1.55 µm; n_eff from `membrane_mode` / `slab_waveguide`. Cell size 1–5 mm. |
| D3 | Physical origin of Δn_eff | Local slab-thickness change Δt (etched/deposited dot) or cladding/analyte change: Δn_eff = (∂n_eff/∂t)·Δt from the slab solver; image grey level ↦ Δt or Δn directly. Typical \|Δn_eff\| 1e-4 … 1e-2, feature size 20–300 µm. |
| D4 | What "SHD" means here | (a) **2D**: circular-harmonic coefficients of the contour, r(θ) = R[1 + Σ_m (a_m cos mθ + b_m sin mθ)], m ≤ 6. (b) **3D**: shape treated as a star-shaped bump h(θ, φ) → Y_lm, l ≤ 4, with Δn ∝ column height. Start with (a); add (b) once the pipeline runs. |
| D5 | Target | Regress a_m, b_m (or the rotation-invariant power \|c_m\|²); classify the dominant m as a second task. |
| D6 | Readout | Ridge regression on NGRC features (constant + linear + quadratic monomials) as the reference; linear SVR/SVC and RBF SVM as the "simple SVM". |
| D7 | Output / detectors | N_d = 16 … 256 point detectors (or pixels) along the output port or in the far field; intensity only (\|E\|²), with optional shot/thermal noise. |
| D8 | Input multiplexing ("virtual nodes") | K launch angles and/or wavelengths per sample, so the feature length is N_d × K. This plays the role of the time delays in NGRC. |

## 1. Foundations

- [ ] **T1.1 Package skeleton.** `ngrc/`, `tests/`, `examples/`, pyproject, conftest that makes
  `math-engines/engines` and `general-mpc/gmpc` importable. *DoD:* `pytest` runs; imports work from a clean checkout.
- [ ] **T1.2 `geometry.py`.** Cell object wrapping a `gmpc.planar.Cell2D` (stadium / circle / polygon,
  perturbed walls); input port (position, width, launch-angle range), output port and detector line;
  slab stack → n_eff(λ), n_g via `membrane_mode`; wall reflectance R(χ) (constant, or the `cell_mirror` DBR).
  *DoD:* test that the wall hits match `gmpc.trace2d` for Δn = 0.
- [ ] **T1.3 Units, RNG and config.** Dataclass configs (cell, source, detectors, perturbation ensemble),
  seeded `numpy.random.Generator` everywhere, JSON round-trip so every dataset can be regenerated.

## 2. Index perturbations as images

- [ ] **T2.1 `index_map.py`: grid.** Δn_eff(x, y) on a uniform grid covering the cell, with an explicit
  pixel pitch and origin; n(x, y) = n_eff,0 + Δn. Bicubic (C¹) interpolation of n and ∇n
  (`scipy.ndimage.map_coordinates` or a hand-written cubic convolution, vectorised over rays).
  *DoD:* ∇n of an analytic Gaussian bump is within 1e-3 relative error; continuity across pixels.
- [ ] **T2.2 Image import.** PNG/TIFF/`.npy` → Δn map: grey level → Δn (linear map or Δt → Δn_eff from
  the slab solver), scale (µm/pixel), placement and rotation in the cell, optional mask to keep shapes
  away from the walls and ports. *DoD:* round-trip of a synthetic disk image gives the expected area and centroid.
- [ ] **T2.3 Smoothing / edge model.** Sharp binary edges make ∇n singular for rays. Model the edge as
  erf or tanh with width w_e (physical: lithography / diffusion blur, ≥ a few λ/n). Report when w_e
  is below the ray-optics validity limit (w_e ≳ λ/n_eff, Fresnel number per feature).
- [ ] **T2.4 `shapes.py`: generator.** Analytic rasteriser for r(θ) = R[1 + Σ_m (a_m cos mθ + b_m sin mθ)]
  with random R, centre, rotation, Δn, edge width; several shapes per sample (optional); 3D bumps
  h(θ, φ) = Σ c_lm Y_lm projected to a column height map. Returns the map **and** the exact ground-truth coefficients.
  *DoD:* `harmonics.py` recovers the input coefficients from the rasterised map to < 1 % for m ≤ 6.
- [ ] **T2.5 `harmonics.py`.** Circular-harmonic decomposition of a contour (boundary extraction +
  FFT in θ around the centroid), of a filled Δn map (polar resampling, Fourier–Bessel or angular moments),
  and the Y_lm decomposition for the 3D bump case; rotation-invariant power spectra. Also used as the
  "ideal SHD" baseline in §7.

## 3. Ray engine (geometric, curved rays)

- [ ] **T3.1 `rays.py`: ray equation.** Integrate dr/dσ = p, dp/dσ = n∇n (σ = ∫ds/n, |p| = n),
  with RK4 fixed step and an adaptive RK45 option; step limited by the local gradient scale. Accumulate
  optical path L = ∫n ds and geometric length.
  *DoD:* analytic checks: uniform n → straight lines and L = n·s; parabolic GRIN slab →
  sinusoidal ray, period 2π/g; radial Luneburg / Maxwell fish-eye → known closed-form trajectories;
  |p| = n conserved to 1e-10.
- [ ] **T3.2 Wall interaction.** Find the hit between steps (segment–wall intersection with the
  `gmpc.planar` element chain, refined by bisection on the RK step), specular reflection, incidence angle χ,
  amplitude factor √R(χ) and reflection phase (DBR phase from `bragg_grating`/`cell_mirror` when used).
  *DoD:* with Δn = 0 the bounce sequence and path match `gmpc.trace2d` to 1e-9 m over 1000 bounces.
- [ ] **T3.3 Ports and termination.** Rays leave through the output port (record position y_d,
  direction, L, amplitude, number of bounces) or are lost (port in, leak, amplitude below threshold,
  max path). Vectorised over many rays at once (numpy arrays, rays as columns).
- [ ] **T3.4 Performance.** Target 10⁴ rays × 10³ bounces in minutes on a laptop. Profile; optional
  numba path; cache the unperturbed (Δn = 0) ray table per cell/source.

## 4. Complex rays / Gaussian beamlets and the speckle field

- [ ] **T4.1 `beamlets.py`: dynamic ray tracing.** Along each central ray carry the 2×2 complex
  (Q, P) system (paraxial Gaussian beam about the ray, complex curvature q): dQ/dσ = P, dP/dσ = (∂²n/∂n⊥²)·n·Q
  in the medium; curvature-mirror transform at each wall (ABCD of a curved/faceted wall at oblique
  incidence, as in `gaussian_beam`); Gouy phase from arg Q.
  *DoD:* free space reproduces w(z), R(z), Gouy of `gaussian_beam`; a quadratic GRIN reproduces its
  ABCD matrix; reflection off a concave arc matches the oblique-incidence ABCD.
- [ ] **T4.2 Source decomposition.** Input-port field (Gaussian mode of the input waveguide/port) →
  set of beamlets over launch angle and position (Gabor frame or equal-angle fan with overlap correction),
  with weights so the summed input field matches the port field. *DoD:* reconstruction error of the
  input field < 1 %.
- [ ] **T4.3 `field.py`: coherent sum.** E(y_d) = Σ_rays A_j √(Q₀/Q_j) exp(i k₀ L_j + i φ_R,j)
  exp(i k₀ n Δ⊥²/(2 q_j)) at the detector line; far field by FFT of the aperture field; intensity
  and polarisation (TE scalar to start). *DoD:* speckle statistics of the chaotic cell: intensity
  pdf ≈ exponential, contrast ≈ 1, correlation length ≈ λ/(2 NA) at the port.
- [ ] **T4.4 Energy and reciprocity checks.** Output power vs Σ R^j over the path ensemble;
  swapping source and detector gives the same transmission amplitude.
- [ ] **T4.5 Validity map.** Ray/beamlet validity vs feature size, Δn and edge width (Fresnel
  number a²/(λL), beamlet width vs feature size, caustics). Flag samples outside the valid regime.

## 5. Fast first-order engine and wave reference

- [ ] **T5.1 `perturbative.py`.** For small Δn: keep the unperturbed rays, add the phase
  Δφ_j = k₀ ∫ Δn ds along them (line integrals = projections of the Δn image along the ray segments,
  i.e. a sum of Radon samples), with optional first-order ray bending. 10–100× faster dataset generation.
  *DoD:* agreement with the full curved-ray engine (field correlation > 0.99) for Δn·L_feature ≪ λ;
  map where it breaks.
- [ ] **T5.2 `wave_ref.py`.** 2D scalar Helmholtz (FDFD, PML, sparse solver) for a small cell
  (≈100 λ), or a wide-angle split-step BPM for a single pass through the perturbation. Validation only.
  *DoD:* ray/beamlet speckle vs FDFD speckle correlation reported vs Δn and feature size.

## 6. Features (the NGRC layer)

- [ ] **T6.1 Multiplexing.** For each sample compute speckle for K launch conditions (angle steps
  in the input port, wavelength steps Δλ ≈ speckle decorrelation bandwidth, or several input ports).
  The K × N_d intensities are the "nodes".
- [ ] **T6.2 `features.py`.** NGRC feature vector: [1, I (linear), unique quadratic monomials I_i I_j
  (optionally subsampled / random-projected)], normalisation (divide by the Δn = 0 reference speckle,
  log, z-score). Also the "differential" feature ΔI = I(Δn) − I(0).
- [ ] **T6.3 Detector realism.** Pixel integration, finite dynamic range / quantisation, shot noise,
  laser phase/frequency noise, thermal drift of n_eff (common-mode Δn), random cell fabrication noise
  (fixed per chip, so it is learned out).

## 7. Datasets and readouts

- [ ] **T7.1 `dataset.py`.** Random ensembles (N = 10³ … 10⁵) of shapes from T2.4: control the
  m-content, random position/rotation (or fixed, as an easier first task), Δn and size ranges. Store
  config + labels + features in `.npz` with a hash of the config; deterministic regeneration.
- [ ] **T7.2 `readout.py`.** Ridge (closed form, regularisation by CV) = NGRC readout; LinearSVR / SVC
  and RBF SVM (scikit-learn), multi-output by one model per coefficient; train/val/test splits,
  hyper-parameter search, metrics: R² and RMSE per (m) or (l, m), rotation-invariant power error,
  classification accuracy of the dominant order.
- [ ] **T7.3 Baselines.** The same readouts on (i) the raw Δn image pixels, (ii) random features of
  the image (random projection + quadratic), (iii) exact harmonics (upper bound), (iv) a "no cell"
  single pass (straight transmission through the perturbation, no multipass). The CMPC has to beat
  (i)/(ii) at equal feature count or equal sample count to be worth it.

## 8. Analysis experiments (answers to the research question)

- [ ] **T8.1 Sensitivity.** Jacobian ∂I/∂a_m by finite differences; singular values → effective rank
  of the optical feature map per m; which m are visible to the cell at all.
- [ ] **T8.2 Learning curves.** Test R² vs number of training samples, N_d, K, and multipass length
  (via R or port size); chaotic stadium vs segmented circle vs single pass.
- [ ] **T8.3 Nonlinearity regime.** Δn sweep: linear-in-Δn field regime (quadratic intensity → ridge
  on NGRC quadratic features is enough) vs strong phase regime (speckle fully decorrelated → only memorisation).
  Locate the optimum Δn·L_path/λ.
- [ ] **T8.4 Invariance.** Rotation and translation of the shapes: does the readout learn the
  invariant power spectrum |c_m|²? Is a translation-invariant feature (speckle autocorrelation /
  far-field power spectrum) better?
- [ ] **T8.5 Robustness.** Noise (T6.3), fabrication noise, drift; retraining cost.
- [ ] **T8.6 3D SHD case (D4b).** Repeat T8.2 with Y_lm targets.

## 9. Integration and output

- [ ] **T9.1 Figures** (`examples/`): cell + Δn map + ray paths; speckle with and without the
  perturbation; validation plots (§3–5); learning curves and per-m R² (§8).
- [ ] **T9.2 Engine spec** in `math-engines` (`spec.yaml` + test vectors) for the propagation
  core, if it should be callable from the web calculator.
- [ ] **T9.3 Applet** (optional, later): JS port of the perturbative engine (T5.1) for an interactive
  "draw a shape, see the speckle and the predicted harmonics" page, checked against the Python vectors.
- [ ] **T9.4 Docs.** `docs/MODELS.md` with equations, assumptions, validity limits and references
  (ray equation in GRIN media, Červený dynamic ray tracing / Gaussian beam summation, NGRC: Gauthier
  et al., Nat. Commun. 12, 5564 (2021)).

## Suggested order of work

T1.1 → T1.2 → T2.1 → T2.4 → T2.5 → T3.1 → T3.2 → T3.3 → T4.3 (point rays, plane-wave phase only) →
T5.1 → T6.1–T6.2 → T7.1–T7.3 (first end-to-end result on the fast path) → T4.1–T4.2 (beamlets) →
T5.2 (validation) → §8 experiments → §9.

With this order an end-to-end "shape → speckle → SVM → harmonics" result comes early on the cheap
first-order engine. The full complex-ray engine and the wave reference then confirm or correct it.
