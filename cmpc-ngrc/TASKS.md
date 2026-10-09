# cmpc-ngrc task list

The tasks are in dependency order. Each one is a module or an experiment, and has a definition of
done (DoD) that is a test or a figure. IDs (T1.2, …) are for commits and cross-references.
Status: `[x]` done, `[~]` partly done (*Now:* says what is there), `[ ]` to do. The explorer applet
(`web/`) reads this file and shows it on its Progress tab.

## 0. Decisions (fixed)

| # | Question | Decision |
|---|---|---|
| D1 | Cell | **Circular** cell first (integrable: sin χ is conserved, so the launch angle sets a caustic disk of radius R_c sin χ that rays never enter). Stadium / segmented cells later, from `gmpc.planar`. |
| D2 | Platform / λ | Si₃N₄-like membrane slab, TE0, λ = 1.55 µm, n_eff ≈ 1.8 (`geometry.membrane_neff`). Cell radius 1 mm. |
| D3 | Physical origin of Δn_eff | Local thickness change Δt or cladding change: Δn_eff = (∂n_eff/∂t)·Δt (`geometry.dneff_dthickness`); image grey level ↦ Δn. Typical \|Δn_eff\| 1e-4 … 1e-2, dots 20–300 µm. |
| D4 | Decomposition | **2D circular-harmonic decomposition (CHD)** of the dot contour, r(θ) = R[1 + Σ_m (a_m cos mθ + b_m sin mθ)], m = 2 … 6 (8). All data are 2D; no spherical harmonics. |
| D5 | Target | Regress a_m, b_m and the rotation-invariant power p_m = a_m² + b_m²; classify the dominant m. |
| D6 | Readout | Ridge regression on NGRC features (constant + linear + quadratic) as the reference; linear and RBF SVR/SVC as the "simple SVM". |
| D7 | Ports | Any number of ports anywhere on the wall, each "in", "out" or "inout", with its own launch angle and fan; symmetric and asymmetric presets. Input ports are holes too. |
| D8 | Detectors | 64-pixel line 100 µm outside every output port; intensity only. Features = all (input, output) pairs. |

## 1. Foundations

- [x] **T1.1 Package skeleton.** `ngrc/`, `tests/`, `examples/`, `web/`, `tools/`, pyproject, conftest (imports `math-engines`).
- [x] **T1.2 Circular cell with multiple ports.** `geometry.CircularCell`, `Port` (angle, width, role, launch, fan), `symmetric_ports`, `ports_at`, detector lines, constant or R(cos χ) wall reflectance, membrane n_eff helpers.
- [~] **T1.3 Configs and reproducibility.** *Now:* dataclass configs with dict round trip, seeded generators, ensemble hash. Missing: one JSON run config that regenerates a whole experiment.

## 2. Index perturbations as images

- [x] **T2.1 Δn grid.** `index_map.GridIndex`: bicubic spline of an image (value, gradient, Hessian), zero outside.
- [x] **T2.2 Image import.** `index_map.from_image`: PNG/TIFF/.npy → Δn (scale, placement, threshold, blur). The applet fits an image contour with circular harmonics and inserts it as a dot.
- [~] **T2.3 Edge model and validity.** *Now:* tanh edges for analytic dots, Gaussian blur for images. Missing: automatic flag when the edge is narrower than λ/n_eff.
- [x] **T2.4 Dot generator.** `shapes.Shape` (analytic Δn, ∇Δn, Hessian, bounding circle, safe step), `random_shape`, `Perturbation` (sum of components).
- [x] **T2.5 CHD.** `harmonics`: exact coefficients of a dot, contour of any Δn map around its centroid, Fourier coefficients, power spectrum. Recovers the coefficients from a binary image to < 0.006.

## 3. Ray engine

- [x] **T3.1 Ray equation in the perturbations.** RK4 of dr/ds = t, dp/ds = ∇n, dL/ds = n inside bounding circles, exact straight segments between them; adaptive steps across flat interiors. Tested: |p| = n, 4th-order convergence, parabolic GRIN period.
- [x] **T3.2 Wall interaction.** Specular reflection on the circle, √R(χ), reflection phase; unperturbed chords 2R_c cos χ reproduced.
- [x] **T3.3 Ports and termination.** Exit records per port, losses by bounce limit and amplitude floor, power per port.
- [x] **T3.4 Performance.** Batched numpy tracer (reference); JS engine through node on all cores (`jsengine.node_fields`, `tools/node_fields.mjs`); phase screen ≈ 1 s per sample (6000 directions, two inputs) with Radon tables, flat typed arrays and record pruning.

## 4. Complex rays / Gaussian beamlets and the speckle field

- [x] **T4.1 Dynamic ray tracing → frozen Gaussians.** Stability matrix from Q, P along each ray; Herman–Kluk prefactor with continuous branch; frozen width w_f. Replaces the evolving-beamlet summation, whose beamlets grew to millimetres in the circle (degenerate mirror system) and applied a dot's lens effect to the whole beamlet. Tested against exact Gaussian beams through clipping apertures and oblique mirrors (`tests/test_fga.py`).
- [x] **T4.2 Source decomposition.** Input beam projected onto coherent states on a phase-space grid (positions × directions); the single-pass field equals the exact beam (power within 0.1 %).
- [x] **T4.3 Ports and detectors.** Smooth aperture model (footprint overlap for the reflected power, field sampled across the opening), far-field detectors (64 directions per output port), E = K (B · phase).
- [~] **T4.4 Energy and reciprocity.** *Now:* power accounting per port and lost rays. Missing: reciprocity test.
- [~] **T4.5 Validity.** *Now:* FGA vs BPM through a dot (E9), phase screen vs curved rays vs Δn (E4), prefactor cut-off for chaotic cells. Missing: convergence vs number of launch directions reported per experiment.

## 5. Fast first-order engine and wave reference

- [x] **T5.1 Phase-screen model.** `perturbative.PhaseScreenModel`: trace once, ΔL = ∫Δn ds on the unperturbed chords (spectrally accurate midpoint rule), E = G @ exp(i k0 ΔL). Identical to the straight-ray tracer; 0.1–0.3 s per sample.
- [~] **T5.2 Wave reference.** *Now:* split-step BPM for one pass through a dot (`wave_ref`, E9: scattered-field correlation ≥ 0.9997). Missing: a full-wave check of multi-bounce speckle in a small cell.

## 6. Features (the NGRC layer)

- [x] **T6.1 Multiplexing.** Input ports × output ports, plus cell variants (launch-angle and wavelength steps, `CircularCell.variant`, E8).
- [x] **T6.2 NGRC features.** `features.stack_intensities` (raw, relative, log, delta) and `ngrc_features` (constant + linear + quadratic, random subset).
- [~] **T6.3 Detector realism.** *Now:* additive detector noise and uniform index (temperature) drift in E11/E12. Missing: fabrication noise.

## 7. Datasets and readouts

- [x] **T7.1 Datasets.** `dataset.Ensemble` → labels (R, a_m, b_m, p_m, dominant m) and features, phase-screen or curved engine.
- [x] **T7.2 Readouts.** `readout`: closed-form ridge with CV, linear/RBF SVR (one per target), linear/RBF SVC, R².
- [~] **T7.3 Baselines.** *Now:* the same readouts on the raw Δn image. Missing: single-pass (no multipass) reference.

## 8. Analysis experiments

- [~] **T8.1 Sensitivity.** *Now:* finite-difference ∂I/∂a_m and singular values vs launch angle and port layout (E1, E2). First result: dots inside the caustic disk R_c sin χ are invisible (zero sensitivity at 57° launch).
- [x] **T8.2 Learning curves.** E6: R² vs number of training dots.
- [x] **T8.3 Nonlinearity regime.** E4 (speckle correlation phase screen vs curved rays, decorrelation vs Δn) and E7/E12 (readout quality vs Δn, curved rays vs phase screen, with detector noise).
- [~] **T8.4 Position and rotation invariance.** *Now:* at a fixed position (800 dots) NGRC ridge recovers a_m, b_m with R² 0.96 / 0.86 / 0.83 / 0.76 / 0.73 (m = 2…6), above the raw-image baseline; with random positions nothing is learned at N = 800. Next: translation-invariant features, more samples, position as an extra target.
- [~] **T8.5 Robustness.** *Now:* detector noise and temperature drift (E11). Missing: fabrication errors (wall tilts), retraining cost.
- [~] **T8.6 Other cells.** *Now:* any `gmpc.planar` wall (`geometry.WallCell`), stadium vs circle readout (E10, Python FGA with the prefactor cut-off). Missing: segmented circle; stadium in the JS engine.

## 9. Integration and output

- [x] **T9.1 Explorer applet.** `web/cmpc-ngrc-explorer.html`: cell view with draggable ports and dots, live curved-ray speckle, CHD, in-browser ridge readout lab, progress board with the Python results.
- [x] **T9.2 JS port check.** `tools/make_vectors.py` + `tools/check_js.mjs`: the JS engine reproduces the Python exits, optical paths, beamlets and fields to machine precision.
- [ ] **T9.3 Engine spec** in `math-engines` (`spec.yaml` + test vectors) for the propagation core.
- [x] **T9.4 Docs.** `docs/MODELS.md`: equations, frozen-Gaussian model, ports and detectors, phase screen, validation, limits, references.

## Suggested order of the next steps

T8.4 (invariant features: speckle autocorrelation / far-field power, or position-aware readout) →
T8.2 learning curves on the best layout → T3.4 node-based curved datasets → T6.1 angle/λ multiplexing →
T5.2 wave reference → T8.6 other cells.
