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
| D8 | Detectors | Far field of every output port, 64 directions (sin θ uniform in ±0.9); intensity only. Features = all (input, output) pairs. |
| D9 | Scope of the study | **One dot, at the cell centre (0, 0).** Moving the dot and adding dots stay available in the code (`Shape.x0/y0`, `Perturbation`, `Ensemble.centre/jitter/n_shapes`) but are later work (section 10). A centred dot is reached only by rays with R_c \|sin χ\| < dot radius, so launch angles near 0. The earlier off-centre results (dot at (250, 100) µm) are archived in `results/progress_offcentre.json`. |
| D10 | Working cell | **24-facet segmented circle** (circumradius 1 mm, flat facets; `geometry.polygon`), ports at the asym4 angles, two inputs, launch 3°, fan ±5.7°, Δn = 3e-5. Flat facets break the smooth circle's parity, so all orders of a centred dot are read (E16, E18, E19). The smooth circle stays as the reference. |

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
- [~] **T5.2 Wave reference.** *Now:* split-step BPM for one pass through a dot (`wave_ref`, E9: scattered-field correlation ≥ 0.9997), and a full-wave multi-bounce reference: `ngrc.fdfd.WaveCell`, 2D scalar Helmholtz by finite differences in a 25 µm cell (circle or n facets, lossy metal wall with port gaps, PML, Gaussian line sources, same far-field detector), empty cell factorised once with MKL PARDISO, each dot by GMRES on its own grid cells (3–8 iterations). Tests: GMRES = direct solve, weak dot = first-order Born. E24 (400 dots, linear readout): the 24-facet cell reads every order in full-wave optics too (Δn_w = 1.2e-3 ≙ 3e-5: 0.95 / 0.85 / 0.86 / 0.64 / 0.79; RBF readout 0.96 / 0.91 / 0.90 / 0.72 / 0.83). At Δn_w = 1.2e-2 (≙ 3e-4) the wave response is more nonlinear than the phase screen (linear 0.94 / 0.81 / 0.79 / 0.30 / 0.49, RBF 0.94 / 0.84 / 0.86 / 0.50 / 0.72), while under 1 % noise waves and rays agree (0.83 / 0.44 / 0.66 / 0.18 / 0.30 vs 0.88 / 0.50 / 0.66 / 0.20 / 0.32, E26). The circle's odd-order suppression, which the ray engine predicts on the same small geometry (0.20 / 0.05, E26), is much weaker in waves at this scale: on the staircased grid the circle reads 0.83 / 0.50 (the staircase acts like fine facets); with smooth sub-pixel walls the circle gives 0.94 / 0.66 / 0.90 / 0.62 / 0.81 and the 24-facet cell 0.97 / 0.92 / 0.89 / 0.81 / 0.79 (RBF 0.98 / 0.96 / 0.92 / 0.84 / 0.85). So the facets still win on the odd orders in waves, but diffraction (a dot of ~3 wavelengths, a ±30° fan) washes out most of the ray-limit cancellation. In the 1 mm cell (dot ~100 wavelengths) the ray limit is closer and the effect should be stronger; the faceted cell avoids it in either picture. Limits: a 25 µm cell (40× smaller), second-order stencil (numerical dispersion), 400 dots.

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
- [~] **T8.4c Centred dot (D9).** `examples/run_centre.py`: E1 launch × fan, E2 layouts, E3 readouts, E6 learning curve, E8 multiplexing, E11 noise/drift, E12 Δn, E13–E15 odd orders. *Now:* launch 3°, fan ±5.7°, Δn = 3e-5 (every low-angle chord crosses the dot; at 1e-3 the phase adds up to tens of radians). Even orders are read almost perfectly (a₂, a₄, a₆: R² 0.99 / 0.96 / 0.96, p₂ 0.96); odd orders only weakly (0.35 / 0.21) at any launch angle (E13). The reason is a symmetry of the circle (see T8.4p). 1 % detector noise keeps a₂ (0.85) but removes m ≥ 3 at this Δn.
- [x] **T8.4p Parity of the circle.** Each ray keeps its offset p = R_c sin χ from the centre, and the chord after a bounce is almost the point reflection of the one before. A point reflection flips the sign of odd harmonics, so odd-order phases cancel chord by chord and even ones add up (E13–E15). Remedies: ports where the near-diametral chords land (asym8 · 2 in: odd 0.80 / 0.53, E2); shorter paths (E14). The real fix is the wall: any wall whose normals stop pointing at the centre. A flat cut (D-cell) does not help (all arc normals still point at the centre); a stadium with straights ≥ 100 µm, or a segmented circle with 12–192 facets, reads all orders (24 facets: 1.00 / 1.00 / 0.99 / 0.99 / 0.98; E16–E18). Working cell: 24 facets (D10).
- [ ] **T8.4r Rotation.** For a centred dot, rotating the dot is rotating the ports: test whether p_m readouts become rotation-robust with symmetric layouts, and what symmetric layouts lose (mirror pairs b_m ↔ −b_m at zero launch).
- [x] **T8.5 Robustness.** Detector noise and temperature drift (E11). Fabrication errors (E22): random facet tilts (1 mrad rms) and offsets (1 µm rms) in the 24-facet cell change nothing if the readout is trained on the chip itself (R² ≥ 0.97 in every order, two chips each). A readout trained on the design (a digital twin) fails already at 1 nm rms facet offset: the speckle is a fingerprint of the exact chip, so the readout must be trained (and periodically re-trained) on the chip.
- [x] **T8.6 Other cells.** Any `gmpc.planar` wall (`geometry.WallCell`, `d_cell`, `polygon`, `stadium`, `ports_by_angle`) in the Python engine and, since this round, in the JS engine (Cell2D elements, ports at boundary coordinates; `WallCell.to_dict`; JS = Python to correlation 1 − 1e-6, `tests/test_jsengine.py`). The applet has a wall-shape selector (circle, segmented, stadium). Chaotic walls are limited to about the Ehrenfest time by the prefactor cut-off (E10, E15); the segmented circle is pseudo-integrable and keeps its long paths.
- [~] **T8.8 Noise robustness of the working cell.** *Now (E19, 1600 dots):* noise-free all readouts reach R² ≥ 0.98 in every order (linear SVR included; p_m 0.98 / 0.94 / 0.87 / 0.78 / 0.64; 100 training dots already give 0.94). With detector noise σ·Ī₀ the readout falls fast: 0.91 / 0.76 / 0.81 / 0.62 / 0.56 at 0.3 %, 0.71 / 0.26 / 0.32 / 0.07 / 0.13 at 1 %. E20 (800 dots): a larger Δn wins under noise. At 1 % noise Δn = 3e-4 gives 0.92 / 0.78 / 0.81 / 0.59 / 0.47 and Δn = 1e-3 gives 0.97 / 0.91 / 0.87 / 0.76 / 0.68 (noise-free 0.99 … 0.93). That is where the phase screen starts to miss ray bending (E4), so T8.7 (curved rays at convergence) now matters for the working point. Next: curved-ray check at Δn = 3e-4 … 1e-3, averaging, wavelength views, larger training sets. E25 (800 dots): independent wavelength views help most at higher noise: Δn = 1e-3, 1 % noise: 1 view 0.96 / 0.91 / 0.87 / 0.78 / 0.69, 5 views 0.98 / 0.94 / 0.92 / 0.82 / 0.78; 3 % noise: 0.92 / 0.78 / 0.71 / 0.51 / 0.44 → 0.95 / 0.87 / 0.84 / 0.70 / 0.62.

## 9. Integration and output

- [x] **T9.1 Explorer applet.** `web/cmpc-ngrc-explorer.html`: cell view with draggable ports and dots, live curved-ray speckle, CHD, in-browser ridge readout lab, progress board with the Python results.
- [x] **T9.2 JS port check.** `tools/make_vectors.py` + `tools/check_js.mjs`: the JS engine reproduces the Python exits, optical paths, beamlets and fields to machine precision.
- [ ] **T9.3 Engine spec** in `math-engines` (`spec.yaml` + test vectors) for the propagation core.
- [x] **T9.4 Docs.** `docs/MODELS.md`: equations, frozen-Gaussian model, ports and detectors, phase screen, validation, limits, references.

- [x] **T8.7 Curved rays at convergence.** E7b (circle) and E21 (24-facet cell): brute-force curved-ray datasets are buried in frozen-Gaussian sampling noise (speckle correlation with the phase screen still rising at 12000 directions: 0.94 at Δn = 3e-4, 0.77 at 1e-3; shape-change correlation ≈ 0.1). E23 measures the bending error without sampling noise by pairing the same rays: in the smooth circle the second-order phase error is 10 / 23 / 46 % of the shape signal at Δn = 1e-4 / 3e-4 / 1e-3 and 75–92 % of the field keeps its ray sequence. In the faceted cell most records have tiny errors (median ≤ 1.3e-3 rad), but µm endpoint shifts move rays across facet corners, so after 20–50 bounces 50–75 % of the field changes its ray sequence. Ray optics cannot say whether the wave field follows (corner diffraction); the full-wave reservoir (E24, T5.2) shows the faceted cell keeps all orders, with a more nonlinear response than the phase screen at Δn ≙ 3e-4 (recovered in part by RBF / NGRC readouts) and the same noise behaviour as the ray model.

## 10. Later: dot position and several dots

Kept in the code, not pursued now (D9).

- [~] **T10.1 Position tolerance.** *Now (off-centre, archived):* a readout is tied to one dot position: a 2 µm shift costs little, higher orders are lost by 5–10 µm; placement jitter in training keeps a_2 at R² 0.80 for 10 µm; dots anywhere in the central 600 µm are not learned with 1600 dots (E5). Redo for the centred dot.
- [ ] **T10.2 Sites.** An array of fixed dot positions, one readout each, or translation-invariant optics.
- [ ] **T10.3 Several dots.** CHD of each dot of a `Perturbation`; cross-talk between dots.

## Suggested order of the next steps

Larger full-wave cells (50–100 µm, finer grid or a fourth-order stencil) to follow the parity effect towards the
ray limit → nonlinear (RBF / NGRC) readouts with more training dots at the noise-robust Δn → T8.4r rotation and
port symmetry → section 10 (position, several dots).
