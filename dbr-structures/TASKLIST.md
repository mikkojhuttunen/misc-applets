# Grating structures (CRIGF, grating couplers, DBRs, curved gratings): task list

Tool code only. Design results, parameter studies and device data stay out of this repository (see README).

## Status (9 Oct 2026)

Python reference engines in `math-engines/`, tested:
- `bragg_grating` v2: κ of rectangular gratings, tanh²(κL), bandwidth, effective length, coupled-mode spectrum, Abelès stacks at normal and oblique incidence (s/p, TIR), `TrenchDBR`, `double_resonant_orders`.
- `slab_waveguide` (three-layer, multilayer), `anisotropic_slab`, `materials`, `qpm_shg`, `gaussian_beam`, `membrane_mode`, `cell_mirror`.
- `grating_coupler` v1 (A1): arbitrary etch profiles, local n_eff table, Fourier κ, out-of-plane radiation per diffraction order with the vertical stack (BOX, Si or Au handle), far field. Reference for `radiation`, `farField`, `profileFn`, `tableAt`, `modeParams` and `planeWave` in `dbr-engine.js`.
- `rcwa` v1 (A3): rigorous Fourier modal method; benchmark of the thin-sheet, EIM-κ and coupled-mode coupler models.
- `crigf` v1 (A2): DBR | coupler | DBR fed by a Gaussian beam, Kazarinov–Henry coupler, transfer-matrix DBRs, power budget, resonance finder. Reference for `crigfSetup`, `crigfAt`, `findResonance`.

`dbr-engine.js` only (no Python reference yet): phase tuning (`tunePhase`), field map along the cavity, SHG in the grating, QPM chirp/apodisation optimiser, pump depletion, ridge effective-index lateral model, curved-DBR Gaussian bounce.

## A. Validation and infrastructure

- [x] A1. `grating_coupler` engine: profile κ, radiation per order, directionality, far field; spec, vectors, JS port check (9 cases, port agrees to 1e-9, 1e-5 for the rounded profile).
  Fixed in `dbr-engine.js` on the way: rectangular teeth sampled as exact cell averages (the fill was rounded to 1/1024, κ off by up to ~2 % near a zero of sin(πmf)); far field block-averages instead of picking 1 sample in 8; the radiation-order loop covers orders reaching a Si handle.
- [x] A2. `crigf` engine: Python port of `crigfAt` and `findResonance`; spec, vectors (5 cases, one through a resonance), JS check (port agrees to 1e-8).
  Independent checks: coupler radiation = `grating_coupler` q = 1 radiation (1e-9); RK4 = matrix exponential in the rotating frame; θ → −θ swaps the escape; no etch = bare slab; strong DBRs return all guided light; passivity; Lorentzian resonance.
  Findings: the coupler alone leaves up to 4e-4 in other modes for a 5 µm beam (the applet note said 1e-4, corrected); on a cavity resonance a few percent go into other modes because the radiated profile follows the cavity decay; at oblique incidence the backward guided wave radiates at −θ and is counted as "other" (B11).
- [x] A3. `rcwa` engine (Fourier modal method, TE/TM, Li's rule): diffraction, leaky and Bloch modes, radiation loss, directionality, DBR κ; benchmark table in `math-engines/engines/rcwa/README.md`.
  TE: thin-sheet radiation within 1–6 % (h ≤ t/3), split within 0.002, κ within 1 %; over BOX/handles within 6 % (Si) and 20 % (gold). Coupled-mode coupler (`crigf.infinite_grating`) vs rigorous resonance: linewidth within 1–7 %, position offset 0.1–0.9 nm from the EIM n_eff.
  TM: thin-sheet radiation ~10× too low, EIM κ 2–7× too high (see A7). Finite CRIGF spectra (DBRs + Gaussian beam) not benchmarked: needs an aperiodic solver (A9).
- [x] A4. Band edges of second-order gratings: `rcwa.resonance_mode` (complex frequency at a real Bloch wavevector, Q), `track_band`, `band_edge` front end; `crigf.band_edge_modes` (coupled-mode bright/dark modes, closed form).
  Rigorous vs coupled-mode: bright Q within 2–8 %, gap within 9 %; dark mode is a symmetry-protected BIC (Q ∝ 1/k_x²), broken by asymmetric teeth. Coupled-mode misses the small f = 0.5 gap (EIM N₂ = 0).
- [ ] A5. `dbr-engine.js` in SI; material table generated from `materials` (LiTaO₃ Bond refits, temperature-dependent LN).
- [ ] A6. `dbr-structures` in `tools/build_standalone.py`.
- [x] A7. TM radiation and κ: `grating_coupler.tm_weights` tooth model (D_x / E_z split, TM plane waves, wall screening c = 0.35 calibrated on `rcwa`), TM radiation, κ_TM = |ρ| κ_EIM with coupling sign −ρ; `crigf` TM (F_R/F_S local fields, complex radiative cross-coupling, −ρ on N₂ and on the DBR slices); JS port with vectors.
  Against `rcwa`: TM radiation mean 7 % (worst 33 %) for h ≤ 50 nm, up/down within 0.03; TM κ exact as h → 0, mean 9 % for h ≤ 20 nm; TM band-edge Q within 10 % with the right ordering; TE unchanged. Also fixed `rcwa.bragg_band` re-centring for deep etches.
  Left: wide teeth (f ≥ 0.7) in air and deep TM etches (corner fields), see A11.
- [ ] A8. Python reference for the lateral models (ridge Γ_lat, curved-DBR `bounce`, `ridgeLateral`) with vectors.
- [ ] A9. Aperiodic rigorous check of a whole CRIGF (supercell RCWA with absorbing boundaries, or 2D FDTD/FEM): DBR + coupler + Gaussian beam, against `crigf.response`.
- [ ] A10. EIM n_eff offset: the coupler resonance sits 0.1–0.9 nm short of the rigorous one (20–50 nm etch); a first-order correction from `rcwa` or a perturbative n_eff shift.
- [ ] A11. TM corner-field correction for wide teeth and deep etches (or a low-order RCWA, M ≈ 3, as the fast TM model: worst κ error 0.07 κ_EIM in the A7 data set).

## B. Physics models

- [ ] B1. Multilayer vertical stacks in the grating (loaded strips, hybrid films, partial etch through a multilayer) via `multilayer_neff`.
- [ ] B2. Anisotropic films via `anisotropic_slab`; angle-dependent n_eff along curved lines on x-cut LN/LT.
- [ ] B3. Apodised and chirped couplers and DBRs (per-period tables); Gaussian-matched out-coupling (> 80 % mode overlap).
- [ ] B4. Phase-shifted (λ/4) DBR, DFB, sampled gratings, doubly resonant guided DBRs (pump and SH).
- [ ] B5. Focusing couplers and curved gratings: elliptical line layout for a focal point, curved-DBR Hermite–Gauss eigenmodes and astigmatism, angle-dependent κ along arcs (`stack_R_oblique`).
- [ ] B6. Sidewall-corrugation gratings; ridge κ from overlap integrals on the 2D finite-difference solver (`waveguide-core`).
- [ ] B7. Mode-mismatch scattering at deep tooth walls (eigenmode expansion).
- [ ] B8. Beam effects: angular spectrum of a finite waist, tilt and offset tolerance, SMF-28 mode coupling.
- [ ] B9. DBR phase, group delay and dispersion; Q, FSR and finesse as outputs.
- [ ] B10. Thermo-optic, Kerr and photorefractive resonance shift.
- [ ] B11. Oblique CRIGF: report the beam the backward guided wave radiates at −θ as its own channel instead of "other"; overlap of the radiated cavity profile with the input beam (beam-shape optimisation, supports C2).

## C. Design and optimisation

- [ ] C1. Coupler optimiser: period, fill, etch depth, BOX thickness for peak coupling efficiency.
- [ ] C2. CRIGF critical-coupling optimiser at the operating power.
- [ ] C3. Apodisation inverse design (Fourier synthesis or layer peeling).
- [ ] C4. Monte Carlo tolerances (etch depth, fill, sidewall angle, stitching) and sensitivities (dλ/dh).
- [ ] C5. 2D maps R(λ, θ), resonance vs (L_s, h).
- [ ] C6. Coupler and CRIGF entries in the Pyodide web calculator.

## D. Export

- [ ] D1. GDS/DXF layout export (gdstk): arcs, focusing ellipses, apodised period lists.
- [ ] D2. MEEP / Tidy3D / Lumerical script stubs for full-wave verification.
- [ ] D3. Save and load the parameter set (JSON or URL hash).

Shared with `cmpc-sim/docs/TASKLIST.md` B9 (taper and grating-coupler port efficiency): use `grating_coupler`.
