# Grating structures (CRIGF, grating couplers, DBRs, curved gratings): task list

Tool code only. Design results, parameter studies and device data stay out of this repository (see README).

## Status (9 Oct 2026)

Python reference engines in `math-engines/`, tested:
- `bragg_grating` v2: κ of rectangular gratings, tanh²(κL), bandwidth, effective length, coupled-mode spectrum, Abelès stacks at normal and oblique incidence (s/p, TIR), `TrenchDBR`, `double_resonant_orders`.
- `slab_waveguide` (three-layer, multilayer), `anisotropic_slab`, `materials`, `qpm_shg`, `gaussian_beam`, `membrane_mode`, `cell_mirror`.
- `grating_coupler` v1 (A1): arbitrary etch profiles, local n_eff table, Fourier κ, out-of-plane radiation per diffraction order with the vertical stack (BOX, Si or Au handle), far field. Reference for `radiation`, `farField`, `profileFn`, `tableAt`, `modeParams` and `planeWave` in `dbr-engine.js`.
- `crigf` v1 (A2): DBR | coupler | DBR fed by a Gaussian beam, Kazarinov–Henry coupler, transfer-matrix DBRs, power budget, resonance finder. Reference for `crigfSetup`, `crigfAt`, `findResonance`.

`dbr-engine.js` only (no Python reference yet): phase tuning (`tunePhase`), field map along the cavity, SHG in the grating, QPM chirp/apodisation optimiser, pump depletion, ridge effective-index lateral model, curved-DBR Gaussian bounce.

## A. Validation and infrastructure

- [x] A1. `grating_coupler` engine: profile κ, radiation per order, directionality, far field; spec, vectors, JS port check (9 cases, port agrees to 1e-9, 1e-5 for the rounded profile).
  Fixed in `dbr-engine.js` on the way: rectangular teeth sampled as exact cell averages (the fill was rounded to 1/1024, κ off by up to ~2 % near a zero of sin(πmf)); far field block-averages instead of picking 1 sample in 8; the radiation-order loop covers orders reaching a Si handle.
- [x] A2. `crigf` engine: Python port of `crigfAt` and `findResonance`; spec, vectors (5 cases, one through a resonance), JS check (port agrees to 1e-8).
  Independent checks: coupler radiation = `grating_coupler` q = 1 radiation (1e-9); RK4 = matrix exponential in the rotating frame; θ → −θ swaps the escape; no etch = bare slab; strong DBRs return all guided light; passivity; Lorentzian resonance.
  Findings: the coupler alone leaves up to 4e-4 in other modes for a 5 µm beam (the applet note said 1e-4, corrected); on a cavity resonance a few percent go into other modes because the radiated profile follows the cavity decay; at oblique incidence the backward guided wave radiates at −θ and is counted as "other" (B11).
- [ ] A3. Rigorous 1D-periodic RCWA/FMM engine (TE/TM, multilayer): benchmark radiation loss, κ, directionality and CRIGF spectra; replaces the "factor of a few" accuracy statement.
- [ ] A4. Complex-β Floquet–Bloch (leaky-mode) solver of the grating section: band edges, bright/dark modes at normal incidence, Q.
- [ ] A5. `dbr-engine.js` in SI; material table generated from `materials` (LiTaO₃ Bond refits, temperature-dependent LN).
- [ ] A6. `dbr-structures` in `tools/build_standalone.py`.
- [ ] A7. Proper TM radiation: the thin-sheet formula and the plane-wave local field are TE expressions (TM uses H_y with the TE formula). Found while writing A1.
- [ ] A8. Python reference for the lateral models (ridge Γ_lat, curved-DBR `bounce`, `ridgeLateral`) with vectors.

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
