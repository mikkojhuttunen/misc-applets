# grating_coupler

Surface-etched gratings on a slab waveguide, by the effective-index method: tooth profiles (rectangular, trapezoidal, rounded, sinusoidal, triangular, sawtooth), the local n_eff table, Fourier coupling coefficients κ_m of any profile, out-of-plane radiation per diffraction order and channel (cladding, substrate or Si handle through the BOX; a gold handle closes the downward channel), directionality, emission angles, the guided Bragg order, and the far field of N periods. SI units, angles in rad.

Version 1. `profile_coupling` and `grating_radiation` return Results; `SurfaceGrating`, `profile_function`, `fourier_coefficient` and `slab_mode` are helpers. `SurfaceGrating` takes indices as numbers or callables n(λ) and a complex handle index.

Python reference for `profileFn`, `sampleProfile`, `tableAt`, `modeParams`, `planeWave`, `radiation` and `farField` in `dbr-structures/dbr-engine.js`, checked by `tools/check_js_ports.mjs` against `test_vectors/vectors.json`. The discretisation is the port's (1024 samples per period, 24-point n_eff table, 128-point far-field period); rectangular teeth are sampled as exact cell averages so the fill is not quantised to 1/1024.

Model limits: first-order (Born) radiation from a thin current sheet at the etch mid-depth, local fields from a plane wave through the unetched stack.

TM (version 1.1, task A7): `tm_weights` is a tooth model. The normal field couples through D_x, continuous across the tooth top, with weight ε₂/ε₁ on the cladding-side E_x; the longitudinal E_z couples directly. Wall screening grows with the tooth height, N_v = w/(w + c h) with w = f Λ and c = `tm_screening` = 0.35, calibrated against `rcwa`. Radiation uses TM plane waves (H_y, (1/ε)∂H continuous) and the mode at unit TM power. In the forward-backward (Bragg) coupling the two terms enter with opposite signs, so κ_TM = |ρ| κ_EIM, ρ = (C_v − C_l)/(C_v⁰ + C_l⁰); the coupling's sign is −ρ times the effective-index one (used by `crigf`). The thin limit (h → 0) is exact: ρ = (N²ε₂ − (N² − ε₂)ε₁)/(N²ε₂ + (N² − ε₂)ε₁), −0.53 for TM0 of LN in air. Accuracy against `rcwa` is in `../rcwa/README.md`.

TE accuracy against `rcwa`: radiation within 1–6 % for etch depths up to a third of the core, κ within 1 %.
