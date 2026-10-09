# grating_coupler

Surface-etched gratings on a slab waveguide, by the effective-index method: tooth profiles (rectangular, trapezoidal, rounded, sinusoidal, triangular, sawtooth), the local n_eff table, Fourier coupling coefficients κ_m of any profile, out-of-plane radiation per diffraction order and channel (cladding, substrate or Si handle through the BOX; a gold handle closes the downward channel), directionality, emission angles, the guided Bragg order, and the far field of N periods. SI units, angles in rad.

Version 1. `profile_coupling` and `grating_radiation` return Results; `SurfaceGrating`, `profile_function`, `fourier_coefficient` and `slab_mode` are helpers. `SurfaceGrating` takes indices as numbers or callables n(λ) and a complex handle index.

Python reference for `profileFn`, `sampleProfile`, `tableAt`, `modeParams`, `planeWave`, `radiation` and `farField` in `dbr-structures/dbr-engine.js`, checked by `tools/check_js_ports.mjs` against `test_vectors/vectors.json`. The discretisation is the port's (1024 samples per period, 24-point n_eff table, 128-point far-field period); rectangular teeth are sampled as exact cell averages so the fill is not quantised to 1/1024.

Model limits: first-order (Born) radiation from a thin current sheet at the etch mid-depth, local fields from a TE plane wave through the unetched stack. TM uses the TE expressions with the H_y profile and is rough. Expect agreement with rigorous solvers within a factor of a few for shallow gratings; an RCWA benchmark is planned (`dbr-structures/TASKLIST.md`, A3).
