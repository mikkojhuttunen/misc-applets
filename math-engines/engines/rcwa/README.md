# rcwa

Rigorous coupled-wave analysis (Fourier modal method) of 1D-periodic multilayer gratings, TE and TM, with Li's inverse rule for TM. Plane-wave diffraction efficiencies per order; guided and leaky Bloch modes (complex k_x) by transverse resonance, det(I − R_up R_down) = 0 against a fixed reference admittance, solved by Muller iteration; radiation loss α = 2 k0 Im N and the up/down split of the radiated power; first-order DBR coupling κ and Bragg wavelength from the guided band outside the stop band. Stable admittance recursion, complex indices allowed. `surface_grating_stack` turns a `grating_coupler.SurfaceGrating` into a layer stack (staircase for non-rectangular profiles). SI units; x along the grating, z up.

Version 1. `grating_leaky_mode` and `dbr_coupling` return Results; `Stack`, `RCWA` (`diffraction`, `leaky_mode`, `bloch_mode`, `mode_function`), `surface_grating_stack` and `bragg_band` are helpers.

Checks in `test_engine.py`: uniform stacks equal the oblique transfer matrix (s and p); lossless gratings conserve energy to 1e-10 and converge in the orders; slab modes equal `slab_waveguide`; a subwavelength lamellar layer converges to the uniaxial effective medium of `anisotropic_slab` as (Λ/λ)² (TE and TM, which tests the factorisation rule).

## Benchmark of the fast models

Stack: substrate 1.444 | core 2.138, 600 nm | air, λ = 1550 nm, rectangular teeth, f = 0.5.

Thin-sheet radiation (`grating_coupler`), second-order coupler detuned by 5 %, α rigorous / α thin sheet:

| etch h | 5 nm | 20 nm | 50 nm | 100 nm | 200 nm |
|---|---|---|---|---|---|
| TE | 0.996 | 0.984 | 0.969 | 0.964 | 0.944 |
| TM | 10.1 | 9.5 | 8.6 | 7.8 | 7.8 |

TE up/down split agrees to 0.002 up to 100 nm. With a BOX on a handle the thin sheet follows the ~10× BOX interference swing of α within ~6 % (Si) and ~20 % (gold). TM is wrong by about 10×: its radiation uses TE expressions (`dbr-structures/TASKLIST.md`, A7).

First-order DBR κ, rigorous / effective-index: TE 1.000 (5 nm), 0.999 (20 nm), 0.997 (50 nm), 0.990 (100 nm); TM 0.51, 0.45, 0.33, 0.15 for the same depths. The effective-index method treats the etched layer as isotropic, but for TM the field component along the guide crosses the tooth walls (harmonic-mean ε), so κ is overestimated 2–7×.

Coupled-mode coupler (`crigf.CRIGF.infinite_grating`, plane-wave limit of the CRIGF coupler) against the rigorous guided-mode resonance, 20 nm etch: linewidth within 3 % (normal incidence, bright band-edge mode) and 1 % (2°), peak reflectance 1 in both, resonance 0.12 nm short of the rigorous one (effective-index n_eff); at 50 nm etch linewidth within 7 %, offset 0.9 nm.

Not covered: finite (aperiodic) structures such as a whole CRIGF with DBRs and a Gaussian beam; that needs a supercell with absorbing boundaries or a 2D FDTD/FEM solver.
