# rcwa

Rigorous coupled-wave analysis (Fourier modal method) of 1D-periodic multilayer gratings, TE and TM, with Li's inverse rule for TM. Plane-wave diffraction efficiencies per order; guided and leaky Bloch modes (complex k_x) by transverse resonance, det(I − R_up R_down) = 0 against a fixed reference admittance, solved by Muller iteration; radiation loss α = 2 k0 Im N and the up/down split of the radiated power; first-order DBR coupling κ and Bragg wavelength from the guided band outside the stop band. Stable admittance recursion, complex indices allowed. `surface_grating_stack` turns a `grating_coupler.SurfaceGrating` into a layer stack (staircase for non-rectangular profiles). SI units; x along the grating, z up.

Version 1. `grating_leaky_mode`, `dbr_coupling` and `band_edge` return Results; `Stack`, `RCWA` (`diffraction`, `leaky_mode`, `bloch_mode`, `mode_function`; the wavelength may be complex), `surface_grating_stack`, `bragg_band`, `resonance_mode` (complex-frequency mode at a real Bloch wavevector, Q) and `track_band` are helpers.

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

Band edges of second-order gratings at normal incidence (Γ point), rigorous complex-frequency modes against the coupled-mode prediction (`crigf.CRIGF.band_edge_modes`): rectangular teeth, f = 0.5: bright-mode Q 8632 vs 8495 (20 nm etch), 1394 vs 1352 (50 nm); the dark mode is a bound state in the continuum (Q > 1e9, and Q ∝ 1/k_x² off Γ). f = 0.3, 50 nm: band gap 4.29 vs 4.70 nm, bright Q 2116 vs 2289. A sawtooth (asymmetric) tooth breaks the protection: both modes radiate (Q 6150 and 9051 rigorous, 6828 and 7602 coupled-mode). At f = 0.5 the effective-index N₂ vanishes, so the coupled-mode gap is zero; the rigorous gap is 0.02 nm (20 nm etch) and 0.19 nm (50 nm). The bright mode is a pole of the rigorous reflection coefficient at the same complex wavelength.

Not covered: finite (aperiodic) structures such as a whole CRIGF with DBRs and a Gaussian beam; that needs a supercell with absorbing boundaries or a 2D FDTD/FEM solver.
