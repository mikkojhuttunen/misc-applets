# waveguide_thermal

Steady-state 2D heat conduction in a waveguide cross-section by finite volumes on a graded rectilinear grid (harmonic-mean face conductances, sparse direct solve with scipy). Isothermal substrate bottom (heat sink), convective top (h_top), adiabatic sides; only the half x ≥ 0 is solved.

- `ridge_heating(heat_per_length, core_material, core_width, core_height, slab_material, slab_thickness, box_material, box_thickness, substrate_material, substrate_thickness, clad_material, clad_thickness, domain_half_width, h_top, k_*, resolution, return_field)`: stack substrate / BOX / optional film / ridge or strip, cladding around it. Heat q′ (W/m) uniformly in the ridge. Returns the mean and peak ΔT, R′ = ΔT_core/q′, ΔT at the BOX and substrate tops, the energy time constant τ_E = ∫ρc_pΔT dA / q′, a heat-balance check and, with `return_field=True`, the grid and ΔT map. `k_core`, `k_slab`, `k_box`, `k_substrate`, `k_clad` override the LUT conductivities (thin-film or worst-case values).
- `thermal_shift(heat_per_length, dneff_dT, wavelength, n_group, length, **geometry)`: the same plus Δn_eff, phase over a length and a resonance shift.
- `ridge_dynamics(**geometry)`: response of the ridge temperature to a heat step (backward Euler, time step doubling) with 10/50/90 % times, and the exact frequency response of the discretised model with its −3 dB frequency. A 500 × 220 nm SOI wire on 2 µm BOX gives t₉₀ ≈ 10 µs and f₃dB ≈ 48 kHz.
- `ridge_mode(wavelength, **geometry)`: fundamental scalar mode on a finite-volume grid whose cell edges follow every interface (window around the ridge, substrate excluded).
- `mode_weighted_heating(heat_per_length, wavelength, heat_in_mode, **geometry)`: per-region Γ_r = ∂n_eff/∂n_r, dn_eff/dT of the channel waveguide, the mode-weighted ΔT and Δn_eff under the computed heating, n_g, and the resonance drift. With `heat_in_mode=True` the heat is deposited ∝ |E|² in the ridge.
- `solve_heat`, `conduction_matrix`, `build_ridge` and `graded_axis` are the building blocks for other geometries.

Checks: the mean step-response time against the exact moment of the discretised model (3 %), the scalar mode of a 12 µm wide strip against the slab TE solution, the Γ sum rule Σ Γ_r n_r = n_eff − λ ∂n_eff/∂λ, and the SOI drift (76 pm/K). Also a layered 1D stack against the exact solution (1e-4), a buried square in a homogeneous medium against the image (Carslaw & Jaeger) solution (3 %), grid convergence (2 %), heat balance, linearity, and the closed-form `thermo_optic.strip_thermal_resistance` within ±30 % for SOI, SiN and TFLN stacks.

Conductivities come from the `thermo_optic` LUT and are temperature independent. The model is 2D (uniform heating along z); for a pump that is absorbed along an amplifier use the local q′(z) (largest at the input) or the 3D extension in the TODO list.

Version 2.
