# waveguide_thermal

Steady-state 2D heat conduction in a waveguide cross-section by finite volumes on a graded rectilinear grid (harmonic-mean face conductances, sparse direct solve with scipy). Isothermal substrate bottom (heat sink), convective top (h_top), adiabatic sides; only the half x ≥ 0 is solved.

- `ridge_heating(heat_per_length, core_material, core_width, core_height, slab_material, slab_thickness, box_material, box_thickness, substrate_material, substrate_thickness, clad_material, clad_thickness, domain_half_width, h_top, k_*, resolution, return_field)`: stack substrate / BOX / optional film / ridge or strip, cladding around it. Heat q′ (W/m) uniformly in the ridge. Returns the mean and peak ΔT, R′ = ΔT_core/q′, ΔT at the BOX and substrate tops, the energy time constant τ_E = ∫ρc_pΔT dA / q′, a heat-balance check and, with `return_field=True`, the grid and ΔT map. `k_core`, `k_slab`, `k_box`, `k_substrate`, `k_clad` override the LUT conductivities (thin-film or worst-case values).
- `thermal_shift(heat_per_length, dneff_dT, wavelength, n_group, length, **geometry)`: the same plus Δn_eff, phase over a length and a resonance shift.
- `solve_heat(x_edges, y_edges, k, q, h_top)` and `graded_axis(...)` are the building blocks for other geometries.

Checks: a layered 1D stack against the exact solution (1e-4), a buried square in a homogeneous medium against the image (Carslaw & Jaeger) solution (3 %), grid convergence (2 %), heat balance, linearity, and the closed-form `thermo_optic.strip_thermal_resistance` within ±30 % for SOI, SiN and TFLN stacks.

Conductivities come from the `thermo_optic` LUT and are temperature independent. The model is 2D (uniform heating along z); for a pump that is absorbed along an amplifier use the local q′(z) (largest at the input) or the 3D extension in the TODO list.

Version 1.
