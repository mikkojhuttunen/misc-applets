# membrane_mode

Guided slab mode of a free-standing membrane (gas | core | gas) and the quantities an evanescent gas sensor needs: effective and group index, absorption factor Γ (α_mode = Γ α_gas, from the field integrals, validated by quadrature), cladding decay constant and 1/e penetration depth, a relative surface-field weight for roughness-scattering scaling, and the evanescent gas volume. Core index from the `materials` Sellmeier fits or a constant. SI units.

Version 1. `evanescent_mode` and `evanescent_volume` return Results; `solve_mode` (→ `MembraneMode` dataclass), `field_profile` and `roughness_scaled_alpha` are helpers.

TM Γ exceeds 1 near cut-off (the E field jumps by n_core²/n_clad² at the surface). `E_edge2` is a proxy, comparable between thicknesses and materials; it is not an absolute loss.
