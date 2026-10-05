# bragg_grating

Coupled-mode κ for rectangular gratings, peak reflectance, bandwidth and effective length, the coupled-mode spectrum, and the exact Abelès transfer matrix of (H L)^N stacks (vectorised over wavelength, complex indices allowed). SI units.

Version 2 adds oblique incidence: `stack_R_oblique` (any layer list, s or p, tilted admittances, TIR and frustrated TIR, renormalised so thick evanescent stacks do not overflow), `cascaded_R` for multi-section mirrors, `TrenchDBR` for the etched air-gap/membrane-tooth mirrors of chip-scale multipass cells (slab TE → p, slab TM → s on the trench walls: `LATERAL_POL`), and `double_resonant_orders` for one periodic DBR on a Bragg condition at two wavelengths.

`coupling_coefficient`, `grating_summary`, `stack_reflectance`, `oblique_reflectance` return Results; `cmt_reflectance`, `stack_R_oblique`, `cascaded_R`, `TrenchDBR`, `double_resonant_orders` are helpers.
