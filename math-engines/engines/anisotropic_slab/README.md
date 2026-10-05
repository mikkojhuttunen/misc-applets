# anisotropic_slab

Guided TE and TM modes of planar waveguides whose layers are anisotropic with principal axes along the slab normal, the transverse direction and the propagation direction. Each layer is given as `(n_normal, n_trans, n_prop)` (a number means isotropic). TE sees only `n_trans`; TM is limited by `n_normal` and weighted by `n_prop`. Transfer-matrix characteristic function, root bracketing on a fine grid.

- `neff_uniaxial_film`: uniaxial film (x-, y- or z-cut, optic axis = crystal z) on isotropic substrate and cladding; spec'd for the web calculator.
- `slab_modes`: arbitrary stacks of anisotropic layers (Python).
- `oriented_indices`, `uniaxial_principal`: cut/propagation orientation helpers for uniaxial and biaxial crystals (KTP: `oriented_indices((n_x, n_y, n_z), "z", "x")`).

Not covered: propagation off a principal axis (TE/TM hybrid modes) and 2D waveguides; use the 2D solver (planned) for those. Checked against the isotropic engine, an analytic symmetric-slab TM relation and an independent finite-difference solver.

Version 1.
