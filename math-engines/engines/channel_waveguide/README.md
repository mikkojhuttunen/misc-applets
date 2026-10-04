# channel_waveguide

Strip-loaded, rib, ridge and buried channel waveguides, SI units.

- `eim_modes`: effective index method. Vertical stacks are solved with `slab_waveguide.multilayer_neff`, the lateral slab with `slab_waveguide.neff_three_layer`. Higher vertical orders that fall below the outside slab fundamental are flagged `leaky`.
- `fd_modes`: semi-vectorial finite-difference modes on any rectangular grid (non-uniform edges allowed) for a given index map; shift-and-invert eigen-solve with SciPy.
- `fd_fundamental`: uniform-grid convenience wrapper for the fundamental mode and its field share in substrate, film/core, strip and superstrate.
- `region_at`, `region_fractions`: geometry and overlap helpers (region codes 0 substrate, 1 film/core, 2 strip, 3 superstrate).

JavaScript port: the solver in `er-waveguide-amplifier/er-engine.js` (lengths in µm inside the port, as in the other ports), checked against `test_vectors/vectors.json` (EIM indices and FD eigenvalues on a shared explicit grid).

Version 1.
