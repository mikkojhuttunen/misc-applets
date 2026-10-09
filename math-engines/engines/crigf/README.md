# crigf

Cavity-resonant integrated grating filter: a second-order grating coupler between two DBRs, fed by a free-space Gaussian beam. The coupler is solved by Kazarinov–Henry coupled-mode theory (forward and backward envelopes with radiative self- and cross-coupling, guided second-order Bragg coupling, propagation loss, beam excitation through the bare-slab local field, BOX interference included). The DBRs and spacers are exact transfer matrices on effective indices. Out-coupling into the reflected and transmitted beams follows by reciprocity from the same coupling constant. Outputs: R into the incident beam mode, T into the substrate beam, guided escape through each DBR, lateral loss at the DBRs, other radiation and absorption, peak circulating power, the coupler radiation loss, and the resonance wavelength and width. SI units.

Version 1. `crigf_response` returns a Result; `CRIGF` (with `response`, `find_resonance` and `infinite_grating`, the plane-wave limit of the coupler used to benchmark it against `rcwa`), `layer_matrix`, `r_t` and `period_slices` are helpers. It builds on `grating_coupler.SurfaceGrating` for the stack, profiles, n_eff tables and plane-wave local fields.

Python reference for `crigfSetup`, `crigfAt` and `findResonance` in `dbr-structures/dbr-engine.js`, checked by `tools/check_js_ports.mjs` against `test_vectors/vectors.json`.

Checks in `test_engine.py`, by routes independent of the RK4 solver:
- the coupler radiation loss equals the `grating_coupler` thin-sheet radiation of the q = 1 orders (to 1e-9), with and without Si or Au handles, at normal and oblique incidence;
- RK4 envelopes against the matrix exponential of the constant-coefficient system in the rotating frame, driven part by Gauss–Legendre quadrature;
- θ → −θ swaps the left and right escape; no etch gives the bare slab; strong DBRs return all guided light;
- passivity, and mode mismatch below 2e-4 for the coupler alone with a matched beam;
- the cavity resonance is Lorentzian in the circulating power.

Limits: effective-index method, first-order radiation from a thin current sheet, TE local fields (TM rough), a Gaussian beam in one transverse dimension (`overlap_y` scales the lateral overlap), lateral DBR losses lumped into `bounce_eta`. No SHG, pump depletion, thermal or Kerr effects here (they stay in the JS port).
