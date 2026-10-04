# phase_matching

Three-wave phase mismatch Δk for ω3 = ω1 + ω2 with arbitrary effective indices, the QPM period of any order, grating strength |d_eff/d| vs duty cycle, and the sinc² response. It does not compute indices itself: feed it indices from `materials`, `slab_waveguide` (and later the rectangular-waveguide and birefringent engines), so modal, birefringent and poled matching all use one mismatch formula.

See `spec.yaml` for inputs, outputs and equations.
