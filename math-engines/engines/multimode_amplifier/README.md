# multimode_amplifier

Highly multimode waveguide amplifiers with mode-dependent gain (MDG), built on the [`ase_noise`](../ase_noise/) covariance core.

| Piece | Functions |
|---|---|
| Mode sets (one polarisation) | `step_index_modes` (LP_lm, Bessel eigenvalue equation), `grin_modes` (LG groups of a parabolic core, g ≤ V/2), `planar_modes` (hard-wall rectangular / planar core) |
| Doping and overlaps | `disk_profile`, `box_profile` (exact cell coverage), `overlaps` → Γ_k |
| Mode-dependent gain | `small_signal_rates`, `linear_channel` (sections with exact gain + spontaneous-emission solution, random coupling unitaries between sections), `coupling_unitary` (Δβ-weighted random Hermitian generator) |
| Diagnostics | `mode_dependent_gain` (eigenmode gains from the singular values of T, MDG, log-gain spread), `modal_noise` (gain and noise figure for a launch into each mode, ASE occupation per mode) |
| Saturation | `saturated_amplifier`: co-pumped two-level rate equations on the transverse grid, with signal and ASE resolved per mode, spatial hole burning, and an optional coupling unitary after every step; returns the channel linearised about the operating point |

The calculator functions `multimode_small_signal` and `saturated_fibre_amplifier` cover the fibre cases. Demo scripts are in [`amplifiers/ase-noise/`](../../../amplifiers/ase-noise/).

Approximations: weakly guiding scalar modes, azimuthally averaged intensities, no off-diagonal gain overlaps between non-degenerate modes, one ASE spectral bin, no backward ASE. Coupled runs cost O(N³) per section, so keep N ≲ 500 there; uncoupled runs handle thousands of modes.

See `spec.yaml` for inputs, outputs and equations.
