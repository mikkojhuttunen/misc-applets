# fiber_mode

Fused-silica step-index fiber: Malitson Sellmeier index, the scalar LP01 mode (exact
eigenvalue equation, Marcuse mode radius) and the phase mismatch it gives three-wave (chi2)
and degenerate four-wave (chi3) mixing, including GVM and β2,s + β2,i.

Conventions: SI units, vacuum wavelengths, constant core index step `delta_n`.
chi2 `dk0 = k_p − k_s − k_i`; chi3 `dk0 = k_s + k_i − 2k_p` (linear part only; add 2γP).
`gvm = β1,i − β1,s` (s/m), `beta2_sum` in s²/m.

Checked against: Malitson index table, the Rudolph–Neumann b(V) approximation, and the
SMF-28 datasheet dispersion D ≈ 17 ps/(nm km) at 1550 nm.

Version history: 1 — first release (physics previously embedded in the parametric-amplifier applet).
