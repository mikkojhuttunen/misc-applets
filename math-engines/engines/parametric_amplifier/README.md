# parametric_amplifier

Fiber optical parametric amplification:

* `coupling` — gain coefficient Γ from d_eff (χ2) or n2 (χ3) and Gaussian mode sizes.
* `small_signal_gain` — exact undepleted-pump gain with continuous or lumped idler loss and
  optional signal loss from a real filter (2×2 matrix exponential), vectorised over the mismatch.
* `amplify` — RK4 coupled-wave solution with pump depletion, SPM/XPM (χ3), idler/signal/pump
  loss, idler dumps and optional SBS/SRS (uses `stimulated_scattering`).
* `gain_bandwidth` — signal and idler −3 dB width in rad/s, Hz and m for a Taylor phase mismatch.

The fiber dispersion that feeds the mismatch comes from `step_index_fiber.phase_mismatch`.

Conventions: SI units; losses are power coefficients in 1/m and dump transmissions are power
ratios (dB conversion happens in the front end). For χ3, `phase_mismatch` is the net
κ = Δβ + 2γP. Gain is symmetric in the sign of the mismatch.

JavaScript port: `amplifiers/parametric-amplifiers/opa_engine.js`, checked against
`test_vectors/vectors.json` by `tools/check_js_ports.mjs`.

Version 1.
