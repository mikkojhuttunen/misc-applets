# idler_loss

Dissipative idler channel for parametric amplifiers, shared by `opa_chi2` and `opa_chi3`, following the "Idler dissipation" panel of `parametric-amplifier.html`.

- **Continuous**: distributed idler power loss α_i (1/m).
- **Lumped**: N idler dumps at z = kL/(N+1), each attenuating the idler by a set number of dB, with lossless segments in between.
- Both can act together. A spectral weight w(λ_i) — `flat`, `pass` (loss inside a band), `stop` (lossless notch) or `custom` (monotone cubic through points) — scales α_i and the dump dB at each idler wavelength. With `loss_acts_on="all"` the curve also attenuates the signal (a real filter rather than an ideal idler-selective loss).

Version 1. Functions (Result): `linear_gain` (exact undepleted-pump gain via a 2×2 matrix exponential per segment, lossless comparison, adiabatic Lorentzian rate), `loss_spectrum`. Helpers: `loss_weight`, `propagate_linear` (checked against the applet's `analyticOut`), `amplifier_with_loss` (gain arrays for the OPA spectra), `db_to_alpha`.

The full nonlinear models with these losses are `opa_chi2.coupled_wave` and `opa_chi3.coupled_wave` (`alpha_idler`, `dumps`, `dump_loss_db`). See `spec.yaml` for inputs, outputs and equations.
