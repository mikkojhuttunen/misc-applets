# opa_chi3

CW fibre optical parametric amplification by degenerate-pump four-wave mixing 2ω_p = ω_s + ω_i, with pump SPM and signal/idler XPM, the same equations and normalisation as the χ3 mode of `parametric-amplifier.html` (γ_j = γ ω_j/ω_p, so the mixing terms conserve photon number).

Version 1 functions (Result): `coupling` (A_eff, γ, nonlinear length), `small_signal_gain` (closed form, κ = Δβ + 2γP), `coupled_wave` (RK4 with depletion, SPM/XPM and per-wave loss, power profiles along z), `gain_spectrum` (Δβ = β2Ω² + β4Ω⁴/12), `fiber_parameters` and `fiber_gain_spectrum` (exact LP01 Δβ, β2–β4 and γ of a silica step-index fibre via `step_index_fiber`; these need SciPy, imported on first use). Helpers: `idler_wavelength`, `effective_area`, `propagate_normalised` (checked against the applet's `simulate3`).

Not included yet: Raman and Brillouin scattering, polarisation effects, non-degenerate (two-pump) FWM, pulsed pumps.

Version 2 adds the dissipative idler channel (via `idler_loss`): `small_signal_gain`, `gain_spectrum` and `fiber_gain_spectrum` take `alpha_idler` (continuous), `dumps` and `dump_loss_db` (lumped), the spectra also a loss profile (`loss_profile`, `band_center`, `band_width`, `band_edge`, `loss_points`, `loss_acts_on`); `coupled_wave` takes `dumps`, `dump_loss_db` and optional signal/pump dump losses alongside the continuous α_j. Lossless results are reported alongside for comparison.

Δβ = β_s + β_i − 2β_p (Agrawal's convention). See `spec.yaml` for inputs, outputs and equations.
