# opa_chi2

CW χ(2) optical parametric amplification ω_p = ω_s + ω_i in guided modes (Gaussian overlap, no diffraction), with the same equations and normalisation as the χ2 mode of `parametric-amplifier.html`.

Version 1 functions (Result): `coupling` (idler wavelength, overlap, Γ), `small_signal_gain` (closed-form undepleted-pump gain for any Δk), `coupled_wave` (RK4 with pump depletion and per-wave loss, power profiles along z), `gain_spectrum` (QPM gain vs signal wavelength from `materials` indices and `phase_matching` mismatch). Helpers: `idler_wavelength`, `mode_overlap`, `propagate_normalised` (the integrator in u = a/√F_p0, checked against the applet's `simulate`).

Version 2 adds the dissipative idler channel (via `idler_loss`): `small_signal_gain` and `gain_spectrum` take `alpha_idler` (continuous), `dumps` and `dump_loss_db` (lumped), and the spectrum also a loss profile (`loss_profile`, `band_offset`, `band_width`, `band_edge`, `loss_points`, `loss_acts_on`); `coupled_wave` takes `dumps`, `dump_loss_db` and optional signal/pump dump losses alongside the continuous α_j. Lossless results are reported alongside for comparison.

Δk = k_p − k_s − k_i − 2πm/Λ, the `phase_matching` convention. See `spec.yaml` for inputs, outputs and equations.
