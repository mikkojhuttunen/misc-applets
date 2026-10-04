# opa_chi2

CW χ(2) optical parametric amplification ω_p = ω_s + ω_i in guided modes (Gaussian overlap, no diffraction), with the same equations and normalisation as the χ2 mode of `parametric-amplifier.html`.

Version 1. Functions (Result): `coupling` (idler wavelength, overlap, Γ), `small_signal_gain` (closed-form undepleted-pump gain for any Δk), `coupled_wave` (RK4 with pump depletion and per-wave loss, power profiles along z), `gain_spectrum` (QPM gain vs signal wavelength from `materials` indices and `phase_matching` mismatch). Helpers: `idler_wavelength`, `mode_overlap`, `propagate_normalised` (the integrator in u = a/√F_p0, checked against the applet's `simulate`).

Δk = k_p − k_s − k_i − 2πm/Λ, the `phase_matching` convention. See `spec.yaml` for inputs, outputs and equations.
