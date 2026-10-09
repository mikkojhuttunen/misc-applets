# thermal_detuning

What heating does to phase matching and to resonators.

- `qpm_thermal(pump_wavelength, signal_wavelength, length, dT_in, decay_length, temperature, alpha_L)` covers type-0 QPM in 5 % MgO:LiNbO₃ (Gayer temperature-dependent Sellmeier, all waves extraordinary; SHG when pump = signal/2). It returns the bulk period, the idler wavelength, dΔk/dT (including expansion of the poling period), the temperature acceptance FWHM = 5.566/(L|dΔk/dT|), and the low-gain efficiency factor for an exponentially decaying heating ΔT(z) = ΔT_in e^(−z/ℓ), with the chip at its original temperature and after the best uniform retuning.
- `phase_matching_factor(z, dk)` and `best_retuned_factor(z, dk)` take any δk(z), for example `dDk_dT * dT` from `amplifier_thermal`.
- `ring_thermal_bistability(wavelength, ring_length, n_group, Q_intrinsic, Q_coupling, absorbing_fraction, R_th, dneff_dT, power)` uses coupled-mode theory with absorption heating (ring resistance R′/L). It returns the bistability threshold P_th = κ³/(3√3 g κ_e), the absorbed power, ΔT and resonance shift when the laser keeps the ring on resonance, and the power buildup. `ring_energy_roots` gives the one or three steady states at a given detuning.

Checks: PPLN periods for 1550 nm and 1064 nm SHG; the 1 cm acceptance is about 10 K, against a measured 1.98 K for a 40 mm waveguide (arXiv:2607.13215); sinc² recovered for uniform heating; the threshold reproduced by counting the cubic's roots; critical-coupling energy balance and buildup F/π.

Limits: bulk indices (a thin-film period is shorter, but dΔk/dT is close to the bulk value); undepleted, low-gain phase matching; single-mode ring with steady-state heating (no thermal oscillations or Kerr/photorefractive terms).

Version 1.
