# Thermal / thermo-optic engines: next steps

Goal: estimate how hard a waveguide amplifier (Er:Al₂O₃ on TFLN/TFLT/SiN, χ⁽²⁾ OPA in TFLN, χ⁽³⁾ in SiN/AlGaAs/Si) can be pumped before absorption heating detunes it, and what limits it first (ΔT, Δn_eff, phase matching, resonance, runaway).

Done: `thermo_optic` (LUT, Δn, slab dn_eff/dT, heat sources, closed-form R′, pump budget), `waveguide_thermal` (2D finite-volume cross-section solver, dynamics, scalar mode solver), `amplifier_thermal` (heating along an Er amplifier, pump limit), `thermal_detuning` (QPM acceptance under non-uniform heating, ring bistability), `examples/pump_heating.py`.

## 1. Better data in the LUT
- [ ] Replace the low-confidence rows with measured or published values (Moretti 2005 citation corrected to J. Appl. Phys. 98, 036101; a literature search found no checkable LiTaO₃ dn/dT at 1550 nm, so the LT rows stay low-confidence): PECVD SiNx (vs N/Si ratio), LiTaO₃ e/o dn/dT (congruent vs stoichiometric), LN ordinary dn/dT, AlN and amorphous Al₂O₃/TiO₂ films, thin-film thermal conductivities (SiN, Al₂O₃, TFLN in-plane vs cross-plane).
- [ ] Add Ta₂O₅, GaP, SiC (4H), diamond, Ge, chalcogenide (As₂S₃), Hydex/SiON, BCB, HSQ, InGaAsP (vs composition), KTP.
- [ ] Temperature-dependent models instead of the linear dn/dT: Si (Frey 2006, Komma 2012), LN e/o (Moretti 2005, Schlarb & Betzler, Gayer 2008 already in `materials`), LT (Bruner 2003), GaAs/AlGaAs (Gehrsitz 2000 with T), SiN. Also k(T) (Si ∝ T^-1.3) for ΔT above ~50 K.
- [ ] Wavelength dispersion of dn/dT (strong near the band edge for GaAs, AlGaAs, InP, Si) so 775/980 nm pumps get their own values.
- [ ] Thin-film size effect for k of the Si device layer (220 nm SOI ≈ 60–100 W/m/K) and of TFLN.
- [ ] Stress-optic (photoelastic) coefficients and thermal-expansion mismatch stress (film on Si), which add to dn/dT in thin films.
- [ ] Tensor dn/dT for anisotropic crystals; LN/LT pyroelectric and photorefractive (green-induced IR absorption, GRIIRA) as separate warnings or models.

## 2. Heat sources
- [x] Longitudinal profile q′(z): pump depletion, signal growth, background absorption (`amplifier_thermal`, same rate equations as `er-waveguide-amplifier.html` with overlap-averaged intensities). Still open: per-cell mode-profile version and ASE.
- [x] Er-specific heat: 980 nm vs 1480 nm quantum defect, upconversion, non-radiative decay (η_rad), quenched ions. Still open: excited-state absorption, concentration-dependent C_up and f_q as in the applet.
- [ ] Three-photon absorption (TFLN, AlGaAs below the half-gap), Urbach-tail TPA in AlGaAs near x ≈ 0.18, defect/OH absorption in oxides near 1.4 µm, H-bond (N–H) absorption at 1520 nm in PECVD SiN.
- [x] Free-carrier dispersion of Si (`si_free_carrier_index`, `absorbed_heat` → `dn_fc_si`). GaAs still open.
- [ ] Split measured propagation loss into absorption vs scattering (only absorption heats); helper with typical splits per platform.

## 3. Thermal solver
- [x] Optical-mode-weighted temperature and heat ∝ |E|² in the ridge (`mode_weighted_heating`, scalar FV mode solver). Still open: semi-vectorial/vectorial TE/TM modes (Si wires differ by up to ~10 %).
- [x] Self-consistent loop T → n(T) → mode → heat → k(T) → T (`self_consistent_heating`): thermal lens (exact vs first-order Δn_eff, A_eff, centroid) and runaway; lumped threshold model `thermo_optic.thermal_runaway` (absorption e^(ΔT/T_a), k ∝ T^-m via the Kirchhoff transform), agreeing with the 2D loop to 1 %. Still open: temperature-dependent TPA/FCA lifetime, runaway dynamics in time.
- [x] Transient solver: step response (backward Euler) and exact frequency response, f₃dB (`ridge_dynamics`). Still open: arbitrary pulse trains.
- [x] Axial heat flow: exact R(κ) by Fourier transform along z and ΔT(z) for any q′(z) (`axial_transfer`, `temperature_along_z`). Result: the axial length is ~µm, so the local 2D R′ is accurate for amplifiers. Still open: package and heat-sink models (thermal paste, TEC), finite chip width.
- [x] Heater P_π and thermal crosstalk at a given pitch (`heater_tuning`). Still open: isolation trenches and undercut, the heater metal's own conductance.
- [x] Temperature-dependent k per region (iteration in `self_consistent_heating`; checked against the Kirchhoff transform for a uniform exponent).
- [x] JS port `waveguide-thermal/thermal-engine.js` (heat solve, scalar mode, mode-weighted heating, runaway, Er heating, QPM, ring), worker-friendly, checked against the `thermal` vectors. Not ported: self-consistent loop, dynamics, axial transfer, heater, athermal design.

## 4. Optical consequences and limits
- [x] QPM temperature acceptance in TFLN and the chirped Δk from z-dependent heating, with and without retuning (`qpm_thermal`). Still open: TFLT (needs a temperature Sellmeier for LT), waveguide n_eff instead of bulk, high-gain OPA.
- [x] Ring thermal bistability threshold and on-resonance heating (`ring_thermal_bistability`). Still open: thermal locking dynamics, OPO threshold shift.
- [ ] MZI and grating devices: phase drift vs pump power, Bragg-wavelength drift of DBR mirrors (`bragg_grating`).
- [ ] Gain-spectrum shift of Er with temperature (cross-section temperature dependence) for the Er amplifier.
- [x] Mode-dependent dn_eff/dT for channel waveguides (Γ_r per region from the 2D mode). Athermal design helper `athermal_design` (bisection on width, height, film or BOX thickness for dλ/dT = 0).
- [ ] Damage and stability thresholds: photorefraction (LN at visible pumps), polymer/SU-8 degradation temperature, Er clustering.

## 5. Front end
- [x] `pump-heating-planner.html`: platform presets and geometry; cross-section ΔT and mode maps; R′, dn_eff/dT, dλ/dT; ΔT against power with feedback up to the runaway threshold; P_max from the ΔT and Δn budgets; Er amplifier ΔT(z). Still open: QPM and ring panels, LUT browser with ranges and sources.
- [x] `er-waveguide-amplifier.html`: a peak-heating tile computed from the applet's own per-cell rate equations (heat = absorbed − radiated + absorbing background) with the closed-form R′. Still open: a thermal pump limit and ΔT(z) plot there.
- [x] Worked-example script `examples/pump_heating.py` (Er:Al₂O₃ on TFLN at 980/1480 nm, Si and AlGaAs wires, TFLN OPA, SiN ring). The engines are in `web/engines_index.json`; the Pyodide page itself has not been tried with them.

## 6. Validation
- [x] PPLN 1550 nm SHG acceptance against a measured 40 mm waveguide (1.98 K, arXiv:2607.13215): model ~10 K·cm, within ~20 %.
- [x] Heater P_π against published values: SOI 19 mW (published 20–25 mW), thick SiN ~120 mW. Still open: measured self-heating resonance shifts, TFLN heaters.
- [ ] Cross-check against an independent FEM (e.g. FEniCS/Elmer) for one SOI, one TFLN and one SiN cross-section.

## Findings so far (examples/pump_heating.py)
- Er:Al₂O₃ strip on TFLN: once the 980 nm pump bleaches the Er absorption, Er adds little heat. The absorbing part of the background loss and quenched ions set the temperature, and 5 K at the input needs several watts. Phase-matched or resonant devices on the same chip will feel it long before the amplifier does.
- Si wires: free-carrier absorption heating grows as P³; with τ = 1 ns, Δn_eff reaches 1e-4 at about 60 mW. A carrier-sweeping p-i-n (10 ps) roughly doubles that. Al₀.₂GaAs (no TPA) stays below 1e-4 to about 450 mW.
- TFLN OPA with 0.3 dB/cm pump absorption: a watt-level 775 nm pump costs about 25 % of the phase-matching efficiency unless the chip temperature is retuned. Retuning recovers almost all of it because the heating decays slowly over 10 mm.
- SiN rings (Q ~ 1e6) become thermally bistable at sub-mW powers.
