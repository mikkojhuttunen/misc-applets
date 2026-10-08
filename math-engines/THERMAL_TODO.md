# Thermal / thermo-optic engines: next steps

Goal: estimate how hard a waveguide amplifier (Er:Al₂O₃ on TFLN/TFLT/SiN, χ⁽²⁾ OPA in TFLN, χ⁽³⁾ in SiN/AlGaAs/Si) can be pumped before absorption heating detunes it, and what limits it first (ΔT, Δn_eff, phase matching, resonance, runaway).

Done in this step: `thermo_optic` (LUT, Δn, slab dn_eff/dT, heat sources, closed-form R′, pump budget) and `waveguide_thermal` (2D finite-volume cross-section solver).

## 1. Better data in the LUT
- [ ] Replace the low-confidence rows with measured or published values: PECVD SiNx (vs N/Si ratio), LiTaO₃ e/o dn/dT (congruent vs stoichiometric), LN ordinary dn/dT, AlN and amorphous Al₂O₃/TiO₂ films, thin-film thermal conductivities (SiN, Al₂O₃, TFLN in-plane vs cross-plane).
- [ ] Add Ta₂O₅, GaP, SiC (4H), diamond, Ge, chalcogenide (As₂S₃), Hydex/SiON, BCB, HSQ, InGaAsP (vs composition), KTP.
- [ ] Temperature-dependent models instead of the linear dn/dT: Si (Frey 2006, Komma 2012), LN e/o (Moretti 2005, Schlarb & Betzler, Gayer 2008 already in `materials`), LT (Bruner 2003), GaAs/AlGaAs (Gehrsitz 2000 with T), SiN. Also k(T) (Si ∝ T^-1.3) for ΔT above ~50 K.
- [ ] Wavelength dispersion of dn/dT (strong near the band edge for GaAs, AlGaAs, InP, Si) so 775/980 nm pumps get their own values.
- [ ] Thin-film size effect for k of the Si device layer (220 nm SOI ≈ 60–100 W/m/K) and of TFLN.
- [ ] Stress-optic (photoelastic) coefficients and thermal-expansion mismatch stress (film on Si), which add to dn/dT in thin films.
- [ ] Tensor dn/dT for anisotropic crystals; LN/LT pyroelectric and photorefractive (green-induced IR absorption, GRIIRA) as separate warnings or models.

## 2. Heat sources
- [ ] Longitudinal profile q′(z): pump depletion along the amplifier (couple to the rate-equation solver of `er-waveguide-amplifier.html`), signal growth, background absorption; the input is the hot spot.
- [ ] Er-specific heat: 980 nm vs 1480 nm quantum defect, cooperative upconversion and excited-state absorption, concentration quenching (fraction of quenched ions turns all absorbed pump into heat).
- [ ] Three-photon absorption (TFLN, AlGaAs below the half-gap), Urbach-tail TPA in AlGaAs near x ≈ 0.18, defect/OH absorption in oxides near 1.4 µm, H-bond (N–H) absorption at 1520 nm in PECVD SiN.
- [ ] Free-carrier dispersion (Si, GaAs) alongside free-carrier absorption: its Δn has the opposite sign to the thermal Δn.
- [ ] Split measured propagation loss into absorption vs scattering (only absorption heats); helper with typical splits per platform.

## 3. Thermal solver
- [ ] Optical-mode-weighted temperature: weight ΔT by |E|² from `waveguide-core` (2D FD mode solver) instead of the ridge mean; heat source distributed as the absorbed mode intensity, not uniform.
- [ ] Self-consistent loop: T → n(T) → mode → absorption/overlap → heat → T (thermal lensing, mode shift, thermal runaway in Si with FCA).
- [ ] Transient solver (implicit time stepping) for pulsed/modulated pumps, thermal bandwidth and τ for thermal locking; compare with τ_E.
- [ ] 3D / quasi-3D: q′(z) along the waveguide with axial conduction; finite chip size, heat sink and package (thermal paste, TEC), top convection.
- [ ] Neighbouring heaters and waveguides (thermal crosstalk), heater design (metal heater above cladding: efficiency in mW/π).
- [ ] Temperature-dependent k in the solver (Kirchhoff transform or iteration).
- [ ] JS port of `solve_heat` / `ridge_heating` (Web Worker) for the applets, checked against `test_vectors/vectors.json`.

## 4. Optical consequences and limits
- [ ] QPM / phase-matching temperature acceptance in TFLN/TFLT: ΔT_FWHM from d(Δk)/dT using `materials.thermal_index` and `qpm_shg`; maximum χ⁽²⁾ OPA pump power before the heated input walks out of phase matching (with the z-dependent heating, i.e. a chirped Δk).
- [ ] Ring/resonator amplifiers and OPOs: thermal bistability (triangular resonances), thermal locking, maximum intracavity power for a given detuning tolerance.
- [ ] MZI and grating devices: phase drift vs pump power, Bragg-wavelength drift of DBR mirrors (`bragg_grating`).
- [ ] Gain-spectrum shift of Er with temperature (cross-section temperature dependence) for the Er amplifier.
- [ ] Mode-dependent dn_eff/dT for channel waveguides (2D mode solver Γ_i per material) instead of the slab approximation; athermal-design helper (cladding thickness to null dλ/dT).
- [ ] Damage and stability thresholds: photorefraction (LN at visible pumps), polymer/SU-8 degradation temperature, Er clustering.

## 5. Front end
- [ ] Applet "Pump heating planner": pick platform and geometry, see R′, ΔT(z), Δn_eff, phase-matching/resonance detuning and P_max for each limit; LUT browser with ranges and sources.
- [ ] Hook into `er-waveguide-amplifier.html`: show the input-facet temperature rise and the thermal pump limit next to the gain results.
- [ ] Add the new engines to the Pyodide web calculator check (already in `web/engines_index.json`) and a worked-example notebook/script for TFLN Er amplifiers and Si/AlGaAs χ⁽³⁾ amplifiers.

## 6. Validation
- [ ] Compare solver R′ and τ with published heater efficiencies (SOI ~20–25 mW/π, TFLN ~ 100s mW/π without undercut) and measured self-heating resonance shifts.
- [ ] Cross-check against an independent FEM (e.g. FEniCS/Elmer) for one SOI, one TFLN and one SiN cross-section.
