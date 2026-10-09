# thermo_optic

Thermal and thermo-optic properties of waveguide materials, the heat a guided beam deposits, and how hard a waveguide (e.g. an amplifier) can be pumped before heating shifts its index too far. SI units in and out.

**Look-up table** (`LUT`, exported to [LUT.md](LUT.md) and [thermo_optic_lut.csv](thermo_optic_lut.csv) by `tools/make_thermo_lut.py`): Si, SiO₂, LPCVD Si₃N₄, PECVD SiNx, LiNbO₃ (TFLN, e and o), LiTaO₃ (TFLT, e and o), GaAs, AlₓGa₁₋ₓAs (any x, `algaas_properties`), InP, AlN, amorphous Al₂O₃ film, sapphire, amorphous TiO₂ film, SU-8/polymer and air. Each row: n (Sellmeier from the `materials` engine where one exists), dn/dT with a range, thermal conductivity k with a range (thin films vary a lot with deposition), ρ, c_p, diffusivity, expansion, band gap, β_TPA and n₂ at 1550 nm, a confidence grade and the source. Values are at about 300 K and 1550 nm; the low-confidence rows (PECVD SiNx, LT, LN ordinary, AlN and oxide films) are design placeholders to replace with measurements.

| Function | Returns |
|---|---|
| `material_properties(material, wavelength, al_fraction)` | one LUT row as a Result |
| `index_change(material, delta_T)` | Δn (nominal and LUT range), phase per length, length for a π shift |
| `slab_thermal_shift(wavelength, core, substrate, cladding, thickness, polarization)` | dn_eff/dT of a slab mode, the share from each layer (Γ_i = ∂n_eff/∂n_i), n_g, resonance drift dλ/dT incl. substrate expansion |
| `resonance_shift(wavelength, dneff_dT, n_group, n_eff, alpha_L, delta_T)` | dλ/dT, Δλ, dν/dT |
| `absorbed_heat(power, wavelength, a_eff, loss_abs_db_per_cm, beta_tpa, carrier_lifetime, sigma_fca)` | heat per length q′ from linear absorption, TPA and free-carrier absorption; carrier density; Si free-carrier index change (`si_free_carrier_index`, Soref & Bennett), which has the opposite sign to the thermal one |
| `amplifier_heat_fraction(pump_wavelength, emission_wavelength, quantum_efficiency)` | quantum defect, η_heat = 1 − η_q λ_p/λ_em |
| `strip_thermal_resistance(width, height, box_thickness, ..., clad_material, slab_thickness)` | closed-form R′ (K m/W), within ±30 % of the `waveguide_thermal` solver for the tested stacks |
| `thermal_runaway(R_th, power, ..., T_scale, k_exponent)` | steady ΔT with feedback (absorption α e^(ΔT/T_a), conductivity k ∝ T^-m through the Kirchhoff transform θ = ∫k/k₀ dT) and the runaway threshold (fold of P(ΔT); with linear absorption alone P_th = T_a/(e R′ α); for m > 1 the conduction limit θ < T₀/(m−1)) |
| `pump_budget(R_th, dneff_dT, power, ...)` | ΔT and Δn_eff at the hottest point (input, undepleted) and the largest power within ΔT_max and Δn_max (handles the P, P², P³ terms) |

Useful rule: R′ in K m/W is the temperature rise in K per mW of heat deposited per mm of waveguide.

Worked numbers (from the tests and `waveguide_thermal`):

- 500 × 220 nm SOI wire, 2 µm BOX: R′ ≈ 0.39 K m/W; dλ/dT ≈ 77 pm/K. At 100 mW with τ = 1 ns, free-carrier absorption (4.5 mW/mm) dominates TPA (0.8 mW/mm): ΔT ≈ 2.2 K, and Δn_eff stays below 1e-4 only up to about 60 mW.
- Er:Al₂O₃ strip (2 × 0.4 µm) on 300 nm TFLN, 4.7 µm BOX, air clad: R′ ≈ 0.51 K m/W. With 3 dB/cm pump absorption and η_q = 0.8, 500 mW at 980 nm heats the input by about 9 K (η_heat = 0.49); 1480 nm pumping halves that (η_heat = 0.23).

Not included yet: temperature dependence of dn/dT and k, stress-optic and pyroelectric/photorefractive effects, wavelength dispersion of dn/dT. Pump depletion along an amplifier is in `amplifier_thermal`, phase-matching and resonator detuning in `thermal_detuning`, and worked examples in `examples/pump_heating.py`. See [../../THERMAL_TODO.md](../../THERMAL_TODO.md).

Version 3.
