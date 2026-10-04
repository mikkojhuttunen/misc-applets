# erbium_amplifier

Erbium-doped waveguide amplifier physics in SI units:

- `cross_sections`: smooth model of the Er:Al₂O₃ 1.5 µm absorption band (peak height adjustable) and the McCumber emission cross-section. It is a model, not measured data; replace it with measured spectra for design work.
- `concentration_effects`: upconversion coefficient and quenched-ion share, either fixed or growing with the Er concentration.
- `upper_population`: steady-state N₂ of the two-level rate equation with energy-transfer upconversion.
- `propagate`: co-propagating pump and signal through a doped region given as cells with mode weights |E|²dA/∫|E|²dA (from any mode solver, e.g. `channel_waveguide`), RK4 in z. Returns P(z), the inversion and the integrals that give the probe gain spectrum.
- `probe_gain_ln`: small-signal gain spectrum from those integrals.
- `uniform_amplifier`: one-cell version with overlaps Γ_p, Γ_s and a doped area, for quick estimates and the web calculator.

Gains are natural-log power ratios (dB = 4.343 × ln). Not modelled: ASE, excited-state absorption, partly inverted ion pairs.

JavaScript port: `er-waveguide-amplifier/er-engine.js`, checked against `test_vectors/vectors.json`.

Version 1.
