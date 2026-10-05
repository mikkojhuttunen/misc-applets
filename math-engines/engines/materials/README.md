# materials

Sellmeier phase index, group index and group-velocity dispersion β2 for SiO₂, Si₃N₄, Si (real part), MgO:LiNbO₃ (e, o), LiTaO₃ (`lt_e`: old approximate fit, ±0.01; `lt_e_bond`, `lt_o_bond`: refits to Bond's table, ±1e-3), KTP (x, y, z), BBO (o, e), AlN (o, e), GaAs, rutile TiO₂ (o, e) and sapphire (o, e). Input: vacuum wavelength in m.

- `refractive_index(material, wavelength)` returns n, n_g and β2 (s²/m). Wavelengths outside a fit's published range (`VALID_RANGE`) are flagged in the assumptions, not rejected.
- `thermal_index(material, wavelength, temperature)` (K) is available for `ln_e_gayer` (5 % MgO:LiNbO₃, extraordinary, Gayer et al. 2008). Use it for phase-matching differences; the absolute dn/dT is not validated and therefore not returned.
- `index(material, wavelength)` is the plain-array helper for other engines.

Anisotropic crystals are one entry per principal axis; choose the axis matching the field polarisation. KTP is room-temperature only (Kato & Takaoka thermo-optic terms not included). Not yet included: Ta₂O₅, GaP, SiON, AlGaAs(x), temperature models for LiTaO₃/KTP/stoichiometric LN (need a source that can be checked).

Version 2.
