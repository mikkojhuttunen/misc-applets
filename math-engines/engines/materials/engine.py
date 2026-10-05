"""Refractive indices of common photonic materials (Sellmeier fits).

Input: vacuum wavelength in m. The Sellmeier coefficients are tabulated for
wavelength in micrometres, so the conversion happens inside the formulas, at their
own boundary; everything returned is dimensionless except the group-velocity
dispersion (s^2/m) and dn/dT (1/K).

Anisotropic crystals are separate entries per principal axis (ln_o/ln_e, ktp_x/y/z, ...):
the caller picks the axis that matches the polarisation of the mode (or the
effective-index weighting done in the waveguide engines).
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_positive

_C = 299792458.0  # m/s

# (A, B) pairs for n^2 = 1 + sum A λ^2 / (λ^2 - B), λ in µm
_SELLMEIER = {
    "sio2": [(0.6961663, 0.0684043**2), (0.4079426, 0.1162414**2), (0.8974794, 9.896161**2)],
    "si3n4": [(3.0249, 0.1353406**2), (40314.0, 1239.842**2)],
    "ln_e": [(2.2454, 0.01242), (1.3005, 0.0513), (6.8972, 331.33)],
    "ln_o": [(2.4272, 0.01478), (1.4617, 0.05612), (9.6536, 371.216)],
    "si": [(10.6684293, 0.301516485**2), (1.54133408, 1104.0**2)],
}
# n^2 = 1 + c0 + sum A λ^2 / (λ^2 - B^2): (c0, [(A, B_um), ...]); the form used by
# Skauli (GaAs), Pastrnak (AlN), Malitson/Dodge (sapphire) and the Bond refits (LiTaO3)
_POLES_SQ = {
    "gaas": (4.372514, [(5.466742, 0.4431307), (0.02429960, 0.8746453), (1.957522, 36.9166)]),
    "aln_o": (2.1399, [(1.3786, 0.1715), (3.861, 15.03)]),
    "aln_e": (2.0729, [(1.6173, 0.1746), (4.139, 15.03)]),
    "sapphire_o": (0.0, [(1.4313493, 0.0726631), (0.65054713, 0.1193242), (5.3414021, 18.028251)]),
    "sapphire_e": (0.0, [(1.5039759, 0.0740288), (0.55069141, 0.1216529), (6.5927379, 20.072248)]),
    "lt_o_bond": (0.0, [(3.5030447406, 0.162397983924), (5.4986874, 15.811749571)]),
    "lt_e_bond": (0.0, [(3.5184224161, 0.162906889986), (3.7507069272, 13.294141074)]),
}
# n^2 = C + sum A / (λ^2 - B) + sum P λ^k : (C, [(A, B_um2), ...], [(P, k), ...]); Kato, Zhang, Devore
_FORM4 = {
    "ktp_x": (3.29100, [(0.04140, 0.03978), (9.35522, 31.45571)], []),
    "ktp_y": (3.45018, [(0.04341, 0.04597), (16.98825, 39.43799)], []),
    "ktp_z": (4.59423, [(0.06206, 0.04763), (110.80672, 86.12171)], []),
    "bbo_o": (2.7359, [(0.01878, 0.01822)], [(-0.01471, 2), (0.0006081, 4), (-0.00006740, 6)]),
    "bbo_e": (2.3753, [(0.01224, 0.01667)], [(-0.01627, 2), (0.0005716, 4), (-0.00006305, 6)]),
    "tio2_o": (5.913, [(0.2441, 0.0803)], []),
    "tio2_e": (7.197, [(0.3322, 0.0843)], []),
}
# Gayer et al. (2008), 5 % MgO:LiNbO3 extraordinary: a1..a6, b1..b4, T0 and 2*273.16 in °C
_GAYER_E = dict(a=(5.756, 0.0983, 0.202, 189.32, 12.52, 1.32e-2), b=(2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4), T0=24.5)

_SOURCES = {
    "sio2": "I. H. Malitson, JOSA 55, 1205 (1965), fused silica",
    "si3n4": "K. Luke et al., Opt. Lett. 40, 4823 (2015), LPCVD Si3N4",
    "ln_e": "D. E. Zelmon et al., JOSA B 14, 3319 (1997), 5% MgO:LiNbO3, extraordinary",
    "ln_o": "D. E. Zelmon et al., JOSA B 14, 3319 (1997), 5% MgO:LiNbO3, ordinary",
    "si": "Salzberg & Villa (1957) fit, real part only, valid above about 1.2 µm",
    "lt_e": "Approximate single-pole fit to LiTaO3 extraordinary index (±0.01); verify before design use",
    "ln_e_gayer": "O. Gayer et al., Appl. Phys. B 91, 343 (2008), 5% MgO:LiNbO3 extraordinary, temperature dependent (T0 = 24.5 °C)",
    "lt_e_bond": "Two-pole Sellmeier fit (this repo) to W. L. Bond, J. Appl. Phys. 36, 1674 (1965), LiTaO3 extraordinary, room temperature; fit residual ≤ 1.0e-3",
    "lt_o_bond": "Two-pole Sellmeier fit (this repo) to W. L. Bond, J. Appl. Phys. 36, 1674 (1965), LiTaO3 ordinary, room temperature; fit residual ≤ 1.1e-3",
    "ktp_x": "K. Kato and E. Takaoka, Appl. Opt. 41, 5040 (2002), KTP n_alpha (x), 20 °C",
    "ktp_y": "K. Kato and E. Takaoka, Appl. Opt. 41, 5040 (2002), KTP n_beta (y), 20 °C",
    "ktp_z": "K. Kato and E. Takaoka, Appl. Opt. 41, 5040 (2002), KTP n_gamma (z), 20 °C",
    "bbo_o": "D. Zhang, Y. Kong, J.-Y. Zhang, Opt. Commun. 184, 485 (2000), BBO ordinary",
    "bbo_e": "D. Zhang, Y. Kong, J.-Y. Zhang, Opt. Commun. 184, 485 (2000), BBO extraordinary",
    "aln_o": "J. Pastrnak and L. Roskovcova, Phys. Stat. Sol. 14, K5 (1966), AlN single crystal, ordinary",
    "aln_e": "J. Pastrnak and L. Roskovcova, Phys. Stat. Sol. 14, K5 (1966), AlN single crystal, extraordinary",
    "gaas": "T. Skauli et al., J. Appl. Phys. 94, 6447 (2003), GaAs, 22 °C",
    "tio2_o": "J. R. Devore, JOSA 41, 416 (1951), rutile ordinary, room temperature",
    "tio2_e": "J. R. Devore, JOSA 41, 416 (1951), rutile extraordinary, room temperature",
    "sapphire_o": "Malitson and Dodge (1972) / Dodge (1986), synthetic sapphire ordinary, 20 °C",
    "sapphire_e": "Malitson and Dodge (1972) / Dodge (1986), synthetic sapphire extraordinary, 20 °C",
}
MATERIALS = tuple(_SOURCES)
# Wavelength range (m) of the published fit; None where it was not recorded. Outside it the formula
# is an extrapolation and refractive_index() says so in its assumptions.
VALID_RANGE = {
    "ln_e_gayer": (0.5e-6, 4.0e-6),
    "lt_e_bond": (0.45e-6, 4.0e-6), "lt_o_bond": (0.45e-6, 4.0e-6),
    "ktp_x": (0.43e-6, 3.54e-6), "ktp_y": (0.43e-6, 3.54e-6), "ktp_z": (0.43e-6, 3.54e-6),
    "bbo_o": (0.64e-6, 3.18e-6), "bbo_e": (0.64e-6, 3.18e-6),
    "aln_o": (0.22e-6, 5.0e-6), "aln_e": (0.22e-6, 5.0e-6),
    "gaas": (0.97e-6, 17e-6),
    "tio2_o": (0.43e-6, 1.53e-6), "tio2_e": (0.43e-6, 1.53e-6),
    "sapphire_o": (0.20e-6, 5.0e-6), "sapphire_e": (0.20e-6, 5.0e-6),
}
THERMAL = ("ln_e_gayer",)


def _gayer_e(lu, temperature_c):
    a, b, T0 = _GAYER_E["a"], _GAYER_E["b"], _GAYER_E["T0"]
    f = (temperature_c - T0) * (temperature_c + T0 + 2 * 273.16)
    l2 = lu * lu
    n2 = a[0] + b[0] * f + (a[1] + b[1] * f) / (l2 - (a[2] + b[2] * f) ** 2) + (a[3] + b[3] * f) / (l2 - a[4] ** 2) - a[5] * l2
    return np.sqrt(n2)


def _sellmeier(material: str, wavelength, temperature_c: float = _GAYER_E["T0"]):
    lu = np.asarray(wavelength, dtype=float) * 1e6
    l2 = lu * lu
    if material == "lt_e":
        return np.sqrt(1 + 3.502 * l2 / (l2 - 0.035) - 0.025 * l2)
    if material == "ln_e_gayer":
        return _gayer_e(lu, temperature_c)
    if material in _POLES_SQ:
        c0, poles = _POLES_SQ[material]
        n2 = 1 + c0 + np.zeros_like(l2)
        for A, B in poles:
            n2 = n2 + A * l2 / (l2 - B * B)
        return np.sqrt(n2)
    if material in _FORM4:
        c, poles, poly = _FORM4[material]
        n2 = c + np.zeros_like(l2)
        for A, B in poles:
            n2 = n2 + A / (l2 - B)
        for P, k in poly:
            n2 = n2 + P * lu**k
        return np.sqrt(n2)
    n2 = np.ones_like(l2)
    for A, B in _SELLMEIER[material]:
        n2 = n2 + A * l2 / (l2 - B)
    return np.sqrt(n2)


def index(material: str, wavelength):
    """Phase index n(λ) as a plain array (helper for other engines)."""
    require_choice("material", material, MATERIALS)
    require_positive(wavelength=wavelength)
    return _sellmeier(material, wavelength)


def _range_note(material: str, wavelength):
    rng = VALID_RANGE.get(material)
    if rng is None:
        return []
    lam = np.asarray(wavelength, dtype=float)
    if np.any(lam < rng[0]) or np.any(lam > rng[1]):
        return [f"WARNING: wavelength outside the published range {rng[0] * 1e6:.2f}-{rng[1] * 1e6:.2f} µm; extrapolated"]
    return []


def _derivs(fn, lam):
    """n, dn/dλ, d²n/dλ² by central differences with a step relative to λ."""
    h = lam * 1e-3
    n0, np_, nm = fn(lam), fn(lam + h), fn(lam - h)
    return n0, (np_ - nm) / (2 * h), (np_ - 2 * n0 + nm) / (h * h)


def refractive_index(material: str, wavelength) -> Result:
    """Phase index n, group index n_g = n - λ dn/dλ and GVD β2 = λ³ n'' / (2π c²) at a vacuum wavelength."""
    n = index(material, wavelength)
    lam = np.asarray(wavelength, dtype=float)
    n0, d1, d2 = _derivs(lambda x: _sellmeier(material, x), lam)
    return Result(
        values={"n": n, "n_group": n0 - lam * d1, "gvd": lam**3 * d2 / (2 * np.pi * _C**2)},
        units={"n": "", "n_group": "", "gvd": "s^2/m"},
        assumptions=[_SOURCES[material], "Isotropic (or a single principal axis), lossless, room temperature"] + _range_note(material, wavelength),
    )


def thermal_index(material: str, wavelength, temperature) -> Result:
    """Phase and group index at temperature (K) for materials with a thermal Sellmeier model (THERMAL).

    Use it for phase-matching differences between wavelengths of the same polarisation (QPM period,
    temperature tuning). It deliberately returns no dn/dT: the Gayer coefficients were fitted to QPM
    tuning data, and their implied absolute drift (about 2.7e-4 1/K at 1550 nm) looks several times larger
    than typical reported thermo-optic coefficients of LiNbO3."""
    require_choice("material", material, THERMAL)
    require_positive(wavelength=wavelength, temperature=temperature)
    lam = np.asarray(wavelength, dtype=float)
    tc = np.asarray(temperature, dtype=float) - 273.15
    n0, d1, _ = _derivs(lambda x: _sellmeier(material, x, tc), lam)
    return Result(
        values={"n": n0, "n_group": n0 - lam * d1},
        units={"n": "", "n_group": ""},
        assumptions=[
            _SOURCES[material],
            "Reliable for index differences between wavelengths (phase matching); the absolute n(T) drift is not validated",
            "Material temperature only; thermal expansion of the waveguide geometry and grating is not included",
        ]
        + _range_note(material, wavelength),
    )
