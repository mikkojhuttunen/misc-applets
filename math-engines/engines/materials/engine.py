"""Refractive indices of common photonic materials (Sellmeier fits).

Input: vacuum wavelength in m. The Sellmeier coefficients are tabulated for
wavelength in micrometres, so the conversion happens inside _sellmeier, at the
formula's own boundary; everything returned is dimensionless.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_positive

# (A, B) pairs for n^2 = 1 + sum A λ^2 / (λ^2 - B), λ in µm
_SELLMEIER = {
    "sio2": [(0.6961663, 0.0684043**2), (0.4079426, 0.1162414**2), (0.8974794, 9.896161**2)],
    "si3n4": [(3.0249, 0.1353406**2), (40314.0, 1239.842**2)],
    "ln_e": [(2.2454, 0.01242), (1.3005, 0.0513), (6.8972, 331.33)],
    "ln_o": [(2.4272, 0.01478), (1.4617, 0.05612), (9.6536, 371.216)],
    "si": [(10.6684293, 0.301516485**2), (1.54133408, 1104.0**2)],
}
_SOURCES = {
    "sio2": "I. H. Malitson, JOSA 55, 1205 (1965), fused silica",
    "si3n4": "K. Luke et al., Opt. Lett. 40, 4823 (2015), LPCVD Si3N4",
    "ln_e": "D. E. Zelmon et al., JOSA B 14, 3319 (1997), 5% MgO:LiNbO3, extraordinary",
    "ln_o": "D. E. Zelmon et al., JOSA B 14, 3319 (1997), 5% MgO:LiNbO3, ordinary",
    "si": "Salzberg & Villa (1957) fit, real part only, valid above about 1.2 µm",
    "lt_e": "Approximate single-pole fit to LiTaO3 extraordinary index (±0.01); verify before design use",
}
MATERIALS = tuple(_SOURCES)


def _sellmeier(material: str, wavelength):
    lu = np.asarray(wavelength, dtype=float) * 1e6
    l2 = lu * lu
    if material == "lt_e":
        return np.sqrt(1 + 3.502 * l2 / (l2 - 0.035) - 0.025 * l2)
    n2 = np.ones_like(l2)
    for A, B in _SELLMEIER[material]:
        n2 = n2 + A * l2 / (l2 - B)
    return np.sqrt(n2)


def index(material: str, wavelength):
    """Phase index n(λ) as a plain array (helper for other engines)."""
    require_choice("material", material, MATERIALS)
    require_positive(wavelength=wavelength)
    return _sellmeier(material, wavelength)


def refractive_index(material: str, wavelength) -> Result:
    """Phase index n and group index n_g = n - λ dn/dλ at a vacuum wavelength."""
    n = index(material, wavelength)
    h = 1e-10
    lam = np.asarray(wavelength, dtype=float)
    dndl = (_sellmeier(material, lam + h) - _sellmeier(material, lam - h)) / (2 * h)
    return Result(
        values={"n": n, "n_group": n - lam * dndl},
        units={"n": "", "n_group": ""},
        assumptions=[_SOURCES[material], "Isotropic, lossless, room temperature"],
    )
