"""Crystal data for TFLN and TFLT: oriented index tensors and second-order coefficients with Miller scaling.

Indices come from math-engines `materials` (congruent LN: Zelmon 1997; 5 % MgO:LN: Zelmon 1997; LT: refits
to Bond 1965). Waveguide frame: x lateral (in the chip plane, across the guide), y vertical (film normal),
z propagation. Cuts:
  x-cut, propagation along crystal y: lateral = crystal z (n_e), vertical = crystal x (n_o)
  z-cut, propagation along crystal x: lateral = crystal y (n_o), vertical = crystal z (n_e)
Type-0 interactions only (all waves in the same polarisation): x-cut TE and z-cut TM use d33; z-cut TE
(E along crystal y) uses d22; x-cut TM (E along crystal x) has d11 = 0 for class 3m.
"""
from __future__ import annotations

import numpy as np

from engines.materials import engine as mat

CRYSTALS = {
    "ln": ("ln_o_cong", "ln_e_cong"),
    "ln_mgo": ("ln_o", "ln_e"),
    "lt": ("lt_o_bond", "lt_e_bond"),
}
CUTS = {"x": "y", "z": "x"}                      # cut -> propagation axis used here

# d coefficients at the reference SHG pump wavelength (m/V). d33: I. Shoji et al., JOSA B 14, 2268 (1997)
# (LiNbO3 25.2, LiTaO3 13.8 pm/V at 1064 nm); MgO:LN taken equal to congruent LN within Shoji's error.
# d22: R. C. Miller et al. / Roberts IEEE JQE 28, 2057 (1992) compilation, ~2.1 pm/V for LN; LT d22 is
# small and uncertain (set to 0 here). Treat d22 values as approximate.
D_TABLE = {
    "ln": {"d33": 25.2e-12, "d22": 2.1e-12, "lambda_ref": 1.064e-6},
    "ln_mgo": {"d33": 25.0e-12, "d22": 2.1e-12, "lambda_ref": 1.064e-6},
    "lt": {"d33": 13.8e-12, "d22": 0.0, "lambda_ref": 1.064e-6},
}
SUBSTRATES = {"sio2": "sio2", "sapphire": "sapphire_o"}


def principal(material, wavelength):
    """(n_o, n_e) at the vacuum wavelength."""
    o, e = CRYSTALS[material]
    return float(mat.index(o, wavelength)), float(mat.index(e, wavelength))


def film_tensor(material, cut, wavelength):
    """(n_lateral, n_vertical, n_propagation) of the film in the waveguide frame."""
    n_o, n_e = principal(material, wavelength)
    if cut == "x":
        return n_e, n_o, n_o
    if cut == "z":
        return n_o, n_e, n_o
    raise ValueError(f"cut must be 'x' or 'z', got {cut!r}")


def isotropic(name, wavelength) -> float:
    if name == "air":
        return 1.0
    return float(mat.index(SUBSTRATES.get(name, name), wavelength))


def coefficient(material, cut, polarization):
    """Name of the type-0 coefficient and whether it uses the extraordinary index (for Miller scaling)."""
    if (cut, polarization) in (("x", "TE"), ("z", "TM")):
        return "d33", True
    if (cut, polarization) == ("z", "TE"):
        return "d22", False
    return None, False


def miller_d(material, cut, polarization, wavelength):
    """Type-0 SHG coefficient at pump wavelength λ by Miller's rule from the tabulated reference:
    d ∝ χ(λ)² χ(λ/2), χ = n² - 1 of the index the fields see."""
    name, extra = coefficient(material, cut, polarization)
    if name is None:
        return 0.0
    row = D_TABLE[material]
    idx = CRYSTALS[material][1 if extra else 0]
    chi = lambda lam: float(mat.index(idx, lam)) ** 2 - 1
    lr = row["lambda_ref"]
    return row[name] * (chi(wavelength) ** 2 * chi(wavelength / 2)) / (chi(lr) ** 2 * chi(lr / 2))
