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


def miller_d3(material, cut, polarization, wavelength_signal, wavelength_idler):
    """Type-0 coefficient for ω_p = ω_s + ω_i by Miller's rule: d ∝ χ(λ_s) χ(λ_i) χ(λ_p), referenced to
    the tabulated SHG value. Reduces to miller_d for λ_s = λ_i."""
    name, extra = coefficient(material, cut, polarization)
    if name is None:
        return 0.0
    row = D_TABLE[material]
    idx = CRYSTALS[material][1 if extra else 0]
    chi = lambda lam: float(mat.index(idx, lam)) ** 2 - 1
    lr = row["lambda_ref"]
    lp = 1 / (1 / wavelength_signal + 1 / wavelength_idler)
    return row[name] * chi(wavelength_signal) * chi(wavelength_idler) * chi(lp) / (chi(lr) ** 2 * chi(lr / 2))


# Deposited loading or cladding layers (isotropic). si3n4: LPCVD stoichiometric fit (Luke 2015); PECVD SiNx and
# amorphous/sputtered SiC vary with recipe: "sinx" uses the Si3N4 dispersion scaled to n = 1.95 at 1550 nm and
# "sic" a constant 2.55 (amorphous SiC near 1550 nm). Pass Ridge(load_index=...) for a measured value.
DEPOSITED = {"si3n4": "si3n4", "sinx": "sinx", "sic": "sic", "sio2": "sio2", "al2o3": "sapphire_o"}


def deposited(name, wavelength) -> float:
    if name == "sic":
        return 2.55
    if name == "sinx":
        return float(mat.index("si3n4", wavelength)) * 1.95 / float(mat.index("si3n4", 1.55e-6))
    return float(mat.index(DEPOSITED[name], wavelength))


def absorption_notes(wavelengths, substrate="sio2", clad="air") -> list[str]:
    """Warnings for wavelengths where the passive layers absorb (no loss model, flags only)."""
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    notes = []
    if "sio2" in (substrate, clad) and np.any(lam > 2.6e-6):
        notes.append("WARNING: SiO2 absorbs beyond ~2.6 µm (OH band near 2.7 µm, multiphonon edge beyond ~3.5 µm); "
                     "mid-IR idler loss into the oxide is not modelled")
    if np.any(lam > 4.5e-6):
        notes.append("WARNING: LiNbO3/LiTaO3 multiphonon absorption beyond ~4.5-5 µm")
    return notes
