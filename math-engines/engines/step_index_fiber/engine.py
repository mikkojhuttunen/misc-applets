"""LP01 mode of a weakly guiding step-index fibre with a fused-silica cladding.

SI units; wavelength is the vacuum wavelength. Requires SciPy (Bessel functions).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1

from ..common import Result, require_positive
from ..materials.engine import index as material_index


def lp01(wavelength, core_radius, delta_n) -> Result:
    """Effective index, V, normalised index b and Marcuse mode-field radius of LP01."""
    require_positive(wavelength=wavelength, core_radius=core_radius, delta_n=delta_n)
    ncl = float(material_index("sio2", wavelength))
    nco = ncl + delta_n
    V = 2 * np.pi / wavelength * core_radius * np.sqrt(nco**2 - ncl**2)
    f = lambda U: U * j1(U) / j0(U) - np.sqrt(V * V - U * U) * k1(np.sqrt(V * V - U * U)) / k0(np.sqrt(V * V - U * U))
    U = brentq(f, 1e-9, min(V, 2.404825557695773) * (1 - 1e-12), xtol=1e-15)
    b = 1 - U * U / (V * V)
    neff = np.sqrt(ncl**2 + (nco**2 - ncl**2) * b)
    w = core_radius * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)
    return Result(
        values={"neff": neff, "V": V, "b": b, "mode_radius": w, "n_clad": ncl},
        units={"neff": "", "V": "", "b": "", "mode_radius": "m", "n_clad": ""},
        assumptions=["Weak guidance (LP modes)", "Fused-silica cladding (Malitson)", "Marcuse fit for the 1/e^2 mode-field radius, valid for 0.8 < V < 2.5"],
    )
