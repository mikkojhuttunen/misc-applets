"""LP01 mode of a weakly guiding step-index fibre with a fused-silica cladding,
and the three-wave / four-wave phase mismatch it produces.

SI units; wavelength is the vacuum wavelength. Requires SciPy (Bessel functions).

Phase-mismatch conventions (CW pump, signal at omega_s0 + Omega):
  chi2: dk(Omega)    = beta(w_p) - beta(w_s) - beta(w_p - w_s)
  chi3: dbeta(Omega) = beta(w_s) + beta(2 w_p - w_s) - 2 beta(w_p)   (linear part; add 2 gamma P)
``gvm`` = beta1_i - beta1_s (s/m) and ``beta2_sum`` = beta2_s + beta2_i (s^2/m).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1

from ..common import Result, require_choice, require_positive
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


C0 = 299_792_458.0


def propagation_constant(omega, core_radius, delta_n):
    """beta(omega) = omega n_eff / c of the LP01 mode in rad/m (helper, scalar)."""
    return omega / C0 * float(lp01(2 * np.pi * C0 / omega, core_radius, delta_n)["neff"])


def phase_mismatch(wavelength_signal, wavelength_pump, core_radius, delta_n, process="chi2", detuning=0.0) -> Result:
    """Linear phase mismatch of the LP01 modes for chi2 (w_p = w_s + w_i) or degenerate chi3
    (2 w_p = w_s + w_i), its change dk_rel at signal detunings Omega (rad/s, scalar or array),
    the group-velocity mismatch, beta2_s + beta2_i and, for chi2, the QPM period cancelling dk0."""
    require_choice("process", process, ("chi2", "chi3"))
    require_positive(wavelength_signal=wavelength_signal, wavelength_pump=wavelength_pump, core_radius=core_radius, delta_n=delta_n)
    wp = 2 * np.pi * C0 / wavelength_pump
    ws = 2 * np.pi * C0 / wavelength_signal
    wsum = wp if process == "chi2" else 2 * wp
    sg = 1.0 if process == "chi2" else -1.0
    k = lambda w: propagation_constant(w, core_radius, delta_n)
    ksum = k(wp) * (1 if process == "chi2" else 2)

    def dk(O):
        w = ws + O
        return sg * (ksum - k(w) - k(wsum - w))

    d0 = dk(0.0)
    det = np.asarray(detuning, dtype=float)
    dk_rel = np.vectorize(lambda O: dk(O) - d0)(det) if det.ndim else dk(float(det)) - d0
    wi = wsum - ws
    h1, h2 = 1e12, 1e13
    b1 = lambda w: (k(w + h1) - k(w - h1)) / (2 * h1)
    b2 = lambda w: (k(w + h2) - 2 * k(w) + k(w - h2)) / (h2 * h2)
    return Result(
        values={"dk0": d0, "dk_rel": dk_rel, "gvm": b1(wi) - b1(ws), "beta2_sum": b2(ws) + b2(wi),
                "wavelength_idler": 2 * np.pi * C0 / wi, "qpm_period": 2 * np.pi / abs(d0) if process == "chi2" else float("nan")},
        units={"dk0": "1/m", "dk_rel": "1/m", "gvm": "s/m", "beta2_sum": "s^2/m", "wavelength_idler": "m", "qpm_period": "m"},
        assumptions=["LP01 of all three waves, weak guidance, fused-silica cladding, constant index step",
                     "CW pump; chi3 mismatch excludes the nonlinear term 2 gamma P",
                     "beta1 and beta2 by central differences with steps 1e12 and 1e13 rad/s"],
    )
