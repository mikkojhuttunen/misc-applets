"""Fused-silica step-index fiber: LP01 mode and three-wave / four-wave phase mismatch.

Conventions
-----------
* SI units. Wavelengths are vacuum wavelengths (m); ``core_radius`` in m;
  ``delta_n`` is the core-cladding index step (constant over wavelength).
* Cladding index from the Malitson (1965) Sellmeier fit of fused silica.
* Scalar, weakly guiding LP01 mode from U J1(U)/J0(U) = W K1(W)/K0(W),
  U^2 + W^2 = V^2. The Gaussian-equivalent mode radius uses Marcuse's fit.
* Phase mismatch for a CW pump and a signal at omega_s0 + Omega:
    chi2: dk(Omega)   = k(w_p) - k(w_s) - k(w_p - w_s)
    chi3: dbeta(Omega) = k(w_s) + k(2 w_p - w_s) - 2 k(w_p)
  The returned ``dk_rel`` is the change relative to Omega = 0.
* ``gvm`` = beta1_i - beta1_s (s/m) and ``beta2_sum`` = beta2_s + beta2_i (s^2/m).
"""
from __future__ import annotations

import math

import numpy as np

from engines.common import C0, Result, require_choice, require_positive

_SELLMEIER = ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161))
_J0_ZERO = 2.404825557695773
_ASSUME = ["Fused-silica cladding (Malitson Sellmeier)", "constant core index step",
           "scalar weakly guiding LP01 mode", "Marcuse Gaussian fit for the mode radius"]


def _n_silica(lam):
    x2 = (np.asarray(lam, float) * 1e6) ** 2
    return np.sqrt(1 + sum(B * x2 / (x2 - C * C) for B, C in _SELLMEIER))


def silica_index(wavelength) -> Result:
    """Refractive index of fused silica (Malitson 1965, valid 0.21-3.7 um)."""
    require_positive("wavelength", wavelength)
    return Result({"n": _n_silica(wavelength)}, {"n": "1"}, ["Malitson Sellmeier, room temperature"])


def _j1_over_j0(x):
    q, t0, t1, j0, j1 = x * x / 4, 1.0, x / 2, 1.0, x / 2
    for m in range(1, 60):
        t0 *= -q / (m * m)
        t1 *= -q / (m * (m + 1))
        j0 += t0
        j1 += t1
        if abs(t0) + abs(t1) < 1e-19:
            break
    return j1 / j0


def _k1_over_k0(x):
    # K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt; trapezoid rule converges exponentially
    h, s0, s1 = 0.2, 0.5, 0.5
    for k in range(1, 20000):
        ch = math.cosh(k * h)
        e = math.exp(-x * (ch - 1))
        s0 += e
        s1 += e * ch
        if e * ch < 1e-18 * s1:
            break
    return s1 / s0


def _lp01_scalar(lam, a, dn):
    ncl = float(_n_silica(lam))
    nco = ncl + dn
    na2 = nco * nco - ncl * ncl
    V = 2 * math.pi / lam * a * math.sqrt(na2)

    def F(U):
        W = math.sqrt(V * V - U * U)
        return U * _j1_over_j0(U) - W * _k1_over_k0(W)

    # Illinois regula falsi on (0, min(V, j01))
    lo, hi = 1e-9, min(V, _J0_ZERO) * (1 - 1e-12)
    flo, fhi, side, U = F(lo), F(hi), 0, lo
    for _ in range(200):
        U = (lo * fhi - hi * flo) / (fhi - flo)
        if not (lo < U < hi):
            U = 0.5 * (lo + hi)
        fu = F(U)
        if fu == 0 or hi - lo < 1e-15 * hi:
            break
        if fu * fhi > 0:
            hi, fhi = U, fu
            if side == -1:
                flo /= 2
            side = -1
        else:
            lo, flo = U, fu
            if side == 1:
                fhi /= 2
            side = 1
    b = 1 - U * U / (V * V)
    neff = math.sqrt(ncl * ncl + na2 * b)
    w = a * (0.65 + 1.619 / V ** 1.5 + 2.879 / V ** 6)
    return neff, V, w, b


_lp01_vec = np.vectorize(_lp01_scalar, otypes=[float, float, float, float])


def lp01_mode(wavelength, core_radius, delta_n) -> Result:
    """Effective index, normalised frequency V, normalised propagation constant b and mode radius."""
    require_positive("wavelength", wavelength)
    require_positive("core_radius", core_radius)
    require_positive("delta_n", delta_n)
    neff, V, w, b = _lp01_vec(wavelength, core_radius, delta_n)
    if np.ndim(neff) == 0:
        neff, V, w, b = float(neff), float(V), float(w), float(b)
    return Result(
        values={"n_eff": neff, "V": V, "b": b, "w_mode": w, "single_mode": np.asarray(V) < _J0_ZERO},
        units={"n_eff": "1", "V": "1", "b": "1", "w_mode": "m", "single_mode": "bool"},
        assumptions=_ASSUME + (["Marcuse fit is accurate for 1.2 < V < 2.4"]),
    )


def wavenumber(omega, core_radius, delta_n):
    """Propagation constant beta(omega) = omega n_eff / c of the LP01 mode (helper)."""
    lam = 2 * math.pi * C0 / omega
    return omega / C0 * _lp01_scalar(lam, core_radius, delta_n)[0]


def idler_wavelength(lambda_signal, lambda_pump, process="chi2"):
    """Idler from energy conservation: chi2 w_i = w_p - w_s, chi3 w_i = 2 w_p - w_s (helper)."""
    lam_sum = lambda_pump if process == "chi2" else lambda_pump / 2
    return 1 / (1 / lam_sum - 1 / np.asarray(lambda_signal, float))


def phase_mismatch(lambda_signal, lambda_pump, core_radius, delta_n, process="chi2", detuning=0.0) -> Result:
    """Linear phase mismatch of the LP01 modes and its group-velocity terms.

    ``detuning`` (rad/s, scalar or array) is the signal offset Omega from omega_s0;
    ``dk_rel`` is the mismatch change relative to Omega = 0. For chi2 the QPM period
    that cancels ``dk0`` is also returned.
    """
    require_choice("process", process, ("chi2", "chi3"))
    for nm, v in (("lambda_signal", lambda_signal), ("lambda_pump", lambda_pump), ("core_radius", core_radius), ("delta_n", delta_n)):
        require_positive(nm, v)
    wp = 2 * math.pi * C0 / lambda_pump
    ws = 2 * math.pi * C0 / lambda_signal
    wsum = wp if process == "chi2" else 2 * wp
    sg = 1.0 if process == "chi2" else -1.0
    k = lambda w: wavenumber(w, core_radius, delta_n)
    ksum = k(wp) * (1 if process == "chi2" else 2)

    def dkD(O):
        w = ws + O
        return sg * (ksum - k(w) - k(wsum - w))

    d0 = dkD(0.0)
    det = np.asarray(detuning, float)
    dk_rel = np.vectorize(lambda O: dkD(O) - d0)(det) if det.ndim else dkD(float(det)) - d0
    wi = wsum - ws
    h1, h2 = 1e12, 1e13
    b1 = lambda w: (k(w + h1) - k(w - h1)) / (2 * h1)
    b2 = lambda w: (k(w + h2) - 2 * k(w) + k(w - h2)) / (h2 * h2)
    gvm = b1(wi) - b1(ws)
    beta2_sum = b2(ws) + b2(wi)
    values = {"dk0": d0, "dk_rel": dk_rel, "gvm": gvm, "beta2_sum": beta2_sum,
              "lambda_idler": 2 * math.pi * C0 / wi,
              "qpm_period": 2 * math.pi / abs(d0) if process == "chi2" else float("nan")}
    units = {"dk0": "rad/m", "dk_rel": "rad/m", "gvm": "s/m", "beta2_sum": "s^2/m", "lambda_idler": "m", "qpm_period": "m"}
    return Result(values, units, _ASSUME + ["CW pump; dk0 > 0 means k_p exceeds k_s + k_i (chi2)",
                                            "chi3 dk excludes the nonlinear term 2 gamma P"])
