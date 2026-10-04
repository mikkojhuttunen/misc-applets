"""Bragg gratings: coupled-mode estimates and the exact transfer matrix of a periodic stack.

SI units; wavelength is the vacuum wavelength; indices are effective indices of
the guided mode (or bulk indices for a thin-film stack). Lossless unless an
imaginary index is passed to stack_reflectance.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_positive, require_range


def coupling_coefficient(delta_n, wavelength, fill_factor=0.5, order=1) -> Result:
    """κ of a rectangular index grating: κ = 2 Δn sin(π m f) / (m λ)."""
    require_positive(delta_n=delta_n, wavelength=wavelength)
    require_range("fill_factor", fill_factor, 0.0, 1.0)
    m = int(order)
    if m < 1:
        raise ValueError("order must be >= 1")
    kappa = 2 * delta_n * abs(np.sin(np.pi * m * fill_factor)) / (m * wavelength)
    return Result(
        values={"kappa": kappa},
        units={"kappa": "1/m"},
        assumptions=["Rectangular profile", "Δn = n_high - n_low of the effective index", "Coupled-mode theory, weak grating"],
    )


def grating_summary(kappa, length, wavelength, n_group) -> Result:
    """Peak reflectance tanh²(κL), null-to-null bandwidth and effective (penetration) length."""
    require_positive(kappa=kappa, length=length, wavelength=wavelength, n_group=n_group)
    kl = kappa * length
    return Result(
        values={
            "R_peak": np.tanh(kl) ** 2,
            "bandwidth_null": wavelength**2 / (np.pi * n_group * length) * np.sqrt(kl**2 + np.pi**2),
            "L_eff": np.tanh(kl) / (2 * kappa),
            "kappa_L": kl,
        },
        units={"R_peak": "", "bandwidth_null": "m", "L_eff": "m", "kappa_L": ""},
        assumptions=["Uniform grating", "Lossless", "Coupled-mode theory"],
    )


def cmt_reflectance(wavelength, bragg_wavelength, n_eff, kappa, length):
    """Coupled-mode reflectance spectrum (helper, vectorised over wavelength).
    Half-detuning δ = 2π n_eff (1/λ - 1/λ_B); material dispersion ignored."""
    require_positive(bragg_wavelength=bragg_wavelength, n_eff=n_eff, length=length)
    lam = np.asarray(wavelength, dtype=float)
    d = 2 * np.pi * n_eff * (1 / lam - 1 / bragg_wavelength)
    s2 = kappa**2 - d**2
    s = np.sqrt(np.abs(s2)) + 1e-300
    L = length
    hyp = kappa**2 * np.sinh(s * L) ** 2 / (d**2 * np.sinh(s * L) ** 2 + s2 * np.cosh(s * L) ** 2)
    osc = kappa**2 * np.sin(s * L) ** 2 / (d**2 - kappa**2 * np.cos(s * L) ** 2)
    return np.where(s2 > 0, hyp, osc)


def stack_reflectance(wavelength, n_high, n_low, d_high, d_low, periods, n_in=None, n_out=None) -> Result:
    """Exact (Abelès) reflectance and transmittance of (H L)^N at normal incidence,
    H facing the input medium. Indices may be complex (loss: positive imaginary part)."""
    require_positive(d_high=d_high, d_low=d_low)
    N = int(periods)
    if N < 1:
        raise ValueError("periods must be >= 1")
    n_in = n_low if n_in is None else n_in
    n_out = n_low if n_out is None else n_out
    lam = np.atleast_1d(np.asarray(wavelength, dtype=float))
    require_positive(wavelength=lam)
    k = 2 * np.pi / lam

    def layer(n, d):
        dl = k * n * d
        c, s = np.cos(dl), np.sin(dl)
        return np.array([[c, -1j * s / n], [-1j * n * s, c]], dtype=complex)  # shape (2, 2, nλ)

    H, Lm = layer(n_high, d_high), layer(n_low, d_low)
    P = np.einsum("ijw,jkw->ikw", H, Lm)
    M = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], P.shape).copy()
    base, e = P, N
    while e:
        if e & 1:
            M = np.einsum("ijw,jkw->ikw", M, base)
        base = np.einsum("ijw,jkw->ikw", base, base)
        e >>= 1
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    x, y = n_in * a + n_in * n_out * b, c + n_out * d
    r = (x - y) / (x + y)
    t = 2 * n_in / (x + y)
    R, T = np.abs(r) ** 2, np.real(n_out) / np.real(n_in) * np.abs(t) ** 2
    if np.ndim(wavelength) == 0:
        R, T = R[0], T[0]
    return Result(
        values={"R": R, "T": T},
        units={"R": "", "T": ""},
        assumptions=["Normal incidence", "Plane waves (or one guided mode with effective indices)", "Abelès characteristic matrices"],
    )
