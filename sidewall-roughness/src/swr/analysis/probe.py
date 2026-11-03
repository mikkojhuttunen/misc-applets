"""Spectral-probe method: coupling kernel H(K) from corrugation runs, loss for any roughness PSD.

A wall corrugation h(x) = A sin(K x) has two spectral lines at +-K with weight A^2/4 each
(sigma^2 = A^2/2). In first-order perturbation theory the radiated power per unit length
is linear in the wall PSD:  alpha = integral_{-inf}^{inf} S(K) H(K) dK  with H(-K) = H(K)
taken as even (README, conventions). So one probe run at K gives

    alpha_probe(K) = (A^2/4) [H(K) + H(-K)] = (A^2/2) H(K)   ->   H(K) = 2 alpha_probe / A^2,

and for random roughness on ONE wall  alpha = 2 * integral_0^inf S(K) H(K) dK. Two walls
with uncorrelated roughness add (factor 2 for a mirror-symmetric guide and mode).
Radiation only exists for |beta - K| < k0 n_rad (n_rad = highest radiating index), i.e.
K in (k0 (n_eff - n_rad), k0 (n_eff + n_rad)); H is zero outside that band apart from
narrow guided-mode conversion lines (bookkept separately, not as radiation).
"""

from __future__ import annotations

import numpy as np

DB_PER_NEPER = 10.0 / np.log(10.0)


def radiation_band(n_eff: float, n_rad: float, lam_um: float) -> tuple[float, float]:
    k0 = 2 * np.pi / lam_um
    return k0 * max(n_eff - n_rad, 0.0), k0 * (n_eff + n_rad)


def conversion_lines(n_eff: float, other_neffs, lam_um: float) -> dict:
    """Wall wavenumbers K that couple the mode to other guided modes (forward and backward)."""
    k0 = 2 * np.pi / lam_um
    out = {}
    for label, n in other_neffs.items():
        out[f"fwd->{label}"] = k0 * abs(n_eff - n)
        out[f"bwd->{label}"] = k0 * (n_eff + n)
    return out


def periods_in_band(n_eff, n_rad, lam_um, count, edge_fraction=0.05):
    """`count` periods evenly spaced in K inside the radiation band (edges trimmed)."""
    kmin, kmax = radiation_band(n_eff, n_rad, lam_um)
    pad = edge_fraction * (kmax - kmin)
    k = np.linspace(kmin + pad, kmax - pad, count)
    return 2 * np.pi / k


def alpha_from_power(p_rad_over_p_in, length_um):
    """Power attenuation coefficient (1/um) from the fraction radiated over length L."""
    frac = np.clip(np.asarray(p_rad_over_p_in, dtype=float), 0.0, 1.0 - 1e-15)
    return -np.log1p(-frac) / length_um


def kernel_from_probe(alpha_probe, amplitude_um):
    """H(K) from one probe run; units 1/um^3 so that alpha [1/um] = int S [um^3] H dK [1/um]."""
    return 2.0 * np.asarray(alpha_probe) / amplitude_um**2


def alpha_from_kernel(k, h, psd_fn, walls: int = 1):
    """alpha (1/um) = walls * 2 * int_0^inf S(K) H(K) dK, trapezoid over the sampled K.

    k must cover the radiation band; H is taken as zero outside the sampled range.
    """
    k = np.asarray(k, dtype=float)
    order = np.argsort(k)
    k, h = k[order], np.asarray(h, dtype=float)[order]
    return walls * 2.0 * np.trapezoid(psd_fn(k) * h, k)


def to_db_per_cm(alpha_per_um):
    return DB_PER_NEPER * np.asarray(alpha_per_um) * 1e4
