"""1D multilayer slab modes with diagonal anisotropic layers (transfer matrix, exact).

Propagation along x, layers stacked along z, uniform in y.
TE (E = Ey):  Ey'' + k0^2 (eps_y - neff^2) Ey = 0, continuous Ey and Ey'.
TM (H = Hy):  (1/eps_x) Hy'' + k0^2 (1 - neff^2 / eps_z) Hy = 0, continuous Hy and Hy'/eps_x.

Used for: slab references (platform E), the etched-region slab modes that decide lateral
leakage of rib modes, cladding decay lengths for domain sizing, and a check of the
Tidy3D mode solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Layer:
    eps: tuple[float, float, float]   # (eps_x, eps_y, eps_z) in the sim frame
    thickness: float


def _q_p(eps, neff, k0, pol):
    ex, ey, ez = eps
    if pol == "TE":
        return np.sqrt(complex(k0**2 * (ey - neff**2))), 1.0
    return np.sqrt(complex(k0**2 * ex * (1.0 - neff**2 / ez))), 1.0 / ex


def _gamma(eps, neff, k0, pol):
    q, p = _q_p(eps, neff, k0, pol)
    return float(np.imag(q)), p      # principal sqrt: q = i*gamma for an evanescent cladding


def _march(layers, below, above, neff, k0, pol):
    gb, pb = _gamma(below, neff, k0, pol)
    psi, u = 1.0 + 0j, pb * gb + 0j
    for layer in layers:
        q, p = _q_p(layer.eps, neff, k0, pol)
        qd = q * layer.thickness
        c, s = np.cos(qd), (np.sin(qd) / q if abs(q) > 1e-14 else layer.thickness)
        psi, u = c * psi + s * u / p, -p * q * np.sin(qd) * psi + c * u
        norm = max(abs(psi), abs(u) / k0, 1e-300)
        psi, u = psi / norm, u / norm
    gt, pt = _gamma(above, neff, k0, pol)
    return float(np.real(u + pt * gt * psi))


def slab_modes(layers, below, above, lam, pol, scan_points):
    """Guided effective indices, descending (mode 0 first)."""
    k0 = 2 * np.pi / lam
    comp = 1 if pol == "TE" else 2
    n_lo = np.sqrt(max(below[comp], above[comp], below[1], above[1]))
    n_hi = np.sqrt(max(max(l.eps[comp], l.eps[1]) for l in layers))
    if n_hi <= n_lo:
        return []
    grid = np.linspace(n_lo, n_hi, scan_points)[1:-1]
    vals = np.array([_march(layers, below, above, n, k0, pol) for n in grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        r = brentq(lambda n: _march(layers, below, above, n, k0, pol), grid[i], grid[i + 1], xtol=1e-12)
        roots.append(r)
    return sorted(roots, reverse=True)


def decay_length(eps_cladding, neff, lam, pol):
    """Field 1/e decay length in a cladding for a mode of index neff."""
    k0 = 2 * np.pi / lam
    g, _ = _gamma(eps_cladding, neff, k0, pol)
    return np.inf if g <= 0 else 1.0 / g


def symmetric_slab(eps_core, eps_clad, thickness):
    return [Layer(tuple(eps_core), thickness)], tuple(eps_clad), tuple(eps_clad)
