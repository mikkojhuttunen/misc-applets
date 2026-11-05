"""Fundamental slab modes of a free-standing (air-clad) film, robust for multimode thicknesses (design_calcs/slab.py
returns the first sign change from below, a higher-order mode once the film carries more than one).

TE0 / TM0 (even, fundamental): with u = h t/2 in (0, pi/2), V = k0 t/2 sqrt(n^2 - 1), w = sqrt(V^2 - u^2):
    TE0: u tan u = w          TM0: u tan u = n^2 w
n_eff = sqrt(n^2 - (2u/(k0 t))^2). Gas overlap, two definitions:
    gamma_E   = int_gas |E|^2 / int |E|^2                         (fraction of |E|^2 in the gas; design_calcs/slab.py,
                                                                    the Gamma used in steps 2-5 and in the brief)
    gamma_abs = (n_g / n_gas) int_gas eps|E|^2 / int eps|E|^2      (absorption confinement factor including the group
                                                                    index, Robinson et al., Opt. Express 16, 4296 (2008);
                                                                    the definition of the existing Fig. 1d, which reaches
                                                                    above 1 for thin Si)
Literature dispersion for LPCVD Si3N4 (stoichiometric): Luke et al., Opt. Lett. 40, 4823 (2015),
n^2 = 1 + 3.0249 L^2/(L^2 - 0.1353406^2) + 40314 L^2/(L^2 - 1239.842^2), L in um [coefficients from memory: VERIFY];
silicon-rich low-stress SiNx is higher (n ~ 2.1-2.3 at 1.55 um) and depends on the recipe."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def n_luke(lam_um):
    L2 = np.asarray(lam_um, float) ** 2
    return np.sqrt(1 + 3.0249 * L2 / (L2 - 0.1353406**2) + 40314 * L2 / (L2 - 1239.842**2))


def neff(n, t, lam, pol="TM"):
    k0 = 2 * math.pi / lam
    V = 0.5 * k0 * t * math.sqrt(n * n - 1)
    r = n * n if pol == "TM" else 1.0
    f = lambda u: u * math.tan(u) - r * math.sqrt(max(V * V - u * u, 0.0))
    hi = min(V, math.pi / 2) * (1 - 1e-12)
    u = brentq(f, 1e-15, hi)
    return math.sqrt(n * n - (2 * u / (k0 * t)) ** 2)


def profile(n, t, lam, pol="TM", npts=60001):
    ne = neff(n, t, lam, pol)
    k0 = 2 * math.pi / lam
    h, g = k0 * math.sqrt(n * n - ne * ne), k0 * math.sqrt(ne * ne - 1)
    x = np.linspace(-t / 2 - 14 / g, t / 2 + 14 / g, npts)
    ins = np.abs(x) <= t / 2
    eps = np.where(ins, n * n, 1.0)
    if pol == "TE":
        E2 = np.where(ins, np.cos(h * x) ** 2, math.cos(h * t / 2) ** 2 * np.exp(-2 * g * (np.abs(x) - t / 2)))
    else:                                            # H_y continuous; E_x = (n_eff/eps) H, E_z = (1/(k0 eps)) dH/dx
        H = np.where(ins, np.cos(h * x), math.cos(h * t / 2) * np.exp(-g * (np.abs(x) - t / 2)))
        dH = np.where(ins, -h * np.sin(h * x), -g * np.sign(x) * math.cos(h * t / 2) * np.exp(-g * (np.abs(x) - t / 2)))
        E2 = (ne * ne * H * H + (dH / k0) ** 2) / eps**2
    return ne, x, ins, eps, E2


def mode(n, t, lam, pol="TM", dlam_rel=1e-4, dn_dlam=0.0):
    """n_eff, n_g (material dispersion dn/dlam optional), gamma_E, gamma_abs, and dn_eff/dn (for thermo-optics)."""
    ne, x, ins, eps, E2 = profile(n, t, lam, pol)
    gE = float(np.trapezoid(np.where(~ins, E2, 0), x) / np.trapezoid(E2, x))
    fe = float(np.trapezoid(np.where(~ins, eps * E2, 0), x) / np.trapezoid(eps * E2, x))
    dl = dlam_rel * lam
    n_p, n_m = n + dn_dlam * dl, n - dn_dlam * dl
    ng = ne - lam * (neff(n_p, t, lam + dl, pol) - neff(n_m, t, lam - dl, pol)) / (2 * dl)
    dn = 1e-5
    dndn = (neff(n + dn, t, lam, pol) - neff(n - dn, t, lam, pol)) / (2 * dn)
    return {"n_eff": ne, "n_g": ng, "gamma_E": gE, "gamma_abs": ng * fe, "dneff_dn": dndn}
