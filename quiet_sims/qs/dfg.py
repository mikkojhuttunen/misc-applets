"""CW difference-frequency generation in MgO:PPLN (congruent LiNbO3, 5 mol% MgO), all waves extraordinary (type 0,
d33), pump 1064 nm, signal 1.50-1.63 um, idler 3.06-3.66 um.

Literature parameters (marked LIT; the ones not checked against the original source here are marked VERIFY):
  * n_e(lambda, T): Gayer, Sacks, Galun, Arie, Appl. Phys. B 91, 343 (2008), 5 % MgO:CLN, extraordinary:
      n^2 = a1 + b1 f + (a2 + b2 f)/(L^2 - (a3 + b3 f)^2) + (a4 + b4 f)/(L^2 - a5^2) - a6 L^2,
      f = (T - 24.5)(T + 570.82), L in um, T in C
      a1..a6 = 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2; b1..b4 = 2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4
      [coefficients from memory: VERIFY against Table 1 of the paper]
  * thermal expansion of the poling period along x: Lambda(T) = Lambda_25 (1 + 1.54e-5 (T - 25) + 5.3e-9 (T - 25)^2)
      (Jundt, Opt. Lett. 22, 1553 (1997), undoped CLN) [VERIFY for MgO:CLN]
  * d33 = 25 pm/V at 1064 nm (LIT, Shoji et al., JOSA B 14, 2268 (1997) give 25.2 pm/V for CLN); Miller scaling to the
    DFG frequencies lowers it to ~ 22-24 pm/V [ASSUMPTION]; d_eff = (2/pi) d33 = 14 pm/V for ideal 50 % duty cycle
    (the value most CW-DFG papers use) [ASSUMPTION]
  * transparency: idler absorption of MgO:LN at 3.3 um ~ 0.01-0.03 /cm (OH and multiphonon tail) [VERIFY], neglected
    in the power estimate.

Single-pass CW power, undepleted pump and signal, Gaussian beams of equal confocal parameter b = L/xi (Boyd-Kleinman):
    P_i = 8 w_i^2 d_eff^2 L P_p P_s h / (eps0 c^3 n_p n_s n_i (lambda_p/n_p + lambda_s/n_s))
derived from the near-field Gaussian overlap P_i = 4 w_i^2 d^2 L^2 P_p P_s / (pi eps0 c^3 n_p n_s n_i (w_p^2 + w_s^2))
with w_j^2 = b lambda_j / (2 pi n_j), and L/b -> h: h = xi in the near field, h_m ~ 1.07 at xi = 2.84 for equal-
wavelength SHG; for DFG with lambda_i / lambda_p ~ 3 the optimum is lower, h ~ 0.8-1.0 [ASSUMPTION, Boyd & Kleinman,
J. Appl. Phys. 39, 3597 (1968)]. Measured CW DFG efficiencies are typically 30-60 % of this [ASSUMPTION]."""
from __future__ import annotations

import math

import numpy as np

C = 299792458.0
EPS0 = 8.8541878128e-12
A = (5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2)
B = (2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4)


def n_e(lam_um, T=25.0):
    L2 = np.asarray(lam_um, float) ** 2
    f = (T - 24.5) * (T + 570.82)
    a1, a2, a3, a4, a5, a6 = A
    b1, b2, b3, b4 = B
    return np.sqrt(a1 + b1 * f + (a2 + b2 * f) / (L2 - (a3 + b3 * f) ** 2) + (a4 + b4 * f) / (L2 - a5**2) - a6 * L2)


def expansion(T):
    d = T - 25.0
    return 1 + 1.54e-5 * d + 5.3e-9 * d * d


def idler(lp_um, ls_um):
    return 1 / (1 / lp_um - 1 / ls_um)


def signal_for_idler(lp_um, li_um):
    return 1 / (1 / lp_um - 1 / li_um)


def period(li_um, T=25.0, lp_um=1.064):
    """QPM period (um, at temperature T, i.e. the grating as it is at T) for first order."""
    ls = signal_for_idler(lp_um, li_um)
    dk = n_e(lp_um, T) / lp_um - n_e(ls, T) / ls - n_e(li_um, T) / li_um      # in 1/um (without 2 pi)
    return 1 / dk


def period_at_25(li_um, T, lp_um=1.064):
    """Grating period to write (at 25 C) so that QPM is reached at temperature T."""
    return period(li_um, T, lp_um) / expansion(T)


def dk(ls_um, T, Lam25_um, lp_um=1.064):
    """Phase mismatch 2 pi (n_p/l_p - n_s/l_s - n_i/l_i - 1/Lambda(T)) in 1/m."""
    li = idler(lp_um, ls_um)
    v = n_e(lp_um, T) / lp_um - n_e(ls_um, T) / ls_um - n_e(li, T) / li - 1 / (Lam25_um * expansion(T))
    return 2 * math.pi * v * 1e6


def fwhm_of(fn, x0, L_m, step):
    """Full width of sinc^2(dk L/2) >= 1/2 around x0 by root finding on |dk| L/2 = 1.39156."""
    from scipy.optimize import brentq
    g = lambda x: abs(fn(x)) * L_m / 2 - 1.391557
    lo = brentq(g, x0 - 50 * step, x0) if g(x0 - 50 * step) > 0 else math.nan
    hi = brentq(g, x0, x0 + 50 * step) if g(x0 + 50 * step) > 0 else math.nan
    return hi - lo, lo, hi


def power(P_p, P_s, L_m, li_um, ls_um, T=25.0, lp_um=1.064, d_eff=14e-12, h=1.0):
    n_p, n_s, n_i = (float(n_e(x, T)) for x in (lp_um, ls_um, li_um))
    w = 2 * math.pi * C / (li_um * 1e-6)
    den = EPS0 * C**3 * n_p * n_s * n_i * (lp_um * 1e-6 / n_p + ls_um * 1e-6 / n_s)
    return 8 * w * w * d_eff**2 * L_m * P_p * P_s * h / den
