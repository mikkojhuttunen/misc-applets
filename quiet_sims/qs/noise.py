"""Noise budget of the QUIET cells (step 5).

Noise-equivalent absorbance after averaging tau (same conventions as the baseline: a white term given per root hertz
and divided by sqrt(tau), plus the fringe residual r(tau) of the ensembles, which is relative and independent of the
power):
    P_det     = P_in 10^(-IL/10)                       (cell insertion loss IL in dB, coupling and propagation)
    shot      = sqrt(q / (Resp P_det))                  (as the baseline sqrt(q/(P T)), Resp in A/W)
    detector  = NEP / P_det                             (NEP in W/sqrt(Hz), referred to the optical power)
    NEA(tau)  = sqrt((shot^2 + detector^2) / tau + r(tau)^2)
Detection limit = NEA / (alpha_peak Gamma <L>) with alpha_peak the HITRAN peak per ppm (step 4)."""
from __future__ import annotations

import math

import numpy as np

Q_E = 1.602176634e-19


def white(P_in, IL_dB, NEP, resp=1.0):
    P = P_in * 10 ** (-IL_dB / 10)
    shot = math.sqrt(Q_E / (resp * P))
    det = NEP / P
    return {"P_det": P, "shot": shot, "detector": det, "white": math.hypot(shot, det)}


def nea(tau, white_per_rthz, r_tau):
    tau = np.asarray(tau, float)
    return np.sqrt(white_per_rthz**2 / tau + np.asarray(r_tau, float) ** 2)


def time_to_reach(target_nea, Tgrid, white_per_rthz, r_curve):
    """Shortest averaging time on the grid (log interpolation) at which NEA <= target; inf if never on the grid."""
    T = np.asarray(Tgrid, float)
    v = nea(T, white_per_rthz, r_curve)
    ok = np.nonzero(v <= target_nea)[0]
    if not ok.size:
        return math.inf
    i = ok[0]
    if i == 0:
        return float(T[0])
    lt, lv = np.log(T[i - 1:i + 1]), np.log(v[i - 1:i + 1])
    return float(math.exp(lt[0] + (math.log(target_nea) - lv[0]) * (lt[1] - lt[0]) / (lv[1] - lv[0])))
