"""Metrics of the ensemble outputs (scripts/ens.js, scripts/ens2.js, scripts/ensemble.js).

Static fringe (Fig. 2): the fringe a detector sees against a stray copy of power fraction eps, after the dither,
averaged over the speckle grains:  V_det = 2 sqrt(eps) V / sqrt(grains)  (V: static floor of the dither mode, V = 1
without dither). Noise-equivalent absorbance (Fig. 3): NEA(T) = sqrt(q/(P T_cell) / T + r(T)^2) with the shot term of
the baseline (per root hertz, sqrt(q/(P T)) at T = 1 s) and r(T) the rms fringe residual after a boxcar of length T
(drift 0.1 rad/s, lobe envelope), medians over the layouts. Detection limit = NEA / (alpha_ppm Gamma <L>)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

Q_E = 1.602176634e-19
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODES = ("none", "angle", "spectral", "both")
V_KEYS = {"none": None, "angle": "V_theta", "spectral": "V_lambda", "both": "V_joint"}


def load(path):
    return json.loads(Path(path).read_text())


def fig3_data():
    return load(RESULTS / "baseline" / "fig3data.json")


def peak_alpha(key):
    """Peak absorption coefficient [1/cm per ppm] of a Fig. 3 strip (HITRAN, breath background)."""
    s = [x for x in fig3_data()["strips"] if x["key"] == key][0]
    return float(max(s["analyte_per_ppm"]))


def vdet(V, grains, eps=1e-6):
    return 2 * math.sqrt(eps) * V / math.sqrt(grains)


def fringe_stats(res, eps=1e-6, thr=5e-7, gamma=0.2):
    """Static-fringe statistics of a Fig. 2 style ensemble: per dither mode the chaotic median, the regular value and
    the share of chaotic layouts below thr; transmission and path medians, Gamma <L>."""
    ch = [c for c in res["chaotic"] if "error" not in c]
    reg = res["regular"]
    out = {"n_layouts": len(ch), "modes": {}}
    for m in MODES:
        k = V_KEYS[m]
        v = np.array([vdet(1.0 if k is None else c[k], c["grains"], eps) for c in ch])
        r = vdet(1.0 if k is None else reg[k], reg["grains"], eps) if "error" not in reg else float("nan")
        out["modes"][m] = {"chaotic_median": float(np.median(v)), "regular": float(r), "share_below": float(np.mean(v < thr)),
                           "chaotic_p25": float(np.percentile(v, 25)), "chaotic_p75": float(np.percentile(v, 75))}
    L = np.array([c["L_mean"] for c in ch])
    T = np.array([c["T"] for c in ch])
    out.update(threshold=thr, L_median_m=float(np.median(L)), T_median=float(np.median(T)), gamma=gamma,
               gammaL_cm=float(gamma * np.median(L) * 100), grains_median=float(np.median([c["grains"] for c in ch])),
               regular_L_m=reg.get("L_mean"), regular_T=reg.get("T"))
    return out


def env_at(env, Tgrid, T):
    """log-log interpolation of a residual-vs-averaging-time curve."""
    return float(np.exp(np.interp(math.log10(T), np.log10(Tgrid), np.log(np.asarray(env, float)))))


def nea_curve(res, Tgrid, P_in=1e-3, mode="both", n_layouts=None, extra_white=0.0):
    """Median-based noise-equivalent absorbance vs averaging time: shot term sqrt(q/(P T_cell)) / sqrt(T) (plus any
    extra white term, per root hertz) and the median fringe residual r(T) of the layouts, added in quadrature."""
    ch = res["chaotic"][:n_layouts] if n_layouts else res["chaotic"]
    Tc = float(np.median([c["T_trans"] for c in ch]))
    env = np.median(np.array([c["env"][mode] for c in ch]), axis=0)
    white = math.sqrt(Q_E / (P_in * Tc) + extra_white**2)
    T = np.asarray(Tgrid, float)
    return T, np.sqrt(white**2 / T + env**2), white, env, Tc


def detection_limit(res, Tgrid, alpha_ppm, gamma, T=1.0, P_in=1e-3, mode="both", n_layouts=None):
    """Detection limit [ppm] at averaging time T from medians over the layouts (as the baseline 110 ppb NH3)."""
    ch = res["chaotic"][:n_layouts] if n_layouts else res["chaotic"]
    L = float(np.median([c["L_mean_m"] for c in ch]))
    Tc = float(np.median([c["T_trans"] for c in ch]))
    fr = float(np.median([env_at(c["env"][mode], Tgrid, T) for c in ch]))
    nea = math.sqrt(Q_E / (P_in * Tc) / T + fr**2)
    A = 100 * alpha_ppm * gamma * L
    return {"ppm": nea / A, "nea": nea, "shot": math.sqrt(Q_E / (P_in * Tc) / T), "fringe": fr, "L_median_m": L, "T_median": Tc,
            "gammaL_cm": gamma * L * 100, "alpha_ppm_per_cm": alpha_ppm}
