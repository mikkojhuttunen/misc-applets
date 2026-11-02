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


def ensemble_stats(res, eps=1e-6, thr=5e-7, gamma=None, alpha_ppm=None, T_avg=1.0, P_in=1e-3, n_nh3=21):
    """Summary of a scripts/ensemble.js output: per dither mode the static fringe (chaotic median, regular, quartiles,
    share of layouts below thr), medians of T, <L>, Gamma <L>, bounce-averaged R, share of bounces above the critical
    angle, mean bounces, and, with alpha_ppm, the detection limit at T_avg from the first n_nh3 layouts (medians)."""
    gamma = res.get("gamma", 0.2) if gamma is None else gamma
    ch = [c for c in res["chaotic"] if "error" not in c]
    reg = res["regular"] if res.get("regular") and "error" not in res["regular"] else None
    out = {"n_layouts": len(ch), "gamma": gamma, "modes": {}}
    for m in MODES:
        v = np.array([c["floor"][m] for c in ch])
        out["modes"][m] = {"chaotic_median": float(np.median(v)), "chaotic_p25": float(np.percentile(v, 25)),
                           "chaotic_p75": float(np.percentile(v, 75)), "share_below": float(np.mean(v < thr)),
                           "regular": float(reg["floor"][m]) if reg else float("nan")}
    med = lambda k: float(np.median([c[k] for c in ch]))
    out.update(threshold=thr, T_median=med("T_trans"), L_median_m=med("L_mean_m"), gammaL_cm=gamma * med("L_mean_m") * 100,
               R_bounce_median=med("R_bounce"), above_critical_median=med("above_critical"), bounces_median=med("mean_bounces"),
               grains_median=med("M2"))
    if reg:
        out["regular"] = {"T": reg["T_trans"], "L_m": reg["L_mean_m"], "gammaL_cm": gamma * reg["L_mean_m"] * 100, "R_bounce": reg.get("R_bounce"),
                          "above_critical": reg.get("above_critical")}
    if alpha_ppm:
        Tg = fig3_data()["T"]
        dl = detection_limit(res, Tg, alpha_ppm, gamma, T=T_avg, P_in=P_in, n_layouts=n_nh3)
        out["detection_limit"] = dict(dl, n_layouts=min(n_nh3, len(res["chaotic"])), T_avg=T_avg)
        if reg:
            r1 = {"chaotic": [reg]}
            out["regular"]["detection_limit_ppm"] = detection_limit(r1, Tg, alpha_ppm, gamma, T=T_avg, P_in=P_in)["ppm"]
    return out


def clean(x):
    """JSON-safe copy: inf / nan become None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def dump(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=1))
