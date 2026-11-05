"""Analysis of the ensemble files written by scripts/ensemble.js (QUIET v32). All formulas explicit:

static residual fringe (per dither mode m, eps the parasitic power fraction, M the number of independent speckle
patterns at the detector = the M^2 "grains" of the detected light):
    dP/P_m = 2 sqrt(eps) V_m / sqrt(M),   V_none = 1, V_angle = V_theta, V_spectral = V_lambda, V_both = V_joint
(V_m: static (0 Hz) visibility of the fringe between the exit light and a stray copy of it, averaged over the dither).
fringe residual after averaging T: r_m(T) = 2 sqrt(eps)/sqrt(M) * sqrt(sum_k P_k |H(f_k T)|^2), the dither spectrum
components P_k at f_k (engine: DitherLab.residual, boxcar envelope, fringe drift 0.1 rad/s) -> stored as env[m] on the
Fig. 3 time grid with eps = 1e-6.
noise-equivalent absorbance (single-pass absorbance A = alpha Gamma L; detection limit c = NEA / (alpha_1ppm Gamma <L>)):
    NEA_m(T) = sqrt( r_m(T)^2 + delta_det(T)^2 )
    old model (Fig. 3b, set A): delta_det = shot only = sqrt(q / (P_in T_cell T))   (1 A/W, T_cell = ray transmission)
    proposal model (T1.3):      delta_det = sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T)),  P_d = P_in 10^(-IL/10)
Medians and interquartile ranges over the inclusion layouts; the regular cell (no inclusions) is a single value.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
Q_E, H_PL, C0 = 1.602176634e-19, 6.62607015e-34, 299792458.0
MODES = ("none", "spectral", "angle", "both")
VKEY = {"none": None, "spectral": "V_lambda", "angle": "V_theta", "both": "V_joint"}


def load(path):
    return json.loads(Path(path).read_text())


def fig3():
    return load(ROOT / "data" / "baseline" / "fig3data.json")


def tgrid():
    return np.array(fig3()["T"])


def chaotic(res):
    return [x for x in res["chaotic"] if "error" not in x]


def fringe(run, mode, eps=1e-6):
    v = 1.0 if VKEY[mode] is None else run[VKEY[mode]]
    return 2 * math.sqrt(eps) * v / math.sqrt(run["grains"])


def stats(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "q1": float(np.quantile(v, .25)), "q3": float(np.quantile(v, .75)),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def at_T(curve, T, Tg=None):
    Tg = tgrid() if Tg is None else Tg
    return float(np.exp(np.interp(math.log(T), np.log(Tg), np.log(np.asarray(curve, float)))))


def shot_old(run, T, P_in=1e-3):
    return math.sqrt(Q_E / (P_in * run["T_trans"] * T))


def det_noise(T, P_in, IL_dB, wavelength, NEP=14e-12):
    """proposal T1.3: delta = sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T))"""
    Pd = P_in * 10 ** (-IL_dB / 10)
    return math.sqrt(NEP ** 2 + 2 * H_PL * C0 / wavelength * Pd) / (Pd * math.sqrt(2 * T))


def nea_old(run, T, mode="both", P_in=1e-3):
    return math.hypot(at_T(run["env"][mode], T), shot_old(run, T, P_in))


def summary(res, eps=1e-6, alpha_ppm=None, T=1.0, thr=5e-7):
    ch, reg = chaotic(res), res["regular"]
    g = res.get("gamma", 0.2)
    out = {"n_layouts": len(ch), "gamma": g}
    for m in MODES:
        out[f"fringe_{m}"] = stats([fringe(x, m, eps) for x in ch])
        out[f"fringe_{m}_regular"] = fringe(reg, m, eps)
        out[f"fringe_{m}_below_{thr:g}"] = float(np.mean([fringe(x, m, eps) < thr for x in ch]))
    out["gain_both_regular_over_chaotic"] = out["fringe_both_regular"] / out["fringe_both"]["median"]
    out["gain_angle_regular_over_chaotic"] = out["fringe_angle_regular"] / out["fringe_angle"]["median"]
    for k, key in (("L_m", "L_mean_m"), ("T_cell", "T_trans"), ("M_grains", "grains"), ("n_pass", "n_pass")):
        out[k] = stats([x[key] for x in ch])
        out[k + "_regular"] = reg[key]
    out["GL_cm"] = stats([g * x["L_mean_m"] * 100 for x in ch])
    out["GL_cm_regular"] = g * reg["L_mean_m"] * 100
    for k in ("N_theta", "N_lambda", "N_eff", "M_spat", "L_rms_m", "sigma_opl", "lam_ftle", "R_bounce", "mean_bounces"):
        if k in ch[0]:
            out[k] = stats([x[k] for x in ch])
            out[k + "_regular"] = reg.get(k)
    if "env" in ch[0]:
        Tg = tgrid()
        for m in MODES:
            out[f"nea_old_{m}_T{T:g}"] = stats([nea_old(x, T, m) for x in ch])
        if alpha_ppm:
            lim = [nea_old(x, T) / (alpha_ppm * g * x["L_mean_m"] * 100) * 1e3 for x in ch]
            out[f"limit_ppb_old_T{T:g}"] = stats(lim)
            out[f"limit_ppb_old_T{T:g}_regular"] = nea_old(reg, T) / (alpha_ppm * g * reg["L_mean_m"] * 100) * 1e3
    return out


def dump(path, obj):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, (np.floating, np.integer)):
            return clean(o.item())
        return o
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), indent=1))
