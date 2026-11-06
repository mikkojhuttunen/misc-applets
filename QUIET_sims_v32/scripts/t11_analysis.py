"""Task 11 analysis: the grating-coupler design (8 x 25 um inclusions, output facet 11, input 75 um +-60 mrad +-100 um) at
5-20 mm, against the 4 mm reference (16 x 50 um). Drift averaging of the static fringe with the engine's envelope
H_d(T) = 1 / sqrt(1 + 2 (0.05 T)^2) (0.1 rad/s); the AC part at the dither frequencies (>= 1 kHz) with
H_ac(T) = 1 / sqrt(1 + 2 (pi 1000 T)^2): r_m(T) = sqrt(S_m^2 H_d^2 + (S_none^2 - S_m^2) H_ac^2) per layout. Detection limit
with the detector (10 mW, transmission T_cell, +8 dB for two grating couplers, NEP 14 pW/rtHz; mid-IR also 1 nW/rtHz).
Scaling: fringe against the number of input states N of each dither design (Task 9 definition).
-> results/t11/summary.json, results/t11/table.md, results/t11/artifact_data.json"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

LN = A.load(ROOT / "results/t7/lines.json")
TG = A.load(ROOT / "data/baseline/fig3data.json")["T"]
BAND = {"nir": (1.5317e-6, 1.5391), "mir": (3.3151e-6, 1.4253)}
GASES = {"nir": [("NH3", "NH3_1.53", {"20pm": 0.896, "35pm": 0.804, "81pm": 0.518})],
         "mir": [("CH4", "CH4_3.31", {"20pm": 0.993, "35pm": 0.979, "81pm": 0.921}), ("C2H6", "C2H6_3.34Q", {"20pm": 0.986, "35pm": 0.959, "81pm": 0.824})]}
DL = {"nir": "35pm", "mir": "81pm"}            # line-compatible wavelength dither of the design
DESIGN, MODES = "w75 A60 X100", ("none", "input", "spectral", "both")
Hd = lambda T: 1 / math.sqrt(1 + 2 * (0.05 * T) ** 2)
Hac = lambda T: 1 / math.sqrt(1 + 2 * (math.pi * 1000 * T) ** 2)


def n_states(w, Aang, Xa, lam, n):
    k = 2 * math.pi * n / lam
    x = np.linspace(-4 * w - Xa, 4 * w + Xa, 12001)
    nA = max(48, math.ceil(2 * math.pi * Aang * k * w)) if Aang else 1
    nX = max(48, math.ceil(4 * math.pi * Xa / w)) if Xa else 1
    sa = np.sin(2 * math.pi * (np.arange(nA) + 0.5) / nA) if Aang else [0.0]
    sx = np.sin(2 * math.pi * (np.arange(nX) + 0.5) / nX) if Xa else [0.0]
    E0 = np.exp(-x ** 2 / w ** 2)
    Eb = np.zeros_like(x, complex)
    for u in sa:
        for v in sx:
            Eb += np.exp(-(x - Xa * v) ** 2 / w ** 2 + 1j * k * Aang * u * (x - Xa * v))
    Eb /= len(sa) * len(sx)
    return float((np.abs(E0) ** 2).sum() / (np.abs(Eb) ** 2).sum())


def det(T, Pd, lam, NEP):
    return math.sqrt(NEP ** 2 + 2 * 6.626e-34 * 2.998e8 / lam * Pd) / (Pd * math.sqrt(2 * T))


def curve(S, m, T):
    s, n0 = S[m], S["none"]
    return math.sqrt((s * Hd(T)) ** 2 + max(n0 ** 2 - s ** 2, 0) * Hac(T) ** 2) if m != "none" else n0 * Hd(T)


def q(v, p):
    return float(np.quantile(v, p))


summ, art, md = {}, {"T": TG, "drift_rad_s": 0.1, "design": DESIGN, "cells": {}, "scaling": {}}, []
for f in sorted((ROOT / "results/t11").glob("final_*_R*.json")):
    d = json.load(open(f))
    band, R = d["band"], d["R"]
    lam, n = BAND[band]
    dl = DL[band]
    Ns = {x["name"]: n_states(x["w"], x["A"], x["Xa"], lam, n) for x in d["dithers"]}
    for cname, C in d["cells"].items():
        ch = [x for x in C["chaotic"] if "error" not in x and x["d"][DESIGN]]
        reg = C["regular"]
        key = f"{band}_R{R}_{cname}"
        S = [x["d"][DESIGN] for x in ch]
        row = {"T": q([s["T"] for s in S], .5), "GL_cm": q([s["GL_cm"] for s in S], .5), "M": q([s["M"] for s in S], .5),
               "static": {m: [q([s["fringe"][dl][m] for s in S], p) for p in (.25, .5, .75)] for m in MODES}, "limits": {}}
        for g, ln, sm in GASES[band]:
            a = LN[ln]["alpha_peak_per_ppm_cm"]
            for T in (0.1, 1, 15, 30, 60):
                for NEP in ((14e-12,) if band == "nir" else (14e-12, 1e-9)):
                    v = [1e3 * math.hypot(curve(s["fringe"][dl], "both", T), det(T, 10e-3 * s["T"] * 10 ** -0.8, lam, NEP)) / (a * s["GL_cm"] * sm[dl]) for s in S]
                    row["limits"][f"{g}_{T}s_NEP{NEP * 1e12:g}pW"] = [q(v, .25), q(v, .5), q(v, .75)]
                row["limits"][f"{g}_{T}s_fringe"] = q([1e3 * curve(s["fringe"][dl], "both", T) / (a * s["GL_cm"] * sm[dl]) for s in S], .5)
        summ[key] = row
        # scaling: fringe (both, line-compatible dither) and input factor against N
        sc = []
        for x in d["dithers"]:
            vals = [r["d"][x["name"]] for r in ch if r["d"][x["name"]]]
            if not vals:
                continue
            fr = [v["fringe"][dl]["both"] for v in vals]
            rin = math.sqrt(np.mean([v["fringe"][dl]["input"] ** 2 for v in vals])) / math.sqrt(np.mean([v["fringe"][dl]["none"] ** 2 for v in vals]))
            sc.append({"name": x["name"], "N": Ns[x["name"]], "both": [q(fr, .25), q(fr, .5), q(fr, .75)], "input_factor": rin,
                       "M": q([v["M"] for v in vals], .5), "T": q([v["T"] for v in vals], .5), "GL_cm": q([v["GL_cm"] for v in vals], .5)})
        art["scaling"][key] = sc
        # artifact pack (Fig. 3 slot): median curves and min-max bands over layouts on the Fig. 3 time grid, per-layout values
        curves = {m: np.array([[curve(s["fringe"][dl], m, T) for T in TG] for s in S]) for m in MODES}
        floor = {m: q([s["fringe"][dl][m] for s in S], .5) for m in MODES}
        mid = int(np.argsort([s["fringe"][dl]["both"] for s in S])[len(S) // 2])
        paths = ch[mid]["paths"]
        cur = {"gamma": BAND and (0.4532 if band == "nir" else 0.5526), "L_mean_m": row["GL_cm"] / 100 / (0.4532 if band == "nir" else 0.5526), "T_trans": row["T"], "M2": row["M"],
               "Ls_m": [p[0] for p in paths], "Ws": [p[1] for p in paths],
               "env": {m: [float(np.median(curves[m][:, i])) for i in range(len(TG))] for m in MODES},
               "env_lo": {m: [float(curves[m][:, i].min()) for i in range(len(TG))] for m in MODES},
               "env_hi": {m: [float(curves[m][:, i].max()) for i in range(len(TG))] for m in MODES},
               "floor": floor, "R_bounce": R,
               "lay": {"fr": {str(T): [curve(s["fringe"][dl], "both", T) for s in S] for T in (0.01, 0.1, 0.5, 1, 15, 30, 60)},
                       "none": {str(T): [curve(s["fringe"][dl], "none", T) for s in S] for T in (0.01, 0.1, 0.5, 1, 15, 30, 60)},
                       "fl": [s["fringe"][dl]["both"] for s in S], "fln": [s["fringe"][dl]["none"] for s in S],
                       "T": [s["T"] for s in S], "L": [s["GL_cm"] / 100 / (0.4532 if band == "nir" else 0.5526) for s in S]}}
        if "error" not in reg and reg["d"][DESIGN]:
            rs = reg["d"][DESIGN]
            rg = {"gamma": cur["gamma"], "L_mean_m": rs["GL_cm"] / 100 / cur["gamma"], "T_trans": rs["T"], "M2": rs["M"], "Ls_m": cur["Ls_m"], "Ws": cur["Ws"],
                  "env": {m: [curve(rs["fringe"][dl], m, T) for T in TG] for m in MODES}, "floor": {m: rs["fringe"][dl][m] for m in MODES}}
        else:
            rg = {k: cur[k] for k in ("gamma", "L_mean_m", "T_trans", "M2", "Ls_m", "Ws", "env", "floor")}
        art["cells"][key] = {"cur": cur, "reg": rg, "dlam": dl, "bars": {m: {"reg": (reg["d"][DESIGN]["fringe"][dl][m] if "error" not in reg and reg["d"][DESIGN] else None),
                                                                             "q": row["static"][m]} for m in MODES},
                             "bars_old": {m: [q([x["d"]["w40 A2"]["fringe"][dl][m] for x in ch if x["d"]["w40 A2"]], p) for p in (.25, .5, .75)] for m in MODES}}
A.dump(ROOT / "results/t11/summary.json", summ)
A.dump(ROOT / "results/t11/artifact_data.json", art)
E = lambda v: f"{v:.2g}"
md.append("| cell | band, R | T | Γ⟨L⟩ cm | M | static fringe, both [IQR] | limit 1 s | 15 s | 30 s | 30 s, fringe only |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for k, r in summ.items():
    band, R, cname = k.split("_", 2)
    for g, _, _ in GASES[band]:
        L = r["limits"]
        nep = "14pW"
        md.append(f"| {cname} | {band} {R[1:]} %, {g} | {r['T']:.4f} | {r['GL_cm']:.1f} | {r['M']:.0f} | {E(r['static']['both'][1])} [{E(r['static']['both'][0])}–{E(r['static']['both'][2])}] | "
                  f"{L[f'{g}_1s_NEP{nep}'][1]:.3g} | {L[f'{g}_15s_NEP{nep}'][1]:.3g} | {L[f'{g}_30s_NEP{nep}'][1]:.3g}"
                  + (f" (1 nW/√Hz: {L[f'{g}_30s_NEP1000pW'][1]:.3g})" if band == "mir" else "") + f" | {L[f'{g}_30s_fringe']:.3g} |")
(ROOT / "results/t11/table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
