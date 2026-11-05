"""Task 9 analysis: static residual fringe 2 sqrt(eps) V / sqrt(M) (eps 1e-6, no drift) of the wave-consistent input dithers,
medians and quartiles over 30 layouts; the input-state count N = |E_in|^2 / |<E_in>_t|^2 of each dither (the number of
orthogonal input states the time-averaged field is spread over) and the expected rms factor N^-1/2; static NH3 / CH4 /
C2H6 floors in ppb with the simulated Gamma<L> of results/t2. -> results/t9/summary.json, results/t9/table.md"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

LINES = A.load(ROOT / "results/t7/lines.json")
BAND = {"nir": (1.5317e-6, 1.5391), "mir": (3.3151e-6, 1.4253)}


DIR = sys.argv[1] if len(sys.argv) > 1 else "t9"
ADAPT = DIR != "t9"                     # t9: 48 samples per period (as run); t9b: adaptive, as scripts/t9_input_states.js now


def n_states(w, Aang, Xa, lam, n):
    k = 2 * math.pi * n / lam
    x = np.linspace(-4 * w - Xa, 4 * w + Xa, 12001)
    nA = max(48, math.ceil(2 * math.pi * Aang * k * w)) if ADAPT else 48
    nX = max(48, math.ceil(4 * math.pi * Xa / w)) if ADAPT else 48
    sa = np.sin(2 * math.pi * (np.arange(nA) + 0.5) / nA) if Aang else [0.0]
    sx = np.sin(2 * math.pi * (np.arange(nX) + 0.5) / nX) if Xa else [0.0]
    E0 = np.exp(-x ** 2 / w ** 2)
    Eb = np.zeros_like(x, complex)
    for u in sa:
        for v in sx:
            Eb += np.exp(-(x - Xa * v) ** 2 / w ** 2 + 1j * k * Aang * u * (x - Xa * v))
    Eb /= len(sa) * len(sx)
    return float((np.abs(E0) ** 2).sum() / (np.abs(Eb) ** 2).sum())


def stats(v):
    v = np.asarray(v)
    return {"median": float(np.median(v)), "q1": float(np.quantile(v, .25)), "q3": float(np.quantile(v, .75)), "rms": float(np.sqrt((v ** 2).mean()))}


out, md = {}, []
# peak reduction of each line under a triangle wavelength dither of +-dlam (uniform average over +-dnu around the peak)
STR = {s["key"]: s for s in A.load(ROOT / "data/baseline/fig3data.json")["strips"]}
STRIP_OF = {"NH3_1.53": "NH3", "CH4_3.31": "CH4 3.3", "C2H6_3.34Q": "C2H6"}


def smear(line, dlam_m):
    st = STR[STRIP_OF[line]]
    x, y = np.array(st["x_ghz"]), np.array(st["analyte_per_ppm"])
    dnu = 2.99792458e8 * dlam_m / (st["lam_um"] * 1e-6) ** 2 / 1e9
    i = int(np.argmax(y))
    m = np.abs(x - x[i]) <= dnu
    return float(y[m].mean() / y[i]) if m.sum() > 1 else 1.0, dnu


for f in sorted((ROOT / f"results/{DIR}").glob("*_0.3dB*.json")):
    d = json.load(open(f))
    band = d["band"]
    lam, n = BAND[band]
    ch, reg = d["layouts"][1:], d["layouts"][0]
    t2 = A.load(ROOT / f"results/t2/circle_{band}_R{d['R']}_0.3dB.json")
    GL = np.median([x["gamma"] * x["L_mean_m"] * 100 for x in A.chaotic(t2)])
    key = f.stem
    out[key] = {"GL_cm": GL, "sigma": d["sigma"], "rows": {}}
    eng_both = np.median([2e-3 * x["V_joint_engine"] / math.sqrt(x["grains"]) for x in ch])
    out[key]["engine_both_20pm"] = float(eng_both)
    for ck in ch[0]["V"]:
        conf, dl = ck.split(" | ")
        c = next(c for c in d["conf"] if c["name"] == conf)
        fr = {m: [2e-3 * x["V"][ck][m] / math.sqrt(x["grains"]) for x in ch] for m in ("none", "input", "spectral", "both")}
        fr_reg = {m: 2e-3 * reg["V"][ck][m] / math.sqrt(reg["grains"]) for m in ("none", "input", "spectral", "both")}
        rv = {m: stats([x["V"][ck][m] for x in ch])["rms"] for m in ("none", "input", "spectral", "both")}
        N = n_states(c["w"], c["A"], c["Xa"], lam, n)
        row = {"conf": conf, "dlam": dl, "N_states": N, "expected_input_factor": N ** -0.5, "input_factor_rms": rv["input"] / rv["none"],
               "both_over_spectral_rms": rv["both"] / rv["spectral"], "fringe": {m: stats(v) for m, v in fr.items()}, "regular": fr_reg,
               "share_below_5e-7": float(np.mean(np.array(fr["both"]) < 5e-7))}
        ppb, sm = {}, {}
        for ln in (["NH3_1.53"] if band == "nir" else ["CH4_3.31", "C2H6_3.34Q"]):
            sm[ln], dnu = smear(ln, float(dl.replace("pm", "")) * 1e-12)
            ppb[ln] = 1e3 * row["fringe"]["both"]["median"] / (LINES[ln]["alpha_peak_per_ppm_cm"] * GL)
        row["static_floor_ppb"] = ppb
        row["line_peak_kept"] = sm
        row["static_floor_ppb_smeared"] = {k: v / sm[k] for k, v in ppb.items()}
        row["dither_GHz"] = dnu
        out[key]["rows"][ck] = row
A.dump(ROOT / f"results/{DIR}/summary.json", out)

E = lambda v: f"{v:.2g}"
for key in ("nir_R0.95_0.3dB", "nir_R0.97_0.3dB", "nir_R0.995_0.3dB", "mir_R0.97_0.3dB"):
    S = out[key]
    md.append(f"\n#### {key} (engine single-ray, both, ±20 pm: {E(S['engine_both_20pm'])}; Γ⟨L⟩ {S['GL_cm']:.2f} cm)\n")
    md.append("| input dither | λ dither | N states | N^-½ | input factor (rms, sim) | spectral only | input only | both [IQR] | regular both | <5e-7 | static floor ppb | line peak kept | floor incl. smearing |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ck, r in S["rows"].items():
        f = r["fringe"]
        md.append(f"| {r['conf']} | ±{r['dlam']} | {r['N_states']:.1f} | {r['expected_input_factor']:.2f} | {r['input_factor_rms']:.2f} | {E(f['spectral']['median'])} | {E(f['input']['median'])} | "
                  f"{E(f['both']['median'])} [{E(f['both']['q1'])}–{E(f['both']['q3'])}] | {E(r['regular']['both'])} | {100 * r['share_below_5e-7']:.0f} % | "
                  + ", ".join(f"{k.split('_')[0]} {v:.3g}" for k, v in r["static_floor_ppb"].items()) + f" | {', '.join(f'{v:.2f}' for v in r['line_peak_kept'].values())} (±{r['dither_GHz']:.2g} GHz) | "
                  + ", ".join(f"{k.split('_')[0]} {v:.3g}" for k, v in r["static_floor_ppb_smeared"].items()) + " |")
md.append("\n#### Beamlet-size check (nir R 0.97, both, median)\n")
md.append("| config | σ 7 µm | σ 10 µm | σ 14 µm |")
md.append("|---|---|---|---|")
for ck in out["nir_R0.97_0.3dB"]["rows"]:
    md.append(f"| {ck} | " + " | ".join(E(out[k]["rows"][ck]["fringe"]["both"]["median"]) if k in out else "–" for k in ("nir_R0.97_0.3dB_sigma7", "nir_R0.97_0.3dB", "nir_R0.97_0.3dB_sigma14")) + " |")
(ROOT / f"results/{DIR}/table.md").write_text("\n".join(md) + "\n")
print("ok", DIR)
