"""Task 10 figure: static detection limits against the cell diameter (24-facet circle, 16 inclusions, best input dither
75 um, +-60 mrad, +-100 um, set B, 0.3 dB/cm). (a) NH3 1.53 um, +-35 pm: fringe floor and the total with the detector at
15 s (10 mW, +8 dB for two grating couplers, NEP 14 pW/rtHz), R 97 % and 99.5 %; (b) CH4 3.31 um, R 97 %, +-35 and +-81 pm,
fringe floor (detector negligible at 14 pW/rtHz). Medians over 30 layouts, bars quartiles. -> figs/t10_cell_size.pdf"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402
from qv import style  # noqa: E402

LN = A.load(ROOT / "results/t7/lines.json")
DIAM = [("base", 4), ("d7", 7), ("d10", 10), ("d14", 14), ("d20", 20)]


def det(T, Pd, lam, NEP=14e-12):
    return math.sqrt(NEP ** 2 + 2 * 6.626e-34 * 2.998e8 / lam * Pd) / (Pd * math.sqrt(2 * T))


def series(f, line, dl, sm, lam, total):
    d = json.load(open(ROOT / f"results/t10/{f}.json"))
    a = LN[line]["alpha_peak_per_ppm_cm"]
    xs, med, lo, hi = [], [], [], []
    for nm, D in DIAM:
        if nm not in d["variants"]:
            continue
        v = []
        for x in d["variants"][nm]["chaotic"]:
            if "error" in x:
                continue
            fr = x["fringe"][dl]["both"]
            if total:
                fr = math.hypot(fr, det(15, 10e-3 * x["T"] * 10 ** -0.8, lam))
            v.append(1e3 * fr / (a * x["GL_cm"] * sm))
        xs.append(D); med.append(np.median(v)); lo.append(np.quantile(v, .25)); hi.append(np.quantile(v, .75))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


plt = style.setup()
fig, (a, b) = plt.subplots(1, 2, figsize=(style.MAXW, 7.5 * style.CM), constrained_layout=True)
for f, col, R in (("nir_R0.97_0.3dB", style.COL["c_cmpc"], "97"), ("nir_R0.995_0.3dB", style.COL["c_herriott"], "99.5")):
    for tot, ls, mk in ((False, "--", "o"), (True, "-", "s")):
        x, m, lo, hi = series(f, "NH3_1.53", "35pm", 0.804, 1.5317e-6, tot)
        a.errorbar(x * (1.02 if tot else 0.98), m, yerr=[m - lo, hi - m], fmt=mk + ls, color=col, ms=4, lw=1.2, capsize=0, mfc="none" if not tot else col,
                   label=f"R {R} %, " + ("with detector, 15 s" if tot else "fringe floor"))
a.axhline(50, color=style.COL["target"], ls=":", lw=1)
a.text(4.2, 56, "50 ppb (proposal)", color=style.COL["target"])
a.set(xscale="log", yscale="log", xlabel="Cell diameter (mm)", ylabel="NH₃ static limit (ppb)", xticks=[4, 7, 10, 14, 20], ylim=(15, 3000))
a.set_xticklabels(["4", "7", "10", "14", "20"]); a.minorticks_off()
a.legend(loc="upper right", fontsize=10, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
a.text(0.02, 0.02, "(a) NH₃ 1.53 µm, ±35 pm", transform=a.transAxes)
for dl, sm, col in (("35pm", 0.979, style.COL["orange"]), ("81pm", 0.921, style.COL["si"])):
    x, m, lo, hi = series("mir_R0.97_0.3dB", "CH4_3.31", dl, sm, 3.3151e-6, False)
    b.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, lw=1.2, capsize=0, label=f"±{dl[:-2]} pm (±{float(dl[:-2]) * 2.998e8 / 3.3151e-6 ** 2 * 1e-21:.2g} GHz)")
b.axhline(10, color=style.COL["target"], ls=":", lw=1)
b.text(4.2, 11, "M7: 10 ppb", color=style.COL["target"])
b.set(xscale="log", yscale="log", xlabel="Cell diameter (mm)", ylabel="CH₄ static fringe limit (ppb)", xticks=[4, 10, 14, 20], ylim=(1, 300))
b.set_xticklabels(["4", "10", "14", "20"]); b.minorticks_off()
b.legend(loc="upper right", fontsize=10, handlelength=1.6, borderaxespad=0.2)
b.text(0.02, 0.02, "(b) CH₄ 3.31 µm, R 97 %", transform=b.transAxes)
fig.savefig(ROOT / "figs/t10_cell_size.pdf")
fig.savefig(ROOT / "figs/t10_cell_size.png", dpi=150)
print("ok")
