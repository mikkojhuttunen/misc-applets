"""Task 9 figure: (a) simulated input-dither factor against the number of input states N (1.53 and 3.3 um), with N^-1/2;
(b) both-dither static fringe (median, IQR) at R 97 % against the input dither for three wavelength-dither depths,
1.53 um, with the engine single-ray value and the M3 target. -> figs/input_states.pdf"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import style  # noqa: E402

S = json.load(open(ROOT / "results/t9/summary.json"))
plt = style.setup()
fig, (a, b) = plt.subplots(1, 2, figsize=(style.MAXW, 9.5 * style.CM), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.55]})
mk = {"angle": "o", "position": "s", "both": "D"}
kind = lambda c: "both" if (" A" in c and " X" in c) else ("position" if " X" in c else "angle")
for key, col, lab in (("nir_R0.97_0.3dB", style.COL["c_cmpc"], "1.53 µm"), ("mir_R0.97_0.3dB", style.COL["orange"], "3.3 µm")):
    seen = set()
    for ck, r in S[key]["rows"].items():
        if not ck.endswith("20pm"):
            continue
        k = kind(r["conf"])
        a.plot(r["N_states"], r["input_factor_rms"], mk[k], color=col, ms=5, mfc="none" if k == "angle" else col,
               label=f"{lab}, {k}" if (lab, k) not in seen else None)
        seen.add((lab, k))
N = np.logspace(0, 1.6, 50)
a.plot(N, N ** -0.5, "k--", lw=1, label="N$^{-1/2}$")
a.set(xscale="log", yscale="log", xlabel="Input states swept N", ylabel="Input-dither factor (rms)", ylim=(0.15, 1.2))
a.set_yticks([0.2, 0.3, 0.5, 1.0]); a.set_yticklabels(["0.2", "0.3", "0.5", "1"]); a.minorticks_off(); a.set_xticks([1, 3, 10, 30]); a.set_xticklabels(["1", "3", "10", "30"])
a.legend(loc="lower left", fontsize=10, handletextpad=0.3, borderaxespad=0.2)
a.text(0.02, 0.98, "(a)", transform=a.transAxes, va="top")
R = S["nir_R0.97_0.3dB"]["rows"]
confs = [c["conf"] for k, c in R.items() if k.endswith("20pm")]
lbl = {"w40 A2": "40\n2\n–", "w75 A15": "75\n15\n–", "w150 A10": "150\n10\n–", "w40 X150": "40\n–\n150", "w40 A15 X150": "40\n15\n150",
       "w75 A15 X100": "75\n15\n100"}
sel = [c for c in confs if c in lbl]
x = np.arange(len(sel))
for j, (dl, col) in enumerate((("20pm", style.COL["none"]), ("81pm", style.COL["lambda"]), ("160pm", style.COL["both"]))):
    med = [R[f"{c} | {dl}"]["fringe"]["both"]["median"] for c in sel]
    lo = [R[f"{c} | {dl}"]["fringe"]["both"]["q1"] for c in sel]
    hi = [R[f"{c} | {dl}"]["fringe"]["both"]["q3"] for c in sel]
    xx = x + (j - 1) * 0.22
    b.errorbar(xx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o", color=col, ms=4, lw=1, capsize=1.5, label=f"±{dl.replace('pm', ' pm')}")
b.axhline(5e-7, color=style.COL["target"], ls="--", lw=1)
b.text(-0.4, 5.6e-7, "target 5×10$^{-7}$", color=style.COL["target"], ha="left", va="bottom")
b.axhline(S["nir_R0.97_0.3dB"]["engine_both_20pm"], color=style.COL["shot"], ls=":", lw=1)
b.text(-0.4, S["nir_R0.97_0.3dB"]["engine_both_20pm"] * 1.15, "engine, single ray (±2 mrad, ±20 pm)", color=style.COL["shot"], fontsize=10, va="bottom")
b.set(yscale="log", xticks=x, ylabel="Static fringe ΔP/P, both dithers", ylim=(8e-8, 6e-5))
b.set_xticklabels([lbl[c] for c in sel], fontsize=10)
b.set_xlabel("waist (µm) / ±angle (mrad) / ±position (µm)", fontsize=10)
b.legend(loc="upper right", title="λ dither", fontsize=10, title_fontsize=10, ncol=3, columnspacing=0.8, handletextpad=0.2)
b.text(0.01, 0.98, "(b)", transform=b.transAxes, va="top")
fig.savefig(ROOT / "figs/input_states.pdf")
fig.savefig(ROOT / "figs/input_states.png", dpi=150)
print("ok")
