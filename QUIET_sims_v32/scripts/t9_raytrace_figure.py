"""Task 9b ray-tracing figure for wide input dithers -> figs/dither_raytrace.pdf
(a) input phase space (position x, angle theta) of the launch beam: 1/e^2 footprints of the input states swept by the old
dither (40 um, +-2 mrad), a wide angle dither (150 um, +-45 mrad) and a combined one (75 um, +-45 mrad, +-100 um), with the
number of orthogonal input states N; (b)-(d) centre-ray traces (28 wall hits) of the chaotic CMPC under +-45 mrad and
+-100 um, and of the regular CMPC under +-45 mrad; (e) overlap |<E(0), E(theta)>|^2 of two input states against the tilt
between them (wave optics, Gaussian beam) for three waists, with the engine single-ray decorrelation angle 1.16 mrad; (f) both-dither static fringe (+-35 pm, line-compatible) against N, R 97 %."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Ellipse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import style  # noqa: E402

RT = json.load(open(ROOT / "results/t9b/raytrace.json"))
S = json.load(open(ROOT / "results/t9b/summary.json"))
kap = 2 * math.pi * RT["n"] / RT["lam"]
plt = style.setup()
fig = plt.figure(figsize=(style.MAXW, 13.0 * style.CM))
BOX = {(0, 0): [0.085, 0.49, 0.23, 0.34], (0, 1): [0.34, 0.44, 0.32, 0.41], (0, 2): [0.67, 0.44, 0.32, 0.41],
       (1, 0): [0.0, 0.0, 0.33, 0.37], (1, 1): [0.43, 0.085, 0.22, 0.28], (1, 2): [0.76, 0.085, 0.22, 0.28]}
axs = {k: fig.add_axes(v) for k, v in BOX.items()}
STC3 = ["#1f77b4", "#000000", "#d62728"]

# (a) input phase space
a = axs[0, 0]
for (w, A, X, col, lab, nst) in ((40e-6, 2e-3, 0, "#7f7f7f", "40 µm, ±2 mrad", 7), (150e-6, 45e-3, 0, style.COL["c_herriott"], "150 µm, ±45 mrad", 7),
                                  (75e-6, 45e-3, 100e-6, style.COL["c_cmpc"], "75 µm, ±45, ±100 µm", 25)):
    th_d = 2 / (kap * w)
    for i in range(nst):
        u = math.sin(2 * math.pi * (i + 0.5) / nst)
        v = math.sin(2 * math.pi * 0.618034 * (i + 0.5) * 3) if X else 0
        a.add_patch(Ellipse((X * v * 1e6, A * u * 1e3), 2 * w * 1e6, 2 * th_d * 1e3, fill=False, ec=col, lw=0.8, alpha=0.9))
    a.plot([], [], color=col, lw=1.4, label=lab)
a.set(xlim=(-270, 270), ylim=(-62, 62), xlabel="Input position (µm)", ylabel="Input angle (mrad)")
a.text(0.5, 1.02, "(a) input states swept", transform=a.transAxes, ha="center", va="bottom")
fig.legend(*a.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=10, handlelength=1.2, columnspacing=1.0)


def cell(ax, c, traces, key, unit, tag):
    o = np.array(c["outline"]) * 1e3
    ax.plot(o[:, 0], o[:, 1], color="k", lw=0.8)
    for x, y, r in zip(c["inc"]["cx"], c["inc"]["cy"], c["inc"]["r"]):
        ax.add_patch(Circle((x * 1e3, y * 1e3), r * 1e3, color=style.COL["inclusion"], alpha=0.8, lw=0))
    for t, col in zip([t for t in traces if t["u"] in (-1, 0, 1)], STC3):
        p = np.array(t["pts"])[:21] * 1e3
        ax.plot(p[:, 0], p[:, 1], color=col, lw=0.6, alpha=0.85)
    for nm, mk in (("in", "s"), ("out", "D")):
        q = np.array(c["ports"][nm]) * 1e3
        ax.plot(q[0], q[1], mk, color="#2ca02c", ms=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, 1.0, tag, transform=ax.transAxes, va="bottom", ha="center")


cell(axs[0, 1], RT["cells"]["chaotic"], RT["cells"]["chaotic"]["angle"], "mrad", "mrad", "(b) chaotic CMPC, ±45 mrad")
cell(axs[0, 2], RT["cells"]["chaotic"], RT["cells"]["chaotic"]["pos"], "um", "µm", "(c) chaotic CMPC, ±100 µm")
cell(axs[1, 0], RT["cells"]["regular"], RT["cells"]["regular"]["angle"], "mrad", "mrad", "(d) regular CMPC, ±45 mrad")
hs = [plt.Line2D([], [], color=c, lw=1.4) for c in STC3]
fig.legend(hs, ["−45 mrad / −100 µm", "centre", "+45 mrad / +100 µm  (b)–(d), 20 wall hits"], loc="upper left", bbox_to_anchor=(0.0, 0.955), ncol=3, fontsize=10,
           handlelength=1.2, columnspacing=1.0)

# (e) overlap of two input states against tilt (wave optics) against the engine's single-ray decorrelation
e = axs[1, 1]
th = np.linspace(0, 20e-3, 400)
for w, col in ((40e-6, "#7f7f7f"), (75e-6, style.COL["c_cmpc"]), (150e-6, style.COL["c_herriott"])):
    e.plot(th * 1e3, np.exp(-(kap * th * w) ** 2 / 4), color=col, lw=1.4, label=f"{w * 1e6:.0f} µm")
e.axvline(1.16, color=style.COL["target"], ls=":", lw=1)
e.text(1.7, 0.04, "single ray", color=style.COL["target"], va="bottom", rotation=90)
e.axvspan(0, 2, color="#000", alpha=0.07, lw=0)
e.set(xlim=(0, 20), ylim=(0, 1.02), xlabel="Tilt between states (mrad)", ylabel="Input-state overlap")
e.legend(loc="upper right", fontsize=10, handlelength=1.0, borderaxespad=0.1, labelspacing=0.2, title="waist", title_fontsize=10)
e.text(0.5, 1.02, "(e) why ±2 mrad fails", transform=e.transAxes, ha="center", va="bottom")

# (f) fringe against N
f = axs[1, 2]
R = S["nir_R0.97_0.3dB"]["rows"]
kind = lambda c: "both" if (" A" in c and " X" in c) else ("position" if " X" in c else "angle")
mk = {"angle": "o", "position": "s", "both": "D"}
seen = set()
for ck, r in R.items():
    if not ck.endswith("35pm"):
        continue
    k = kind(r["conf"]); q = r["fringe"]["both"]
    f.errorbar(r["N_states"], q["median"], yerr=[[q["median"] - q["q1"]], [q["q3"] - q["median"]]], fmt=mk[k], color=style.COL["c_cmpc"],
               mfc="none" if k == "angle" else style.COL["c_cmpc"], ms=4, lw=0.7, capsize=0, label=k if k not in seen else None)
    seen.add(k)
N = np.logspace(0, 2.7, 40)
f0 = R["w40 A2 | 35pm"]["fringe"]["both"]["median"]
f.plot(N, f0 * N ** -0.5, "k--", lw=0.9, label="N$^{-1/2}$")
f.axhline(5e-7, color=style.COL["target"], ls="--", lw=0.9)
f.text(1.2, 3.5e-7, "target", color=style.COL["target"])
f.set(xscale="log", yscale="log", xlabel="Input states N", ylabel="ΔP/P (±35 pm)", ylim=(2e-7, 2e-5))
f.legend(loc="upper right", fontsize=10, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
f.text(0.5, 1.02, "(f) fringe, R 97 %", transform=f.transAxes, ha="center", va="bottom")
fig.savefig(ROOT / "figs/dither_raytrace.pdf")
fig.savefig(ROOT / "figs/dither_raytrace.png", dpi=150)
print("ok")
