"""Fig. 2 (e, f) for set B: (e) Poincare sections (s / perimeter, sin chi) of R-/C-Herriott and R-/C-CMPC at the set-B
near-IR index (results/fig2e_poincare.json from scripts/poincare_dump.js, inclusion layout seed 1; dots rasterised at
600 dpi inside the vector PDF), (f) static residual fringe 2 sqrt(eps) V / sqrt(M) with no / spectral (+-20 pm) / angle
(+-2 mrad) / both dithers, chaotic bars = medians over 30 layouts with IQR whiskers, regular = single values, eps 1e-6;
one figure per mirror reflectivity R (membrane loss 0.3 dB/cm). -> figs/fig2ef_setB_R<R>.pdf"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import style  # noqa: E402

MODES = ("none", "spectral", "angle", "both")


def main():
    plt = style.setup()
    C = style.COL
    P = json.loads((ROOT / "results" / "fig2e_poincare.json").read_text())
    S = json.loads((ROOT / "results" / "t2" / "summary.json").read_text())
    maps = [("herriott_regular", "R-Herriott", C["r_herriott"]), ("circle_regular", "R-CMPC", C["r_cmpc"]),
            ("herriott_chaotic", "C-Herriott", C["c_herriott"]), ("circle_chaotic", "C-CMPC", C["c_cmpc"])]
    for R in (0.95, 0.97, 0.995):
        fig = plt.figure(figsize=(16.8 * style.CM, 7.0 * style.CM), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
        for i, (k, lab, col) in enumerate(maps):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.scatter(P[k]["S"], P[k]["C"], s=0.08, color=col, alpha=0.8, lw=0, rasterized=True)
            ax.set(xlim=(0, 1), ylim=(-1, 1), xticks=[0, 0.5, 1], yticks=[-1, 0, 1])
            ax.text(0.03, 0.95, lab, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
            if i % 2:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("sin χ")
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("s / perimeter")
            if i == 0:
                ax.set_title("(e) Poincaré sections", loc="left", fontsize=10)
        b = fig.add_subplot(gs[:, 2])
        xs = np.arange(4)
        for j, (cell, kind, lab, col) in enumerate([("herriott", "regular", "R-Herriott", C["r_herriott"]), ("herriott", "chaotic", "C-Herriott", C["c_herriott"]),
                                                    ("circle", "regular", "R-CMPC", C["r_cmpc"]), ("circle", "chaotic", "C-CMPC", C["c_cmpc"])]):
            s = S[f"{cell}_nir_R{R}_0.3dB"]
            v = np.array([s[f"fringe_{m}_regular"] if kind == "regular" else s[f"fringe_{m}"]["median"] for m in MODES])
            x = xs + (j - 1.5) * 0.2
            b.bar(x, v, 0.2, color=col, label=lab)
            if kind == "chaotic":
                lo = np.array([s[f"fringe_{m}"]["q1"] for m in MODES]); hi = np.array([s[f"fringe_{m}"]["q3"] for m in MODES])
                b.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", elinewidth=0.8, capsize=1.5)
        b.axhline(5e-7, color=C["target"], ls="--", lw=1)
        b.text(-0.45, 6e-7, "target", color=C["target"], fontsize=10, ha="left", va="bottom")
        b.set(yscale="log", ylim=(1e-8, 1e-1), xticks=xs, ylabel="Fringe amplitude ΔP/P", xlabel="Dither")
        b.set_xticklabels(["none", "spectral", "angle", "both"])
        b.legend(ncol=2, loc="upper right", fontsize=10, handlelength=1.0, columnspacing=0.6, labelspacing=0.2)
        b.set_title(f"(f) R = {R * 100:g} %, 0.3 dB/cm", loc="left", fontsize=10)
        out = ROOT / "figs" / f"fig2ef_setB_R{R}.pdf"
        fig.savefig(out, dpi=600)
        fig.savefig(out.with_suffix(".png"), dpi=150)
        plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
