"""One-row overview figure: (a) CMPC chip, (b) integrated Herriott cell, (c) evanescent-field cross-section,
(d) Γ against membrane thickness.

(a) and (b) are the oblique renders of membrane_3d.py without labels (embedded raster, full resolution); (c) and (d)
are drawn as vectors by the panel functions of evanescent_sensing.py.

    python combined_figure.py                         183 mm wide (double column), CMPC star {24/11}
    python combined_figure.py --width 250 --star 7
    python combined_figure.py --no-render             reuse figs/*_bare.png from the last run
Writes figs/combined_overview.pdf, .svg and .png.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evanescent_sensing as ev  # noqa: E402
import membrane_3d as m3  # noqa: E402

OUT = HERE / "figs"


def trimmed(path, pad=0.01):
    """The render with its white margin cut off (pad: fraction of the larger side kept around the drawing)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    ys, xs = np.nonzero((img < 248).any(axis=2))
    p = int(pad * max(img.shape[:2]))
    return img[max(ys.min() - p, 0):ys.max() + p + 1, max(xs.min() - p, 0):xs.max() + p + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=183, help="figure width, mm")
    ap.add_argument("--star", type=int, default=11, help="CMPC star orbit step m (24 facets: 1..11)")
    ap.add_argument("--thickness", type=float, default=250, help="membrane, nm")
    ap.add_argument("--pol", default="TM", choices=("TE", "TM"))
    ap.add_argument("--seed", type=int, default=4, help="molecule placement in (c)")
    ap.add_argument("--font", type=float, default=7, help="font size, pt")
    ap.add_argument("--no-render", action="store_true", help="reuse the 3D renders from the last run")
    a = ap.parse_args()

    if not a.no_render:
        args3d = m3.parser().parse_args(["--bare", "--star", str(a.star), "--thickness", str(a.thickness), "--pol", a.pol])
        m3.chip(args3d)
        m3.herriott(args3d)
    img_a, img_b = trimmed(OUT / "membrane_chip_3d_bare.png"), trimmed(OUT / "membrane_herriott_3d_bare.png")

    plt.rcParams.update({"font.size": a.font, "font.family": ["Inter", "DejaVu Sans"], "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    # panel widths in units of the row height: the renders keep their aspect, (c) and (d) are fixed
    asp = [img_a.shape[1] / img_a.shape[0], img_b.shape[1] / img_b.shape[0], 1.35, 0.8]
    W = a.width / 25.4
    gap, left_d, bottom = 0.06, 0.32, 0.28                 # inches: between panels, (d) y-label, (d) x-label
    H = (W - 3 * gap - left_d - 0.05) / sum(asp)
    fig = plt.figure(figsize=(W, H + bottom + 0.05))
    FH = H + bottom + 0.05
    x = 0.0
    axes = []
    for i, r in enumerate(asp):
        if i == 3:
            x += left_d
        axes.append(fig.add_axes([x / W, bottom / FH, r * H / W, H / FH]))
        x += r * H + gap
    for ax, img in zip(axes[:2], (img_a, img_b)):
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()

    md = ev.mode_data(a.thickness, a.pol)
    ac = axes[2]
    ev.cross_section(fig, ac, md, a.seed, L=4.0, text=False)
    ac.set_xticks([])
    ac.set_yticks([])
    for sp in ac.spines.values():
        sp.set_visible(True)
        sp.set_color("#9aa3a8")
    um, t, dl = 1e6, md["t"], md["delta"]
    ac.text(3.95, (t / 2 + dl) * um, "δ", ha="right", va="bottom", color="#a12a1f", fontsize=a.font)
    ac.text(3.95, (t / 2 + 3 * dl) * um, "3δ", ha="right", va="bottom", color="#a12a1f", fontsize=a.font)

    ad = axes[3]
    ev.gamma_curves(ad, md, text=False, lw=1.3)
    ad.set_xticks([200, 400, 600])
    ad.set_ylim(0, 1.3 * ad.get_ylim()[1])                # headroom for the panel label and the legend
    ad.set_xlabel("t (nm)", labelpad=1)
    ad.set_ylabel("Γ", labelpad=1)
    ad.tick_params(pad=1.5)
    for sp in ("top", "right"):
        ad.spines[sp].set_visible(False)
    ad.legend(frameon=False, loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2)

    for ax, lab in zip(axes, "abcd"):
        ax.text(0.015, 0.985, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontsize=a.font + 2, zorder=20)
    stem = OUT / "combined_overview"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=600 if ext == "png" else 300, pad_inches=0.02, bbox_inches="tight")
    svg = Path(f"{stem}.svg")                               # drop matplotlib's DOCTYPE line (not needed; some hosts reject DTDs)
    svg.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    print("wrote", ", ".join(f"{stem.name}.{e}" for e in ("pdf", "svg", "png")), f"({a.width:g} mm wide)")


if __name__ == "__main__":
    main()
