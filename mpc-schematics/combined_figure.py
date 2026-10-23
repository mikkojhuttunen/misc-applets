"""Overview figures at journal width.

row     one row: (a) CMPC chip, (b) integrated Herriott cell, (c) evanescent-field cross-section, (d) Γ against
        membrane thickness                                               -> figs/combined_overview.{pdf,svg,png}
setup   two rows: (a) and (b) on top; (c), (d) and (e) the system top view of setup_schematic.py (light source in the
        red block, sensing chip, detection in the blue block) below      -> figs/combined_setup.{pdf,svg,png}

(a) and (b) are the oblique renders of membrane_3d.py without labels (embedded raster, full resolution); (c), (d) and
(e) are vector. (e) is the simplified schematic by default (I Source, II Gas sensor, III Detection; designed at the
183 mm panel size, text in fixed point sizes); --setup-style full draws the labelled schematic, scaled down with its
text (the smallest size is printed; use a larger --width for bigger text).

    python combined_figure.py                         both layouts, 183 mm wide (double column), CMPC star {24/11}
    python combined_figure.py setup --width 250 --setup-kind herriott
    python combined_figure.py --no-render             reuse figs/*_bare.png from the last run
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
import setup_schematic as ss  # noqa: E402
import components  # noqa: E402

OUT = HERE / "figs"


def trimmed(path, pad=0.01):
    """The render with its white margin cut off (pad: fraction of the larger side kept around the drawing)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    ys, xs = np.nonzero((img < 248).any(axis=2))
    p = int(pad * max(img.shape[:2]))
    return img[max(ys.min() - p, 0):ys.max() + p + 1, max(xs.min() - p, 0):xs.max() + p + 1]


def panel_c(fig, ac, md, a):
    ev.cross_section(fig, ac, md, a.seed, L=4.0, text=False)
    ac.set_xticks([])
    ac.set_yticks([])
    for sp in ac.spines.values():
        sp.set_visible(True)
        sp.set_color("#9aa3a8")
    um, t, dl = 1e6, md["t"], md["delta"]
    ac.text(3.95, (t / 2 + dl) * um, "δ", ha="right", va="bottom", color="#a12a1f", fontsize=a.font)
    ac.text(3.95, (t / 2 + 3 * dl) * um, "3δ", ha="right", va="bottom", color="#a12a1f", fontsize=a.font)


def panel_d(ad, md):
    ev.gamma_curves(ad, md, text=False, lw=1.3)
    ad.set_xticks([200, 400, 600])
    ad.set_ylim(0, 1.3 * ad.get_ylim()[1])                # headroom for the panel label and the legend
    ad.set_xlabel("t (nm)", labelpad=1)
    ad.set_ylabel("Γ", labelpad=1)
    ad.tick_params(pad=1.5)
    for sp in ("top", "right"):
        ad.spines[sp].set_visible(False)
    ad.legend(frameon=False, loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2)


def labels(axes, a):
    for ax, lab in zip(axes, "abcde"):
        ax.text(0.015, 0.985, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontsize=a.font + 2, zorder=20)


def save(fig, name, a):
    stem = OUT / name
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=600 if ext == "png" else 300, pad_inches=0.02, bbox_inches="tight")
    svg = Path(f"{stem}.svg")                               # drop matplotlib's DOCTYPE line (not needed; some hosts reject DTDs)
    svg.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    plt.close(fig)
    print("wrote", ", ".join(f"{stem.name}.{e}" for e in ("pdf", "svg", "png")), f"({a.width:g} mm wide)")


def row_layout(a, img_a, img_b, md):
    # panel widths in units of the row height: the renders keep their aspect, (c) and (d) are fixed
    asp = [img_a.shape[1] / img_a.shape[0], img_b.shape[1] / img_b.shape[0], 1.35, 0.8]
    W = a.width / 25.4
    gap, left_d, bottom = 0.06, 0.32, 0.28                 # inches: between panels, (d) y-label, (d) x-label
    H = (W - 3 * gap - left_d - 0.05) / sum(asp)
    FH = H + bottom + 0.05
    fig = plt.figure(figsize=(W, FH))
    x, axes = 0.0, []
    for i, r in enumerate(asp):
        if i == 3:
            x += left_d
        axes.append(fig.add_axes([x / W, bottom / FH, r * H / W, H / FH]))
        x += r * H + gap
    for ax, img in zip(axes[:2], (img_a, img_b)):
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
    panel_c(fig, axes[2], md, a)
    panel_d(axes[3], md)
    labels(axes, a)
    save(fig, "combined_overview", a)


def setup_layout(a, img_a, img_b, md):
    W = a.width / 25.4
    gap, left_d, bottom, credit = 0.06, 0.32, 0.28, 0.1   # inches; credit: line for the component-library attribution
    asp1 = [img_a.shape[1] / img_a.shape[0], img_b.shape[1] / img_b.shape[0]]
    H1 = (W - gap) / sum(asp1)
    simple = a.setup_style == "simple"
    XL, YL = ((0, ss.SX), (0, ss.SY)) if simple else (ss.XLIM, ss.YLIM)
    margin = 0.0 if simple else 1.3                        # data units left of the schematic for the panel label
    ex = (XL[1] - XL[0] + margin) / (YL[1] - YL[0])
    asp2 = [1.1, 0.75, ex]
    H2 = (W - 2 * gap - left_d) / sum(asp2)
    FH = credit + bottom + H2 + gap + H1
    fig = plt.figure(figsize=(W, FH))
    axes, x = [], 0.0
    y1 = (credit + bottom + H2 + gap) / FH
    for r, img in zip(asp1, (img_a, img_b)):
        ax = fig.add_axes([x / W, y1, r * H1 / W, H1 / FH])
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        axes.append(ax)
        x += r * H1 + gap
    x = 0.0
    for i, r in enumerate(asp2):
        if i == 1:
            x += left_d
        axes.append(fig.add_axes([x / W, (credit + bottom) / FH, r * H2 / W, H2 / FH]))
        x += r * H2 + gap
    panel_c(fig, axes[2], md, a)
    panel_d(axes[3], md)
    ae = axes[4]
    sa = ss.parser().parse_args([a.setup_kind, "--star", str(a.setup_star)])
    if simple:                                             # designed at this size: lines follow the width, text stays in pt
        k = H2 / ss.SY / ss.SIPU
        ss.draw_simple(ae, a.setup_kind, sa, font=a.font, panel_label=True)
        ss.scale_artists(ae, k, text=False)
    else:
        ref = plt.figure(figsize=(16, 8.4), dpi=130)         # the scale the standalone schematic is designed at
        k = H2 / (YL[1] - YL[0]) / ss.inch_per_unit(ref.add_subplot(111, xlim=ss.XLIM, ylim=ss.YLIM, aspect="equal"))
        plt.close(ref)
        ss.draw(ae, a.setup_kind, sa, title=False, legend=True, attribution=False)
        ae.set_xlim(XL[0] - margin, XL[1])
        ss.scale_artists(ae, k)
    smallest = min(t.get_fontsize() for t in ae.texts)
    labels(axes, a)
    if components.used_library():
        fig.text(1.0, 0.0, components.ATTRIBUTION, ha="right", va="bottom", fontsize=max(a.font - 2.5, 4), color="#5a6a6f")
    save(fig, "combined_setup", a)
    print(f"(e) lines at {k:.2f} of the design size, smallest text {smallest:.1f} pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layouts", nargs="*", default=["row", "setup"], help="row | setup")
    ap.add_argument("--width", type=float, default=183, help="figure width, mm")
    ap.add_argument("--star", type=int, default=11, help="CMPC star orbit step m (24 facets: 1..11)")
    ap.add_argument("--thickness", type=float, default=250, help="membrane, nm")
    ap.add_argument("--pol", default="TM", choices=("TE", "TM"))
    ap.add_argument("--seed", type=int, default=4, help="molecule placement in (c)")
    ap.add_argument("--font", type=float, default=7, help="font size, pt")
    ap.add_argument("--setup-style", default="simple", choices=("simple", "full"),
                    help="panel (e): simple (blocks I Source, II Gas sensor, III Detection) or the full labelled schematic")
    ap.add_argument("--setup-star", type=int, default=7, help="CMPC star orbit step drawn in panel (e)")
    ap.add_argument("--setup-kind", default="cmpc", choices=("cmpc", "herriott"), help="cell drawn in panel (e)")
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
    md = ev.mode_data(a.thickness, a.pol)
    for lay in a.layouts:
        {"row": row_layout, "setup": setup_layout}[lay](a, img_a, img_b, md)


if __name__ == "__main__":
    main()
