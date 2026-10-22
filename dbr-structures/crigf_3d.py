"""CRIGF geometry schematic in 3D: Python port of crigf_schematic.m (same section, tooth and offset conventions).

Layout along x: left DBR | left phase section | grating coupler (GC) | right phase section | right DBR.
Each grating section holds N full unit cells with one centred tooth per cell; tooth width = fill * period;
section length = N * period. Phase lengths are gaps between SECTION edges. A positive offset moves the whole GC
toward the right DBR: actual left phase = phaseL + offset, actual right phase = phaseR - offset.
Teeth are additive rectangles on a flat multilayer stack (no etch profile, residual film, field or spectrum).
Thick layers are drawn at most display_cap thick; every physical value stays in the parameters and the text panel.

Defaults follow Hemsley et al., Opt. Express 31, 27274 (2023), doi:10.1364/OE.498125: NGC = 21, 400 DBR periods
per side, periods 884 / 442 nm, fill 0.5, Si3N4 teeth 47 nm, LiNbO3 300 nm on fused silica 500 µm, width 20 µm.
The 300 nm phase sections are example values, not taken from the paper.

    python crigf_3d.py
    python crigf_3d.py --NGC 31 --offset 50 --NL 200 --NR 200
    python crigf_3d.py --layers "Fused silica:500000,LiNbO3:300" --grating-material Si3N4 --grating-height 47

Outputs figs/crigf_3d.png and .pdf. The 3D drawing uses ../mpc-schematics/render3d.py (numpy, Pillow, shapely).
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from shapely.geometry import box  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mpc-schematics"))
from render3d import Scene  # noqa: E402

OUT = HERE / "figs"
LAYER_COLORS = [(0.78, 0.80, 0.80), (0.57, 0.68, 0.65), (0.83, 0.75, 0.62), (0.72, 0.64, 0.76)]   # as in the .m file
GRATING_COLORS = [(0.13, 0.52, 0.38), (0.91, 0.48, 0.16), (0.13, 0.52, 0.38)]                   # DBR, GC, DBR
SECTION_NAMES = ["left DBR", "phase L", "GC", "phase R", "right DBR"]


@dataclass
class CRIGF:
    """All lengths in µm (as in crigf_schematic.m)."""
    NGC: int = 21
    periodGC: float = 0.884
    fillGC: float = 0.5
    NL: int = 400
    periodL: float = 0.442
    fillL: float = 0.5
    NR: int = 400
    periodR: float = 0.442
    fillR: float = 0.5
    phaseL: float = 0.3
    phaseR: float = 0.3
    offset: float = 0.0
    width: float = 20.0
    grating_material: str = "Si3N4"
    grating_height: float = 0.047
    layer_names: list = field(default_factory=lambda: ["Fused silica", "LiNbO3"])
    layer_thickness: list = field(default_factory=lambda: [500.0, 0.300])
    display_cap: float = 0.6
    zoom_margin: float = 1.5

    def check(self):
        for n in ("NGC", "NL", "NR"):
            v = getattr(self, n)
            if int(v) != v or v < 0:
                raise ValueError(f"{n} must be a non-negative integer")
        if self.NGC < 1:
            raise ValueError("GC must contain at least one period")
        if self.NGC + self.NL + self.NR > 100000:
            raise ValueError("use at most 100000 periods for this schematic")
        for n in ("periodGC", "periodL", "periodR", "width", "grating_height", "display_cap"):
            if not getattr(self, n) > 0:
                raise ValueError(f"{n} must be positive")
        for n in ("fillGC", "fillL", "fillR"):
            if not 0 < getattr(self, n) <= 1:
                raise ValueError(f"{n} must be in (0, 1]")
        for n in ("phaseL", "phaseR", "zoom_margin"):
            if getattr(self, n) < 0:
                raise ValueError(f"{n} must be non-negative")
        if self.phaseL + self.offset < 0 or self.phaseR - self.offset < 0:
            raise ValueError("offset must satisfy -phaseL <= offset <= phaseR; the GC cannot enter a DBR")
        if len(self.layer_names) != len(self.layer_thickness) or not 1 <= len(self.layer_names) <= 20:
            raise ValueError("give one name per layer thickness, 1 to 20 layers")
        if min(self.layer_thickness) <= 0:
            raise ValueError("layer thicknesses must be positive")

    def build(self):
        """Section lengths and edges, tooth table [(left, right, section 0/1/2)], displayed layer z edges."""
        self.check()
        self.lengths = np.array([self.NL * self.periodL, self.phaseL + self.offset, self.NGC * self.periodGC,
                                 self.phaseR - self.offset, self.NR * self.periodR])
        self.edges = np.r_[0, np.cumsum(self.lengths)]
        self.total = self.edges[-1]
        counts, periods = [self.NL, self.NGC, self.NR], [self.periodL, self.periodGC, self.periodR]
        fills, starts = [self.fillL, self.fillGC, self.fillR], self.edges[[0, 2, 4]]
        teeth = []
        for j in range(3):
            left = starts[j] + np.arange(counts[j]) * periods[j] + (1 - fills[j]) * periods[j] / 2
            teeth.append(np.c_[left, left + fills[j] * periods[j], np.full(counts[j], j)])
        self.teeth = np.vstack(teeth)
        self.dt = np.minimum(self.layer_thickness, self.display_cap)
        self.zedge = np.r_[-self.dt.sum(), -self.dt.sum() + np.cumsum(self.dt)]
        self.zedge[-1] = 0.0
        return self


def scene_3d(c: CRIGF, x0, x1, sx, sy, sz, cut=None, azim=-58, elev=24, edge_alpha=0.75):
    """Stack and teeth between x0 and x1, coordinates scaled by (sx, sy, sz) for display. cut = x where the front
    half (y < width/2) is removed up to x1, exposing a section through the stack and the teeth."""
    sc = Scene(azim=azim, elev=elev, edge_alpha=edge_alpha, ambient=0.62)
    W = c.width
    cutbox = box(cut * sx, -1, x1 * sx + 1, W / 2 * sy) if cut is not None else None
    tol = 1e-9

    def section(p, q):
        if cut is None:
            return False
        on_y = abs(p[1] - W / 2 * sy) < tol and abs(q[1] - W / 2 * sy) < tol and min(p[0], q[0]) >= cut * sx - tol
        on_x = abs(p[0] - cut * sx) < tol and abs(q[0] - cut * sx) < tol and max(p[1], q[1]) <= W / 2 * sy + tol
        return on_y or on_x

    def lighter(col, f=0.35):
        return tuple(np.asarray(col) * (1 - f) + f)

    for k in range(len(c.dt)):
        col = LAYER_COLORS[k % len(LAYER_COLORS)]
        fp = box(x0 * sx, 0, x1 * sx, W * sy)
        if cutbox is not None:
            fp = fp.difference(cutbox)
        sc.prism(fp, c.zedge[k] * sz, c.zedge[k + 1] * sz, col, section, lighter(col))
    sel = (c.teeth[:, 1] > x0) & (c.teeth[:, 0] < x1)
    for a, b, j in c.teeth[sel]:
        col = GRATING_COLORS[int(j)]
        fp = box(max(a, x0) * sx, 0, min(b, x1) * sx, W * sy)
        if cutbox is not None:
            fp = fp.difference(cutbox)
        sc.prism(fp, 0, c.grating_height * sz, col, section, lighter(col))
    return sc


def draw_side(ax, c: CRIGF, xl, title):
    """Longitudinal section (true lengths, displayed layer thicknesses), all teeth in range drawn."""
    for k in range(len(c.dt)):
        ax.add_patch(plt.Rectangle((0, c.zedge[k]), c.total, c.dt[k], fc=LAYER_COLORS[k % len(LAYER_COLORS)], ec="none"))
    for j in range(3):
        t = c.teeth[(c.teeth[:, 2] == j) & (c.teeth[:, 1] >= xl[0]) & (c.teeth[:, 0] <= xl[1])]
        if len(t):
            verts = [[(a, 0), (b, 0), (b, c.grating_height), (a, c.grating_height)] for a, b, _ in t]
            ax.add_collection(PolyCollection(verts, facecolors=[GRATING_COLORS[j]], edgecolors="none"))
    for e in c.edges[1:-1]:
        ax.axvline(e, color="0.4", ls=":", lw=0.8)
    ax.set(xlim=xl, ylim=(c.zedge[0], 1.7 * c.grating_height), xlabel="longitudinal x (µm)", ylabel="display z (µm)", title=title)
    ax.title.set_fontsize(10)


def text_panel(c: CRIGF):
    L = c.lengths
    t = [f"TOTAL LENGTH = {c.total:.6g} µm", f"Transverse width = {c.width:.6g} µm", "",
         "LEFT DBR | PS-L | GC | PS-R | RIGHT DBR", "Section lengths (µm):", " | ".join(f"{x:.6g}" for x in L), "",
         f"GC: {c.NGC} periods, pitch {c.periodGC * 1e3:.6g} nm",
         f"GC fill {c.fillGC:.4g}; tooth width {c.fillGC * c.periodGC * 1e3:.6g} nm",
         f"Left DBR: {c.NL} periods, pitch {c.periodL * 1e3:.6g} nm",
         f"Left fill {c.fillL:.4g}; tooth width {c.fillL * c.periodL * 1e3:.6g} nm",
         f"Right DBR: {c.NR} periods, pitch {c.periodR * 1e3:.6g} nm",
         f"Right fill {c.fillR:.4g}; tooth width {c.fillR * c.periodR * 1e3:.6g} nm",
         f"GC offset = {c.offset * 1e3:.6g} nm (positive: right)",
         f"Actual phase sections: {L[1] * 1e3:.6g} / {L[3] * 1e3:.6g} nm", "",
         f"Teeth: {c.grating_material}, height {c.grating_height * 1e3:.6g} nm",
         "FLAT LAYERS (bottom to top; true thickness):"]
    for n, d, dd in zip(c.layer_names, c.layer_thickness, c.dt):
        t.append(f"{n}: {d * 1e3:.6g} nm" + (f"  (drawn {dd * 1e3:.6g} nm)" if dd < d else ""))
    t += ["", "Orange = GC; green = DBRs.", "Phase sections contain no grating teeth.",
          "All flat layers span the total length.", "Display heights are not to common scale.",
          "Dimensions above are physical values.", "Ambient above the teeth is not drawn."]
    return "\n".join(t)


def figure(c: CRIGF, width_px=1500, azim=-58, elev=24):
    c.build()
    OUT.mkdir(exist_ok=True)
    zspan = c.zedge[-1] - c.zedge[0] + c.grating_height
    # overview: whole device, aspect x : y : z = 5 : 1.5 : 1 (pbaspect of the .m file)
    sx, sy, sz = 5 / c.total, 1.5 / c.width, 1 / zspan
    over, to_o = scene_3d(c, 0, c.total, sx, sy, sz, azim=azim, elev=elev, edge_alpha=0.25).render(width=width_px)
    # close-up: GC +- zoom margin + a few DBR periods, cut open from the GC centre to the right end
    x0 = max(0.0, c.edges[2] - c.zoom_margin - 6 * c.periodL)
    x1 = min(c.total, c.edges[3] + c.zoom_margin + 6 * c.periodR)
    gc_mid = 0.5 * (c.edges[2] + c.edges[3])
    s = 3.0 / (x1 - x0)
    zc = 0.55 / zspan                                     # displayed stack height 0.55 against a 3-unit length
    close, to_c = scene_3d(c, x0, x1, s, s, zc, cut=gc_mid, azim=azim, elev=elev).render(width=width_px)
    zx_over, zx_close = sz / sx, zc / s

    fig = plt.figure(figsize=(15, 12.5), dpi=130)
    gs = fig.add_gridspec(4, 2, width_ratios=[2.15, 1], height_ratios=[1.0, 1.25, 0.5, 0.5], hspace=0.5, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(over); ax.set_axis_off()
    ax.set_title(f"3D, whole device | x compressed, height exaggerated ×{zx_over:.0f} vs x; thick layers capped", fontsize=10, loc="left")
    for j, name in enumerate(SECTION_NAMES):
        xm = 0.5 * (c.edges[j] + c.edges[j + 1])
        if name.startswith("phase"):
            continue
        px, py = to_o(np.array([[xm * sx, c.width * sy, c.grating_height * sz]]))
        ax.annotate(name, (px[0], py[0]), (px[0], py[0] - 40), ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="-", lw=0.7), annotation_clip=False)
    ax2 = fig.add_subplot(gs[1, 0]); ax2.imshow(close); ax2.set_axis_off()
    ax2.set_title(f"3D close-up: GC, phase sections and the first DBR periods, cut open at the GC centre "
                  f"(height ×{zx_close:.1f} vs x)", fontsize=10, loc="left")
    for (P, txt, off) in (([(c.edges[2] + 4 * c.periodGC) * s, c.width * s, c.grating_height * zc], f"GC, {c.NGC} × {c.periodGC * 1e3:.0f} nm", (90, -60)),
                          ([(x1 - 2 * c.periodR) * s, c.width * 0.5 * s, c.grating_height * zc], "cut face (section)", (60, -50)),
                          ([(c.edges[1] + c.lengths[1] / 2) * s, c.width * s, 0], f"phase L {c.lengths[1] * 1e3:.0f} nm", (-70, -60))):
        px, py = to_c(np.array([P]))
        ax2.annotate(txt, (px[0], py[0]), (px[0] + off[0], py[0] + off[1]), fontsize=9, ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.7), annotation_clip=False)
    for k, n in enumerate(c.layer_names):
        zmid = 0.5 * (c.zedge[k] + c.zedge[k + 1])
        px, py = to_c(np.array([[x1 * s, c.width / 2 * s, zmid * zc]]))
        ax2.annotate(n, (px[0], py[0]), (px[0] + 60, py[0] + 20), fontsize=9,
                     arrowprops=dict(arrowstyle="-", lw=0.7), annotation_clip=False)
    a3 = fig.add_subplot(gs[2, 0]); draw_side(a3, c, (0, c.total), "Full longitudinal section (all periods drawn)")
    a4 = fig.add_subplot(gs[3, 0])
    draw_side(a4, c, (max(0, c.edges[2] - c.zoom_margin), min(c.total, c.edges[3] + c.zoom_margin)), "GC and phase-section close-up")
    at = fig.add_subplot(gs[:, 1]); at.set_axis_off()
    at.text(0.02, 0.98, text_panel(c), va="top", ha="left", fontsize=10, family="DejaVu Sans")
    fig.suptitle("CRIGF geometry: DBR | phase | grating coupler | phase | DBR", x=0.02, ha="left", fontweight="bold", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"crigf_3d.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    L = c.lengths
    print(f"CRIGF section lengths (µm): DBR-L {L[0]:.9g} | PS-L {L[1]:.9g} | GC {L[2]:.9g} | PS-R {L[3]:.9g} | DBR-R {L[4]:.9g}")
    print(f"Total length = {c.total:.9g} µm; transverse width = {c.width:.9g} µm -> {OUT / 'crigf_3d.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = CRIGF()
    for n in ("NGC", "NL", "NR"):
        ap.add_argument(f"--{n}", type=int, default=getattr(d, n))
    for n, unit in (("periodGC", "nm"), ("periodL", "nm"), ("periodR", "nm"), ("phaseL", "nm"), ("phaseR", "nm"),
                    ("offset", "nm"), ("grating-height", "nm"), ("display-cap", "nm"), ("zoom-margin", "nm")):
        key = n.replace("-", "_")
        ap.add_argument(f"--{n}", type=float, default=getattr(d, key) * 1e3, help=unit)
    for n in ("fillGC", "fillL", "fillR"):
        ap.add_argument(f"--{n}", type=float, default=getattr(d, n))
    ap.add_argument("--width", type=float, default=d.width, help="µm")
    ap.add_argument("--grating-material", default=d.grating_material)
    ap.add_argument("--layers", default="Fused silica:500000,LiNbO3:300", help="name:thickness_nm, bottom to top")
    ap.add_argument("--azim", type=float, default=-58)
    ap.add_argument("--elev", type=float, default=24)
    a = ap.parse_args()
    names, thick = zip(*[(s.split(":")[0].strip(), float(s.split(":")[1]) / 1e3) for s in a.layers.split(",")])
    c = CRIGF(NGC=a.NGC, periodGC=a.periodGC / 1e3, fillGC=a.fillGC, NL=a.NL, periodL=a.periodL / 1e3, fillL=a.fillL,
              NR=a.NR, periodR=a.periodR / 1e3, fillR=a.fillR, phaseL=a.phaseL / 1e3, phaseR=a.phaseR / 1e3,
              offset=a.offset / 1e3, width=a.width, grating_material=a.grating_material,
              grating_height=a.grating_height / 1e3, layer_names=list(names), layer_thickness=list(thick),
              display_cap=a.display_cap / 1e3, zoom_margin=a.zoom_margin / 1e3)
    figure(c, azim=a.azim, elev=a.elev)


if __name__ == "__main__":
    main()
