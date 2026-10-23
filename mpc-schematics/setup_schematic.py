"""Top-view setup schematic of the whole system: bulk laser and detectors, on-chip dithering, the sensing cell and the
output-port waveguides.

    python setup_schematic.py                  both variants: figs/setup_cmpc.{pdf,png}, figs/setup_herriott.{pdf,png}
    python setup_schematic.py cmpc             cmpc | herriott
    python setup_schematic.py --outputs 3      CMPC output ports (1-3)
    python setup_schematic.py --no-modules     without the red light-source and blue detection blocks

Light path: laser (bulk, with its driver) -> isolator -> lens -> edge coupler -> tap coupler (a few % to a reference
detector for power normalisation) -> dither section -> cell input port -> cell -> output ports (dark blue) -> output
waveguides -> edge couplers -> lenses -> bulk photodiodes -> lock-in / DAQ.

Dithering (the fringe-washout methods of fringe-washout.html and cmpc-ray-tracer.html):
    frequency dither    laser current modulation, from the function generator through the laser driver
    angle dither        on chip: a 1x2 splitter feeds two waveguides with thermo-optic heaters that recombine side by side
                        at the input port; their relative phase steers the launch angle into the cell
    phase dither        a heater on the common input waveguide
The lock-in / DAQ takes the detector signals and the dither reference. Drawing only: positions, sizes and component
choices are schematic, not a layout. Bulk optics use gwoptics ComponentLibrary SVGs from components/ when present
(see components/README.md), otherwise vector stand-ins in the same style.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

import components  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
C = dict(chip="#eef1f3", chip_ec="#8a949b", mem="#c9d6ea", mem_ec="#1f4e79", dbr="#3d4e66", wg_in="#c9952a", wg_out="#1d3f8f",
         wg_ref="#6b7378", beam="#d0352b", heater="#e07b00", fibre="#4f9d4f", elec="#5a6a6f", box="#ffffff", box_ec="#1b2328",
         ink="#1b2328", muted="#5a6a6f", gas="#8a4fc8", fs="#e8574a")
plt.rcParams.update({"font.size": 10, "font.family": ["Inter", "DejaVu Sans"]})


def box(ax, x, y, w, h, title, sub=None, fc=C["box"], ec=C["box_ec"], lw=1.4, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center", fontsize=fs, fontweight="bold", color=C["ink"], zorder=5)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.5, color=C["muted"], zorder=5)


def curve(ax, pts, color, lw=2.6, ls="-", z=3, arrow=False):
    """Smooth guide through control points (cubic Bézier segments with tangent handles)."""
    P = np.asarray(pts, float)
    verts, codes = [P[0]], [MPath.MOVETO]
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ta = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]) / 2 if 0 < i else b - a
        tb = (P[min(i + 2, len(P) - 1)] - P[i]) / 2 if i + 2 < len(P) else b - a
        verts += [a + ta / 3, b - tb / 3, b]
        codes += [MPath.CURVE4] * 3
    ax.add_patch(PathPatch(MPath(verts, codes), fc="none", ec=color, lw=lw, ls=ls, zorder=z, capstyle="round"))
    if arrow:
        d = P[-1] - P[-2]
        ax.add_patch(FancyArrowPatch(P[-1] - 0.05 * d, P[-1], arrowstyle="-|>", mutation_scale=13, color=color, lw=0, zorder=z + 1))


def heater(ax, x0, x1, y, n=7, amp=0.07):
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = y + amp * np.array([0 if i in (0, 2 * n) else (1 if i % 2 else -1) for i in range(2 * n + 1)])
    ax.plot(xs, ys, color=C["heater"], lw=1.6, zorder=5)
    for xe in (x0, x1):
        ax.add_patch(Rectangle((xe - 0.07, y - 0.07), 0.14, 0.14, fc="#f2c38b", ec=C["heater"], lw=0.8, zorder=5))


def elec(ax, pts, label=None, at=0.5, color=None):
    P = np.asarray(pts, float)
    ax.plot(P[:, 0], P[:, 1], color=color or C["elec"], lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    if label:
        i = int(at * (len(P) - 1))
        m = P[i] + (P[min(i + 1, len(P) - 1)] - P[i]) * 0.5
        ax.text(m[0], m[1] + 0.12, label, fontsize=8.3, color=C["muted"], ha="center", va="bottom", zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def cmpc_cell(ax, c, r, N=24, m=7, band=0.22, outs=(30, -30)):
    """Segmented cell with a DBR band, star orbit, input port at 180°, output ports at the given angles (deg)."""
    a0 = np.cos(np.pi / N)
    ang = 2 * np.pi * (np.arange(N) / N) - np.pi / N
    for j, (rr, col, alpha) in enumerate(((1 + band, C["dbr"], 0.85), (1, C["mem"], 1.0))):
        pts = c + r * rr * np.c_[np.cos(ang), np.sin(ang)]
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec=C["mem_ec"], lw=0.8, alpha=alpha, zorder=2))
    for k in range(4):                                     # trench lines in the band
        rr = 1 + band * (k + 0.5) / 4
        pts = c + r * rr * np.c_[np.cos(ang), np.sin(ang)]
        ax.add_patch(Polygon(pts, closed=True, fc="none", ec="#a9b8cc", lw=0.6, zorder=2))
    p = N // 2
    ks = (p + m * np.arange(N + 1)) % N
    mids = c + r * a0 * np.c_[np.cos(2 * np.pi * ks / N), np.sin(2 * np.pi * ks / N)]
    ax.plot(mids[:, 0], mids[:, 1], color=C["beam"], lw=0.8, alpha=0.85, zorder=3)
    ports = []
    for deg in outs:                                       # output ports: band drawn thinner (fewer periods)
        th = np.radians(deg)
        u = np.array([np.cos(th), np.sin(th)])
        v = np.array([-u[1], u[0]])
        q = [c + r * (1 + band * 0.5) * u - 0.11 * r * v, c + r * (1 + band) * u - 0.11 * r * v,
             c + r * (1 + band) * u + 0.11 * r * v, c + r * (1 + band * 0.5) * u + 0.11 * r * v]
        ax.add_patch(Polygon(q, closed=True, fc=C["mem"], ec="none", zorder=2.5))
        ports.append((c + r * (1 + band) * u, u))
    d0 = (mids[1] - mids[0]) / np.linalg.norm(mids[1] - mids[0])
    q = np.array([mids[0] - 0.13 * r * np.array([-d0[1], d0[0]]) - d0 * band * r * 1.4,
                  mids[0] - 0.13 * r * np.array([-d0[1], d0[0]]), mids[0] + 0.13 * r * np.array([-d0[1], d0[0]]),
                  mids[0] + 0.13 * r * np.array([-d0[1], d0[0]]) - d0 * band * r * 1.4])
    ax.add_patch(Polygon(q, closed=True, fc=C["mem"], ec="none", zorder=2.5))
    return mids[0], d0, ports


def herriott_cell(ax, x0, y0, d, D, R):
    """Two curved mirrors (drawn as arcs with a trench band), V of input/output at the centre of mirror 1."""
    y = np.linspace(-D, D, 60)
    for k, sgn in ((0, -1), (1, 1)):
        sx = R - np.sqrt(R * R - y * y)
        xs = x0 + (sx if k == 0 else d - sx)
        for j in range(4):
            off = sgn * 0.045 * (j + 0.5)
            yy = np.where((k == 0) & (np.abs(y) < 0.1), np.nan, y)
            ax.plot(xs + off, y0 + yy, color=C["dbr"], lw=2.2, zorder=3)
    rng = np.random.default_rng(1)
    th = np.arccos(1 - d / R)
    n = np.arange(0, 23)
    ys = -D / 1.4 * np.sin(n * th)
    xs = np.where(n % 2 == 0, x0, x0 + d)
    ax.plot(xs, y0 + ys, color=C["beam"], lw=0.8, alpha=0.85, zorder=3)
    return np.array([x0, y0])


XLIM, YLIM = (-0.2, 18.5), (-0.6, 9.3)
MOD = dict(src=("#e8574a", "#c0392b", "#a12a1f"), det=("#3a62c4", "#1d3f8f", "#1d3f8f"))   # fill, edge, text


def module(ax, x0, y0, x1, y1, kind, label):
    """Shaded, framed block of bulk parts (red: light source, blue: detection) with its name in the bottom right corner."""
    fc, ec, ink = MOD[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15", fc=fc, alpha=0.09, ec="none", zorder=0.5))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15", fc="none", ec=ec, lw=1.8, zorder=0.6))
    ax.text(x1 - 0.12, y0 + 0.1, label, ha="right", va="bottom", fontsize=10, fontweight="bold", color=ink, zorder=6)


def draw(ax, kind, a, title=True, legend=True, attribution=True):
    """The whole schematic into ax (data units: the chip is 10 x 7.15)."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    if getattr(a, "modules", True):
        module(ax, -0.1, 3.3, 4.27, 9.15, "src", "LIGHT SOURCE")
        module(ax, 14.58, -0.05, 18.45, 9.15, "det", "DETECTION")
    # ------------------------------------------------------------ bulk: laser, driver, isolator, fibre
    yb = 4.57                                              # free-space beam height (chip input)
    hl = components.place(ax, "laser", 1.05, yb, 1.6)
    ax.text(1.05, yb - hl / 2 - 0.08, "Laser", ha="center", va="top", fontsize=11, fontweight="bold", color=C["ink"])
    ax.text(1.05, yb - hl / 2 - 0.36, "bulk DFB / ICL, CW", ha="center", va="top", fontsize=8.5, color=C["muted"])
    box(ax, 0.1, 6.4, 2.3, 1.05, "Laser driver", "current + TEC")
    box(ax, 0.1, 8.05, 2.3, 0.95, "Function generator", "dither waveforms", fs=10)
    elec(ax, [[1.25, 6.4], [1.25, yb + hl / 2]], "frequency dither (current)")
    elec(ax, [[1.25, 8.05], [1.25, 7.45]])
    # free space: laser -> isolator -> lens focusing onto the edge coupler
    ax.plot([2.05, 2.62], [yb, yb], color=C["fs"], lw=2.4, zorder=3)
    components.place(ax, "isolator", 3.0, yb, 0.72)
    ax.text(3.0, yb - 0.33, "isolator", ha="center", va="top", fontsize=8.5, color=C["muted"])
    ax.add_patch(Polygon([[3.37, yb - 0.09], [3.85, yb - 0.09], [4.4, yb], [3.85, yb + 0.09], [3.37, yb + 0.09]], fc=C["fs"], ec="none", alpha=0.85, zorder=3))
    components.place(ax, "lens", 3.85, yb, 0.5)
    ax.text(3.85, yb + 0.32, "lens", ha="center", va="bottom", fontsize=8.5, color=C["muted"])
    # ------------------------------------------------------------ chip
    cx0, cy0, cx1, cy1 = 4.4, 0.6, 14.4, 7.75
    ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, cy1 - cy0, boxstyle="round,pad=0,rounding_size=0.1", fc=C["chip"], ec=C["chip_ec"], lw=1.4, zorder=1))
    ax.text(cx0 + 0.15, cy1 - 0.15, "Si membrane sensing chip (SOI)", fontsize=11, fontweight="bold", va="top", color=C["ink"])
    # edge coupler and tap
    yin = 4.57
    ax.add_patch(Polygon([[cx0, yin - 0.18], [cx0 + 0.55, yin - 0.04], [cx0 + 0.55, yin + 0.04], [cx0, yin + 0.18]], fc=C["wg_in"], ec="none", zorder=3))
    ax.text(cx0 + 0.3, yin - 0.42, "edge\ncoupler", fontsize=8, ha="center", va="top", color=C["muted"])
    curve(ax, [[cx0 + 0.55, yin], [5.6, yin]], C["wg_in"])
    # tap coupler to the reference detector
    curve(ax, [[5.15, yin - 0.14], [5.5, yin - 0.14], [5.85, yin - 1.3], [6.1, 1.4], [6.9, cy0 + 0.35], [cx1, cy0 + 0.35]], C["wg_ref"], lw=2.0)
    ax.text(6.15, 2.2, "tap (~5 %) to the\nreference detector", fontsize=8, color=C["muted"], ha="left", va="center")
    # dither section: phase heater, 1x2 split, two heated arms, recombine side by side at the port
    curve(ax, [[5.6, yin], [6.55, yin]], C["wg_in"])
    heater(ax, 5.75, 6.4, yin + 0.0, n=4)
    ax.text(6.08, yin + 0.2, "phase dither", fontsize=8.3, color=C["heater"], ha="center")
    ax.add_patch(Rectangle((6.55, yin - 0.2), 0.4, 0.4, fc=C["wg_in"], ec="#8a6a12", lw=0.8, zorder=4))
    ax.text(6.75, yin - 0.28, "1×2", fontsize=8, ha="center", va="top", color=C["muted"])
    if kind == "cmpc":
        ccen, rc = np.array([11.1, 4.0]), 1.85
        cmpc_ = cmpc_cell(ax, ccen, rc, outs=tuple(a.out_angles[: a.outputs]))
        p_in, d0, ports = cmpc_
        port_in = p_in - d0 * 0.22 * rc * 1.4
        approach = port_in - d0 * 0.75
    else:
        x0, y0, d, D, R = 9.0, 4.4, 4.2, 1.35, 5.0
        herriott_cell(ax, x0, y0, d, D, R)
        port_in = np.array([x0 - 0.2, y0])
        approach = port_in + np.array([-0.9, 0.42])
    for sgn, arm in ((1, "a"), (-1, "b")):
        nb = np.array([-(approach - [7.7, yin])[1], (approach - [7.7, yin])[0]])
        nb = nb / np.linalg.norm(nb)
        pts = [[6.95, yin + sgn * 0.1], [7.2, yin + sgn * 0.42], [7.65, yin + sgn * 0.42], approach + sgn * 0.07 * nb, port_in + sgn * 0.05 * nb]
        curve(ax, pts, C["wg_in"], lw=2.0)
        heater(ax, 7.2, 7.65, yin + sgn * 0.42, n=3, amp=0.05)
    ax.text(7.42, yin + 0.62, "angle dither:\nheated arm pair", fontsize=8.3, color=C["heater"], ha="center", va="bottom")
    ax.add_patch(Rectangle((5.6, yin - 0.75), 2.45, 2.15, fc="none", ec=C["heater"], lw=0.9, ls=(0, (3, 2)), zorder=2))
    ax.text(5.65, yin - 0.7, "on-chip dither section", fontsize=8.3, color=C["heater"], va="bottom")
    # bond pads and electrical lines to the heaters
    pads = (5.75, 6.4, 7.2, 7.65)
    for xp in pads:
        ax.add_patch(Rectangle((xp - 0.13, cy1 - 1.05), 0.26, 0.26, fc="#d9c27a", ec="#8a6a12", lw=0.8, zorder=4))
    ax.text(6.7, cy1 - 1.12, "bond pads", fontsize=8, color=C["muted"], ha="center", va="top")
    for xp, yh in zip(pads, (yin, yin, yin + 0.42, yin + 0.42)):
        ax.plot([xp, xp], [cy1 - 1.05, yh + 0.08], color=C["heater"], lw=0.7, alpha=0.7, zorder=2)
    elec(ax, [[2.4, 8.3], [6.7, 8.3], [6.7, cy1 - 0.79]], "heater drive (angle / phase dither)", at=0.3)
    # membrane window
    if kind == "cmpc":
        ax.add_patch(Circle(ccen, rc * 1.42, fc="none", ec=C["mem_ec"], lw=1.0, ls=(0, (5, 3)), zorder=2))
        ax.text(ccen[0] - rc * 0.95, ccen[1] - rc * 1.3, "released membrane\n(backside window)", fontsize=8.3, color=C["mem_ec"], ha="right", va="top")
        ax.text(ccen[0], ccen[1] + 0.15, "CMPC", fontsize=12, fontweight="bold", ha="center", color="#12324f", zorder=4)
        ax.text(ccen[0], ccen[1] - 0.25, "segmented cell, trench DBRs\nstar orbit {24/7}", fontsize=8.3, ha="center", va="top", color="#12324f", zorder=4)
    else:
        ax.add_patch(FancyBboxPatch((8.55, 2.75), 5.1, 3.3, boxstyle="round,pad=0,rounding_size=0.3", fc="none", ec=C["mem_ec"], lw=1.0, ls=(0, (5, 3)), zorder=2))
        ax.text(8.7, 2.85, "released membrane (backside window)", fontsize=8.3, color=C["mem_ec"], ha="left", va="bottom")
        ax.text(11.1, 6.2, "integrated Herriott cell: curved trench-DBR mirrors", fontsize=9.5, fontweight="bold", ha="center", color="#12324f")
    # ------------------------------------------------------------ outputs to the detectors
    det_y = []
    xe = cx1
    if kind == "cmpc":
        for i, (pp, u) in enumerate(ports):
            yt = [6.3, 2.45, 4.0][i] if len(ports) > 1 else 4.0
            pts = [pp, pp + u * 0.45, [xe - 0.7, yt], [xe, yt]]
            curve(ax, pts, C["wg_out"], lw=2.6)
            det_y.append(yt)
    else:
        u_out = np.array([-0.95, -0.32])
        pp = port_in
        pts = [pp, pp + u_out * 0.35, [8.45, 3.3], [9.3, 2.35], [xe - 0.7, 2.35], [xe, 2.35]]
        curve(ax, pts, C["wg_out"], lw=2.6)
        det_y.append(2.35)
    det_y_ref = cy0 + 0.35

    def detector(yt, title, sub, ref=False):
        """Edge coupler -> diverging beam -> collecting lens -> focused onto a bulk photodiode."""
        ax.add_patch(Polygon([[xe, yt - 0.03], [14.95, yt - 0.11], [14.95, yt + 0.11], [xe, yt + 0.03]], fc=C["fs"], ec="none", alpha=0.85, zorder=3))
        ax.add_patch(Polygon([[14.95, yt - 0.11], [15.45, yt - 0.02], [15.45, yt + 0.02], [14.95, yt + 0.11]], fc=C["fs"], ec="none", alpha=0.85, zorder=3))
        components.place(ax, "lens", 14.95, yt, 0.42)
        components.place(ax, "photodiode_ref" if ref else "photodiode", 15.68, yt, 0.4)
        ax.text(16.15, yt + 0.12, title, ha="left", va="center", fontsize=9.5, fontweight="bold", color=C["ink"])
        ax.text(16.15, yt - 0.18, sub, ha="left", va="center", fontsize=8.2, color=C["muted"])
    detector(det_y_ref, "Reference detector", "power normalisation", ref=True)
    for i, yt in enumerate(det_y):
        ax.add_patch(Polygon([[xe - 0.5, yt - 0.04], [xe, yt - 0.18], [xe, yt + 0.18], [xe - 0.5, yt + 0.04]], fc=C["wg_out"], ec="none", zorder=3))
        detector(yt, f"Detector {i + 1}", "bulk InGaAs / MCT")
    # lock-in / DAQ
    box(ax, 15.6, 7.95, 2.4, 1.05, "Lock-in / DAQ", "demodulation, averaging", fs=10)
    for yt in det_y + [det_y_ref]:
        elec(ax, [[16.0, yt - 0.33], [16.0, yt - 0.45], [18.3, yt - 0.45], [18.3, 8.3], [18.0, 8.3]])
    elec(ax, [[2.4, 8.75], [15.6, 8.75]], "dither reference", at=0.8)
    # gas
    gx = 10.6 if kind == "cmpc" else 11.1
    for dx in (-1.2, 0.0, 1.2):
        ax.add_patch(FancyArrowPatch((gx + dx - 0.35, 0.1), (gx + dx + 0.35, 0.1), arrowstyle="-|>", mutation_scale=12, color=C["gas"], lw=1.4, zorder=5))
    ax.text(gx, -0.15, "sample gas flows over and under the membrane", fontsize=8.8, color=C["gas"], ha="center", va="top")
    # legend for the guides
    lg = [("input waveguide (strip-loaded)", C["wg_in"], "-"), ("output-port waveguides", C["wg_out"], "-"), ("reference tap", C["wg_ref"], "-"),
          ("free-space beam (bulk optics)", C["fs"], "-"), ("ray path in the cell", C["beam"], "-"), ("electrical", C["elec"], (0, (4, 2.5))),
          ("heater (thermo-optic)", C["heater"], "-")]
    for i, (lab, col, ls) in enumerate(lg if legend else []):
        y = 2.85 - 0.34 * i
        ax.plot([0.15, 0.6], [y, y], color=col, lw=2.4, ls=ls)
        ax.text(0.7, y, lab, fontsize=8.5, va="center", color=C["ink"])
    cell = "CMPC" if kind == "cmpc" else "integrated Herriott cell"
    if title:
        ax.text(-0.1, 9.3, f"Setup, top view: bulk laser and detectors, on-chip dithering, {cell} (schematic, not to scale)",
                fontsize=13, fontweight="bold", va="bottom", color=C["ink"])
    if attribution and components.used_library():
        ax.text(18.45, -0.55, components.ATTRIBUTION, ha="right", va="top", fontsize=7, color=C["muted"])


def inch_per_unit(ax):
    """Drawn size of one data unit (equal aspect)."""
    ax.apply_aspect()
    bb = ax.get_position().transformed(ax.figure.transFigure)
    return min(bb.width / (XLIM[1] - XLIM[0]), bb.height / (YLIM[1] - YLIM[0])) / ax.figure.dpi


def scale_artists(ax, k):
    """Scale text, line widths, markers and arrow heads by k: the schematic drawn smaller keeps its proportions."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    for c in ax.get_children():
        if isinstance(c, Text):
            c.set_fontsize(c.get_fontsize() * k)
        elif isinstance(c, Line2D):
            c.set_linewidth(c.get_linewidth() * k)
            c.set_markersize(c.get_markersize() * k)
        elif isinstance(c, Patch):
            c.set_linewidth(c.get_linewidth() * k)
            if isinstance(c, FancyArrowPatch):
                c.set_mutation_scale(c.get_mutation_scale() * k)


def figure(kind, a):
    fig, ax = plt.subplots(figsize=(16, 8.4), dpi=130)
    draw(ax, kind, a)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"setup_{kind}.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kinds", nargs="*", default=["cmpc", "herriott"])
    ap.add_argument("--outputs", type=int, default=2, help="CMPC output ports (1-3)")
    ap.add_argument("--out-angles", type=float, nargs="+", default=[45, -45, 0], help="CMPC output-port facet angles, deg")
    ap.add_argument("--no-modules", dest="modules", action="store_false", help="omit the light-source (red) and detection (blue) blocks")
    return ap


if __name__ == "__main__":
    a = parser().parse_args()
    for k in a.kinds:
        figure(k, a)
        print("wrote", OUT / f"setup_{k}.pdf")
