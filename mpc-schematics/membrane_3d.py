"""Oblique 3D schematics of the chip-scale CMPC: a free-standing Si membrane with a segmented cell whose facets are
etched-trench DBR mirrors, with a cutaway, and a to-scale close-up of one DBR.

    python membrane_3d.py                       both figures into figs/ (PNG and PDF)
    python membrane_3d.py chip                  chip | dbr | herriott
    python membrane_3d.py herriott --R 5 --N 22 --M 5 --A 0.5 --phase 90
    python membrane_3d.py --facets 12 --periods 3 --azim -40 --elev 28

chip   schematic, NOT to scale: membrane thickness, trench widths and the DBR band are exaggerated so they are visible
       next to a 1 cm cell (the real DBR is ~2 µm deep against a 5 mm radius). Layer stack: Si handle, buried oxide,
       Si device layer released over a backside window. Trenches end short of the facet corners: the corner bridges
       are what holds the inner membrane once every trench is etched through (an assumption of the drawing; the
       simulations do not model them). One facet has no DBR over the port width (input port); the beam drawn is the
       star orbit {N/m} of the cmpc-sim launch.
dbr    to scale: Si membrane thickness, air-gap and tooth widths from cmpc_sim.membrane_mode and cmpc_sim.DBR
       (quarter-wave, orders 1, 1), with the slab-mode intensity profile on the cut face.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "cmpc-sim" / "cmpc"))
from render3d import Scene, ngon  # noqa: E402
import cmpc_sim as cs  # noqa: E402

OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
COL = dict(si="#8fa3c4", si_sec="#c9d4e6", handle="#9a9fa6", handle_sec="#c8ccd1", box="#e8d9a8", box_sec="#f3ead0",
           beam="#d0352b", port="#1f8a4c", strip="#d4a63c", strip_sec="#ead39b", strip_ink="#8a6a12", out="#7a3fb0", field="#7a3fb0", ink="#1b2328")
plt.rcParams.update({"font.size": 10, "font.family": ["Inter", "DejaVu Sans"]})


def annotate(ax, to_px, P_anchor, P_text, text, color=COL["ink"], ha="center", **kw):
    a, t = np.ravel(to_px(np.atleast_2d(P_anchor))), np.ravel(to_px(np.atleast_2d(P_text)))
    ax.annotate(text, (a[0], a[1]), (t[0], t[1]), ha=ha, va="center", color=color, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=2, shrinkB=0), annotation_clip=False, **kw)


def chip(a):
    N, Np, m = a.facets, a.periods, a.star
    t, g = 0.05, 0.040                                   # exaggerated membrane thickness and air-gap width (cell radius 1)
    tooth = g * 193 / 388                                 # keeps the Si 250 nm TM0 gap : tooth ratio
    tb, th = 0.045, 0.30                                  # buried oxide, handle
    a0 = np.cos(np.pi / N)                                # facet apothem, circumradius 1
    p = N // 2                                            # port facet (at 180°, away from the cut)
    band = a0 + Np * (g + tooth)
    slots = unary_union([ngon(a0 + j * (g + tooth) + g, N).difference(ngon(a0 + j * (g + tooth), N)) for j in range(Np)])
    bw = 0.028
    bridges = []
    for k in range(N):
        ang = 2 * np.pi * k / N - np.pi / N
        u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
        q = [0.9 * u - bw / 2 * v, (band + 0.2) * u - bw / 2 * v, (band + 0.2) * u + bw / 2 * v, 0.9 * u + bw / 2 * v]
        bridges.append(Polygon(q))
    # input port: no trenches along the input beam (it enters at the launch angle, along the first star chord)
    ks = (p + m * np.arange(N + 1)) % N
    mids = np.c_[a0 * np.cos(2 * np.pi * ks / N), a0 * np.sin(2 * np.pi * ks / N)]
    d0 = (mids[1] - mids[0]) / np.linalg.norm(mids[1] - mids[0])
    wp = 0.10
    ring = ngon(band + 0.02, N).difference(ngon(a0 - 0.02, N))
    in_win = LineString([mids[0] + 0.2 * d0, mids[0] - 1.0 * d0]).buffer(wp / 2, cap_style=2).intersection(ring)
    # output ports: past the first n_out periods the membrane is solid over the port width (partial transmission)
    n_out = a.out_periods if a.out_periods is not None else Np // 2
    out_k = [round(N * f) % N for f in (0.625, 0.25, 0.375, 0.125)][: a.outputs]
    out_win = []
    for k in out_k:
        ang = 2 * np.pi * k / N
        u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
        r0 = a0 + n_out * (g + tooth) - 0.25 * g
        out_win.append(Polygon([r0 * u - wp / 2 * v, (band + 0.05) * u - wp / 2 * v, (band + 0.05) * u + wp / 2 * v, r0 * u + wp / 2 * v]))
    slots = slots.difference(unary_union(bridges)).difference(unary_union([in_win] + out_win))

    # tapered strip-loaded waveguides on the outer membrane: wide at the port, single-mode width further out
    w1, w2, lt, ts = wp, 0.035, 0.32, 0.6 * t

    def taper(start, u):
        v_ = np.array([-u[1], u[0]])
        far = start + 4 * u
        return unary_union([Polygon([start - w1 / 2 * v_, start + lt * u - w2 / 2 * v_, start + lt * u + w2 / 2 * v_, start + w1 / 2 * v_]),
                            Polygon([start + lt * u - w2 / 2 * v_, far - w2 / 2 * v_, far + w2 / 2 * v_, start + lt * u + w2 / 2 * v_])])

    s_in = mids[0] - d0 * ((band - a0) / abs(d0 @ (mids[0] / a0)) + 0.01)
    guides = [(taper(s_in, -d0), s_in, -d0)]
    for k in out_k:
        u = np.array([np.cos(2 * np.pi * k / N), np.sin(2 * np.pi * k / N)])
        guides.append((taper((band + 0.01) * u, u), (band + 0.01) * u, u))

    L = 1.75
    chipsq = box(-L, -L, L, L)
    cut = box(0, -L - 1, L + 1, 0)                        # cutaway quadrant x > 0, y < 0
    win = box(-1.38, -1.38, 1.38, 1.38)                    # backside window in the handle
    undercut = box(-1.46, -1.46, 1.46, 1.46)              # oxide etched slightly past it

    def on_cut(p_, q_, tol=1e-9):
        return (abs(p_[0]) < tol and abs(q_[0]) < tol and max(p_[1], q_[1]) <= tol) or \
               (abs(p_[1]) < tol and abs(q_[1]) < tol and min(p_[0], q_[0]) >= -tol)

    sc = Scene(azim=a.azim, elev=a.elev)
    sc.prism(chipsq.difference(win).difference(cut), -tb - th, -tb, COL["handle"], on_cut, COL["handle_sec"])
    sc.prism(chipsq.difference(undercut).difference(cut), -tb, 0, COL["box"], on_cut, COL["box_sec"])
    mem = chipsq.difference(slots).difference(cut)
    sc.prism(mem, 0, t, COL["si"], on_cut, COL["si_sec"])
    # guided-mode intensity on the two cut faces: cell, then decaying through the DBR (heights scaled to the drawn membrane)
    mode_g = cs.membrane_mode("Si", a.thickness * 1e-9, a.lam * 1e-6, a.pol, with_group_index=False)
    xf, If = cs.field_profile(mode_g, 1.5)
    zsc = t / (a.thickness * 1e-9)
    ext = (xf.max() - a.thickness * 1e-9 / 2) * zsc
    Rm = band + 0.0

    def glow_alpha(s_, tt):
        z = (-ext + tt * (t + 2 * ext) - t / 2) / zsc
        r = s_ * Rm
        return 0.62 * np.interp(z, xf, If / If.max()) * np.where(r <= a0, 1.0, np.exp(-3 * (r - a0) / (band - a0)))
    sc_glow = [([0, -1e-4, -ext], [Rm, 0, 0], [0, 0, t + 2 * ext]), ([1e-4, 0, -ext], [0, -Rm, 0], [0, 0, t + 2 * ext])]
    strips = unary_union([gd[0] for gd in guides]).intersection(chipsq).difference(cut)
    sc.prism(strips, t, t + ts, COL["strip"], on_cut, COL["strip_sec"])

    if not a.no_glow:
        for O, U, V in sc_glow:
            sc.glow(O, U, V, glow_alpha)
    # star orbit {N/m} through the facet midpoints, starting at the port, on the membrane surface
    keep = mem.buffer(1e-6).union(ngon(a0, N)).difference(cut)
    for seg in zip(mids[:-1], mids[1:]):
        piece = LineString(seg).intersection(keep)
        for ls in getattr(piece, "geoms", [piece]):
            if ls.length > 0 and ls.geom_type == "LineString":
                xy = np.asarray(ls.coords)
                sc.line(np.c_[xy, np.full(len(xy), t + 1e-3)], COL["beam"], width=1.4)
    p_in = s_in - 0.55 * d0
    sc.line(np.r_[[np.r_[p_in, t + ts + 1e-3]], [np.r_[s_in, t + ts + 1e-3]]], COL["port"], width=2.2)
    sc.line(np.r_[[np.r_[s_in, t + 1e-3]], [np.r_[mids[0], t + 1e-3]]], COL["port"], width=2.2)
    outs = []
    for _, st, u in guides[1:]:
        e = LineString([st, st + 4 * u]).intersection(chipsq.difference(cut))
        if e.is_empty:
            continue
        xy = np.asarray(e.coords)
        sc.line(np.c_[xy, np.full(len(xy), t + ts + 1e-3)], COL["out"], width=2.0)
        outs.append((xy[0], xy[-1]))

    img, to_px = sc.render(width=a.width)
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    A0, B0 = to_px(np.atleast_2d(np.r_[p_in, t + ts])), to_px(np.atleast_2d(np.r_[s_in - 0.1 * d0, t + ts]))
    ax.annotate("", (B0[0][0], B0[1][0]), (A0[0][0], A0[1][0]), arrowprops=dict(arrowstyle="-|>", color=COL["port"], lw=2))
    for st, en in outs:
        A_, B_ = to_px(np.atleast_2d(np.r_[st + 0.6 * (en - st), t + ts])), to_px(np.atleast_2d(np.r_[en, t + ts]))
        ax.annotate("", (B_[0][0], B_[1][0]), (A_[0][0], A_[1][0]), arrowprops=dict(arrowstyle="-|>", color=COL["out"], lw=2))
    k1 = N * 3 // 8
    fm = a0 * np.array([np.cos(2 * np.pi * k1 / N), np.sin(2 * np.pi * k1 / N)])
    annotate(ax, to_px, np.r_[fm * (a0 + 0.12) / a0, t], [-0.9, 2.15, t], f"etched-trench DBR on every facet ({Np} periods, drawn wide)")
    annotate(ax, to_px, np.r_[mids[0] - 0.08 * d0, t], [-2.3, -0.5, t], f"input port: no DBR, beam enters at χ = {90 - 180 * m / N:.1f}°", COL["port"])
    annotate(ax, to_px, np.r_[s_in - 0.2 * d0, t + ts], [-2.1, 0.9, t], "tapered strip-loaded waveguides\n(strip on the membrane, to the chip edge)", COL["strip_ink"])
    if out_k:
        u = np.array([np.cos(2 * np.pi * out_k[0] / N), np.sin(2 * np.pi * out_k[0] / N)])
        mode = cs.membrane_mode("Si", a.thickness * 1e-9, a.lam * 1e-6, a.pol, with_group_index=False)
        D = cs.DBR(n_tooth=mode.neff, lam_design=a.lam * 1e-6, N=n_out, m_gap=1, m_tooth=1, slab_pol=a.pol, bounce_loss=0)
        T0 = 1 - float(D.R(a.lam * 1e-6, np.array(0.0)))
        annotate(ax, to_px, np.r_[(a0 + 0.5 * (band - a0)) * u, t], [0.6, 2.55, t],
                 f"output ports: {n_out} of {Np} periods, T(0°) = {T0 * 100:.0f} %\n(no output above the {np.degrees(np.arcsin(1 / mode.neff)):.0f}° TIR angle)", COL["out"])
    annotate(ax, to_px, np.r_[0.15, 0.55, t], [1.2, 2.25, t], f"star orbit {{{N}/{m}}} in the membrane", COL["beam"])
    annotate(ax, to_px, [0.9, -0.02, t / 2], [2.45, 0.9, 0.0], "section through the DBR\n(gaps and teeth)")
    annotate(ax, to_px, [0.45, 0, t], [2.6, 0.2, -0.25], f"free-standing Si membrane\n(thickness ×{t / 250e-9 * 5e-3:.0f})")
    annotate(ax, to_px, [1.6, 0, -tb / 2], [2.7, -0.6, -0.4], "buried oxide")
    annotate(ax, to_px, [1.6, 0, -tb - th / 2], [2.6, -1.4, -0.6], "Si handle")
    annotate(ax, to_px, [0, -1.0, -tb - th], [0.4, -2.6, -0.65], "backside window: membrane released")
    b0 = 2 * np.pi * (N * 5 // 8) / N - np.pi / N
    annotate(ax, to_px, [(a0 + 0.1) * np.cos(b0) / np.cos(np.pi / N), (a0 + 0.1) * np.sin(b0) / np.cos(np.pi / N), t],
             [-2.6, -1.3, t], "corner bridges carry the cell\n(trenches stop short of the corners)")
    ax.text(0.0, 1.03, f"Chip-scale CMPC, {N}-facet cell in a free-standing Si membrane (schematic, not to scale)",
            transform=ax.transAxes, va="bottom", fontsize=12, color=COL["ink"], fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"membrane_chip_3d.{ext}", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def dbr(a):
    lam, d_t = a.lam * 1e-6, a.thickness * 1e-9
    mode = cs.membrane_mode("Si", d_t, lam, a.pol, with_group_index=False)
    D = cs.DBR(n_tooth=mode.neff, lam_design=lam, N=a.periods, m_gap=1, m_tooth=1, slab_pol=a.pol, bounce_loss=0)
    um = 1e6
    t, g, tooth = d_t * um, D.d_gap * um, D.d_tooth * um
    cav, tail, Ly = 1.8, 0.9, 2.4
    z0, z1 = -t / 2, t / 2
    sc = Scene(azim=a.azim - 30, elev=a.elev)
    on_front = lambda p_, q_: abs(p_[1]) < 1e-9 and abs(q_[1]) < 1e-9
    sc.prism(box(-cav, 0, 0, Ly), z0, z1, COL["si"], on_front, COL["si_sec"])
    x = 0.0
    teeth = []
    for _ in range(D.N):
        x += g
        sc.prism(box(x, 0, x + tooth, Ly), z0, z1, COL["si"], on_front, COL["si_sec"])
        teeth.append(x)
        x += tooth
    x_end = x
    sc.prism(box(x_end + tail, 0, x_end + tail + 0.5, Ly), z0, z1, COL["si"], on_front, COL["si_sec"])

    xf, zf = cs.field_profile(mode, 3.0)
    zf_um, I = xf * um, zf / zf.max()
    if not a.no_glow:                                     # guided-mode intensity on the cut face, decaying through the mirror
        ez = zf_um.max()
        sc.glow([-cav, -1e-3, -ez], [cav + x_end, 0, 0], [0, 0, 2 * ez],
                lambda s_, tt: 0.55 * np.interp(-ez + 2 * ez * tt, zf_um, I) *
                np.where(s_ * (cav + x_end) <= cav, 1.0, np.exp(-3 * (s_ * (cav + x_end) - cav) / x_end)))
    sc.line(np.c_[-cav * 0.55 - 0.8 * I, np.full(len(I), -2e-3), zf_um], COL["field"], width=1.6, depth_test=False)
    sc.line([[-cav * 0.55, -2e-3, zf_um.min()], [-cav * 0.55, -2e-3, zf_um.max()]], "#8a8f96", width=0.6, depth_test=False)

    chi = np.radians(a.chi)
    yh, Lr = Ly * 0.55, cav * 0.9
    zt = z1 + 1e-3
    pin = [-Lr, yh - Lr * np.tan(chi), zt]
    pout = [-Lr, yh + Lr * np.tan(chi), zt]
    sc.line([pin, [0, yh, zt]], COL["beam"], width=2.0)
    sc.line([[0, yh, zt], pout], COL["beam"], width=2.0)

    img, to_px = sc.render(width=a.width, extra_points=[[-cav - 0.3, 0, z0 - 0.6], [x_end + tail + 0.8, Ly, z1 + 0.6]])
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    P = lambda *xyz: np.ravel(to_px(np.atleast_2d(xyz)))
    for p0, p1, col in ((pin, [-0.25 * Lr, yh - 0.25 * Lr * np.tan(chi), zt], COL["beam"]), ([0, yh, zt], [-0.75 * Lr, yh + 0.75 * Lr * np.tan(chi), zt], COL["beam"])):
        A_, B_ = P(*p0), P(*p1)
        ax.annotate("", (B_[0], B_[1]), (A_[0], A_[1]), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
    annotate(ax, to_px, [-0.5, yh - 0.05, zt], [-cav - 1.2, Ly + 0.2, zt], f"slab mode {a.pol}₀, χ = {a.chi:g}°\n(n_eff = {mode.neff:.3f})", COL["beam"])
    annotate(ax, to_px, [g / 2, 0, z0], [g / 2 - 0.3, -0.75, z0 - 0.35], f"air gap {g * 1e3:.0f} nm")
    annotate(ax, to_px, [teeth[0] + tooth / 2, 0, z0], [teeth[0] + 1.0, -0.75, z0 - 0.35], f"Si tooth {tooth * 1e3:.0f} nm")
    annotate(ax, to_px, [-cav, 0, 0], [-cav - 0.2, -0.9, 0.35], f"Si membrane {t * 1e3:.0f} nm")
    annotate(ax, to_px, [-cav * 0.55 - 0.8, -2e-3, 0], [-cav - 0.1, -0.2, -0.75], f"|{'H' if a.pol == 'TM' else 'E'}|² across the membrane\n(Γ = {mode.Gamma:.2f} in the gas)", COL["field"])
    annotate(ax, to_px, [x_end + tail / 2, Ly * 0.7, z1], [x_end + 1.3, Ly + 0.7, z1], "air (cmpc_sim.DBR output medium)")
    R0 = float(D.R(lam, np.array(0.0)))
    ax.text(0.0, 1.17, f"Etched-trench DBR, {D.N} periods, to scale: Si {t * 1e3:.0f} nm {a.pol}₀ at {a.lam:g} µm, "
            f"R(0°) = {R0:.4f}", transform=ax.transAxes, va="bottom", fontsize=12, color=COL["ink"], fontweight="bold")
    ax.text(0.0, 1.12, f"period {(g + tooth) * 1e3:.0f} nm, mirror depth {x_end:.2f} µm; total internal reflection at the "
            f"first gap above χ = {np.degrees(np.arcsin(1 / mode.neff)):.1f}°", transform=ax.transAxes, va="bottom",
            fontsize=10, color="#4a565c")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"membrane_dbr_3d.{ext}", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def herriott(a):
    """Integrated Herriott cell in the membrane: two curved trench-DBR mirrors, centre port with a V of tapered
    strip-loaded waveguides, exact ray path, cutaway through mirror 2 and the cell. Lengths in mm (drawn)."""
    from herriott_planar import PlanarHerriott
    c = PlanarHerriott(R=a.R * 1e-3, N=a.N, M=a.M, A=a.A * 1e-3, phase=a.phase, lam=a.lam * 1e-6,
                       thickness=a.thickness * 1e-9, pol=a.pol, periods=a.periods)
    tr = c.trace()
    mm = 1e3
    R, d, A, D, y0, wp = c.R * mm, c.d * mm, c.A * mm, c.D * mm, c.y0 * mm, c.wp * mm
    Np = a.periods
    band = 0.22                                           # drawn DBR band (real ~2.3 µm)
    g = band / Np * c.dbr.d_gap / (c.dbr.d_gap + c.dbr.d_tooth)
    tooth = band / Np - g
    t, tb, th = 0.05, 0.04, 0.28
    X0, X1, Y1 = -2.2, d + 0.9, D + 1.0
    chipsq = box(X0, -Y1, X1, Y1)
    win = box(-0.7, -(D + 0.55), d + 0.55, D + 0.55)
    undercut = box(-0.76, -(D + 0.61), d + 0.61, D + 0.61)
    Xc = d / 2
    cut = box(Xc, -Y1 - 1, X1 + 1, 0)

    def on_cut(p_, q_, tol=1e-9):
        return (abs(p_[0] - Xc) < tol and abs(q_[0] - Xc) < tol and max(p_[1], q_[1]) <= tol) or \
               (abs(p_[1]) < tol and abs(q_[1]) < tol and min(p_[0], q_[0]) >= Xc - tol)

    # trenches: arcs concentric with each mirror, behind it, over the aperture |Y| <= D
    slots = []
    for k, (cx, side) in enumerate(((R, -1), (d - R, 1))):
        lim = box(-50, -D, cx, D) if side < 0 else box(cx, -D, 50, D)
        for j in range(Np):
            r0 = R + j * (g + tooth)
            ring = Point(cx, 0).buffer(r0 + g, 512).difference(Point(cx, 0).buffer(r0, 512)).intersection(lim)
            if k == 0:
                ring = ring.difference(box(-50, y0 - wp / 2, 50, y0 + wp / 2))
            slots.append(ring)
    slots = unary_union(slots)

    # waveguides behind mirror 1: input along the launch direction, output along the exit direction
    P = tr["points"] * mm
    p0 = P[0]
    guides, beams_io = [], []
    for u, kind in ((-tr["in_dir"], "in"), (tr["exit_dir"], "out")):
        if kind == "out" and tr["exit"] != "port":
            continue
        v = np.array([-u[1], u[0]])
        st = p0 + u * (band + 0.03)
        w1, w2, lt = wp, wp * 0.35, 0.55
        far = st + 6 * u
        guides.append(unary_union([Polygon([st - w1 / 2 * v, st + lt * u - w2 / 2 * v, st + lt * u + w2 / 2 * v, st + w1 / 2 * v]),
                                   Polygon([st + lt * u - w2 / 2 * v, far - w2 / 2 * v, far + w2 / 2 * v, st + lt * u + w2 / 2 * v])]))
        beams_io.append((kind, st, u))
    ts = 0.6 * t

    sc = Scene(azim=a.azim - 30, elev=a.elev + 4)
    sc.prism(chipsq.difference(win).difference(cut), -tb - th, -tb, COL["handle"], on_cut, COL["handle_sec"])
    sc.prism(chipsq.difference(undercut).difference(cut), -tb, 0, COL["box"], on_cut, COL["box_sec"])
    mem = chipsq.difference(slots).difference(cut)
    sc.prism(mem, 0, t, COL["si"], on_cut, COL["si_sec"])
    sc.prism(unary_union(guides).intersection(chipsq).difference(cut), t, t + ts, COL["strip"], on_cut, COL["strip_sec"])

    if not a.no_glow:                                     # guided-mode intensity on the cut faces
        mode_g = cs.membrane_mode("Si", a.thickness * 1e-9, a.lam * 1e-6, a.pol, with_group_index=False)
        xf, If = cs.field_profile(mode_g, 1.5)
        zsc = t / (a.thickness * 1e-9)
        ext = (xf.max() - a.thickness * 1e-9 / 2) * zsc
        prof = lambda tt: np.interp((-ext + tt * (t + 2 * ext) - t / 2) / zsc, xf, If / If.max())
        L1 = d + band - Xc
        sc.glow([Xc, -1e-4, -ext], [L1, 0, 0], [0, 0, t + 2 * ext],
                lambda s_, tt: 0.62 * prof(tt) * np.where(Xc + s_ * L1 <= d, 1.0, np.exp(-3 * (Xc + s_ * L1 - d) / band)))
        L2 = 1.25 * A
        sc.glow([Xc + 1e-4, 0, -ext], [0, -L2, 0], [0, 0, t + 2 * ext],
                lambda s_, tt: 0.62 * prof(tt) * np.where(s_ * L2 <= A, 1.0, np.exp(-4 * (s_ * L2 - A) / A)))

    keep = chipsq.difference(cut)
    for i in range(len(P) - 1):
        piece = LineString(P[i:i + 2]).intersection(keep)
        for ls in getattr(piece, "geoms", [piece]):
            if ls.length > 0 and ls.geom_type == "LineString":
                xy = np.asarray(ls.coords)
                sc.line(np.c_[xy, np.full(len(xy), t + 1e-3)], COL["beam"], width=1.1)
    io_px = []
    for kind, st, u in beams_io:
        e = LineString([st, st + 6 * u]).intersection(chipsq.difference(cut))
        xy = np.asarray(e.coords)
        col = COL["port"] if kind == "in" else COL["out"]
        sc.line(np.c_[xy, np.full(len(xy), t + ts + 1e-3)], col, width=2.0)
        sc.line([[*st, t + 1e-3], [*p0, t + 1e-3]], col, width=2.0)
        io_px.append((kind, xy[0], xy[-1]))

    img, to_px = sc.render(width=a.width)
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    for kind, s0, s1 in io_px:
        col = COL["port"] if kind == "in" else COL["out"]
        A_, B_ = (to_px(np.atleast_2d(np.r_[s1, t + ts])), to_px(np.atleast_2d(np.r_[s0 + 0.3 * (s1 - s0), t + ts])))
        if kind == "out":
            A_, B_ = B_, A_
        ax.annotate("", (B_[0][0], B_[1][0]), (A_[0][0], A_[1][0]), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    bud = c.budget(tr)
    ang = np.degrees(np.arccos(np.clip(-tr["in_dir"] @ tr["exit_dir"], -1, 1)))
    annotate(ax, to_px, [-band / 2, 0.9 * D, t], [-0.6, Y1 + 0.9, t], f"mirror 1: curved trench DBR, {Np} periods\n(R = {R:g} mm, drawn wide)")
    annotate(ax, to_px, [d + band / 2, 0.6 * D, t], [d + 0.9, Y1 + 0.4, t], "mirror 2")
    annotate(ax, to_px, [p0[0] - 0.02, p0[1], t], [-1.6, -Y1 - 0.9, t], f"centre port: no DBR over {wp * 1e3:.0f} µm", COL["port"])
    annotate(ax, to_px, [*(beams_io[0][1] + beams_io[0][2] * 0.8), t + ts], [-2.9, -0.4, t],
             f"tapered strip-loaded waveguides,\ninput and output {ang:.0f}° apart", COL["strip_ink"])
    annotate(ax, to_px, [d * 0.3, 0.2 * A, t], [d * 0.5, Y1 + 1.4, t],
             f"{tr['n_hits'] + 1} passes, M = {c.M}: path {tr['path'] * 100:.1f} cm, out {bud['T_out'] * 100:.0f} %", COL["beam"])
    annotate(ax, to_px, [d + 0.02, 0, t / 2], [X1 + 0.9, 0.4, 0], "section through mirror 2")
    annotate(ax, to_px, [d * 0.75, 0, -tb - th], [X1 + 0.5, -1.6, -0.5], "backside window: membrane released")
    ax.text(0.0, 1.03, f"Integrated Herriott cell in a free-standing Si membrane (schematic, not to scale): N = {c.N}, "
            f"M = {c.M}, φ = {c.phase:g}°, d = {d:.2f} mm", transform=ax.transAxes, va="bottom", fontsize=12,
            color=COL["ink"], fontweight="bold")
    for ext_ in ("png", "pdf"):
        fig.savefig(OUT / f"membrane_herriott_3d.{ext_}", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["chip", "dbr", "herriott"])
    ap.add_argument("--facets", type=int, default=24)
    ap.add_argument("--star", type=int, default=7)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--outputs", type=int, default=2, help="number of output ports (0-4)")
    ap.add_argument("--out-periods", type=int, default=None, help="DBR periods kept at an output port (default periods // 2)")
    ap.add_argument("--thickness", type=float, default=250, help="membrane thickness, nm")
    ap.add_argument("--lam", type=float, default=1.55, help="wavelength, µm")
    ap.add_argument("--pol", default="TM", choices=("TE", "TM"))
    ap.add_argument("--chi", type=float, default=20.0, help="angle of incidence drawn in the DBR close-up, deg")
    ap.add_argument("--azim", type=float, default=-38.0)
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1800, help="image width in pixels")
    ap.add_argument("--no-glow", action="store_true", help="omit the guided-mode glow on the cut faces")
    ap.add_argument("--R", type=float, default=5.0, help="Herriott: mirror radius, mm")
    ap.add_argument("--N", type=int, default=22, help="Herriott: passes to re-entry")
    ap.add_argument("--M", type=int, default=5, help="Herriott: turns")
    ap.add_argument("--A", type=float, default=0.5, help="Herriott: pattern half-width, mm")
    ap.add_argument("--phase", type=float, default=90.0, help="Herriott: launch phase, deg (90: centre port)")
    args = ap.parse_args()
    for f in args.figs:
        {"chip": chip, "dbr": dbr, "herriott": herriott}[f](args)
        print("wrote", f)
