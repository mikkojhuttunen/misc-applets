"""Simple schematics of the multipass-cell structures used in the simulations, drawn from the simulators' own classes
(so the drawing is the geometry that is traced, not a sketch of it):

    cmpc-sim (chaotic multipass cell in a Si membrane), cmpc_sim.SegmentedCell / Stadium
        regular       24 flat facets, circumradius 5 mm (1 cm cell), star {24/7} launch (chi = 37.5 deg)
        tilt          the same with 1 deg rms random facet tilts (seed 1)
        curved        every facet convex with rho = 5 cm (the 'chaotic' cell of run_figures / run_midir)
        stadium       r = 5 mm, straights a = 5 mm (smooth, chaotic reference)
        mirror        etched-trench DBR on each facet: cavity | [air gap, Si tooth] x N | air, λ/4 orders (1, 1)
    general-mpc, gmpc.planar / gmpc.herriott
        stadiums      smooth, 16 facets per cap, facets + 1 mrad tilt, facets + curved caps (1/rho = 40 /m rms)
        Herriott      R = 0.5 m, N = 30, M = 7, A = 12 mm (side view and spot pattern)
        astigmatic    Lissajous cell re-entrant after 50 hits (Mx, My = 11, 13)

Facet tilts and curvature sags (a few µm over a 1.3 mm facet) are invisible at cell scale, so the cmpc panels are
drawn twice: true scale, and with the perturbation exaggerated by EXAG (marked in the title).

    python draw_cells.py              all figures into figs/
    python draw_cells.py cmpc         cmpc | dbr | stadium | herriott
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cmpc-sim" / "cmpc"))
sys.path.insert(0, str(ROOT / "general-mpc"))
import cmpc_sim as cs  # noqa: E402
from gmpc import herriott as H, planar as P, trace2d as T  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

CMPC = dict(R=5e-3, N=24, star_m=7, tilt_deg=1.0, curv_radius=0.05, port_w=30e-6, lam=1.55e-6)
EXAG = 10.0
WALL, RAY, PORT = "#1f4e79", "#c0504d", "#4f9d4f"


def chi_star():
    return np.pi / 2 - np.pi * CMPC["star_m"] / CMPC["N"]


def launch_from_facet0(cell, chi, n_hits):
    """Ray from the middle of facet 0 at angle chi from the inward normal, as cmpc_sim.trace_rays launches."""
    x0, y0, nx, ny = cell.point_at(np.array([cell.h]))
    ux, uy = -nx[0], -ny[0]
    dx, dy = ux * np.cos(chi) - uy * np.sin(chi), ux * np.sin(chi) + uy * np.cos(chi)
    return cs.trace_path(cell, x0[0], y0[0], dx, dy, n_hits)


def draw_cmpc_cell(ax, cell, title, n_hits=48):
    x, y = cell.outline(2400)
    ax.plot(x * 1e3, y * 1e3, color=WALL, lw=1.6)
    xs, ys = launch_from_facet0(cell, chi_star(), n_hits)
    ax.plot(xs * 1e3, ys * 1e3, color=RAY, lw=0.6, alpha=0.8)
    px, py, _, _ = cell.point_at(np.array([cell.h]))
    ax.plot(px * 1e3, py * 1e3, "o", color=PORT, ms=5, zorder=5)
    ax.annotate("port", (px[0] * 1e3, py[0] * 1e3), (px[0] * 1e3 + 0.6, py[0] * 1e3 + 1.2), color=PORT, fontsize=8)
    ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")


def fig_cmpc():
    R, N = CMPC["R"], CMPC["N"]
    t = np.radians(CMPC["tilt_deg"])
    k = 1 / CMPC["curv_radius"]
    cells = [(cs.SegmentedCell(R, N), f"regular: {N} flat facets, Ø {2*R*1e3:.0f} mm\nstar {{{N}/{CMPC['star_m']}}} launch, χ = {np.degrees(chi_star()):.1f}°"),
             (cs.SegmentedCell(R, N, tilt_rms=t, seed=1), f"tilted facets, {CMPC['tilt_deg']:g}° rms\n(true scale)"),
             (cs.SegmentedCell(R, N, curvature=k), f"curved facets ρ = {CMPC['curv_radius']*100:.0f} cm (chaotic)\n(true scale, sag {cs.SegmentedCell(R, N).h**2/2/CMPC['curv_radius']*1e6:.1f} µm)"),
             (cs.SegmentedCell(R, N, curvature=k * 25), "curved facets, curvature ×25\n(exaggerated to show the shape)"),
             (cs.SegmentedCell(R, N, tilt_rms=t * EXAG, seed=1), f"tilted facets ×{EXAG:.0f} ({CMPC['tilt_deg']*EXAG:g}° rms)\n(exaggerated)"),
             (cs.Stadium(R, R), "stadium r = 5 mm, a = 5 mm\n(smooth, chaotic reference)")]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.2))
    for ax, (c, ttl) in zip(axs.flat, cells):
        if isinstance(c, cs.Stadium):
            x, y = c.outline(800)
            ax.plot(x * 1e3, y * 1e3, color=WALL, lw=1.6)
            th = 0.9
            xs, ys = cs.trace_path(c, -c.a / 2, -c.r + 1e-9, np.sin(th), np.cos(th), 40)
            ax.plot(xs * 1e3, ys * 1e3, color=RAY, lw=0.6, alpha=0.8)
            ax.set(title=ttl, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
        else:
            draw_cmpc_cell(ax, c, ttl)
    fig.suptitle("cmpc-sim cells (Si membrane, in-plane DBR mirror on every facet); red: first 48 bounces of one ray")
    fig.tight_layout()
    fig.savefig(OUT / "cmpc_cells.png")
    plt.close(fig)


def fig_dbr(N=4, t=250e-9, pol="TM"):
    """Top view of one facet mirror: the stack drawn to scale for Si t TM0 at 1.55 µm, quarter-wave orders (1, 1)."""
    lam = CMPC["lam"]
    m = cs.membrane_mode("Si", t, lam, pol, with_group_index=False)
    d = cs.DBR(n_tooth=m.neff, lam_design=lam, N=N, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    x = 0.0
    a.add_patch(Rectangle((-1.5, 0), 1.5, 3, color="#d9d9d9"))
    a.text(-0.75, 3.2, f"cell (membrane)\nn_eff = {m.neff:.3f}", ha="center", fontsize=8)
    for i in range(N):
        a.add_patch(Rectangle((x, 0), d.d_gap * 1e6, 3, color="white", ec="k", lw=0.5))
        x += d.d_gap * 1e6
        a.add_patch(Rectangle((x, 0), d.d_tooth * 1e6, 3, color="#8faadc"))
        x += d.d_tooth * 1e6
    a.add_patch(Rectangle((x, 0), 0.8, 3, color="white", ec="k", lw=0.5, ls=":"))
    a.text(x + 0.4, 3.2, "air", ha="center", fontsize=8)
    a.annotate(f"air gap {d.d_gap*1e9:.0f} nm", (d.d_gap * 1e6 / 2, 0), (-1.4, -0.7), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.6))
    a.annotate(f"Si tooth {d.d_tooth*1e9:.0f} nm", (d.d_gap * 1e6 + d.d_tooth * 1e6 / 2, 0), (1.2, -0.7), fontsize=8,
               color="#1f4e79", arrowprops=dict(arrowstyle="->", lw=0.6, color="#1f4e79"))
    a.annotate("", (1.2, 1.6), (-1.1, 0.4), arrowprops=dict(arrowstyle="->", color=RAY, lw=1.5))
    a.text(-1.2, 0.2, "χ", color=RAY, fontsize=10)
    a.set(xlim=(-1.6, x + 1.0), ylim=(-1.4, 3.8), xlabel="distance into the mirror (µm)", yticks=[],
          title=f"Etched-trench DBR, {N} periods, Si {t*1e9:.0f} nm {pol}0 at {lam*1e6:.2f} µm (top view, to scale)")
    sg = np.linspace(0, 0.999, 800)
    for n_ in (2, 3, 4, 6):
        dd = cs.DBR(n_tooth=m.neff, lam_design=lam, N=n_, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
        ax[1].plot(np.degrees(np.arcsin(sg)), dd.R(lam, sg), label=f"N = {n_}")
    ax[1].axhline(0.99, color="k", ls="--", lw=0.8)
    ax[1].axvline(np.degrees(chi_star()), color=RAY, ls=":", lw=1)
    ax[1].set(xlabel="χ (deg)", ylabel="R", ylim=(0.85, 1.002), title="R(χ): TIR beyond the first gap above "
              f"{np.degrees(np.arcsin(1/m.neff)):.0f}°")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trench_dbr.png")
    plt.close(fig)


def fig_stadium():
    r = a = 5e-3
    nf = 16
    base = P.stadium_cell(r, a, cap_facets=nf, straight_segments=4)
    cells = [("smooth stadium", P.stadium_cell(r, a)),
             (f"{nf} facets per cap", base),
             (f"facets + tilt, ×{EXAG:.0f} of 1 mrad rms", P.perturb(base, tilt_rms=1e-3 * EXAG, seed=2)),
             (f"curved caps, ×{EXAG:.0f} of 40 /m rms", P.perturb(base, curvature_rms=40 * EXAG, seed=2, select="cap"))]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (ttl, c) in zip(axs, cells):
        x, y = c.outline(48)
        ax.plot(x * 1e3, y * 1e3, color=WALL, lw=1.5)
        xs, ys, *_ = T.trace_path(c, 0.1 * c.perimeter, 0.45, 40)
        ax.plot(xs * 1e3, ys * 1e3, color=RAY, lw=0.6, alpha=0.8)
        ax.set(title=ttl, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    fig.suptitle("general-mpc planar cells (stadium family); red: 40 bounces of one ray")
    fig.tight_layout()
    fig.savefig(OUT / "gmpc_stadiums.png")
    plt.close(fig)


def fig_herriott():
    R, N, M, A = 0.5, 30, 7, 0.012
    c = H.herriott_cell(R, N, M, A)
    tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 200)
    Rx, Ry, d = H.astigmatic_reentrant(R, 50, 11, 13)
    ca = H.astigmatic_cell(Rx, Ry, d, A, 0.7 * A, 1.2e-3)
    ta = H.trace3d(ca["mirrors"], ca["p0"], ca["d0"], 200)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[2, 1, 1]))
    a = ax[0]
    d0 = c["d"]
    yy = np.linspace(-0.0254, 0.0254, 100)
    for z0, sgn in ((0, 1), (d0, -1)):
        a.plot((z0 + sgn * (R - np.sqrt(R**2 - yy**2))) * 100, yy * 1e3, color=WALL, lw=3)
    n = tr.n_hits[0]
    pts = np.vstack([tr.start[0], tr.hits[0, :n]])
    a.plot(pts[:, 2] * 100, pts[:, 0] * 1e3, color=RAY, lw=0.6)
    hx, hy, hr = c["mirrors"][0].holes[0]
    a.plot([0, 0], [(hx - hr) * 1e3, (hx + hr) * 1e3], color="white", lw=4)
    a.annotate("input/output hole", (0, hx * 1e3), (d0 * 15, hx * 1e3 + 9), color=PORT, fontsize=8,
               arrowprops=dict(arrowstyle="->", color=PORT))
    a.set(xlabel="z (cm)", ylabel="x (mm)", title=f"Herriott side view: R = {R} m, d = {d0*100:.2f} cm, N = {N}, M = {M}, "
          f"path {tr.path():.2f} m")
    for ax_, t_, ttl in ((ax[1], tr, f"Herriott spots (circle A = {A*1e3:.0f} mm)"),
                         (ax[2], ta, f"astigmatic: Rx = {Rx*100:.1f}, Ry = {Ry*100:.1f} cm\nLissajous, re-entrant after 50")):
        k = t_.n_hits[0]
        for mi, col, lab in ((0, WALL, "M1"), (1, RAY, "M2")):
            s = np.nonzero(t_.mirror[0, :k] == mi)[0]
            ax_.plot(t_.local[0, s, 0] * 1e3 * (1 if mi == 0 else -1), t_.local[0, s, 1] * 1e3, "o", ms=3.5, color=col, label=lab)
        ax_.add_patch(plt.Circle((0, 0), 25.4, fill=False, color="0.6", lw=1))
        ax_.set(title=ttl, xlabel="x (mm)", ylabel="y (mm)", aspect="equal", xlim=(-27, 27), ylim=(-27, 27))
        ax_.legend(fontsize=7, loc="lower right")
    fig.suptitle("general-mpc Herriott-type cells (2-inch mirrors)")
    fig.tight_layout()
    fig.savefig(OUT / "herriott_cells.png")
    plt.close(fig)


FIGS = {"cmpc": fig_cmpc, "dbr": fig_dbr, "stadium": fig_stadium, "herriott": fig_herriott}

if __name__ == "__main__":
    for k in sys.argv[1:] or FIGS:
        FIGS[k]()
        print("wrote", k)
