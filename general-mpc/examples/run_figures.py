"""Overview figures for general-mpc: Herriott, astigmatic and deformed Herriott cells; smooth, faceted and perturbed
stadium cells.

    python examples/run_figures.py            all figures into general-mpc/figs
    python examples/run_figures.py 2          selected
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmpc import herriott as H, planar as P, trace2d as T  # noqa: E402

OUT = ROOT / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})

CFG = dict(R=0.5, N=30, M=7, A=0.012, lam=1.55e-6,       # Herriott: 0.5 m mirrors, 30 hits, 7 turns, 12 mm pattern radius
           r=5e-3, a=5e-3, cap_facets=16, port_w=150e-6)  # planar: 1 cm wide stadium with 5 mm straights


def spots(ax, tr, k, mirror, label, **kw):
    sel = np.nonzero(tr.mirror[0, : tr.n_hits[0]] == mirror)[0]
    ax.plot(tr.hits[0, sel, 0] * 1e3, tr.hits[0, sel, 1] * 1e3, "o", ms=4, label=label, **kw)


def fig1():
    """Herriott cells: ideal, astigmatic (Lissajous), and slightly deformed."""
    R, N, M, A = CFG["R"], CFG["N"], CFG["M"], CFG["A"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    c = H.herriott_cell(R, N, M, A, wavelength=CFG["lam"])
    tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 200)
    n = np.arange(1, tr.n_hits[0] + 1)
    xp, yp = H.paraxial_spots(n, c["d"], A, A, R, R)
    spots(ax[0], tr, 0, 0, "M1 (exact)")
    spots(ax[0], tr, 0, 1, "M2 (exact)")
    ax[0].plot(xp * 1e3, yp * 1e3, "k+", ms=7, label="paraxial")
    hx, hy, hr = c["mirrors"][0].holes[0]
    ax[0].add_patch(plt.Circle((hx * 1e3, hy * 1e3), hr * 1e3, fill=False, color="C3", lw=1.5))
    r = H.reentrance(tr)
    ax[0].set(title=f"Herriott N={N}, M={M}: d = {c['d']*100:.2f} cm, path {r['path']:.2f} m\nre-entry offset {r['exit_offset']*1e6:.0f} µm (spherical aberration)",
              xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax[0].legend(fontsize=7, loc="lower right")

    Rx, Ry, d = H.astigmatic_reentrant(R, 50, 11, 13)
    ca = H.astigmatic_cell(Rx, Ry, d, A, 0.7 * A, 1.2e-3)
    ta = H.trace3d(ca["mirrors"], ca["p0"], ca["d0"], 200)
    spots(ax[1], ta, 0, 0, "M1")
    spots(ax[1], ta, 0, 1, "M2")
    ra = H.reentrance(ta)
    ax[1].set(title=f"Astigmatic (Rx = {Rx*100:.2f}, Ry = {Ry*100:.2f} cm): Lissajous, N = 50\n{ra['exit']} after {ra['n_hits']} hits, path {ra['path']:.2f} m",
              xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax[1].legend(fontsize=7)

    m2 = H.perturb_mirror(c["mirrors"][1], dRx=+0.002 * R, dRy=-0.002 * R)     # 0.4 % astigmatism on one mirror
    td = H.trace3d([c["mirrors"][0], m2], c["p0"], c["d0"], 400)
    spots(ax[2], td, 0, 0, "M1")
    spots(ax[2], td, 0, 1, "M2")
    ax[2].add_patch(plt.Circle((hx * 1e3, hy * 1e3), hr * 1e3, fill=False, color="C3", lw=1.5))
    rd = H.reentrance(td)
    ax[2].set(title=f"Slightly deformed: M2 astigmatic ±0.2 % in R\n{rd['exit']} after {rd['n_hits']} hits, path {rd['path']:.2f} m",
              xlabel="x (mm)", ylabel="y (mm)", aspect="equal")

    errs = np.geomspace(1e-5, 1e-2, 25)
    off_r, off_a = [], []
    for e in errs:
        for kind, store in (("radius", off_r), ("astig", off_a)):
            dR = (e * R, e * R) if kind == "radius" else (e * R, -e * R)
            ms = [H.perturb_mirror(m, dRx=dR[0], dRy=dR[1]) for m in c["mirrors"]]
            t = H.trace3d(ms, c["p0"], c["d0"], 400)
            rr = H.reentrance(t)
            store.append(rr["exit_offset"] if rr["exit"] == "hole" and rr["n_hits"] == N else np.nan)
    ax[3].loglog(errs, np.array(off_r) * 1e3, "o-", label="radius error (both mirrors)")
    ax[3].loglog(errs, np.array(off_a) * 1e3, "s-", label="astigmatism ±ΔR (both mirrors)")
    ax[3].axhline(hr * 1e3, color="C3", lw=1, ls="--")
    ax[3].text(1.2e-5, hr * 1e3 * 0.75, "hole radius", color="C3", fontsize=8)
    ax[3].set_ylim(top=hr * 1e3 * 2)
    ax[3].set(title="Re-entry offset after N hits (missing points:\nthe beam no longer leaves after N hits)", xlabel="relative error ΔR/R", ylabel="re-entry offset (mm)")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_herriott_ideal_astigmatic_deformed.png")


def fig2():
    """Planar cells: smooth, faceted and perturbed stadiums; trajectories and phase space."""
    r, a, nf = CFG["r"], CFG["a"], CFG["cap_facets"]
    cells = [("smooth stadium", P.stadium_cell(r, a)),
             (f"{nf} facets per cap", P.stadium_cell(r, a, cap_facets=nf, straight_segments=4)),
             ("facets + 1 mrad rms tilt", P.perturb(P.stadium_cell(r, a, cap_facets=nf, straight_segments=4), tilt_rms=1e-3, seed=2)),
             ("facets + curvature 1/ρ = 40 /m rms", P.perturb(P.stadium_cell(r, a, cap_facets=nf, straight_segments=4), curvature_rms=40, seed=2, select="cap"))]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for k, (name, cell) in enumerate(cells):
        x, y = cell.outline()
        ax[0, k].plot(x * 1e3, y * 1e3, "k", lw=1.2)
        for th, col in ((0.3, "C0"), (0.62, "C1")):
            xs, ys, *_ = T.trace_path(cell, 0.13 * cell.perimeter, th, 60)
            ax[0, k].plot(xs * 1e3, ys * 1e3, lw=0.6, color=col)
        sep, fit = T.twin_divergence(cell, 0.13 * cell.perimeter, 0.45, 300)
        lam = "λ ≈ 0 (no exponential growth)" if fit is None or fit["lam"] < 0.05 else f"λ = {fit['lam']:.2f} per reflection"
        ax[0, k].set(title=f"{name}\n{lam}", aspect="equal", xlabel="x (mm)", ylabel="y (mm)")
        ax[0, k].grid(False)
        S, SC = T.poincare(cell, 30, 400, seed=k)
        ax[1, k].plot(S.ravel() / cell.perimeter, SC.ravel(), ".", ms=0.6, color="k", alpha=0.5)
        ax[1, k].set(xlabel="s / perimeter", ylabel="sin χ", ylim=(-1, 1), title="phase space at the walls")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stadium_smooth_faceted_perturbed.png")


def fig3():
    """Path statistics of the planar cells with ports, vs mirror reflectance; mean-field comparison."""
    r, a, nf, w = CFG["r"], CFG["a"], CFG["cap_facets"], CFG["port_w"]
    cells = [("circle (regular)", P.circle_cell(r)), ("smooth stadium", P.stadium_cell(r, a)),
             (f"faceted stadium ({nf}/cap)", P.stadium_cell(r, a, cap_facets=nf, straight_segments=4)),
             ("faceted + curved facets", P.perturb(P.stadium_cell(r, a, cap_facets=nf, straight_segments=4), curvature_rms=40, seed=2, select="cap"))]
    Rs = np.array([0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, cell in cells:
        tab = T.trace_rays(cell, w, 2000, 3000, theta0=0.35, seed=4)
        L, Tdet = [], []
        for R in Rs:
            res = T.evaluate(tab, lambda s, R=R: np.full_like(s, R))
            L.append(res.L_mean)
            Tdet.append(res.T_det)
        ax[0].semilogx(1 - Rs, np.array(L) * 100, "o-", label=name)
        ax[1].semilogx(1 - Rs, Tdet, "o-", label=name)
    mf = [T.mean_field_estimate(cells[1][1], w, R)["L_mean"] * 100 for R in Rs]
    ax[0].semilogx(1 - Rs, mf, "k--", label="mean field (ergodic), stadium")
    ax[0].set(xlabel="mirror loss 1 − R", ylabel="mean detected path ⟨L⟩ (cm)", title=f"Path to the output port ({w*1e6:.0f} µm ports)")
    ax[1].set(xlabel="mirror loss 1 − R", ylabel="transmission to the output port", title="Detected fraction")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_planar_path_statistics.png")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1", "2", "3"]:
        print("figure", w, flush=True)
        globals()[f"fig{w}"]()
    print("done ->", OUT)
