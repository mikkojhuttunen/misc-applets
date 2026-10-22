"""Integrated (planar) Herriott cell in a free-standing Si membrane: geometry, exact 2D ray trace and a plotter.

Two concave in-plane mirrors (curved etched-trench DBRs, radius R) face each other at spacing d = R (1 - cos 2πM/N).
The beam enters through a gap in the DBR of mirror 1 and follows the paraxial spot pattern x_n = A cos(nθ + φ)
(cos θ = 1 - d/R; the general-mpc herriott.injection relations in one transverse dimension). After N hits (M turns)
it is back at the gap and leaves. The launch phase φ matters in a planar cell: with φ = 0 (port at the pattern edge,
the classic 3D choice) the pattern is symmetric in n, the exit ray retraces the input path and couples back into the
input waveguide, and every spot is hit twice. With φ = 90° the port sits at the mirror centre, the spots x_n =
-A sin nθ are all distinct, and the output leaves at an angle ≈ 2A sin θ / d to the input: input and output
waveguides form a V behind mirror 1.

Coordinates in the membrane plane: X along the cell axis (mirror 1 vertex at X = 0, mirror 2 at X = d), Y across.
The trace is exact (circular mirrors, no paraxial approximation); each hit's angle of incidence gives the trench-DBR
reflectance from cmpc_sim.DBR (Si slab mode n_eff, quarter-wave gap and tooth), so the throughput is the product of
R over the hits.

    python herriott_planar.py                                  R = 5 mm, N = 22, M = 5, A = 0.5 mm, φ = 90°, 4 DBR periods
    python herriott_planar.py --phase 0                        classic edge port: the output retraces the input
    python herriott_planar.py --R 8 --N 30 --M 7 --A 1.5 --periods 6
    python herriott_planar.py --bounce-loss 3e-4               add the cmpc-sim per-bounce loss
Writes figs/herriott_planar.png and .pdf.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "cmpc-sim" / "cmpc"))
import cmpc_sim as cs  # noqa: E402

OUT = HERE / "figs"


@dataclass
class PlanarHerriott:
    """Lengths in m. D: half-aperture of each mirror (default 1.4 A). wp: port gap width (default: 3 mode radii,
    at most 0.4 x the nearest other spot on mirror 1)."""
    R: float = 5e-3
    N: int = 22
    M: int = 5
    A: float = 0.5e-3
    phase: float = 90.0                                         # launch phase φ, degrees
    D: float | None = None
    wp: float | None = None
    lam: float = 1.55e-6
    thickness: float = 250e-9
    pol: str = "TM"
    periods: int = 4
    bounce_loss: float = 0.0
    mode: object = field(init=False, default=None)

    def __post_init__(self):
        if not 0 < self.M < self.N / 2:
            raise ValueError("need 0 < M < N/2 for a stable re-entrant cell")
        self.theta = 2 * np.pi * self.M / self.N
        self.d = self.R * (1 - np.cos(self.theta))
        self.D = 1.4 * self.A if self.D is None else self.D
        ph = np.radians(self.phase)
        self.y0 = self.A * np.cos(ph)
        # paraxial: x_n = x0 cos nθ + sqrt(d / (2R - d)) (x0 + R x0') sin nθ = A cos(nθ + φ)
        self.slope0 = (-self.A * np.sin(ph) / np.sqrt(self.d / (2 * self.R - self.d)) - self.y0) / self.R
        self.mode = cs.membrane_mode("Si", self.thickness, self.lam, self.pol, with_group_index=False)
        g = 1 - self.d / self.R
        lam_m = self.lam / self.mode.neff                     # wavelength in the slab mode
        self.w = float(np.sqrt(lam_m * self.d / np.pi / np.sqrt(1 - g * g)))
        if self.wp is None:
            n = np.arange(2, self.N, 2)
            sp = np.abs(self.paraxial(n) - self.y0)
            sp = sp[sp > 1e-9 * self.A]
            self.wp = float(min(2 * 3 * self.w, 0.8 * sp.min())) if sp.size else 6 * self.w
        self.dbr = cs.DBR(n_tooth=self.mode.neff, lam_design=self.lam, N=self.periods, m_gap=1, m_tooth=1,
                          slab_pol=self.pol, bounce_loss=self.bounce_loss)

    def paraxial(self, n):
        return self.A * np.cos(np.asarray(n, float) * self.theta + np.radians(self.phase))

    # mirror k: 0 at X = 0 (centre of curvature at +R), 1 at X = d (centre at d - R)
    def centre(self, k):
        return np.array([self.R, 0.0]) if k == 0 else np.array([self.d - self.R, 0.0])

    def surface_x(self, k, y):
        s = self.R - np.sqrt(self.R**2 - np.asarray(y, float) ** 2)
        return s if k == 0 else self.d - s

    def trace(self, n_max=None):
        """Exact trace from the port. Returns dict(points (n+1, 2), mirror, angle_deg, exit: 'port' | 'edge' | 'trapped',
        n_hits, path, exit_dir, in_dir)."""
        n_max = n_max or 4 * self.N
        y0 = self.y0
        p = np.array([self.surface_x(0, y0), y0])
        u = np.array([1.0, self.slope0])
        u /= np.linalg.norm(u)
        pts, mir, ang = [p.copy()], [], []
        exit_kind, k_next = "trapped", 1
        in_dir = u.copy()
        for _ in range(n_max):
            k = k_next
            C = self.centre(k)
            o = p - C
            b = o @ u
            disc = b * b - (o @ o - self.R**2)
            if disc < 0:
                exit_kind = "edge"
                break
            sq = np.sqrt(disc)
            cand = [t for t in (-b - sq, -b + sq) if t > 1e-12 * self.R]
            ok = [t for t in cand if ((p + t * u)[0] - C[0]) * (1 if k == 1 else -1) > 0]
            if not ok:
                exit_kind = "edge"
                break
            t = min(ok)
            p = p + t * u
            pts.append(p.copy())
            if abs(p[1]) > self.D:
                exit_kind = "edge"
                break
            if k == 0 and abs(p[1] - self.y0) <= self.wp / 2:
                exit_kind = "port"
                break
            nrm = (C - p) / self.R
            ang.append(float(np.degrees(np.arccos(min(1.0, abs(u @ nrm))))))
            mir.append(k)
            u = u - 2 * (u @ nrm) * nrm
            k_next = 1 - k
        pts = np.array(pts)
        return dict(points=pts, mirror=np.array(mir), angle_deg=np.array(ang), exit=exit_kind,
                    n_hits=len(mir), path=float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), exit_dir=u, in_dir=in_dir)

    def budget(self, tr):
        """Mirror reflectance at each hit (trench DBR at that angle of incidence) and the cumulative throughput."""
        s = np.sin(np.radians(tr["angle_deg"]))
        R = np.atleast_1d(self.dbr.R(self.lam, s))
        I = np.cumprod(np.r_[1.0, R])
        seg = np.linalg.norm(np.diff(tr["points"], axis=0), axis=1)
        L_eff = float(np.sum(I[: len(seg)] * seg))                      # absorption-weighted path, Σ I_j ℓ_j
        return dict(R=R, T_out=float(I[-1]), L_eff=L_eff)


def plot(c: PlanarHerriott, out=OUT, band_mm=0.12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    tr = c.trace()
    bud = c.budget(tr)
    mm = 1e3
    fig = plt.figure(figsize=(14, 7.6), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    y = np.linspace(-c.D, c.D, 400)
    for k in (0, 1):
        sgn = -1 if k == 0 else 1                       # DBR band lies behind each mirror
        for j in range(c.periods):
            g0 = j / c.periods * band_mm / mm
            g1 = g0 + 0.67 / c.periods * band_mm / mm
            xs0 = c.surface_x(k, y) + sgn * g0
            xs1 = c.surface_x(k, y) + sgn * g1
            gap = np.where((k == 0) & (np.abs(y - c.y0) <= c.wp / 2), np.nan, 1.0)
            ax.fill_betweenx(y * mm, xs0 * mm * gap, xs1 * mm * gap, color="#8fa3c4", lw=0)
        ax.plot(c.surface_x(k, y) * mm, y * mm, color="#1f4e79", lw=1.8)
    P = tr["points"] * mm
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(P) - 1))
    for i in range(len(P) - 1):
        ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color=cols[i], lw=1.1)
    sp = P[1:1 + tr["n_hits"]]
    mk = tr["mirror"]
    ax.plot(sp[mk == 0, 0], sp[mk == 0, 1], "o", ms=4, color="#1f4e79", label="spots on mirror 1")
    ax.plot(sp[mk == 1, 0], sp[mk == 1, 1], "o", ms=4, color="#c0504d", label="spots on mirror 2")
    # input and output waveguides behind mirror 1 (tapered strips)
    p0 = P[0] / mm
    for u, col, lab in ((-tr["in_dir"], "#1f8a4c", "input"), (tr["exit_dir"], "#7a3fb0", "output")):
        if lab == "output" and tr["exit"] != "port":
            continue
        st = p0 + u * band_mm / mm * 1.2
        v = np.array([-u[1], u[0]])
        w1, w2, lt, L2 = c.wp, c.wp * 0.3, 0.6e-3, 1.4e-3
        poly = np.array([st - w1 / 2 * v, st + lt * u - w2 / 2 * v, st + L2 * u - w2 / 2 * v, st + L2 * u + w2 / 2 * v,
                         st + lt * u + w2 / 2 * v, st + w1 / 2 * v]) * mm
        ax.fill(poly[:, 0], poly[:, 1], color="#d4a63c", alpha=0.9, lw=0)
        e = (st + L2 * u) * mm
        ax.annotate(lab, e, e + u * 0.15, color=col, fontsize=9, ha="center", va="center")
        a0, a1 = (p0 + (L2 * 0.9) * u) * mm, (p0 + 0.2e-3 * u) * mm
        if lab == "input":
            ax.annotate("", a1, a0, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        else:
            ax.annotate("", a0, a1, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    ax.set(aspect="equal", xlabel="X along the cell (mm)", ylabel="Y across (mm)",
           title=f"Integrated Herriott cell, top view: R = {c.R * mm:g} mm, N = {c.N}, M = {c.M}, φ = {c.phase:g}°, d = {c.d * mm:.3f} mm"
                 f"\n{tr['exit']} after {tr['n_hits']} hits, path {tr['path'] * 100:.2f} cm (DBR band drawn {band_mm} mm wide)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    a2 = fig.add_subplot(gs[0, 1])
    n = np.arange(1, tr["n_hits"] + 1)
    yp = c.paraxial(n)
    hy = tr["points"][1:tr["n_hits"] + 1, 1]
    a2.plot(n, yp * mm, "k+", ms=7, label="paraxial A cos(nθ + φ)")
    a2.plot(n[tr["mirror"] == 0], hy[tr["mirror"] == 0] * mm, "o", ms=4, color="#1f4e79", label="mirror 1 (exact)")
    a2.plot(n[tr["mirror"] == 1], hy[tr["mirror"] == 1] * mm, "o", ms=4, color="#c0504d", label="mirror 2 (exact)")
    a2.axhspan((c.y0 - c.wp / 2) * mm, (c.y0 + c.wp / 2) * mm, color="#1f8a4c", alpha=0.15, label="port gap")
    a2.set(xlabel="hit n", ylabel="Y (mm)", title="Spot positions")
    a2.legend(fontsize=7, loc="lower left")
    a2.grid(alpha=0.25)

    a3 = fig.add_subplot(gs[1, 1])
    a3.plot(n, bud["R"], "o-", ms=3, color="#1f4e79", label="R at each hit")
    a3b = a3.twinx()
    a3b.plot(np.r_[0, n], np.cumprod(np.r_[1.0, bud["R"]]), color="#c0504d", label="power left")
    a3.set(xlabel="hit n", ylabel="DBR reflectance", title=f"Trench DBR, {c.periods} periods, Si {c.thickness * 1e9:.0f} nm {c.pol}₀: "
           f"angles {tr['angle_deg'].min():.1f}–{tr['angle_deg'].max():.1f}°")
    a3b.set_ylabel("power left", color="#c0504d")
    a3.grid(alpha=0.25)
    fig.text(0.62, 0.005, f"mode radius w = {c.w * 1e6:.1f} µm (in-plane, λ/n_eff), port gap {c.wp * 1e6:.0f} µm, aperture ±{c.D * mm:.2f} mm; "
             f"out: {bud['T_out'] * 100:.1f} % after {tr['n_hits']} hits, absorption path Σ I ℓ = {bud['L_eff'] * 100:.2f} cm",
             fontsize=8.5, ha="center", color="#4a565c")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"herriott_planar.{ext}", bbox_inches="tight")
    plt.close(fig)
    return tr, bud


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=float, default=5.0, help="mirror radius, mm")
    ap.add_argument("--N", type=int, default=22)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--A", type=float, default=0.5, help="pattern half-width, mm")
    ap.add_argument("--phase", type=float, default=90.0, help="launch phase φ, deg (0: port at the edge, output retraces the input)")
    ap.add_argument("--D", type=float, default=None, help="mirror half-aperture, mm (default 1.4 A)")
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--thickness", type=float, default=250, help="membrane, nm")
    ap.add_argument("--lam", type=float, default=1.55, help="µm")
    ap.add_argument("--pol", default="TM", choices=("TE", "TM"))
    ap.add_argument("--bounce-loss", type=float, default=0.0)
    a = ap.parse_args()
    c = PlanarHerriott(R=a.R * 1e-3, N=a.N, M=a.M, A=a.A * 1e-3, phase=a.phase, D=None if a.D is None else a.D * 1e-3, lam=a.lam * 1e-6,
                       thickness=a.thickness * 1e-9, pol=a.pol, periods=a.periods, bounce_loss=a.bounce_loss)
    tr, bud = plot(c)
    print(f"d = {c.d * 1e3:.4f} mm, theta = {np.degrees(c.theta):.2f} deg, mode radius {c.w * 1e6:.1f} um, port gap {c.wp * 1e6:.0f} um")
    ang = np.degrees(np.arccos(np.clip(-tr["in_dir"] @ tr["exit_dir"], -1, 1)))
    print(f"input/output waveguide angle {ang:.1f} deg")
    print(f"exit: {tr['exit']} after {tr['n_hits']} hits, path {tr['path'] * 1e3:.2f} mm, angles {tr['angle_deg'].min():.2f}-{tr['angle_deg'].max():.2f} deg")
    print(f"DBR R per hit {bud['R'].min():.5f}-{bud['R'].max():.5f}; output {bud['T_out'] * 100:.2f} %; sum I l = {bud['L_eff'] * 1e3:.2f} mm")
    print("wrote", OUT / "herriott_planar.png")


if __name__ == "__main__":
    main()
