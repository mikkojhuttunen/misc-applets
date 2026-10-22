"""How many DBR periods give a 99 % mirror on a free-standing Si membrane?

Uses the misc-applets reference engines (no new physics):
    engines.slab_waveguide   n_eff of the air | Si | air slab mode (full and partially etched membrane)
    engines.bragg_grating    TrenchDBR (etched air-trench DBR, exact oblique-incidence transfer matrix),
                             stack_R_oblique (any layer stack), coupling_coefficient (coupled-mode κ)
    engines.cell_mirror      reflectance the light actually sees in a segmented cell (traced angles of incidence)

Mirror families:
    trench     cavity (membrane) | [air gap, Si tooth] x N | air      deep (through-membrane) etch, as in cmpc-sim
    shallow    cavity | [etched to t - e, full t] x N | full t        partial etch of depth e (dbr-structures style)

For each design, the smallest N with R >= target at:
    0 deg              normal incidence
    star {24/7}        chi = 90 - 180*7/24 = 37.5 deg, the launch used in the cmpc-sim figures
    all angles         min over |sin chi| in [0, 1] (what a fully chaotic cell can hit)
    uniform avg        <R> over uniform sin chi (invariant measure of a fully chaotic billiard)
    curved cell        power-weighted R_eff in the 24-facet cell with rho = 5 cm facets (cell_mirror engine)

    python dbr_periods.py                     Si 250 nm TM0 + Si 220 nm TE0 at 1.55 um, target 0.99
    python dbr_periods.py --target 0.999
    python dbr_periods.py --plot              also writes figs/dbr_periods.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "math-engines"))
from engines.materials import engine as mat  # noqa: E402
from engines.slab_waveguide.engine import neff_three_layer  # noqa: E402
from engines.bragg_grating.engine import TrenchDBR, stack_R_oblique, coupling_coefficient, LATERAL_POL  # noqa: E402
from engines.cell_mirror.engine import cell_mirror_reflectance  # noqa: E402

SG = np.linspace(0.0, 0.999, 2001)
S_STAR = np.sin(np.radians(90 - 180 * 7 / 24))
N_MAX = 60


def neff(t, lam, pol):
    return neff_three_layer(lam, 1.0, mat.index("si", lam), 1.0, t, pol)


def trench(n_t, lam, pol, N, m_gap=1, m_tooth=1, sin_design=0.0):
    d = TrenchDBR(n_tooth=n_t, lam_design=lam, N=N, m_gap=m_gap, m_tooth=m_tooth, bounce_loss=0.0, slab_pol=pol,
                  sin_design=sin_design)
    return lambda s: d.R(lam, s)


def shallow(n_full, n_etch, lam, pol, N):
    """Quarter-wave partial-etch grating, normal-incidence design, ending in the full membrane."""
    lay = [(n_etch, lam / (4 * n_etch)), (n_full, lam / (4 * n_full))] * N
    return lambda s: stack_R_oblique(lam, s, n_full, lay, n_full, LATERAL_POL[pol])


def metrics(Rf):
    R = Rf(SG)
    return dict(R0=float(Rf(np.array(0.0))), Rstar=float(Rf(np.array(S_STAR))), Rmin=float(R.min()),
                Ravg=float(np.trapezoid(R, SG) / (SG[-1] - SG[0])))


def first_N(make, key, target):
    for N in range(1, N_MAX + 1):
        if metrics(make(N))[key] >= target:
            return N
    return None


def first_N_cell(n_t, lam, pol, target, m_tooth=1, max_N=30):
    for N in range(1, max_N + 1):
        r = cell_mirror_reflectance(radius=5e-3, n_facets=24, theta_c=np.radians(90 - 180 * 7 / 24), fan=np.radians(1),
                                    n_beams=7, n_hits=150, curvature=1 / 0.05, n_eff=n_t, polarization=pol, periods=N,
                                    m_gap=1, m_tooth=m_tooth, wavelength=lam).values
        if r["R_eff"] >= target:
            return N
    return None


def fmt(N):
    return f"{N:>4d}" if N else f">{N_MAX:<3d}" if N is None else "   -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=1.55e-6)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-cell", action="store_true", help="skip the traced-cell column (slowest)")
    a = ap.parse_args()
    lam, tg = a.lam, a.target
    print(f"Si membrane, lambda = {lam*1e6:.3f} um, n_Si = {mat.index('si', lam):.4f}, target R >= {tg}")
    print(f"star {{24/7}} launch: chi = {np.degrees(np.arcsin(S_STAR)):.1f} deg\n")

    rows = []
    for t, pol in ((250e-9, "TM"), (220e-9, "TE"), (300e-9, "TM")):
        n_t = neff(t, lam, pol)
        tir = 1 / n_t
        print(f"Si {t*1e9:.0f} nm {pol}0 ({LATERAL_POL[pol]}-pol on the trench walls): n_eff = {n_t:.4f}, "
              f"TIR at the first gap for chi > {np.degrees(np.arcsin(tir)):.1f} deg")
        for (mg, mt) in ((1, 1), (1, 3)):
            make = lambda N, mg=mg, mt=mt: trench(n_t, lam, pol, N, mg, mt)
            d = TrenchDBR(n_tooth=n_t, lam_design=lam, N=1, m_gap=mg, m_tooth=mt)
            Ns = {k: first_N(make, k, tg) for k in ("R0", "Rstar", "Ravg", "Rmin")}
            Nc = None if a.no_cell else first_N_cell(n_t, lam, pol, tg, mt)
            rows.append((f"trench {t*1e9:.0f}{pol} ({mg},{mt})", Ns, Nc))
            print(f"  trench gap {d.d_gap*1e9:.0f} nm / tooth {d.d_tooth*1e9:.0f} nm (orders {mg},{mt}): "
                  f"N(0deg)={fmt(Ns['R0'])} N(37.5deg)={fmt(Ns['Rstar'])} N(<R>uniform)={fmt(Ns['Ravg'])} "
                  f"N(all angles)={fmt(Ns['Rmin'])}" + ("" if a.no_cell else f" N(curved cell R_eff)={fmt(Nc)}"))
        if pol == "TE":
            print(f"    TE0 is p-polarised on the trench walls: Brewster hole near "
                  f"{np.degrees(np.arcsin(1/np.hypot(n_t, 1))):.1f} deg, so 'all angles' never reaches the target")
        for e in (30e-9, 70e-9, 120e-9):
            n_e = neff(t - e, lam, pol)
            dn = n_t - n_e
            kap = coupling_coefficient(dn, lam).values["kappa"]
            L_need = np.arctanh(np.sqrt(tg)) / kap
            Lam = lam / (2 * (n_t + n_e) / 2)
            make = lambda N, n_e=n_e: shallow(n_t, n_e, lam, pol, N)
            N0 = first_N(make, "R0", tg)
            print(f"  shallow etch {e*1e9:.0f} nm: dn = {dn:.4f}, CMT kappa = {kap*1e-3:.1f}/mm -> L = {L_need*1e6:.1f} um "
                  f"= {L_need/Lam:.0f} periods; TMM N(0deg) = {fmt(N0)}")
        print()

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parent / "figs"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        n_t = neff(250e-9, lam, "TM")
        Ns = np.arange(1, 9)
        for key, lab in (("R0", "0°"), ("Rstar", "37.5° (star 24/7)"), ("Ravg", "uniform sin χ avg"), ("Rmin", "min over χ")):
            ax[0].semilogy(Ns, [1 - metrics(trench(n_t, lam, "TM", N))[key] for N in Ns], "o-", label=lab)
        ax[0].axhline(1 - tg, color="k", ls="--", lw=0.8)
        ax[0].set(xlabel="periods N", ylabel="1 − R", title="Si 250 nm TM0 trench DBR, λ/4 gap and tooth")
        ax[0].legend(fontsize=8)
        for N in (2, 3, 4, 6):
            ax[1].plot(np.degrees(np.arcsin(SG)), trench(n_t, lam, "TM", N)(SG), label=f"N = {N}")
        ax[1].axhline(tg, color="k", ls="--", lw=0.8)
        ax[1].set(xlabel="angle of incidence χ (deg)", ylabel="R", ylim=(0.8, 1.001), title="R(χ)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "dbr_periods.png", dpi=130)
        print("wrote", out / "dbr_periods.png")


if __name__ == "__main__":
    main()
