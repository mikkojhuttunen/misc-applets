"""Milestone 3 example: x-cut TFLN and TFLT ridges for 1550 -> 775 nm SHG from the 2D FD mode solver.

    python examples/ridge_shg.py [--fig figs/ridge_shg.png] [--cache cache/]

Prints period, indices, A_eff, Miller-scaled d33, normalised efficiency, ordered-grating bandwidth and the
width tolerance; then repeats the 50 nm band designs of band_designs.py on the modal dispersion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chi2sim import synthesis, waveguide as W  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--length", type=float, default=4e-3)
    a = ap.parse_args(argv)
    lam0, L = 1.55e-6, a.length
    lams = np.linspace(1.40e-6, 1.70e-6, 9)
    geoms = {"TFLN x-cut": W.Ridge(material="ln"), "TFLT x-cut": W.Ridge(material="lt")}
    models = {k: W.SHGModel.cached(g, lams, cache_dir=a.cache) for k, g in geoms.items()}
    g0 = geoms["TFLN x-cut"]
    print(f"ridge: film {g0.film * 1e9:.0f} nm, etch {g0.etch * 1e9:.0f} nm, top width {g0.width * 1e6:.2f} µm, "
          f"sidewall {g0.sidewall:.0f}°, {g0.clad} clad on {g0.substrate}; TE00 at 1550 and 775 nm; L = {L * 1e3:g} mm")
    for name, m in models.items():
        s = m.summary(lam0, L)
        ws = W.width_scan(m.geom, [m.geom.width - 50e-9, m.geom.width, m.geom.width + 50e-9], lam0)
        tol = 2 * 1.391557 / (L * abs(ws["ddk_dw"][1]))
        print(f"\n{name}: period {s['period'] * 1e6:.3f} µm, n_p {s['n_pump']:.4f}, n_SH {s['n_sh']:.4f}, "
              f"n_g {s['ng_pump']:.3f}/{s['ng_sh']:.3f}")
        print(f"  A_eff {s['a_eff'] * 1e12:.3f} µm², d33(1550) {s['d'] * 1e12:.1f} pm/V, "
              f"η0 {s['eta_norm'] * 1e-2:.0f} %/(W cm²) unpoled-matched, first-order QPM {s['eta_qpm'] * 1e-2:.0f} %/(W cm²)")
        print(f"  ordered FWHM {s['fwhm'] * 1e9:.2f} nm; dΛ/dw {np.gradient(ws['period'], ws['width'])[1]:.4f}; "
              f"width tolerance (uniform error within FWHM) ±{tol * 1e9:.1f} nm")
        kappa = float(m.coupling(lam0)["kappa"])
        for fw in (50e-9,):
            for mode in ("chirp", "random"):
                d = synthesis.design(m, lam0, fw, L, mode=mode, n_real=32)
                r = d.report
                print(f"  {fw * 1e9:.0f} nm {mode:6}: <η> {kappa**2 * r['band_stats']['mean'] * 100:6.2f} %/W, "
                      f"{r['fraction_of_ceiling']:.2f} of ceiling, ripple80 {r['core_stats']['ripple_db']:.2f} dB"
                      + (f", speckle {r['speckle_contrast']:.2f}" if 'speckle_contrast' in r else ""))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = models["TFLN x-cut"]
        gr = W.build_grid(m.geom, lam0)
        mp, ms = W.fundamental(m.geom, lam0, "TE", gr), W.fundamental(m.geom, lam0 / 2, "TE", gr)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        for k, (md, t) in enumerate(((mp, "TE00, 1550 nm"), (ms, "TE00, 775 nm"))):
            ax[k].pcolormesh(gr.xe * 1e6, gr.ye * 1e6, md.field, cmap="magma", shading="flat")
            ax[k].contour(gr.xc * 1e6, gr.yc * 1e6, gr.f_film, [0.5], colors="w", linewidths=0.7)
            ax[k].set_xlim(-2, 2)
            ax[k].set_ylim(-0.8, 1.0)
            ax[k].set_aspect("equal")
            ax[k].set_title(f"{t}: n_eff {md.neff:.4f}")
            ax[k].set_xlabel("x (µm)")
        ax[0].set_ylabel("y (µm)")
        ll = np.linspace(lams[0], lams[-1], 200)
        for name, mm in models.items():
            ax[2].plot(ll * 1e9, mm.period(ll) * 1e6, label=name)
        ax[2].set_xlabel("pump λ (nm)")
        ax[2].set_ylabel("first-order period (µm)")
        ax[2].legend()
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=130)
        print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
