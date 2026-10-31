"""Examples A, B, E and hybrid of the plan: 1550 -> 775 nm SHG band designs against the efficiency ceiling.

    python examples/band_designs.py [--length 5e-3] [--backend jax] [--fig figs/band_designs.png]

Dispersion: LinearIndexSHG (TFLN-ridge-like phase and group indices) until the waveguide layer exists.
Efficiencies use κ from math-engines cwe_shg.coupling_shg (d33 = 25 pm/V, A_eff = 1 µm²; placeholders).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chi2sim import dispersion, fast, poling, synthesis  # noqa: E402
from engines.cwe_shg import engine as cwe  # noqa: E402
from engines.qpm_shg import engine as qpm  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=5e-3)
    ap.add_argument("--backend", default="numpy", choices=("numpy", "jax"))
    ap.add_argument("--n-real", type=int, default=64)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args(argv)
    dkf = dispersion.LinearIndexSHG()
    lam0, L = 1.55e-6, a.length
    kappa = float(cwe.coupling_shg(lam0, dkf.n_pump, dkf.n_sh, 25e-12, 1e-12)["kappa"])

    # A: ordered grating reproduces qpm_shg (period, bandwidth) and the (2/π)² η0 L² peak
    ref = qpm.phase_matching(lam0, dkf.n_pump, dkf.n_sh, dkf.ng_pump, dkf.ng_sh, L)
    p = poling.ordered(L, ref["period"])
    band = synthesis.band_to_dk(dkf, lam0, 3 * ref["fwhm"])
    lam = synthesis.lambda_grid(band, L, span=2.0)
    G2 = np.abs(fast.spectrum(p, dkf(lam))[0]) ** 2
    from chi2sim.metrics import fwhm
    print(f"A ordered: period {ref['period'] * 1e6:.4f} µm, FWHM {fwhm(lam, G2) * 1e9:.3f} nm (qpm_shg {ref['fwhm'] * 1e9:.3f} nm), "
          f"peak η {kappa**2 * G2.max() * 100:.0f} %/W")

    rows, curves = [], []
    for fw in (10e-9, 50e-9, 100e-9):
        for mode, kw in (("chirp", {}), ("random", {"n_real": a.n_real}), ("hybrid", {"sigma_wall": 100e-9, "n_real": a.n_real})):
            d = synthesis.design(dkf, lam0, fw, L, mode=mode, backend=a.backend, **kw)
            r = d.report
            rows.append((fw, mode, r["fwhm"], r["fraction_of_ceiling"], r["core_stats"]["ripple_db"],
                         r.get("speckle_contrast", float("nan")), kappa**2 * r["band_stats"]["mean"] * 100, r.get("min_domain", np.nan)))
            curves.append((fw, mode, d.lam, kappa**2 * d.G2 * 100, d.report.get("mc")))
    print(f"\nL = {L * 1e3:g} mm; ceiling = (2/π)² 2πL/B_k; η in %/W (d = 25 pm/V, A_eff = 1 µm²)")
    print(f"{'FWHM':>6} {'mode':>7} {'got':>7} {'ceil.':>6} {'ripple80':>8} {'speckle':>8} {'<η>':>8} {'min dom':>8}")
    for fw, mode, got, frac, rip, sc, eta, md in rows:
        print(f"{fw * 1e9:5.0f}n {mode:>7} {got * 1e9:6.1f}n {frac:6.3f} {rip:7.2f}dB {sc:8.2f} {eta:8.2f} {md * 1e6:7.2f}µ")
    print("\nchirp: one deterministic mask; random: ensemble mean (each realization is speckle, contrast ≈ 1);"
          "\nhybrid: chirp with 100 nm rms wall jitter, ensemble mean and its contrast")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for ax, fw in zip(axs, (10e-9, 50e-9, 100e-9)):
            for f, mode, x, y, mc in curves:
                if f != fw:
                    continue
                ax.plot((x - lam0) * 1e9, y, label=mode + (" (mean)" if mode != "chirp" else ""))
                if mode == "random" and mc is not None:
                    ax.fill_between((x - lam0) * 1e9, kappa**2 * mc["q10"] * 100, kappa**2 * mc["q90"] * 100, alpha=0.2, label="random 10-90 %")
            ceil = synthesis.efficiency_ceiling(L, synthesis.band_to_dk(dkf, lam0, fw).b_k)["mean_G2_max"] * kappa**2 * 100
            ax.axhline(ceil, ls="--", lw=0.8, color="k", label="ceiling")
            ax.set_title(f"FWHM {fw * 1e9:.0f} nm, L = {L * 1e3:g} mm")
            ax.set_xlabel("pump detuning (nm)")
            ax.set_ylabel("η (%/W)")
        axs[0].legend(fontsize=8)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=130)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
