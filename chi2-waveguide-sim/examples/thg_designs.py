"""Milestone 5 example: cascaded THG 1550 -> 517 nm in an x-cut TFLN ridge.

    python examples/thg_designs.py [--length 4e-3] [--devices 100] [--fig figs/thg_designs.png]

Compares double-matching masks (two-section, optimised superlattice, optimised blend, commensurate single
period) at λ0, their spectra, the TE10 3ω modal option, depletion with the full cascade solver, and the
±5 nm film-thickness spread across devices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chi2sim import thg, wafer, waveguide as W  # noqa: E402
from engines.cwe_cascade import engine as casc  # noqa: E402

LAM0 = 1.55e-6
LAMS = np.linspace(1.50e-6, 1.60e-6, 5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=4e-3)
    ap.add_argument("--devices", type=int, default=100)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args(argv)
    L = a.length
    m = W.THGModel(W.Ridge(), LAMS)
    pa, pb = m.periods(LAM0)
    c = m.coupling(LAM0)
    ref = thg.reference_efficiencies(m, LAM0, L)
    print(f"TFLN x-cut ridge 600/300 nm, 1.4 µm; TE00 at 1550, 775, 517 nm; L = {L * 1e3:g} mm")
    print(f"  Λ_a (SHG) {pa * 1e6:.3f} µm, Λ_b (SFG) {pb * 1e6:.3f} µm, Δk_b/Δk_a = {pa / pb:.4f}")
    print(f"  A_eff SHG {float(m.n('aa', LAM0)) * 1e12:.3f} µm², SFG {float(m.n('ab', LAM0)) * 1e12:.3f} µm²; "
          f"d {m.d_coeffs(LAM0)[0] * 1e12:.1f} / {m.d_coeffs(LAM0)[1] * 1e12:.1f} pm/V; κ_a {float(c['kappa_a']):.0f}, κ_3 {float(c['kappa_3']):.0f} 1/(m√W)")
    print(f"  references: both matched at full strength {ref['matched_uniform'] * 100:.0f} %/W², "
          f"two-section closed form {ref['two_section_closed'] * 100:.0f} %/W²")

    designs = {"two-section": thg.two_section(m, LAM0, L)}
    p, w, ph, _ = thg.optimise_superlattice(m, LAM0, L)
    designs[f"superlattice w={w:.2f} ψ={ph:.2f}"] = p
    p, prm, _ = thg.optimise_blend(m, LAM0, L)
    designs[f"blend {prm['w_start']:.1f}->{prm['w_end']:.1f} ψ={prm['phase']:.2f}"] = p
    pc, row = thg.commensurate(m, LAM0, L, max_order=9)
    designs[f"single period m=({row['m_a']},{row['m_b']})"] = pc
    lam = np.linspace(1.53e-6, 1.57e-6, 801)
    spectra = {}
    print(f"\n  {'design':>34} {'η_THG(1550)':>12} {'FWHM':>8} {'min domain':>11}")
    from chi2sim.metrics import fwhm
    for name, p in designs.items():
        eta = thg.thg_spectrum(m, p, lam)[0]
        spectra[name] = eta
        i0 = int(np.argmin(np.abs(lam - LAM0)))
        print(f"  {name:>34} {eta[i0] * 100:10.1f} %/W² {fwhm(lam, eta) * 1e9:6.2f} nm {p.lengths[1:-1].min() * 1e6:9.2f} µm")
    print(f"  (single period: residual Δk_b {row['residual']:.3g} 1/m, strength {row['strength']:.3g})")

    g10 = W.Ridge(etch=0.6e-6, width=1.2e-6)
    m10 = W.THGModel(g10, LAMS, order3=(1, 0))
    m00 = W.THGModel(g10, LAMS)
    print(f"\n  fully etched 1.2 µm ridge, 3ω in TE10 instead of TE00: Λ_b {m10.periods(LAM0)[1] * 1e6:.2f} vs "
          f"{m00.periods(LAM0)[1] * 1e6:.2f} µm, SFG overlap area {float(m10.n('ab', LAM0)) * 1e12:.0f} vs "
          f"{float(m00.n('ab', LAM0)) * 1e12:.2f} µm² (SFG efficiency × {float(m00.n('ab', LAM0)) / float(m10.n('ab', LAM0)):.1e})")

    best_name = max(designs, key=lambda k: spectra[k][int(np.argmin(np.abs(lam - LAM0)))])
    bp = designs[best_name]
    print(f"\n  depletion, {best_name} (full cascade RK4 at 1550 nm):")
    for P in (0.01, 0.1, 0.3, 1.0):
        r = casc.cascade_coupled(bp.boundaries, bp.g, abs(float(m.dk_a(LAM0))), abs(float(m.dk_b(LAM0))),
                                 float(c["kappa_a"]), float(c["kappa_1"]), float(c["kappa_2"]), float(c["kappa_3"]), P)
        und = float(thg.thg_spectrum(m, bp, [LAM0], P)[0][0]) * P**3
        print(f"    P = {P:5.2f} W: P3 {float(r['P3'][0]) * 1e3:8.3f} mW (undepleted {und * 1e3:9.3f} mW), "
              f"P2 {float(r['P2'][0]) * 1e3:8.2f} mW, residual {float(r['power_residual'][0]):.1e}")

    fam = wafer.ThicknessFamily(lambda g: W.THGModel(g, LAMS), W.Ridge())
    s = thg.double_matching_shift(fam, LAM0)
    print(f"\n  film thickness: re-matching shift dλ/dt {s['dlam_a_dt']:+.2f} nm/nm (SHG step), {s['dlam_b_dt']:+.2f} nm/nm (SFG step)")
    lam_d = np.linspace(1.50e-6, 1.60e-6, 801)
    dev = {}
    for name in ("two-section", best_name):
        r = thg.thg_devices(fam, designs[name], lam_d, LAM0, n_devices=a.devices, seed=4)
        dev[name] = r
        print(f"  {name:>34}: nominal peak {r['nominal'].max() * 100:.0f} %/W², device peak median "
              f"{np.median(r['peak_eta']) * 100:.1f} %/W², best {r['peak_eta'].max() * 100:.0f}, "
              f"yield (peak ≥ ½ nominal, any λ in 1500-1600 nm) {np.mean(r['peak_eta'] >= 0.5 * r['nominal'].max()):.2f}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
        for name, eta in spectra.items():
            ax[0].plot(lam * 1e9, eta * 100, label=name)
        ax[0].set_xlabel("fundamental λ (nm)")
        ax[0].set_ylabel("η_THG (%/W²)")
        ax[0].set_title(f"THG designs, L = {L * 1e3:g} mm")
        ax[0].legend(fontsize=7)
        r = dev[best_name]
        for e in r["eta"][:30]:
            ax[1].plot(r["lam"] * 1e9, e * 100, lw=0.5, alpha=0.6)
        ax[1].plot(r["lam"] * 1e9, r["nominal"] * 100, "k", lw=1.5, label="nominal")
        ax[1].set_xlabel("fundamental λ (nm)")
        ax[1].set_ylabel("η_THG (%/W²)")
        ax[1].set_title(f"{best_name}: 30 devices, film ±5 nm")
        ax[1].legend(fontsize=8)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=130)
        print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
