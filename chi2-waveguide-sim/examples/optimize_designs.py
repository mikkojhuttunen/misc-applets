"""Milestone 6 example: optimised masks for flat bands and for the ±5 nm wafer thickness spread.

    python examples/optimize_designs.py [--length 4e-3] [--devices 200] [--fig figs/optimize_designs.png]

A  20 and 40 nm flat-top SHG bands on the TFLN ridge: Hann chirp -> envelope IFTA -> wall-level gradient (JAX)
B  robust SHG at 1550 nm over ±5 nm film (TFLN and TFLT): ordered, 100 nm chirp, DE on the geometric mean over
   devices, then gradient polish; checked on random devices (mean, 10 % quantile, worst)
C  robust THG 1550 -> 517 nm: two-section and blend vs DE of a blended chirped superlattice (geometric mean)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chi2sim import fast, optimize as O, poling, synthesis as S, thg as T, wafer, waveguide as W  # noqa: E402

LAM0 = 1.55e-6


def stats(x):
    return f"mean {np.mean(x) * 100:7.2f}, p10 {np.quantile(x, 0.1) * 100:7.2f}, worst {np.min(x) * 100:7.2f}"


def part_a(L):
    m = W.SHGModel(W.Ridge(), np.linspace(1.45e-6, 1.65e-6, 8))
    kap2 = float(m.coupling(LAM0)["kappa"]) ** 2
    print(f"\nA  flat-top SHG bands, TFLN ridge, L = {L * 1e3:g} mm (ordered FWHM about 6 nm; core = central 80 % of the band)")
    curves = {}
    for fw in (20e-9, 40e-9):
        b = S.band_to_dk(m, LAM0, fw)
        klo, khi = sorted((abs(b.k_lo), abs(b.k_hi)))
        ceil = S.efficiency_ceiling(L, khi - klo)["mean_G2_max"]
        chk = np.abs(m(LAM0 + np.linspace(-0.4, 0.4, 201) * fw))
        out = {"Hann chirp": S.design(m, LAM0, fw, L, mode="chirp").poling}
        out["IFTA"] = O.ifta(abs(b.k0), klo, khi, L, chk, iters=300, level=0.8, edge=0.3)["poling"]
        tg = O.band_target(m, LAM0 + np.linspace(-0.4, 0.4, 81) * fw, kind="flat", alpha=3.0)
        out["IFTA + gradient"] = O.gradient_optimise(out["IFTA"], tg)["poling"]
        print(f"  {fw * 1e9:.0f} nm band (L·B_k = {L * (khi - klo):.0f}, ceiling {kap2 * ceil * 100:.1f} %/W):")
        lam = LAM0 + np.linspace(-1.0, 1.0, 801) * fw
        for k, p in out.items():
            f = O.core_flatness(p, m, LAM0, fw)
            if fw == 40e-9:
                curves[k] = kap2 * np.abs(fast.spectrum(p, np.abs(m(lam)))[0]) ** 2
            print(f"    {k:>16}: core mean {kap2 * f['mean'] * 100:6.1f} %/W = {f['mean'] / ceil:.2f} of ceiling, "
                  f"ripple {f['ripple_db']:.2f} dB, min domain {p.lengths[1:-1].min() * 1e6:.2f} µm")
    return lam, curves


def part_b(L, n_dev):
    res = {}
    print(f"\nB  robust SHG at 1550 nm, film ±5 nm uniform, L = {L * 1e3:g} mm; η at 1550 nm over {n_dev} random devices (%/W)")
    for mat in ("ln", "lt"):
        fam = wafer.ThicknessFamily(lambda g: W.SHGModel(g, np.linspace(1.40e-6, 1.70e-6, 7)), W.Ridge(material=mat))
        nom = fam.nominal
        k2, k0 = float(nom.coupling(LAM0)["kappa"]) ** 2, abs(float(nom(LAM0)))
        tg = O.robust_target(fam, LAM0, k2, kind="geo", n=21)
        designs = {"ordered": poling.ordered(L, float(nom.period(LAM0))),
                   "chirp 100 nm (M4)": S.design(nom, LAM0, 100e-9, L, mode="chirp").poling}
        de = O.evolve_chirp(tg, L, k0, maxiter=20, popsize=10, bounds=[(-0.02, 0.02), (0.0, 0.1), (0.0, 0.45)])
        designs[f"DE geo (span {de['x'][1] * 100:.2f} %, edge {de['x'][2]:.2f})"] = de["poling"]
        designs["DE + gradient"] = O.gradient_optimise(de["poling"], tg)["poling"]
        print(f"  {'TFLN' if mat == 'ln' else 'TFLT'}: nominal ordered peak {k2 * (2 / np.pi * L) ** 2 * 100:.0f} %/W")
        for k, p in designs.items():
            r = wafer.shg_devices(fam, p, np.linspace(1.545e-6, 1.555e-6, 21), LAM0, n_devices=n_dev, seed=7)
            res[(mat, k)] = r["at_target"]
            print(f"    {k:>36}: {stats(r['at_target'])}")
    return res


def part_c(L, n_dev):
    lams = np.linspace(1.50e-6, 1.60e-6, 5)
    fam = wafer.ThicknessFamily(lambda g: W.THGModel(g, lams), W.Ridge())
    nom = fam.nominal
    ka, kb = abs(float(nom.dk_a(LAM0))), abs(float(nom.dk_b(LAM0)))
    tg = O.thg_robust_target(fam, LAM0, n=21, kind="geo")
    designs = {"two-section": T.two_section(nom, LAM0, L), "blend 0.9->0.1": T.blend(nom, LAM0, L, 0.9, 0.1, 2.36)}
    de = O.evolve_thg(tg, L, ka, kb, maxiter=12, popsize=8,
                      bounds=((0.4, 1.0), (0.0, 0.6), (0.0, np.pi), (0.0, 0.04), (0.0, 0.04)))
    x = de["x"]
    designs[f"DE geo (w {x[0]:.2f}->{x[1]:.2f}, spans {x[3] * 100:.2f}/{x[4] * 100:.2f} %)"] = de["poling"]
    print(f"\nC  robust THG 1550 -> 517 nm, TFLN ridge, film ±5 nm, L = {L * 1e3:g} mm; η_THG at 1550 nm over {n_dev} devices (%/W²)")
    res = {}
    for k, p in designs.items():
        r = T.thg_devices(fam, p, np.array([1.5499e-6, LAM0, 1.5501e-6]), LAM0, n_devices=n_dev, seed=9)
        res[k] = r["at_target"]
        print(f"    {k:>40}: nominal {r['nominal_at_target'] * 100:8.1f}; devices {stats(r['at_target'])}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=4e-3)
    ap.add_argument("--devices", type=int, default=200)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args(argv)
    lam, spec = part_a(a.length)
    shg = part_b(a.length, a.devices)
    thg = part_c(a.length, min(a.devices, 100))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
        for k, y in spec.items():
            ax[0].plot((lam - LAM0) * 1e9, y * 100, label=k)
        ax[0].set_xlabel("pump detuning (nm)")
        ax[0].set_ylabel("η (%/W)")
        ax[0].set_title("A: 40 nm band, TFLN")
        ax[0].legend(fontsize=7)
        bins = np.logspace(-3, 3, 50)
        for (mat, k), v in shg.items():
            if mat == "ln":
                ax[1].hist(np.maximum(v * 100, 1e-3), bins=bins, histtype="step", label=k)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("η at 1550 nm (%/W)")
        ax[1].set_ylabel("devices")
        ax[1].set_title("B: TFLN SHG over ±5 nm film")
        ax[1].legend(fontsize=7)
        bins = np.logspace(-6, 4, 50)
        for k, v in thg.items():
            ax[2].hist(np.maximum(v * 100, 1e-6), bins=bins, histtype="step", label=k)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("η_THG at 1550 nm (%/W²)")
        ax[2].set_title("C: TFLN THG over ±5 nm film")
        ax[2].legend(fontsize=7)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=130)
        print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
