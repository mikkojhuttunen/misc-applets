"""Milestone 4 example: device-to-device film (and deposited-layer) thickness spread of ±5 nm across a wafer.

    python examples/wafer_spread.py [--devices 200] [--length 4e-3] [--backend jax] [--fig figs/wafer_spread.png]

Each device has a constant thickness offset (uniform in ±5 nm). One poling mask per design is applied to all
devices. Reports peak-shift sensitivity, the spread of the phase-matching peak, and the yield (fraction of
devices reaching half the nominal efficiency at the target wavelength) for ordered and broadband designs;
then the same for a SiNx-loaded TFLN guide (film and strip thicknesses both spread) and for OPA with
775, 532 and 1064 nm pumps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chi2sim import poling, synthesis, wafer, waveguide as W  # noqa: E402

LAM0 = 1.55e-6
LAMS = np.linspace(1.40e-6, 1.70e-6, 7)


def shg_study(name, geom, L, n_dev, backend, fwhms=(30e-9, 60e-9, 100e-9)):
    fam = wafer.ThicknessFamily(lambda g: W.SHGModel(g, LAMS), geom)
    nom = fam.nominal
    lam = np.linspace(1.42e-6, 1.68e-6, 2601)
    designs = [("ordered", poling.ordered(L, float(nom.period(LAM0))))]
    for fw in fwhms:
        designs.append((f"chirp {fw * 1e9:.0f} nm", synthesis.design(nom, LAM0, fw, L, mode="chirp").poling))
    rows, keep = [], {}
    for label, p in designs:
        r = wafer.shg_devices(fam, p, lam, LAM0, n_devices=n_dev, seed=1, backend=backend)
        y = wafer.yield_fraction(r["at_target"], r["nominal_at_target"])
        rows.append((label, r["nominal_at_target"] * 100, np.median(r["at_target"]) * 100,
                     np.min(r["at_target"]) * 100, y))
        keep[label] = r
    sens = fam.sensitivity(LAM0, L)
    extra = (f", dλ/dt_load {sens['dpeak_dload']:+.2f} nm/nm (ordered-grating tolerance ±{sens['tolerance_load'] * 1e9:.2f} nm)"
             if "dpeak_dload" in sens else "")
    print(f"\n{name}: SHG 1550 -> 775 nm, L = {L * 1e3:g} mm, {n_dev} devices, ±5 nm uniform")
    print(f"  phase matching: dλ/dt_film {sens['dpeak_dfilm']:+.2f} nm/nm (ordered-grating tolerance ±{sens['tolerance_film'] * 1e9:.2f} nm)"
          f"{extra}; rms peak shift over the devices {keep['ordered']['peak_shift_rms'] * 1e9:.1f} nm")
    print(f"  {'design':>14} {'η nominal':>10} {'median':>8} {'worst':>8} {'yield≥50%':>10}   (η at 1550 nm in %/W)")
    for label, n0, med, worst, y in rows:
        print(f"  {label:>14} {n0:10.1f} {med:8.1f} {worst:8.2f} {y:10.2f}")
    return keep


def opa_study(name, geom, lp, L, n_dev, pump_power, signals):
    fam = wafer.ThicknessFamily(lambda g: W.OPAModel(g, lp, np.linspace(signals[0] - 20e-9, signals[-1] + 20e-9, 7)), geom)
    nom = fam.nominal
    p = poling.ordered(L, float(nom.period(LAM0)))
    r = wafer.opa_devices(fam, p, signals, pump_power, n_devices=n_dev, seed=2)
    i0 = int(np.argmin(np.abs(signals - LAM0)))
    at = r["gain_db"][:, i0]
    nom_db = r["nominal_db"][i0]
    print(f"\n{name}: OPA pump {lp * 1e9:.0f} nm, signal 1550 nm -> idler {float(nom.idler(LAM0)) * 1e9:.0f} nm, "
          f"period {float(nom.period(LAM0)) * 1e6:.3f} µm, P_p {pump_power:g} W, L {L * 1e3:g} mm")
    sens = fam.sensitivity(LAM0, L)
    shift = "∞ (degenerate: ∂Δk/∂λ_s = 0)" if not np.isfinite(sens["dpeak_dfilm"]) else f"{sens['dpeak_dfilm']:+.2f} nm/nm"
    print(f"  nominal gain at 1550 {nom_db:.2f} dB; devices: median {np.median(at):.2f} dB, worst {at.min():.2f} dB, "
          f"yield (≥ half the nominal dB gain) {wafer.yield_fraction(at, nom_db):.2f}")
    print(f"  dλ_s/dt_film {shift}; film tolerance of the grating ±{sens['tolerance_film'] * 1e9:.2f} nm; "
          f"device peak gain in 1450-1650 nm: median {np.median(r['peak_db']):.2f} dB")
    for n in nom.notes:
        print("  " + n)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=200)
    ap.add_argument("--length", type=float, default=4e-3)
    ap.add_argument("--backend", default="numpy", choices=("numpy", "jax"))
    ap.add_argument("--fig", default=None)
    a = ap.parse_args(argv)
    L, n = a.length, a.devices
    ln = shg_study("TFLN x-cut ridge (600 nm film, 300 nm etch, 1.4 µm)", W.Ridge(material="ln"), L, n, a.backend)
    shg_study("TFLT x-cut ridge (600 nm film, 300 nm etch, 1.4 µm)", W.Ridge(material="lt"), L, n, a.backend, fwhms=(60e-9,))
    shg_study("SiNx-loaded TFLN (300 nm LN unetched, 300 nm SiNx strip, 1.2 µm)",
              W.Ridge(film=0.3e-6, etch=0.0, width=1.2e-6, load="sinx", load_thickness=0.3e-6), L, n, a.backend, fwhms=(60e-9,))
    sig = np.linspace(1.45e-6, 1.65e-6, 201)
    o775 = opa_study("TFLN ridge", W.Ridge(), 775e-9, L, min(n, 100), 0.3, sig)
    opa_study("TFLN ridge", W.Ridge(), 532e-9, L, min(n, 100), 0.3, sig)
    opa_study("TFLN thick ridge (900 nm film, 450 nm etch, 2.4 µm)", W.Ridge(film=0.9e-6, etch=0.45e-6, width=2.4e-6),
              1064e-9, L, min(n, 100), 0.3, sig)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
        for k, label in enumerate(("ordered", "chirp 60 nm")):
            r = ln[label]
            for e in r["eta"][:40]:
                ax[k].plot(r["lam"] * 1e9, e * 100, lw=0.5, alpha=0.5)
            ax[k].plot(r["lam"] * 1e9, r["nominal"] * 100, "k", lw=1.5, label="nominal")
            ax[k].axvline(1550, color="k", ls=":", lw=0.8)
            ax[k].set_title(f"TFLN SHG, {label}, 40 of {n} devices")
            ax[k].set_xlabel("pump λ (nm)")
            ax[k].set_ylabel("η (%/W)")
            ax[k].legend(fontsize=8)
        for gdb in o775["gain_db"][:40]:
            ax[2].plot(o775["signal"] * 1e9, gdb, lw=0.5, alpha=0.5)
        ax[2].plot(o775["signal"] * 1e9, o775["nominal_db"], "k", lw=1.5)
        ax[2].set_title("OPA, 775 nm pump, 0.3 W, ordered grating")
        ax[2].set_xlabel("signal λ (nm)")
        ax[2].set_ylabel("signal gain (dB)")
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=130)
        print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
