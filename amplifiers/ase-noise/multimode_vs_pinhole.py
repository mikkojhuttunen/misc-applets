"""Simulation 2: multimode ASE and spatial filtering.

An amplifier gives the same gain G to every LG mode up to order N_max (M = n_pol (N+1)(N+2)/2
modes), the signal is in LG_00. Detection on a large-area photodiode with no filter, behind a
pinhole spatial filter (radius in units of the focal-plane LG_00 radius w_f), or through a
single-mode fibre. Signal-spontaneous beat noise only sees ASE in the signal mode; the
spontaneous-spontaneous term grows with the number of detected modes, which the pinhole cuts
at the price of signal loss 1 - exp(-2 a²/w_f²).

Usage: python multimode_vs_pinhole.py [output_dir]
"""
import sys

import numpy as np

from _style import INK2, SERIES, out_path, plt
from engines.ase_noise import engine as an

BASE = dict(signal_power=1e-6, gain=1e3, n_sp=1.5, wavelength=1.064e-6,
            optical_bandwidth=100e9, electrical_bandwidth=1e9, n_pol=2)
ORDERS = np.arange(0, 21)
N_FIXED = 10
RADII = np.linspace(0.3, 3.0, 55)


def snr_db(r):
    return 10 * np.log10(r["snr"])


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    cases = [("No filter", "none", 1.0), ("Pinhole a = 1.5 w_f", "pinhole", 1.5),
             ("Pinhole a = 1.0 w_f", "pinhole", 1.0), ("Pinhole a = 0.7 w_f", "pinhole", 0.7),
             ("Single-mode fibre", "single_mode", 1.0)]
    for c, (lab, f, a) in zip(SERIES, cases):
        rs = [an.multimode_snr(max_order=int(n), spatial_filter=f, pinhole_radius=a, **BASE) for n in ORDERS]
        m = [r["amplifier_modes"] for r in rs]
        ax1.plot(m, [snr_db(r) for r in rs], color=c, marker="o", ms=3, label=lab)
    ax1.set(xscale="log", xlabel="Amplifier modes M (incl. 2 polarisations)", ylabel="Electrical SNR (dB)",
            title="SNR vs number of amplified modes")
    ax1.legend(loc="lower left")

    rp = [an.multimode_snr(max_order=N_FIXED, spatial_filter="pinhole", pinhole_radius=float(a), **BASE) for a in RADII]
    bare = an.multimode_snr(max_order=N_FIXED, spatial_filter="none", pinhole_radius=1.0, **BASE)
    smf = an.multimode_snr(max_order=N_FIXED, spatial_filter="single_mode", pinhole_radius=1.0, **BASE)
    s = np.array([snr_db(r) for r in rp])
    ax2.plot(RADII, s, color=SERIES[2], label="Pinhole")
    ax2.axhline(snr_db(bare), color=SERIES[0], lw=1.5, ls="--", label="No filter")
    ax2.axhline(snr_db(smf), color=SERIES[4], lw=1.5, ls="--", label="Single-mode fibre")
    k = int(np.argmax(s))
    ax2.plot(RADII[k], s[k], "o", color=SERIES[2], ms=8, mec="#fcfcfb", mew=2)
    ax2.annotate(f"optimum a = {RADII[k]:.2f} w_f\n{s[k]:.1f} dB", (RADII[k], s[k]), xytext=(10, -28),
                 textcoords="offset points", color=INK2)
    ax2.set(xlabel="Pinhole radius a / w_f", ylabel="Electrical SNR (dB)",
            title=f"Pinhole size trade-off, M = {bare['amplifier_modes']} modes")
    ax2.legend(loc="lower right")

    for c, key, lab in [(SERIES[0], "var_shot", "Shot"), (SERIES[1], "var_s_sp", "Signal–spontaneous"),
                        (SERIES[2], "var_sp_sp", "Spontaneous–spontaneous")]:
        ax3.plot(RADII, [r[key] for r in rp], color=c, label=lab)
    ax3.set(yscale="log", xlabel="Pinhole radius a / w_f", ylabel="Noise current variance (A²)",
            title="Noise budget behind the pinhole")
    ax3.legend(loc="lower right")
    meff = [r["effective_modes"] for r in rp]
    i1 = int(np.argmin(abs(RADII - 1.0)))
    ax3.annotate(f"M_eff ≈ {meff[i1]:.1f} at a = w_f\n(M = {bare['amplifier_modes']} unfiltered)",
                 (RADII[i1], rp[i1]["var_sp_sp"]), xytext=(40, -60), textcoords="offset points", color=INK2,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    p = out_path("multimode_vs_pinhole.png", sys.argv)
    fig.savefig(p, dpi=150)
    print(p)


if __name__ == "__main__":
    main()
