"""Simulation 4: saturated, cladding-pumped highly multimode amplifier with spatial hole burning.

Yb-like two-level medium, 100 µm / NA 0.2 step-index core (881 LP modes per polarisation),
400 µm inner cladding, 200 W pump at 976 nm co-propagating, 3 m, signal at 1064 nm in LP01,
forward ASE resolved in every mode (one 3 THz bin, both polarisations). The LP01 signal burns
a hole in the central inversion, so higher-order modes see more gain than the signal and the
ASE builds up in them; confining the doping (ρ < 1) restores the signal's advantage.

Usage: python saturated_multimode.py [output_dir]
"""
import sys

import numpy as np

from _style import INK2, SERIES, out_path, plt
from engines.multimode_amplifier import engine as mm
from multimode_mdg import u_over_v

LAM, A, NA, LENGTH, PUMP = 1.064e-6, 50e-6, 0.2, 3.0, 200.0
CLAD = np.pi * (200e-6) ** 2
YB = dict(dopant_density=6e25, sigma_es=2.5e-25, sigma_as=5e-27, sigma_ep=2.5e-24, sigma_ap=2.6e-24,
          lifetime=1e-3, pump_wavelength=0.976e-6)
STEPS = 120


def run(modes, rho, seed_power):
    return mm.saturated_amplifier(modes, LENGTH, PUMP, seed_power, mm.disk_profile(modes, rho * A),
                                  cladding_area=CLAD, steps=STEPS, **YB)


def main():
    modes = mm.step_index_modes(A, NA, LAM)
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)

    ax = axs[0, 0]
    for c, rho in zip(SERIES, [1.0, 0.6]):
        r = run(modes, rho, 0.1)
        ax.plot(r["z"], r["signal_power"], color=c, label=f"Signal, ρ = {rho}")
        ax.plot(r["z"], np.maximum(r["ase_power"], 1e-9), color=c, ls="--", label=f"ASE, ρ = {rho}")
    ax.plot(r["z"], r["pump_power"], color=INK2, lw=1, ls=":", label="Pump, ρ = 0.6")
    ax.set(yscale="log", ylim=(1e-6, 400), xlabel="Position z (m)", ylabel="Power (W)",
           title=f"Power evolution, 100 mW seed, {PUMP:.0f} W pump")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[0, 1]
    x = u_over_v(modes)
    fam = modes.family_of_mode
    first = np.unique(fam, return_index=True)[1]
    for c, ps in zip(SERIES, [0.01, 1.0, 10.0]):
        r = run(modes, 1.0, ps)
        g = 10 * np.log10(mm.modal_noise(r["channel"])["gain"][first])
        ax.plot(x, g, "o", ms=3.5, mec="none", color=c, alpha=0.85, label=f"Seed {ps:g} W")
        ax.plot(x[0], g[0], "o", ms=9, color=c, mec="#fcfcfb", mew=2)
    ax.plot([], [], "o", ms=9, color=INK2, mec="#fcfcfb", mew=2, label="LP01 (signal mode)")
    ax.set(xlabel="Normalised transverse wavenumber U/V", ylabel="Modal gain at operating point (dB)",
           title="Spatial hole burning, full-core doping (ρ = 1)", ylim=(8, 54))
    ax.legend(loc="upper center", ncol=2, markerscale=1.5)

    seeds = np.logspace(-2, 1, 7)
    res = {}
    for rho in [1.0, 0.8, 0.6]:
        res[rho] = [run(modes, rho, ps) for ps in seeds]
    ax = axs[1, 0]
    for c, rho in zip(SERIES, res):
        ex = []
        for r in res[rho]:
            G = mm.modal_noise(r["channel"])["gain"]
            ex.append(10 * np.log10(G.max() / G[0]))
        ax.plot(seeds, ex, "o-", color=c, label=f"ρ = {rho}")
    ax.axhline(0, color=INK2, lw=1)
    ax.set(xscale="log", xlabel="Seed power in LP01 (W)", ylabel="Best higher-order gain − LP01 gain (dB)",
           title="Gain advantage of competing modes")
    ax.legend(loc="center right")

    ax = axs[1, 1]
    for c, rho in zip(SERIES, res):
        ax.plot(seeds, [r["ase_power"][-1] / (r["ase_power"][-1] + r["signal_power"][-1]) for r in res[rho]],
                "o-", color=c, label=f"ρ = {rho}")
    ax.set(xscale="log", yscale="log", xlabel="Seed power in LP01 (W)", ylabel="ASE / total output power",
           title="Output ASE fraction")
    ax.legend(loc="upper right")

    p = out_path("saturated_multimode.png", sys.argv)
    fig.savefig(p, dpi=150)
    print(p)


if __name__ == "__main__":
    main()
