"""Simulation 3: mode-dependent gain (MDG) and ASE in highly multimode fibre amplifiers.

Small-signal amplifiers (medium gain g0 = 3 /m, n_sp = 1.2, L = 2 m) in a 100 µm / NA 0.2
core at 1064 nm: 881 LP modes (step index) or 435 LG modes (graded index) per polarisation.
Doping is confined to r ≤ ρ a. Panels: modal gain vs normalised transverse wavenumber U/V
for both profiles, the distribution of the output ASE over the modes, and the growth of
the MDG spread with length without and with random linear mode coupling (Ho & Kahn:
∝ L uncoupled, ∝ sqrt(L) strongly coupled).

Usage: python multimode_mdg.py [output_dir]
"""
import sys

import numpy as np

from _style import INK2, SERIES, out_path, plt
from engines.multimode_amplifier import engine as mm

LAM, A, NA, G0, NSP, L = 1.064e-6, 50e-6, 0.2, 3.0, 1.2, 2.0
RHOS = [1.0, 0.7, 0.4]


def u_over_v(modes, na=NA, n_clad=1.45):
    k = 2 * np.pi / LAM
    nco2 = n_clad**2 + na**2
    return np.sqrt(np.maximum(k * k * nco2 - modes.beta**2, 0)) / (k * na)


def modal_gain_db(modes, rho):
    ge, ga, al = mm.small_signal_rates(modes, mm.disk_profile(modes, rho * A), G0, NSP)
    return 10 * np.log10(np.exp((ge - ga) * L)), mm.linear_channel(modes, ge, ga, al, L)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    profiles = [("Step index", mm.step_index_modes(A, NA, LAM)), ("Graded index", mm.grin_modes(A, NA, LAM))]
    for ax, (name, modes) in zip(axs[0], profiles):
        x = u_over_v(modes)
        for c, rho in zip(SERIES, RHOS):
            gdb, _ = modal_gain_db(modes, rho)
            ax.plot(x, gdb, "o", ms=3.5, color=c, mec="none", alpha=0.85, label=f"ρ = {rho}")
        ax.set(xlabel="Normalised transverse wavenumber U/V (0 = fundamental)", ylabel="Modal gain (dB)",
               title=f"{name}: {modes.n_modes} modes per polarisation", ylim=(-1, 33))
        ax.legend(title="Doped radius / core radius", loc="upper center", ncol=3, markerscale=2)

    ax = axs[1, 0]
    for (name, modes), ls in zip(profiles, ["-", "--"]):
        for c, rho in zip(SERIES, RHOS):
            _, ch = modal_gain_db(modes, rho)
            occ = np.sort(mm.modal_noise(ch)["ase_occupation"])[::-1]
            meff = occ.sum() ** 2 / np.sum(occ**2)
            frac = np.arange(1, occ.size + 1) / occ.size
            ax.plot(frac, np.cumsum(occ) / occ.sum(), ls=ls, color=c,
                    label=f"{name}, ρ = {rho}: M_eff = {meff:.0f} / {modes.n_modes}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    ax.set(xlabel="Fraction of modes (sorted by ASE power)", ylabel="Cumulative share of output ASE",
           title="Where the ASE goes: concentration in high-gain modes")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[1, 1]
    small = mm.step_index_modes(20e-6, 0.1, LAM)
    ge, ga, al = mm.small_signal_rates(small, mm.disk_profile(small, 10e-6), G0, NSP)
    lengths = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
    unc = [4.343 * mm.mode_dependent_gain(mm.linear_channel(small, ge, ga, al, x))["log_gain_std"] for x in lengths]
    ax.plot(lengths, unc, "o-", color=SERIES[0], label="Uncoupled")
    for c, th in zip(SERIES[1:], [0.1, 1.0]):
        v = [np.mean([4.343 * mm.mode_dependent_gain(mm.linear_channel(
            small, ge, ga, al, x, sections=int(64 * x), coupling_strength=th, beta_correlation=1e9, seed=s))["log_gain_std"]
            for s in range(8)]) for x in lengths]
        ax.plot(lengths, v, "o-", color=c, label=f"Random coupling θ = {th} rad per 1.6 cm")
        last = v
    ax.plot(lengths, unc[3] * lengths, color=INK2, lw=1, ls="--", label="∝ L")
    ax.plot(lengths, last[3] * np.sqrt(lengths), color=INK2, lw=1, ls=":", label="∝ √L")
    ax.set(xscale="log", yscale="log", xlabel="Amplifier length (m)", ylabel="Std of eigenmode gains (dB)",
           title=f"MDG accumulation, {small.n_modes} modes, ρ = 0.5 (8-seed mean)")
    ax.set_xticks(lengths, [f"{x:g}" for x in lengths])
    ax.minorticks_off()
    ax.legend(loc="upper left", fontsize=8)

    p = out_path("multimode_mdg.png", sys.argv)
    fig.savefig(p, dpi=150)
    print(p)


if __name__ == "__main__":
    main()
