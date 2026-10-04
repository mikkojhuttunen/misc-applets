"""Simulation 1: noise build-up in a chain of N × (lossy span + amplifier).

Each amplifier exactly compensates its span. The engine propagates the signal amplitude and
the ASE correlation through the chain; the dashed lines are the textbook rule
OSNR_dB ≈ 58 + P_dBm - L_dB - NF_dB - 10 log10 N (0.1 nm at 1550 nm).

Usage: python cascade_osnr.py [output_dir]
"""
import sys

import numpy as np

from _style import INK2, SERIES, out_path, plt
from engines.ase_noise import engine as an

LAM, P_IN, NSP, B_REF = 1.55e-6, 1e-3, 1.5, 12.5e9
SPAN_LOSS_DB = [15, 20, 25]
N = np.arange(1, 41)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for c, L in zip(SERIES, SPAN_LOSS_DB):
        eta, g = 10 ** (-L / 10), 10 ** (L / 10)
        res = [an.span_cascade(int(n), eta, g, NSP, P_IN, LAM, B_REF) for n in N]
        osnr = 10 * np.log10([r["osnr"] for r in res])
        nf = 10 * np.log10([r["noise_figure_total"] for r in res])
        nf_amp = 10 * np.log10(an.amplifier_noise(g, NSP)["noise_figure"])
        rule = 58 + 10 * np.log10(P_IN / 1e-3) - L - nf_amp - 10 * np.log10(N)
        ax1.plot(N, osnr, color=c, label=f"{L} dB spans")
        ax1.plot(N, rule, color=INK2, lw=1, ls="--")
        ax1.annotate(f"{L} dB", (N[-1], osnr[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK2)
        ax2.plot(N, nf, color=c, label=f"{L} dB spans")
        ax2.annotate(f"{L} dB", (N[-1], nf[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK2)
    ax1.plot([], [], color=INK2, lw=1, ls="--", label="58 + P − L − NF − 10 log N")
    ax1.set(xscale="log", xlabel="Number of spans N", ylabel="OSNR in 0.1 nm (dB)",
            title=f"OSNR after N spans, launch {10*np.log10(P_IN/1e-3):.0f} dBm, n_sp = {NSP}")
    ax2.set(xscale="log", xlabel="Number of spans N", ylabel="Chain noise figure (dB)",
            title="Chain noise figure (Friis, referred to input)")
    ax1.legend(loc="lower left")
    ax2.legend(loc="upper left")
    p = out_path("cascade_osnr.png", sys.argv)
    fig.savefig(p, dpi=150)
    print(p)


if __name__ == "__main__":
    main()
