"""Task 7: detection limits with the proposal's noise model (T1.3) and set-B fringe residuals; Fig. 3 (a, b) for set B.

    NEA(T) = sqrt( r_m(T)^2 + delta_det(T)^2 ),  delta_det(T) = sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T)),  P_d = P_in 10^(-IL/10)
    limit [ppb] = 1e3 * NEA / (alpha_1ppm * Gamma<L>)
r_m(T): fringe residual of the dither mode m after averaging T (eps = 1e-6, drift 0.1 rad/s), per layout from the set-B
ensembles (near-IR: NH3 at 1531.7 nm; mid-IR: CH4 / C2H6 on the 1000 nm cell at 3315.1 nm). The cell loss IL (14 / 20 /
25 dB, input coupler to detector) replaces the simulated ray transmission (the budget of T1.3 includes the mirrors and
the membrane); Gamma<L> is either the simulated chaotic <L> of each layout times Gamma_E, or the 2 cm milestone. Medians and
IQR over the 30 layouts. NEP 14 pW/sqrt(Hz), P_in 1 / 3 / 10 mW, T = 1 and 15 s. alpha_1ppm from results/t7/lines.json.
-> results/t7/limits.csv, results/t7/summary.json, figs/fig3_setB.pdf
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402
from qv import style  # noqa: E402

LINES = A.load(ROOT / "results" / "t7" / "lines.json")
LAM = {"NH3_1.53": 1.5317e-6, "CH4_3.31": 3.3151e-6, "C2H6_3.34Q": 3.3368e-6}
BAND = {"NH3_1.53": "nir", "CH4_3.31": "mir", "C2H6_3.34Q": "mir"}
GAMMA_LINE = {"NH3_1.53": 0.4532, "CH4_3.31": 0.5526, "C2H6_3.34Q": 0.5622}
TG = A.tgrid()
WORK = ("R0.97", "0.3dB")            # working point for the figure


def limits(res, line, IL, P, T, mode="both", GL=None):
    a = LINES[line]["alpha_peak_per_ppm_cm"]
    out = []
    for x in A.chaotic(res):
        gl = GL if GL is not None else GAMMA_LINE[line] * x["L_mean_m"] * 100
        nea = math.hypot(A.at_T(x["env"][mode], T), A.det_noise(T, P, IL, LAM[line]))
        out.append(1e3 * nea / (a * gl))
    return out


def t_needed(res, line, IL, P, target, GL=None):
    """smallest T on the grid where the median limit <= target (inf if not within the grid)"""
    for T in TG[TG >= 1e-2]:
        if np.median(limits(res, line, IL, P, T, GL=GL)) <= target:
            return float(T)
    return math.inf


def main():
    rows, S = [], {}
    for line in LINES:
        if line not in LAM:
            continue
        for R in ("R0.95", "R0.97", "R0.995"):
            for dB in ("0.3dB", "0.5dB"):
                res = A.load(ROOT / "results" / "t2" / f"circle_{BAND[line]}_{R}_{dB}.json")
                for IL in (14, 20, 25):
                    for P in (1e-3, 3e-3, 1e-2):
                        for T in (1.0, 15.0):
                            for basis, GL in (("sim", None), ("2cm", 2.0)):
                                st = A.stats(limits(res, line, IL, P, T, GL=GL))
                                rows.append({"line": line, "R": R[1:], "dBcm": dB[:-2], "IL_dB": IL, "P_in_mW": P * 1e3, "T_s": T, "GL_basis": basis,
                                             "limit_ppb_median": st["median"], "q1": st["q1"], "q3": st["q3"],
                                             "fringe_1s_median": float(np.median([A.at_T(x["env"]["both"], T) for x in A.chaotic(res)])),
                                             "det_noise": A.det_noise(T, P, IL, LAM[line])})
    with open(ROOT / "results" / "t7" / "limits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ---- the proposal statements (T1.3, M7, Sec. 3.2)
    ch4, c2 = LINES["CH4_3.31"]["alpha_peak_per_ppm_cm"], LINES["C2H6_3.34Q"]["alpha_peak_per_ppm_cm"]
    S["delta_needed_10ppb_2cm"] = {"CH4_3.31": ch4 * 0.01 * 2, "C2H6_3.34Q": c2 * 0.01 * 2}
    S["delta_needed_2ppb_2cm_C2H6"] = c2 * 0.002 * 2
    S["det_noise_1s"] = {f"{IL}dB_{P*1e3:g}mW": A.det_noise(1, P, IL, LAM["CH4_3.31"]) for IL in (14, 20, 25) for P in (1e-3, 3e-3, 1e-2)}
    S["T_for_det_3e-7"] = {f"{IL}dB_{P*1e3:g}mW": (A.det_noise(1, P, IL, LAM["CH4_3.31"]) / 3e-7) ** 2 for IL in (14, 20, 25) for P in (1e-3, 3e-3, 1e-2)}
    S["line_ratio_to_NH3"] = LINES["ratio_to_NH3"]
    # time needed (median over layouts, fringe of each R / loss case) for the M7 targets at the 2 cm milestone and simulated path
    tn = {}
    for R in ("R0.95", "R0.97", "R0.995"):
        for dB in ("0.3dB", "0.5dB"):
            mir = A.load(ROOT / "results" / "t2" / f"circle_mir_{R}_{dB}.json")
            nir = A.load(ROOT / "results" / "t2" / f"circle_nir_{R}_{dB}.json")
            for IL, P in ((14, 1e-3), (20, 1e-3), (20, 1e-2), (25, 1e-3), (25, 3e-3)):
                for basis, GL in (("2cm", 2.0), ("sim", None)):
                    k = f"{R}_{dB}_{IL}dB_{P*1e3:g}mW_{basis}"
                    tn[k] = {"CH4_10ppb": t_needed(mir, "CH4_3.31", IL, P, 10, GL), "C2H6_10ppb": t_needed(mir, "C2H6_3.34Q", IL, P, 10, GL),
                             "C2H6_2ppb": t_needed(mir, "C2H6_3.34Q", IL, P, 2, GL), "NH3_100ppb": t_needed(nir, "NH3_1.53", IL, P, 100, GL),
                             "GL_sim_cm_mir": float(np.median([GAMMA_LINE["CH4_3.31"] * x["L_mean_m"] * 100 for x in A.chaotic(mir)])),
                             "GL_sim_cm_nir": float(np.median([GAMMA_LINE["NH3_1.53"] * x["L_mean_m"] * 100 for x in A.chaotic(nir)])),
                             "fringe_r1s_mir": float(np.median([A.at_T(x["env"]["both"], 1) for x in A.chaotic(mir)])),
                             "fringe_r1s_nir": float(np.median([A.at_T(x["env"]["both"], 1) for x in A.chaotic(nir)]))}
    S["T_needed_s"] = tn
    A.dump(ROOT / "results" / "t7" / "summary.json", S)
    figure()
    print("ok", len(rows))


def figure():
    plt = style.setup()
    C = style.COL
    R, dB = WORK
    nir = A.load(ROOT / "results" / "t2" / f"circle_nir_{R}_{dB}.json")
    mir = A.load(ROOT / "results" / "t2" / f"circle_mir_{R}_{dB}.json")
    IL, P = 14, 1e-3
    fig = plt.figure(figsize=(16.8 * style.CM, 6.6 * style.CM), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1])
    b = fig.add_subplot(gs[:, 1])
    ch = A.chaotic(nir)
    det = np.array([A.det_noise(T, P, IL, LAM["NH3_1.53"]) for T in TG])
    tot = {}
    for m, col, lw, lab in (("none", C["none"], 1.3, "none"), ("spectral", C["lambda"], 1.3, "λ"), ("angle", C["angle"], 1.3, "angle"), ("both", "k", 2.0, "both")):
        arr = np.array([np.sqrt(np.array(x["env"][m]) ** 2 + det ** 2) for x in ch])
        tot[m] = np.median(arr, 0)
        b.fill_between(TG, np.quantile(arr, .25, 0), np.quantile(arr, .75, 0), color=C["blue"] if m == "both" else col, alpha=0.15, lw=0)
        b.loglog(TG, tot[m], color=col, lw=lw, label=lab)
    reg = np.sqrt(np.array(nir["regular"]["env"]["both"]) ** 2 + det ** 2)
    b.loglog(TG, reg, ":", color=C["blue"], lw=1.5, label="both, regular")
    for ILx, ls in ((14, "--"), (20, "-."), (25, (0, (1, 1.5)))):
        b.loglog(TG, [A.det_noise(T, P, ILx, LAM["NH3_1.53"]) for T in TG], ls=ls, color=C["shot"], lw=1.0)
        b.text(TG[-1] * 0.6, A.det_noise(TG[-1], P, ILx, LAM["NH3_1.53"]) * 1.25, f"{ILx} dB", color=C["shot"], fontsize=10, ha="right")
    gl = float(np.median([GAMMA_LINE["NH3_1.53"] * x["L_mean_m"] * 100 for x in ch]))
    a = LINES["NH3_1.53"]["alpha_peak_per_ppm_cm"]
    v1 = float(np.exp(np.interp(0.0, np.log(TG), np.log(tot["both"]))))
    b.plot(1, v1, "o", color="k", ms=4)
    b.axvline(1, color="0.5", ls="--", lw=0.7)
    b.annotate(f"{1e3 * v1 / (a * gl):.0f} ppb", (1, v1), (2e-3, v1 / 20), fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.6))
    b.set(xlim=(1e-4, TG[-1]), ylim=(1e-8, 1e-2), xlabel="Averaging time T (s)", ylabel="Noise-equivalent absorbance")
    b.set_title("(b) NH$_3$, set B, 14 dB, 1 mW", loc="left", fontsize=10)
    b.legend(fontsize=8.5, loc="lower left", ncol=2, handlelength=1.2, columnspacing=0.8)
    r = b.twinx(); r.set_yscale("log"); lo, hi = b.get_ylim(); r.set_ylim(1e3 * lo / (a * gl), 1e3 * hi / (a * gl)); r.set_ylabel("NH$_3$ (ppb)")
    # (a) spectra with noise realisations of the representative layout's exit paths
    strips = {s["key"]: s for s in A.fig3()["strips"]}
    rng = np.random.default_rng(7)
    panels = [("NH3", "NH$_3$ 1.53 µm", 10.0, 1e-6, C["blue"], nir, "NH3_1.53", 1.5391), ("CH4 1.6", "CH$_4$ 1.64 µm", 1.0, 1e-7, C["blue"], nir, "NH3_1.53", 1.5391),
              ("CH4 3.3", "CH$_4$ 3.31 µm", 1.0, 1e-5, C["orange"], mir, "CH4_3.31", 1.4253), ("C2H6", "C$_2$H$_6$ 3.34 µm", 1.0, 1e-5, C["orange"], mir, "C2H6_3.34Q", 1.4253)]
    axs = []
    for k, (key, lab, conc, sc, col, res, line, neff) in enumerate(panels):
        ax = fig.add_subplot(gs[k, 0], sharex=axs[0] if axs else None); axs.append(ax)
        s = strips[key]
        x = np.array(s["x_ghz"]); an = conc * np.array(s["analyte_per_ppm"]); bg = sum(np.array(v) for v in s["bg"].values()); tt = an + bg
        ax.fill_between(x, 0, bg / sc, color="0.82", lw=0)
        ax.fill_between(x, bg / sc, tt / sc, color=col, alpha=0.9, lw=0)
        chv = A.chaotic(res)
        Lm = np.median([q["L_mean_m"] for q in chv])
        rep = min(chv, key=lambda q: abs(q["L_mean_m"] - Lm))
        opd = neff * np.array(rep["Ls_m"]); amp = np.sqrt(np.array(rep["Ws"]))
        ph = rng.uniform(0, 2 * math.pi, opd.size)
        pat = (amp[None, :] * np.cos(2 * math.pi * x[:, None] * 1e9 * opd[None, :] / 2.998e8 + ph[None, :])).sum(1)
        pat /= np.sqrt(np.mean(pat ** 2))
        glv = GAMMA_LINE[line] * Lm * 100
        d1 = A.det_noise(1, P, IL, LAM[line])
        for m, tc in (("none", C["none"]), ("spectral", C["lambda"]), ("angle", C["angle"]), ("both", "k")):
            r1 = float(np.median([A.at_T(q["env"][m], 1) for q in chv]))
            y = (tt + (r1 * pat + d1 * rng.standard_normal(x.size)) / glv) / sc
            ax.plot(x, y, color=tc, lw=0.7 if m != "both" else 1.0, zorder=3 if m == "both" else 2)
        box = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85)
        ax.text(0.015, 0.95, lab, transform=ax.transAxes, color=col, fontsize=10, va="top", bbox=box, zorder=6)
        ax.text(0.995, 0.95, f"×10$^{{{int(round(math.log10(sc)))}}}$", transform=ax.transAxes, fontsize=10, ha="right", va="top", bbox=box, zorder=6)
        ax.set_ylim(0, float(np.max(tt / sc)) * 1.6); ax.set_xlim(x[0], x[-1])
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.spines[["top", "right"]].set_visible(False)
        if k < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
    axs[-1].set_xlabel("Detuning (GHz)")
    fig.supylabel("Absorption coefficient α (cm$^{-1}$)", fontsize=10)
    h = [plt.Line2D([], [], color=C["angle"], label="angle"), plt.Line2D([], [], color="k", label="both"),
         plt.Line2D([], [], color=C["lambda"], label="wavelength"), plt.Line2D([], [], color=C["none"], label="no dither, 1 s")]
    axs[0].legend(handles=h, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1, handlelength=1.2, columnspacing=0.8, fontsize=10)
    axs[0].set_title("(a)", loc="left", pad=16)
    fig.savefig(ROOT / "figs" / "fig3_setB.pdf")
    fig.savefig(ROOT / "figs" / "fig3_setB.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
