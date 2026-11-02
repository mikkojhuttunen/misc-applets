"""Step 9: proposal figures from the new runs (PDF, Liberation Sans 10 pt, <= 16.8 cm wide, house colours of the
existing figures; the plotted numbers go to results/step9_figdata.json). Medians and quartiles over the layouts,
never single layouts (the fringe pattern in Fig. 3a is an illustration with the median residual as its rms).

Fig. 2f: static residual fringe (NEA = 2 sqrt(eps) V / sqrt(M) per layout) for no dither / spectral / angle / both:
  R-Herriott, C-Herriott, R-CMPC, C-CMPC of the 600 nm SiNx cells (step 2, R(sin chi), 0.1 dB/cm); chaotic medians with
  interquartile whiskers over 30 layouts, regular single values, target 5e-7 dashed.
Fig. 3b: noise-equivalent absorbance vs averaging time, sqrt(q/(P T)/tau + r(tau)^2) per dither mode, medians with
  min-max bands over the first 21 layouts of the 600 nm circle; shot term dashed, regular cell (both) dotted, NH3 axis
  with the median Gamma <L> and the breath-broadened NH3 peak.
Fig. 3a: HITRAN strips (Fig. 3 lines; NH3 10 ppm, CH4 and C2H6 1 ppm on 6 % H2O, 4.5 % CO2: concentrations inferred from
  the scales of the existing panels) with the 1 s measurement for each dither mode: a static fringe pattern (40 stray
  delays n_eff L with L drawn from an exponential distribution of mean <L>, the dwell statistics of a chaotic cell with a
  small port, random phases, scaled to the median residual at 1 s) plus white shot noise, in alpha units
  (divided by Gamma <L>); near-IR strips with the 600 nm circle (step 2), 3.3 um strips with the 1.2 um / 0.3 dB/cm
  mid-IR circle (step 4).  python scripts/step9_figures.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402
from qs import style  # noqa: E402

FIG = M.RESULTS / "figs"
MODES = ("none", "spectral", "angle", "both")
EPS = 1e-6
P_IN = 1e-3


def fig2f(plt, C, out):
    cells = [("R-Herriott", "herriott_neff1.66", "regular", C["r_herriott"]), ("C-Herriott", "herriott_neff1.66", "chaotic", C["c_herriott"]),
             ("R-CMPC", "circle_neff1.66", "regular", C["r_cmpc"]), ("C-CMPC", "circle_neff1.66", "chaotic", C["c_cmpc"])]
    data = {}
    fig, a = plt.subplots(figsize=(8.4 * style.CM, 6.2 * style.CM), constrained_layout=True)
    w = 0.2
    for j, (lab, f, kind, col) in enumerate(cells):
        res = M.load(M.RESULTS / "step2" / f"{f}.json")
        vals, lo, hi = [], [], []
        for m in MODES:
            if kind == "regular":
                v = res["regular"]["floor"][m]; vals.append(v); lo.append(v); hi.append(v)
            else:
                x = np.array([c["floor"][m] for c in res["chaotic"] if "error" not in c])
                q1, md, q3 = np.percentile(x, [25, 50, 75]); vals.append(md); lo.append(q1); hi.append(q3)
        xs = np.arange(4) + (j - 1.5) * w
        a.bar(xs, vals, w, color=col, label=lab, zorder=2)
        if kind == "chaotic":
            a.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor="k", elinewidth=0.8, capsize=2, zorder=3)
        data[lab] = {"modes": MODES, "value": vals, "q1": lo, "q3": hi, "kind": kind, "source": f"results/step2/{f}.json"}
    a.axhline(5e-7, color=C["target"], ls="--", lw=1.2, zorder=4)
    a.text(-0.45, 5e-7 * 1.25, "target", color=C["target"], fontsize=10)
    a.set_yscale("log"); a.set_ylim(1e-8, 1e-2); a.set_xticks(range(4)); a.set_xticklabels(MODES)
    a.set_xlabel("Dither"); a.set_ylabel("Fringe amplitude ΔP/P"); a.set_title("(f) Static residual fringe", loc="left")
    a.legend(ncol=2, loc="upper right", handlelength=1.2, columnspacing=0.8, borderaxespad=0.1, fontsize=10)
    a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig2f_static_fringe.pdf"); fig.savefig(out / "fig2f_static_fringe.png", dpi=150)
    return data


def curves(res, n=21):
    Tg = np.array(M.fig3_data()["T"])
    ch = [c for c in res["chaotic"][:n] if "error" not in c]
    out = {}
    for m in MODES:
        tot = np.array([np.sqrt(M.Q_E / (P_IN * c["T_trans"]) / Tg + np.array(c["env"][m]) ** 2) for c in ch])
        out[m] = {"median": np.median(tot, 0), "min": tot.min(0), "max": tot.max(0)}
    Tc = float(np.median([c["T_trans"] for c in ch]))
    out["shot"] = np.sqrt(M.Q_E / (P_IN * Tc) / Tg)
    r = res["regular"]
    out["regular_both"] = np.sqrt(M.Q_E / (P_IN * r["T_trans"]) / Tg + np.array(r["env"]["both"]) ** 2)
    gl = res["gamma"] * float(np.median([c["L_mean_m"] for c in ch])) * 100
    return Tg, out, gl, ch


def fig3(plt, C, out):
    res = M.load(M.RESULTS / "step2" / "circle_neff1.66.json")
    mir = M.load(M.RESULTS / "step4" / "circle_neff1.59_0.3dB.json")
    Tg, cv, gl, ch = curves(res)
    Tg4, cv4, gl4, ch4 = curves(mir)
    aNH3 = M.peak_alpha("NH3")
    fig = plt.figure(figsize=(16.8 * style.CM, 6.4 * style.CM), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1])
    data = {"3b": {}, "3a": {}}
    # (b)
    b = fig.add_subplot(gs[:, 1])
    spec = [("none", C["none"], 1.4, "none"), ("spectral", C["lambda"], 1.4, "λ"), ("angle", C["angle"], 1.4, "angle"), ("both", "k", 2.2, "both")]
    for m, col, lw, lab in spec:
        band = C["blue"] if m == "both" else col
        b.fill_between(Tg, cv[m]["min"], cv[m]["max"], color=band, alpha=0.15, lw=0)
        b.loglog(Tg, cv[m]["median"], color=col, lw=lw)
        data["3b"][m] = {k: v.tolist() for k, v in cv[m].items()}
    b.loglog(Tg, cv["shot"], "--", color=C["shot"], lw=1.1)
    b.loglog(Tg, cv["regular_both"], ":", color=C["blue"], lw=1.6)
    i1 = int(np.argmin(abs(np.log(Tg))))
    v1 = float(np.exp(np.interp(0.0, np.log10(Tg), np.log(cv["both"]["median"]))))
    ppb = v1 / (aNH3 * gl) * 1e3
    b.axvline(1, color="0.45", ls="--", lw=0.8)
    b.plot(1, v1, "o", color="k", ms=5, zorder=5)
    b.annotate(f"{ppb:.0f} ppb", (1, v1), (3e-3, v1 / 40), fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.6))
    at = lambda arr, t: float(np.exp(np.interp(math.log10(t), np.log10(Tg), np.log(arr))))
    for m, col, t, f in (("none", C["none"], 1e-2, 2.2), ("spectral", C["lambda"], 1e-2, 2.6), ("angle", C["angle"], 3e-2, 2.6), ("both", "k", 3e-4, 0.12)):
        b.text(t, at(cv[m]["median"], t) * f, {"spectral": "λ"}.get(m, m), color=col, fontsize=10, ha="center")
    b.text(8, at(cv["shot"], 8) * 0.18, "shot", color=C["shot"], fontsize=10, ha="center")
    b.set(xlim=(Tg[0] * 3, Tg[-1]), ylim=(1e-9, 1e-3), xlabel="Averaging time T (s)", ylabel="Noise-equivalent absorbance")
    b.set_title("(b) NH$_3$ noise floor", loc="left")
    r = b.twinx(); r.set_yscale("log"); lo, hi = b.get_ylim(); r.set_ylim(lo / (aNH3 * gl) * 1e3, hi / (aNH3 * gl) * 1e3); r.set_ylabel("NH$_3$ (ppb)")
    for ax_ in (b, r):
        ax_.spines["top"].set_visible(False)
    data["3b"].update(T_s=Tg.tolist(), shot=cv["shot"].tolist(), regular_both=cv["regular_both"].tolist(), gammaL_cm=gl, alpha_NH3=aNH3,
                      ppb_at_1s=ppb, nea_both_1s=v1, source="results/step2/circle_neff1.66.json (21 layouts)")
    # (a)
    strips = {s["key"]: s for s in M.fig3_data()["strips"]}
    panels = [("NH3", "NH$_3$ 1.53 µm", 10.0, 1e-6, C["blue"], (cv, gl, ch)), ("CH4 1.6", "CH$_4$ 1.64 µm", 1.0, 1e-7, C["blue"], (cv, gl, ch)),
              ("CH4 3.3", "CH$_4$ 3.31 µm", 1.0, 1e-5, C["orange"], (cv4, gl4, ch4)), ("C2H6", "C$_2$H$_6$ 3.34 µm", 1.0, 1e-5, C["orange"], (cv4, gl4, ch4))]
    rng = np.random.default_rng(7)
    n_eff_of = {"NH3": 1.66, "CH4 1.6": 1.66, "CH4 3.3": 1.59, "C2H6": 1.59}
    axs = []
    for k, (key, lab, conc, sc, col, (cvv, glv, chv)) in enumerate(panels):
        a = fig.add_subplot(gs[k, 0], sharex=axs[0] if axs else None); axs.append(a)
        s = strips[key]
        x = np.array(s["x_ghz"]); an = conc * np.array(s["analyte_per_ppm"]); bg = sum(np.array(v) for v in s["bg"].values())
        tot = an + bg
        a.fill_between(x, 0, bg / sc, color="0.82", lw=0)
        a.fill_between(x, bg / sc, tot / sc, color=col, alpha=0.9, lw=0)
        Lmed = float(np.median([q["L_mean_m"] for q in chv]))
        opd = n_eff_of[key] * rng.exponential(Lmed, 40)                    # exit paths of a chaotic cell: ~ exponential dwell
        ph = rng.uniform(0, 2 * math.pi, opd.size)
        pat = np.sum(np.cos(2 * math.pi * x[:, None] * 1e9 * opd[None, :] / 2.998e8 + ph[None, :]), 1)
        pat /= np.sqrt(np.mean(pat**2))
        shot1 = math.sqrt(M.Q_E / (P_IN * float(np.median([c["T_trans"] for c in chv]))))
        d = {"x_ghz": x.tolist(), "analyte": an.tolist(), "background": bg.tolist(), "concentration_ppm": conc, "scale": sc, "traces": {}}
        for m, tc in (("none", C["none"]), ("spectral", C["lambda"]), ("angle", C["angle"]), ("both", "k")):
            r1 = float(np.exp(np.interp(0.0, np.log10(Tg), np.log(np.median([np.array(c["env"][m]) for c in chv], 0)))))
            noise = (r1 * pat + shot1 * rng.standard_normal(x.size)) / (glv)                    # absorbance -> alpha [1/cm]
            y = (tot + noise) / sc
            a.plot(x, y, color=tc, lw=0.7 if m != "both" else 1.0, zorder=3 if m == "both" else 2)
            d["traces"][m] = {"alpha_noise_rms_per_cm": float(np.sqrt(r1**2 + shot1**2) / glv), "fringe_rms_1s": r1}
        box = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85)
        a.text(0.015, 0.95, lab, transform=a.transAxes, color=col, fontsize=10, va="top", bbox=box, zorder=6)
        a.text(0.995, 0.95, f"×10$^{{{int(round(math.log10(sc)))}}}$", transform=a.transAxes, fontsize=10, ha="right", va="top", bbox=box, zorder=6)
        top = float(np.max(tot / sc)) * 1.6
        a.set_ylim(0, top); a.set_xlim(x[0], x[-1])
        a.yaxis.set_major_locator(plt.MaxNLocator(2))
        a.spines[["top", "right"]].set_visible(False)
        if k < 3:
            plt.setp(a.get_xticklabels(), visible=False)
        data["3a"][key] = d
    axs[-1].set_xlabel("Detuning (GHz)")
    fig.supylabel("Absorption coefficient α (cm$^{-1}$)", fontsize=10)
    h = [plt.Line2D([], [], color=C["angle"], label="angle"), plt.Line2D([], [], color="k", label="both"),
         plt.Line2D([], [], color=C["lambda"], label="wavelength"), plt.Line2D([], [], color=C["none"], label="no dither, 1 s")]
    axs[0].legend(handles=h, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1, handlelength=1.2, columnspacing=0.8, fontsize=10)
    axs[0].set_title("(a)", loc="left", pad=16)
    fig.savefig(out / "fig3_spectroscopy.pdf"); fig.savefig(out / "fig3_spectroscopy.png", dpi=150)
    return data


def main():
    plt = style.setup()
    FIG.mkdir(parents=True, exist_ok=True)
    C = style.COL
    d2 = fig2f(plt, C, FIG)
    d3 = fig3(plt, C, FIG)
    M.dump(M.RESULTS / "step9_figdata.json", {"fig2f": d2, "fig3": d3})
    for k, v in d2.items():
        print(k, " ".join(f"{m}: {x:.2g}" for m, x in zip(MODES, v["value"])))
    print(f"fig 3b: NH3 at 1 s {d3['3b']['ppb_at_1s']:.0f} ppb (NEA {d3['3b']['nea_both_1s']:.2g}, Gamma<L> {d3['3b']['gammaL_cm']:.2f} cm)")


if __name__ == "__main__":
    main()
