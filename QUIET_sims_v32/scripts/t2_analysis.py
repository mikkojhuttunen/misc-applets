"""Task 2: realistic Fig. 2f (set B). Tables of the static residual fringe per geometry x dither, the chaotic/regular
gain, the share of layouts below 5e-7, physical path <L>, effective bounces, transmission, Gamma<L>, the speckle numbers
N_lambda / N_theta / M, and the fringe at the frequency-equivalent of +-0.55 GHz (read from the stored fringe-vs-depth
curve). -> results/t2/summary.json, results/t2/table_fringe.csv, figs/fig2f_setB_<loss>dB.pdf

Definitions (qv/analysis.py): dP/P = 2 sqrt(eps) V / sqrt(M), eps = 1e-6; gain = regular / chaotic median (both dithers);
effective bounces = power-weighted mean number of wall reflections of the detected light (bounceStats in ensemble.js);
N_lambda, N_theta = (tr G)^2 / ||G||_F^2 of the exit-field coherence matrix over 16 wavelength / 24 angle dither states
(engine speckle model); M = M^2 grains of the detected light.
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

R_LIST, DB_LIST = (0.95, 0.97, 0.995), (0.3, 0.5)
GEOM = [("herriott", "nir", "regular", "R-Herriott"), ("herriott", "nir", "chaotic", "C-Herriott"),
        ("circle", "nir", "regular", "R-CMPC"), ("circle", "nir", "chaotic", "C-CMPC")]
C = 299792458.0


def stat_at(run, dl_grid, pm):
    """static fringe (both dithers) at wavelength-dither depth pm, nearest stored grid point (32 per decade)"""
    j = int(np.argmin(np.abs(np.log(dl_grid) - math.log(pm * 1e-12))))
    return run["stat"]["both"][j], dl_grid[j] * 1e12


def case(cell, band, R, dB):
    return ROOT / "results" / "t2" / f"{cell}_{band}_R{R}_{dB}dB.json"


def analyse():
    out, rows = {}, []
    for band, lam in (("nir", 1.5317e-6), ("mir", 3.3151e-6)):
        pm055 = 0.55e9 * lam ** 2 / C * 1e12             # +-0.55 GHz in pm at this wavelength
        for cell in ("circle", "herriott"):
            for R in R_LIST:
                for dB in DB_LIST:
                    f = case(cell, band, R, dB)
                    if not f.exists():
                        continue
                    res = A.load(f)
                    s = A.summary(res)
                    dl = np.array(res["stat_dl"])
                    ch = A.chaotic(res)
                    s["fringe_both_055GHz"] = A.stats([stat_at(x, dl, pm055)[0] for x in ch])
                    s["fringe_both_055GHz_regular"] = stat_at(res["regular"], dl, pm055)[0]
                    s["dither_pm_055GHz_grid"] = stat_at(ch[0], dl, pm055)[1]
                    s["gain_both_055GHz"] = s["fringe_both_055GHz_regular"] / s["fringe_both_055GHz"]["median"]
                    s["mean_chord_mm_regular"] = 1e3 * res["regular"]["L_mean_m"] / max(res["regular"].get("mean_bounces", float("nan")), 1e-9)
                    key = f"{cell}_{band}_R{R}_{dB}dB"
                    out[key] = s
                    for kind in ("regular", "chaotic"):
                        r = {"cell": cell, "band": band, "R": R, "dBcm": dB, "kind": kind}
                        for m in A.MODES:
                            r[f"fringe_{m}"] = s[f"fringe_{m}_regular"] if kind == "regular" else s[f"fringe_{m}"]["median"]
                            if kind == "chaotic":
                                r[f"fringe_{m}_q1"], r[f"fringe_{m}_q3"] = s[f"fringe_{m}"]["q1"], s[f"fringe_{m}"]["q3"]
                        r["fringe_both_055GHz"] = s["fringe_both_055GHz_regular"] if kind == "regular" else s["fringe_both_055GHz"]["median"]
                        for k in ("L_m", "T_cell", "M_grains", "mean_bounces", "N_lambda", "N_theta", "GL_cm"):
                            r[k] = s.get(k + "_regular") if kind == "regular" else (s[k]["median"] if isinstance(s.get(k), dict) else None)
                        rows.append(r)
    A.dump(ROOT / "results" / "t2" / "summary.json", out)
    with open(ROOT / "results" / "t2" / "table_fringe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    return out


def figure(out, dB):
    """Fig. 2f style: bars per geometry for none / spectral / angle / both, one panel per R; chaotic: median + IQR."""
    plt = style.setup()
    COL = {"R-Herriott": style.COL["r_herriott"], "C-Herriott": style.COL["c_herriott"], "R-CMPC": style.COL["r_cmpc"], "C-CMPC": style.COL["c_cmpc"]}
    fig, axs = plt.subplots(1, 3, figsize=(16.8 * style.CM, 6.4 * style.CM), sharey=True, constrained_layout=True)
    xs = np.arange(4)
    for ax, R in zip(axs, R_LIST):
        for j, (cell, band, kind, lab) in enumerate(GEOM):
            s = out.get(f"{cell}_{band}_R{R}_{dB}dB")
            if s is None:
                continue
            v = [s[f"fringe_{m}_regular"] if kind == "regular" else s[f"fringe_{m}"]["median"] for m in A.MODES]
            x = xs + (j - 1.5) * 0.2
            ax.bar(x, v, 0.2, color=COL[lab], label=lab if R == R_LIST[0] else None)
            if kind == "chaotic":
                lo = [s[f"fringe_{m}"]["q1"] for m in A.MODES]
                hi = [s[f"fringe_{m}"]["q3"] for m in A.MODES]
                ax.errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor="k", elinewidth=0.8, capsize=1.5)
        ax.axhline(5e-7, color=style.COL["target"], ls="--", lw=1)
        ax.set(yscale="log", ylim=(5e-9, 3e-3), xticks=xs)
        ax.set_xticklabels(["none", "λ", "angle", "both"])
        ax.set_title(f"R = {R * 100:g} %", fontsize=10)
    axs[0].set_ylabel("Static residual fringe ΔP/P")
    fig.legend(loc="outside upper center", ncol=4, frameon=False)
    p = ROOT / "figs" / f"fig2f_setB_{dB}dB.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    o = analyse()
    for dB in DB_LIST:
        figure(o, dB)
    print("ok", len(o))
