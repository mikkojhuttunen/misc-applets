"""Tasks 3 and 4.

Task 3 (eps sweep): dP/P(eps) = 2 sqrt(eps) V / sqrt(M) for every chaotic layout of the best set-B chaotic cell (lowest
median both-dither fringe) and, for reference, every set-B case; eps on a log grid 1e-8 .. 1e-3 (31 points). In the
model eps enters only as the stray amplitude sqrt(eps): V and M are properties of the exit field (they depend on R,
loss, n_eff, geometry and dither, not on eps), so the scaling is exact; the thresholds eps_max(target) =
eps0 (target / dP/P(eps0))^2 are computed for 5e-7 (M3) and 8e-8 (ethane stretch) from the median and the quartiles.
The second-order term eps (stray power) is a static offset, not a fringe; it is checked to stay below the fringe.

Task 4 (path / N_lambda): for the 2 cm gas-equivalent milestone with Gamma = 0.45: L_phys = 2 cm / Gamma, the number of
bounces L_phys / ell (ell = mean chord between wall hits from the simulated cells), the loss
10 log10(1/R) * bounces + alpha_dB * L_phys. The spectral decorrelation of the exit speckle is computed from the
simulated exit-path distribution p(L) of each layout (power weights W_k, path L_k):
    C(dnu) = | sum_k W_k exp(2 pi i dnu n L_k / c) |^2 / (sum_k W_k)^2,
with n = n_eff (as the engine) and n = n_g (the group index of the slab mode, the physically correct one for a frequency
change). For a dither uniform in frequency over +-Dnu the number of independent patterns is
    N_lambda = 1 / <C(nu1 - nu2)>,  nu1, nu2 uniform on [-Dnu, Dnu] (triangular pdf of the difference on [-2Dnu, 2Dnu]);
dnu_c = first dnu with C <= 1/e. The detection-limit scaling to Gamma<L> = 2 cm uses the set-B NEA at 1 s.
-> results/t3/eps_sweep.csv, results/t3/summary.json, figs/eps_sweep.pdf, results/t4/summary.json
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

C0 = 299792458.0
NG = {"nir": 2.2772, "mir": 2.1482}          # slab TM0 group index (results/slab_setB.json)
NEFF = {"nir": 1.5391, "mir": 1.4253}
LAM = {"nir": 1.5317e-6, "mir": 3.3151e-6}
EPS = np.logspace(-8, -3, 31)


def t3(summary):
    keys = [k for k in summary if "_nir_" in k and k.startswith(("circle", "herriott"))]
    best = min((k for k in keys), key=lambda k: summary[k]["fringe_both"]["median"])
    res_rows, out = [], {"best_cell": best, "cases": {}}
    for k in keys + [k for k in summary if "_mir_" in k]:
        s = summary[k]
        f0 = s["fringe_both"]
        th = {}
        for tgt in (5e-7, 8e-8):
            th[f"{tgt:g}"] = {q: 1e-6 * (tgt / f0[q]) ** 2 for q in ("median", "q1", "q3")}
            th[f"{tgt:g}"]["regular"] = 1e-6 * (tgt / s["fringe_both_regular"]) ** 2
        out["cases"][k] = {"fringe_both_eps1e-6": f0, "fringe_both_regular_eps1e-6": s["fringe_both_regular"], "eps_max": th}
    # the sweep itself for the best cell, per layout
    res = A.load(ROOT / "results" / "t2" / f"{best}.json")
    ch = A.chaotic(res)
    for e in EPS:
        v = [A.fringe(x, "both", e) for x in ch]
        st = A.stats(v)
        res_rows.append({"eps": e, "median": st["median"], "q1": st["q1"], "q3": st["q3"], "regular": A.fringe(res["regular"], "both", e),
                         "static_offset_eps": e, "sqrt_eps_slope_check": st["median"] / math.sqrt(e)})
    slopes = np.polyfit(np.log10(EPS), np.log10([r["median"] for r in res_rows]), 1)
    out["loglog_slope"] = float(slopes[0])
    out["eps_where_offset_exceeds_fringe"] = float(next((r["eps"] for r in res_rows if r["eps"] > r["median"]), float("nan")))
    (ROOT / "results" / "t3").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "t3" / "eps_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res_rows[0]))
        w.writeheader()
        w.writerows(res_rows)
    A.dump(ROOT / "results" / "t3" / "summary.json", out)
    plt = style.setup()
    fig, ax = plt.subplots(figsize=(8.4 * style.CM, 6.4 * style.CM), constrained_layout=True)
    ax.fill_between(EPS, [r["q1"] for r in res_rows], [r["q3"] for r in res_rows], color=style.COL["c_cmpc"], alpha=0.25, lw=0)
    ax.plot(EPS, [r["median"] for r in res_rows], color=style.COL["c_cmpc"], label="chaotic, median")
    ax.plot(EPS, [r["regular"] for r in res_rows], color=style.COL["r_cmpc"], label="regular")
    for tgt, ls in ((5e-7, "--"), (8e-8, ":")):
        ax.axhline(tgt, color=style.COL["target"], ls=ls, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Parasitic power fraction ε", ylabel="ΔP/P, both dithers")
    ax.legend(loc="upper left")
    fig.savefig(ROOT / "figs" / "eps_sweep.pdf")
    fig.savefig(ROOT / "figs" / "eps_sweep.png", dpi=150)
    plt.close(fig)
    return out


def corr(Ls, Ws, n, dnu):
    tau = n * np.asarray(Ls) / C0
    w = np.asarray(Ws)
    z = (w[None, :] * np.exp(2j * math.pi * np.asarray(dnu)[:, None] * tau[None, :])).sum(1)
    return np.abs(z) ** 2 / w.sum() ** 2


def n_lambda(Ls, Ws, n, Dnu, npts=401):
    u = np.linspace(-2 * Dnu, 2 * Dnu, npts)
    pdf = (2 * Dnu - np.abs(u)) / (4 * Dnu ** 2)
    c = corr(Ls, Ws, n, u)
    return 1.0 / np.trapezoid(pdf * c, u)


def dnu_c(Ls, Ws, n):
    f = np.logspace(6, 11, 400)
    c = corr(Ls, Ws, n, f)
    i = np.argmax(c <= math.exp(-1))
    return float(f[i]) if c[i] <= math.exp(-1) else float("nan")


def t4(summary):
    out = {"milestone_GL_cm": 2.0, "gamma": 0.45, "L_phys_cm_needed": 2.0 / 0.45, "cases": {}}
    for k, s in summary.items():
        band = "nir" if "_nir_" in k else "mir"
        res = A.load(ROOT / "results" / "t2" / f"{k}.json")
        ch = A.chaotic(res)
        R = float(k.split("_R")[1].split("_")[0])
        dB = float(k.split("_")[-1].replace("dB", ""))
        chord = np.median([x["L_mean_m"] / x["mean_bounces"] for x in ch]) * 100            # cm per bounce
        Lneed = 2.0 / s["gamma"]
        bounces = Lneed / chord
        loss = bounces * 10 * math.log10(1 / R) + dB * Lneed
        row = {"chord_cm": chord, "L_needed_cm": Lneed, "bounces_needed": bounces, "loss_needed_dB": loss,
               "L_sim_cm": s["L_m"]["median"] * 100, "GL_sim_cm": s["GL_cm"], "bounces_sim": s["mean_bounces"]["median"],
               "T_sim": s["T_cell"]["median"]}
        for lab, Dnu in (("0.55GHz", 0.55e9), ("20pm", 20e-12 * C0 / LAM[band] ** 2)):
            for nk, n in (("n_eff", NEFF[band]), ("n_g", NG[band])):
                row[f"N_lambda_pL_{lab}_{nk}"] = A.stats([n_lambda(x["Ls_m"], x["Ws"], n, Dnu) for x in ch])
        for nk, n in (("n_eff", NEFF[band]), ("n_g", NG[band])):
            row[f"dnu_c_GHz_{nk}"] = A.stats([dnu_c(x["Ls_m"], x["Ws"], n) / 1e9 for x in ch])
        row["N_lambda_engine_20pm"] = s.get("N_lambda")
        out["cases"][k] = row
    # old assumption: ~10 cm physical (n_eff 2) -> N_lambda ~ 2-2.7 at +-0.55 GHz; exponential p(L) with mean L:
    # C = 1 / (1 + (2 pi dnu n L / c)^2)
    for L_cm, n in ((10, 2.0), (4.44, NEFF["nir"]), (4.44, NG["nir"])):
        Ls = np.random.default_rng(1).exponential(L_cm / 100, 4000)
        out[f"exponential_pL_L{L_cm}cm_n{n:.2f}"] = {"N_lambda_0.55GHz": float(n_lambda(Ls, np.ones_like(Ls), n, 0.55e9)),
                                                     "dnu_c_GHz": dnu_c(Ls, np.ones_like(Ls), n) / 1e9}
    A.dump(ROOT / "results" / "t4" / "summary.json", out)
    return out


if __name__ == "__main__":
    S = A.load(ROOT / "results" / "t2" / "summary.json")
    a = t3(S)
    print("t3 best", a["best_cell"], "slope", round(a["loglog_slope"], 4))
    b = t4(S)
    print("t4 ok", len(b["cases"]))
