"""Tasks 5, 6 and 8 (set-B near-IR working point: 500 nm SiNx, 1531.7 nm, n_eff 1.5391, Gamma_E 0.4532, R 0.97, 0.3 dB/cm).

Task 5 (angle-dither validity). Speckle model of the engine: the exit field on phase-space channels of one mode each
(frozen-Gaussian beamlets through the cell), C_theta(dtheta) = |<E(0), E(dtheta)>|^2 / (|E(0)|^2 |E(dtheta)|^2), and the
number of independent patterns over a dither, N = (tr G)^2 / ||G||_F^2, G = sum_t E_t E_t^dagger over the dither states
(24 angle states, 16 wavelength states). K_eff = N_theta over +-A, A = 2, 5, 10, 20, 50 mrad; dtheta_c = first dtheta with
C <= 1/e. Port mode count M_modes = 2 n_eff w / lambda (w = 150 um). Scaling test: V_theta vs N_theta^(-1/2) and
V_joint vs (N_theta N_lambda)^(-1/2) and vs N_eff^(-1/2) (log-log fit over layouts; slope -1/2 and unit prefactor
expected for independent random patterns).

Task 6 (inclusions). One parameter at a time around 16 discs x dn 0.01 x r 50 um: dn 0.001-0.01, number 4-32, radius
25-100 um. Chaos metrics: FTLE of the launch beam (lam_ftle, per wall hit), Poincare coverage and entropy of the launch beam
(scripts/chaos_metrics.js), relative width of p(L) (power-weighted std / mean of the exit paths). Fresnel redirection at the
disc edges (in-plane s-polarisation for the TM0 slab mode, n1 = n_eff, n2 = n_eff + dn): <R> over a uniform impact
parameter, entering and leaving (total internal reflection when leaving at grazing incidence); crossings per unit path
from Cauchy's formula for random chords: interfaces per length = n_inc 2 pi r / (pi A), A the area of the inclusion ring.
The redirected light stays in the cell (it is the chaos mechanism), so it is not a loss; a loss only arises from
out-of-plane scattering at the etched step edge, not modelled (measured in T2.3).

Task 8 (layout spread). 100 layouts of the working point: NH3 limit at 1 s (proposal noise model, 14 dB, 1 mW, simulated
Gamma<L>) and both-dither fringe; Spearman rank correlation with the candidate predictors; a one-threshold rule on the best
predictor and its hit rate for picking layouts in the best half.
-> results/t5/summary.json, results/t6/summary.json, results/t8/summary.json, figs/angle_dither.pdf, figs/layout_spread.pdf
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402
from qv import style  # noqa: E402

NEFF, LAM, W_PORT = 1.5391, 1.5317e-6, 150e-6
M_MODES = 2 * NEFF * W_PORT / LAM
ADITH = (2, 5, 10, 20, 50)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_power(x, y):
    p = np.polyfit(np.log(x), np.log(y), 1)
    return {"slope": float(p[0]), "prefactor": float(math.exp(p[1]))}


def t5():
    out = {"M_modes": M_MODES, "cases": {}}
    plt = style.setup()
    fig, ax = plt.subplots(1, 2, figsize=(16.8 * style.CM, 6.2 * style.CM), constrained_layout=True)
    for cell, cr, cc in (("circle", style.COL["r_cmpc"], style.COL["c_cmpc"]), ("herriott", style.COL["r_herriott"], style.COL["c_herriott"])):
        K, Kr = [], []
        for Ad in ADITH:
            f = ROOT / "results" / "t5" / f"{cell}_A{Ad}.json"
            if not f.exists():
                continue
            res = A.load(f)
            ch = A.chaotic(res)
            reg = res["regular"]
            row = {"N_theta": A.stats([x["N_theta"] for x in ch]), "N_theta_regular": reg["N_theta"],
                   "dtheta_c_mrad": A.stats([1e3 * x["dtheta_c"] for x in ch if x.get("dtheta_c") is not None and math.isfinite(x["dtheta_c"])]),
                   "dtheta_c_mrad_regular": 1e3 * reg["dtheta_c"] if reg.get("dtheta_c") else None,
                   "N_lambda": A.stats([x["N_lambda"] for x in ch]), "N_eff": A.stats([x["N_eff"] for x in ch]),
                   "V_theta": A.stats([x["V_theta"] for x in ch]), "V_joint": A.stats([x["V_joint"] for x in ch]),
                   "fringe_angle": A.stats([A.fringe(x, "angle") for x in ch]), "fringe_both": A.stats([A.fringe(x, "both") for x in ch]),
                   "fringe_angle_regular": A.fringe(reg, "angle"), "fringe_both_regular": A.fringe(reg, "both"), "n_ch": A.stats([x["n_ch"] for x in ch])}
            nt = np.array([x["N_theta"] for x in ch]); nl = np.array([x["N_lambda"] for x in ch]); ne = np.array([x["N_eff"] for x in ch])
            vt = np.array([x["V_theta"] for x in ch]); vj = np.array([x["V_joint"] for x in ch])
            row["fit_Vtheta_vs_Ntheta"] = fit_power(nt, vt)
            row["fit_Vjoint_vs_NthetaNlambda"] = fit_power(nt * nl, vj)
            row["ratio_Vjoint_over_(NthNl)^-1/2"] = A.stats(vj * np.sqrt(nt * nl))
            row["ratio_Vjoint_over_Neff^-1/2"] = A.stats(vj * np.sqrt(ne))
            row["ratio_Vtheta_over_Ntheta^-1/2"] = A.stats(vt * np.sqrt(nt))
            out["cases"][f"{cell}_A{Ad}"] = row
            K.append((Ad, row["N_theta"])); Kr.append((Ad, reg["N_theta"]))
            if Ad == 5:                                    # 0.25 mrad sampling up to 10 mrad
                d = np.array(ch[0]["corr_theta"]["d"]) * 1e3
                cm = np.median([x["corr_theta"]["c"] for x in ch], 0)
                ax[0].plot(d, cm, color=cc, lw=1.5, label=f"C-{'CMPC' if cell == 'circle' else 'Herriott'}")
                ax[0].plot(np.array(reg["corr_theta"]["d"]) * 1e3, reg["corr_theta"]["c"], color=cr, lw=1.2, ls="--", label=f"R-{'CMPC' if cell == 'circle' else 'Herriott'}")
        if K:
            a = np.array([k[0] for k in K])
            ax[1].errorbar(a, [k[1]["median"] for k in K], yerr=[[k[1]["median"] - k[1]["q1"] for k in K], [k[1]["q3"] - k[1]["median"] for k in K]],
                           color=cc, marker="o", ms=3.5, capsize=2, label=f"C-{'CMPC' if cell == 'circle' else 'Herriott'}")
            ax[1].plot(a, [k[1] for k in Kr], color=cr, marker="s", ms=3.5, ls="--", label=f"R-{'CMPC' if cell == 'circle' else 'Herriott'}")
    ax[0].axhline(math.exp(-1), color="0.5", lw=0.7, ls=":")
    ax[0].set(xlim=(0, 10), ylim=(0, 1.02), xlabel="Launch-angle change Δθ (mrad)", ylabel="Speckle correlation C(Δθ)")
    ax[0].legend(fontsize=10)
    ax[0].set_title("(a)", loc="left", fontsize=10)
    ax[1].axhline(M_MODES, color=style.COL["target"], lw=1, ls="--")
    ax[1].text(2.2, M_MODES * 0.6, f"M = 2n$_{{eff}}$w/λ = {M_MODES:.0f}", color=style.COL["target"], fontsize=10)
    ax[1].set(xscale="log", yscale="log", xlabel="Angle dither ±A (mrad)", ylabel="Independent angle patterns K$_{eff}$", ylim=(0.8, 600))
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax[1].xaxis.set_major_locator(FixedLocator(list(ADITH))); ax[1].set_xticklabels([str(a) for a in ADITH])
    ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].legend(fontsize=10, loc="center left")
    ax[1].set_title("(b)", loc="left", fontsize=10)
    fig.savefig(ROOT / "figs" / "angle_dither.pdf")
    fig.savefig(ROOT / "figs" / "angle_dither.png", dpi=150)
    plt.close(fig)
    A.dump(ROOT / "results" / "t5" / "summary.json", out)
    return out


def fresnel_s(n1, n2, sin_i):
    """s-polarised power reflectance from n1 into n2 at sin(incidence) (1 beyond the critical angle)"""
    st = n1 * sin_i / n2
    if st >= 1:
        return 1.0
    ci, ct = math.sqrt(1 - sin_i ** 2), math.sqrt(1 - st ** 2)
    r = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    return r * r


def mean_fresnel(dn, n=NEFF, k=4001):
    b = (np.arange(k) + 0.5) / k                      # impact parameter / r, uniform; sin(incidence) = b
    ent = np.mean([fresnel_s(n, n + dn, s) for s in b])
    # leaving the disc: incidence angle inside equals the entering refraction angle: sin = n b / (n + dn)
    lev = np.mean([fresnel_s(n + dn, n, n * s / (n + dn)) for s in b])
    return float(ent), float(lev)


def pl_width(x):
    L, W = np.array(x["Ls_m"]), np.array(x["Ws"])
    m = np.sum(W * L) / np.sum(W)
    return float(math.sqrt(np.sum(W * (L - m) ** 2) / np.sum(W)) / m)


def t6():
    configs = [("dn", 0.001, "circle_dn0.001", "cov_dn0.001"), ("dn", 0.002, "circle_dn0.002", "cov_dn0.002"), ("dn", 0.005, "circle_dn0.005", "cov_dn0.005"),
               ("default", "16 x 0.01 x 50 um", None, "cov_dn0.01"), ("n", 4, "circle_n4", "cov_n4"), ("n", 8, "circle_n8", "cov_n8"), ("n", 32, "circle_n32", "cov_n32"),
               ("radius_um", 25, "circle_r25e-6", "cov_r25e-6"), ("radius_um", 100, "circle_r100e-6", "cov_r100e-6")]
    out = {"cases": {}}
    r2 = (2e-3 * math.cos(math.pi / 24)) ** 2 * math.pi        # cell (apothem disc) area, m^2
    for par, val, ens, cov in configs:
        f = ROOT / "results" / "t2" / "circle_nir_R0.97_0.3dB.json" if ens is None else ROOT / "results" / "t6" / f"{ens}.json"
        if not f.exists():
            continue
        res = A.load(f)
        ch = A.chaotic(res)
        dn = val if par == "dn" else 0.01
        n_inc = val if par == "n" else 16
        rad = (val if par == "radius_um" else 50) * 1e-6
        ent, lev = mean_fresnel(dn)
        per_len = n_inc * 2 * math.pi * rad / (math.pi * r2)               # interfaces per metre of path (Cauchy)
        Lm = float(np.median([x["L_mean_m"] for x in ch]))
        row = {"param": par, "value": val, "fringe_both": A.stats([A.fringe(x, "both") for x in ch]), "fringe_angle": A.stats([A.fringe(x, "angle") for x in ch]),
               "fringe_both_regular": A.fringe(res["regular"], "both"), "lam_ftle": A.stats([x["lam_ftle"] for x in ch]),
               "f_launch_chaotic": A.stats([x["f_launch"] for x in ch]), "N_theta": A.stats([x["N_theta"] for x in ch]),
               "N_lambda": A.stats([x["N_lambda"] for x in ch]), "GL_cm": A.stats([0.4532 * x["L_mean_m"] * 100 for x in ch]),
               "T_cell": A.stats([x["T_trans"] for x in ch]),
               "R_normal_per_interface": (dn / (2 * NEFF)) ** 2, "R_mean_entering": ent, "R_mean_leaving_incl_TIR": lev,
               "interfaces_per_cm": per_len / 100, "interfaces_per_path": per_len * Lm,
               "redirected_fraction_per_path_normal_R": per_len * Lm * (dn / (2 * NEFF)) ** 2,
               "redirected_fraction_per_path_mean_R": per_len * Lm * 0.5 * (ent + lev)}
        if "Ls_m" in ch[0]:
            row["pL_rel_width"] = A.stats([pl_width(x) for x in ch])
        cf = ROOT / "results" / "t6" / f"{cov}.json"
        if cf.exists():
            c = A.load(cf)
            row["coverage"] = A.stats([x["coverage"] for x in c["chaotic"]])
            row["entropy"] = A.stats([x["entropy"] for x in c["chaotic"]])
            row["coverage_regular"] = c["regular"]["coverage"]
        out["cases"][f"{par}={val}"] = row
    hc = ROOT / "results" / "t6" / "cov_herriott.json"
    if hc.exists():
        c = A.load(hc)
        out["herriott_coverage"] = {"regular": c["regular"], "chaotic": A.stats([x["coverage"] for x in c["chaotic"]])}
    A.dump(ROOT / "results" / "t6" / "summary.json", out)
    return out


def t8():
    a = A.load(ROOT / "results" / "t2" / "circle_nir_R0.97_0.3dB.json")
    b8 = ROOT / "results" / "t8" / "circle_seeds31_100.json"
    if not b8.exists():
        return None
    ch = A.chaotic(a) + A.chaotic(A.load(b8))
    cov = {x["seed"]: x for x in A.load(ROOT / "results" / "t8" / "cov_100.json")["chaotic"]} if (ROOT / "results/t8/cov_100.json").exists() else {}
    from t7_analysis import LINES  # noqa: E402
    al = LINES["NH3_1.53"]["alpha_peak_per_ppm_cm"]
    rows = []
    for x in ch:
        nea = math.hypot(A.at_T(x["env"]["both"], 1), A.det_noise(1, 1e-3, 14, 1.5317e-6))
        gl = 0.4532 * x["L_mean_m"] * 100
        r = {"seed": x["seed"], "limit_ppb": 1e3 * nea / (al * gl), "limit_ppb_2cm": 1e3 * nea / (al * 2.0), "fringe_both": A.fringe(x, "both"),
             "fringe_angle": A.fringe(x, "angle"), "GL_cm": gl, "T_cell": x["T_trans"], "lam_ftle": x["lam_ftle"], "f_launch": x["f_launch"],
             "N_theta": x["N_theta"], "N_lambda": x["N_lambda"], "N_eff": x["N_eff"], "M_grains": x["grains"], "pL_rel_width": pl_width(x),
             "sigma_opl_cm": 100 * x["sigma_opl"], "n_pass": x["n_pass"]}
        if x["seed"] in cov:
            r["coverage"], r["entropy"] = cov[x["seed"]]["coverage"], cov[x["seed"]]["entropy"]
        rows.append(r)
    preds = [k for k in rows[0] if k not in ("seed", "limit_ppb", "limit_ppb_2cm", "fringe_both", "fringe_angle")]
    out = {"n_layouts": len(rows), "limit_ppb": A.stats([r["limit_ppb"] for r in rows]), "limit_ppb_2cm": A.stats([r["limit_ppb_2cm"] for r in rows]),
           "fringe_both": A.stats([r["fringe_both"] for r in rows]), "share_fringe_below_5e-7": float(np.mean([r["fringe_both"] < 5e-7 for r in rows])),
           "spearman_vs_limit": {}, "spearman_vs_fringe_both": {}}
    for p in preds:
        v = [r.get(p, float("nan")) for r in rows]
        if any(not math.isfinite(q) for q in v):
            continue
        out["spearman_vs_limit"][p] = spearman(v, [r["limit_ppb"] for r in rows])
        out["spearman_vs_fringe_both"][p] = spearman(v, [r["fringe_both"] for r in rows])
    best = max(out["spearman_vs_fringe_both"], key=lambda p: abs(out["spearman_vs_fringe_both"][p]))
    sgn = np.sign(out["spearman_vs_fringe_both"][best])
    v = np.array([r[best] for r in rows]); f = np.array([r["fringe_both"] for r in rows])
    good = f <= np.median(f)
    thr = np.median(v)
    pick = v <= thr if sgn > 0 else v >= thr
    out["design_rule"] = {"predictor": best, "spearman": out["spearman_vs_fringe_both"][best], "rule": f"{best} {'<=' if sgn > 0 else '>='} {thr:.4g} (its median)",
                          "hit_rate_best_half": float(np.mean(good[pick])), "fringe_median_picked": float(np.median(f[pick])),
                          "fringe_median_rejected": float(np.median(f[~pick]))}
    A.dump(ROOT / "results" / "t8" / "summary.json", out | {"layouts": rows})
    plt = style.setup()
    fig, ax = plt.subplots(1, 2, figsize=(16.8 * style.CM, 6.0 * style.CM), constrained_layout=True)
    lim = np.array([r["limit_ppb"] for r in rows])
    ax[0].hist(np.log10(lim), bins=20, color=style.COL["c_cmpc"], alpha=0.85)
    ax[0].set(xlabel="log$_{10}$ NH$_3$ limit at 1 s (ppb)", ylabel="Layouts")
    ax[0].set_title(f"(a) {len(rows)} layouts, median {np.median(lim):.0f} ppb", loc="left", fontsize=10)
    ax[1].scatter(v, f, s=10, color=style.COL["c_cmpc"])
    ax[1].axhline(5e-7, color=style.COL["target"], ls="--", lw=1)
    ax[1].axvline(thr, color="0.5", ls=":", lw=0.8)
    ax[1].set(yscale="log", xlabel={"n_pass": "Mean passes of the detected light"}.get(best, best), ylabel="ΔP/P, both dithers")
    ax[1].set_title(f"(b) Spearman ρ = {out['design_rule']['spearman']:.2f}", loc="left", fontsize=10)
    fig.savefig(ROOT / "figs" / "layout_spread.pdf")
    fig.savefig(ROOT / "figs" / "layout_spread.png", dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    o5 = t5(); print("t5", len(o5["cases"]))
    o6 = t6(); print("t6", len(o6["cases"]))
    o8 = t8(); print("t8", None if o8 is None else o8["design_rule"])
