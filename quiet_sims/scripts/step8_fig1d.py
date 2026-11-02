"""Step 8: Fig. 1d data for SiNx membranes: TE0 and TM0 n_eff and gas overlap Gamma vs thickness (50-1500 nm) at
1.55 um and 3.0 um, with the constant indices of the brief (n = 2.0, 1.97) and with the LPCVD Si3N4 dispersion of Luke
et al. 2015 (marked; qs/slabmodes.py), both Gamma definitions (gamma_abs = n_g-weighted confinement factor, the
definition of the existing Fig. 1d; gamma_E = |E|^2 fraction in the gas, used in steps 2-5), and the design points
(600 nm at 1.55 um, 1.2 um at 3.3 um). -> results/step8_fig1d.csv, results/step8_fig1d.json, results/figs/fig1d_SiNx.pdf
python scripts/step8_fig1d.py"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import slabmodes as SM  # noqa: E402
from qs import style  # noqa: E402
from qs.metrics import RESULTS, dump  # noqa: E402

T_NM = np.r_[np.arange(50, 1501, 10)]
CASES = [("1.55", 1.55e-6, 2.0), ("3.0", 3.0e-6, 1.97)]


def dndl_luke(lam):
    h = 1e-4
    return float((SM.n_luke(lam * 1e6 + h) - SM.n_luke(lam * 1e6 - h)) / (2 * h * 1e-6))


def main():
    rows, curves = [], {}
    for key, lam, n0 in CASES:
        nl = float(SM.n_luke(lam * 1e6))
        for pol in ("TE", "TM"):
            for src, n, dn in (("const", n0, 0.0), ("Luke2015", nl, dndl_luke(lam))):
                c = {"t_nm": [], "n_eff": [], "n_g": [], "gamma_abs": [], "gamma_E": []}
                for t in T_NM:
                    m = SM.mode(n, t * 1e-9, lam, pol, dn_dlam=dn)
                    for k in ("n_eff", "n_g", "gamma_abs", "gamma_E"):
                        c[k].append(m[k])
                    c["t_nm"].append(int(t))
                    rows.append([key, pol, src, f"{n:.4f}", int(t), f"{m['n_eff']:.5f}", f"{m['n_g']:.5f}", f"{m['gamma_abs']:.5f}", f"{m['gamma_E']:.5f}"])
                curves[f"{key}_{pol}_{src}"] = c
    with open(RESULTS / "step8_fig1d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_um", "pol", "index_model", "n_core", "t_nm", "n_eff", "n_g", "gamma_abs", "gamma_E"])
        w.writerows(rows)
    design = {}
    for name, lam, n, t in (("600 nm @ 1.55 um", 1.55e-6, 2.0, 600e-9), ("1.2 um @ 3.3 um", 3.3e-6, 1.97, 1.2e-6),
                            ("1.2 um @ 3.0 um (on the dashed curve)", 3.0e-6, 1.97, 1.2e-6)):
        design[name] = {pol: SM.mode(n, t, lam, pol) for pol in ("TM", "TE")}
        design[name]["Luke2015_TM"] = SM.mode(float(SM.n_luke(lam * 1e6)), t, lam, "TM", dn_dlam=dndl_luke(lam))
    dump(RESULTS / "step8_fig1d.json", {"design_points": design, "luke_n": {"1.55": float(SM.n_luke(1.55)), "3.0": float(SM.n_luke(3.0)), "3.3": float(SM.n_luke(3.3))},
                                        "notes": ["Luke et al., Opt. Lett. 40, 4823 (2015) Sellmeier for LPCVD Si3N4: coefficients from memory, VERIFY",
                                                  "gamma_abs = n_g * int_gas eps|E|^2 / int eps|E|^2 (Robinson 2008), as in the existing Fig. 1d"]})
    plt = style.setup()
    C = style.COL
    fig, ax = plt.subplots(1, 2, figsize=(15.5 * style.CM, 5.6 * style.CM), constrained_layout=True)
    for key, ls in (("1.55", "-"), ("3.0", "--")):
        for pol, col in (("TM", C["sin"]), ("TE", C["sin_light"])):
            c = curves[f"{key}_{pol}_const"]
            ax[0].plot(c["t_nm"], c["gamma_abs"], ls, color=col)
            ax[1].plot(c["t_nm"], c["n_eff"], ls, color=col)
    for name, lam_key, t, col in (("600 nm @ 1.55 um", None, 600, "k"), ("1.2 um @ 3.0 um (on the dashed curve)", None, 1200, "k")):
        m = design[name]["TM"]
        ax[0].plot(t, m["gamma_abs"], "o", color=col, ms=5, zorder=5)
        ax[1].plot(t, m["n_eff"], "o", color=col, ms=5, zorder=5)
    ax[0].set(xlabel="t (nm)", ylabel="Γ", xlim=(50, 1500), ylim=(0, 1.15))
    ax[1].set(xlabel="t (nm)", ylabel="$n_\\mathrm{eff}$", xlim=(50, 1500), ylim=(1, 2.0))
    ax[0].set_title("(d) SiN$_x$ gas overlap", loc="left")
    ax[1].set_title("effective index", loc="left")
    h = [plt.Line2D([], [], color=C["sin"], label="TM$_0$"), plt.Line2D([], [], color=C["sin_light"], label="TE$_0$"),
         plt.Line2D([], [], color="0.3", ls="-", label="1.55 µm"), plt.Line2D([], [], color="0.3", ls="--", label="3.0 µm"),
         plt.Line2D([], [], color="k", marker="o", ls="", ms=5, label="design")]
    ax[1].legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.6, borderaxespad=0.0, labelspacing=0.3)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    out = RESULTS / "figs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig1d_SiNx.pdf")
    fig.savefig(out / "fig1d_SiNx.png", dpi=150)
    for k, v in design.items():
        print(k, {p: {kk: round(vv, 3) for kk, vv in v[p].items() if kk in ('n_eff', 'gamma_abs', 'gamma_E', 'n_g')} for p in v})


if __name__ == "__main__":
    main()
