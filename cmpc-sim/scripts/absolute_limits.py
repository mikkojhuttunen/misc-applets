"""Absolute speckle-limited concentration limits for the Fig. 4 budget, from the HITRAN peak absorption.

    c_min = sigma_A / (Gamma * <L> * alpha_ppm)        sigma_A: speckle noise in absorbance (coherence_model.speckle_noise_A)

Needs results/multibeam_nir.pkl (scripts/multibeam_nir.py) and the HITRAN spectra of ../HITRAN/spectra_out
(HITRAN/get_spectra.py, alpha columns already scaled to 1 ppm). Writes results/absolute_limits.json and
results/figs_prelim/fig4_budget_abs.pdf (Fig. 4 with a ppb axis). Speckle only: shot/detector noise, line-fit
details, humidity interference and coupling losses are NOT included. Same model assumptions as multibeam_nir.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))
import json, pickle
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import cmpc_sim as cs, coherence_model as cm

for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito-*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "axes.grid": True, "grid.alpha": 0.25, "axes.linewidth": 0.8})
LAM = 1.55e-6; C = 299792458.0
HIT = Path(__file__).resolve().parents[2] / "HITRAN" / "spectra_out"
MB = pickle.load(open("results/multibeam_nir.pkl", "rb"))
mTM = cs.membrane_mode("Si", 250e-9, LAM, "TM")
inc = lambda pl: cm.merge_incoherent(pl) if len(pl) > 1 else pl[0]


def peak_alpha(csvfile, nu0, half=0.15):
    d = np.genfromtxt(HIT / csvfile, delimiter=",", skip_header=1); m = abs(d[:, 0] - nu0) < half
    return float(d[m, 1].max()), float(d[m][np.argmax(d[m, 1]), 0])


# model line of Fig. 4: nu0 = c / 1.5317 um (NH3, 6528.8 cm^-1), width 2 x 0.095 cm^-1 (5.7 GHz)
a_nh3, nu_nh3 = peak_alpha("spec_NH3_1.5um.csv", 6528.8)          # 1/cm per ppm
nu0 = C / 1.5317e-6; line = 2 * 0.095 * 29979245800.0; dndT = 1.5e-4
dnu = np.concatenate([np.linspace(0, 2e9, 1000), np.linspace(2e9, 30e9, 1000)])
ang = MB[("long", "angles", 16)]; k1 = MB[("long", "ports", 1)][0]; cache = {}


def sigmaA(p, laser, dT):
    if id(p) not in cache: cache[id(p)] = cm.autocovariance(p, dnu)
    return cm.speckle_noise_A(p, dnu, cache[id(p)], dT, nu0, dndT, laser, line, True, 0.0)


d4, d16 = inc(ang[:4]), inc(ang)
steps = [("start: 1 beam, 1 MHz laser, 1 mK drift", k1, 1e6, 1e-3), ("+ 1 GHz laser linewidth", k1, 1e9, 1e-3),
         ("+ beam dither, 4 input modes", d4, 1e9, 1e-3), ("+ beam dither, 16 input modes", d16, 1e9, 1e-3),
         ("+ drift 10 µK", d16, 1e9, 1e-5), ("+ drift 1 µK", d16, 1e9, 1e-6)]
res = []
for name, p, lw, dT in steps:
    sA = sigmaA(p, lw, dT); Leff = mTM.Gamma * p.L_mean * 100.0            # gas-equivalent path [cm]
    res.append(dict(step=name, sigma_A=sA, L_mean_cm=p.L_mean * 100, GammaL_cm=Leff, c_min_ppb=1e3 * sA / (Leff * a_nh3)))
    print("%-42s sigma_A=%.2e  Gamma<L>=%5.1f cm  c_min=%.3g ppb" % (name, sA, Leff, res[-1]["c_min_ppb"]))
json.dump(dict(line="NH3 %.3f cm^-1" % nu_nh3, alpha_per_ppm_cm=a_nh3, Gamma=mTM.Gamma, T_det_long_K1=k1.T_det, steps=res,
               note="speckle-limited only; old assumptions 0.04 dB/cm, 3e-4 bounce loss, R=1-3e-4; to be re-run"),
          open("results/absolute_limits.json", "w"), indent=1)

v = np.array([r["c_min_ppb"] for r in res]); rel = v / v[0]
fig, a = plt.subplots(figsize=(6.69, 2.95)); y = np.arange(len(v))[::-1]
a.barh(y, v, color=["0.55", "C2", "C0", "C0", "C1", "C1"]); a.set_xscale("log"); a.set_xlim(v.min() / 4, v.max() * 3)
for yy, vv, r in zip(y, v, rel): a.text(vv * 1.15, yy, f"{vv:.2g} ppb (×{r:.2g})", va="center", fontsize=10)
a.set_yticks(y); a.set_yticklabels([s[0].replace(", 1 MHz", ",\n1 MHz") for s in steps]); a.grid(axis="y", alpha=0)
a.set_xlabel("speckle-limited detection limit, NH$_3$ line at 1.5317 µm (ppb, single measurement)")
fig.tight_layout(pad=0.4); Path("results/figs_prelim").mkdir(parents=True, exist_ok=True); fig.savefig("results/figs_prelim/fig4_budget_abs.pdf")

# ------------------------------------------------------------------ HITRAN figure: line absorbance vs speckle noise floors
def col(f, nu, half=0.0):
    d = np.genfromtxt(HIT / f, delimiter=",", skip_header=1); return np.interp(nu, d[:, 0], d[:, 1])

nu_w = np.linspace(6528.77 - 0.5, 6528.77 + 0.5, 800); dn_ghz = (nu_w - 6528.77) * 29.9792458
Leff = mTM.Gamma * k1.L_mean * 100.0                                   # gas-equivalent path of the long-path cell [cm]
aN = col("spec_NH3_1.5um.csv", nu_w); aB = col("interf_H2O_NH3_1.5um.csv", nu_w) + col("interf_CO2_NH3_1.5um.csv", nu_w)
fig, ax = plt.subplots(1, 2, figsize=(6.69, 3.1), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
b = ax[0]
lines = [("NH$_3$\n1.532 µm", 6528.77, "spec_NH3_1.5um.csv", "interf_H2O_NH3_1.5um.csv", "C0"),
         ("CH$_4$\n1.635 µm", 6114.652, "spec_CH4_1.6um.csv", "interf_H2O_CH4_1.6um.csv", "C0"),
         ("CH$_4$\n3.313 µm", 3017.80, "spec_CH4_3.3um.csv", "interf_H2O_CH4_3.3um.csv", "C1"),
         ("C$_2$H$_6$\n3.337 µm", 2996.852, "spec_C2H6_3.3um.csv", "interf_H2O_C2H6_3.3um.csv", "C1"),
         ("NO\n5.263 µm", 1900.076, "spec_NO_5.3um.csv", "interf_H2O_NO_5.3um.csv", "C3")]
tab = []
for k, (lab, nu, f, fw, cc) in enumerate(lines):
    pk, nup = peak_alpha(f, nu, 0.05); h2o = col(fw, nup)
    tab.append(dict(line=lab.replace("\n", " "), nu_cm=nup, alpha_per_ppm_cm=pk, H2O_6pct_cm=h2o))
    b.bar(k, pk, color=cc, width=0.6); b.plot([k], [h2o], "ko", ms=5)
b.set_yscale("log"); b.set_xticks(range(len(lines))); b.set_xticklabels([l[0] for l in lines], fontsize=10); b.set_ylim(5e-9, 3e-4)
b.set_ylabel("peak absorption per ppm (cm$^{-1}$)"); b.grid(axis="x", alpha=0); b.set_title("(a) HITRAN line strengths", loc="left", fontsize=10)
b.text(0.02, 0.97, "dot: 6 % H$_2$O at the line", transform=b.transAxes, fontsize=10, va="top")
c = ax[1]
for ppb, ls in ((1000, "-"), (100, "--"), (10, ":")):
    c.semilogy(dn_ghz, np.maximum(Leff * aN * ppb * 1e-3, 1e-12), "C0", ls=ls, lw=1.3, label=f"NH$_3$ {ppb} ppb")
c.semilogy(dn_ghz, np.maximum(Leff * aB, 1e-12), color="0.5", lw=0.9, label="H$_2$O 6 % + CO$_2$ 4.5 %")
for (nm, v, col_) in (("start", res[0]["sigma_A"], "k"), ("+ dither, 16 modes", res[3]["sigma_A"], "C2"), ("+ drift 1 µK", res[5]["sigma_A"], "C1")):
    c.axhline(v, color=col_, lw=1.0); c.text(-14.6, v * 1.2, "speckle: " + nm, ha="left", fontsize=10, color=col_, bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5))
c.set(xlim=(-15, 15), ylim=(1e-7, 3.0), xlabel="detuning from 6528.77 cm$^{-1}$ (GHz)", ylabel=r"absorbance $\Gamma\langle L\rangle\alpha$")
c.legend(loc="upper left", ncol=2, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.0); c.set_title("(b) NH$_3$ line vs speckle floor", loc="left", fontsize=10)
fig.tight_layout(pad=0.4); fig.savefig("results/figs_prelim/fig6_hitran.pdf")
json.dump(tab, open("results/hitran_lines_summary.json", "w"), indent=1)
for t in tab: print("%-16s %.3f cm-1  alpha/ppm=%.2e  H2O6%%=%.2e" % (t["line"], t["nu_cm"], t["alpha_per_ppm_cm"], t["H2O_6pct_cm"]))
