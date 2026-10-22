"""Fig. 6: (a) HITRAN spectra of the five candidate lines as a laser scan would record them, (b) noise floor of the NH3
measurement against averaging time.

(a) Line-by-line Voigt spectra from the cached HITRAN records (../HITRAN/hitran_lbl.py, checked against HAPI): 1 ppm
    analyte in a breath-like background (6 % H2O, 4.5 % CO2) at 310 K and 1 atm, over a +-15 GHz current-tuning scan
    around each line (line wings to 25 cm^-1, pedestal subtracted); in the non-CH4 windows also 1.9 ppm CH4 where HITRAN
    data is cached (the C2H6 window). Stacked: background (grey) + analyte (colour fill); black: the total a detector
    sees; coloured dashed: the analyte alone. Overlaid, one illustrative realisation of the stray-light speckle fringe
    of the cell of (b) on top of the total, converted to alpha with its Gamma <L>: the fringe is a sum of etalon terms,
    one per detected exit path L_k of the cell (weights W_k from the Chaos Lab tracing), sqrt(W_k) cos(2 pi nu n_eff
    L_k / c + phi_k) with random phases, scaled to the rms level of (b): purple, undithered (static floor); black, both
    dithers after 1 s (plus white shot noise); purple band: +-1 sigma of the undithered fringe. The mid-IR strips assume the same cell and absorbance noise.
(b) Noise-equivalent absorbance of the circular multipass cell (A) of the Multipass Chaos Lab artifact, NH3 line at
    1531.68 nm, from its DitherLab engine (data/chaoslab_nh3.json, made by scripts/chaoslab_nh3.js): the residual
    stray-light fringe 2 sqrt(eps) V(T)/sqrt(M^2) after a boxcar average of length T (lobe-averaged), with no dither,
    angle dither (2 mrad, 1 kHz sine), wavelength dither (20 pm, 1297 Hz triangle) or both, with a 0.1 rad/s fringe
    drift; shot noise sqrt(q/(P T_cell T)) for 1 mW; the total of shot noise and the both-dither fringe. Dotted: the
    regular cell (flat facets, no index disks) with both dithers. Right axis: NH3 mixing ratio, with the HITRAN peak
    alpha of (a) and the cell's Gamma <L>.

Writes results/figs_prelim/fig6_spectra_noise.pdf and .png."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "HITRAN"))
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import hitran_lbl as hl

for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito-*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8})
ROOT = Path(__file__).resolve().parents[1]
GHZ = 29.9792458                                                      # GHz per cm^-1
WING = 25.0                      # line wings cut at 25 cm^-1, pedestal subtracted: no steps from cut strong lines nearby
Q_E = 1.602176634e-19
BG = {"H2O": 0.06, "CO2": 0.045}
CH4_AMBIENT = 1.9e-6
LINES = [("NH$_3$", "NH3", 6528.77, "C0"), ("CH$_4$", "CH4", 6114.652, "C0"), ("CH$_4$", "CH4", 3017.80, "C1"),
         ("C$_2$H$_6$", "C2H6", 2996.852, "C1"), ("NO", "NO", 1900.076, "C3")]

fig = plt.figure(figsize=(6.69, 3.95))
gsA = fig.add_gridspec(len(LINES), 1, hspace=0.18, left=0.085, right=0.445, top=0.845, bottom=0.12)
gsB = fig.add_gridspec(1, 1, left=0.555, right=0.92, top=0.935, bottom=0.12)

# ------------------------------------------------------------------ speckle fringe of the cell of (b), for the overlay in (a)
d = json.load(open(ROOT / "data" / "chaoslab_nh3.json"))
T = np.array(d["T"]); cur, reg = d["cells"]["circle_current"], d["cells"]["circle_regular"]
P_in, n_eff = d["settings"]["circle"]["P_in"], d["settings"]["circle"]["n_eff"]
GL = cur["gamma"] * cur["L_mean_m"] * 100                                                     # Gamma <L> [cm]
i1 = int(np.argmin(abs(T - 1.0)))
shot = np.sqrt(Q_E / (P_in * cur["T_trans"] * T))                     # boxcar of length T: noise bandwidth 1/(2T)
both = np.array(cur["scen"]["both"]["env"]); total = np.hypot(both, shot)
Lk, Wk = np.array(cur["Ls_m"]), np.array(cur["Ws"])
rng = np.random.default_rng(6)


def speckle(dnu_hz, rms):
    """One realisation of the stray-light fringe over the scan [absorbance], rms over the scan = rms."""
    f = (np.sqrt(Wk)[:, None] * np.cos(2 * np.pi * dnu_hz[None, :] * n_eff * Lk[:, None] / 299792458.0
                                        + rng.uniform(0, 2 * np.pi, Lk.size)[:, None])).sum(0)
    return rms * f / np.sqrt(np.mean(f ** 2))

# ------------------------------------------------------------------ (a) spectra
tab, axs = [], []
for k, (lab, sp, nu0, cc) in enumerate(LINES):
    a = fig.add_subplot(gsA[k, 0], sharex=axs[0] if axs else None); axs.append(a)
    nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 1201); x = (nu - nu0) * GHZ
    an = hl.alpha(nu, sp, 1e-6, wing_cm=WING); bg = sum(hl.alpha(nu, g, xg, wing_cm=WING) for g, xg in BG.items())
    if sp != "CH4" and hl.find_tables("CH4", nu.min(), nu.max()):
        bg = bg + hl.alpha(nu, "CH4", CH4_AMBIENT, wing_cm=WING)
    tot = an + bg
    e = 10.0 ** np.floor(np.log10(tot.max())); top = tot.max() / e
    a.fill_between(x, 0, bg / e, color="0.82", lw=0)
    a.fill_between(x, bg / e, tot / e, color=cc, lw=0, alpha=0.85)
    a.plot(x, tot / e, "k", lw=0.7); a.plot(x, an / e, color=cc, lw=0.9, ls=(0, (3, 1.5)))
    dnu = x * 1e9
    s0 = cur["scen"]["none"]["floor"] / GL
    a.fill_between(x, (tot - s0) / e, (tot + s0) / e, color="C4", alpha=0.13, lw=0, zorder=2.5)
    a.plot(x, (tot + speckle(dnu, cur["scen"]["none"]["floor"]) / GL) / e, color="C4", lw=0.6, alpha=0.5, zorder=3)
    a.plot(x, (tot + (speckle(dnu, both[i1]) + rng.normal(0, shot[i1], x.size)) / GL) / e, color="k", lw=0.5, alpha=0.75, zorder=4)
    a.set_ylim(0, top * 1.12); a.set_yticks([0, np.floor(top * 10) / 10 if top < 2 else np.floor(top)])
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.text(1.0, 0.98, f"×10$^{{{int(np.log10(e))}}}$", transform=a.transAxes, ha="right", va="top", fontsize=8, color="0.35")
    a.text(0.015, 0.97, f"{lab} {1e4 / nu0:.3f} µm", transform=a.transAxes, ha="left", va="top", fontsize=9, color=cc,
           bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.3))
    i = int(np.argmax(an)); tab.append(dict(line=f"{sp} {1e4 / nu0:.3f} um", nu_cm=float(nu[i]), alpha_per_ppm_cm=float(an[i]),
                                         background_at_line_cm=float(bg[i]), background_max_scan_cm=float(bg.max())))
    a.tick_params(labelbottom=k == len(LINES) - 1, length=2.5)
    a.set_xlim(-15, 15)
axs[-1].set_xlabel("detuning (GHz)")
axs[len(LINES) // 2].set_ylabel(r"absorption coefficient $\alpha$ (cm$^{-1}$)")
axs[0].set_title("(a) 1 ppm in 6 % H$_2$O + 4.5 % CO$_2$, 310 K", loc="left", fontsize=10, pad=24)
from matplotlib.lines import Line2D
axs[0].legend([(matplotlib.patches.Patch(color="C4", alpha=0.13, lw=0), Line2D([], [], color="C4", lw=1.0, alpha=0.6)), Line2D([], [], color="k", lw=0.8, alpha=0.75)],
              ["speckle ±1σ, no dither", "both dithers, 1 s"], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False,
              fontsize=8.5, handlelength=1.5, columnspacing=1.0, borderaxespad=0.1)

# ------------------------------------------------------------------ (b) noise floor against averaging time
b = fig.add_subplot(gsB[0, 0])
SC = [("none", "no dither", "C4"), ("angle", "angle dither", "C2"), ("spectral", "wavelength dither", "C1"), ("both", "both dithers", "C0")]
for key, name, cc in SC:
    b.loglog(T, cur["scen"][key]["env"], color=cc, lw=1.4, label=name)
b.loglog(T, reg["scen"]["both"]["env"], color="C0", lw=1.4, ls=":", label="both, regular cell")
b.loglog(T, shot, color="0.45", lw=1.0, ls="--", label="shot noise, 1 mW")
b.loglog(T, total, color="k", lw=2.0, label="total (both + shot)")
b.set(xlim=(T[0], T[-1]), ylim=(3e-10, 1e-3), xlabel="averaging time $T$ (s)", ylabel="noise-equivalent absorbance")
b.grid(alpha=0.25, which="major")
b.set_title("(b) NH$_3$ 1531.68 nm, chaotic circular cell", loc="left", fontsize=10)
a_nh3 = tab[0]["alpha_per_ppm_cm"]
ppb = lambda A: A / (GL * a_nh3) * 1e3
r = b.secondary_yaxis("right", functions=(ppb, lambda c: c * GL * a_nh3 * 1e-3)); r.set_ylabel("NH$_3$ (ppb)")
r.spines["right"].set_visible(True)
b.plot([1.0], [total[i1]], "ko", ms=4)
b.annotate(f"{ppb(total[i1]):.0f} ppb at 1 s", (1.0, total[i1]), xytext=(6.0, 4e-9), ha="center", fontsize=9,
           arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
b.legend(loc="lower left", frameon=False, fontsize=8.5, handlelength=1.8, labelspacing=0.25)
b.text(0.98, 0.97, f"$\\Gamma\\langle L\\rangle$ = {GL:.1f} cm, ε = 10$^{{-6}}$, M$^2$ = {cur['M2']:.0f}", transform=b.transAxes, ha="right", va="top", fontsize=8.5, color="0.3")

out = ROOT / "results" / "figs_prelim"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig6_spectra_noise.pdf"); fig.savefig(out / "fig6_spectra_noise.png", dpi=200)
json.dump(dict(lines=tab, cell=dict(Gamma=cur["gamma"], L_mean_cm=cur["L_mean_m"] * 100, T_cell=cur["T_trans"], M2=cur["M2"]),
               nec_ppb_1s=dict(total=ppb(total[i1]), shot=ppb(shot[i1]), fringe_both=ppb(both[i1]),
                               fringe_both_regular=ppb(reg["scen"]["both"]["env"][i1]))), open(out / "fig6_spectra_noise.json", "w"), indent=1)
for t in tab: print("%-14s %.3f cm-1  alpha/ppm=%.2e  background at line=%.2e" % (t["line"], t["nu_cm"], t["alpha_per_ppm_cm"], t["background_at_line_cm"]))
print("NH3 NEC at 1 s: total %.0f ppb (shot %.0f, fringe %.0f; regular cell fringe %.0f)" % (ppb(total[i1]), ppb(shot[i1]), ppb(both[i1]), ppb(reg["scen"]["both"]["env"][i1])))
