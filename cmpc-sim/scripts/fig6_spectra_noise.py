"""Fig. 6: (a) HITRAN spectra of the five candidate lines as a laser scan would record them, (b) noise floor of the NH3
measurement against averaging time.

(a) Line-by-line Voigt spectra from the cached HITRAN records (../HITRAN/hitran_lbl.py, checked against HAPI): 1 ppm
    analyte in a breath-like background (6 % H2O, 4.5 % CO2) at 310 K and 1 atm, over a +-15 GHz current-tuning scan
    around each line (line wings to 25 cm^-1, pedestal subtracted); in the non-CH4 windows also 1.9 ppm CH4 where HITRAN
    data is cached (the C2H6 window). Stacked: background (grey) + analyte (colour fill); black: the total a detector
    sees; coloured dashed: the analyte alone. Overlaid, one illustrative realisation of the stray-light speckle fringe
    of the cell of (b) on top of the total, converted to alpha with its Gamma <L>: the fringe is a sum of etalon terms,
    one per detected exit path L_k of the cell (weights W_k from the Chaos Lab tracing), sqrt(W_k) cos(2 pi nu n_eff
    L_k / c + phi_k) with random phases, scaled to the rms level of (b) after averaging for T_OVERLAY, plus white shot
    noise. Which dither modes are drawn on which strip is set by OVERLAY (the Fig. 6 Playground artifact writes it):
    none (purple), angle (green), spectral (orange), both (black); "band" adds a +-1 sigma band of a mode. The mid-IR
    strips assume the same cell and absorbance noise.
(b) Noise-equivalent absorbance of the circular multipass cell (A) of the Multipass Chaos Lab artifact, NH3 line at
    1531.68 nm, from its DitherLab engine (data/chaoslab_nh3.json, made by scripts/chaoslab_nh3.js): the residual
    stray-light fringe 2 sqrt(eps) V(T)/sqrt(M^2) after a boxcar average of length T (lobe-averaged), with no dither,
    angle dither (2 mrad, 1 kHz sine), wavelength dither (20 pm, 1297 Hz triangle) or both, with a 0.1 rad/s fringe
    drift; shot noise sqrt(q/(P T_cell T)) for 1 mW; the total of shot noise and the both-dither fringe. Dotted: the
    regular cell (flat facets, no index disks) with both dithers. Right axis: NH3 mixing ratio, with the HITRAN peak
    alpha of (a) and the cell's Gamma <L> (2.2 cm; stray power eps = 1e-6). Curves are labelled directly.
All text Arial 12 pt (Liberation Sans, metrically identical, where Arial is not installed); size FIG_W_CM x FIG_H_CM, margins
fixed in inches so the text keeps its spacing at any size.

Writes results/figs_prelim/fig6_spectra_noise.pdf and .png, and fig6_settings.json with every setting behind them."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "HITRAN"))
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import hitran_lbl as hl

FONT = "Arial" if any(f.name == "Arial" for f in fm.fontManager.ttflist) else "Liberation Sans"   # Liberation Sans: Arial metrics
FS = 12                                                                                             # every text in the figure, pt
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
                     "mathtext.bf": FONT + ":bold", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titlesize": FS, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                     "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8})
ROOT = Path(__file__).resolve().parents[1]
GHZ = 29.9792458                                                      # GHz per cm^-1
WING = 25.0                      # line wings cut at 25 cm^-1, pedestal subtracted: no steps from cut strong lines nearby
Q_E = 1.602176634e-19
BG = {"H2O": 0.06, "CO2": 0.045}
CH4_AMBIENT = 1.9e-6
# noise overlays in (a): per strip, the dither modes drawn (none, angle, spectral, both); optional "band": mode for a +-1 sigma band
OVERLAY = {"NH3": ["angle", "both"], "CH4 1.6": ["angle", "both"], "CH4 3.3": ["angle", "both"], "C2H6": ["angle", "both"], "NO": ["angle", "both"]}
OVERLAY_BAND = {}
T_OVERLAY, OVERLAY_ALPHA, OVERLAY_SEED = 1.0, 0.75, 6
ANALYTE_PPM = 1.0
# figure size [cm] and the share of the panel width that goes to (a)
FIG_W_CM, FIG_H_CM, A_SHARE = 22.0, 7.6, 0.6
HEADROOM = 1.4                    # y range of each strip in (a) as a multiple of its peak: room for the labels
# curves of (b): none, angle, spectral, both, regular, shot, total
B_CURVES = ["none", "angle", "spectral", "both", "regular", "shot", "total"]
MODE_STYLE = {"none": ("C4", "no dither"), "angle": ("C2", "angle"), "spectral": ("C6", "wavelength"), "both": ("k", "both")}
LINES = [("NH$_3$", "NH3", 6528.77, "C0", "NH3"), ("CH$_4$", "CH4", 6114.652, "C0", "CH4 1.6"), ("CH$_4$", "CH4", 3017.80, "C1", "CH4 3.3"),
         ("C$_2$H$_6$", "C2H6", 2996.852, "C1", "C2H6"), ("NO", "NO", 1900.076, "C3", "NO")]

FW, FH = FIG_W_CM / 2.54, FIG_H_CM / 2.54
ML, GAP, MR, MT_A, MB = 45.2 / 72, 67.7 / 72, 39 / 72, 17.1 / 72, 30.5 / 72   # margins [in]: left, between panels, right, top (both panels), bottom
WA = (FW - ML - GAP - MR) * A_SHARE                                          # width of (a) [in]
fig = plt.figure(figsize=(FW, FH))
gsA = fig.add_gridspec(len(LINES), 1, hspace=0.22, left=ML / FW, right=(ML + WA) / FW, top=1 - MT_A / FH, bottom=MB / FH)
gsB = fig.add_gridspec(1, 1, left=(ML + WA + GAP) / FW, right=1 - MR / FW, top=1 - MT_A / FH, bottom=MB / FH)

# ------------------------------------------------------------------ speckle fringe of the cell of (b), for the overlay in (a)
d = json.load(open(ROOT / "data" / "chaoslab_nh3.json"))
T = np.array(d["T"]); cur, reg = d["cells"]["circle_current"], d["cells"]["circle_regular"]
P_in, n_eff = d["settings"]["circle"]["P_in"], d["settings"]["circle"]["n_eff"]
GL = cur["gamma"] * cur["L_mean_m"] * 100                                                     # Gamma <L> [cm]
i1 = int(np.argmin(abs(T - 1.0))); iov = int(np.argmin(abs(np.log(T / T_OVERLAY))))
shot = np.sqrt(Q_E / (P_in * cur["T_trans"] * T))                     # boxcar of length T: noise bandwidth 1/(2T)
both = np.array(cur["scen"]["both"]["env"]); total = np.hypot(both, shot)
Lk, Wk = np.array(cur["Ls_m"]), np.array(cur["Ws"])
rng = np.random.default_rng(OVERLAY_SEED)


def speckle(dnu_hz, rms):
    """One realisation of the stray-light fringe over the scan [absorbance], rms over the scan = rms."""
    f = (np.sqrt(Wk)[:, None] * np.cos(2 * np.pi * dnu_hz[None, :] * n_eff * Lk[:, None] / 299792458.0
                                        + rng.uniform(0, 2 * np.pi, Lk.size)[:, None])).sum(0)
    return rms * f / np.sqrt(np.mean(f ** 2))

# ------------------------------------------------------------------ (a) spectra
tab, axs = [], []
used = []
for k, (lab, sp, nu0, cc, key) in enumerate(LINES):
    a = fig.add_subplot(gsA[k, 0], sharex=axs[0] if axs else None); axs.append(a)
    nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 1201); x = (nu - nu0) * GHZ
    an = hl.alpha(nu, sp, ANALYTE_PPM * 1e-6, wing_cm=WING); bg = sum(hl.alpha(nu, g, xg, wing_cm=WING) for g, xg in BG.items())
    if sp != "CH4" and hl.find_tables("CH4", nu.min(), nu.max()):
        bg = bg + hl.alpha(nu, "CH4", CH4_AMBIENT, wing_cm=WING)
    tot = an + bg
    e = 10.0 ** np.floor(np.log10(tot.max())); top = tot.max() / e
    a.fill_between(x, 0, bg / e, color="0.82", lw=0)
    a.fill_between(x, bg / e, tot / e, color=cc, lw=0, alpha=0.85)
    a.plot(x, tot / e, "k", lw=0.7); a.plot(x, an / e, color=cc, lw=0.9, ls=(0, (3, 1.5)))
    dnu = x * 1e9
    if key in OVERLAY_BAND:
        sb = np.hypot(cur["scen"][OVERLAY_BAND[key]]["env"][iov], shot[iov]) / GL
        a.fill_between(x, (tot - sb) / e, (tot + sb) / e, color=MODE_STYLE[OVERLAY_BAND[key]][0], alpha=0.13, lw=0, zorder=2.5)
    for j, mode in enumerate(OVERLAY.get(key, [])):
        lev = cur["scen"][mode]["env"][iov]
        a.plot(x, (tot + (speckle(dnu, lev) + rng.normal(0, shot[iov], x.size)) / GL) / e, color=MODE_STYLE[mode][0],
               lw=0.5, alpha=OVERLAY_ALPHA, zorder=3 + j)
        if mode not in used: used.append(mode)
    a.set_ylim(0, top * HEADROOM); a.set_yticks([0, np.floor(top * 10) / 10 if top < 2 else np.floor(top)])
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.text(1.0, 0.98, f"×10$^{{{int(np.log10(e))}}}$", transform=a.transAxes, ha="right", va="top", fontsize=FS, color="0.35")
    a.text(0.015, 0.97, f"{lab} {1e4 / nu0:.2f} µm", transform=a.transAxes, ha="left", va="top", fontsize=FS, color=cc,
           bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.3))
    i = int(np.argmax(an)); tab.append(dict(line=f"{sp} {1e4 / nu0:.3f} um", nu_cm=float(nu[i]), alpha_per_ppm_cm=float(an[i]) / ANALYTE_PPM,
                                         background_at_line_cm=float(bg[i]), background_max_scan_cm=float(bg.max())))
    a.tick_params(labelbottom=k == len(LINES) - 1, length=2.5)
    a.set_xlim(-15, 15); a.set_xticks([-10, 0, 10])
axs[-1].set_xlabel("Detuning (GHz)", labelpad=0)
axs[len(LINES) // 2].set_ylabel(r"Absorption coefficient $\alpha$ (cm$^{-1}$)")
axs[len(LINES) // 2].yaxis.set_label_coords(-26 / 72 / WA, 0.5)
fig.text(21.8 / 72 / FW, 1 - (MT_A - 6 / 72) / FH, f"(a) {ANALYTE_PPM:g} ppm in breath background", fontsize=FS, va="baseline")
from matplotlib.lines import Line2D
Tlab = f"{T_OVERLAY:g} s" if T_OVERLAY >= 1 else f"{T_OVERLAY * 1e3:g} ms"
order = [m for m in MODE_STYLE if m in used]
axs[0].legend([Line2D([], [], color=MODE_STYLE[m][0], lw=0.9, alpha=OVERLAY_ALPHA) for m in order],
              [MODE_STYLE[m][1] + (f", {Tlab}" if i == len(order) - 1 else "") for i, m in enumerate(order)], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(order), frameon=False,
              fontsize=FS, handlelength=0.8, handletextpad=0.2, columnspacing=0.75, borderaxespad=0, borderpad=0)

# ------------------------------------------------------------------ (b) noise floor against averaging time
b = fig.add_subplot(gsB[0, 0])
SC = [("none", "no dither", "C4"), ("angle", "angle dither", "C2"), ("spectral", "wavelength dither", "C6"), ("both", "both dithers", "C0")]
for key, name, cc in SC:
    if key in B_CURVES: b.loglog(T, cur["scen"][key]["env"], color=cc, lw=1.4, label=name)
if "regular" in B_CURVES: b.loglog(T, reg["scen"]["both"]["env"], color="C0", lw=1.4, ls=":", label="both, regular cell")
if "shot" in B_CURVES: b.loglog(T, shot, color="0.45", lw=1.0, ls="--", label="shot noise, 1 mW")
if "total" in B_CURVES: b.loglog(T, total, color="k", lw=2.0, label="total (both + shot)")
b.set(xlim=(T[0], T[-1]), ylim=(1e-9, 1e-3), xlabel="Averaging time $T$ (s)", ylabel="Noise-equivalent absorbance")
b.xaxis.labelpad = 0
b.yaxis.set_label_coords(-41 / 72 / ((FW - ML - GAP - MR) * (1 - A_SHARE)), 0.5)
b.set_xticks([1e-4, 1e-2, 1, 1e2]); b.set_yticks([1e-9, 1e-7, 1e-5, 1e-3])
b.grid(alpha=0.25, which="major")
fig.text((ML + WA + GAP) / FW, 1 - (MT_A - 6 / 72) / FH, "(b) NH$_3$ noise floor", fontsize=FS, va="baseline")
a_nh3 = tab[0]["alpha_per_ppm_cm"]
ppb = lambda A: A / (GL * a_nh3) * 1e3
r = b.secondary_yaxis("right", functions=(ppb, lambda c: c * GL * a_nh3 * 1e-3)); r.set_ylabel("NH$_3$ (ppb)", labelpad=1)
r.set_yticks([1e0, 1e2, 1e4, 1e6])
r.spines["right"].set_visible(True)
b.axvline(1.0, color="k", lw=0.6, ls=(0, (2, 2)), alpha=0.6)
b.plot([1.0], [total[i1]], "ko", ms=4)
b.annotate(f"{ppb(total[i1]):.0f} ppb", (1.0, total[i1]), xytext=(-36.8, -34.3), textcoords="offset points", ha="center", va="baseline", fontsize=FS,
           arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
# direct labels instead of a legend: (curve key, T where the label sits, factor above (>1) or below (<1) the curve, text)
LBL = [("none", 0.13, 2.2, "none", "C4"), ("spectral", 0.6, 2.6, "λ", "C6"), ("angle", 1.0, 2.5, "angle", "C2"),
       ("shot", 3.1, 0.2, "shot", "0.35"), ("total", 3.6, 2.0, "both", "k")]
curve_of = {"regular": reg["scen"]["both"]["env"], "shot": shot, "total": total, **{k: cur["scen"][k]["env"] for k in ("none", "angle", "spectral", "both")}}
for key, tl, fac, text, cc in LBL:
    if key in B_CURVES:
        b.text(tl, np.interp(np.log(tl), np.log(T), curve_of[key]) * fac, text, color=cc, fontsize=FS, ha="center", va="center")

out = ROOT / "results" / "figs_prelim"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig6_spectra_noise.pdf"); fig.savefig(out / "fig6_spectra_noise.png", dpi=200)
json.dump(dict(lines=tab, cell=dict(Gamma=cur["gamma"], L_mean_cm=cur["L_mean_m"] * 100, T_cell=cur["T_trans"], M2=cur["M2"]),
               nec_ppb_1s=dict(total=ppb(total[i1]), shot=ppb(shot[i1]), fringe_both=ppb(both[i1]),
                               fringe_both_regular=ppb(reg["scen"]["both"]["env"][i1]))), open(out / "fig6_spectra_noise.json", "w"), indent=1)

# ------------------------------------------------------------------ every setting behind the figure, with provenance
import datetime, subprocess
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
except OSError:
    commit = None
cs = d["settings"]["circle"]
strips = []
for (lab, sp, nu0, cc, key), t in zip(LINES, tab):
    gases = [sp] + list(BG) + (["CH4"] if sp != "CH4" and hl.find_tables("CH4", nu0 - 0.5, nu0 + 0.5) else [])
    strips.append(dict(key=key, species=sp, nu_center_cm=nu0, lambda_um=1e4 / nu0, colour=cc, peak_nu_cm=t["nu_cm"],
                       alpha_per_ppm_cm=t["alpha_per_ppm_cm"], background_at_line_cm=t["background_at_line_cm"],
                       background_gases=gases[1:], noise_overlays=OVERLAY.get(key, []), band=OVERLAY_BAND.get(key),
                       hitran_files=sorted({str(f.relative_to(ROOT.parent)) for g in gases for f in hl.find_tables(g, nu0 - 0.5, nu0 + 0.5)})))
settings = dict(
    generated=dict(script="cmpc-sim/scripts/fig6_spectra_noise.py", date=datetime.datetime.now().isoformat(timespec="seconds"), git_commit=commit,
                   outputs=["fig6_spectra_noise.pdf", "fig6_spectra_noise.png"]),
    figure=dict(width_cm=FIG_W_CM, height_cm=FIG_H_CM, a_width_share=A_SHARE, strip_headroom=HEADROOM, font=FONT, font_size_pt=FS,
                margins_pt=dict(left=ML * 72, between=GAP * 72, right=MR * 72, top=MT_A * 72, bottom=MB * 72), png_dpi=200),
    panel_a=dict(
        spectra=dict(method="line-by-line Voigt from cached HITRAN records (HITRAN/hitran_lbl.py), natural abundance",
                     temperature_K=310.0, pressure_atm=1.0, line_wing_cutoff_cm=WING, wing_pedestal_subtracted=True,
                     scan_full_width_cm=1.0, scan_full_width_GHz=1.0 * GHZ, grid_points=1201, analyte_ppm=ANALYTE_PPM,
                     background_mole_fractions={**BG, "CH4 (non-CH4 strips, where cached)": CH4_AMBIENT}),
        noise_overlay=dict(averaging_time_s=T_OVERLAY, opacity=OVERLAY_ALPHA, seed=OVERLAY_SEED, white_shot_noise=True,
                           model="sum over the cell's detected exit paths L_k of sqrt(W_k) cos(2 pi nu n_eff L_k / c + random phase), "
                                 "scaled to the rms residual of panel (b) at the averaging time, divided by Gamma <L>",
                           mode_colours={k: v[0] for k, v in MODE_STYLE.items()}),
        strips=strips),
    panel_b=dict(
        curves=B_CURVES, marked_time_s=1.0,
        cell=dict(source=d["settings"]["source"], kind="circular, chaotic (index inclusions)", radius_m=cs["c_radius"], facets=cs["n_facets"],
                  launch_deg=float(np.degrees(cs["c_theta"])), n_eff=n_eff, mirror_R=cs["R_mirror"], Gamma=cur["gamma"],
                  L_mean_cm=cur["L_mean_m"] * 100, Gamma_L_cm=GL, transmission=cur["T_trans"], speckle_grains_M2=cur["M2"],
                  inclusions=dict(n=cs["n_inc"], radius_m=cs["inc_radius"], dn=cs["inc_dn"], sign=cs["inc_sign"], seed=cs["inc_seed"])),
        regular_reference="same cell, flat facets, no inclusions",
        dithers=dict(angle=dict(amplitude_rad=cs["dither"], waveform=cs["waveform"], rate_Hz=cs["f_theta"]),
                     wavelength=dict(amplitude_m=cs["dlam"], waveform=cs["lwave"], rate_Hz=cs["f_lambda"])),
        stray_light=dict(power_fraction_eps=cs["f_parasitic"], path_share=cs["stray"], fringe_drift_rad_s=d["settings"]["drift_rad_s"]),
        averaging="boxcar of length T, lobe-averaged; residual 2 sqrt(eps) V(T) / sqrt(M2)",
        shot_noise=dict(input_power_W=P_in, formula="sqrt(q / (P_in T_cell T))"),
        ppb_axis=dict(line="NH3 6528.77 cm^-1", alpha_per_ppm_cm=a_nh3)),
    results=dict(nec_ppb_at_1s=dict(total=ppb(total[i1]), shot=ppb(shot[i1]), fringe_both=ppb(both[i1]),
                                    fringe_both_regular=ppb(reg["scen"]["both"]["env"][i1]))),
)
json.dump(settings, open(out / "fig6_settings.json", "w"), indent=1, ensure_ascii=False)
print("settings:", out / "fig6_settings.json")
for t in tab: print("%-14s %.3f cm-1  alpha/ppm=%.2e  background at line=%.2e" % (t["line"], t["nu_cm"], t["alpha_per_ppm_cm"], t["background_at_line_cm"]))
print("NH3 NEC at 1 s: total %.0f ppb (shot %.0f, fringe %.0f; regular cell fringe %.0f)" % (ppb(total[i1]), ppb(shot[i1]), ppb(both[i1]), ppb(reg["scen"]["both"]["env"][i1])))
