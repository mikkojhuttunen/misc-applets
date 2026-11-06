"""Task 12: breath-resolved measurements at 100 ms (asthma / breath analysis), on the Task 11 grating-coupler design.

python scripts/t12_fast.py      -> results/t12/fast.json, results/t12/table.md, figs/t12_fast.pdf (+png)

1. Detection limits at T = 0.1, 0.5, 1, 2 s from the Task 11 runs (results/t11/final_*.json, 30 layouts per cell):
   noise-equivalent fractional power = hypot(fringe residual, detector) / (Gamma<L> alpha_peak smear), medians and quartiles.
   - fringe residual: the Task 11 model r(T) = sqrt(S_both^2 H_d^2 + (S_none^2 - S_both^2) H_ac^2) with the drift envelope
     H_d(T) = 1/sqrt(1 + 2 (w T / 2)^2) for a fringe-phase drift rate w (rad/s; Task 11 used 0.1) and the AC part at the
     >= 1 kHz dither frequencies H_ac(T) = 1/sqrt(1 + 2 (pi 1000 T)^2).
   - breath referencing: a baseline (inhaled or zero air through the same cartridge) is subtracted every tau seconds; the
     static fringe is left only as its change over tau, |exp(i w tau) - 1| = 2 sin(w tau / 2) (0.30 at 0.1 rad/s, 3 s).
   - detector: sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T)), P_d = P x T_cell x 10^-0.8 (two grating couplers, 8 dB);
     NEP 14 pW/rtHz (TE-cooled photodiode) and, in the mid-IR, 1 nW/rtHz (uncooled); P = 10 and 20 mW.
   Grid: cells d5-d20 (+ the 4 mm reference), R 97 / 99.5 %, w = 0.1 / 0.05 / 0.02 rad/s, tau = none / 3 s.
2. Breath averaging: the end-tidal plateau (1 s) and n = 5, 10 breaths (white and referenced-fringe parts both taken as
   independent between breaths: sigma / sqrt(n); an assumption).
3. One-laser CH4 + C2H6: HITRAN (HAPI, breath background: 35 C, 5.5 % H2O, 4.5 % CO2) spectra in three candidate scan
   windows, convolved with the design's +-81 pm (+-0.073 /cm) triangle wavelength dither (uniform offset distribution),
   least-squares fit of C2H6, CH4, H2O and a linear baseline with white per-point noise. k_g = sigma_fit / sigma_peak-only,
   relative to the unsmeared main line of each gas (C2H6 2996.85 /cm, CH4 3016.49 /cm, lines.json): multiply the
   single-line limit (fringe + detector, unsmeared alpha) by k_g. The fringe is spectrally correlated; applying k to it
   too is an approximation.
4. Gas exchange of the cartridge: flush time V/Q of the channel above the membrane (height h) and diffusion time h^2/(2D).
5. NO (FeNO) at 5.3 um: peak alpha of the strongest line in breath, for reference only (no membrane model at 5.3 um).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

OUT = ROOT / "results/t12"
OUT.mkdir(parents=True, exist_ok=True)
LN = A.load(ROOT / "results/t7/lines.json")
DESIGN = "w75 A60 X100"
BAND = {"nir": 1.5317e-6, "mir": 3.3151e-6}
GASES = {"nir": [("NH3", "NH3_1.53", 0.804)], "mir": [("CH4", "CH4_3.31", 0.921), ("C2H6", "C2H6_3.34Q", 0.824)]}
DL = {"nir": "35pm", "mir": "81pm"}
TS = (0.1, 0.5, 1, 2)
DRIFTS = (0.1, 0.05, 0.02)
TAU = 3.0
CELLS = ("d4ref", "d5", "d10", "d15", "d20")


def Hd(T, w):
    return 1 / math.sqrt(1 + 2 * (w / 2 * T) ** 2)


def Hac(T):
    return 1 / math.sqrt(1 + 2 * (math.pi * 1000 * T) ** 2)


def fringe(S, T, w, tau):
    s, n0 = S["both"], S["none"]
    r = math.sqrt((s * Hd(T, w)) ** 2 + max(n0 ** 2 - s ** 2, 0) * Hac(T) ** 2)
    return r * (2 * math.sin(w * tau / 2) if tau else 1)


def det(T, Pd, lam, NEP):
    return math.sqrt(NEP ** 2 + 2 * 6.626e-34 * 2.998e8 / lam * Pd) / (Pd * math.sqrt(2 * T))


def qs(v):
    return [float(np.quantile(v, p)) for p in (.25, .5, .75)]


# ---------------------------------------------------------------- 1, 2: limits
lim = {}
for band in ("nir", "mir"):
    lam = BAND[band]
    for R in (0.97, 0.995):
        d = json.load(open(ROOT / f"results/t11/final_{band}_R{R}.json"))
        for c in CELLS:
            S = [x["d"][DESIGN] for x in d["cells"][c]["chaotic"] if "error" not in x and x["d"][DESIGN]]
            for g, ln, sm in GASES[band]:
                a = LN[ln]["alpha_peak_per_ppm_cm"]
                for NEP in ((14e-12,) if band == "nir" else (14e-12, 1e-9)):
                    for P in (10e-3, 20e-3):
                        for w in DRIFTS:
                            for tau in (None, TAU):
                                for T in TS:
                                    v, vf, vd = [], [], []
                                    for s in S:
                                        k = 1e3 / (a * s["GL_cm"] * sm)
                                        f = fringe(s["fringe"][DL[band]], T, w, tau)
                                        e = det(T, P * s["T"] * 10 ** -0.8, lam, NEP)
                                        v.append(k * math.hypot(f, e)); vf.append(k * f); vd.append(k * e)
                                    key = f"{band}|R{R}|{c}|{g}|NEP{NEP * 1e12:g}pW|P{P * 1e3:g}mW|w{w}|{'ref' if tau else 'raw'}|T{T}"
                                    lim[key] = {"total": qs(v), "fringe": float(np.median(vf)), "det": float(np.median(vd))}
breath = {}
for key, v in lim.items():
    if key.endswith("|T1") and "|ref|" in key:
        m = v["total"][1]
        breath[key.replace("|T1", "")] = {"plateau_1s": m, "n5": m / math.sqrt(5), "n10": m / math.sqrt(10)}

# ---------------------------------------------------------------- 3: one-laser CH4 + C2H6 windows
from qv import gas as G  # noqa: E402

DNU = 81e-12 / 3.3368e-6 ** 2 * 1e-2        # +-81 pm in /cm at 3.337 um
WINDOWS = {"W2990": (2988.55, 2990.45, "C2H6 2990.06 + CH4 2988.93/2989.02 (1.1 /cm apart)"),
           "W2980": (2978.55, 2980.45, "C2H6 2980.08 + CH4 2978.92/2979.00 (1.1 /cm; H2O on the C2H6 peak)"),
           "W2997": (2996.45, 2999.30, "C2H6 2996.85 (cleanest) + CH4 2999.00 (2.15 /cm apart)")}
MIX = {"CH4": 2.5e-6, "C2H6": 3e-9}         # an example breath: CH4 2.5 ppm, C2H6 3 ppb


def spec(table, x, lo, hi, step=0.002):
    nu, a = G.alpha(table, x, lo - 0.3, hi + 0.3, step)
    return nu, a


def dither(nu, a):
    n = max(1, int(round(DNU / (nu[1] - nu[0]))))
    k = np.ones(2 * n + 1) / (2 * n + 1)
    return np.convolve(a, k, mode="same")


win = {}
for wk, (lo, hi, desc) in WINDOWS.items():
    tab = {"C2H6": "C2H6_2880_3000", "CH4": "CH4_2950_3100", "H2O": "H2O_2950_3100" if lo > 2950 else "H2O_2880_3000"}
    cols, raw = {}, {}
    nu0 = None
    for g, t in tab.items():
        x = G.X_H2O if g == "H2O" else 1e-6
        nu, a = spec(t, x, lo, hi)
        if nu0 is None:
            nu0 = nu
        a = np.interp(nu0, nu, a)
        raw[g] = a
        cols[g] = dither(nu0, a)
    sel = (nu0 >= lo) & (nu0 <= hi)
    nu = nu0[sel]
    X = np.column_stack([cols["C2H6"][sel], cols["CH4"][sel], cols["H2O"][sel], np.ones(sel.sum()), (nu - nu.mean()) / (hi - lo)])
    K = sel.sum()
    cov = np.linalg.inv(X.T @ X)
    k = {}
    for i, (g, ln) in enumerate((("C2H6", "C2H6_3.34Q"), ("CH4", "CH4_3.31"))):
        k[g] = float(math.sqrt(K * cov[i, i]) * LN[ln]["alpha_peak_per_ppm_cm"])
    corr = float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))
    pk = {g: float(cols[g][sel].max()) for g in ("C2H6", "CH4")}      # dither-smeared peak alpha in the window (line hopping)
    win[wk] = {"alpha_sm_peak_per_ppm_cm": pk, "range_cm": [lo, hi], "span_cm": hi - lo, "desc": desc, "k": k, "corr_C2H6_CH4": corr,
               "peak_alpha_per_ppm_cm": {g: float(raw[g][sel].max()) for g in ("C2H6", "CH4")}, "H2O_max_per_cm": float(raw["H2O"][sel].max()),
               "nu": nu[::5].round(4).tolist(), "abs_per_cm": {g: (cols[g][sel] * (MIX[g] / 1e-6 if g != "H2O" else 1))[::5].tolist() for g in ("C2H6", "CH4", "H2O")}}

# single-scan limits: k x the single-line limit with the unsmeared main-line alpha (smear is inside the fit)
scan = {}
for wk, W in win.items():
    for R in (0.97, 0.995):
        for c in ("d10", "d15", "d20"):
            for w in DRIFTS:
                for tau in ("raw", "ref"):
                    for T in TS:
                        out = {}
                        for g, sm in (("CH4", 0.921), ("C2H6", 0.824)):
                            out[g] = lim[f"mir|R{R}|{c}|{g}|NEP14pW|P10mW|w{w}|{tau}|T{T}"]["total"][1] * sm * W["k"][g]
                        scan[f"{wk}|R{R}|{c}|w{w}|{tau}|T{T}"] = out

# line hopping on one laser: dwell on the C2H6 peak for a fraction f of each frame, on the CH4 line for 1 - f (current
# steps of ~1 cm^-1 at the frame rate); the referenced fringe is a fixed residual, only the detector part grows as 1/sqrt(f)
HOP = (0.8, 0.5)
hop = {}
for wk, W in win.items():
    for R in (0.97, 0.995):
        for c in ("d10", "d15", "d20"):
            for w in DRIFTS:
                for tau in ("raw", "ref"):
                    for T in TS:
                        for f in HOP:
                            out = {}
                            for g, ln, sm, fg in (("C2H6", "C2H6_3.34Q", 0.824, f), ("CH4", "CH4_3.31", 0.921, 1 - f)):
                                r = lim[f"mir|R{R}|{c}|{g}|NEP14pW|P10mW|w{w}|{tau}|T{T}"]
                                scale = sm * LN[ln]["alpha_peak_per_ppm_cm"] / W["alpha_sm_peak_per_ppm_cm"][g]
                                out[g] = math.hypot(r["fringe"], r["det"] / math.sqrt(fg)) * scale
                            hop[f"{wk}|R{R}|{c}|w{w}|{tau}|T{T}|f{f}"] = out

# ---------------------------------------------------------------- 4: gas exchange
DIFF = {"CH4": 0.22, "C2H6": 0.15, "NH3": 0.24, "NO": 0.20}   # cm^2/s in air near 35 C (approximate)
exch = {}
for side in (10, 15, 20):
    for h in (0.2, 0.3, 0.5):
        V = side * side * h * 1e-3          # mL
        exch[f"{side}mm_h{h}mm"] = {"V_uL": V * 1e3, "flush_ms": {f"{Q}mL/s": 1e3 * V / Q for Q in (0.2, 0.5, 1, 2)},
                                     "diff_ms": {g: 1e3 * (h * 0.1) ** 2 / (2 * D) for g, D in DIFF.items()}}

# ---------------------------------------------------------------- 5: NO at 5.3 um (reference only)
nu, aNO = G.alpha("NO_1860_1930", 1e-6, 1860, 1930, 0.002)
_, aW = G.alpha("H2O_1860_1930", G.X_H2O, 1860, 1930, 0.002)
_, aC = G.alpha("CO2_1860_1930", G.X_CO2, 1860, 1930, 0.002)
n = min(len(aNO), len(aW), len(aC))
nu, aNO, aW, aC = nu[:n], aNO[:n], aW[:n], aC[:n]
score = aNO / (aW + aC + 1e-3 * aNO.max())          # strong NO with little water / CO2 under it
i_best, i_max = int(np.argmax(score * (aNO > 0.3 * aNO.max()))), int(np.argmax(aNO))
no = {"strongest": {"nu_cm": float(nu[i_max]), "alpha_per_ppm_cm": float(aNO[i_max]), "bg_per_cm": float(aW[i_max] + aC[i_max])},
      "cleanest_strong": {"nu_cm": float(nu[i_best]), "alpha_per_ppm_cm": float(aNO[i_best]), "bg_per_cm": float(aW[i_best] + aC[i_best])},
      "ratio_to_CH4_3.31": float(aNO[i_max] / LN["CH4_3.31"]["alpha_peak_per_ppm_cm"])}

json.dump({"settings": {"T_s": TS, "drift_rad_s": DRIFTS, "tau_ref_s": TAU, "design": DESIGN, "dither": DL, "mix": MIX, "dnu_dither_cm": DNU,
                        "note": "medians (and quartiles) over 30 layouts; limits in ppb, 1 sigma"},
           "limits": lim, "breath": breath, "windows": win, "scan": scan, "hop": hop, "exchange": exch, "NO": no},
          open(OUT / "fast.json", "w"), indent=1)

# ---------------------------------------------------------------- table.md
L = lambda *a: lim["|".join(a)]["total"][1]
md = ["# Task 12: 100 ms breath measurements (generated by scripts/t12_fast.py)", "",
      "Limits in ppb (1σ, median over 30 layouts), grating-coupler design, 10 mW, two grating couplers (8 dB), NEP 14 pW/√Hz unless noted. "
      "raw: no referencing; ref: baseline subtracted every 3 s. w: fringe-phase drift rate (rad/s).", "",
      "## 0.1 s", "", "| cell | R | gas | raw, w 0.1 | ref, w 0.1 | ref, w 0.02 | fringe / detector part (ref, w 0.1) | ref, w 0.1, 20 mW | 1 nW/√Hz, ref |", "|---|---|---|---|---|---|---|---|---|"]
for band, gl in (("mir", ("CH4", "C2H6")), ("nir", ("NH3",))):
    for c in ("d5", "d10", "d15", "d20"):
        for R in (0.97, 0.995):
            for g in gl:
                b = f"{band}|R{R}|{c}|{g}"
                rr = lim[f"{b}|NEP14pW|P10mW|w0.1|ref|T0.1"]
                nep = f"{L(b, 'NEP1000pW', 'P10mW', 'w0.1', 'ref', 'T0.1'):.3g}" if band == "mir" else "–"
                md.append(f"| {c[1:]} mm | {R * 100:g} % | {g} | {L(b, 'NEP14pW', 'P10mW', 'w0.1', 'raw', 'T0.1'):.3g} | {rr['total'][1]:.3g} | "
                          f"{L(b, 'NEP14pW', 'P10mW', 'w0.02', 'ref', 'T0.1'):.3g} | {rr['fringe']:.3g} / {rr['det']:.3g} | {L(b, 'NEP14pW', 'P20mW', 'w0.1', 'ref', 'T0.1'):.3g} | {nep} |")
md += ["", "## Against averaging time (10 mm, R 99.5 %, ref, w 0.1)", "", "| gas | 0.1 s | 0.5 s | 1 s | 2 s | 1 s, 5 breaths | 1 s, 10 breaths |", "|---|---|---|---|---|---|---|"]
for band, g in (("mir", "CH4"), ("mir", "C2H6"), ("nir", "NH3")):
    b = f"{band}|R0.995|d10|{g}|NEP14pW|P10mW|w0.1|ref"
    md.append(f"| {g} | " + " | ".join(f"{lim[b + f'|T{T}']['total'][1]:.3g}" for T in TS) + f" | {breath[b]['n5']:.3g} | {breath[b]['n10']:.3g} |")
md += ["", "## One laser for CH₄ + C₂H₆ (fit of C₂H₆, CH₄, H₂O, linear baseline; ±81 pm dither)", "",
       "| window | span (cm⁻¹) | k C₂H₆ | k CH₄ | corr | CH₄ / C₂H₆ at 0.1 s, 10 mm R 99.5 %, ref w 0.1 | at 1 s |", "|---|---|---|---|---|---|---|"]
for wk, W in win.items():
    a_, b_ = scan[f"{wk}|R0.995|d10|w0.1|ref|T0.1"], scan[f"{wk}|R0.995|d10|w0.1|ref|T1"]
    md.append(f"| {W['desc']} | {W['span_cm']:.2f} | {W['k']['C2H6']:.2f} | {W['k']['CH4']:.2f} | {W['corr_C2H6_CH4']:+.2f} | "
              f"{a_['CH4']:.3g} / {a_['C2H6']:.3g} | {b_['CH4']:.3g} / {b_['C2H6']:.3g} |")
md += ["", "Line hopping on one laser (dwell fraction f on the C₂H₆ peak, 1 − f on the CH₄ line; smeared window peaks), 10 mm, R 99.5 %, ref, w 0.1:", "",
       "| window | f | CH₄ / C₂H₆ at 0.1 s | at 1 s | w 0.02, 0.1 s |", "|---|---|---|---|---|"]
for wk in ("W2990", "W2997"):
    for f in HOP:
        a_, b_, c_ = (hop[f"{wk}|R0.995|d10|w{w}|ref|T{T}|f{f}"] for w, T in ((0.1, 0.1), (0.1, 1), (0.02, 0.1)))
        md.append(f"| {wk} | {f} | {a_['CH4']:.3g} / {a_['C2H6']:.3g} | {b_['CH4']:.3g} / {b_['C2H6']:.3g} | {c_['CH4']:.3g} / {c_['C2H6']:.3g} |")
md += ["", "## Gas exchange (channel above the membrane)", "", "| membrane | h (mm) | V (µL) | flush at 0.5 / 1 / 2 mL/s (ms) | diffusion CH₄ / C₂H₆ (ms) |", "|---|---|---|---|---|"]
for k_, e in exch.items():
    md.append(f"| {k_.split('_')[0]} | {k_.split('_h')[1][:-2]} | {e['V_uL']:.0f} | {e['flush_ms']['0.5mL/s']:.0f} / {e['flush_ms']['1mL/s']:.0f} / {e['flush_ms']['2mL/s']:.0f} | "
              f"{e['diff_ms']['CH4']:.1f} / {e['diff_ms']['C2H6']:.1f} |")
md += ["", f"NO (5.3 µm, reference only): strongest line {no['strongest']['nu_cm']:.2f} cm⁻¹, α {no['strongest']['alpha_per_ppm_cm']:.3g} /(ppm cm) "
       f"({no['ratio_to_CH4_3.31']:.2f} × CH₄ 3.31 µm), breath H₂O + CO₂ under it {no['strongest']['bg_per_cm']:.2g} /cm; "
       f"cleanest strong line {no['cleanest_strong']['nu_cm']:.2f} cm⁻¹ (α {no['cleanest_strong']['alpha_per_ppm_cm']:.3g}, background {no['cleanest_strong']['bg_per_cm']:.2g} /cm)."]
(OUT / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- figure
from qv import style  # noqa: E402

plt = style.setup()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.MAXW, 7.2 * style.CM), constrained_layout=True)
Tg = np.array(TS)
for g, band, col in (("CH4", "mir", style.COL["sin"]), ("C2H6", "mir", style.COL["orange"]), ("NH3", "nir", style.COL["c_cmpc"])):
    b = f"{band}|R0.995|d10|{g}|NEP14pW|P10mW"
    for tag, ls, lab in (("w0.1|raw", ":", "raw"), ("w0.1|ref", "-", "ref."), ("w0.02|ref", "--", "ref., 0.02 rad/s")):
        y = [lim[f"{b}|{tag}|T{T}"]["total"][1] for T in TS]
        ax.plot(Tg, y, ls, color=col, label=f"{g.replace('4', '₄').replace('2H6', '₂H₆').replace('3', '₃')}, {lab}" if g == "CH4" or tag == "w0.1|ref" else None)
ax.set(xscale="log", yscale="log", xlabel="Averaging time (s)", ylabel="Limit, 1σ (ppb)", xticks=[0.1, 0.5, 1, 2], ylim=(0.2, 3e4))
ax.set_xticklabels(["0.1", "0.5", "1", "2"]); ax.minorticks_off()
ax.legend(loc="upper center", fontsize=10, handlelength=1.6, borderaxespad=0.2, labelspacing=0.15, columnspacing=0.8, ncol=2)
ax.text(0.02, 0.02, "(a) 10 mm, R 99.5 %, 10 mW", transform=ax.transAxes)
W = win["W2990"]
x = np.array(W["nu"])
for g, col, sc, lab in (("H2O", style.COL["none"], 1, "H₂O 5.5 %"), ("CH4", style.COL["sin"], 1, "CH₄ 2.5 ppm"), ("C2H6", style.COL["orange"], 1000, "C₂H₆ 3 ppb × 1000")):
    y = 1e6 * np.array(W["abs_per_cm"][g]) * sc
    bx.plot(x, y, color=col)
    j = int(np.argmax(np.where(x > (2989.3 if g == "H2O" else 2988.75), y, 0)))   # label each curve at its own peak (the H2O wing at the left edge excluded)
    if g == "H2O":
        bx.text(x[j] + 0.05, y[j] - 12, lab, color=col, ha="left", va="center")
    else:
        bx.text(x[j], y[j] + 4, lab.replace(" × ", "\n× "), color=col, ha="center", va="bottom")
bx.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="α (10$^{-6}$ cm$^{-1}$)", xlim=(W["range_cm"][0], W["range_cm"][1]), ylim=(0, 150))
bx.ticklabel_format(axis="x", useOffset=False)
bx.text(0.02, 0.96, "(b) 2990 cm$^{-1}$ window,\n±81 pm dither", transform=bx.transAxes, ha="left", va="top", bbox=dict(fc="white", ec="none", pad=1))
fig.savefig(ROOT / "figs/t12_fast.pdf")
fig.savefig(ROOT / "figs/t12_fast.png", dpi=150)
print((OUT / "table.md").read_text())
