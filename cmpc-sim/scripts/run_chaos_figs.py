"""Figures for: speckle vs chaoticity / number of beams, hole-free TE mirror, perturbed regular cell, suppression budget.
Needs chaos_scan.pkl, budget_scan.pkl, mirror_designs.json (python chaos_scan.py; python budget_scan.py after mirror design)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import json, pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cmpc_sim as cs, coherence_model as cm, gas_spectra as gs, mirror_design as md

OUT = Path("results/figs_chaos"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
LAM = 5.2629e-6; NU0 = 1900.08 * gs.C_CM
mA = cs.membrane_mode("Ge", 300e-9, LAM, "TE")
S = pickle.load(open("results/chaos_scan.pkl", "rb")); B = pickle.load(open("results/budget_scan.pkl", "rb"))
DES = json.load(open("data/mirror_designs.json"))["130_16"]
names = list(S.keys()); short = [n.replace("curved ρ=", "curved\nρ=").replace("tilt ", "tilt\n") for n in names]
MULTI = [n for n in names if len(S[n]["paths"]) > 1]
x = np.arange(len(names))


def con(p, lw=1e6): return cm.contrast(p, lw, True)


def figA():
    fig = plt.figure(figsize=(17, 9.6)); gs_ = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs_[0, 0]); ax2 = fig.add_subplot(gs_[0, 1]); ax3 = fig.add_subplot(gs_[0, 2])
    c1 = [con(S[n]["paths"][1][0]) for n in names]
    ax.semilogy(x, c1, "o-", color="k", lw=2, label="K = 1 beam")
    for K, col in ((2, "C0"), (4, "C3")):
        xs = [names.index(n) for n in MULTI]
        ax.semilogy(xs, [con(cm.merge_incoherent(S[n]["paths"][K])) for n in MULTI], "s-", color=col, label=f"K = {K} mutually incoherent beams")
    xs = [names.index(n) for n in MULTI]
    ax.semilogy(xs, [con(cm.merge_coherent(S[n]["paths"][4])) for n in MULTI], "v--", color="C3", mfc="none", label="K = 4 beams from ONE laser (coherent)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set(ylabel="speckle contrast σ_I/⟨I⟩ (all modes summed, 1 MHz laser)", title="Speckle contrast vs cell perturbation and number of beams", ylim=(0.03, 1))
    ax.axvspan(-0.4, 0.4, color="0.9", zorder=0); ax.text(0, 0.045, "regular\npolygon", ha="center", fontsize=7)
    ax.legend(fontsize=7, loc="upper right")
    Ms = [S[n]["paths"][1][0].M_spat() for n in names]; Np = [S[n]["paths"][1][0].n_paths_eff()[0] for n in names]
    ax2.bar(x, Ms, color="C2", alpha=0.8, label="independent output modes M")
    ax2.set_ylabel("M (output modes summed by the detector)"); ax2.set_xticks(x); ax2.set_xticklabels(short, fontsize=7)
    t = ax2.twinx(); t.plot(x, Np, "o-", color="C1", label="paths per mode N_p"); t.set_ylabel("paths per mode N_p", color="C1"); t.grid(False)
    pred = [np.sqrt((1 - 1 / n) / m) for n, m in zip(Np, Ms)]
    ax2.set_title("Why: contrast² ≈ (1 − 1/N_p)/(K·M)")
    t2 = ax2.twinx(); t2.spines["right"].set_position(("axes", 1.18)); t2.plot(x, [c / p for c, p in zip(c1, pred)], "k:", lw=0); t2.set_visible(False)
    ax2.legend(loc="upper left", fontsize=6); t.legend(loc="center left", fontsize=6)
    gl = np.geomspace(1e5, 3e9, 20)
    for n, col in (("regular", "C3"), ("curved ρ=5 cm", "C0")):
        ax3.loglog(gl / 1e6, [con(S[n]["paths"][1][0], g) for g in gl], color=col, label=f"{n}, K = 1")
    ax3.loglog(gl / 1e6, [con(cm.merge_incoherent(S["curved ρ=5 cm"]["paths"][4]), g) for g in gl], "--", color="C0", label="curved ρ=5 cm, K = 4 incoherent")
    ax3.set(xlabel="laser linewidth FWHM (MHz)", ylabel="speckle contrast", title="Linewidth averaging (≤ line width ~3.5 GHz)", ylim=(3e-3, 1)); ax3.legend(fontsize=7)
    dn = np.linspace(-1.5e9, 1.5e9, 3001)
    cases = (("regular", S["regular"]["paths"][1][0], "regular 24-gon, K = 1"), ("curved ρ=5 cm", S["curved ρ=5 cm"]["paths"][1][0], "curved (ρ = 5 cm), K = 1"),
             ("curved4", cm.merge_incoherent(S["curved ρ=5 cm"]["paths"][4]), "curved, K = 4 incoherent beams"))
    for k, (n, p, ttl) in enumerate(cases):
        a = fig.add_subplot(gs_[1, k])
        I = cm.simulate_spectrum(p, dn, NU0, None, "all", seed=3)
        a.plot(dn / 1e9, I, lw=0.7, color=["C3", "C0", "C2"][k])
        a.axhline(1, color="k", lw=0.5)
        a.set(xlabel="detuning (GHz)", ylabel="I(ν) / ⟨I⟩, no gas", ylim=(0, 2.6), title=f"{ttl}: σ = {np.std(I)/np.mean(I):.2f} (model {con(p):.2f})", )
        a.title.set_fontsize(9)
    fig.suptitle("Ge 300 nm TE0, 5.26 µm, 1 cm segmented cell, 100 µm ports, flat mirror R = 0.9997. Spectra: one random-phase realisation of the path model, noise only (no gas).", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "A_speckle_vs_chaoticity_and_beams.png")


def figB():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    sg = np.linspace(0, 0.9999, 2000)
    dbr = cs.DBR(n_tooth=mA.neff, lam_design=LAM, N=20, m_gap=1, m_tooth=1, slab_pol="TE", bounce_loss=0)
    ax[0].semilogy(sg, np.maximum(1 - dbr.R(LAM, sg), 1e-6), "C3", label="two-material (air | Ge 300 nm), N = 20")
    ax[0].semilogy(sg, np.maximum(1 - md.R_tri(LAM, sg, DES["nA"], DES["nC"], DES["g"], DES["c"], DES["a"], DES["N"]), 1e-6), "C0", label="tri-index (+ thinned Ge 130 nm), N = 16")
    ax[0].axvline(1 / mA.neff, color="k", ls=":", lw=0.8); ax[0].axvline(1 / np.sqrt(1 + mA.neff**2), color="k", ls="--", lw=0.8)
    ax[0].set(xlabel="sin χ", ylabel="1 − R", ylim=(1e-6, 2), title="p-pol mirror loss vs angle (dashed: Brewster, dotted: 1/n_eff)"); ax[0].legend(fontsize=7, loc="lower right")
    for sig, col in ((10e-9, "C2"), (20e-9, "C1")):
        ax[1].hist(md.tolerance(LAM, DES, sig, 300), bins=30, alpha=0.6, color=col, label=f"tri-index, σ = {sig*1e9:.0f} nm layer error")
    ax[1].axvline(0.1023, color="C3", lw=2, label="two-material (no error)"); ax[1].axvline(DES["mean_loss"], color="C0", lw=2, ls="--", label="tri-index design")
    ax[1].set(xlabel="angle-averaged ⟨1 − R⟩", ylabel="count", title="Fabrication tolerance (300 Monte-Carlo stacks)"); ax[1].legend(fontsize=7)
    ax[2].axis("off")
    pos = 0; y0 = 0.5; cols = {"gap": "white", "thin": "#9ecae1", "tooth": "#08306b"}
    tot = 3 * (DES["g"] + DES["c"] + DES["a"]); sc = 0.95 / tot
    for p in range(3):
        for key, lab in (("g", "gap"), ("c", "thin"), ("a", "tooth")):
            w = DES[key] * sc
            ax[2].add_patch(plt.Rectangle((0.02 + pos, y0), w, 0.12, fc=cols[lab], ec="k", lw=0.6, transform=ax[2].transAxes)); pos += w
    ax[2].text(0.02, 0.68, "stack from the cell (3 of 16 periods shown): air gap | thinned Ge | full-thickness tooth", fontsize=8, transform=ax[2].transAxes)
    ax[2].text(0.02, 0.36, f"air gap {DES['g']*1e9:.0f} nm (minimum feature)\nthinned-Ge region {DES['c']*1e9:.0f} nm, n_eff = {DES['nC']:.2f} (Ge 130 nm)\nfull tooth {DES['a']*1e9:.0f} nm, n_eff = {DES['nA']:.2f} (Ge 300 nm)\n"
               f"depth {DES['N']*(DES['g']+DES['c']+DES['a'])*1e6:.0f} µm;  ⟨1−R⟩ = {DES['mean_loss']:.4f}, worst angle {DES['max_loss']:.2f}\n\nTwo-material stacks of ANY thickness have R_p ≈ 0 at the\nBrewster angle (all interfaces share it); a third index breaks that.\nThinned region = shallow etch of the same membrane.",
               fontsize=8, va="top", family="monospace", transform=ax[2].transAxes)
    fig.tight_layout(); fig.savefig(OUT / "B_hole_free_TE_mirror.png")


def figC():
    tilt = [n for n in names if n == "regular" or n.startswith("tilt")] + [n for n in names if n.startswith("curved")]
    xi = np.arange(len(tilt)); w = 0.27
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, (lab, col) in enumerate((("two-material", "C3"), ("tri-index", "C0"), ("flat", "0.6"))):
        ax[0].bar(xi + (k - 1) * w, [S[n]["mirrors"][lab]["loss"] for n in tilt], w, color=col, label=lab if lab != "flat" else "ideal flat mirror")
        ax[1].bar(xi + (k - 1) * w, [S[n]["mirrors"][lab]["GL"] * (100 if lab != "flat" else 100) for n in tilt], w, color=col)
    ref = con(S["regular"]["paths"][1][0]) / (S["regular"]["mirrors"]["two-material"]["GL"] * 100)
    for k, (lab, col) in enumerate((("two-material", "C3"), ("tri-index", "C0"))):
        ax[2].bar(xi + (k - 0.5) * w * 1.3, [con(S[n]["paths"][1][0]) / (S[n]["mirrors"][lab]["GL"] * 100) / ref for n in tilt], w * 1.3, color=col, label=lab)
    ax[0].set_yscale("log"); ax[0].set(ylabel="mean loss per bounce 1 − R along detected paths", title="Chaos pushes rays into the Brewster hole of the two-material mirror"); ax[0].legend(fontsize=8)
    ax[1].set(ylabel="Γ⟨L⟩ (cm)", title="Gas-equivalent path (α_bg = 0.03 dB/cm)")
    ax[2].axhline(1, color="k", lw=0.8); ax[2].set(ylabel="relative detection limit", title="∝ contrast / Γ⟨L⟩, rel. to regular + two-material (lower is better)")
    ax[2].title.set_fontsize(8); ax[0].title.set_fontsize(8)
    for a in ax:
        a.set_xticks(xi); a.set_xticklabels([n.replace("curved ρ=", "curved\nρ=").replace("tilt ", "tilt\n") for n in tilt], fontsize=7)
    fig.suptitle("Perturbed regular cell with the two-material TE mirror stays hole-free only for tilt ≲ 0.05°; with the tri-index mirror any perturbation is allowed.", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "C_perturbed_regular_cell_hole_free.png")


def figD():
    a1, _ = gs.peak_alpha_per_ppm("NO"); line_fwhm = 2 * 0.058 * gs.C_CM
    dnu = np.concatenate([np.linspace(0, 2e9, 1000), np.linspace(2e9, 30e9, 1000)]); dn_dT = 3e-4
    cache = {}

    def mdc(key, laser=1e6, span=0.0, dT=1e-3, incoh=True):
        pl = B[key]; p = cm.merge_incoherent(pl) if (len(pl) > 1 and incoh) else pl[0]
        if key not in cache: cache[key] = (p, cm.autocovariance(p, dnu))
        p, g = cache[key]
        A1 = a1 * 1e-3 * p.Gamma * p.L_mean * 100
        return cm.speckle_noise_A(p, dnu, g, dT, NU0, dn_dT, laser, line_fwhm, True, span) / A1, gs.shot_noise_A(1e-3, p.T_det, 1.0) / A1
    k1, k4, k8, w4 = (100e-6, 1), (100e-6, 4), (100e-6, 8), (400e-6, 1)
    steps = [("start: K=1, 100 µm port, 1 MHz laser,\nno dither, 1 mK drift", mdc(k1)),
             ("wider port 400 µm (K=1)", mdc(w4)),
             ("K = 4 incoherent beams", mdc(k4)),
             ("K = 8 incoherent beams", mdc(k8)),
             ("K = 4 + laser 1 GHz", mdc(k4, laser=1e9)),
             ("  + dither (≙ 10 GHz)", mdc(k4, laser=1e9, span=10e9)),
             ("  + drift 10 µK", mdc(k4, laser=1e9, span=10e9, dT=1e-5)),
             ("  + drift 1 µK", mdc(k4, laser=1e9, span=10e9, dT=1e-6))]
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5))
    yy = np.arange(len(steps))[::-1]
    ax[0].barh(yy, [s[1][0] for s in steps], color=["0.5", "C1", "C0", "C0", "C2", "C2", "C2", "C2"])
    for y, s in zip(yy, steps): ax[0].text(s[1][0] * 1.15, y, f"{s[1][0]:.3g} ppb", va="center", fontsize=8)
    ax[0].set_yticks(yy); ax[0].set_yticklabels([s[0] for s in steps], fontsize=8); ax[0].set_xscale("log")
    ax[0].axvspan(5, 50, color="g", alpha=0.15); ax[0].text(6, yy[0] + 0.45, "FeNO 5–50 ppb", color="g", fontsize=8)
    ax[0].axvline(steps[-1][1][1], color="k", ls="-.", lw=1); ax[0].text(steps[-1][1][1] * 1.1, yy[-1] - 0.45, "shot-noise floor (1 mW, 1 Hz)", fontsize=7)
    ax[0].set(xlabel="speckle-limited NO detection limit (ppb)", title="Suppression budget (curved facets + tri-index mirror, Ge 300 nm TE)")
    for key, lab, col in (((100e-6, 1), "K=1", "C3"), ((100e-6, 2), "K=2", "C1"), ((100e-6, 4), "K=4", "C0"), ((100e-6, 8), "K=8", "C2")):
        pass
    Ks = [1, 2, 4, 8]; v = [mdc((100e-6, K))[0] for K in Ks]
    L = [np.mean([p.L_mean for p in B[(100e-6, K)]]) * 100 for K in Ks]; Cc = [con(cm.merge_incoherent(B[(100e-6, K)]) if K > 1 else B[(100e-6, K)][0]) for K in Ks]
    ax[1].plot(Ks, np.array(Cc) / Cc[0], "o-", color="C0", label="speckle contrast (relative)")
    ax[1].plot(Ks, np.array(L) / L[0], "s-", color="C3", label="⟨L⟩ per beam (relative): extra ports leak light")
    ax[1].plot(Ks, np.array(v) / v[0], "^-", color="k", lw=2, label="speckle-limited detection limit (relative)")
    ax[1].plot(Ks, 1 / np.sqrt(Ks), ":", color="C0", label="1/√K")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xticks(Ks); ax[1].set_xticklabels(Ks)
    ax[1].set(xlabel="number of incoherent beams K", title="Beams reduce contrast but every port leaks light (100 µm ports)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "D_suppression_budget.png")
    return steps, Ks, v, L, Cc


if __name__ == "__main__":
    figA(); figB(); figC()
    st, Ks, v, L, Cc = figD()
    for s in st: print("%-45s %10.3g ppb" % (s[0].replace("\n", " "), s[1][0]))
    print("K scaling: K", Ks, "MDC rel", np.round(np.array(v) / v[0], 3), "L rel", np.round(np.array(L) / L[0], 3), "contrast rel", np.round(np.array(Cc) / Cc[0], 3))
    for n in S:
        r = S[n]
        print(f"{n:15s} lyap {r['lyap']:.3f} " + " ".join(f"K{K}={con(cm.merge_incoherent(r['paths'][K])) if K>1 else con(r['paths'][1][0]):.3f}" + (f"(coh {con(cm.merge_coherent(r['paths'][K])):.3f})" if K > 1 else "") for K in r["paths"]))
