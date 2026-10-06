"""Proposal-ready preliminary figures (17 cm wide, >= 10 pt Carlito text, vector PDF) in figs_prelim/.
Main design: free-standing Si membrane (released SOI device layer), TM0, 1.55 um; mid-IR example: Ge 300 nm TE0, 5.26 um."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import json, os, pickle, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import cmpc_sim as cs, coherence_model as cm, mirror_design as md

for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "mathtext.fontset": "custom", "mathtext.rm": "Carlito",
                     "mathtext.it": "Carlito:italic", "mathtext.bf": "Carlito:bold", "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.linewidth": 0.8})
OUT = Path("results/figs_prelim"); OUT.mkdir(parents=True, exist_ok=True)
WIDTH = 6.69
LAM = 1.55e-6; C = 299792458.0
mTM = cs.membrane_mode("Si", 250e-9, LAM, "TM")
W = 30e-6; TH0 = LAM / (mTM.neff * W); CHI = np.pi / 2 - np.pi * 7 / 24
ALPHA = cs.dB_per_cm_to_alpha(0.1); BL = 3e-4
R_flat = lambda s: np.full_like(s, 1 - BL)
deg = np.radians
CELLS = {"regular": {}, "tilt 0.05°": dict(tilt_rms=deg(0.05)), "tilt 0.1°": dict(tilt_rms=deg(0.1)), "tilt 0.3°": dict(tilt_rms=deg(0.3)),
         "tilt 1°": dict(tilt_rms=deg(1.0)), "curved ρ=20 cm": dict(curvature=1 / 0.20), "curved ρ=10 cm": dict(curvature=1 / 0.10),
         "curved ρ=5 cm": dict(curvature=1 / 0.05)}
MULTI = ("regular", "tilt 0.3°", "curved ρ=5 cm")


def mkcell(name): return cs.SegmentedCell(5e-3, 24, seed=1, **CELLS[name])


# ---------------------------------------------------------------- data: 1.55 um coherence scan
if not Path("results/chaos_scan_nir.pkl").exists():
    out = {}; t0 = time.time()
    for name in CELLS:
        cell = mkcell(name); P = cell.perimeter
        rec = dict(paths={})
        for K in ((1, 2, 4) if name in MULTI else (1,)):
            ports = [cell.h + j * P / K for j in range(K)]; rec["paths"][K] = []
            for k in range(K):
                tab = cs.trace_rays(cell, W, 6000, 800, TH0, seed=5 + 10 * K + k, max_path=3.0, keep_start=False, theta_c=CHI,
                                    s_in=ports[k], s_out=cell.h + 18 * cell.h, extra_ports=[q for j, q in enumerate(ports) if j != k])
                rec["paths"][K].append(cm.prepare(tab, R_flat, ALPHA, mTM.Gamma, LAM, mTM.neff, mTM.n_group, W)); del tab
        out[name] = rec
        p = rec["paths"][1][0]
        print(f"[{time.time()-t0:3.0f}s] {name:15s} T={p.T_det:.3f} <L>={p.L_mean*100:.0f}cm paths/mode={p.n_paths_eff()[0]:.1f} M={p.M_spat():.0f} C={cm.contrast(p,1e6,True):.3f}", flush=True)
    pickle.dump(out, open("results/chaos_scan_nir.pkl", "wb"))
S = pickle.load(open("results/chaos_scan_nir.pkl", "rb"))
con = lambda p, lw=1e6: cm.contrast(p, lw, True)
inc = lambda n, K: cm.merge_incoherent(S[n]["paths"][K]) if K > 1 else S[n]["paths"][1][0]


# ---------------------------------------------------------------- Fig 1: concept
def fig1():
    fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 4.9), gridspec_kw=dict(height_ratios=[1, 0.95]))
    for j, (name, ttl) in enumerate((("regular", "(a) regular cell, flat facets"), ("curved ρ=5 cm", "(b) curved facets (ρ = 5 cm)"))):
        cell = mkcell(name); a = ax[0, j]
        x, y = cell.outline(); a.plot(x * 1e3, y * 1e3, "k", lw=1.0)
        x0, y0, nx, ny = cell.point_at(np.array([cell.h]))
        for off, col in ((0.0, "C0"), (0.004, "C3")):
            th = np.arctan2(-ny[0], -nx[0]) + CHI + off
            xs, ys = cs.trace_path(cell, x0[0], y0[0], np.cos(th), np.sin(th), n_hits=40)
            a.plot(xs * 1e3, ys * 1e3, lw=0.6, color=col)
        a.plot(x0 * 1e3, y0 * 1e3, "o", color="tab:green", ms=6)
        xo, yo, _, _ = cell.point_at(np.array([cell.h + 18 * cell.h])); a.plot(xo * 1e3, yo * 1e3, "o", color="tab:red", ms=6)
        a.set_aspect("equal"); a.axis("off"); a.set_title(ttl, loc="left")
        Sg, SC = cs.poincare(cell, 40, 300, seed=j)
        b = ax[1, j]; b.plot(Sg.ravel() / cell.perimeter, SC.ravel(), ".", ms=0.9, color="k", alpha=0.6, rasterized=True)
        b.set(xlim=(0, 1), ylim=(-1, 1), xlabel="boundary position s / perimeter", ylabel="sin χ", xticks=[0, 0.5, 1], yticks=[-1, 0, 1])
        b.set_title("(c) Poincaré section" if j == 0 else "(d) Poincaré section", loc="left")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig1_concept.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: speckle law
def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.35), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]; dn = np.linspace(-1.5e9, 1.5e9, 3001); nu0 = C / 1.5317e-6
    cases = (("regular", S["regular"]["paths"][1][0], "regular", "C3"), ("curved", S["curved ρ=5 cm"]["paths"][1][0], "chaotic", "C0"),
             ("curved4", inc("curved ρ=5 cm", 4), "chaotic, 4 beams", "C2"))
    info = []
    for k, (key, p, lab, col) in enumerate(cases):
        I = cm.simulate_spectrum(p, dn, nu0, None, "all", seed=3)
        off = (2 - k) * 2.3
        a.plot(dn / 1e9, I - 1 + off, lw=0.7, color=col); a.axhline(off, color="0.6", lw=0.5)
        a.text(-1.45, off + 1.05, f"{lab}: σ = {np.std(I)/np.mean(I):.2f}", fontsize=10, va="bottom")
        info.append((lab, np.std(I) / np.mean(I), con(p)))
    a.set(xlabel="detuning (GHz)", yticks=[], ylabel="intensity, no gas (offset)", xlim=(-1.5, 1.5), ylim=(-1.4, 6.9)); a.grid(False)
    a.set_title("(a) simulated spectral speckle", loc="left")
    b = ax[1]
    fam = {"regular": ("k", "o"), "tilt": ("C1", "^"), "curved": ("C0", "s")}
    for n in S:
        p = S[n]["paths"][1][0]; f = "regular" if n == "regular" else ("tilt" if n.startswith("tilt") else "curved")
        b.loglog(p.M_spat(), con(p), fam[f][1], color=fam[f][0], ms=6, mfc="none")
    lab_done = False
    for K, mk in ((2, "D"), (4, "D")):
        for n in MULTI:
            p = inc(n, K); b.loglog(p.M_spat(), con(p), mk, color="C3", ms=5)
    for n in MULTI:
        p = cm.merge_coherent(S[n]["paths"][4]); b.loglog(p.M_spat(), con(p), "v", color="C3", mfc="none", ms=6)
    xs = np.geomspace(3, 300, 20); b.loglog(xs, 1 / np.sqrt(xs), "k:", lw=1)
    b.plot([], [], "o", color="k", mfc="none", label="one beam"); b.plot([], [], "D", color="C3", label="K = 2, 4 incoherent"); b.plot([], [], "v", color="C3", mfc="none", label="K = 4, one laser")
    b.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    b.set(xlabel="independent output modes M", ylabel="speckle contrast", xlim=(3, 300), ylim=(0.03, 1.1)); b.set_title("(b) contrast ≈ M$^{-1/2}$", loc="left")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig2_speckle.pdf"); plt.close(fig)
    return info


# ---------------------------------------------------------------- Fig 3: device (1.55 um Si membrane)
def fig3():
    import run_figures as rf
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.15))
    a = ax[0]; sg = np.linspace(0, 0.9999, 2000)
    for pol, d0, col, lab in (("TE", 220e-9, "C3", "TE$_0$ (p-pol)"), ("TM", 250e-9, "C0", "TM$_0$ (s-pol)")):
        m = cs.membrane_mode("Si", d0, LAM, pol, with_group_index=False)
        dbr = cs.DBR(n_tooth=m.neff, lam_design=LAM, N=10, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
        a.semilogy(sg, np.maximum(1 - dbr.R(LAM, sg), 1e-7), color=col, lw=1.4, label=lab)
    a.annotate("Brewster\nhole", (0.31, 0.5), (0.05, 2e-3), color="C3", arrowprops=dict(arrowstyle="->", color="C3"), fontsize=10)
    a.set(xlabel="sin χ at the mirror", ylabel="mirror loss 1 − R", ylim=(1e-7, 2), xlim=(0, 1)); a.legend(loc="lower right", frameon=False)
    a.set_title("(a) mirror loss vs angle", loc="left")
    b = ax[1]
    tab = rf.make_table("curved", rf.ref_mode(), n_rays=3000)
    d = np.linspace(150e-9, 600e-9, 17); res = {}
    for pol in ("TE", "TM"):
        g1, g2 = [], []
        for x in d:
            m = cs.membrane_mode("Si", x, LAM, pol, with_group_index=False)
            Rf, _ = rf.mirror_fn(m)
            g1.append(cs.evaluate(tab, Rf, cs.dB_per_cm_to_alpha(0.1), m.Gamma).L_eff_gas * 100)
            g2.append(cs.evaluate(tab, Rf, rf.alpha_bg_for(m, "Si"), m.Gamma).L_eff_gas * 100)
        res[pol] = (g1, g2)
    b.semilogy(d * 1e9, res["TM"][0], color="C0", lw=1.4, label="TM$_0$, 0.1 dB/cm")
    b.semilogy(d * 1e9, res["TM"][1], "--", color="C0", lw=1.4, label="TM$_0$, roughness-scaled")
    b.semilogy(d * 1e9, res["TE"][0], color="C3", lw=1.4, label="TE$_0$, 0.1 dB/cm")
    b.axhline(50, color="k", ls=":", lw=1); b.text(155, 56, "milestone M4: 50 cm", fontsize=10)
    b.set(xlabel="Si membrane thickness (nm)", ylabel=r"gas-equivalent path $\Gamma\langle L\rangle$ (cm)", ylim=(0.1, 120), xlim=(150, 600))
    b.legend(loc="lower right", frameon=False, borderaxespad=0.1, fontsize=10); b.set_title("(b) path with gas overlap", loc="left")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig3_device.pdf"); plt.close(fig)
    return d, res, tab


# ---------------------------------------------------------------- Fig 4: suppression budget (1.55 um)
def fig4():
    dnu = np.concatenate([np.linspace(0, 2e9, 1000), np.linspace(2e9, 30e9, 1000)])
    nu0 = C / 1.5317e-6; line = 2 * 0.095 * 29979245800.0; dndT = 1.5e-4
    p1 = S["curved ρ=5 cm"]["paths"][1][0]; p4 = inc("curved ρ=5 cm", 4); cache = {}

    def lim(p, laser, span, dT):
        k = id(p)
        if k not in cache: cache[k] = cm.autocovariance(p, dnu)
        return cm.speckle_noise_A(p, dnu, cache[k], dT, nu0, dndT, laser, line, True, span) / p.L_mean
    steps = [("start: one beam, 1 MHz laser,\n1 mK drift", lim(p1, 1e6, 0, 1e-3)), ("+ 4 incoherent beams", lim(p4, 1e6, 0, 1e-3)),
             ("+ 1 GHz laser linewidth", lim(p4, 1e9, 0, 1e-3)), ("+ dither (10 GHz equiv.)", lim(p4, 1e9, 10e9, 1e-3)),
             ("+ drift 10 µK", lim(p4, 1e9, 10e9, 1e-5)), ("+ drift 1 µK", lim(p4, 1e9, 10e9, 1e-6))]
    v = np.array([s[1] for s in steps]) / steps[0][1]
    fig, a = plt.subplots(figsize=(WIDTH, 2.75)); y = np.arange(len(v))[::-1]
    a.barh(y, v, color=["0.55", "C0", "C2", "C2", "C1", "C1"]); a.set_xscale("log"); a.set_xlim(1e-3, 3)
    for yy, vv in zip(y, v): a.text(vv * 1.15, yy, f"×{vv:.2g}", va="center", fontsize=10)
    a.set_yticks(y); a.set_yticklabels([s[0] for s in steps]); a.set_xlabel("speckle-limited detection limit, relative to start"); a.grid(axis="y", alpha=0)
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig4_budget.pdf"); plt.close(fig)
    return steps, v


# ---------------------------------------------------------------- Fig 5: mid-IR Ge, hole-free mirror
def fig5():
    LM = 5.2629e-6; mA = cs.membrane_mode("Ge", 300e-9, LM, "TE", with_group_index=False)
    DES = json.load(open("data/mirror_designs.json"))["130_16"]; S5 = pickle.load(open("results/chaos_scan.pkl", "rb"))
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.15)); a = ax[0]; sg = np.linspace(0, 0.9999, 2000)
    dbr = cs.DBR(n_tooth=mA.neff, lam_design=LM, N=20, m_gap=1, m_tooth=1, slab_pol="TE", bounce_loss=0)
    a.semilogy(sg, np.maximum(1 - dbr.R(LM, sg), 1e-6), "C3", lw=1.4, label="two-material")
    a.semilogy(sg, np.maximum(1 - md.R_tri(LM, sg, DES["nA"], DES["nC"], DES["g"], DES["c"], DES["a"], DES["N"]), 1e-6), "C0", lw=1.4, label="three-index")
    a.set(xlabel="sin χ at the mirror", ylabel="mirror loss 1 − R", ylim=(1e-6, 2), xlim=(0, 1)); a.legend(loc="lower right", frameon=False); a.set_title("(a) Ge 300 nm TE$_0$, 5.26 µm", loc="left")
    pc = lambda n: cm.contrast(S5[n]["paths"][1][0], 1e6, True)
    ref = pc("regular") / (S5["regular"]["mirrors"]["two-material"]["GL"] * 100)
    fom = {(n, m): pc(n) / (S5[n]["mirrors"][m]["GL"] * 100) / ref for n in ("regular", "curved ρ=5 cm") for m in ("two-material", "tri-index")}
    b = ax[1]; w = 0.36
    for k, (m, col, lab) in enumerate((("two-material", "C3", "two-material"), ("tri-index", "C0", "three-index"))):
        vals = [fom[("regular", m)], fom[("curved ρ=5 cm", m)]]
        b.bar(np.arange(2) + (k - 0.5) * w, vals, w, color=col, label=lab)
        for xx, vv in zip(np.arange(2) + (k - 0.5) * w, vals): b.text(xx, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=10)
    b.axhline(1, color="k", lw=0.8); b.set_xticks([0, 1]); b.set_xticklabels(["regular", "curved facets"]); b.set_ylim(0, 1.5)
    b.set_ylabel("relative detection limit"); b.legend(loc="upper right", frameon=False, ncol=1, fontsize=10); b.grid(axis="x", alpha=0); b.set_title("(b) cell × mirror", loc="left")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig5_midir.pdf"); plt.close(fig)
    return mA, DES, fom, S5


if __name__ == "__main__":
    fig1(); info = fig2(); d, res, tab = fig3(); steps, v = fig4(); mA, DES, fom, S5 = fig5()
    print("FIG2 spectra (label, measured sigma, model):", [(a, round(b, 3), round(c, 3)) for a, b, c in info])
    for n in S:
        p = S[n]["paths"][1][0]
        print(f"{n:15s} M={p.M_spat():.1f} Np={p.n_paths_eff()[0]:.1f} C1={con(p):.3f}" + "".join(f" K{K}inc={con(inc(n,K)):.3f}" + f" K{K}coh={con(cm.merge_coherent(S[n]['paths'][K])):.3f}" for K in (2, 4) if K in S[n]["paths"]))
    p = S["curved ρ=5 cm"]["paths"][1][0]; print("curved K1: T_det=%.3f (%.1f dB) <L>=%.1f cm G<L>=%.1f cm" % (p.T_det, 10*np.log10(p.T_det), p.L_mean*100, p.L_mean*100*mTM.Gamma))
    print("Si TM250 neff=%.2f Gamma=%.2f ng=%.2f" % (mTM.neff, mTM.Gamma, mTM.n_group))
    print("fig3 thickness nm:", np.round(d*1e9).astype(int)); print("TM const:", np.round(res['TM'][0],1)); print("TM rough:", np.round(res['TM'][1],1)); print("TE const:", np.round(res['TE'][0],2))
    print("budget:", [(s[0].split(chr(10))[0], round(float(x), 4)) for s, x in zip(steps, v)])
    print("fom mid-IR:", {k: round(x, 3) for k, x in fom.items()})
    print("mid-IR Ge300 TE: neff %.2f Gamma %.3f zp %.0f nm; design gap %.0f thin %.0f tooth %.0f nm mean loss %.4f" % (mA.neff, mA.Gamma, mA.z_p*1e9, DES['g']*1e9, DES['c']*1e9, DES['a']*1e9, DES['mean_loss']))
