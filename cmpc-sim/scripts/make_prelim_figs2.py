"""v0.3 figures: single-port multi-beam concept (Fig 1), coherence vs laser linewidth (Fig 2), budget with beam dither (Fig 4)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import pickle
import matplotlib.pyplot as plt
import numpy as np
import make_prelim_figs as mp
import cmpc_sim as cs, coherence_model as cm

OUT = mp.OUT; WIDTH = mp.WIDTH; C = mp.C; LAM = mp.LAM
MB = pickle.load(open("results/multibeam_nir.pkl", "rb"))
mTM = mp.mTM; TH0 = mp.TH0; CHI = mp.CHI
L = lambda tag, kind, K: MB[(tag, kind, K)]
inc = lambda pl: cm.merge_incoherent(pl) if len(pl) > 1 else pl[0]
coh = lambda pl: cm.merge_coherent(pl) if len(pl) > 1 else pl[0]
con = lambda p, lw: cm.contrast(p, lw, True)


def fig1():
    fig = plt.figure(figsize=(WIDTH, 4.3)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95])
    cols = ["C0", "C3", "C2", "C1"]
    for j, (name, ttl, nb) in enumerate((("regular", "(a) regular cell", 1), ("curved ρ=5 cm", "(b) chaotic cell", 1), ("curved ρ=5 cm", "(c) one port, K beams", 4))):
        cell = mp.mkcell(name); a = fig.add_subplot(gs[0, j])
        x, y = cell.outline(); a.plot(x * 1e3, y * 1e3, "k", lw=1.0)
        x0, y0, nx, ny = cell.point_at(np.array([cell.h]))
        offs = [(k - 1.5) * 2.2 * TH0 for k in range(4)] if nb > 1 else [0.0, 0.004]
        for k, off in enumerate(offs):
            th = np.arctan2(-ny[0], -nx[0]) + CHI + off
            xs, ys = cs.trace_path(cell, x0[0], y0[0], np.cos(th), np.sin(th), n_hits=40 if nb == 1 else 28)
            a.plot(xs * 1e3, ys * 1e3, lw=0.6, color=(cols[k] if nb > 1 else ["C0", "C3"][k]))
        a.plot(x0 * 1e3, y0 * 1e3, "o", color="tab:green", ms=6)
        xo, yo, _, _ = cell.point_at(np.array([cell.h + 18 * cell.h])); a.plot(xo * 1e3, yo * 1e3, "o", color="tab:red", ms=6)
        a.set_aspect("equal"); a.axis("off"); a.set_title(ttl, loc="left")
    for j, name in enumerate(("regular", "curved ρ=5 cm")):
        cell = mp.mkcell(name); b = fig.add_subplot(gs[1, j])
        Sg, SC = cs.poincare(cell, 40, 300, seed=j)
        b.plot(Sg.ravel() / cell.perimeter, SC.ravel(), ".", ms=0.9, color="k", alpha=0.6, rasterized=True)
        b.set(xlim=(0, 1), ylim=(-1, 1), xlabel="position s/perimeter", ylabel="sin χ", xticks=[0, 1], yticks=[-1, 0, 1])
        b.set_title("(d) Poincaré, regular" if j == 0 else "(e) Poincaré, chaotic", loc="left", fontsize=10)
    f = fig.add_subplot(gs[1, 2]); dm = LAM / (mTM.neff * mp.W)
    for i in range(-12, 13): f.axhline(0.55 + i * dm, color="0.85", lw=0.5)
    for k in range(4):
        f.axhspan(0.55 + (k - 1.5) * 2.2 * dm - dm / 2, 0.55 + (k - 1.5) * 2.2 * dm + dm / 2, color=cols[k], alpha=0.9)
    f.annotate("", xy=(0.5, 0.55 + 0.5 * 2.2 * dm + 3.5 * dm), xytext=(0.5, 0.55 + 0.5 * 2.2 * dm + 0.8 * dm), arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
    f.text(0.56, 0.55 + 0.5 * 2.2 * dm + 3.0 * dm, "dither", fontsize=10)
    f.set(xlim=(0, 1), ylim=(0.55 - 8 * dm, 0.55 + 8 * dm), xlabel="position in port", ylabel="sin χ", xticks=[0, 1], yticks=[]); f.grid(False)
    f.set_title("(f) port phase space", loc="left", fontsize=10)
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig1_concept.pdf"); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.35), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]; dn = np.linspace(-1.5e9, 1.5e9, 3001); nu0 = C / 1.5317e-6
    cell = mp.mkcell("regular"); P = cell.perimeter
    tab = cs.trace_rays(cell, mp.W, 5000, 1000, TH0, seed=7, max_path=5.0, keep_start=False, theta_c=CHI, s_in=cell.h, s_out=cell.h + 18 * cell.h)
    preg = cm.prepare(tab, mp.R_flat, cs.dB_per_cm_to_alpha(0.04), mTM.Gamma, LAM, mTM.neff, mTM.n_group, mp.W); del tab
    global preg_g; preg_g = preg
    cases = ((preg, "regular, 1 beam", "C3"), (L("long", "ports", 1)[0], "chaotic, 1 beam", "C0"), (inc(L("long", "angles", 8)), "chaotic, 8 beams", "C2"))
    info = []
    for k, (p, lab, col) in enumerate(cases):
        I = cm.simulate_spectrum(p, dn, nu0, None, "all", seed=3); off = (2 - k) * 2.3
        a.plot(dn / 1e9, I - 1 + off, lw=0.7, color=col); a.axhline(off, color="0.6", lw=0.5)
        a.text(-1.45, off + 1.05, f"{lab}: σ = {np.std(I)/np.mean(I):.2f}", fontsize=10, va="bottom"); info.append((lab, np.std(I) / np.mean(I)))
    a.set(xlabel="detuning (GHz)", yticks=[], ylabel="intensity, no gas (offset)", xlim=(-1.5, 1.5), ylim=(-1.4, 6.9)); a.grid(False)
    a.set_title("(a) simulated spectral speckle", loc="left")
    b = ax[1]; gl = np.geomspace(1e5, 3e9, 20)
    b.loglog(gl / 1e6, [con(L("long", "ports", 1)[0], g) for g in gl], "k", lw=1.5, label="1 beam")
    b.loglog(gl / 1e6, [con(coh(L("long", "angles", 8)), g) for g in gl], color="C0", lw=1.5, label="8 beams, one laser")
    b.loglog(gl / 1e6, [con(coh(L("short", "angles", 8)), g) for g in gl], "--", color="C0", lw=1.5, label="same, short path")
    b.loglog(gl / 1e6, [con(inc(L("long", "angles", 8)), g) for g in gl], color="C3", lw=1.5, label="8 incoherent / dithered")
    b.set(xlabel="laser linewidth (MHz)", ylabel="speckle contrast", ylim=(3e-3, 0.4)); b.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.1)
    b.set_title("(b) beams from one laser", loc="left")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig2_speckle.pdf"); plt.close(fig)
    return info


def fig4():
    dnu = np.concatenate([np.linspace(0, 2e9, 1000), np.linspace(2e9, 30e9, 1000)])
    nu0 = C / 1.5317e-6; line = 2 * 0.095 * 29979245800.0; dndT = 1.5e-4; ang = L("long", "angles", 16); k1 = L("long", "ports", 1)[0]; cache = {}

    def lim(p, laser, dT):
        key = id(p)
        if key not in cache: cache[key] = cm.autocovariance(p, dnu)
        return cm.speckle_noise_A(p, dnu, cache[key], dT, nu0, dndT, laser, line, True, 0.0) / p.L_mean
    d4 = inc(ang[:4]); d16 = inc(ang)
    steps = [("start: 1 beam, 1 MHz laser,\n1 mK drift", lim(k1, 1e6, 1e-3)), ("+ 1 GHz laser linewidth", lim(k1, 1e9, 1e-3)),
             ("+ beam dither, 4 input modes", lim(d4, 1e9, 1e-3)), ("+ beam dither, 16 input modes", lim(d16, 1e9, 1e-3)),
             ("+ drift 10 µK", lim(d16, 1e9, 1e-5)), ("+ drift 1 µK", lim(d16, 1e9, 1e-6))]
    v = np.array([s[1] for s in steps]) / steps[0][1]
    fig, a = plt.subplots(figsize=(WIDTH, 2.75)); y = np.arange(len(v))[::-1]
    a.barh(y, v, color=["0.55", "C2", "C0", "C0", "C1", "C1"]); a.set_xscale("log"); a.set_xlim(2e-4, 3)
    for yy, vv in zip(y, v): a.text(vv * 1.15, yy, f"×{vv:.2g}", va="center", fontsize=10)
    a.set_yticks(y); a.set_yticklabels([s[0] for s in steps]); a.set_xlabel("speckle-limited detection limit, relative to start"); a.grid(axis="y", alpha=0)
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig4_budget.pdf"); plt.close(fig)
    return steps, v


if __name__ == "__main__":
    fig1(); info = fig2(); steps, v = fig4()
    print("regular K=1 model contrast at 1 MHz (alpha 0.04):", round(con(preg_g, 1e6), 3)); print("spectra sigma:", [(a, round(b, 3)) for a, b in info])
    for tag in ("long", "short"):
        k1 = L(tag, "angles", 1)[0] if tag == "long" else L("short", "angles", 1)[0]
        print(tag, "K=1 contrast at 1MHz/100MHz/1GHz:", [round(con(k1, g), 3) for g in (1e6, 1e8, 1e9)], " <L>=%.0f cm" % (k1.L_mean * 100))
        for K in (8,):
            pl = L(tag, "angles", K)
            print(tag, f"K={K} one-laser coherent:", [round(con(coh(pl), g), 3) for g in (1e6, 1e7, 1e8, 3e8, 1e9, 3e9)], " incoherent:", [round(con(inc(pl), g), 3) for g in (1e6, 1e9)])
    print("long incoherent K=2,4,8,16 at 1MHz:", [round(con(inc(L("long", "angles", K)), 1e6), 3) for K in (2, 4, 8, 16)], " ports K=2,4,8:", [round(con(inc(L("long", "ports", K)), 1e6), 3) for K in (2, 4, 8)])
    print("<L> rel ports K=2,4,8:", [round(np.mean([p.L_mean for p in L("long", "ports", K)]) / L("long", "ports", 1)[0].L_mean, 3) for K in (2, 4, 8)], " angles K=2,4,8,16:", [round(np.mean([p.L_mean for p in L("long", "angles", K)]) / L("long", "ports", 1)[0].L_mean, 3) for K in (2, 4, 8, 16)])
    print("budget:", [(s[0].split(chr(10))[0], round(float(x), 4)) for s, x in zip(steps, v)])
    print("K=1 long: T=%.3f (%.1f dB) <L>=%.1f cm G<L>=%.1f cm" % (L("long", "ports", 1)[0].T_det, 10 * np.log10(L("long", "ports", 1)[0].T_det), L("long", "ports", 1)[0].L_mean * 100, L("long", "ports", 1)[0].L_mean * 100 * mTM.Gamma))
    print("mode width dsin=%.4f ; cells in [-1,1]=%.0f" % (LAM / (mTM.neff * mp.W), 2 / (LAM / (mTM.neff * mp.W))))

    import dataclasses
    pl = L("long", "angles", 8)
    print("delay test K=8 one laser, 1 GHz: D(m) -> contrast; incoherent limit", round(con(inc(pl), 1e9), 4))
    for D in (0, 0.1, 0.3, 1.0, 2.0, 4.0):
        sh = [dataclasses.replace(p, cells=[(P, Lc + k * D) for P, Lc in p.cells]) for k, p in enumerate(pl)]
        print("  D = %.1f m (total span %.1f m): %.4f" % (D, 7 * D, con(cm.merge_coherent(sh), 1e9)))
    sp = np.concatenate([p.L_all for p in pl]); print("path-length spread of detected light: median %.0f cm, 90%% < %.0f cm" % (np.median(sp) * 100, np.percentile(sp, 90) * 100))
