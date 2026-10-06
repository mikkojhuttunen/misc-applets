"""Mid-IR (NO fundamental, ~5.26 um) simulations for free-standing Ge (and Si) membranes.

    python run_midir.py          # all figures + printed summary
    python run_midir.py 1 4      # selected

Reuses cmpc_sim / coherence_model / gas_spectra. ASSUMED inputs are in CFG and in ASSUMPTIONS of cmpc_sim.py.
Gas line parameters are ILLUSTRATIVE unless hitran_lines_mir.json exists (python fetch_hitran.py mir).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cmpc_sim as cs
import coherence_model as cm
import gas_spectra as gs

NU_NO = 1900.08                      # cm^-1 (NO line used as the design wavelength)
LAM = 1.0 / (NU_NO * 100.0)          # m  (~5.263 um)
CFG = dict(R_cell=5e-3, n_facets=24, port_w=100e-6, alpha_ref_dB_cm=0.1, bounce_loss=3e-4, dbr_N=20,
           mat="Ge", d=300e-9, pol="TE", curv_radius=0.05, n_rays=12000, n_bounce=2500, max_path=8.0, seed=5,
           dneff_dT=3e-4, laser_fwhm=1e6, line_hwhm_cm=0.058)
OUT = Path("results/figs_midir")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {("Ge", "TE"): "C3", ("Ge", "TM"): "C1", ("Si", "TE"): "C0", ("Si", "TM"): "C2"}
_REF = cs.membrane_mode("Si", 220e-9, 1.55e-6, "TE", with_group_index=False)


def mode(mat=None, d=None, pol=None, gi=True):
    return cs.membrane_mode(mat or CFG["mat"], d or CFG["d"], LAM, pol or CFG["pol"], with_group_index=gi)


def alpha_bg(m):
    """Background loss [1/m]: alpha_ref (Si 220 nm TE, 1.55 um) x surface-field proxy x (index contrast)^2 x (k0/k0_ref)^2.
    ASSUMES the same rms roughness as the 1.55 um reference. Crude."""
    a_ref = cs.dB_per_cm_to_alpha(CFG["alpha_ref_dB_cm"])
    return a_ref * (m.E_edge2 / _REF.E_edge2) * ((m.n_core**2 - 1) / (_REF.n_core**2 - 1)) ** 2 * (1.55e-6 / m.wavelength) ** 2


def mirror(m, N=None):
    dbr = cs.DBR(n_tooth=m.neff, lam_design=LAM, N=N or CFG["dbr_N"], m_gap=1, m_tooth=1, bounce_loss=CFG["bounce_loss"], slab_pol=m.pol)
    return (lambda s: dbr.R(LAM, s)), dbr


def cell(kind):
    R, N = CFG["R_cell"], CFG["n_facets"]
    return cs.SegmentedCell(R, N, curvature=1 / CFG["curv_radius"]) if kind == "chaotic" else cs.SegmentedCell(R, N)


def table(kind, m, n_rays=None, star_m=None):
    """chaotic: curved facets, star {24/7} launch. regular: flat facets, launch at 45 deg (star_m = 6): all incidence
    angles are multiples of 15 deg, so sin chi in {0, .26, .5, .71, .87, .97}: none inside a TE Brewster hole near 0.34-0.42."""
    sm = star_m or (7 if kind == "chaotic" else 6)
    chi = np.pi / 2 - np.pi * sm / CFG["n_facets"]
    return cs.trace_rays(cell(kind), CFG["port_w"], n_rays or CFG["n_rays"], CFG["n_bounce"], LAM / (m.neff * CFG["port_w"]),
                         seed=CFG["seed"], max_path=CFG["max_path"], keep_start=False, theta_c=chi)


_ST = {}


def state(kind):
    if kind not in _ST:
        m = mode()
        Rf, _ = mirror(m)
        tab = table(kind, m)
        _ST[kind] = (m, cm.prepare(tab, Rf, alpha_bg(m), m.Gamma, LAM, m.neff, m.n_group, CFG["port_w"]), tab, Rf)
    return _ST[kind]


def fig1():
    d = np.geomspace(60e-9, 1.5e-6, 70)
    A = np.pi * (CFG["R_cell"]) ** 2
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for (mat, pol), c in COL.items():
        ms = [cs.membrane_mode(mat, x, LAM, pol, with_group_index=False) for x in d]
        ls = "-" if pol == "TE" else "--"
        ax[0].loglog(d * 1e9, [m.Gamma for m in ms], ls, color=c, label=f"{mat} {pol}0")
        ax[1].plot(d * 1e9, [m.neff for m in ms], ls, color=c)
        ax[2].loglog(d * 1e9, [m.z_p * 1e9 for m in ms], ls, color=c)
        ax[3].loglog(d * 1e9, [cs.evanescent_volume(m, A)["V_ev_mm3"] for m in ms], ls, color=c)
    for a in ax:
        a.axvline(300, color="k", ls=":", lw=0.8); a.set_xlabel("membrane thickness (nm)")
    ax[0].set(ylabel="Γ", title=f"Γ at {LAM*1e6:.2f} µm", ylim=(2e-3, 2)); ax[0].legend(fontsize=8)
    ax[1].set(ylabel="n_eff", title="Effective index (dotted: 300 nm)", ylim=(1, 3.7))
    ax[2].set(ylabel="z_p per side (nm)", title="1/e intensity penetration depth")
    ax[3].set(ylabel="V_ev (mm³)", title="Evanescent gas volume, 1 cm disc, both faces")
    fig.suptitle("Free-standing Ge and Si membranes, 5.26 µm. TM at 300 nm is at cut-off (n_eff ≈ 1.02, field extends ~2 µm): not a confined mode.", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "m1_modes_gamma_volume.png")


def fig2():
    sg = np.linspace(0, 0.9999, 1500)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for (mat, pol, d0), c in ((("Ge", "TE", 300e-9), "C3"), (("Ge", "TM", 600e-9), "C1"), (("Ge", "TE", 600e-9), "C4")):
        m = cs.membrane_mode(mat, d0, LAM, pol, with_group_index=False)
        dbr = cs.DBR(n_tooth=m.neff, lam_design=LAM, N=20, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
        ax[0].semilogy(sg, np.maximum(1 - dbr.R(LAM, sg), 1e-9), color=c, label=f"{mat} {d0*1e9:.0f} nm {pol}0, n_eff {m.neff:.2f}")
        ax[0].axvline(1 / m.neff, color=c, ls=":", lw=0.8)
    for k, a in enumerate(np.sin(np.radians([0, 15, 30, 45, 60, 75]))):
        ax[0].axvline(a, color="0.6", lw=0.6, ls="-", alpha=0.6)
    ax[0].text(0.01, 0.04, "grey: sinχ of the hole-avoiding regular cell", transform=ax[0].transAxes, fontsize=7)
    ax[0].set(xlabel="sin χ at the mirror", ylabel="1 − R (N = 20)", title="Mirror loss vs angle (dotted: 1/n_eff)", ylim=(1e-9, 2)); ax[0].legend(fontsize=7, loc="upper right")
    d = np.geomspace(150e-9, 1.2e-6, 30)
    for (mat, pol), c in COL.items():
        y = []
        for x in d:
            m = cs.membrane_mode(mat, x, LAM, pol, with_group_index=False)
            dbr = cs.DBR(n_tooth=m.neff, lam_design=LAM, N=20, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
            y.append(1 - cs.uniform_average(lambda s: dbr.R(LAM, s)))
        ax[1].semilogy(d * 1e9, y, "-" if pol == "TE" else "--", color=c, label=f"{mat} {pol}0")
    ax[1].set(xlabel="membrane thickness (nm)", ylabel="⟨1 − R⟩ over uniform sinχ", title="Angle-averaged mirror loss (chaotic cell)", ylim=(1e-4, 2)); ax[1].legend(fontsize=8)
    m = mode(); dbr = mirror(m)[1]
    ax[2].axis("off")
    ax[2].text(0, 1, f"DBR design, Ge {CFG['d']*1e9:.0f} nm TE0 at {LAM*1e6:.3f} µm\n\n"
                      f"n_eff = {m.neff:.3f}, n_g = {m.n_group:.2f}\n"
                      f"air gap (first order, λ/4):   {dbr.d_gap*1e9:.0f} nm\n"
                      f"membrane tooth (λ/4n_eff):    {dbr.d_tooth*1e9:.0f} nm\n"
                      f"period: {(dbr.d_gap+dbr.d_tooth)*1e6:.2f} µm; N = {dbr.N} → {dbr.N*(dbr.d_gap+dbr.d_tooth)*1e6:.0f} µm deep\n\n"
                      "(1.55 µm Si: gap 390 nm, tooth 140 nm.\nMid-IR features are 3-4× larger: easier lithography.)",
               va="top", fontsize=9, family="monospace")
    fig.tight_layout(); fig.savefig(OUT / "m2_mirrors.png")


def fig3():
    d = np.geomspace(150e-9, 1.2e-6, 18)
    tab = table("chaotic", mode())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for (mat, pol), c in COL.items():
        R = []
        for x in d:
            m = cs.membrane_mode(mat, x, LAM, pol, with_group_index=False)
            Rf, _ = mirror(m, 20)
            r1 = cs.evaluate(tab, Rf, cs.dB_per_cm_to_alpha(CFG["alpha_ref_dB_cm"]), m.Gamma)
            r2 = cs.evaluate(tab, Rf, alpha_bg(m), m.Gamma)
            R.append([r1.L_eff_gas * 100, r2.L_eff_gas * 100, r2.S1 * 100, m.Gamma])
        R = np.array(R); ls = "-" if pol == "TE" else "--"
        ax[0].semilogy(d * 1e9, R[:, 0], ls, color=c, label=f"{mat} {pol}0")
        ax[1].semilogy(d * 1e9, R[:, 1], ls, color=c)
        ax[2].semilogy(d * 1e9, R[:, 2], ls, color=c)
    ax[0].set(ylabel="Γ⟨L⟩ (cm)", title="Gas-equivalent path, 0.1 dB/cm", ylim=(0.02, 100)); ax[0].legend(fontsize=8)
    ax[1].set(ylabel="Γ⟨L⟩ (cm)", title="roughness-scaled loss (k0², surface field, index contrast)", ylim=(0.02, 100))
    ax[2].set(ylabel="S1 = Γ T ⟨L⟩ (cm)", title="Signal yield (roughness-scaled loss)", ylim=(1e-5, 10))
    for a in ax:
        a.set_xscale("log"); a.axvline(300, color="k", ls=":", lw=0.8); a.set_xlabel("membrane thickness (nm)")
    ax[0].axhline(50, color="k", lw=0.7, ls=":")
    fig.suptitle(f"Curved-facet segmented cell, D = 1 cm, {CFG['port_w']*1e6:.0f} µm ports, DBR N = 20, λ = {LAM*1e6:.2f} µm", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "m3_thickness_tradeoff.png")


def fig4():
    dnu = np.concatenate([np.linspace(0, 2e9, 1500), np.linspace(2e9, 30e9, 1400)])
    gl = np.geomspace(1e5, 1e10, 24)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for kind, c, lab in (("chaotic", "C0", "curved facets (chaotic)"), ("regular", "C3", "regular, hole-avoiding launch")):
        m, p, tab, Rf = state(kind)
        g = cm.autocovariance(p, dnu)
        ax[0].semilogx(np.maximum(dnu, 1e5) / 1e6, g, color=c, label=f"{lab}: paths/mode {p.n_paths_eff()[0]:.1f}, M = {p.M_spat():.0f}")
        ax[1].loglog(gl / 1e6, [cm.contrast(p, x, True) for x in gl], color=c, label=lab)
        ax[1].loglog(gl / 1e6, [cm.contrast(p, x, False) for x in gl], "--", color=c)
    ax[0].set(xlabel="frequency offset (MHz)", ylabel="speckle autocovariance g", title="Spectral correlation"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="laser linewidth FWHM (MHz)", ylabel="speckle contrast", title="Solid: sum of modes, dashed: single mode", ylim=(1e-3, 1.2)); ax[1].legend(fontsize=7)
    names, vals = [], []
    for kind in ("chaotic", "regular"):
        m, p, tab, Rf = state(kind)
        names.append(kind); vals.append((p.T_det, p.L_mean * 100, m.Gamma * p.L_mean * 100))
    x = np.arange(2); w = 0.27
    ax[2].bar(x - w, [v[1] for v in vals], w, label="⟨L⟩ (cm)", color="C2")
    ax[2].bar(x, [v[2] for v in vals], w, label="Γ⟨L⟩ (cm)", color="C4")
    ax[2].bar(x + w, [100 * v[0] for v in vals], w, label="T_det (%)", color="0.6")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["curved facets\n(chaotic)", "regular\n(hole-avoiding)"]); ax[2].legend(fontsize=8)
    ax[2].set_title("Throughput and path, Ge 300 nm TE0")
    fig.suptitle("TE Brewster hole: the chaotic cell (all angles) loses ~10 % per bounce. A regular cell launched at 45° only uses sinχ outside the hole: 4× the signal, but fewer output modes (M = 6 vs 16), so higher contrast at narrow laser linewidth.", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "m4_chaotic_vs_regular_TE.png")
    return vals


def fig5():
    lines, src = gs.load_lines()
    m, p, tab, Rf = state("chaotic")
    Lgas = m.Gamma * p.L_mean * 100
    nu = np.linspace(1880, 2240, 6000)
    al, parts = gs.alpha_mixture(nu, gs.MIDIR_BREATH, lines)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    col = {"H2O": "C0", "NO": "C3", "CO": "C1", "N2O": "C2"}
    for sp, a in parts.items():
        ax[0].semilogy(nu, np.maximum(a * Lgas, 1e-12), color=col[sp], label=f"{sp} {gs.MIDIR_BREATH[sp]*1e6:g} ppm" if gs.MIDIR_BREATH[sp] >= 1e-6 else f"{sp} {gs.MIDIR_BREATH[sp]*1e9:g} ppb")
    ax[0].set(xlabel="wavenumber (cm⁻¹)", ylabel=f"absorbance Γ⟨L⟩ α  (Γ⟨L⟩ = {Lgas:.1f} cm)", ylim=(1e-9, 1), title="Breath-like mixture near 5 µm, 1 atm"); ax[0].legend(fontsize=8)
    nz = np.linspace(NU_NO - 0.6, NU_NO + 0.6, 1500)
    base = {k: v for k, v in gs.MIDIR_BREATH.items() if k != "NO"}
    al0, _ = gs.alpha_mixture(nz, base, lines)
    ax[1].plot(nz, 1 - gs.through_cell(al0, p.L_all, p.W_all, m.Gamma), "k--", lw=0.8, label="no NO")
    for c, cc in ((5e-9, "C0"), (25e-9, "C1"), (50e-9, "C3")):
        mix = dict(base); mix["NO"] = c
        al1, _ = gs.alpha_mixture(nz, mix, lines)
        ax[1].plot(nz, 1 - gs.through_cell(al1, p.L_all, p.W_all, m.Gamma), color=cc, label=f"NO {c*1e9:g} ppb")
    ax[1].set(xlabel="wavenumber (cm⁻¹)", ylabel="absorbed fraction 1 − T", title="NO line (FeNO range), p(L)-weighted"); ax[1].legend(fontsize=8)
    import run_figures as rf
    m15, p15 = rf.state("curved")
    a_nh3, _ = gs.peak_alpha_per_ppm("NH3")
    a_no, _ = gs.peak_alpha_per_ppm("NO")
    A_nh3 = a_nh3 * 1e-3 * m15.Gamma * p15.L_mean * 100
    A_no = a_no * 1e-3 * Lgas
    ax[2].bar([0, 1], [A_nh3, A_no], color=["C0", "C3"])
    ax[2].set_yscale("log"); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels([f"NH₃ 1.53 µm\nSi TM 250 nm\nΓ⟨L⟩={m15.Gamma*p15.L_mean*100:.1f} cm", f"NO 5.26 µm\nGe TE 300 nm\nΓ⟨L⟩={Lgas:.1f} cm"])
    ax[2].set(ylabel="peak absorbance per ppb", title="Per-ppb signal (illustrative line strengths)")
    for i, v in enumerate((A_nh3, A_no)):
        ax[2].text(i, v * 1.2, f"{v:.1e}", ha="center", fontsize=9)
    fig.suptitle(f"Line data: {src}. Strengths are order-of-magnitude placeholders; absorbance ratio is the point, not the absolute value.", fontsize=9, color="C3")
    fig.tight_layout(); fig.savefig(OUT / "m5_midIR_gas_spectra.png")
    return A_nh3, A_no, Lgas


def fig6():
    lines, src = gs.load_lines()
    a1, _ = gs.peak_alpha_per_ppm("NO")
    nu0 = NU_NO * gs.C_CM
    line_fwhm = 2 * CFG["line_hwhm_cm"] * gs.C_CM
    dnu = np.concatenate([np.linspace(0, 2e9, 1500), np.linspace(2e9, 30e9, 1400)])
    dT = np.geomspace(1e-5, 1e-1, 30)
    spans = (0.0, 0.1e9, 1e9, 10e9)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    out = {}
    for kind, ls in (("chaotic", "-"), ("regular", "--")):
        m, p, tab, Rf = state(kind)
        g = cm.autocovariance(p, dnu)
        A1 = a1 * 1e-3 * m.Gamma * p.L_mean * 100
        for span, c in zip(spans, ("C3", "C1", "C2", "C0")):
            tK = span * p.n_g / (nu0 * CFG["dneff_dT"])
            sig = np.array([cm.speckle_noise_A(p, dnu, g, x, nu0, CFG["dneff_dT"], CFG["laser_fwhm"], line_fwhm, True, span) for x in dT])
            ax[0].loglog(dT * 1e3, sig / A1, ls, color=c, label=(f"{kind}, dither ≙ {span/1e9:g} GHz = {tK:.2g} K" if kind == "chaotic" else None))
            out[(kind, span)] = sig / A1
        spn = np.geomspace(1e6, 1e11, 40)
        ax[1].loglog(spn / 1e9, [cm.contrast(p, CFG["laser_fwhm"], True, s) for s in spn], ls, color="C0" if kind == "chaotic" else "C3", label=kind)
        out[(kind, "shot")] = gs.shot_noise_A(1e-3, p.T_det, 1.0) / A1
        out[(kind, "A1")] = A1
    shot = out[("chaotic", "shot")]
    ax[0].axhspan(5, 50, color="g", alpha=0.12)
    ax[0].axhline(shot, color="k", ls="-.", lw=1); ax[0].text(0.02, 0.05, f"shot-noise floor (1 mW in, 1 Hz): {shot:.2g} ppb", transform=ax[0].transAxes, fontsize=8)
    ax[0].text(0.55, 0.92, "green band: FeNO 5–50 ppb", transform=ax[0].transAxes, fontsize=8, color="g")
    ax[0].set(xlabel="temperature drift between reference and measurement (mK)", ylabel="speckle-limited NO detection limit (ppb)", title="Curved-facet cell (dashed lines: regular cell, same dithers)")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].set(xlabel="phase dither expressed as equivalent frequency span (GHz)", ylabel="speckle contrast (sum of modes)", title="Averaging by phase dither (1 MHz laser)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"NO near 5.26 µm, Ge 300 nm TE0, 1 cm cell, 1 atm. Line data: {src}. dn_eff/dT = {CFG['dneff_dT']:.0e} /K assumed.", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "m6_speckle_limited_NO_detection.png")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    res = {}
    m = mode()
    print(f"lambda = {LAM*1e6:.4f} um;  Ge 300 nm TE0: n_eff={m.neff:.3f} n_g={m.n_group:.2f} Gamma={m.Gamma:.3f} z_p={m.z_p*1e9:.0f} nm; V_ev(1 cm disc)={cs.evanescent_volume(m, np.pi*CFG['R_cell']**2)['V_ev_mm3']:.4f} mm3")
    print(f"alpha_bg (roughness-scaled) = {alpha_bg(m)/(np.log(10)/10*100):.3f} dB/cm")
    for w in which:
        print("figure", w, flush=True)
        res[w] = globals()[f"fig{w}"]()
    for k in ("chaotic", "regular"):
        if k in _ST:
            mm, p, tab, Rf = _ST[k]
            print(f"{k:8s}: T_det={p.T_det:.3f} <L>={p.L_mean*100:.0f} cm  Gamma<L>={mm.Gamma*p.L_mean*100:.1f} cm  <1-R>={1-cs.uniform_average(Rf):.4f}  paths/mode={p.n_paths_eff()[0]:.1f}  M_spat={p.M_spat():.0f}  contrast(sum,1MHz)={cm.contrast(p,1e6,True):.3f}")
    if "6" in res and res["6"]:
        o = res["6"]
        for k in ("chaotic", "regular"):
            print(k, "A per ppb = %.2e; shot-limited %.2g ppb" % (o[(k, 'A1')], o[(k, 'shot')]))
            for span in (0.0, 0.1e9, 1e9, 10e9):
                print(f"   dither {span/1e9:g} GHz-eq: MDC at dT=0.1 mK: {np.interp(0.1e-3, np.geomspace(1e-5,1e-1,30), o[(k,span)]):.3g} ppb ; at 10 mK: {np.interp(1e-2, np.geomspace(1e-5,1e-1,30), o[(k,span)]):.3g} ppb")
    print("done ->", OUT)
