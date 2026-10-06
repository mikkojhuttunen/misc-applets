"""CMPC simulator figures, v0.2 (corrected polarisation mapping, segmented cells, gas spectra, coherence model).

    python run_figures.py            # all
    python run_figures.py 3 4        # selected

Needs numpy, scipy, matplotlib and a clone of mikkojhuttunen/misc-applets (env MISC_APPLETS if not next to this file).
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

# ------------------------------------------------------------------ CONFIG (all assumptions that move the answer are here)
CFG = dict(
    lam=1.55e-6, R_cell=5e-3, n_facets=24, star_m=7,           # 1 cm wide cell, star polygon {24/7}
    port_w=30e-6,                                              # port width; launch half-angle = lambda/(n_eff w)
    alpha_ref_dB_cm=0.1,                                       # ASSUMED background loss of Si 220 nm TE membrane (scaled for others)
    bounce_loss=3e-4,                                          # ASSUMED extra per-bounce loss (slot radiation, roughness)
    dbr_N=10, material="Si", d=250e-9, pol="TM",               # reference design: Si 250 nm TM0
    curv_radius=0.05,                                          # facet curvature radius of the perturbed cell [m]
    tilt_deg=1.0,
    n_rays=12000, n_bounce=2500, max_path=8.0, seed=5,
    laser_fwhm=1e6, dneff_dT=1.5e-4, nu_NH3=6528.76, nu_CH4=6057.10,
)
OUT = Path("results/figs")
OUT.mkdir(parents=True, exist_ok=True)
COL = {"Si": "#1f4e79", "SiNx": "#c0504d", "Al2O3": "#4f9d4f"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})
LAM = CFG["lam"]


def ref_mode(**kw):
    p = dict(material=CFG["material"], thickness=CFG["d"], wavelength=LAM, pol=CFG["pol"])
    p.update(kw)
    return cs.membrane_mode(**p)


def mirror_fn(mode, N=None, mat="Si"):
    N = (CFG["dbr_N"] if mat == "Si" else 25) if N is None else N
    dbr = cs.DBR(n_tooth=mode.neff, lam_design=LAM, N=N, m_gap=1, m_tooth=1, bounce_loss=CFG["bounce_loss"], slab_pol=mode.pol)
    return (lambda s: dbr.R(LAM, s)), dbr


def make_cell(kind):
    R, N = CFG["R_cell"], CFG["n_facets"]
    if kind == "regular":
        return cs.SegmentedCell(R, N)
    if kind == "tilt":
        return cs.SegmentedCell(R, N, tilt_rms=np.radians(CFG["tilt_deg"]), seed=1)
    if kind == "curved":
        return cs.SegmentedCell(R, N, curvature=1 / CFG["curv_radius"])
    raise KeyError(kind)


def make_table(kind, mode, port_w=None, n_rays=None, keep_start=False):
    w = CFG["port_w"] if port_w is None else port_w
    th0 = LAM / (mode.neff * w)
    chi = np.pi / 2 - np.pi * CFG["star_m"] / CFG["n_facets"]
    return cs.trace_rays(make_cell(kind), w, CFG["n_rays"] if n_rays is None else n_rays, CFG["n_bounce"], th0,
                         seed=CFG["seed"], max_path=CFG["max_path"], keep_start=keep_start, theta_c=chi)


def alpha_bg_for(mode, mat, ref=None):
    """Background loss [1/m]: roughness-scaled from alpha_ref at Si 220 nm TE (surface field weight x index-contrast^2)."""
    ref = ref or cs.membrane_mode("Si", 220e-9, LAM, "TE", with_group_index=False)
    a_ref = cs.dB_per_cm_to_alpha(CFG["alpha_ref_dB_cm"])
    n_si = ref.n_core
    return a_ref * (mode.E_edge2 / ref.E_edge2) * ((mode.n_core**2 - 1) / (n_si**2 - 1)) ** 2


# ------------------------------------------------------------------ F1
def fig1():
    d = np.geomspace(50e-9, 1.2e-6, 70)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for mat in ("Si", "SiNx", "Al2O3"):
        for pol, ls in (("TE", "-"), ("TM", "--")):
            ms = [cs.membrane_mode(mat, x, LAM, pol, with_group_index=False) for x in d]
            ax[0].loglog(d * 1e9, [m.Gamma for m in ms], ls, color=COL[mat], label=f"{mat} {pol}0")
            ax[1].plot(d * 1e9, [m.neff for m in ms], ls, color=COL[mat], label=f"{mat} {pol}0")
            ax[2].loglog(d * 1e9, [m.z_p * 1e9 for m in ms], ls, color=COL[mat], label=f"{mat} {pol}0")
    ax[0].set(xlabel="membrane thickness (nm)", ylabel="Γ  (α_modal = Γ α_gas)", title="Evanescent absorption factor Γ (1550 nm)", ylim=(1e-3, 2))
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(xlabel="membrane thickness (nm)", ylabel="n_eff", title="Effective index (mirror contrast vs air; TIR for sinχ > 1/n_eff)", ylim=(1, 3.6))
    ax[2].set(xlabel="membrane thickness (nm)", ylabel="z_p = 1/(2γ) per side (nm)", title="1/e intensity penetration depth into the gas")
    fig.suptitle("Free-standing membranes in air. TE = E in the membrane plane (p-pol on a vertical trench mirror); TM = E_z dominant (s-pol). "
                 "TM Γ > 1 only where the mode is close to cut-off.", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_gamma_neff_penetration.png")


# ------------------------------------------------------------------ F2
def fig2():
    A = np.pi * (1e-2 / 2) ** 2
    d = np.geomspace(60e-9, 1.2e-6, 50)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mat in ("Si", "SiNx", "Al2O3"):
        for pol, ls in (("TE", "-"), ("TM", "--")):
            ms = [cs.membrane_mode(mat, x, LAM, pol, with_group_index=False) for x in d]
            ax[0].loglog(d * 1e9, [cs.evanescent_volume(m, A)["V_ev_mm3"] for m in ms], ls, color=COL[mat], label=f"{mat} {pol}0")
    ax[0].set(xlabel="membrane thickness (nm)", ylabel="V_ev = 2 A z_p  (mm³)", title="Gas volume within the 1/e depth, 1 cm disc, both faces")
    ax[0].text(0.03, 0.04, "1 mm³ of air ≈ 2.5×10¹⁶ molecules;  1 ppb → 2.5×10⁷ per mm³", transform=ax[0].transAxes, fontsize=8)
    ax[0].legend(fontsize=7, ncol=2)
    for pol, d0, ls in (("TE", 220e-9, "-"), ("TM", 250e-9, "--"), ("TM", 400e-9, ":")):
        m = cs.membrane_mode("Si", d0, LAM, pol, with_group_index=False)
        x, I = cs.field_profile(m, 3.0)
        ax[1].semilogy(x * 1e9, I / I.max(), ls, color=COL["Si"], label=f"Si {d0*1e9:.0f} nm {pol}0, Γ={m.Gamma:.2f}")
    ax[1].set(xlabel="x across membrane (nm)", ylabel="|E_y|² (TE) or |H_y|² (TM), normalised", title="Guided field in Si membranes", ylim=(2e-3, 1.5))
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_evanescent_volume.png")


# ------------------------------------------------------------------ F3: mirrors TE (p) vs TM (s)
def fig3():
    sg = np.linspace(0, 0.9999, 1500)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for pol, d0, c in (("TE", 220e-9, "C3"), ("TM", 250e-9, "C0")):
        m = cs.membrane_mode("Si", d0, LAM, pol, with_group_index=False)
        dbr = cs.DBR(n_tooth=m.neff, N=10, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
        ax[0].semilogy(sg, np.maximum(1 - dbr.R(LAM, sg), 1e-9), color=c, label=f"Si {d0*1e9:.0f} nm {pol}0 ({cs.LATERAL_POL[pol]}-pol), n_eff={m.neff:.2f}")
        ax[0].axvline(1 / m.neff, color=c, ls=":", lw=0.8)
        if pol == "TE":
            ax[0].axvline(1 / np.sqrt(1 + m.neff**2), color=c, ls="--", lw=0.8)
            ax[0].annotate("Brewster hole\n(sinχ ≈ 0.27–0.36)", (0.31, 0.5), (0.45, 3e-3), fontsize=8, color=c, arrowprops=dict(arrowstyle="->", color=c))
    ax[0].set(xlabel="sin χ at the mirror", ylabel="1 − R (N = 10, no extra loss)", title="Mirror loss vs angle: dotted 1/n_eff (TIR), dashed Brewster", ylim=(1e-9, 2))
    ax[0].legend(fontsize=7, loc="lower right")
    d = np.geomspace(120e-9, 700e-9, 30)
    for pol, ls in (("TE", "-"), ("TM", "--")):
        for N, c in ((10, "C0"), (25, "C2")):
            y = []
            for x in d:
                m = cs.membrane_mode("Si", x, LAM, pol, with_group_index=False)
                if not m.guided:
                    y.append(np.nan); continue
                dbr = cs.DBR(n_tooth=m.neff, N=N, m_gap=1, m_tooth=1, slab_pol=pol, bounce_loss=0)
                y.append(1 - cs.uniform_average(lambda s: dbr.R(LAM, s)))
            ax[1].semilogy(d * 1e9, y, ls, color=c, label=f"{pol}0, N={N}")
    ax[1].set(xlabel="Si membrane thickness (nm)", ylabel="⟨1 − R⟩ over uniform sinχ", title="Angle-averaged mirror loss (chaotic cell sees all angles)")
    ax[1].legend(fontsize=8)
    # first-order DBR spectra TM vs TE at normal incidence
    lam = np.linspace(1.3e-6, 1.9e-6, 1500)
    for pol, d0, c in (("TE", 220e-9, "C3"), ("TM", 250e-9, "C0"), ("TM", 300e-9, "C1")):
        grid = np.linspace(lam[0], lam[-1], 40)
        nt = np.interp(lam, grid, [cs.membrane_mode("Si", d0, x, pol, with_group_index=False).neff for x in grid])
        n0 = cs.membrane_mode("Si", d0, LAM, pol, with_group_index=False).neff
        lay = [(1.0, LAM / 4), (nt, LAM / (4 * n0))] * 6
        ax[2].plot(lam * 1e9, cs.stack_R_oblique(lam, np.zeros_like(lam), nt, lay, 1.0, cs.LATERAL_POL[pol]), color=c, label=f"Si {d0*1e9:.0f} nm {pol}0 (n_eff {n0:.2f}), N=6")
    ax[2].set(xlabel="wavelength (nm)", ylabel="R at normal incidence", title="First-order DBR stop band (membrane dispersion included)", ylim=(0, 1.02))
    ax[2].legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_mirrors_TE_vs_TM.png")


# ------------------------------------------------------------------ F4: thickness trade-off
def fig4():
    d = np.geomspace(110e-9, 700e-9, 22)
    ref = cs.membrane_mode("Si", 220e-9, LAM, "TE", with_group_index=False)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
    tab = make_table("curved", ref_mode(), n_rays=2500)        # one geometric ray table for all thicknesses (launch spread barely matters for path statistics)
    for mat in ("Si", "SiNx", "Al2O3"):
        for pol, ls in (("TE", "-"), ("TM", "--")):
            rows = []
            for x in d:
                m = cs.membrane_mode(mat, x, LAM, pol, with_group_index=False)
                if not m.guided or m.neff < 1.05:
                    rows.append([x] + [np.nan] * 6); continue
                Rf, _ = mirror_fn(m, mat=mat)
                a_c = cs.dB_per_cm_to_alpha(CFG["alpha_ref_dB_cm"])
                r_c = cs.evaluate(tab, Rf, a_c, m.Gamma)
                r_r = cs.evaluate(tab, Rf, alpha_bg_for(m, mat, ref), m.Gamma)
                rows.append([x, m.Gamma, 1 - cs.uniform_average(Rf), r_c.L_eff_gas, r_c.S1, r_r.L_eff_gas, r_r.S1])
            R = np.array(rows)
            lab = f"{mat} {pol}0"
            ax[0].plot(d * 1e9, R[:, 1], ls, color=COL[mat], label=lab)
            ax[1].semilogy(d * 1e9, R[:, 2], ls, color=COL[mat])
            ax[2].semilogy(d * 1e9, R[:, 3] * 100, ls, color=COL[mat])
            ax[2].semilogy(d * 1e9, R[:, 5] * 100, ":", color=COL[mat])
            ax[3].semilogy(d * 1e9, R[:, 4] * 100, ls, color=COL[mat])
            ax[3].semilogy(d * 1e9, R[:, 6] * 100, ":", color=COL[mat])
    ax[0].set(ylabel="Γ", title="Evanescent factor Γ"); ax[0].legend(fontsize=7, ncol=2, loc="upper right")
    ax[1].set(ylabel="⟨1 − R⟩ (angle average)", title="Mirror loss (N = 10 Si, 25 others)", ylim=(1e-4, 2))
    ax[2].set(ylabel="Γ⟨L⟩ (cm)", title="Γ⟨L⟩: 0.1 dB/cm; dotted: roughness-scaled", ylim=(0.05, 100))
    ax[2].axhline(50, color="k", lw=0.8, ls=":"); ax[2].text(115, 55, "50 cm milestone", fontsize=8)
    ax[3].set(ylabel="S1 = Γ T_det ⟨L⟩ (cm)", title="Signal yield (S/N scales with S1)", ylim=(1e-4, 10))
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("membrane thickness (nm)")
    fig.suptitle(f"Curved-facet segmented cell, D = 1 cm, {CFG['port_w']*1e6:.0f} µm ports, single-mode injection. Solid = TE0 (p-pol mirror), dashed = TM0 (s-pol mirror). "
                 f"Assumed loss {CFG['alpha_ref_dB_cm']} dB/cm (Si 220 nm TE), bounce loss {CFG['bounce_loss']:.0e}.", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_thickness_tradeoff_TE_TM.png")


# ------------------------------------------------------------------ F5: segmented cells, Poincaré sections
def fig5():
    m = ref_mode()
    Rf, _ = mirror_fn(m)
    kinds = (("regular", "regular 24-gon (flat facets)"), ("tilt", f"random facet tilt {CFG['tilt_deg']}° rms"),
             ("curved", f"curved facets, ρ = {CFG['curv_radius']*100:.0f} cm"))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for k, (kind, title) in enumerate(kinds):
        cell = make_cell(kind)
        tab = make_table(kind, m, n_rays=1500, keep_start=True)
        x, y = cell.outline()
        a = ax[0, k]
        a.plot(x * 1e3, y * 1e3, "k", lw=1.2)
        det = np.nonzero(tab.exit_port == 1)[0]
        for i, c in zip(det[:3], ("C0", "C1", "C2")):
            xs, ys = cs.trace_path(cell, *tab.start[i], n_hits=min(int(tab.exit_idx[i]) + 1, 70))
            a.plot(xs * 1e3, ys * 1e3, lw=0.6, color=c)
        for s0, c in ((tab.s_in, "tab:green"), (tab.s_out, "tab:red")):
            px, py, _, _ = cell.point_at(np.array([s0]))
            a.plot(px * 1e3, py * 1e3, "o", color=c, ms=7)
        a.set_aspect("equal"); a.grid(False); a.set_title(title, fontsize=10)
        S, SC = cs.poincare(cell, 40, 500, seed=k)
        b = ax[1, k]
        b.plot(S.ravel() / cell.perimeter, SC.ravel(), ".", ms=0.7, color="k", alpha=0.5)
        b.set(xlabel="boundary coordinate s / perimeter", ylabel="sin χ (signed)", ylim=(-1, 1), title="Poincaré section: 40 trajectories × 500 bounces")
    fig.suptitle("Segmented circular cell (green: input port, red: output port). Flat facets: pseudo-integrable (lines in phase space). "
                 "Small designed perturbations fill phase space, spreading path lengths and exit directions.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_segmented_cells_poincare.png")


# ------------------------------------------------------------------ F6: port width / loss map
def fig6():
    m = ref_mode()
    Rf, _ = mirror_fn(m)
    widths = np.array([15, 30, 60, 120, 250, 500]) * 1e-6
    a_dB = np.geomspace(0.01, 2.0, 14)
    L = np.zeros((a_dB.size, widths.size)); T = np.zeros_like(L)
    for j, w in enumerate(widths):
        tab = make_table("curved", m, port_w=w, n_rays=4000)
        for i, a in enumerate(a_dB):
            r = cs.evaluate(tab, Rf, cs.dB_per_cm_to_alpha(a), m.Gamma)
            L[i, j], T[i, j] = r.L_mean, r.T_det
    G = m.Gamma
    X, Y = np.meshgrid(widths * 1e6, a_dB)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    c0 = ax[0].contourf(X, Y, np.log10(L * 100), 20, cmap="viridis"); plt.colorbar(c0, ax=ax[0], label="log10 ⟨L⟩ [cm]")
    ax[0].contour(X, Y, L * 100, [10, 50, 100, 500], colors="w", linewidths=0.8); ax[0].set_title("Geometric path ⟨L⟩ (white: 10, 50, 100, 500 cm)", fontsize=9)
    c1 = ax[1].contourf(X, Y, np.log10(G * L * 100), 20, cmap="magma"); plt.colorbar(c1, ax=ax[1], label="log10 Γ⟨L⟩ [cm]")
    ax[1].contour(X, Y, G * L * 100, [50], colors="c", linewidths=1.5)
    ax[1].set_title(f"Gas-equivalent path, Γ = {G:.2f} (cyan: 50 cm milestone)", fontsize=9)
    S1 = G * T * L * 100
    c2 = ax[2].contourf(X, Y, np.log10(S1), 20, cmap="cividis"); plt.colorbar(c2, ax=ax[2], label="log10 S1 [cm]")
    ax[2].set_title("Signal yield Γ T_det ⟨L⟩", fontsize=9)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set(xlabel="port width w (µm)", ylabel="background loss α_bg (dB/cm)")
    fig.suptitle(f"{CFG['material']} {CFG['d']*1e9:.0f} nm {CFG['pol']}0, curved-facet cell, D = 1 cm, DBR N = {CFG['dbr_N']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_path_vs_port_width_and_loss.png")
    return widths, a_dB, L, T


# ------------------------------------------------------------------ shared state for gas / coherence figures
_STATE = {}


def state(kind="curved"):
    if kind not in _STATE:
        m = ref_mode()
        Rf, _ = mirror_fn(m)
        a = alpha_bg_for(m, "Si")
        tab = make_table(kind, m)
        p = cm.prepare(tab, Rf, a, m.Gamma, LAM, m.neff, m.n_group, CFG["port_w"])
        _STATE[kind] = (m, p)
    return _STATE[kind]


# ------------------------------------------------------------------ F7: gas spectra
def fig7():
    m, p = state("curved")
    lines, src = gs.load_lines()
    nu = np.linspace(5880, 6670, 6000)
    alpha, parts = gs.alpha_mixture(nu, gs.BREATH_EXAMPLE, lines)
    Lgas = m.Gamma * p.L_mean * 100
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    col = {"CO2": "C1", "H2O": "C0", "CH4": "C2", "NH3": "C3"}
    for sp, a in parts.items():
        ax[0].semilogy(1e7 / nu, np.maximum(a * Lgas, 1e-12), color=col[sp], label=f"{sp} {gs.BREATH_EXAMPLE[sp]*1e6:g} ppm")
    ax[0].set(xlabel="wavelength (nm)", ylabel=f"absorbance A = α Γ⟨L⟩  (Γ⟨L⟩ = {Lgas:.0f} cm)", ylim=(1e-9, 1), title="Breath-like mixture, 1 atm")
    ax[0].legend(fontsize=8)
    for k, (sp, nu0, concs, w) in enumerate((("NH3", CFG["nu_NH3"], (0.1e-6, 0.5e-6, 2e-6), 0.6), ("CH4", CFG["nu_CH4"], (2e-6, 20e-6, 200e-6), 0.6))):
        nz = np.linspace(nu0 - w, nu0 + w, 1500)
        for c, cc in zip(concs, ("C0", "C1", "C3")):
            mix = dict(gs.BREATH_EXAMPLE); mix[sp] = c
            al, _ = gs.alpha_mixture(nz, mix, lines)
            T = gs.through_cell(al, p.L_all, p.W_all, m.Gamma)
            ax[1 + k].plot(1e7 / nz, 1 - T, color=cc, label=f"{sp} {c*1e6:g} ppm")
        al0, _ = gs.alpha_mixture(nz, {k_: v for k_, v in gs.BREATH_EXAMPLE.items() if k_ != sp}, lines)
        ax[1 + k].plot(1e7 / nz, 1 - gs.through_cell(al0, p.L_all, p.W_all, m.Gamma), "k--", lw=0.8, label="CO₂ + H₂O only")
        ax[1 + k].set(xlabel="wavelength (nm)", ylabel="absorbed fraction 1 − T", title=f"{sp} line at {1e7/nu0:.1f} nm, p(L)-weighted")
        ax[1 + k].legend(fontsize=8)
    fig.suptitle(f"Source of line data: {src}.  Peak absorbances ~1e-5 per ppm need baseline noise at the 1e-6 level.", fontsize=9, color="C3")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_gas_spectra_through_cell.png")
    a1, nu1 = gs.peak_alpha_per_ppm("NH3")
    print(f"NH3 peak alpha {a1:.2e} /cm per ppm at {1e7/nu1:.1f} nm; A per ppm = {a1*Lgas:.2e} for Γ<L>={Lgas:.1f} cm")


# ------------------------------------------------------------------ F8: simulated coherent spectra with a gas line
def fig8():
    lines, _ = gs.load_lines()
    fig, ax = plt.subplots(3, 2, figsize=(14, 9.5), sharex=True)
    nu0 = CFG["nu_NH3"] * gs.C_CM
    dn = np.linspace(-5e9, 5e9, 5001)
    nz = CFG["nu_NH3"] + dn / gs.C_CM
    mix = {"NH3": 2000e-6}
    for k, (kind, title) in enumerate((("regular", "regular 24-gon"), ("curved", "curved facets"))):
        m, p = state(kind)
        al, _ = gs.alpha_mixture(nz, mix, lines)
        ideal = gs.through_cell(al, p.L_all, p.W_all, m.Gamma)
        I0s = cm.simulate_spectrum(p, dn, nu0, None, "single", seed=1)
        I0 = cm.simulate_spectrum(p, dn, nu0, None, "all", seed=1)
        ax[0, k].plot(dn / 1e9, I0s, color="0.6", lw=0.7, label="single-mode port")
        ax[0, k].plot(dn / 1e9, I0, color="C0", lw=1.0, label="detector sums all modes")
        ax[0, k].set(ylabel="I(ν), no gas", title=f"{title}: C(sum) = {np.std(I0)/np.mean(I0):.2f}, C(single) = {np.std(I0s)/np.mean(I0s):.2f}")
        ax[0, k].legend(fontsize=8, loc="upper right")
        for row, (dT, lab) in enumerate(((0.0, "static baseline (same speckle pattern)"), (10e-3, "baseline drifted by 10 mK")), start=1):
            Ig = cm.simulate_spectrum(p, dn, nu0, al, "all", dT=dT, dneff_dT=CFG["dneff_dT"], seed=1)
            ax[row, k].plot(dn / 1e9, Ig / I0, color="C3", lw=0.9, label="measured / reference")
            ax[row, k].plot(dn / 1e9, ideal, "k", lw=1.2, label="gas only (ideal)")
            ax[row, k].set(ylabel="transmission ratio", title=lab, ylim=(0.97, 1.006) if row == 1 else (0.6, 1.4))
            ax[row, k].legend(fontsize=8, loc="lower right")
        ax[2, k].set_xlabel("detuning from line centre (GHz)")
    fig.suptitle("Simulated coherent output (random-phase path model). NH₃ 2000 ppm, illustrative line parameters; at breath levels (<1 ppm) the line is ~1000× weaker.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_simulated_spectra_gas_plus_speckle.png")


# ------------------------------------------------------------------ F9: coherence metrics
def fig9():
    dnu = np.concatenate([np.geomspace(1e5, 2e9, 600), np.linspace(2e9, 30e9, 1400)])
    gl = np.geomspace(1e5, 1e10, 26)
    dT = np.geomspace(1e-5, 1e-1, 40)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for kind, c, lab in (("regular", "C3", "regular 24-gon"), ("tilt", "C1", "tilt 1°"), ("curved", "C0", "curved facets")):
        m, p = state(kind) if kind != "tilt" else _tilt_state()
        g = cm.autocovariance(p, dnu)
        ax[0].semilogx(dnu / 1e6, g, color=c, label=f"{lab}  (paths/mode {p.n_paths_eff()[0]:.1f}, M={p.M_spat():.0f})")
        ax[1].loglog(gl / 1e6, [cm.contrast(p, x, True) for x in gl], color=c, label=f"{lab}: sum of modes")
        ax[1].loglog(gl / 1e6, [cm.contrast(p, x, False) for x in gl], "--", color=c, label=f"{lab}: single mode")
        sh = cm.thermal_shift_hz(dT, CFG["nu_NH3"] * gs.C_CM, CFG["dneff_dT"], p.n_g)
        ax[2].semilogx(dT * 1e3, 1 - np.interp(sh, dnu, g), color=c, label=lab)
    ax[0].set(xlabel="frequency offset Δν (MHz)", ylabel="speckle autocovariance g(Δν)", title="Spectral correlation (revivals = etalon-like fringes)")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="laser linewidth FWHM (MHz)", ylabel="speckle contrast σ_I/⟨I⟩", title="Contrast vs laser linewidth", ylim=(1e-3, 1.2))
    ax[1].legend(fontsize=6, ncol=1)
    ax[2].set(xlabel="temperature drift ΔT (mK)", ylabel="pattern decorrelation 1 − ρ", title=f"Pattern decorrelation, dn_eff/dT = {CFG['dneff_dT']:.1e} /K", ylim=(1e-4, 1.2))
    ax[2].set_yscale("log"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_coherence_metrics.png")


def _tilt_state():
    if "tilt" not in _STATE:
        m = ref_mode(); Rf, _ = mirror_fn(m)
        tab = make_table("tilt", m)
        _STATE["tilt"] = (m, cm.prepare(tab, Rf, alpha_bg_for(m, "Si"), m.Gamma, LAM, m.neff, m.n_group, CFG["port_w"]))
    return _STATE["tilt"]


# ------------------------------------------------------------------ F10: speckle-limited detection limit
def fig10():
    lines, src = gs.load_lines()
    dnu = np.concatenate([np.linspace(0, 2e9, 1500), np.linspace(2e9, 30e9, 1400)])
    nu0 = CFG["nu_NH3"] * gs.C_CM
    a1, _ = gs.peak_alpha_per_ppm("NH3")                                   # 1/cm per ppm
    line_fwhm = 2 * 0.095 * gs.C_CM                                        # 1 atm, Hz
    dT = np.geomspace(1e-5, 1e-1, 30)
    gl = np.geomspace(1e5, 1e10, 26)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for kind, c, lab in (("regular", "C3", "regular"), ("curved", "C0", "curved facets")):
        m, p = state(kind)
        g = cm.autocovariance(p, dnu)
        A1 = a1 * m.Gamma * p.L_mean * 100                                 # absorbance per ppm
        for spatial, ls, dl in ((True, "-", "sum of modes"), (False, "--", "single-mode port")):
            sig = [cm.speckle_noise_A(p, dnu, g, x, nu0, CFG["dneff_dT"], CFG["laser_fwhm"], line_fwhm, spatial) for x in dT]
            ax[0].loglog(dT * 1e3, np.array(sig) / A1 * 1e3, ls, color=c, label=f"{lab}, {dl}")
            sig2 = [cm.speckle_noise_A(p, dnu, g, 1e-2, nu0, CFG["dneff_dT"], x, line_fwhm, spatial) for x in gl]
            ax[1].loglog(gl / 1e6, np.array(sig2) / A1 * 1e3, ls, color=c, label=f"{lab}, {dl}")
        shot = gs.shot_noise_A(1e-3, p.T_det, 1.0) / A1 * 1e3
    for a in ax:
        a.axhline(shot, color="k", lw=1, ls="-."); a.text(0.02, 0.04, f"shot-noise floor, 1 mW in, 1 Hz: {shot:.2g} ppb", transform=a.transAxes, fontsize=8)
        a.axhline(100, color="g", lw=0.8, ls=":"); a.axhline(1e4, color="g", lw=0.8, ls=":")
        a.set_ylabel("speckle-limited NH₃ detection limit (ppb)")
    ax[0].set(xlabel="temperature drift between reference and measurement (mK)", title=f"Laser {CFG['laser_fwhm']/1e6:g} MHz FWHM")
    ax[1].set(xlabel="laser linewidth FWHM (MHz)", title="ΔT = 10 mK")
    ax[0].legend(fontsize=7)
    ax[0].text(0.55, 0.9, "green dotted: 100 ppb and 10 ppm milestones", transform=ax[0].transAxes, fontsize=7, color="g")
    fig.suptitle(f"NH₃ line near 1531.7 nm ({src}); Γ⟨L⟩-based absorbance; 1 atm; Si 250 nm TM0 cell, D = 1 cm", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_speckle_limited_detection_limit.png")


if __name__ == "__main__":
    which = sys.argv[1:] or [str(i) for i in range(1, 11)]
    for w in which:
        print("figure", w, flush=True)
        globals()[f"fig{w}"]()
    print("done ->", OUT)
