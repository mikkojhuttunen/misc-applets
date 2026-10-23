"""How a free-standing membrane senses gas: the evanescent field of the guided slab mode reaches into the gas on both
sides, and only molecules inside it absorb.

Panel 1  cross-section through the membrane (heights to scale), mode intensity as a red shading, molecules drawn
         (enlarged) inside and outside the evanescent layer, the intensity decay length δ = 1/(2γ).
Panel 2  intensity profile |H|^2 (TM) or |E|^2 (TE) across the membrane; the part in the gas is shaded; Γ from
         cmpc_sim.membrane_mode (modal absorption per unit gas absorption: α_mode = Γ α_gas).
Panel 3  Γ against membrane thickness for TE0 and TM0: a thinner membrane puts more of the mode in the gas.

    python evanescent_sensing.py                       Si 250 nm, TM0, 1.55 µm
    python evanescent_sensing.py --thickness 220 --pol TE
Writes figs/evanescent_sensing.png and .pdf (vector).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "cmpc-sim" / "cmpc"))
import cmpc_sim as cs  # noqa: E402

OUT = HERE / "figs"
CPK = {"C": "#3d3d3d", "H": "#f4f4f4", "O": "#d33a2c", "N": "#3a5bd9"}
RAD = {"C": 0.40, "H": 0.27, "O": 0.38, "N": 0.39}
SHAPES2D = {"CH4": [("C", 0, 0), ("H", 1, 0), ("H", -1, 0), ("H", 0, 1), ("H", 0, -1)],
            "CO2": [("C", 0, 0), ("O", 1, 0), ("O", -1, 0)],
            "H2O": [("O", 0, 0), ("H", 0.61, 0.79), ("H", 0.61, -0.79)],
            "NH3": [("N", 0, 0), ("H", 0.94, -0.34), ("H", -0.94, -0.34), ("H", 0, 1)]}
plt.rcParams.update({"font.size": 10, "font.family": ["Inter", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False})


def molecule(ax, kind, x, z, b, ang, sx, sensed):
    """2D ball-and-stick molecule; sx stretches x so atoms stay round on an axis with unequal scales."""
    c, s_ = np.cos(ang), np.sin(ang)
    atoms = [(el, x + b * (c * px - s_ * pz) * sx, z + b * (s_ * px + c * pz)) for el, px, pz in SHAPES2D[kind]]
    if sensed:
        for f, al in ((2.3, 0.12), (1.8, 0.15), (1.35, 0.18)):
            ax.add_patch(matplotlib.patches.Ellipse((x, z), 2 * f * b * sx, 2 * f * b, color="#ff2a1a", alpha=al, lw=0, zorder=3))
    for el, px, pz in atoms[1:]:
        ax.plot([atoms[0][1], px], [atoms[0][2], pz], color="#6b7378", lw=1.2, zorder=4)
    for el, px, pz in atoms:
        ax.add_patch(matplotlib.patches.Ellipse((px, pz), 2 * RAD[el] * b * sx, 2 * RAD[el] * b, fc=CPK[el], ec="#2b3135", lw=0.5, zorder=5))


def mode_data(thickness_nm=250, pol="TM", lam_um=1.55):
    """Slab mode, intensity profile and decay lengths shared by the panels (lengths in m, profile x in µm)."""
    lam, t = lam_um * 1e-6, thickness_nm * 1e-9
    m = cs.membrane_mode("Si", t, lam, pol, with_group_index=False)
    delta = 1 / (2 * m.gamma_ev)                          # intensity decay length; gamma_ev is the field decay constant
    x, I = cs.field_profile(m, 10.0, 2001)
    return dict(m=m, t=t, lam=lam, pol=pol, thickness=thickness_nm, lam_um=lam_um, delta=delta, x=x, I=I / I.max(),
                zmax=(t / 2 + 9 * delta) * 1e6)


def cross_section(fig, ax, md, seed=4, L=4.0, text=True, n_in=5, n_out=4):
    """Panel 1: membrane cross-section (heights to scale), intensity shading, δ and 3δ lines, molecules."""
    um, t, delta, zmax, m = 1e6, md["t"], md["delta"], md["zmax"], md["m"]
    zz = np.linspace(-zmax, zmax, 600)
    Iz = np.interp(zz / um, md["x"], md["I"], left=0, right=0)
    rgba = np.zeros((len(zz), 2, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb("#ff2a1a")
    rgba[..., 3] = (0.75 * Iz ** 0.35)[:, None]                 # compressed scale so the tails show
    ax.imshow(rgba, extent=(0, L, -zmax, zmax), origin="lower", aspect="auto", zorder=1, interpolation="bilinear")
    ax.add_patch(Rectangle((0, -t / 2 * um), L, t * um, fc="#8fa3c4", ec="#1f4e79", lw=1.0, alpha=0.75, zorder=2))
    for sgn in (1, -1):
        ax.axhline(sgn * (t / 2 + delta) * um, color="#c0392b", ls="--", lw=0.9, zorder=2)
        ax.axhline(sgn * (t / 2 + 3 * delta) * um, color="#c0392b", ls=":", lw=0.8, zorder=2)
    ax.add_patch(FancyArrowPatch((0.25, 0), (1.35, 0), arrowstyle="-|>", mutation_scale=14, color="#1b2328", lw=1.6, zorder=6))
    if text:
        ax.text(L - 0.03, (t / 2 + delta) * um + 0.01, f"δ = 1/(2γ) = {delta * 1e9:.0f} nm (intensity /e)", ha="right", va="bottom", color="#a12a1f", fontsize=9)
        ax.text(L - 0.03, (t / 2 + 3 * delta) * um + 0.01, "3δ: intensity down to 5 %", ha="right", va="bottom", color="#a12a1f", fontsize=9)
        ax.text(0.27, 0.012, f"guided {md['pol']}₀ mode, n_eff = {m.neff:.3f}", fontsize=9, va="bottom", zorder=6)
        ax.text(L / 2, -t / 2 * um + 0.01, f"Si membrane, {md['thickness']:.0f} nm", ha="center", va="bottom", fontsize=9, color="#12324f", zorder=6)
    # molecules: enlarged, half inside the evanescent layer, half beyond it
    rng = np.random.default_rng(seed)
    b = 0.035 * zmax / 0.6
    ax.set(xlim=(0, L), ylim=(-zmax, zmax))
    # keep atoms round: data aspect correction (x span L over panel width, z span 2 zmax over panel height)
    fig.canvas.draw()
    bb = ax.get_window_extent()
    sx = (L / bb.width) / (2 * zmax / bb.height)
    placed = []
    kinds = list(SHAPES2D)
    for side in (1, -1):
        for zone, n in (("in", n_in), ("out", n_out)):
            for _ in range(n):
                for _try in range(200):
                    xm = rng.uniform(0.25, 2.55 if zone == "in" else L - 0.25)
                    if zone == "in":
                        zm = side * (t / 2 * um + rng.uniform(0.9 * b, 2.6 * delta * um))
                    else:
                        zm = side * rng.uniform((t / 2 + 4.5 * delta) * um + b, zmax - 1.4 * b)
                    if all(abs(xm - p[0]) > 3 * b * sx or abs(zm - p[1]) > 3 * b for p in placed) and not (xm < 1.7 and abs(zm) < 0.06):
                        break
                placed.append((xm, zm))
                molecule(ax, kinds[rng.integers(4)], xm, zm, b, rng.uniform(0, np.pi), sx, zone == "in")


def mode_profile(ax, md):
    """Panel 2: intensity across the membrane, gas part shaded, Γ."""
    um, x, I, t, m = 1e6, md["x"], md["I"], md["t"], md["m"]
    inside = np.abs(x) <= t / 2
    ax.fill_betweenx(x * um, 0, np.where(inside, I, np.nan), color="#8fa3c4", alpha=0.8, lw=0)
    ax.fill_betweenx(x * um, 0, np.where(~inside, I, np.nan), color="#ff2a1a", alpha=0.45, lw=0)
    ax.plot(I, x * um, color="#7a2a20", lw=1.4)
    ax.set(xlabel=f"{'|H|²' if md['pol'] == 'TM' else '|E|²'} (norm.)", xlim=(0, 1.08), title="Mode profile")
    ax.text(0.5, 0.97, f"Γ = {m.Gamma:.2f}:\nα_mode = Γ α_gas", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#a12a1f")


def gamma_curves(ax, md, text=True, lw=1.8):
    """Panel 3: Γ against membrane thickness for TM0 and TE0, the design point marked."""
    d = np.linspace(100, 600, 51) * 1e-9
    for pol, col in (("TM", "#c0392b"), ("TE", "#1f4e79")):
        G = [cs.membrane_mode("Si", di, md["lam"], pol, with_group_index=False).Gamma for di in d]
        ax.plot(d * 1e9, G, color=col, lw=lw, label=f"{pol}₀")
    G0 = md["m"].Gamma
    ax.plot([md["thickness"]], [G0], "o", color="#1b2328", ms=6 if text else 4, zorder=5)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    if text:
        ax.annotate(f"this design\nΓ = {G0:.2f}", (md["thickness"], G0), (md["thickness"] + 55, G0 - 0.05), fontsize=9,
                    arrowprops=dict(arrowstyle="-", lw=0.8))
        ax.set(xlabel="Si membrane thickness (nm)", ylabel="Γ (modal absorption / gas absorption)",
               title=f"Thinner membrane, more field in the gas ({md['lam_um']:g} µm)")
        ax.legend(frameon=False)
        ax.text(0.40, 0.80, "Γ > 1 for thin TM₀ membranes: E_z jumps by\nn_Si² at the surface, so the gas side carries\nthe stronger field. Cost of going thinner:\nn_eff → 1 (weaker DBR contrast, mode spreads\nfar out) and more surface-roughness loss",
                transform=ax.transAxes, fontsize=8.5, color="#4a565c", va="top")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thickness", type=float, default=250, help="membrane, nm")
    ap.add_argument("--pol", default="TM", choices=("TE", "TM"))
    ap.add_argument("--lam", type=float, default=1.55, help="µm")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    md = mode_data(a.thickness, a.pol, a.lam)
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(15, 5.4), dpi=130)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 0.8, 1.25], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    cross_section(fig, ax, md, a.seed)
    ax.set(xlabel="along the membrane (µm, schematic)", ylabel="height z (µm, to scale)",
           title="Evanescent-field sensing: only molecules inside the red layer absorb")
    ax.text(0.01, 0.01, "molecules enlarged ~1000×; halo = inside the field; shading = intensity (compressed scale)", transform=ax.transAxes, fontsize=8.5, color="#4a565c")
    a2 = fig.add_subplot(gs[1], sharey=ax)
    mode_profile(a2, md)
    plt.setp(a2.get_yticklabels(), visible=False)
    gamma_curves(fig.add_subplot(gs[2]), md)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"evanescent_sensing.{ext}", bbox_inches="tight")
    m = md["m"]
    print(f"{a.pol}0 {a.thickness:.0f} nm: n_eff {m.neff:.4f}, Gamma {m.Gamma:.3f}, delta (intensity) {md['delta'] * 1e9:.1f} nm -> {OUT / 'evanescent_sensing.png'}")


if __name__ == "__main__":
    main()
