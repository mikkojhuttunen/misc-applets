"""Guided slab mode of a free-standing membrane (n_clad | core | n_clad) and its evanescent gas-sensing quantities.

Symmetric three-layer slab, core centred on x = 0, thickness d. Field f(x) = cos(κx) (even orders) or sin(κx)
(odd orders) inside, f(±d/2) exp(-γ(|x| - d/2)) outside; f is E_y for TE and H_y for TM.

Γ is the absorption-coupling factor: power dissipated in the cladding gas divided by the Poynting flux, so that
the modal absorption coefficient is α_mode = Γ α_gas (α_gas: bulk absorption coefficient of the gas).

    TE:  Γ = (n_c / n_eff) ∫_clad |E|² / ∫ |E|²
    TM:  Γ = n_c ∫_clad (β² H² + H'²) / n_c⁴  /  ( k β ∫ H² / n² )

TM Γ can exceed 1 close to cut-off (the E field jumps by n_core²/n_clad² at the surface). SI units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Result, require_choice, require_positive
from ..materials import engine as _mat
from ..slab_waveguide.engine import neff_three_layer

_POL = ("TE", "TM")
MATERIAL_CHOICES = ("constant",) + _mat.MATERIALS
N_AIR_NUMBER_DENSITY = 2.5e25  # m^-3, air at ~293 K, 1 atm


@dataclass
class MembraneMode:
    """Mode quantities (helper output). Lengths in m, gamma_ev and E_edge2 in 1/m."""

    thickness: float
    wavelength: float
    pol: str
    order: int
    n_core: float
    n_clad: float
    neff: float
    n_group: float
    gamma_ev: float     # cladding field decay constant, field ~ exp(-gamma_ev |x|)
    Gamma: float        # alpha_mode = Gamma * alpha_gas
    f_clad_E: float     # share of ∫f² outside the core (both sides)
    z_p: float          # 1/e intensity penetration depth per side = 1/(2 gamma_ev)
    E_edge2: float      # |E| on the core side of the surface per unit guided power (common factors dropped)
    material: str = "constant"

    @property
    def guided(self) -> bool:
        return bool(np.isfinite(self.neff))


def core_index(material, wavelength, n_core=None):
    """n_core if material is 'constant' (or None), else the Sellmeier index of the materials engine."""
    if material in (None, "constant"):
        return float(n_core)
    return float(_mat.index(material, wavelength))


def _surface_weight(pol, even, kap, beta, k, nf, nc, d, core, clad, edge):
    """|E|² on the core side of the surface per unit guided power, 1/m. A proxy for surface-roughness
    scattering: comparable between thicknesses and materials, not an absolute loss."""
    if pol == "TE":
        return edge**2 / (beta / k * (core + clad))
    fp = kap * (np.sin(kap * d / 2) if even else np.cos(kap * d / 2))
    return (fp**2 + beta**2 * edge**2) / (k * beta * nf**4 * (core / nf**2 + clad / nc**2))


def solve_mode(wavelength, thickness, n_core=None, n_clad=1.0, polarization="TE", order=0,
               material="constant", group_index=True) -> MembraneMode:
    """Symmetric membrane mode (helper). NaN fields below cut-off. The group index is a central difference
    of n_eff(λ) including material dispersion when a Sellmeier material is named."""
    lam, d, nc = float(wavelength), float(thickness), float(n_clad)
    nf = core_index(material, lam, n_core)
    neff = neff_three_layer(lam, nc, nf, nc, d, polarization, order)
    if not np.isfinite(neff):
        nan = np.nan
        return MembraneMode(d, lam, polarization, order, nf, nc, nan, nan, nan, nan, nan, nan, nan, material or "constant")
    k = 2 * np.pi / lam
    kap = k * np.sqrt(nf**2 - neff**2)
    gam = k * np.sqrt(neff**2 - nc**2)
    beta = k * neff
    even = order % 2 == 0
    edge = np.cos(kap * d / 2) if even else np.sin(kap * d / 2)
    core = d / 2 + (1 if even else -1) * np.sin(kap * d) / (2 * kap)   # ∫_core f² dx
    clad = edge**2 / gam                                               # both sides
    f_clad = clad / (core + clad)
    if polarization == "TE":
        Gamma = (nc / neff) * f_clad
    else:
        Gamma = (nc * edge**2 * (beta**2 + gam**2) / gam / nc**4) / (k * beta * (core / nf**2 + clad / nc**2))
    n_g = np.nan
    if group_index:
        h = 1e-3 * lam
        lo, hi = (neff_three_layer(l, nc, core_index(material, l, n_core), nc, d, polarization, order) for l in (lam - h, lam + h))
        n_g = neff - lam * (hi - lo) / (2 * h)
    return MembraneMode(d, lam, polarization, order, nf, nc, float(neff), float(n_g), float(gam), float(Gamma), float(f_clad),
                        float(1 / (2 * gam)), float(_surface_weight(polarization, even, kap, beta, k, nf, nc, d, core, clad, edge)),
                        material or "constant")


def field_profile(mode: MembraneMode, extent=4.0, n=801):
    """x [m] and f(x)² (|E_y|² for TE, |H_y|² for TM, unnormalised) across the membrane, out to
    `extent` decay lengths 1/γ beyond each surface (helper, for plots and quadrature checks)."""
    k = 2 * np.pi / mode.wavelength
    kap = k * np.sqrt(mode.n_core**2 - mode.neff**2)
    gam, half = mode.gamma_ev, mode.thickness / 2
    x = np.linspace(-(half + extent / gam), half + extent / gam, n)
    even = mode.order % 2 == 0
    inside = np.cos(kap * x) if even else np.sin(kap * x)
    edge = np.cos(kap * half) if even else np.sign(x) * np.sin(kap * half)
    f = np.where(np.abs(x) <= half, inside, edge * np.exp(-gam * (np.abs(x) - half)))
    return x, f**2


def roughness_scaled_alpha(mode: MembraneMode, mode_ref: MembraneMode, alpha_ref, index_contrast=True):
    """Background loss [1/m] scaled from a reference membrane with loss alpha_ref, assuming surface-roughness
    scattering dominates: α ∝ E_edge2 (× (n_core² - n_clad²)² when index_contrast). Crude; thin membranes scatter more."""
    a = alpha_ref * mode.E_edge2 / mode_ref.E_edge2
    if index_contrast:
        a *= ((mode.n_core**2 - mode.n_clad**2) / (mode_ref.n_core**2 - mode_ref.n_clad**2)) ** 2
    return a


def evanescent_mode(wavelength, thickness, n_core=3.476, n_clad=1.0, polarization="TE", order=0, material="constant") -> Result:
    """Effective and group index, evanescent absorption factor Γ, decay constant, penetration depth and
    surface-field weight of a membrane slab mode. material='constant' uses n_core; otherwise a Sellmeier fit."""
    require_positive(wavelength=wavelength, thickness=thickness, n_clad=n_clad)
    require_choice("polarization", polarization, _POL)
    require_choice("material", material, MATERIAL_CHOICES)
    if material == "constant":
        require_positive(n_core=n_core)
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    m = solve_mode(wavelength, thickness, n_core, n_clad, polarization, order, material)
    assumptions = ["Symmetric lossless slab, claddings semi-infinite", "Weak gas absorption: α_mode = Γ α_gas",
                   "NaN below cut-off", "E_edge2 is a relative roughness-scattering weight, not an absolute loss"]
    if material == "constant":
        assumptions.append("Constant core index: n_group contains waveguide dispersion only")
    else:
        assumptions.append(_mat._SOURCES[material])
    return Result(
        values={"neff": m.neff, "n_group": m.n_group, "n_core": m.n_core, "Gamma": m.Gamma, "f_clad": m.f_clad_E,
                "gamma_ev": m.gamma_ev, "z_p": m.z_p, "E_edge2": m.E_edge2, "guided": m.guided},
        units={"neff": "", "n_group": "", "n_core": "", "Gamma": "", "f_clad": "", "gamma_ev": "1/m", "z_p": "m",
               "E_edge2": "1/m", "guided": ""},
        assumptions=assumptions,
    )


def evanescent_volume(z_p, area, sides=2, number_density=N_AIR_NUMBER_DENSITY) -> Result:
    """Gas volume within the 1/e intensity depth z_p over a footprint `area` (one or both faces),
    and the number of analyte molecules in it per ppb of mole fraction."""
    require_positive(z_p=z_p, area=area, number_density=number_density)
    sides = int(sides)
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    V = sides * area * z_p
    return Result(
        values={"V_ev": V, "molecules_per_ppb": number_density * V * 1e-9},
        units={"V_ev": "m^3", "molecules_per_ppb": ""},
        assumptions=["Volume = sides × area × z_p (1/e intensity depth)", "Ideal gas, default air at ~293 K and 1 atm"],
    )
