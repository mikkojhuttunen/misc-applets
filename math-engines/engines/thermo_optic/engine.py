"""Thermo-optic and thermal properties of waveguide materials, absorbed heat and the pump power a waveguide tolerates.

Look-up table (LUT) per material, at room temperature (about 300 K) and 1550 nm unless the entry says
otherwise: dn/dT, thermal conductivity k, density, specific heat, linear expansion, band gap, two-photon
absorption β_TPA and Kerr n2, each with the source and a confidence grade. Thin-film thermal conductivities
(PECVD/LPCVD nitride, sputtered or ALD oxides, AlN) depend strongly on deposition; the LUT gives a nominal
value with a range, use the range for worst-case estimates.

Linear model: n(T) = n(T0) + dn/dT (T - T0), valid for ΔT up to some tens of K (dn/dT of Si rises by about
20 % from 300 K to 400 K, k of crystalline Si falls as T^-1.3). Heat sources per unit waveguide length q' (W/m):

    linear absorption   q' = α_abs P                           (α_abs: the absorbing part of the loss, not scattering)
    two-photon (TPA)    q' = β P² / A_eff
    free carriers (FCA) q' = σ N P,  N = τ β P² / (2 hν A_eff²)
    gain medium         q' = η_heat α_p P_p,  η_heat = 1 - η_q λ_p / λ_em   (quantum defect + non-radiative decay)

Temperature rise ΔT = R' q' with R' (K m / W) the thermal resistance per unit length of the cross-section
(strip_thermal_resistance here, or the finite-volume solver in waveguide_thermal). SI units throughout.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range
from ..materials import engine as _mat
from ..slab_waveguide.engine import neff_three_layer

H_PLANCK = 6.62607015e-34
C0 = 299792458.0
Q_E = 1.602176634e-19
DB_TO_NEPER = math.log(10) / 10  # α [1/m] = loss [dB/m] * DB_TO_NEPER

# Every entry: n (at 1550 nm; replaced by the Sellmeier index when `sellmeier` is set), dn_dT [1/K] with a
# (low, high) range, k [W/(m K)] with a range, rho [kg/m^3], cp [J/(kg K)], alpha_L [1/K], Eg [eV],
# beta_tpa [m/W] at 1550 nm (0 when 2hν < Eg, None when not tabulated), n2 [m^2/W] (None when not tabulated),
# confidence ("high", "medium", "low": an order-of-magnitude design value to be replaced by a measurement), source.
LUT: dict[str, dict] = {
    "si": dict(
        name="Crystalline Si (SOI device layer)", sellmeier="si", n=3.476,
        dn_dT=1.84e-4, dn_dT_range=(1.75e-4, 1.90e-4),
        k=148.0, k_range=(50.0, 148.0), rho=2329.0, cp=705.0, alpha_L=2.6e-6, Eg=1.12,
        beta_tpa=8.0e-12, n2=5.0e-18, sigma_fca=1.45e-21, confidence="high",
        source="dn/dT: Komma et al., APL 101, 041905 (2012); TPA, n2: Bristow et al., APL 90, 191104 (2007); FCA: Soref & Bennett (1987); "
               "k: bulk 148, a 220 nm SOI film about 60-100 (phonon boundary scattering)"),
    "sio2": dict(
        name="SiO2 (fused silica, thermal oxide)", sellmeier="sio2", n=1.444,
        dn_dT=0.95e-5, dn_dT_range=(0.8e-5, 1.1e-5),
        k=1.38, k_range=(1.0, 1.4), rho=2200.0, cp=740.0, alpha_L=0.55e-6, Eg=9.0,
        beta_tpa=0.0, n2=2.6e-20, confidence="high",
        source="dn/dT: Leviton & Frey, Proc. SPIE 6273 (2006), Malitson (1965); k: fused silica 1.38, PECVD/thermal oxide films 1.0-1.4"),
    "si3n4": dict(
        name="Si3N4, stoichiometric LPCVD", sellmeier="si3n4", n=1.996,
        dn_dT=2.45e-5, dn_dT_range=(2.0e-5, 4.0e-5),
        k=10.0, k_range=(2.5, 30.0), rho=3100.0, cp=700.0, alpha_L=3.3e-6, Eg=5.0,
        beta_tpa=0.0, n2=2.5e-19, confidence="medium",
        source="dn/dT: Arbabi & Goddard, Opt. Lett. 38, 3878 (2013); k: thin-film values 2.5-30 depending on film and thickness, bulk ceramic ~30"),
    "sinx": dict(
        name="SiNx, PECVD (hydrogenated, N/Si ratio dependent)", sellmeier=None, n=1.90,
        dn_dT=4.0e-5, dn_dT_range=(2.5e-5, 6.0e-5),
        k=1.5, k_range=(0.7, 4.0), rho=2500.0, cp=750.0, alpha_L=2.0e-6, Eg=4.0,
        beta_tpa=0.0, n2=None, confidence="low",
        source="Typical PECVD film values; n 1.8-2.1 and dn/dT both rise with Si content (Si-rich nitride: TPA and higher n2). Measure the film"),
    "ln_e": dict(
        name="LiNbO3 (TFLN), extraordinary (z-polarised)", sellmeier="ln_e", n=2.138,
        dn_dT=3.4e-5, dn_dT_range=(3.0e-5, 4.0e-5),
        k=4.6, k_range=(3.5, 5.6), rho=4650.0, cp=630.0, alpha_L=7.5e-6, Eg=3.8,
        beta_tpa=0.0, n2=None, confidence="medium",
        source="dn/dT: Moretti et al., J. Appl. Phys. 98, 036101 (2005), 1523 nm, congruent; k, ρ, cp: crystal data sheets; "
               "alpha_L along c (15.4e-6 along a). Pyroelectric and photorefractive effects are not included"),
    "ln_o": dict(
        name="LiNbO3 (TFLN), ordinary", sellmeier="ln_o", n=2.211,
        dn_dT=0.3e-5, dn_dT_range=(-0.2e-5, 0.8e-5),
        k=4.6, k_range=(3.5, 5.6), rho=4650.0, cp=630.0, alpha_L=15.4e-6, Eg=3.8,
        beta_tpa=0.0, n2=None, confidence="low",
        source="dn_o/dT is small and its sign differs between reports (Moretti 2005, Schlarb & Betzler 1993); alpha_L along a"),
    "lt_e": dict(
        name="LiTaO3 (TFLT), extraordinary", sellmeier="lt_e_bond", n=2.119,
        dn_dT=1.5e-5, dn_dT_range=(1.0e-5, 2.5e-5),
        k=4.6, k_range=(3.0, 8.8), rho=7450.0, cp=424.0, alpha_L=4.1e-6, Eg=4.6,
        beta_tpa=0.0, n2=None, confidence="low",
        source="dn/dT: estimate from temperature-dependent Sellmeier fits (Abedin & Ito 1996, Bruner et al. 2003 for stoichiometric LT); "
               "verify for the TFLT film. alpha_L along c (16.1e-6 along a)"),
    "lt_o": dict(
        name="LiTaO3 (TFLT), ordinary", sellmeier="lt_o_bond", n=2.123,
        dn_dT=0.5e-5, dn_dT_range=(0.0, 1.2e-5),
        k=4.6, k_range=(3.0, 8.8), rho=7450.0, cp=424.0, alpha_L=16.1e-6, Eg=4.6,
        beta_tpa=0.0, n2=None, confidence="low",
        source="As lt_e; ordinary-axis dn/dT is poorly known"),
    "gaas": dict(
        name="GaAs", sellmeier="gaas", n=3.374,
        dn_dT=2.35e-4, dn_dT_range=(2.0e-4, 2.7e-4),
        k=55.0, k_range=(46.0, 55.0), rho=5317.0, cp=330.0, alpha_L=5.73e-6, Eg=1.424,
        beta_tpa=1.0e-10, n2=1.6e-17, confidence="medium",
        source="dn/dT: Della Corte et al., J. Appl. Phys. 88, 7115 (2000), 1.5 µm; TPA 6-26 cm/GW reported; k, ρ, cp: Adachi, GaAs and related materials (1993)"),
    "algaas": dict(
        name="Al(x)Ga(1-x)As, x = 0.20 (see algaas_properties for other x)", sellmeier=None, n=None,
        dn_dT=None, dn_dT_range=None, k=None, k_range=None, rho=None, cp=None, alpha_L=None, Eg=None,
        beta_tpa=None, n2=None, confidence="low", source=""),
    "inp": dict(
        name="InP", sellmeier=None, n=3.167,
        dn_dT=2.0e-4, dn_dT_range=(1.8e-4, 2.2e-4),
        k=68.0, k_range=(60.0, 68.0), rho=4810.0, cp=310.0, alpha_L=4.6e-6, Eg=1.344,
        beta_tpa=None, n2=None, confidence="medium",
        source="dn/dT: Della Corte et al., J. Appl. Phys. 88, 7115 (2000); TPA at 1550 nm is non-zero (2hν > Eg) but not tabulated here"),
    "aln": dict(
        name="AlN (sputtered or epitaxial film), ordinary", sellmeier="aln_o", n=2.12,
        dn_dT=2.3e-5, dn_dT_range=(2.0e-5, 3.0e-5),
        k=30.0, k_range=(5.0, 285.0), rho=3255.0, cp=740.0, alpha_L=4.2e-6, Eg=6.0,
        beta_tpa=0.0, n2=2.3e-19, confidence="low",
        source="dn/dT: AlN ring-resonator measurements (Xiong et al. 2012, Pernice et al. 2012), about 2.3e-5; "
               "k: bulk single crystal 285, sputtered polycrystalline films 5-100+"),
    "al2o3_film": dict(
        name="Al2O3, amorphous film (ALD, reactive sputtering; Er host)", sellmeier=None, n=1.65,
        dn_dT=1.3e-5, dn_dT_range=(0.5e-5, 2.5e-5),
        k=1.6, k_range=(1.0, 3.3), rho=3200.0, cp=780.0, alpha_L=4.2e-6, Eg=6.5,
        beta_tpa=0.0, n2=None, confidence="low",
        source="n: reactive-sputtered Al2O3 1.60-1.67 at 1550 nm; k: ALD and sputtered films 1.0-3.3; dn/dT taken close to sapphire"),
    "sapphire": dict(
        name="Sapphire (crystalline Al2O3), ordinary", sellmeier="sapphire_o", n=1.746,
        dn_dT=1.3e-5, dn_dT_range=(1.1e-5, 1.5e-5),
        k=35.0, k_range=(30.0, 42.0), rho=3980.0, cp=760.0, alpha_L=5.0e-6, Eg=8.8,
        beta_tpa=0.0, n2=3.0e-20, confidence="medium",
        source="dn/dT: Tropf & Thomas, Handbook of Optical Constants III (1998); k at 300 K 30-42"),
    "tio2_film": dict(
        name="TiO2, amorphous film (negative dn/dT, used for athermal cladding)", sellmeier=None, n=2.30,
        dn_dT=-1.0e-4, dn_dT_range=(-2.5e-4, -0.5e-4),
        k=1.0, k_range=(0.5, 3.0), rho=3800.0, cp=690.0, alpha_L=8.0e-6, Eg=3.3,
        beta_tpa=0.0, n2=None, confidence="low",
        source="Athermal Si/SiN waveguides with TiO2 cladding (Guha et al. 2013, Djordjevic et al. 2013); value depends on deposition and anneal"),
    "su8": dict(
        name="SU-8 / generic optical polymer", sellmeier=None, n=1.57,
        dn_dT=-1.1e-4, dn_dT_range=(-2.0e-4, -0.5e-4),
        k=0.2, k_range=(0.15, 0.3), rho=1200.0, cp=1200.0, alpha_L=52e-6, Eg=4.0,
        beta_tpa=0.0, n2=None, confidence="low",
        source="Typical polymer values; polymers have negative dn/dT of order -1e-4 1/K"),
    "air": dict(
        name="Air, 1 atm", sellmeier=None, n=1.00027,
        dn_dT=-0.92e-6, dn_dT_range=(-1.0e-6, -0.85e-6),
        k=0.026, k_range=(0.024, 0.028), rho=1.2, cp=1005.0, alpha_L=0.0, Eg=12.0,
        beta_tpa=0.0, n2=None, confidence="high",
        source="dn/dT = -(n - 1)/T at constant pressure; k of air at 300 K. Convection is not included"),
}
MATERIALS = tuple(LUT)
CONFIDENCE = ("high", "medium", "low")
_REF_WAVELENGTH = 1.55e-6


def algaas_properties(x: float) -> dict:
    """Al(x)Ga(1-x)As entry at Al fraction x (0..1) by interpolation between GaAs and AlAs.

    k from Adachi's alloy-scattering fit 1/k = 2.27 + 28.83 x - 30 x² (cm K / W); Γ-gap 1.424 + 1.247 x eV
    (x < 0.45), X-gap above; index at 1550 nm and dn/dT linearly between GaAs (3.374, 2.35e-4) and AlAs (2.92,
    1.4e-4), about ±0.02 and ±20 %; Gehrsitz et al. (2000) is the reference model for n(x, λ)."""
    require_range("al_fraction", x, 0.0, 1.0)
    x = float(x)
    Eg = 1.424 + 1.247 * x if x < 0.45 else 1.900 + 0.125 * x + 0.143 * x * x
    k = 100.0 / (2.27 + 28.83 * x - 30.0 * x * x)
    two_photon = 2 * H_PLANCK * C0 / _REF_WAVELENGTH / Q_E
    gaas = LUT["gaas"]
    return dict(
        name=f"Al{x:.2f}Ga{1 - x:.2f}As", sellmeier=None, n=3.374 - 0.454 * x,
        dn_dT=2.35e-4 - 0.95e-4 * x, dn_dT_range=((2.35e-4 - 0.95e-4 * x) * 0.8, (2.35e-4 - 0.95e-4 * x) * 1.2),
        k=k, k_range=(0.8 * k, 1.2 * k), rho=5317.0 - 1557.0 * x, cp=330.0 + 120.0 * x, alpha_L=(5.73 - 0.53 * x) * 1e-6, Eg=Eg,
        beta_tpa=gaas["beta_tpa"] if two_photon > Eg else 0.0, n2=None, confidence="low",
        source="Linear GaAs-AlAs interpolation; k: S. Adachi, J. Appl. Phys. 54, 1844 (1983). x >= 0.18 keeps 2hν(1550 nm) below the gap "
               "(no TPA, the usual choice for AlGaAs-on-insulator); Urbach-tail TPA close to the threshold is not modelled",
    )


LUT["algaas"] = algaas_properties(0.20)


def entry(material: str, al_fraction: float = 0.20) -> dict:
    """The LUT row (a copy), with AlGaAs evaluated at al_fraction."""
    require_choice("material", material, MATERIALS)
    return dict(algaas_properties(al_fraction) if material == "algaas" else LUT[material])


def index_at(material: str, wavelength: float, al_fraction: float = 0.20) -> float:
    """Room-temperature index: the materials-engine Sellmeier fit where one exists, else the LUT value at 1550 nm."""
    e = entry(material, al_fraction)
    return float(_mat.index(e["sellmeier"], wavelength)) if e["sellmeier"] else float(e["n"])


def _notes(material, e, wavelength):
    out = [f"{e['name']}: {e['source']}", f"Confidence: {e['confidence']}"]
    if e["sellmeier"] is None:
        out.append("n is the LUT value at 1550 nm (no Sellmeier fit for this material)")
    if abs(wavelength - _REF_WAVELENGTH) > 0.3e-6:
        out.append("WARNING: dn/dT and β_TPA are tabulated at 1550 nm; they change with wavelength, strongly near the band edge")
    return out


def material_properties(material: str, wavelength=1.55e-6, al_fraction=0.20) -> Result:
    """LUT row for a material: index, dn/dT (nominal and range), thermal conductivity (nominal and range),
    ρ, c_p, thermal diffusivity D = k/(ρ c_p), expansion, band gap, TPA and n2."""
    require_positive(wavelength=wavelength)
    e = entry(material, al_fraction)
    nan = float("nan")
    v = {
        "n": index_at(material, wavelength, al_fraction),
        "dn_dT": e["dn_dT"], "dn_dT_min": e["dn_dT_range"][0], "dn_dT_max": e["dn_dT_range"][1],
        "k": e["k"], "k_min": e["k_range"][0], "k_max": e["k_range"][1],
        "rho": e["rho"], "cp": e["cp"], "diffusivity": e["k"] / (e["rho"] * e["cp"]),
        "alpha_L": e["alpha_L"], "Eg": e["Eg"],
        "beta_tpa": nan if e["beta_tpa"] is None else e["beta_tpa"],
        "n2": nan if e["n2"] is None else e["n2"],
    }
    u = {"n": "", "dn_dT": "1/K", "dn_dT_min": "1/K", "dn_dT_max": "1/K", "k": "W/(m K)", "k_min": "W/(m K)", "k_max": "W/(m K)",
         "rho": "kg/m^3", "cp": "J/(kg K)", "diffusivity": "m^2/s", "alpha_L": "1/K", "Eg": "eV", "beta_tpa": "m/W", "n2": "m^2/W"}
    return Result(values=v, units=u, assumptions=_notes(material, e, wavelength) + ["Room temperature (about 300 K); NaN = not tabulated"])


def lut_rows(wavelength=1.55e-6) -> list[dict]:
    """The whole LUT as plain rows (for the CSV/Markdown export in tools/make_thermo_lut.py)."""
    rows = []
    for m in MATERIALS:
        e = entry(m)
        rows.append({
            "material": m, "name": e["name"], "n_1550": round(index_at(m, wavelength), 4),
            "dn_dT": e["dn_dT"], "dn_dT_min": e["dn_dT_range"][0], "dn_dT_max": e["dn_dT_range"][1],
            "k": round(e["k"], 3), "k_min": round(e["k_range"][0], 3), "k_max": round(e["k_range"][1], 3),
            "rho": round(e["rho"], 1), "cp": round(e["cp"], 1), "diffusivity": float(f"{e['k'] / (e['rho'] * e['cp']):.3e}"),
            "alpha_L": e["alpha_L"], "Eg": round(e["Eg"], 3), "beta_tpa": e["beta_tpa"], "n2": e["n2"],
            "confidence": e["confidence"], "source": e["source"],
        })
    return rows


def index_change(material: str, delta_T, wavelength=1.55e-6, al_fraction=0.20) -> Result:
    """Δn = dn/dT ΔT for a material (nominal and the LUT range), and the phase shift per metre 2π Δn / λ."""
    require_positive(wavelength=wavelength)
    e = entry(material, al_fraction)
    dT = np.asarray(delta_T, dtype=float)
    dn = e["dn_dT"] * dT
    a, b = e["dn_dT_range"][0] * dT, e["dn_dT_range"][1] * dT
    with np.errstate(divide="ignore"):
        l_pi = np.abs(wavelength / (2 * dn))
    return Result(
        values={"delta_n": dn, "delta_n_min": np.minimum(a, b), "delta_n_max": np.maximum(a, b),
                "phase_per_length": 2 * np.pi * dn / wavelength, "length_for_pi": l_pi},
        units={"delta_n": "", "delta_n_min": "", "delta_n_max": "", "phase_per_length": "rad/m", "length_for_pi": "m"},
        assumptions=_notes(material, e, wavelength) + ["Linear in ΔT (first-order dn/dT); strain-optic and expansion terms not included"],
    )


def slab_thermal_shift(wavelength=1.55e-6, core="si", substrate="sio2", cladding="sio2", thickness=220e-9,
                       polarization="TE", substrate_alpha_L=None, al_fraction=0.20) -> Result:
    """Thermal shift of a three-layer slab mode: dn_eff/dT, the share of it from each layer (Γ_i dn_i/dT with
    Γ_i = ∂n_eff/∂n_i), group index and the resonance drift dλ/dT = λ (dn_eff/dT + n_eff α_sub) / n_g.

    α_sub: expansion of the substrate that sets the cavity length (defaults to the substrate's LUT value)."""
    require_positive(wavelength=wavelength, thickness=thickness)
    require_choice("polarization", polarization, ("TE", "TM"))
    mats = (substrate, core, cladding)
    n = [index_at(m, wavelength, al_fraction) for m in mats]
    dndT = [entry(m, al_fraction)["dn_dT"] for m in mats]

    def neff(idx, lam=wavelength):
        return neff_three_layer(lam, idx[0], idx[1], idx[2], thickness, polarization, 0)

    n0 = neff(n)
    if not np.isfinite(n0):
        raise ValueError("no guided mode for this slab")
    h = 1e-4
    gam = []
    for i in range(3):
        up, dn = list(n), list(n)
        up[i] += h
        dn[i] -= h
        gam.append((neff(up) - neff(dn)) / (2 * h))
    contrib = [g * d for g, d in zip(gam, dndT)]
    dneff_dT = float(sum(contrib))
    dl = wavelength * 1e-3
    np_ = neff([index_at(m, wavelength + dl, al_fraction) for m in mats], wavelength + dl)
    nm_ = neff([index_at(m, wavelength - dl, al_fraction) for m in mats], wavelength - dl)
    ng = n0 - wavelength * (np_ - nm_) / (2 * dl)
    a_sub = entry(substrate, al_fraction)["alpha_L"] if substrate_alpha_L is None else float(substrate_alpha_L)
    dlam = wavelength * (dneff_dT + n0 * a_sub) / ng
    return Result(
        values={"neff": n0, "n_group": ng, "dneff_dT": dneff_dT,
                "Gamma_sub": gam[0], "Gamma_core": gam[1], "Gamma_clad": gam[2],
                "contrib_sub": contrib[0], "contrib_core": contrib[1], "contrib_clad": contrib[2],
                "dlambda_dT": dlam},
        units={"neff": "", "n_group": "", "dneff_dT": "1/K", "Gamma_sub": "", "Gamma_core": "", "Gamma_clad": "",
               "contrib_sub": "1/K", "contrib_core": "1/K", "contrib_clad": "1/K", "dlambda_dT": "m/K"},
        assumptions=[
            "Fundamental three-layer slab mode (slab_waveguide engine); a channel waveguide has extra lateral cladding share",
            "Uniform temperature over the mode; Γ_i = ∂n_eff/∂n_i by central differences",
            "Resonance drift includes substrate expansion (n_eff α_sub), not the stress-optic effect",
        ] + [f"{m}: confidence {entry(m, al_fraction)['confidence']}" for m in mats],
    )


def resonance_shift(wavelength, dneff_dT, n_group, n_eff=0.0, alpha_L=0.0, delta_T=1.0) -> Result:
    """Ring/Fabry-Perot/grating resonance drift: dλ/dT = λ (dn_eff/dT + n_eff α_L) / n_g, and Δλ at ΔT."""
    require_positive(wavelength=wavelength, n_group=n_group)
    d = wavelength * (dneff_dT + n_eff * alpha_L) / n_group
    return Result(values={"dlambda_dT": d, "delta_lambda": d * delta_T, "dnu_dT": -C0 / wavelength**2 * d},
                  units={"dlambda_dT": "m/K", "delta_lambda": "m", "dnu_dT": "Hz/K"},
                  assumptions=["First order in ΔT; n_g includes waveguide dispersion"])


def db_per_cm_to_per_m(loss_db_per_cm):
    """Power attenuation coefficient α (1/m) from a loss in dB/cm."""
    return np.asarray(loss_db_per_cm, dtype=float) * 100.0 * DB_TO_NEPER


def heat_coefficients(wavelength, a_eff, alpha_abs=0.0, beta_tpa=0.0, carrier_lifetime=0.0, sigma_fca=0.0):
    """(c1, c2, c3) with q' = c1 P + c2 P² + c3 P³ (W/m): linear absorption, TPA, free-carrier absorption."""
    hnu = H_PLANCK * C0 / wavelength
    c1 = float(alpha_abs)
    c2 = float(beta_tpa) / a_eff
    c3 = float(sigma_fca) * float(carrier_lifetime) * float(beta_tpa) / (2 * hnu * a_eff**2)
    return c1, c2, c3


def si_free_carrier_index(carrier_density):
    """Free-carrier (plasma) dispersion of Si at 1550 nm, equal electron and hole densities N (1/m^3):
    Δn = -(8.8e-22 N + 8.5e-18 N^0.8) with N in cm^-3 (Soref & Bennett 1987). Opposite in sign to the thermal Δn."""
    N = np.asarray(carrier_density, dtype=float) * 1e-6
    return -(8.8e-22 * N + 8.5e-18 * np.power(np.maximum(N, 0.0), 0.8))


def absorbed_heat(power, wavelength=1.55e-6, a_eff=0.1e-12, loss_abs_db_per_cm=0.0, beta_tpa=0.0,
                  carrier_lifetime=0.0, sigma_fca=0.0) -> Result:
    """Heat deposited per unit length q' (W/m) by a guided power P: linear absorption, TPA and FCA.

    loss_abs_db_per_cm: only the absorbing part of the propagation loss (scattering leaves the chip);
    A_eff: TPA effective area of the mode in the absorbing material; carrier_lifetime τ and σ_FCA
    for semiconductors (Si: σ = 1.45e-21 m², τ 0.5-5 ns in SOI wires, ~10 ps with a reverse-biased p-i-n)."""
    require_positive(wavelength=wavelength, a_eff=a_eff)
    require_nonnegative(power=power, loss_abs_db_per_cm=loss_abs_db_per_cm, beta_tpa=beta_tpa,
                        carrier_lifetime=carrier_lifetime, sigma_fca=sigma_fca)
    P = np.asarray(power, dtype=float)
    a = db_per_cm_to_per_m(loss_abs_db_per_cm)
    c1, c2, c3 = heat_coefficients(wavelength, a_eff, a, beta_tpa, carrier_lifetime, sigma_fca)
    hnu = H_PLANCK * C0 / wavelength
    N = carrier_lifetime * beta_tpa * (P / a_eff) ** 2 / (2 * hnu)
    return Result(
        values={"q_linear": c1 * P, "q_tpa": c2 * P**2, "q_fca": c3 * P**3, "q_total": c1 * P + c2 * P**2 + c3 * P**3,
                "carrier_density": N, "alpha_fca": sigma_fca * N, "dn_fc_si": si_free_carrier_index(N)},
        units={"q_linear": "W/m", "q_tpa": "W/m", "q_fca": "W/m", "q_total": "W/m", "carrier_density": "1/m^3", "alpha_fca": "1/m",
               "dn_fc_si": ""},
        assumptions=["All absorbed power becomes heat (TPA carriers recombine non-radiatively)",
                     "Undepleted power (local value at one point of the waveguide); carriers from TPA only (no doping, no linear defect absorption)",
                     "dn_fc_si: Si plasma dispersion of the TPA carriers (Soref & Bennett); not meaningful for other materials"],
    )


def amplifier_heat_fraction(pump_wavelength=0.98e-6, emission_wavelength=1.532e-6, quantum_efficiency=1.0) -> Result:
    """Fraction of absorbed pump power that ends up as heat: η_heat = 1 - η_q λ_p / λ_em.

    η_q: radiative quantum efficiency (fluorescence + stimulated emission out of all decays); concentration
    quenching, upconversion and excited-state absorption lower it. Er: 980 nm pumping heats about 12x more per
    absorbed watt than in-band 1480 nm pumping."""
    require_positive(pump_wavelength=pump_wavelength, emission_wavelength=emission_wavelength)
    require_range("quantum_efficiency", quantum_efficiency, 0.0, 1.0)
    qd = 1 - pump_wavelength / emission_wavelength
    eta = 1 - quantum_efficiency * pump_wavelength / emission_wavelength
    return Result(values={"quantum_defect": qd, "eta_heat": eta}, units={"quantum_defect": "", "eta_heat": ""},
                  assumptions=["Every absorbed pump photon is either re-emitted at λ_em (probability η_q) or turned into heat"])


def strip_thermal_resistance(width=0.5e-6, height=0.22e-6, box_thickness=2e-6, box_material="sio2",
                             substrate_thickness=500e-6, substrate_material="si", clad_material="sio2",
                             slab_thickness=0.0, slab_material="si", box_k=None, substrate_k=None, slab_k=None) -> Result:
    """Quick closed-form estimate of the thermal resistance per unit length R' (K m / W) from a uniformly heated
    waveguide (width w, height h) on a buried layer (BOX, thickness t) to a heat sink at the substrate bottom.

    Waveguide embedded in a cladding (clad_material not air), no film:  R_box = acosh(d/r) / (2π k_box),
        r = (w + 2h)/4 (equivalent cylinder), d = t + h/2 (its height above the substrate, which acts as an isothermal plane).
    Air cladding, or a film (slab) under the ridge: R_box = acosh(d/r) / (π k_box), r = w_eff/4, d = t + t_slab,
        w_eff = w + 2 L_f, L_f = sqrt(k_slab t_slab t / (2 k_box)) (lateral spreading in the film).
    Substrate: R_sub = ln(8H / (π w'')) / (π k_sub), w'' = w_eff + 2t.
    Within ±30 % of waveguide_thermal for SOI, SiN and air-clad TFLN stacks (tests); an oxide-clad ridge on a film
    comes out up to 40 % high. Use the solver for design."""
    require_positive(width=width, height=height, box_thickness=box_thickness, substrate_thickness=substrate_thickness)
    require_nonnegative(slab_thickness=slab_thickness)
    require_choice("clad_material", clad_material, MATERIALS)
    kb = entry(box_material)["k"] if box_k is None else float(box_k)
    ks = entry(substrate_material)["k"] if substrate_k is None else float(substrate_k)
    kf = entry(slab_material)["k"] if slab_k is None else float(slab_k)
    require_positive(box_k=kb, substrate_k=ks, slab_k=kf)
    t = box_thickness
    if slab_thickness > 0 or clad_material == "air":
        L_f = math.sqrt(kf * slab_thickness * t / (2 * kb))
        w_eff = width + 2 * L_f
        d, r, m = t + slab_thickness, w_eff / 4, 1
    else:
        w_eff = width
        d, r, m = t + height / 2, (width + 2 * height) / 4, 2
    R_box = math.acosh(max(d / r, 1.0)) / (m * math.pi * kb)
    w2 = w_eff + 2 * t
    arg = 8 * substrate_thickness / (math.pi * w2)
    R_sub = math.log(arg) / (math.pi * ks) if arg > math.e else substrate_thickness / (ks * w2)
    return Result(values={"R_th": R_box + R_sub, "R_box": R_box, "R_sub": R_sub, "w_eff": w_eff},
                  units={"R_th": "K m/W", "R_box": "K m/W", "R_sub": "K m/W", "w_eff": "m"},
                  assumptions=["2D (long waveguide, uniform along z), steady state, isothermal substrate bottom",
                               "Chip width much larger than the substrate thickness; top surface adiabatic",
                               "ΔT [K] for 1 mW of heat per mm of waveguide equals R' in K m/W"])


def _max_power(c, budget):
    """Largest P >= 0 with c1 P + c2 P² + c3 P³ <= budget (monotone for c >= 0)."""
    c1, c2, c3 = c
    if budget <= 0:
        return 0.0
    if c1 == c2 == c3 == 0:
        return math.inf
    lo, hi = 0.0, 1.0
    while c1 * hi + c2 * hi**2 + c3 * hi**3 < budget:
        hi *= 2
        if hi > 1e12:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c1 * mid + c2 * mid**2 + c3 * mid**3 < budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pump_budget(R_th=10.0, dneff_dT=1.8e-4, power=0.1, wavelength=1.55e-6, a_eff=0.1e-12, loss_abs_db_per_cm=0.0,
                beta_tpa=0.0, carrier_lifetime=0.0, sigma_fca=0.0, pump_absorption_db_per_cm=0.0, eta_heat=0.0,
                dT_max=10.0, dneff_max=1e-3) -> Result:
    """Temperature rise and index shift at a guided power P, and the largest power that stays within ΔT_max and Δn_eff,max.

    Heat per length at the hottest point (the input facet, undepleted): q' = (α_abs + η_heat α_p) P + β P²/A_eff + FCA.
    ΔT = R' q', Δn_eff = dn_eff/dT ΔT. α_p: pump absorption of the gain medium (dB/cm, small-signal; bleaching lowers it,
    so this is the worst case), η_heat from amplifier_heat_fraction. R' from strip_thermal_resistance or waveguide_thermal."""
    require_positive(R_th=R_th, wavelength=wavelength, a_eff=a_eff, dT_max=dT_max, dneff_max=dneff_max)
    require_nonnegative(power=power, pump_absorption_db_per_cm=pump_absorption_db_per_cm)
    require_range("eta_heat", eta_heat, 0.0, 1.0)
    a = float(db_per_cm_to_per_m(loss_abs_db_per_cm) + eta_heat * db_per_cm_to_per_m(pump_absorption_db_per_cm))
    c = heat_coefficients(wavelength, a_eff, a, beta_tpa, carrier_lifetime, sigma_fca)
    P = float(power)
    q = c[0] * P + c[1] * P**2 + c[2] * P**3
    dT = R_th * q
    budget_T = dT_max / R_th
    budget_n = dneff_max / (abs(dneff_dT) * R_th) if dneff_dT != 0 else math.inf
    P_T = _max_power(c, budget_T)
    P_n = _max_power(c, budget_n)
    return Result(
        values={"q": q, "dT": dT, "dneff": dneff_dT * dT, "P_max_dT": P_T, "P_max_dn": P_n, "P_max": min(P_T, P_n),
                "q_linear": c[0] * P, "q_tpa": c[1] * P**2, "q_fca": c[2] * P**3},
        units={"q": "W/m", "dT": "K", "dneff": "", "P_max_dT": "W", "P_max_dn": "W", "P_max": "W",
               "q_linear": "W/m", "q_tpa": "W/m", "q_fca": "W/m"},
        assumptions=["Steady state, local (input-facet) heating with undepleted power; heat flows only through the cross-section (2D)",
                     "Linear thermo-optic response; no thermal runaway feedback through temperature-dependent absorption",
                     "P_max = inf when nothing absorbs"],
    )
