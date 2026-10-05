"""cmpc_sim — effective path length and evanescent-volume estimates for chip-scale chaotic multipass cells (CMPCs)
in free-standing high-index membranes.

All physics lives in misc-applets/math-engines; this module keeps the v0.2 script API on top of it:
    engines.membrane_mode   slab-mode field integrals -> Γ, penetration depth, group index, surface-field weight
    engines.bragg_grating   oblique-incidence transfer matrix, etched-trench DBR (TrenchDBR), two-wavelength orders
    engines.billiard_cell   stadium / segmented-cell ray tracer with ports, re-weighting of one ray table
    engines.gaussian_beam   Rayleigh range (injection / diffraction check)

SI units everywhere unless a name ends in _um, _nm, _cm. Read ASSUMPTIONS at the bottom before quoting numbers.
"""
from __future__ import annotations

import numpy as np

import _engines  # noqa: F401  (puts math-engines on sys.path)
from engines.billiard_cell.engine import (CellResult, RayTable, SegmentedCell, Stadium, dB_per_cm_to_alpha,  # noqa: F401
                                          evaluate, mean_field_estimate, occupancy_map, poincare, trace_path, trace_rays)
from engines.bragg_grating.engine import LATERAL_POL, TrenchDBR, cascaded_R, stack_R_oblique, stack_reflectance  # noqa: F401
from engines.bragg_grating.engine import double_resonant_orders as _dro
from engines.gaussian_beam.engine import beam_parameters
from engines.membrane_mode import engine as _mm
from engines.membrane_mode.engine import N_AIR_NUMBER_DENSITY, MembraneMode  # noqa: F401

C0 = 299792458.0
DBR = TrenchDBR

# name -> (materials-engine key | constant index, note)
MATERIALS = {
    "Si": ("si", "Salzberg-Villa fit via engines.materials (real part, valid >~1.2 um)"),
    "SiNx": ("si3n4", "stoichiometric LPCVD Si3N4 via engines.materials (n~1.996); Si-rich SiNx is higher: use n_const"),
    "Al2O3": (1.65, "ASSUMED constant n=1.65 (amorphous ALD/sputtered); engines only has crystalline sapphire (n~1.75)"),
}


def _material_args(name, n_const=None):
    key = MATERIALS[name][0]
    if n_const is not None:
        return "constant", float(n_const)
    if isinstance(key, float):
        return "constant", key
    return key, None


def n_material(name: str, lam: float, n_const: float | None = None) -> float:
    mat, n = _material_args(name, n_const)
    return _mm.core_index(mat, lam, n)


def membrane_mode(material: str, thickness: float, wavelength: float = 1.55e-6, pol: str = "TE", order: int = 0,
                  n_clad: float = 1.0, n_const: float | None = None, with_group_index: bool = True) -> MembraneMode:
    """Symmetric (n_clad | membrane | n_clad) slab mode; material is a key of MATERIALS."""
    mat, n = _material_args(material, n_const)
    m = _mm.solve_mode(wavelength, thickness, n, n_clad, pol, order, mat, with_group_index)
    m.material = material
    return m


def roughness_scaled_alpha(mode, mode_ref, alpha_ref):
    """Background loss if surface-roughness scattering dominates: α ∝ E_edge2, scaled from a reference membrane."""
    return _mm.roughness_scaled_alpha(mode, mode_ref, alpha_ref, index_contrast=False)


def evanescent_volume(mode, area, sides=2):
    r = _mm.evanescent_volume(mode.z_p, area, sides)
    return dict(V_ev=r["V_ev"], V_ev_mm3=r["V_ev"] * 1e9, molecules_per_ppb=r["molecules_per_ppb"])


def field_profile(mode, z_extent_factor=4.0, n=801):
    return _mm.field_profile(mode, z_extent_factor, n)


def double_resonant_orders(lam1, lam2, n_t1, n_t2, max_order=31, n_gap=1.0, top=5):
    out = _dro(lam1, lam2, n_t1, n_t2, max_order, n_gap, top)
    for c in out:
        c["d_tooth_nm"], c["d_gap_nm"] = c.pop("d_tooth") * 1e9, c.pop("d_gap") * 1e9
    return out


def uniform_average(Rfun, n=513):
    """⟨R⟩ over sin χ uniform in [0, 1] (the invariant measure a chaotic cell samples)."""
    sg = np.linspace(0, 1, n)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(Rfun(sg), sg))


def rayleigh_check(w0: float, lam: float, n: float, distance: float):
    """(z_R, distance): is in-plane beam diffraction negligible over `distance`?"""
    return beam_parameters(wavelength=lam, w0=w0, n=n)["z_R"], distance


# --------------------------------------------------------------------------- self-test
def selftest(verbose=True):
    """Quick end-to-end check of the engines as this script uses them (the engines' own pytest suites go further)."""
    ok = True
    lam = np.linspace(1.45e-6, 1.65e-6, 41)
    mine = stack_R_oblique(lam, np.zeros_like(lam), 1.0, [(2.8, 138e-9), (1.0, 387.5e-9)] * 6, 1.0)
    e1 = float(np.max(np.abs(mine - stack_reflectance(lam, 2.8, 1.0, 138e-9, 387.5e-9, 6)["R"])))
    ok &= e1 < 1e-9
    g_thin = membrane_mode("Si", 5e-9).Gamma
    te, tm = membrane_mode("Si", 220e-9, pol="TE").Gamma, membrane_mode("Si", 220e-9, pol="TM").Gamma
    ok &= g_thin > 0.9 and tm > te
    circ = trace_rays(Stadium(5e-3, 0.0), 100e-6, 400, 200, seed=3)
    longlived = np.nonzero((circ.exit_idx < 0) | (circ.exit_idx >= 6))[0][:50]
    spread = float(np.abs(circ.sinchi[longlived, :6] - circ.sinchi[longlived, :1]).max())
    ok &= spread < 1e-5
    st = Stadium(5e-3, 5e-3)
    mc = evaluate(trace_rays(st, 150e-6, 3000, 2500, seed=2), lambda s: np.full_like(s, 0.999))
    mf = mean_field_estimate(st, 150e-6, 0.999)
    e4 = abs(mc.L_mean - mf["L_mean"]) / mf["L_mean"]
    ok &= e4 < 0.25
    n1 = 2.8
    rp = float(stack_R_oblique(1.55e-6, np.sin(np.arctan(1 / n1)), n1, [], 1.0, "p"))
    ok &= rp < 1e-12
    if verbose:
        print(f"[1] oblique TMM vs Abeles at normal incidence max|dR| = {e1:.2e}")
        print(f"[2] Gamma(5 nm Si) = {g_thin:.3f}, Gamma_TE(220 nm) = {te:.3f}, Gamma_TM(220 nm) = {tm:.3f}")
        print(f"[3] circle sin(chi) drift = {spread:.1e}; stadium <L> MC {mc.L_mean*100:.1f} cm vs mean field {mf['L_mean']*100:.1f} cm (rel {e4:.2f})")
        print(f"[4] Brewster p-pol R = {rp:.1e}")
    return bool(ok)


ASSUMPTIONS = """
v0.3 assumptions (the ones that move the answer most are marked **):
- Polarisation: slab TE (E in membrane plane) is p-pol and slab TM (E_z) is s-pol on a vertical trench mirror.
- Cell is 2D: the guided mode propagates in-plane as a ray; diffraction enters only via the launch spread lambda/(n_eff w).
- Mirrors: in-plane DBR of etched air gaps and membrane teeth, 1D effective-index transfer matrix at oblique incidence (TIR and
  frustrated TIR included). ** 3D radiation loss at the slot, TE<->TM conversion, sidewall roughness and finite etch depth are lumped
  into bounce_loss (3e-4, ASSUMED) and are the first thing the full-wave numerics must replace.
- ** Background loss alpha_bg is an input (0.1 dB/cm at Si 220 nm TE, ASSUMED); other membranes scale with the surface-field proxy
  E_edge2 x (n^2-1)^2, which is crude (TM Si 250 nm comes out about 3x TE 220 nm).
- Gas absorption is weak: signal ~ alpha_gas * Gamma * <L>; Gamma from the field integrals of the symmetric slab, validated by quadrature.
- Al2O3: constant n = 1.65 (assumed). Si Sellmeier is a real-part fit valid above ~1.2 um. SiNx is stoichiometric Si3N4.
- Ports are ideal windows (no taper loss); single-mode injection with spread lambda/(n_eff w).
- Rays still trapped after n_bounce / max_path are dropped from the path statistics (RayTable.trapped_fraction).
- Coherence model: random-phase statistics on discrete paths (facet itineraries), reflection phases neglected, dn_eff/dT = 1.5e-4 /K assumed.
  A prediction model only; not validated against wave simulations.
- Gas lines: ILLUSTRATIVE hand-entered parameters unless a HITRAN export exists (math-engines/tools/fetch_hitran.py). Do not quote.
"""

if __name__ == "__main__":
    print("selftest passed" if selftest() else "selftest FAILED")
