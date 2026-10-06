"""cmpc_sim — effective path length and evanescent-volume estimates for chip-scale
chaotic multipass cells (CMPCs) in free-standing high-index membranes.

Built on the Python reference engines in mikkojhuttunen/misc-applets/math-engines:
    engines.materials          Sellmeier n(lambda), Si / Si3N4 / sapphire
    engines.slab_waveguide     neff of the (air | membrane | air) slab mode
    engines.bragg_grating      normal-incidence Abeles stack (used to validate the
                               oblique-incidence transfer matrix written here)
    engines.gaussian_beam      Rayleigh range (injection / diffraction check)

New physics written here (not available in the engines):
    * slab-mode field integrals -> evanescent power fraction Gamma, penetration depth
    * oblique-incidence DBR transfer matrix  R_m(sin chi, lambda)
    * 2D billiard ray tracer (circle / stadium) with input and output ports
    * re-weighting of one geometric ray table for any mirror / loss / port setting

SI units everywhere unless a name ends in _um, _nm, _cm.
Model assumptions are listed in ASSUMPTIONS at the bottom; read them before quoting numbers.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- engines
def _locate_engines() -> Path:
    """Find the Python math-engines of misc-applets. This package lives in misc-applets/cmpc-sim/, so the engines are normally
    in ../math-engines; override with $MISC_APPLETS=/path/to/misc-applets (the folder that contains math-engines/)."""
    root = Path(__file__).resolve().parents[1]                     # .../misc-applets/cmpc-sim
    cands = [os.environ.get("MISC_APPLETS"), root.parent, "./misc-applets", "../misc-applets", Path.home() / "misc-applets"]
    for c in cands:
        if c and (Path(c) / "math-engines" / "engines").is_dir():
            return Path(c) / "math-engines"
    raise ImportError("math-engines not found: cmpc-sim must sit inside a misc-applets checkout (next to math-engines/), "
                      "or set MISC_APPLETS=/path/to/misc-applets")


_ENG = _locate_engines()
sys.path.insert(0, str(_ENG))
from engines.materials import engine as _mat                      # noqa: E402
from engines.slab_waveguide.engine import neff_three_layer        # noqa: E402
from engines.bragg_grating.engine import stack_reflectance        # noqa: E402
from engines.gaussian_beam.engine import beam_parameters          # noqa: E402

C0 = 299792458.0
N_AIR_NUMBER_DENSITY = 2.5e25   # m^-3, air at ~293 K, 1 atm


# --------------------------------------------------------------------------- materials
# name -> (engine key | constant index, note)
MATERIALS = {
    "Si":    ("si",    "Salzberg-Villa fit via engines.materials (real part, valid >~1.2 um)"),
    "SiNx":  ("si3n4", "stoichiometric LPCVD Si3N4 via engines.materials (n~1.996); Si-rich SiNx is higher: use n_const"),
    "Ge":    ("ge",    "Barnes & Piltch (1979) Sellmeier fit, 2-14 um, room temperature (coefficients entered from memory: verify before design use)"),
    "Al2O3": (1.65,    "ASSUMED constant n=1.65 (amorphous ALD/sputtered); engines only has crystalline sapphire (n~1.75)"),
}


def n_material(name: str, lam: float, n_const: float | None = None) -> float:
    if n_const is not None:
        return float(n_const)
    key = MATERIALS[name][0]
    if isinstance(key, float):
        return key
    if key == "ge":
        l2 = (lam * 1e6) ** 2
        return float(np.sqrt(9.28156 + 6.72880 * l2 / (l2 - 0.44105) + 0.21307 * l2 / (l2 - 3870.1)))
    return float(_mat.index(key, lam))


# --------------------------------------------------------------------------- slab mode
@dataclass
class MembraneMode:
    material: str
    thickness: float
    wavelength: float
    pol: str
    order: int
    n_core: float
    neff: float
    n_group: float
    gamma_ev: float      # cladding decay constant [1/m]  (field ~ exp(-gamma z))
    Gamma: float         # absorption-coupling factor: alpha_wg = Gamma * alpha_gas
    f_clad_E: float      # fraction of |E|^2 integral outside the membrane (both sides)
    z_p: float           # 1/e intensity penetration depth per side = 1/(2 gamma)
    E_edge2: float = np.nan   # |field at the membrane surface|^2 / int |field|^2 dx  [1/m]  (surface-scattering weight)

    @property
    def guided(self) -> bool:
        return np.isfinite(self.neff)


def _surface_weight(pol, even, kap, gam, beta, k, nf, nc, d, core, clad, edge):
    """|E|^2 on the core side of the membrane surface per unit guided power (common factors dropped), 1/m.
    Crude proxy for surface-roughness scattering; comparable between thicknesses, not an absolute loss."""
    if pol == "TE":
        return edge**2 / (beta / k * (core + clad))
    fp = kap * (np.sin(kap * d / 2) if even else np.cos(kap * d / 2))
    return (fp**2 + beta**2 * edge**2) / (k * beta * nf**4 * (core / nf**2 + clad / nc**2))


def membrane_mode(material: str, thickness: float, wavelength: float = 1.55e-6,
                  pol: str = "TE", order: int = 0, n_clad: float = 1.0,
                  n_const: float | None = None, with_group_index: bool = True) -> MembraneMode:
    """Symmetric (n_clad | membrane | n_clad) slab mode and its gas-sensing quantities.

    Gamma follows from dissipation in the cladding gas divided by the Poynting flux:
        TE:  Gamma = (n_c/neff) * int_clad |E|^2 / int |E|^2
        TM:  Gamma = n_c int_clad (b^2 H^2 + H'^2)/n_c^4  /  ( k b int H^2/n^2 )
    so that the modal absorption is alpha_wg = Gamma * alpha_gas (bulk gas absorption coefficient).
    """
    lam, d = wavelength, thickness
    nf = n_material(material, lam, n_const)
    neff = neff_three_layer(lam, n_clad, nf, n_clad, d, pol, order)
    if not np.isfinite(neff):
        return MembraneMode(material, d, lam, pol, order, nf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    k = 2 * np.pi / lam
    kap = k * np.sqrt(nf**2 - neff**2)
    gam = k * np.sqrt(neff**2 - n_clad**2)
    beta = k * neff
    even = (order % 2 == 0)
    edge = np.cos(kap * d / 2) if even else np.sin(kap * d / 2)
    core = d / 2 + (1 if even else -1) * np.sin(kap * d) / (2 * kap)      # int_core f^2 dx
    clad = edge**2 / gam                                                  # both sides
    if pol == "TE":
        f_clad = clad / (core + clad)
        Gamma = (n_clad / neff) * f_clad
    else:
        num = n_clad * edge**2 * (beta**2 + gam**2) / gam / n_clad**4
        den = k * beta * (core / nf**2 + clad / n_clad**2)
        Gamma = num / den
        f_clad = clad / (core + clad)
    n_g = np.nan
    if with_group_index:
        h = 1e-3 * lam
        vals = []
        for l in (lam - h, lam + h):
            vals.append(neff_three_layer(l, n_clad, n_material(material, l, n_const), n_clad, d, pol, order))
        n_g = neff - lam * (vals[1] - vals[0]) / (2 * h)
    return MembraneMode(material, d, lam, pol, order, nf, float(neff), float(n_g), float(gam),
                        float(Gamma), float(f_clad), float(1 / (2 * gam)), float(_surface_weight(pol, even, kap, gam, beta, k, nf, n_clad, d, core, clad, edge)))


def roughness_scaled_alpha(mode: MembraneMode, mode_ref: MembraneMode, alpha_ref: float) -> float:
    """Background loss if surface-roughness scattering dominates: alpha ~ |E_surface|^2 per unit power
    E_edge2, scaled from a reference membrane. Thin membranes scatter more."""
    return alpha_ref * mode.E_edge2 / mode_ref.E_edge2


def evanescent_volume(mode: MembraneMode, area: float, sides: int = 2) -> dict:
    """Gas volume inside the 1/e intensity penetration depth above (and below) a cell of footprint `area`,
    and the number of analyte molecules in it per ppb."""
    V = sides * area * mode.z_p
    return dict(V_ev=V, V_ev_mm3=V * 1e9, molecules_per_ppb=N_AIR_NUMBER_DENSITY * V * 1e-9)


def field_profile(mode: MembraneMode, z_extent_factor: float = 4.0, n: int = 801):
    """|E_y|^2 (TE) or |H_y|^2 (TM) vs x across the membrane (x=0 at centre); for plotting."""
    lam, d = mode.wavelength, mode.thickness
    k = 2 * np.pi / lam
    kap = k * np.sqrt(mode.n_core**2 - mode.neff**2)
    gam = mode.gamma_ev
    half = d / 2
    xmax = half + z_extent_factor / gam
    x = np.linspace(-xmax, xmax, n)
    even = mode.order % 2 == 0
    f = np.where(np.abs(x) <= half, (np.cos(kap * x) if even else np.sin(kap * x)),
                 (np.cos(kap * half) if even else np.sign(x) * np.sin(kap * half)) * np.exp(-gam * (np.abs(x) - half)))
    return x, f**2


# --------------------------------------------------------------------------- DBR (oblique TMM)
def _norm(M):
    s = np.max(np.abs(M), axis=(0, 1), keepdims=True)
    return M / np.where(s == 0, 1, s)


def stack_R_oblique(lam, sin_in, n_in, layers, n_out, pol="s"):
    """Reflectance of a layer stack at oblique incidence, E perpendicular (pol='s') or parallel
    (pol='p') to the plane of incidence. Here the plane of incidence is the MEMBRANE plane, so

        slab TM mode (dominant E_z, perpendicular to the membrane)  ->  pol = 's'
        slab TE mode (E in the membrane plane)                      ->  pol = 'p'   (Brewster behaviour)

    layers : [(n, d), ...] from the input medium; complex n allowed. sin_in = sin(theta) in the input medium.
    Evanescent layers handled (branch Im(cos) >= 0), so TIR and frustrated TIR are included. Reduces to
    engines.bragg_grating.stack_reflectance at normal incidence (checked in selftest()).
    """
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    lam = np.asarray(lam, dtype=float)
    s = n_in * np.minimum(np.asarray(sin_in, dtype=float), 1 - 1e-9)   # avoid exact grazing
    shape = np.broadcast(lam, s).shape
    k = 2 * np.pi / lam

    def cosine(n):
        return np.sqrt(1 - (s / n) ** 2 + 1e-18j)

    def adm(n):                       # tilted optical admittance
        c = cosine(n)
        return n * c if pol == "s" else n / c

    M = np.zeros((2, 2) + shape, dtype=complex)
    M[0, 0] = M[1, 1] = 1
    for n, d in layers:
        c = cosine(n)
        q = adm(n)
        dl = k * n * d * c
        cs, sn = np.cos(dl), np.sin(dl)
        L = np.zeros_like(M)
        L[0, 0], L[0, 1], L[1, 0], L[1, 1] = cs, -1j * sn / q, -1j * q * sn, cs
        M = _norm(np.einsum("ij...,jk...->ik...", M, L))
    q_in, q_out = adm(n_in), adm(n_out)
    A, B, Cc, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    x, y = q_in * (A + B * q_out), Cc + D * q_out
    r = (x - y) / (x + y)
    return np.abs(r) ** 2


LATERAL_POL = {"TE": "p", "TM": "s"}     # slab-mode label -> polarisation w.r.t. a vertical trench mirror


@dataclass
class DBR:
    """In-plane DBR of a membrane cell: cavity(membrane) | [air gap, membrane tooth] x N | air.
    d_gap, d_tooth are odd multiples of a quarter wavelength (order m_gap, m_tooth) at lam_design."""
    n_tooth: float          # effective index of the membrane slab mode at the design wavelength
    lam_design: float = 1.55e-6
    N: int = 8
    m_gap: int = 1
    m_tooth: int = 3
    n_gap: float = 1.0
    bounce_loss: float = 3e-4    # extra per-bounce scattering/absorption loss (roughness)
    slab_pol: str = "TM"         # polarisation of the guided slab mode that hits the mirror

    @property
    def d_gap(self): return self.m_gap * self.lam_design / (4 * self.n_gap)

    @property
    def d_tooth(self): return self.m_tooth * self.lam_design / (4 * self.n_tooth)

    def layers(self):
        return [(self.n_gap, self.d_gap), (self.n_tooth, self.d_tooth)] * self.N

    def R(self, lam, sin_chi, n_tooth_at_lam=None):
        n_t = self.n_tooth if n_tooth_at_lam is None else n_tooth_at_lam
        lay = [(self.n_gap, self.d_gap), (n_t, self.d_tooth)] * self.N
        return stack_R_oblique(lam, sin_chi, n_t, lay, self.n_gap, LATERAL_POL[self.slab_pol]) * (1 - self.bounce_loss)


def cascaded_R(lam, sin_chi, n_in, sections, n_out=1.0, pol="s"):
    """Two or more DBR sections in series (each a list of layers) -> one stack; the simplest
    'doubly resonant' mirror: section 1 tuned to lam1, section 2 to lam2."""
    lay = [l for sec in sections for l in sec]
    return stack_R_oblique(lam, sin_chi, n_in, lay, n_out, pol)


def double_resonant_orders(lam1, lam2, n_t1, n_t2, max_order=31, n_gap=1.0, top=5):
    """Odd integer orders for ONE periodic DBR to be on a Bragg condition at both wavelengths:
        tooth: a1 lam1/(4 n_t1) = a2 lam2/(4 n_t2),   gap: b1 lam1/(4 n_gap) = b2 lam2/(4 n_gap)
    Returns the best candidates with relative thickness mismatches (small is good)."""
    odds = range(1, max_order + 1, 2)
    out = []
    for a1 in odds:
        for a2 in odds:
            ea = abs(a1 * lam1 / n_t1 - a2 * lam2 / n_t2) / (a1 * lam1 / n_t1)
            for b1 in odds:
                for b2 in odds:
                    eb = abs(b1 * lam1 - b2 * lam2) / (b1 * lam1)
                    out.append((max(ea, eb) + 1e-4 * (a1 + b1), ea, eb, (a1, a2), (b1, b2),
                                a1 * lam1 / (4 * n_t1), b1 * lam1 / (4 * n_gap)))
    out.sort(key=lambda t: t[0])
    return [dict(tooth_orders=t[3], gap_orders=t[4], tooth_mismatch=t[1], gap_mismatch=t[2],
                 d_tooth_nm=t[5] * 1e9, d_gap_nm=t[6] * 1e9) for t in out[:top]]


# --------------------------------------------------------------------------- billiard
@dataclass
class Stadium:
    """Stadium billiard: two half-discs of radius r joined by straight edges of length a.
    a = 0 is the circle (regular, integrable baseline). Footprint width = 2 r."""
    r: float
    a: float = 0.0

    @property
    def area(self): return 2 * self.r * self.a + np.pi * self.r**2

    @property
    def perimeter(self): return 2 * self.a + 2 * np.pi * self.r

    @property
    def mean_chord(self): return np.pi * self.area / self.perimeter   # Cauchy, 2D

    def point_at(self, s):
        """Boundary point and outward normal at arclength s (CCW from the bottom-left corner of the straight)."""
        s = np.mod(np.asarray(s, float), self.perimeter)
        r, a = self.r, self.a
        x = np.empty_like(s); y = np.empty_like(s); nx = np.empty_like(s); ny = np.empty_like(s)
        m1 = s < a
        m2 = (s >= a) & (s < a + np.pi * r)
        m3 = (s >= a + np.pi * r) & (s < 2 * a + np.pi * r)
        m4 = s >= 2 * a + np.pi * r
        x[m1], y[m1], nx[m1], ny[m1] = s[m1] - a / 2, -r, 0, -1
        ph = s[m2] - a
        ph = ph / r - np.pi / 2
        x[m2], y[m2], nx[m2], ny[m2] = a / 2 + r * np.cos(ph), r * np.sin(ph), np.cos(ph), np.sin(ph)
        x[m3], y[m3], nx[m3], ny[m3] = a / 2 - (s[m3] - a - np.pi * r), r, 0, 1
        ph = (s[m4] - 2 * a - np.pi * r) / r + np.pi / 2
        x[m4], y[m4], nx[m4], ny[m4] = -a / 2 + r * np.cos(ph), r * np.sin(ph), np.cos(ph), np.sin(ph)
        return x, y, nx, ny

    def inside(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        cx = np.clip(x, -self.a / 2, self.a / 2)
        return (x - cx) ** 2 + y**2 <= self.r**2

    def outline(self, n=400):
        s = np.linspace(0, self.perimeter, n)
        x, y, _, _ = self.point_at(s)
        return x, y

    def _hit(self, px, py, dx, dy):
        """First boundary hit from an interior/boundary point (convex => smallest valid t > eps).
        Returns t, nx, ny, s."""
        r, a = self.r, self.a
        eps = 1e-9 * r
        n = px.size
        T = np.full((6, n), np.inf)
        # lines y = +-r
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (r - py) / dy
            xh = px + tt * dx
            T[0] = np.where((dy > 0) & (tt > eps) & (np.abs(xh) <= a / 2 + 1e-12), tt, np.inf)
            tb = (-r - py) / dy
            xh = px + tb * dx
            T[1] = np.where((dy < 0) & (tb > eps) & (np.abs(xh) <= a / 2 + 1e-12), tb, np.inf)
        # caps
        for idx, cx, sign in ((2, a / 2, +1), (4, -a / 2, -1)):
            ox, oy = px - cx, py
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - r * r
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0))
            for j, t in enumerate((-b - sq, -b + sq)):
                xh = px + t * dx - cx
                ok = (disc >= 0) & (t > eps) & (sign * xh >= -1e-12)
                T[idx + j] = np.where(ok, t, np.inf)
        k = np.argmin(T, axis=0)
        t = T[k, np.arange(n)]
        hx, hy = px + t * dx, py + t * dy
        nx = np.zeros(n); ny = np.zeros(n); s = np.zeros(n)
        top, bot = k == 0, k == 1
        rc, lc = (k == 2) | (k == 3), (k == 4) | (k == 5)
        ny[top], ny[bot] = 1, -1
        s[bot] = hx[bot] + a / 2
        s[top] = a + np.pi * r + (a / 2 - hx[top])
        ph = np.arctan2(hy[rc], hx[rc] - a / 2)
        nx[rc], ny[rc] = np.cos(ph), np.sin(ph)
        s[rc] = a + r * (ph + np.pi / 2)
        ph = np.mod(np.arctan2(hy[lc], hx[lc] + a / 2), 2 * np.pi)
        nx[lc], ny[lc] = np.cos(ph), np.sin(ph)
        s[lc] = 2 * a + np.pi * r + r * (ph - np.pi / 2)
        return t, nx, ny, s


@dataclass
class SegmentedCell:
    """Circular segmented cell: a regular N-gon of facets (circumradius r), each facet an in-plane DBR mirror.

    The unperturbed polygon billiard is NOT chaotic (zero Lyapunov exponent: rays keep a few conserved
    directions). Designed perturbations break that:
      tilt_rms    rms random tilt of each facet about its midpoint [rad]  (fixed pattern, set by lithography)
      offset_rms  rms radial displacement of each facet [m]
      curvature   1/rho of every facet [1/m]; >0 convex into the cell (dispersing -> strongly mixing), <0 concave
    Facet gaps/overlaps at the corners are neglected (facets are extended by 15 % so rays never leak).
    The boundary coordinate s runs along the facets (length 2h each); ports are windows in s.
    """
    r: float
    n_facets: int = 24
    tilt_rms: float = 0.0
    curvature: float = 0.0
    offset_rms: float = 0.0
    seed: int = 0
    a: float = 0.0

    def __post_init__(self):
        N = self.n_facets
        rng = np.random.default_rng(self.seed)
        th = 2 * np.pi * np.arange(N) / N
        self.h = self.r * np.sin(np.pi / N)
        self.apothem = self.r * np.cos(np.pi / N)
        phi = th + self.tilt_rms * rng.standard_normal(N)
        dr = self.offset_rms * rng.standard_normal(N)
        self.n_k = np.stack([np.cos(phi), np.sin(phi)], 1)
        self.t_k = np.stack([-np.sin(phi), np.cos(phi)], 1)
        self.p_k = (self.apothem + dr)[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
        self.s_in_default = self.h

    @property
    def perimeter(self): return 2 * self.h * self.n_facets

    @property
    def area(self): return 0.5 * self.n_facets * self.r**2 * np.sin(2 * np.pi / self.n_facets)

    @property
    def mean_chord(self): return np.pi * self.area / self.perimeter

    def _surface(self, k, u):
        """Point and outward normal on facet k at tangential offset u (arrays)."""
        nk, tk, pk = self.n_k[k], self.t_k[k], self.p_k[k]
        if self.curvature == 0:
            return pk[:, 0] + u * tk[:, 0], pk[:, 1] + u * tk[:, 1], nk[:, 0], nk[:, 1]
        rho = 1 / self.curvature
        R = abs(rho)
        sag = R - np.sqrt(np.maximum(R * R - u * u, 0))
        sg = np.sign(rho)
        x, y = pk[:, 0] + u * tk[:, 0] + sg * sag * nk[:, 0], pk[:, 1] + u * tk[:, 1] + sg * sag * nk[:, 1]
        cx, cy = pk[:, 0] + rho * nk[:, 0], pk[:, 1] + rho * nk[:, 1]
        return x, y, -sg * (x - cx) / R, -sg * (y - cy) / R

    def point_at(self, s):
        s = np.mod(np.asarray(s, float), self.perimeter)
        k = np.minimum((s // (2 * self.h)).astype(int), self.n_facets - 1)
        u = s - k * 2 * self.h - self.h
        return self._surface(k, u)

    def outline(self, n=400):
        per = max(n // self.n_facets, 4)
        xs, ys = [], []
        for k in range(self.n_facets):
            u = np.linspace(-self.h, self.h, per)
            x, y, _, _ = self._surface(np.full(per, k), u)
            xs.append(x); ys.append(y)
        xs.append(xs[0][:1]); ys.append(ys[0][:1])
        return np.concatenate(xs), np.concatenate(ys)

    def inside(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        ok = np.ones(x.shape, bool)
        for k in range(self.n_facets):
            ok &= (x - self.p_k[k, 0]) * self.n_k[k, 0] + (y - self.p_k[k, 1]) * self.n_k[k, 1] <= 0
        return ok

    def _hit(self, px, py, dx, dy):
        n = px.size
        eps = 1e-9 * self.r
        hh = 1.15 * self.h
        bt = np.full(n, np.inf); bnx = np.zeros(n); bny = np.zeros(n); bs = np.zeros(n)
        rho = 1 / self.curvature if self.curvature != 0 else None
        for k in range(self.n_facets):
            nk, tk, pk = self.n_k[k], self.t_k[k], self.p_k[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                if rho is None:
                    dn = dx * nk[0] + dy * nk[1]
                    t = ((pk[0] - px) * nk[0] + (pk[1] - py) * nk[1]) / dn
                    ok = (dn > 1e-12) & (t > eps)
                    u = (px + t * dx - pk[0]) * tk[0] + (py + t * dy - pk[1]) * tk[1]
                    ok &= np.abs(u) <= hh
                    nx_, ny_ = nk[0], nk[1]
                else:
                    R, sg = abs(rho), np.sign(rho)
                    cx, cy = pk[0] + rho * nk[0], pk[1] + rho * nk[1]
                    ox, oy = px - cx, py - cy
                    b = ox * dx + oy * dy
                    disc = b * b - (ox * ox + oy * oy - R * R)
                    sq = np.sqrt(np.maximum(disc, 0))
                    t = -b - sq if sg > 0 else -b + sq
                    hx, hy = px + t * dx, py + t * dy
                    side = ((hx - cx) * nk[0] + (hy - cy) * nk[1]) * sg < 0
                    u = (hx - pk[0]) * tk[0] + (hy - pk[1]) * tk[1]
                    ok = (disc >= 0) & (t > eps) & (np.abs(u) <= hh) & side
                    nrm = np.hypot(hx - cx, hy - cy)
                    nx_, ny_ = -sg * (hx - cx) / nrm, -sg * (hy - cy) / nrm
            upd = ok & (t < bt)
            bt = np.where(upd, t, bt)
            bnx = np.where(upd, nx_, bnx); bny = np.where(upd, ny_, bny)
            bs = np.where(upd, k * 2 * self.h + np.clip(u, -self.h, self.h) + self.h, bs)
        return bt, bnx, bny, bs


def lyapunov(cell, n=300, n_fit=400, delta=1e-12, seed=0, j0=40):
    """Finite-time largest Lyapunov exponent per bounce: pairs of trajectories from the same boundary point with launch
    angles differing by delta; separation d = sqrt(dr^2 + (l dtheta)^2), l = mean chord. The exponent is the slope of the
    ensemble-mean log d over bounces j0..jmax, jmax = where the median separation reaches 1e-3 (before saturation). ~0 for regular
    (polygon) billiards (separation grows only linearly in time)."""
    rng = np.random.default_rng(seed)
    s0 = rng.random(n) * cell.perimeter
    x0, y0, nx, ny = cell.point_at(s0)
    ch = np.arcsin(rng.uniform(-0.9, 0.9, n))
    ux, uy = -nx, -ny
    th = np.arctan2(uy, ux) + ch
    ell = cell.mean_chord
    st = []
    for off in (0.0, delta):
        st.append([x0.copy(), y0.copy(), np.cos(th + off), np.sin(th + off)])
    logd = np.full((n, n_fit), np.nan)
    pos = [None, None]
    ang = [None, None]
    for j in range(n_fit):
        for q in (0, 1):
            x, y, dx, dy = st[q]
            t, hnx, hny, hs = cell._hit(x, y, dx, dy)
            x, y = x + t * dx, y + t * dy
            dn = dx * hnx + dy * hny
            dx, dy = dx - 2 * dn * hnx, dy - 2 * dn * hny
            nn = np.hypot(dx, dy)
            st[q] = [x, y, dx / nn, dy / nn]
        dr2 = (st[0][0] - st[1][0]) ** 2 + (st[0][1] - st[1][1]) ** 2
        dth = np.arctan2(st[0][2] * st[1][3] - st[0][3] * st[1][2], st[0][2] * st[1][2] + st[0][3] * st[1][3])
        logd[:, j] = 0.5 * np.log(dr2 + (ell * dth) ** 2 + 1e-300)
    med = np.nanmedian(logd, 0)
    sat = int(np.argmax(med > np.log(1e-3))) if (med > np.log(1e-3)).any() else None
    m = np.nanmean(logd, 0)
    jj = np.arange(1, n_fit + 1)
    if sat is not None and sat >= 8:                  # fast exponential growth: fit before saturation
        return float(np.polyfit(jj[2:sat], m[2:sat], 1)[0])
    return float(np.polyfit(jj[j0:], m[j0:], 1)[0])    # slow / non-exponential growth


def poincare(cell, n_rays=40, n_hits=300, seed=0):
    """Boundary phase-space samples (s, signed sin chi) of long trajectories (no ports).
    Invariant measure of billiards is uniform in (s, sin chi): initial conditions are drawn uniformly."""
    rng = np.random.default_rng(seed)
    s0 = rng.random(n_rays) * cell.perimeter
    x, y, nx, ny = cell.point_at(s0)
    sc = rng.uniform(-0.98, 0.98, n_rays)
    ch = np.arcsin(sc)
    ux, uy = -nx, -ny
    dx, dy = ux * np.cos(ch) - uy * np.sin(ch), ux * np.sin(ch) + uy * np.cos(ch)
    S = np.full((n_rays, n_hits), np.nan); SC = np.full((n_rays, n_hits), np.nan)
    ok = np.ones(n_rays, bool)
    for j in range(n_hits):
        t, hnx, hny, hs = cell._hit(x, y, dx, dy)
        ok &= np.isfinite(t)
        t = np.where(ok, t, 0)
        x, y = x + t * dx, y + t * dy
        dn = dx * hnx + dy * hny
        S[:, j] = np.where(ok, hs, np.nan); SC[:, j] = np.where(ok, dx * hny - dy * hnx, np.nan)
        dx, dy = dx - 2 * dn * hnx, dy - 2 * dn * hny
        nn = np.hypot(dx, dy)
        dx, dy = dx / nn, dy / nn
    return S, SC


@dataclass
class RayTable:
    """Geometry-only ray data: reusable for any mirror R(chi), loss and wavelength."""
    cell: Stadium
    port_w: float
    s_in: float
    s_out: float
    n_rays: int
    n_bounce: int
    chord: np.ndarray      # (n_rays, n_bounce) float32  chord j ends at hit j
    sinchi: np.ndarray     # (n_rays, n_bounce) float32  |sin| of incidence angle at hit j
    exit_idx: np.ndarray   # (n_rays,) index of the hit that lands in a port (-1: trapped)
    exit_port: np.ndarray  # (n_rays,) 0 = input port (lost), 1 = output port (detected)
    theta0: float
    start: np.ndarray = field(default=None, repr=False)   # (n_rays, 4) x, y, dx, dy for plotting
    hits: np.ndarray = field(default=None, repr=False)    # (n_rays, n_bounce, 2) hit positions (keep_hits=True only)
    itin: np.ndarray = field(default=None, repr=False)    # (n_rays,) uint64 hash of the facet sequence (path identity)


def _in_window(s, s0, w, P):
    d = np.mod(s - s0 + P / 2, P) - P / 2
    return np.abs(d) <= w / 2


def trace_rays(cell: Stadium, port_w: float, n_rays: int = 3000, n_bounce: int = 2500,
               theta0: float = np.radians(20), s_in: float | None = None, s_out_frac: float = 0.37,
               seed: int = 1, keep_start: bool = True, keep_hits: bool = False, max_path: float | None = None,
               theta_c: float = 0.0, s_out: float | None = None, extra_ports=(), launch_w: float | None = None,
               launch_off: float = 0.0) -> RayTable:
    """Launch rays from the input port (angle uniform within +-theta0 of the inward normal) and trace
    specular reflections until they reach the input or output port window, or n_bounce is exhausted."""
    rng = np.random.default_rng(seed)
    P = cell.perimeter
    s_in = getattr(cell, "s_in_default", cell.a / 2) if s_in is None else s_in
    s_out = s_in + s_out_frac * P if s_out is None else s_out
    lw = port_w if launch_w is None else launch_w                 # beam footprint inside the port (narrow footprint = divergent beam)
    s0 = s_in + launch_off + (rng.random(n_rays) - 0.5) * lw
    x, y, nx, ny = cell.point_at(s0)
    ang = theta_c + (rng.random(n_rays) * 2 - 1) * theta0       # launch angle from the inward normal: centre theta_c, half-width theta0
    ux, uy = -nx, -ny                                    # inward normal
    dx, dy = ux * np.cos(ang) - uy * np.sin(ang), ux * np.sin(ang) + uy * np.cos(ang)
    start = np.stack([x, y, dx, dy], 1).copy() if keep_start else None
    chord = np.zeros((n_rays, n_bounce), np.float32)
    sinchi = np.zeros((n_rays, n_bounce), np.float32)
    exit_idx = np.full(n_rays, -1, int)
    exit_port = np.full(n_rays, -1, int)
    hits = np.zeros((n_rays, n_bounce, 2), np.float32) if keep_hits else None
    alive = np.ones(n_rays, bool)
    Lc = np.zeros(n_rays)
    itin = np.zeros(n_rays, np.uint64)
    nseg = getattr(cell, "n_facets", 64)
    for j in range(n_bounce):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        t, hnx, hny, hs = cell._hit(x[ia], y[ia], dx[ia], dy[ia])
        bad = ~np.isfinite(t)
        if bad.any():                                    # leaked through a corner gap: counted as lost
            exit_port[ia[bad]] = -2
            alive[ia[bad]] = False
            t = np.where(bad, 0.0, t); hs = np.where(bad, -1e9, hs)
        hx, hy = x[ia] + t * dx[ia], y[ia] + t * dy[ia]
        fac = np.clip(np.floor(np.nan_to_num(hs) / (P / nseg)), 0, nseg - 1).astype(np.uint64) + np.uint64(1)
        itin[ia] = itin[ia] * np.uint64(1000003) + fac
        dn = dx[ia] * hnx + dy[ia] * hny
        chord[ia, j] = t
        if keep_hits:
            hits[ia, j, 0], hits[ia, j, 1] = hx, hy
        sinchi[ia, j] = dx[ia] * hny - dy[ia] * hnx          # signed (conserved angular momentum in a circle)
        pin = _in_window(hs, s_in, port_w, P)
        for s_x in extra_ports:                       # other beams' input ports: light leaving through them is lost
            pin = pin | _in_window(hs, s_x, port_w, P)
        pout = _in_window(hs, s_out, port_w, P)
        done = (pin | pout) & ~bad
        exit_idx[ia[done]] = j
        exit_port[ia[done]] = np.where(pout[done], 1, 0)
        alive[ia[done]] = False
        keep = ~done & ~bad
        if max_path is not None:
            Lc[ia] += t
            keep &= Lc[ia] < max_path
            alive[ia[~done & ~bad & ~keep]] = False
        ik = ia[keep]
        x[ik], y[ik] = hx[keep], hy[keep]
        ndx, ndy = dx[ia][keep] - 2 * dn[keep] * hnx[keep], dy[ia][keep] - 2 * dn[keep] * hny[keep]
        nn = np.hypot(ndx, ndy)
        dx[ik], dy[ik] = ndx / nn, ndy / nn
    return RayTable(cell, port_w, s_in, s_out, n_rays, n_bounce, chord, sinchi, exit_idx, exit_port, theta0, start, hits, itin)


def trace_path(cell: Stadium, x0, y0, dx0, dy0, n_hits=40):
    """Single trajectory polyline (no ports) for plotting."""
    x, y, dx, dy = (np.array([v], float) for v in (x0, y0, dx0, dy0))
    xs, ys = [x[0]], [y[0]]
    for _ in range(n_hits):
        t, nx, ny, _ = cell._hit(x, y, dx, dy)
        x, y = x + t * dx, y + t * dy
        dn = dx * nx + dy * ny
        dx, dy = dx - 2 * dn * nx, dy - 2 * dn * ny
        xs.append(x[0]); ys.append(y[0])
    return np.array(xs), np.array(ys)


# --------------------------------------------------------------------------- path-length statistics
@dataclass
class CellResult:
    T_det: float          # fraction of launched power reaching the output port
    L_mean: float         # mean geometric path of detected power [m]
    L_eff_gas: float      # Gamma * L_mean: equivalent bulk-gas path [m]
    S1: float             # signal yield: dP_det/P_in per unit alpha_gas = Gamma*T_det*L_mean [m]
    L: np.ndarray         # path of each detected ray [m]
    W: np.ndarray         # its power weight
    bounces: np.ndarray
    n_det: int
    L_mean_err: float


def evaluate(table: RayTable, R_of_sinchi, alpha_bg: float = 0.0, Gamma: float = 1.0) -> CellResult:
    """Re-weight the geometric ray table. R_of_sinchi(sin chi) -> mirror power reflectance (array in/out);
    alpha_bg: background power attenuation of the guided mode [1/m] (scattering + absorption, NOT gas)."""
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    if sel.size == 0:
        return CellResult(0, np.nan, np.nan, 0, np.array([]), np.array([]), np.array([]), 0, np.nan)
    e = table.exit_idx[sel]
    j = np.arange(table.n_bounce)[None, :]
    path_mask = j <= e[:, None]
    int_mask = j < e[:, None]
    ch = table.chord[sel].astype(float) * path_mask
    L = ch.sum(1)
    sg = np.linspace(0, 1, 513)
    lnR = np.log(np.clip(R_of_sinchi(sg), 1e-12, 1.0))
    lnRt = np.interp(np.abs(table.sinchi[sel]), sg, lnR) * int_mask
    lnW = lnRt.sum(1) - alpha_bg * L
    W = np.exp(lnW)
    T = W.sum() / table.n_rays
    Lm = (W * L).sum() / W.sum()
    var = (W * (L - Lm) ** 2).sum() / W.sum()
    neff_n = W.sum() ** 2 / (W**2).sum()
    err = np.sqrt(var / max(neff_n, 1))
    return CellResult(T, Lm, Gamma * Lm, Gamma * T * Lm, L, W, e + 1, sel.size, err)


def occupancy_map(cell: Stadium, port_w: float, R_of_sinchi, alpha_bg: float = 0.0, n_rays: int = 500,
                  n_bounce: int = 1500, nb: int = 160, theta0: float = np.radians(20), seed: int = 4,
                  samples: int = 8):
    """Power-weighted time-averaged ray density over the cell (all launched rays, until absorbed or
    exited). Returns x edges, y edges, density normalised to its mean over the cell interior, and the
    coverage fraction = area share where density > 10 % of that mean."""
    tab = trace_rays(cell, port_w, n_rays, n_bounce, theta0, seed=seed, keep_hits=True)
    sg = np.linspace(0, 1, 513)
    lnR = np.log(np.clip(R_of_sinchi(sg), 1e-12, 1.0))
    j = np.arange(n_bounce)[None, :]
    last = np.where(tab.exit_idx >= 0, tab.exit_idx, n_bounce - 1)[:, None]
    valid = j <= last
    ch = tab.chord.astype(float) * valid
    lnr = np.interp(np.abs(tab.sinchi), sg, lnR) * (j < last)
    Lcum_prev = np.cumsum(ch, 1) - ch
    surv = np.exp(np.cumsum(lnr, 1) - lnr - alpha_bg * Lcum_prev)
    x0 = np.concatenate([tab.start[:, :1], tab.hits[:, :-1, 0]], 1)
    y0 = np.concatenate([tab.start[:, 1:2], tab.hits[:, :-1, 1]], 1)
    x1, y1 = tab.hits[..., 0], tab.hits[..., 1]
    wch = surv * ch
    r, a = cell.r, cell.a
    xe = np.linspace(-(a / 2 + r), a / 2 + r, nb + 1)
    ye = np.linspace(-r, r, max(int(nb * 2 * r / (a + 2 * r)), 8) + 1)
    H = np.zeros((len(xe) - 1, len(ye) - 1))
    m = valid & (wch > 0)
    rng = np.random.default_rng(seed + 100)
    for k in range(samples):
        f = (k + rng.random(x0.shape)) / samples        # stratified random positions along every chord
        px, py = (x0 + f * (x1 - x0))[m], (y0 + f * (y1 - y0))[m]
        H += np.histogram2d(px, py, bins=(xe, ye), weights=wch[m] / samples)[0]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    ins = cell.inside(XX, YY)
    dens = H / max(H[ins].mean(), 1e-300)
    dens = np.where(ins, dens, np.nan)
    cover = float((dens[ins] > 0.1).mean())
    return xe, ye, dens, cover


def mean_field_estimate(cell: Stadium, port_w: float, R_mean: float, alpha_bg: float = 0.0, Gamma: float = 1.0):
    """Analytic cross-check (ergodic, uniform-boundary-flux assumption): geometric series over bounces.
    Per bounce: exit prob eta_out = w/P (output), eta_in = w/P (lost); survive prob R_mean * exp(-alpha*chord)."""
    ell = cell.mean_chord
    eta = port_w / cell.perimeter
    surv = (1 - 2 * eta) * R_mean * np.exp(-alpha_bg * ell)
    T = eta * np.exp(-alpha_bg * ell) / (1 - surv)
    # mean number of chords of detected power: sum n eta surv^(n-1) / sum eta surv^(n-1) = 1/(1-surv)
    L = ell / (1 - surv)
    return dict(T_det=T, L_mean=L, n_bounce=1 / (1 - surv), L_eff_gas=Gamma * L)


# --------------------------------------------------------------------------- helpers
def dB_per_cm_to_alpha(db_cm: float) -> float:
    """dB/cm power loss -> alpha [1/m] (power attenuation coefficient)."""
    return db_cm * 100 * math.log(10) / 10


def uniform_average(Rfun, n=513):
    sg = np.linspace(0, 1, n)
    return float(np.trapezoid(Rfun(sg), sg))


def rayleigh_check(w0: float, lam: float, n: float, distance: float):
    """Is the in-plane beam diffraction negligible over `distance`?  (z_R vs path length)"""
    return beam_parameters(wavelength=lam, w0=w0, n=n)["z_R"], distance


# --------------------------------------------------------------------------- self-test
def selftest(verbose=True):
    ok = True
    # 1. oblique TMM at normal incidence == engines.bragg_grating.stack_reflectance
    lam = np.linspace(1.45e-6, 1.65e-6, 41)
    nH, nL, dH, dL, N = 2.8, 1.0, 138e-9, 387.5e-9, 6
    mine = stack_R_oblique(lam, np.zeros_like(lam), nL, [(nH, dH), (nL, dL)] * N, nL)
    ref = stack_reflectance(lam, nH, nL, dH, dL, N)["R"]
    e1 = float(np.max(np.abs(mine - ref)))
    ok &= e1 < 1e-9
    # 2. TE slab Gamma -> 1 as thickness -> 0, and TM > TE
    g_thin = membrane_mode("Si", 5e-9, 1.55e-6).Gamma
    te, tm = membrane_mode("Si", 220e-9, 1.55e-6, "TE").Gamma, membrane_mode("Si", 220e-9, 1.55e-6, "TM").Gamma
    ok &= (g_thin > 0.9) and (tm > te)
    # 3. Gamma by brute-force numerical quadrature of the field profile (TE and TM)
    e3 = 0.0
    for pol in ("TE", "TM"):
        m = membrane_mode("Si", 300e-9, 1.55e-6, pol)
        x, F = field_profile(m, 14.0, 320001)
        F = np.sqrt(F) * np.where(x < 0, 1.0, 1.0)         # |f|; sign irrelevant for |.|^2 integrals
        outside = (np.abs(x) > 150e-9).astype(float)
        n2 = np.where(outside > 0, 1.0, m.n_core**2)
        k = 2 * np.pi / 1.55e-6
        beta = k * m.neff
        if pol == "TE":
            I = F**2
            g_num = (1.0 / m.neff) * np.trapezoid(I * outside, x) / np.trapezoid(I, x)
        else:
            Hp = np.gradient(F, x)                                 # |H'| up to the kink at the interface (measure zero)
            E2 = (beta**2 * F**2 + Hp**2) / n2**2
            g_num = np.trapezoid(E2 * outside, x) / (k * beta * np.trapezoid(F**2 / n2, x))
        e3 = max(e3, abs(g_num - m.Gamma) / m.Gamma)
    ok &= e3 < 3e-3     # TM quadrature converges ~1/N at the H' kink
    # 4. ray tracer: circle conserves sin(chi); mean chord = pi A / P; mean-field vs MC
    circ = Stadium(5e-3, 0.0)
    t = trace_rays(circ, 100e-6, 400, 200, seed=3)
    longlived = np.nonzero((t.exit_idx < 0) | (t.exit_idx >= 6))[0][:50]
    spread = float(np.abs(t.sinchi[longlived, :6] - t.sinchi[longlived, :1]).max())
    ok &= spread < 1e-5
    st = Stadium(5e-3, 5e-3)
    tab = trace_rays(st, 150e-6, 3000, 2500, seed=2)
    Rm = 0.999
    mc = evaluate(tab, lambda s: np.full_like(s, Rm), 0.0)
    mf = mean_field_estimate(st, 150e-6, Rm)
    e4 = abs(mc.L_mean - mf["L_mean"]) / mf["L_mean"]
    if verbose:
        print(f"[1] TMM vs engine max|dR| = {e1:.2e}")
        print(f"[2] Gamma(20nm Si)={g_thin:.3f}, Gamma_TE(220nm)={te:.3f}, Gamma_TM(220nm)={tm:.3f}")
        print(f"[3] Gamma analytic vs quadrature rel.err = {e3:.2e}")
        print(f"[4] circle sin(chi) drift = {spread:.1e};  stadium <L> MC {mc.L_mean*100:.1f} cm vs mean-field {mf['L_mean']*100:.1f} cm (rel {e4:.2f})")
    # 5. Brewster: single air/membrane interface, p-pol, zero reflectance at tan(theta_B)=n2/n1; s-pol and p-pol agree at normal incidence
    n1 = 2.8
    sb = np.sin(np.arctan(1 / n1))
    rp = float(stack_R_oblique(1.55e-6, sb, n1, [], 1.0, "p"))
    rs = float(stack_R_oblique(1.55e-6, sb, n1, [], 1.0, "s"))
    d0 = abs(float(stack_R_oblique(1.55e-6, 0.0, n1, [(1.0, 0.3e-6), (n1, 0.15e-6)] * 3, 1.0, "p"))
             - float(stack_R_oblique(1.55e-6, 0.0, n1, [(1.0, 0.3e-6), (n1, 0.15e-6)] * 3, 1.0, "s")))
    ok &= (rp < 1e-12) and (rs > 0.05) and d0 < 1e-12
    # 6. segmented cell: flat unperturbed facets keep every hit on the polygon; reflection law and port mapping consistent
    sc = SegmentedCell(5e-3, 24, 0.0)
    tabs = trace_rays(sc, 100e-6, 300, 300, seed=1, max_path=3.0)
    S_, SC_ = poincare(SegmentedCell(5e-3, 24, 0.01), 20, 200, seed=2)
    xo, yo = sc.outline(2000)
    e6 = float(np.max(np.abs(np.hypot(xo, yo) - 5e-3 * np.cos(np.pi / 24) / np.cos(np.mod(np.arctan2(yo, xo) + np.pi / 24, 2 * np.pi / 24) - np.pi / 24))))
    ok &= e6 < 1e-12 and np.all(np.isfinite(S_)) and np.max(np.abs(SC_)) <= 1.0 + 1e-9
    if verbose:
        print(f"[5] Brewster p-pol R={rp:.1e} (s-pol {rs:.3f}); s/p at normal incidence differ by {d0:.1e}")
        print(f"[6] segmented-cell geometry residual {e6:.1e}; detected {int((tabs.exit_port == 1).sum())} of 300 rays")
    return bool(ok and e4 < 0.25)


ASSUMPTIONS = """
v0.2 assumptions (the ones that move the answer most are marked **):
- Polarisation: slab TE (E in membrane plane) is p-pol and slab TM (E_z) is s-pol on a vertical trench mirror. [v0.1 had this backwards.]
- Cell is 2D: the guided mode propagates in-plane as a ray; diffraction enters only via the launch spread lambda/(n_eff w).
- Mirrors: in-plane DBR of etched air gaps and membrane teeth, 1D effective-index transfer matrix at oblique incidence (TIR and
  frustrated TIR included). ** 3D radiation loss at the slot, TE<->TM conversion, sidewall roughness and finite etch depth are lumped
  into bounce_loss (3e-4, ASSUMED) and are the first thing the full-wave numerics must replace.
- ** Background loss alpha_bg is an input (0.1 dB/cm at Si 220 nm TE, ASSUMED); other membranes scale with the surface-field proxy
  E_edge2 x (n^2-1)^2, which is crude (TM Si 250 nm comes out about 3x TE 220 nm).
- Gas absorption is weak: signal ~ alpha_gas * Gamma * <L>; Gamma from the field integrals of the symmetric slab, validated by quadrature.
- Al2O3: constant n = 1.65 (assumed). Si Sellmeier is a real-part fit valid above ~1.2 um. SiNx is stoichiometric Si3N4.
- Ports are ideal windows (no taper loss); single-mode injection with spread lambda/(n_eff w).
- Coherence model: random-phase statistics on discrete paths (facet itineraries), reflection phases neglected, dn_eff/dT = 1.5e-4 /K assumed.
  A prediction model only; not validated against wave simulations.
- Gas lines: ILLUSTRATIVE hand-entered parameters unless hitran_lines.json exists (run fetch_hitran.py). Do not quote.
"""

if __name__ == "__main__":
    print("selftest passed" if selftest() else "selftest FAILED")
