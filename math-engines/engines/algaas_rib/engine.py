"""AlGaAs-on-insulator rib waveguides for layer-poled, modally phase-matched SHG.

Geometry (x lateral, y vertical, z propagation; SI units everywhere):

    top cladding (air or SiO2)
         ┌─────────┐   rib,  height h_rib, width w (at its base), chi2 < 0   (sign s = -1)
    ─────┴─────────┴─────   flat core, height h_core,               chi2 > 0   (sign s = +1)
    bottom cladding (SiO2 or air)

The rib is etched down to the core/rib interface, so outside the rib only the flat core
remains. Core and rib may have different Al fractions (x_core, x_rib), e.g. different MQW
designs whose average index differs; both are treated as homogeneous alloys.

Polarisation (zinc-blende (001) wafer, propagation along [110]): only d14 couples, and it couples
a TE pump (E in the plane, along the waveguide x axis) to a TM second harmonic (E along [001],
the waveguide y axis). So the overlap integral is

    Γ = ∫ s(x,y) Ex_p(x,y)² Ey_SH(x,y) dA,   with ∫Ex_p² dA = ∫Ey_SH² dA = 1.

Because s flips *vertically* (core +, rib -), no axial grating is needed: the pump fundamental
(even in y) and a vertically odd SH mode (TM with one vertical node placed near the
core/rib interface) have a non-zero Γ, and Δk = 4π (n_SH - n_p) / λ = 0 is the modal
phase-matching condition itself.

Solvers
-------
* eim_neff:   effective index method. Instant, but approximate (typically 1e-2 in n_eff for
              high-contrast ribs, worse for higher-order modes). For sweeps and maps.
* rib_modes:  semi-vectorial finite-difference solver (quasi-TE: Ex, quasi-TM: Ey) on a
              non-uniform cell-centred grid whose cell edges coincide with every interface.
              Richardson extrapolation in the grid step removes the leading O(h²) error.

Material: Afromowitz (1974) single-oscillator model for Al_xGa_(1-x)As below the gap, SiO2 from
the `materials` engine. The absolute AlGaAs index matters at the 1e-3 level for modal phase
matching; `dn_algaas` is a calibration offset for a measured Sellmeier fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from ..common import Result, require_positive, require_range
from ..materials.engine import index as _material_index

_C = 299792458.0
_EPS0 = 8.8541878128e-12
_HC_EV_UM = 1.239841984  # eV·µm
_CLADDINGS = ("sio2", "air")
MIN_MARGIN_EV = 0.10     # warn when the gap (or MQW edge) is less than this above the photon energy


# ----------------------------------------------------------------------------- material
def _afromowitz_params(x):
    x = np.asarray(x, dtype=float)
    e0 = 3.65 + 0.871 * x + 0.179 * x**2
    ed = 36.1 - 2.45 * x
    eg = 1.424 + 1.266 * x + 0.266 * x**2
    return e0, ed, eg


def algaas_n(x, wavelength, eg=None):
    """Phase index of Al_xGa_(1-x)As (Afromowitz 1974). NaN at or above the direct gap.

    eg (eV) replaces the gap inside the model, e.g. with the confinement-shifted e1-hh1 edge of an MQW well
    (an approximation: the oscillator strength parameters E_0, E_d stay those of the bulk alloy)."""
    e0, ed, eg_bulk = _afromowitz_params(x)
    eg = eg_bulk if eg is None else np.asarray(eg, dtype=float)
    e = _HC_EV_UM / (np.asarray(wavelength, dtype=float) * 1e6)
    eta = np.pi * ed / (2 * e0**3 * (e0**2 - eg**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = (2 * e0**2 - eg**2 - e**2) / (eg**2 - e**2)
        n2 = 1 + ed / e0 + ed * e**2 / e0**3 + (eta / np.pi) * e**4 * np.log(arg)
        n = np.sqrt(n2)
    return np.where(e < eg, n, np.nan)


def algaas_index(x, wavelength) -> Result:
    """Refractive index of Al_xGa_(1-x)As, its direct gap and the gap margin at the photon energy."""
    require_range("x", x, 0.0, 1.0)
    require_positive(wavelength=wavelength)
    n = algaas_n(x, wavelength)
    eg_lin = bandgap(x, "linear")
    eg_idx = bandgap(x, "afromowitz")
    e = _HC_EV_UM / (np.asarray(wavelength, dtype=float) * 1e6)
    margin = np.minimum(eg_lin, eg_idx) - e
    notes = ["Afromowitz (1974) single-oscillator model, 300 K, below the direct gap; accuracy of the "
             "absolute index is of order 1e-2, so calibrate with dn_algaas before trusting a phase-matching width",
             "bandgap is the linear direct-gap formula 1.424 + 1.247 x; the index model itself uses 1.424 + 1.266 x + 0.266 x²; "
             "margin is taken against the lower of the two"]
    if np.any(margin < MIN_MARGIN_EV):
        notes.append(f"WARNING: gap margin below {MIN_MARGIN_EV} eV: band-edge (Urbach) absorption, and the model is least accurate there")
    if np.any(np.asarray(x) > 0.45):
        notes.append("WARNING: x > 0.45, the gap is indirect and the direct-gap formulas do not apply")
    return Result(values={"n": n, "bandgap": eg_lin, "photon_energy": e, "margin": margin},
                  units={"n": "", "bandgap": "eV", "photon_energy": "eV", "margin": "eV"}, assumptions=notes)


# ----------------------------------------------------------------------------- absorption and MQW edge
GAP_MODELS = ("linear", "afromowitz")
URBACH_EV = 0.010          # assumed Urbach energy (room temperature, clean material); no source in the project files
ALPHA_EDGE = 1.0e6         # assumed absorption at E = E_g, 1/m (1e4 1/cm)


def bandgap(x, model="linear"):
    """Direct gap of Al_xGa_(1-x)As in eV. 'linear' = 1.424 + 1.247 x (project note, Casey-Panish type, direct
    gap only, x < 0.45); 'afromowitz' = 1.424 + 1.266 x + 0.266 x² (the gap inside the index model)."""
    x = np.asarray(x, dtype=float)
    if model == "linear":
        return 1.424 + 1.247 * x
    if model == "afromowitz":
        return 1.424 + 1.266 * x + 0.266 * x**2
    raise ValueError(f"model must be one of {GAP_MODELS}")


def urbach_alpha(edge_ev, wavelength, e_urbach=URBACH_EV, alpha_edge=ALPHA_EDGE):
    """Sub-gap absorption α = α_edge exp(-(E_edge - E)/E_U), 1/m. An order-of-magnitude estimate of the intrinsic
    tail only (not defect, free-carrier or scattering loss); both parameters are assumptions."""
    e = _HC_EV_UM / (np.asarray(wavelength, dtype=float) * 1e6)
    return alpha_edge * np.exp(-(np.asarray(edge_ev, dtype=float) - e) / e_urbach)


def absorption_report(x, wavelength, model="linear", edge_ev=None) -> Result:
    """Photon energy, gap margin and Urbach-tail loss of an AlGaAs layer (or of an MQW edge edge_ev) at a wavelength."""
    require_positive(wavelength=wavelength)
    eg = bandgap(x, model) if edge_ev is None else np.asarray(edge_ev, dtype=float)
    e = _HC_EV_UM / (np.asarray(wavelength, dtype=float) * 1e6)
    alpha = urbach_alpha(eg, wavelength)
    notes = [f"gap model: {model if edge_ev is None else 'explicit edge'}; Urbach tail with E_U = {URBACH_EV * 1e3:.0f} meV and "
             f"α(E_g) = {ALPHA_EDGE * 1e-2:.0e} 1/cm are assumptions",
             "intrinsic tail only; defects, free carriers, TPA of the pump with the SH (ħω_p + ħω_SH = 2.4 eV), and scattering are not included"]
    if edge_ev is None and np.any(np.asarray(x) > 0.45):
        notes.append("WARNING: x > 0.45, the gap is indirect and the direct-gap formula does not apply")
    return Result(values={"photon_energy": e, "edge": eg, "margin": eg - e, "alpha": alpha,
                          "loss_db_per_cm": alpha * 4.343e-2},
                  units={"photon_energy": "eV", "edge": "eV", "margin": "eV", "alpha": "1/m", "loss_db_per_cm": "dB/cm"},
                  assumptions=notes)


def min_al_fraction(wavelength, margin_ev, model="linear"):
    """Smallest x whose direct gap exceeds the photon energy at `wavelength` by margin_ev."""
    target = _HC_EV_UM / (wavelength * 1e6) + margin_ev
    if model == "linear":
        return (target - 1.424) / 1.247
    xs = np.linspace(0, 1, 100001)
    return float(xs[np.argmax(bandgap(xs, model) >= target)])


@dataclass(frozen=True)
class MQW:
    """Periodic well/barrier stack, GaAs-like well and AlGaAs barrier by default. Lengths in m.

    Single-well finite-barrier envelope model (BenDaniel-Duke, no inter-well coupling, no strain): valid for barriers
    of several nm. Masses and band offset are approximate defaults and can be overridden."""

    well: float
    barrier: float
    x_well: float = 0.0
    x_barrier: float = 0.4
    q_c: float = 0.65                  # conduction-band share of the gap difference
    exciton_ev: float = 0.008          # binding energy subtracted from the e1-hh1 edge
    m_e: tuple | None = None           # (well, barrier) electron masses in m0; default 0.067 + 0.083 x
    m_hh: tuple | None = None          # (well, barrier) heavy-hole masses along z; default 0.34 + 0.42 x

    def masses(self):
        me = self.m_e or (0.067 + 0.083 * self.x_well, 0.067 + 0.083 * self.x_barrier)
        mh = self.m_hh or (0.34 + 0.42 * self.x_well, 0.34 + 0.42 * self.x_barrier)
        return me, mh

    @property
    def fill(self):
        return self.well / (self.well + self.barrier)

    @property
    def average_x(self):
        return self.fill * self.x_well + (1 - self.fill) * self.x_barrier


_HBAR2_2M0 = 0.0380998          # ħ²/(2 m0) in eV nm²


def _ground_state(v, a, m_w, m_b):
    """Lowest bound level (eV, above the well bottom) of a symmetric finite well of half-width a (nm), depth v (eV)."""
    if v <= 0:
        return 0.0
    k = lambda e: np.sqrt(e * m_w / _HBAR2_2M0)
    kap = lambda e: np.sqrt(max(v - e, 0.0) * m_b / _HBAR2_2M0)
    h = lambda e: m_b * k(e) * np.sin(k(e) * a) - m_w * kap(e) * np.cos(k(e) * a)
    e_pi = _HBAR2_2M0 * (np.pi / (2 * a)) ** 2 / m_w       # k a = π/2
    hi = min(v, e_pi)
    return brentq(h, 1e-9, hi * (1 - 1e-12), xtol=1e-12)


def mqw_edge(q: MQW, model="linear") -> Result:
    """Confinement energies and the e1-hh1 absorption edge of one well (eV)."""
    require_positive(well=q.well)
    eg_w, eg_b = float(bandgap(q.x_well, model)), float(bandgap(q.x_barrier, model))
    d = max(eg_b - eg_w, 0.0)
    (me_w, me_b), (mh_w, mh_b) = q.masses()
    a = q.well * 1e9 / 2
    e1 = _ground_state(q.q_c * d, a, me_w, me_b)
    h1 = _ground_state((1 - q.q_c) * d, a, mh_w, mh_b)
    edge = eg_w + e1 + h1 - q.exciton_ev
    notes = ["single finite well, envelope approximation: barriers must be thick enough that wells do not couple",
             "mass and offset defaults are approximate; the edge is only as good as the Qc and mass inputs",
             f"gap model {model}"]
    if q.barrier < 3e-9:
        notes.append("WARNING: barrier < 3 nm, inter-well coupling (miniband) will lower the edge")
    return Result(values={"e1": e1, "hh1": h1, "edge": edge, "well_gap": eg_w, "barrier_gap": eg_b, "band_offset": d},
                  units={k: "eV" for k in ("e1", "hh1", "edge", "well_gap", "barrier_gap", "band_offset")},
                  assumptions=notes)


def mqw_max_well(x_well, x_barrier, edge_target_ev, model="linear", **kw) -> float:
    """Largest well width (m) whose e1-hh1 edge stays at or above edge_target_ev (NaN if even 0.5 nm fails)."""
    f = lambda w: mqw_edge(MQW(w, 10e-9, x_well, x_barrier, **kw), model)["edge"] - edge_target_ev
    lo, hi = 0.5e-9, 30e-9
    if f(lo) < 0:
        return float("nan")
    if f(hi) > 0:
        return hi
    return brentq(f, lo, hi, xtol=1e-12)


def mqw_indices(q: MQW, wavelength, model="linear") -> dict:
    """Effective indices of the stack for E in the layer plane (n_par, TE) and along growth (n_perp, TM), from the
    long-wavelength form-birefringence averages ε_par = <ε>, 1/ε_perp = <1/ε>. The well index uses the confined edge
    in place of the bulk gap (see algaas_n)."""
    edge = float(mqw_edge(q, model)["edge"])
    n_w = float(algaas_n(q.x_well, wavelength, eg=edge + q.exciton_ev))
    n_b = float(algaas_n(q.x_barrier, wavelength))
    f = q.fill
    e_w, e_b = n_w**2, n_b**2
    return {"n_par": float(np.sqrt(f * e_w + (1 - f) * e_b)), "n_perp": float(np.sqrt(1 / (f / e_w + (1 - f) / e_b))),
            "n_well": n_w, "n_barrier": n_b, "edge": edge}


# ----------------------------------------------------------------------------- structure
@dataclass(frozen=True)
class RibStack:
    """Layer-poled AlGaAs rib on a flat core. Lengths in m."""

    h_core: float
    h_rib: float
    width: float                 # rib width at its base (the core/rib interface)
    x_core: float = 0.30         # Al fraction of the chi2 > 0 core (0.35 is the recommended low-absorption value)
    x_rib: float = 0.30          # Al fraction of the chi2 < 0 rib
    bottom: str = "sio2"
    top: str = "air"
    sidewall_deg: float = 0.0    # rib wall angle from vertical; the rib narrows upward
    dn_algaas: float = 0.0       # additive index calibration applied to both AlGaAs layers
    dn_rib: float = 0.0          # extra bulk index of the chi2<0 rib relative to the core (MQW / composition
                                 # engineering) at the pump, 1550 nm; negative = rib lower than core
    dn_rib_sh: float | None = None   # the same offset at the SH, 775 nm; None = equal to dn_rib at every wavelength,
                                     # otherwise linear in 1/λ between the two (offsets are dispersive in practice)
    core_mqw: MQW | None = None      # if set, the core is this well/barrier stack: x_core is ignored, TE sees n_par,
    rib_mqw: MQW | None = None       # TM sees n_perp (form birefringence), and the absorption edge is the e1-hh1 edge

    def validate(self):
        require_positive(h_core=self.h_core, width=self.width)
        if self.h_rib < 0:
            raise ValueError("h_rib must be non-negative")
        require_range("x_core", self.x_core, 0.0, 1.0)
        require_range("x_rib", self.x_rib, 0.0, 1.0)
        require_range("sidewall_deg", self.sidewall_deg, 0.0, 45.0)
        require_range("dn_rib", self.dn_rib, -0.5, 0.5)
        if self.dn_rib_sh is not None:
            require_range("dn_rib_sh", self.dn_rib_sh, -0.5, 0.5)
        for name, v in (("bottom", self.bottom), ("top", self.top)):
            if v not in _CLADDINGS:
                raise ValueError(f"{name} must be one of {_CLADDINGS}, got {v!r}")
        if self.h_rib > 0 and self.width / 2 - self.h_rib * np.tan(np.radians(self.sidewall_deg)) <= 0:
            raise ValueError("sidewall angle closes the rib before its top")


def _clad_n(name, lam):
    return 1.0 if name == "air" else float(_material_index("sio2", lam))


def rib_offset(st: RibStack, lam: float) -> float:
    """Engineered index offset of the rib at wavelength lam: dn_rib at 1550 nm, dn_rib_sh at 775 nm,
    linear in 1/λ in between and outside (a constant offset when dn_rib_sh is None)."""
    if st.dn_rib_sh is None:
        return st.dn_rib
    t = (1 / lam - 1 / 1.55e-6) / (1 / 0.775e-6 - 1 / 1.55e-6)
    return st.dn_rib + (st.dn_rib_sh - st.dn_rib) * t


def _layer_n(x, mqw, lam, pol, gap_model="linear"):
    if mqw is None:
        return float(algaas_n(x, lam))
    return mqw_indices(mqw, lam, gap_model)["n_par" if pol.upper() == "TE" else "n_perp"]


def layer_indices(st: RibStack, lam: float, pol: str = "TE") -> dict:
    """Refractive indices at vacuum wavelength lam for polarisation pol (matters only for MQW layers)."""
    nc = _layer_n(st.x_core, st.core_mqw, lam, pol) + st.dn_algaas
    nr = _layer_n(st.x_rib, st.rib_mqw, lam, pol) + st.dn_algaas + rib_offset(st, lam)
    if not (np.isfinite(nc) and np.isfinite(nr)):
        raise ValueError(f"wavelength {lam * 1e9:.0f} nm is above the AlGaAs/MQW edge of the core or rib")
    return {"core": nc, "rib": nr, "bottom": _clad_n(st.bottom, lam), "top": _clad_n(st.top, lam)}


def sh_absorption(st: RibStack, lam_sh: float, model="linear") -> dict:
    """Gap margin and Urbach-tail loss of the core and the rib at the second-harmonic wavelength; the worse layer
    decides. MQW layers use their e1-hh1 edge."""
    out = {}
    for name, x, q in (("core", st.x_core, st.core_mqw), ("rib", st.x_rib, st.rib_mqw)):
        edge = None if q is None else float(mqw_edge(q, model)["edge"])
        r = absorption_report(x if q is None else q.average_x, lam_sh, model, edge)
        out[name] = {"edge": float(r["edge"]), "margin": float(r["margin"]), "loss_db_per_cm": float(r["loss_db_per_cm"])}
    worst = min(out, key=lambda k: out[k]["margin"])
    out["worst"] = worst
    out["margin"] = out[worst]["margin"]
    out["loss_db_per_cm"] = out[worst]["loss_db_per_cm"]
    return out


# ----------------------------------------------------------------------------- 1D multilayer slab
def _slab_f(neff, k0, n_b, ns, ds, n_t, tm):
    """Characteristic function of a multilayer slab (zero at guided modes); vectorised in neff."""
    neff = np.atleast_1d(np.asarray(neff, dtype=float))
    p = lambda n: n * n if tm else 1.0
    u = np.ones_like(neff)
    v = k0 * np.sqrt(np.maximum(neff**2 - n_b**2, 0.0)) / p(n_b)
    for n, d in zip(ns, ds):
        k2 = (k0 * n) ** 2 - (k0 * neff) ** 2
        kap = np.sqrt(k2.astype(complex))
        small = np.abs(kap) < 1e-9 * k0
        kd = kap * d
        cs = np.cos(kd).real
        sn = np.where(small, d, (np.sin(kd) / np.where(small, 1, kap)).real)
        pn = p(n)
        u, v = u * cs + pn * v * sn, -(k2 / pn) * sn * u + v * cs
    gt = k0 * np.sqrt(np.maximum(neff**2 - n_t**2, 0.0))
    return v + gt * u / p(n_t)


def slab_modes(lam, n_bottom, indices, thicknesses, n_top, pol="TE", max_modes=12, npts=1500):
    """Effective indices (descending) of all guided modes of a multilayer slab."""
    k0 = 2 * np.pi / lam
    lo = max(n_bottom, n_top) + 1e-9
    hi = max(indices) - 1e-9
    if hi <= lo:
        return []
    tm = pol.upper() == "TM"
    grid = np.linspace(hi, lo, npts)
    f = _slab_f(grid, k0, n_bottom, indices, thicknesses, n_top, tm)
    out = []
    for a in np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
        g = lambda t: float(_slab_f(t, k0, n_bottom, indices, thicknesses, n_top, tm)[0])
        out.append(brentq(g, grid[a + 1], grid[a], xtol=1e-12))
        if len(out) >= max_modes:
            break
    return out


# ----------------------------------------------------------------------------- effective index method
def eim_neff(st: RibStack, lam: float, pol: str = "TE", v_order: int = 0, l_order: int = 0) -> float:
    """Effective-index-method n_eff of mode (v_order, l_order); NaN if not guided.

    The vertical slab problem is solved with the polarisation of the mode (TE for quasi-TE), the lateral
    one with the opposite (TM-type for quasi-TE: E normal to the side walls)."""
    st.validate()
    n = layer_indices(st, lam, pol)
    pol = pol.upper()
    lat_pol = "TM" if pol == "TE" else "TE"
    ii = slab_modes(lam, n["bottom"], [n["core"], n["rib"]] if st.h_rib > 0 else [n["core"]],
                    [st.h_core, st.h_rib] if st.h_rib > 0 else [st.h_core], n["top"], pol)
    if len(ii) <= v_order:
        return float("nan")
    n_i = ii[v_order]
    oo = slab_modes(lam, n["bottom"], [n["core"]], [st.h_core], n["top"], pol)
    n_o = oo[v_order] if len(oo) > v_order else max(n["bottom"], n["top"])
    if st.h_rib <= 0 or n_i <= n_o:
        return float("nan") if st.h_rib > 0 else n_i
    ll = slab_modes(lam, n_o, [n_i], [st.width], n_o, lat_pol)
    return ll[l_order] if len(ll) > l_order else float("nan")


def eim_dispersion(st: RibStack, lams, pol="TE", orders=((0, 0),)):
    """n_eff(λ) by EIM for each (v_order, l_order); array [len(orders), len(lams)] (NaN where cut off)."""
    return np.array([[eim_neff(st, l, pol, v, m) for l in lams] for v, m in orders])


# ----------------------------------------------------------------------------- 2D finite differences
def _axis_edges(keys, lo, hi, d, inner_lo, inner_hi, grow=1.3):
    """Cell edges on [lo, hi]: cells ≈ d between the keys and `inner` beyond them (interfaces are
    exact edges), then geometrically graded to the box."""
    pts = [keys[0] - inner_lo] + list(keys) + [keys[-1] + inner_hi]
    edges = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil((b - a) / d - 1e-9)))
        edges.extend(np.linspace(a, b, n + 1)[1:])
    left, w = [], d
    x = edges[0]
    while x > lo:
        w *= grow
        x = max(x - w, lo)
        left.append(x)
    right, w, x = [], d, edges[-1]
    while x < hi:
        w *= grow
        x = min(x + w, hi)
        right.append(x)
    return np.array(left[::-1] + edges + right)


def _build_grid(st: RibStack, d, margin_x=2.0e-6, margin_bot=2.0e-6, margin_top=1.5e-6, inner=0.35e-6):
    keys_y = [0.0, st.h_core] + ([st.h_core + st.h_rib] if st.h_rib > 0 else [])
    keys_x = [-st.width / 2, st.width / 2]
    xe = _axis_edges(keys_x, -st.width / 2 - margin_x, st.width / 2 + margin_x, d, inner, inner)
    ye = _axis_edges(keys_y, -margin_bot, keys_y[-1] + margin_top, d, inner, inner)
    return xe, ye


def _material_maps(st: RibStack, lam, xe, ye, pol="TE"):
    """Cell-averaged eps and chi2 sign s on the grid (rows = y, columns = x)."""
    n = layer_indices(st, lam, pol)
    ss = 1 if st.sidewall_deg == 0 else 4
    def fine(e):
        c = 0.5 * (e[:-1] + e[1:])
        if ss == 1:
            return c, ss
        w = np.diff(e)
        return (e[:-1, None] + (np.arange(ss) + 0.5)[None, :] / ss * w[:, None]).ravel(), ss
    xs, _ = fine(xe)
    ys, _ = fine(ye)
    X, Y = np.meshgrid(xs, ys)
    t = np.tan(np.radians(st.sidewall_deg))
    in_core = (Y >= 0) & (Y < st.h_core)
    in_rib = (Y >= st.h_core) & (Y < st.h_core + st.h_rib) & (np.abs(X) < st.width / 2 - (Y - st.h_core) * t)
    eps = np.where(Y < 0, n["bottom"] ** 2, n["top"] ** 2)
    eps = np.where(in_core, n["core"] ** 2, eps)
    eps = np.where(in_rib, n["rib"] ** 2, eps)
    s = np.where(in_core, 1.0, 0.0) - np.where(in_rib, 1.0, 0.0)
    if ss > 1:
        ny, nx = len(ye) - 1, len(xe) - 1
        eps = eps.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
        s = s.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    return eps, s


def _op_axis(eps, d, axis, weighted, face="arith"):
    """Second-derivative operator along `axis` (1 = x, 0 = y) on the flattened (ny, nx) grid.
    weighted=True gives d/dξ[(1/ε) d(εu)/dξ] (ξ normal to the interfaces), else d²/dξ². Dirichlet box."""
    ny, nx = eps.shape
    e = eps if axis == 1 else eps.T          # work along the last axis
    m, n = e.shape
    dd = np.asarray(d, dtype=float)
    hf = 0.5 * (dd[:-1] + dd[1:])
    if weighted:
        ef = 0.5 * (e[:, :-1] + e[:, 1:]) if face == "arith" else 2 / (1 / e[:, :-1] + 1 / e[:, 1:])
        cf = 1.0 / (ef * hf[None, :])
        up = cf * e[:, 1:] / dd[None, :-1]                    # coupling to i+1
        dn = cf * e[:, :-1] / dd[None, 1:]                    # coupling to i-1
        diag = np.zeros_like(e)
        diag[:, :-1] -= cf * e[:, :-1] / dd[None, :-1]
        diag[:, 1:] -= cf * e[:, 1:] / dd[None, 1:]
    else:
        cf = np.broadcast_to(1.0 / hf[None, :], (m, n - 1))
        up = cf / dd[None, :-1]
        dn = cf / dd[None, 1:]
        diag = np.zeros_like(e)
        diag[:, :-1] -= cf / dd[None, :-1]
        diag[:, 1:] -= cf / dd[None, 1:]
    diag[:, 0] -= 2.0 / dd[0] ** 2
    diag[:, -1] -= 2.0 / dd[-1] ** 2
    # assemble for (m, n) layout, then permute to (ny, nx) layout if axis == 0
    idx = np.arange(m * n).reshape(m, n)
    rows = np.concatenate([idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    cols = np.concatenate([idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()])
    vals = np.concatenate([diag.ravel(), up.ravel(), dn.ravel()])
    if axis == 0:
        perm = np.arange(ny * nx).reshape(ny, nx).T.ravel()   # (m,n) index -> (ny,nx) flat index
        rows, cols = perm[rows], perm[cols]
    return sp.csr_matrix((vals, (rows, cols)), shape=(ny * nx, ny * nx))


def _assemble(eps, dx, dy, k0, pol, face="arith"):
    """Semi-vectorial operator; eigenvalue β². TE: Ex (weighted in x), TM: Ey (weighted in y)."""
    te = pol.upper() == "TE"
    return (_op_axis(eps, dx, 1, weighted=te, face=face) + _op_axis(eps, dy, 0, weighted=not te, face=face)
            + sp.diags(k0**2 * eps.ravel()))


@dataclass
class ModeSet:
    """Result of rib_modes: guided modes of one polarisation at one wavelength."""

    pol: str
    wavelength: float
    neff: np.ndarray
    fields: list              # each (ny, nx), normalised so ∫F² dA = 1, sign fixed (positive at the peak)
    labels: list              # (m_x, m_y): number of lateral / vertical nodes
    xc: np.ndarray
    yc: np.ndarray
    xe: np.ndarray
    ye: np.ndarray
    sign: np.ndarray          # chi2 sign map s(x, y)
    eps: np.ndarray
    notes: list = field(default_factory=list)

    @property
    def cell_area(self):
        return np.outer(np.diff(self.ye), np.diff(self.xe))

    def find(self, label):
        for i, lb in enumerate(self.labels):
            if lb == tuple(label):
                return i
        return None


def _nodes(profile, rel=0.05):
    a = np.abs(profile)
    keep = a > rel * a.max()
    sg = np.sign(profile[keep])
    return int(np.sum(sg[:-1] * sg[1:] < 0)) if sg.size > 1 else 0


def _solve_once(st, lam, pol, nmodes, d, face):
    xe, ye = _build_grid(st, d)
    eps, s = _material_maps(st, lam, xe, ye, pol)
    dx, dy = np.diff(xe), np.diff(ye)
    k0 = 2 * np.pi / lam
    A = _assemble(eps, dx, dy, k0, pol, face)
    sigma = (k0 * np.sqrt(eps.max())) ** 2 * 1.0005
    vals, vecs = spla.eigs(A.tocsc(), k=nmodes, sigma=sigma, which="LM")
    order = np.argsort(-vals.real)
    n_clad = max(np.sqrt(eps[0, 0]), np.sqrt(eps[-1, 0]))
    neff, fields = [], []
    area = np.outer(dy, dx)
    for i in order:
        ne = np.sqrt(max(vals[i].real, 0.0)) / k0
        if ne <= n_clad * 1.0005:
            continue
        f = vecs[:, i].real.reshape(eps.shape)
        f = f / np.sqrt(np.sum(f**2 * area))
        if f.flat[np.argmax(np.abs(f))] < 0:
            f = -f
        neff.append(ne)
        fields.append(f)
    return xe, ye, eps, s, np.array(neff), fields


def rib_modes(st: RibStack, lam: float, pol: str = "TE", nmodes: int = 8, step: float = 30e-9,
              extrapolate: bool = True, face: str = "arith") -> ModeSet:
    """Semi-vectorial FD modes. With extrapolate=True the eigenvalue is also computed on a grid twice as
    coarse and n_eff² is Richardson-extrapolated (error ∝ step²); fields come from the fine grid."""
    st.validate()
    require_positive(wavelength=lam, step=step)
    xe, ye, eps, s, neff, fields = _solve_once(st, lam, pol, nmodes, step, face)
    notes = [f"semi-vectorial FD, {len(xe) - 1}×{len(ye) - 1} cells, step {step * 1e9:.0f} nm"]
    if extrapolate and len(neff):
        _, _, _, _, neff2, _ = _solve_once(st, lam, pol, nmodes, 2 * step, face)
        m = min(len(neff), len(neff2))
        if m:
            neff[:m] = np.sqrt(np.maximum((4 * neff[:m] ** 2 - neff2[:m] ** 2) / 3, 0.0))
            notes.append("n_eff² Richardson-extrapolated from steps h and 2h")
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    labels = []
    for f in fields:
        j, i = np.unravel_index(np.argmax(np.abs(f)), f.shape)
        labels.append((_nodes(f[j, :]), _nodes(f[:, i])))
    return ModeSet(pol=pol.upper(), wavelength=lam, neff=neff, fields=fields, labels=labels, xc=xc, yc=yc,
                   xe=xe, ye=ye, sign=s, eps=eps, notes=notes)


# ----------------------------------------------------------------------------- overlap and SHG figures
def overlap(pump: ModeSet, ip: int, sh: ModeSet, ish: int) -> dict:
    """Layer-poled overlap of a TE pump mode and a TM SH mode on the same (SH-wavelength) grid.

    Both ModeSets must share the grid, so build them with `shg_modes` (same RibStack => same grid)."""
    if pump.fields[ip].shape != sh.fields[ish].shape:
        raise ValueError("pump and SH modes live on different grids")
    area = sh.cell_area
    fp, fs, s = pump.fields[ip], sh.fields[ish], sh.sign
    gam_signed = float(np.sum(s * fp**2 * fs * area))
    gam_plain = float(np.sum(np.abs(s) * fp**2 * fs * area))     # all-positive chi2 in the AlGaAs, same d
    gam_abs = float(np.sum(np.abs(s) * fp**2 * np.abs(fs) * area))  # best case: sign follows the SH field
    return {"gamma": abs(gam_signed), "gamma_uniform": abs(gam_plain), "gamma_max": gam_abs,
            "a_eff": 1.0 / gam_signed**2 if gam_signed != 0 else float("inf")}


def shg_design(st: RibStack, lam_pump: float, pump_label=(0, 0), sh_label=(0, 1), d_eff=100e-12,
               length=None, step=30e-9, extrapolate=True, nmodes=40) -> Result:
    """Modal phase matching, Δk and layer-poled overlap for TE(pump_label) -> TM(sh_label) SHG.

    Labels are (lateral nodes, vertical nodes). Returns n_eff of both modes, Δk, coherence length,
    Γ, A_eff = 1/Γ², and the normalised efficiency η = 2 ω² d_eff² Γ² / (ε0 c³ n_p² n_SH)."""
    require_positive(lam_pump=lam_pump, d_eff=d_eff)
    pump = rib_modes(st, lam_pump, "TE", min(nmodes, 12), step, extrapolate)
    sh = rib_modes(st, lam_pump / 2, "TM", nmodes, step, extrapolate)
    ip, ish = pump.find(pump_label), sh.find(sh_label)
    if ip is None or ish is None:
        raise ValueError(f"mode not found: pump {pump_label} in {pump.labels}, SH {sh_label} in {sh.labels}")
    # pump and SH grids differ (both built from the same step, hence identical): check
    if pump.fields[ip].shape != sh.fields[ish].shape:
        raise ValueError("internal: grids differ")
    ov = overlap(pump, ip, sh, ish)
    n_p, n_s = float(pump.neff[ip]), float(sh.neff[ish])
    dk = 4 * np.pi * (n_s - n_p) / lam_pump
    omega = 2 * np.pi * _C / lam_pump
    eta = 2 * omega**2 * d_eff**2 * ov["gamma"] ** 2 / (_EPS0 * _C**3 * n_p**2 * n_s)   # 1/(W) per L²
    ab = sh_absorption(st, lam_pump / 2)
    vals = {"n_pump": n_p, "n_sh": n_s, "delta_k": dk, "delta_n": n_s - n_p,
            "sh_margin": ab["margin"], "sh_loss_db_per_cm": ab["loss_db_per_cm"],
            "coherence_length": np.pi / abs(dk) if dk != 0 else float("inf"),
            "gamma": ov["gamma"], "gamma_uniform": ov["gamma_uniform"], "gamma_max": ov["gamma_max"],
            "a_eff": ov["a_eff"], "eta_norm": eta}
    units = {"n_pump": "", "n_sh": "", "delta_k": "1/m", "delta_n": "", "sh_margin": "eV", "sh_loss_db_per_cm": "dB/cm", "coherence_length": "m",
             "gamma": "1/m", "gamma_uniform": "1/m", "gamma_max": "1/m", "a_eff": "m²", "eta_norm": "1/(W m²)"}
    if length is not None:
        vals["eta"] = eta * length**2 * float(np.sinc(dk * length / 2 / np.pi) ** 2)
        units["eta"] = "1/W"
    notes = []
    if ab["margin"] < MIN_MARGIN_EV:
        notes.append(f"WARNING: SH absorption margin {ab['margin']:.3f} eV in the {ab['worst']} (< {MIN_MARGIN_EV} eV)")
    return Result(values=vals, units=units,
                  assumptions=notes + pump.notes + sh.notes + [
                      "Semi-vectorial: quasi-TE uses Ex, quasi-TM uses Ey; the weak longitudinal field is neglected",
                      "Scalar normalisation ∫F²dA = 1 per mode; n_eff enters the efficiency prefactor",
                      "TE pump -> TM SH is the only d14 channel for propagation along [110] in (001) AlGaAs; "
                      "d_eff = d14 there (user-supplied; the MQW value is not the bulk one)"])


# ----------------------------------------------------------------------------- design helpers
_PARAMS = ("h_core", "h_rib", "width", "x_core", "x_rib", "dn_rib", "dn_rib_sh")


def _pick(ms: ModeSet, label):
    i = ms.find(label)
    return None if i is None else i


def mode_neff(st: RibStack, lam: float, pol: str, label, step=30e-9, extrapolate=True, nmodes=40) -> float:
    """n_eff of one labelled mode (lateral nodes, vertical nodes); NaN if it is not found/guided."""
    ms = rib_modes(st, lam, pol, nmodes, step, extrapolate)
    i = ms.find(label)
    return float("nan") if i is None else float(ms.neff[i])


def delta_n(st: RibStack, lam_pump: float, pump_label=(0, 0), sh_label=(0, 2), step=30e-9,
            extrapolate=True, nmodes=40) -> float:
    """n_eff(TM sh_label @ λ/2) - n_eff(TE pump_label @ λ): zero at modal phase matching."""
    return (mode_neff(st, lam_pump / 2, "TM", sh_label, step, extrapolate, nmodes)
            - mode_neff(st, lam_pump, "TE", pump_label, step, extrapolate, nmodes))


def tune_parameter(st: RibStack, lam_pump: float, name: str, lo: float, hi: float, pump_label=(0, 0),
                   sh_label=(0, 2), npts=12, step=30e-9, extrapolate=True, nmodes=40):
    """Values of one geometry parameter in [lo, hi] at which Δn = 0 (all sign changes found on an npts scan,
    refined by Brent). Returns a list of (value, slope d(Δn)/d(parameter)) and the scan (values, Δn)."""
    if name not in _PARAMS:
        raise ValueError(f"name must be one of {_PARAMS}")
    vals = np.linspace(lo, hi, npts)
    f = lambda v: delta_n(replace(st, **{name: float(v)}), lam_pump, pump_label, sh_label, step, extrapolate, nmodes)
    scan = np.array([f(v) for v in vals])
    roots = []
    for a in range(npts - 1):
        if np.isfinite(scan[a]) and np.isfinite(scan[a + 1]) and scan[a] * scan[a + 1] < 0:
            try:
                r = brentq(f, vals[a], vals[a + 1], xtol=1e-10 * max(abs(vals[a]), 1e-9) + 1e-12, rtol=1e-9)
            except ValueError:      # a mode vanished (NaN) inside the bracket: not a usable root
                continue
            roots.append((r, (scan[a + 1] - scan[a]) / (vals[a + 1] - vals[a])))
    return roots, (vals, scan)


def sweep(st: RibStack, lam_pump: float, name: str, values, pump_label=(0, 0), sh_label=(0, 2), d_eff=100e-12,
          step=30e-9, extrapolate=True, nmodes=40) -> dict:
    """Δn, Δk, Γ, Γ_uniform, Γ_max, A_eff and η_norm vs one geometry parameter (NaN where a mode is missing)."""
    keys = ("n_pump", "n_sh", "delta_n", "delta_k", "gamma", "gamma_uniform", "gamma_max", "a_eff", "eta_norm")
    out = {k: np.full(len(values), np.nan) for k in keys}
    for i, v in enumerate(values):
        try:
            r = shg_design(replace(st, **{name: float(v)}), lam_pump, pump_label, sh_label, d_eff,
                           step=step, extrapolate=extrapolate, nmodes=nmodes)
        except ValueError:
            continue
        for k in keys:
            out[k][i] = r[k]
    return out


def dispersion(st: RibStack, lams, pol="TE", max_vertical=4, max_lateral=1, step=40e-9, extrapolate=True,
               nmodes=40) -> dict:
    """n_eff(λ) of every labelled mode with lateral order <= max_lateral and vertical order <= max_vertical.
    Returns {(m_x, m_y): array over lams} with NaN where the mode is not guided (or not among the nmodes solved).
    Where two modes share a label (a heuristic miss), the higher-index one is kept."""
    lams = np.asarray(lams, dtype=float)
    out = {(i, j): np.full(len(lams), np.nan) for i in range(max_lateral + 1) for j in range(max_vertical + 1)}
    for k, lam in enumerate(lams):
        try:
            ms = rib_modes(st, lam, pol, nmodes, step, extrapolate)
        except ValueError:
            continue
        for lb, ne in zip(ms.labels, ms.neff):
            if lb in out and np.isnan(out[lb][k]):
                out[lb][k] = ne
    return {lb: v for lb, v in out.items() if np.isfinite(v).any()}


def group_index(lams, neff):
    """n_g = n - λ dn/dλ from a sampled n_eff(λ) (central differences; NaN-aware)."""
    lams = np.asarray(lams, dtype=float)
    neff = np.asarray(neff, dtype=float)
    return neff - lams * np.gradient(neff, lams)


def shg_modal_pm(wavelength, h_core, h_rib, width, x_core=0.35, x_rib=0.35, pump_vertical=0, sh_vertical=1,
                 d_eff=100e-12, length=1e-3) -> Result:
    """Flat-argument wrapper of shg_design for the web calculator: TE(0, pump_vertical) -> TM(0, sh_vertical),
    SiO2 below, air above, vertical side walls, 30 nm grid with Richardson extrapolation."""
    st = RibStack(h_core=h_core, h_rib=h_rib, width=width, x_core=x_core, x_rib=x_rib)
    r = shg_design(st, wavelength, (0, int(pump_vertical)), (0, int(sh_vertical)), d_eff, length=length)
    keep = ("n_pump", "n_sh", "delta_n", "sh_margin", "sh_loss_db_per_cm", "delta_k", "coherence_length", "gamma", "gamma_max", "a_eff", "eta_norm", "eta")
    return Result(values={k: r[k] for k in keep}, units={k: r.units[k] for k in keep}, assumptions=r.assumptions)
