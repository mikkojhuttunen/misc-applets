"""Gas-sensing multipass cells on a chip: an elongated, deformed integrated Herriott cell (the baseline) against a
circular multipass cell (CMPC) of the same footprint area -- the path you gain against the chaos you lose.

CELLS (planar_cell). Baseline: two concave in-plane mirrors at spacing d = L (cell length) with half aperture
a = W/2, re-entrant design R = L / (1 − cos 2πM/N), one window on M1 for input and output, distinct spots
(phase π/N) by default. The mirrors are smooth profile elements, S(η) = conic sag + a3 η³ + a4 η⁴, deformed by
eps4 (quartic) and eps3 (cubic) given as fractions of the paraxial edge sag a²/(2R); eps = 0 is the ideal cell.
Optional flat side mirrors close the cell (walls). Comparison: a regular polygon of n_facets flat facets (the CMPC;
or a smooth circle) with the same area, input port in the middle of facet 0 and an output port out_frac of the
perimeter further on, launched at θc from the input normal. Both cells get the same input beam.

INPUT BEAM. The eigenmode of the undeformed baseline's first round trip at its window (q0; the waveguide taper is
designed for the ideal cell), sampled as rays from its Wigner function: n_side × n_side rays on a quantile grid,
offsets y = (w/2) z_i across the beam and angles y/R_w + (θw/2) z_j (θw = λm/(π w), R_w the wavefront radius),
equal weights; z = Φ⁻¹((i + ½)/n_side) (Acklam's rational approximation, as in the JS port).

PATH (what you gain). Rays are traced port to port (planar_cell.trace_rays); every reflection multiplies by
R_mirror. Detected power T, its power-weighted mean path ⟨L⟩, spread, leaked share. For the gas:
    absorbance per ppm   A_ppm = 100 α_ppm Γ ⟨L⟩        (α_ppm [1/cm] of the strongest listed line, linear regime)
    shot-noise limit     NEC_shot = sqrt(2 q B / (ℜ P_in T)) / A_ppm   [ppm]

CHAOS (what you lose).
  * Lyapunov exponent λ per reflection: twin rays 1e-9 rad apart from the window at θc (planar_cell.twin_divergence,
    least-squares slope of ln separation over its exponential stretch; 0 when there is none: regular motion).
  * Dither visibility V at the exit pass (ray_phase.dither_analysis): the fringe left when the launch angle is
    dithered with amplitude A. Chaotic cells scramble the path within the dither (V → 0, fringes wash out); a
    regular, re-entrant Herriott cell keeps it (V → 1). With a parasitic copy of power fraction f_par the fringe
    floor is 2 sqrt(f_par) V in absorbance (f_par = 1e-6: a 2e-3 fringe before washout):  NEC_fringe = 2 sqrt(f_par) V / A_ppm;  NEC = sqrt(NEC_shot² + NEC_fringe²).

GAUSSIAN-BEAM TWO-MIRROR MODEL. The input q is propagated along the centre ray's traced chords with the local wall
curvature κ and angle of incidence χ at every hit (tangential ABCD: free space [[1, ℓ], [0, 1]], mirror
[[1, 0], [−2κ/cos χ, 1]]; flat facets κ = 0). Per hit: beam radius w_j and the share of its footprint (radius
w_j / cos χ along the wall) that lands on the reflecting part of the same element (not past its ends, not in a
port). T_G = Π shares × R^reflections is the power that stays in one Gaussian beam; w_max/w_in its breathing or
spreading; m_max the largest |(A + D)/2| of the round trips (two consecutive chords) along the path (> 1: locally
unstable); λ_G = ln σ_max(Π M_j)/hits (matrices in units of the mean chord) the linearised growth along the
centre ray -- 0 for a stable two-mirror cell, > 0 where the deformation makes the orbit unstable. A chaotic
cell loses the Gaussian beam (it spreads and splits over facets) while it gains washout.

Deformation sweep (deformation_sweep): the baseline's path, sensitivity and NEC against its chaos (λ, V) as
eps4 grows, with the CMPC as the reference point. SI units unless named otherwise.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range
from ..planar_cell import engine as P
from ..ray_phase import engine as RP

Q_E = 1.602176634e-19
# peak absorption coefficient [1/cm per ppm] and wavelength [m] of the strongest illustrative line (trace_gas,
# 296 K, 1 atm; checked against trace_gas.peak_alpha_per_ppm in the tests)
SPECIES = {"CH4": (1.9404499757559918e-07, 1653.7262586923987e-9), "NH3": (9.070328505920917e-07, 1531.6844239947554e-9),
           "CO2": (1.8544907877951942e-09, 1572.3344606971416e-9), "H2O": (2.9303442119426514e-09, 1528.9584734878601e-9)}

_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def inv_normal(p):
    """Φ⁻¹(p) by Acklam's rational approximation (relative error < 1.2e-9)."""
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > 1 - 0.02425:
        return -inv_normal(1 - p)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def quantiles(n):
    return np.array([inv_normal((i + 0.5) / n) for i in range(int(n))])


def gfrac(lo, hi, s, w):
    """Share of a Gaussian footprint (1/e² radius w, centre s) inside [lo, hi]."""
    k = math.sqrt(2) / w
    return 0.5 * (math.erf(k * (hi - s)) - math.erf(k * (lo - s)))


# ---------------------------------------------------------------- cells
def design_radius(L, N, M):
    """Mirror radius of the re-entrant design with spacing L: L = R (1 − cos 2πM/N)."""
    return L / (1 - math.cos(2 * math.pi * M / N))


def mirror_radius_w(L, R, lam_m):
    """1/e² mode radius on the mirrors of the symmetric two-mirror resonator: w² = λm L / (π sqrt(1 − g²))."""
    g = 1 - L / R
    return math.sqrt(lam_m * L / (math.pi * math.sqrt(1 - g * g)))


def baseline_cell(L=15e-3, W=1.5e-3, N=14, M=3, distinct=True, port_w=0.0, eps4=0.0, eps3=0.0, conic=0.0, deform="both",
                  walls=False, lam_m=1.6537e-6 / 2.0):
    """Elongated (deformed) Herriott cell L × W; port_w = 0: 2.5 × the mode radius on the mirrors."""
    require_positive(L=L, W=W)
    R = design_radius(L, N, M)
    pw = port_w if port_w > 0 else 2.5 * mirror_radius_w(L, R, lam_m)
    a = W / 2
    A = a - pw
    if A <= 0:
        raise ValueError("the window does not fit: make the cell wider or the window narrower")
    c = P.deformed_herriott_cell(R, N, M, A, pw, a, None, L, math.pi / N if distinct else 0.0, conic, eps3, eps4, deform, walls)
    c.meta.update(kind="herriott", L=L, W=W)
    return c


def circle_reference(area, n_facets=24, port_w=150e-6, out_frac=0.37, smooth=False):
    """CMPC of the same area: regular polygon (flat facets, circumradius from N/2 r² sin(2π/N) = area) or a smooth
    circle; input port mid facet 0, output port out_frac × perimeter further on."""
    require_positive(area=area)
    if smooth:
        r = math.sqrt(area / math.pi)
        c = P.circle_cell(r)
    else:
        N = int(n_facets)
        r = math.sqrt(2 * area / (N * math.sin(2 * math.pi / N)))
        c = P.polygon_cell(r, N)
    c.s_out_default = c.s_in_default + out_frac * c.perimeter
    c.meta.update(kind="circle", r=r, port_w=port_w, n_facets=0 if smooth else int(n_facets))
    return c


# ---------------------------------------------------------------- Gaussian-beam two-mirror model
def centre_path(cell, s_in, theta, s_out, port_w, n_max=2000):
    """Centre ray from s_in at θ until it hits the output window (kept) or leaves: chords, curvature and cos χ per hit."""
    xs, ys, S, SC, K = P.trace_path(cell, s_in, theta, n_max, port=(s_out, port_w))
    chords = np.hypot(np.diff(xs), np.diff(ys))
    exits = bool(len(S)) and bool(P.in_window(S[-1], s_out, port_w, cell.perimeter))
    kap = cell.curvature_at(S) if len(S) else np.zeros(0)
    return dict(xs=xs, ys=ys, s=S, sinchi=SC, element=K, chords=chords, kappa=kap, cos=np.sqrt(np.clip(1 - SC**2, 1e-12, 1.0)),
                exits=exits, n_hits=len(S))


def eigen_q(Mrt):
    """q of the eigenmode of a round-trip matrix (None when unstable): 1/q = (D − A)/(2B) − i sqrt(1 − m²)/|B|."""
    A, B, C, D = Mrt[0, 0], Mrt[0, 1], Mrt[1, 0], Mrt[1, 1]
    m = 0.5 * (A + D)
    if abs(m) >= 1 or B == 0:
        return None
    return 1 / complex((D - A) / (2 * B), -math.sqrt(1 - m * m) / abs(B))


def input_q(L=15e-3, W=1.5e-3, N=14, M=3, distinct=True, port_w=0.0, lam_m=1.6537e-6 / 2.0, walls=False):
    """Input beam: eigenmode of the first round trip of the undeformed baseline at its window, and that cell."""
    c0 = baseline_cell(L, W, N, M, distinct, port_w, lam_m=lam_m, walls=walls)
    p = centre_path(c0, c0.s_in_default, c0.meta["theta_launch"], c0.s_out_default, c0.meta["port_w"], 3)
    Mt = np.eye(2)
    for j in range(2):
        Mt = np.array([[1.0, p["chords"][j]], [0.0, 1.0]]) @ Mt
        Mt = np.array([[1.0, 0.0], [-2 * p["kappa"][j] / p["cos"][j], 1.0]]) @ Mt
    q = eigen_q(Mt)
    if q is None:
        raise ValueError("the undeformed baseline is not a stable two-mirror cell")
    return q, c0


def q_width(q, lam_m):
    """(1/e² radius, wavefront radius) of the Gaussian beam q."""
    iq = 1 / q
    return math.sqrt(-lam_m / (math.pi * iq.imag)), (1 / iq.real if iq.real != 0 else math.inf)


def gaussian_path(cell, path, q0, lam_m, ports, R_mirror=1.0):
    """q along the traced centre path (centre_path): radius per hit, footprint share on the reflecting element,
    T_G, w_max/w_in and the linearised growth λ_G (module docstring). ports: list of (s0, width) windows."""
    n = path["n_hits"]
    q = q0
    w_in = q_width(q0, lam_m)[0]
    ell = float(np.mean(path["chords"])) if n else 1.0
    Mt = np.eye(2)
    W, share = np.zeros(n), np.ones(n)
    T = 1.0
    steps = []                                                    # one matrix per chord + reflection
    for j in range(n):
        q = q + path["chords"][j]
        Mt = np.array([[1.0, path["chords"][j] / ell], [0.0, 1.0]]) @ Mt
        steps.append(np.array([[1.0, path["chords"][j] / ell], [0.0, 1.0]]))
        W[j] = q_width(q, lam_m)[0]
        last = j == n - 1 and path["exits"]
        if not last:
            k = int(path["element"][j])
            s, wf = path["s"][j], W[j] / path["cos"][j]
            lo, hi = cell.s0[k], cell.s0[k] + cell.L[k]
            sh = gfrac(lo, hi, s, wf)
            for s0, pw in ports:
                d = (s0 - s + cell.perimeter / 2) % cell.perimeter - cell.perimeter / 2
                sh -= gfrac(max(s + d - pw / 2, lo), min(s + d + pw / 2, hi), s, wf) if s + d + pw / 2 > lo and s + d - pw / 2 < hi else 0.0
            share[j] = max(sh, 0.0)
            T *= share[j] * R_mirror
            Cm = -2 * path["kappa"][j] / path["cos"][j]
            q = q / (Cm * q + 1)
            Mt = np.array([[1.0, 0.0], [Cm * ell, 1.0]]) @ Mt
            steps[-1] = np.array([[1.0, 0.0], [Cm * ell, 1.0]]) @ steps[-1]
    sv = np.linalg.svd(Mt, compute_uv=False)[0] if n else 1.0
    m_rt = [0.5 * np.trace(steps[j + 1] @ steps[j]) for j in range(len(steps) - 1)] if len(steps) > 1 else [1.0]
    return dict(w=W, share=share, T_G=float(T), w_max_ratio=float(W.max() / w_in) if n else 1.0,
                lam_G=float(math.log(sv) / n) if n else 0.0, m_max=float(np.max(np.abs(m_rt))), w_in=w_in, q_end=q)


# ---------------------------------------------------------------- ray ensemble, chaos, figures
def beam_rays(cell, s_in, theta_c, q0, lam_m, n_side=24):
    """Launch rays sampling the Gaussian beam q0 (Wigner function) at the window: (x, y, dx, dy)."""
    w, Rw = q_width(q0, lam_m)
    thw = lam_m / (math.pi * w)
    z = quantiles(n_side)
    _, _, nx, ny = cell.point_at(np.array([s_in]))
    cos_t = max(math.cos(theta_c), 1e-6)
    off = np.repeat(w / 2 * z, len(z))
    ang = np.tile(thw / 2 * z, len(z)) + (off / Rw if math.isfinite(Rw) else 0.0)
    return P.launch(cell, s_in + off / cos_t, theta_c + ang)


def cell_figures(cell, theta_c, q0, lam_m, n_eff=2.0, gamma=0.2, alpha_ppm=SPECIES["CH4"][0], R_mirror=0.995, dither=2e-3,
                 waveform="sine", f_par=1e-6, P_in=1e-3, bandwidth=1.0, n_side=24, n_bounce=2000, n_lyap=300, n_pass_max=60,
                 n_dither_max=12801):
    """All figures of one cell (module docstring) as a dict."""
    s_in, s_out = cell.s_in_default, cell.s_out_default
    pw = cell.meta["port_w"]
    lam = lam_m * n_eff
    ports = [(s_in, pw)] + ([] if abs(s_out - s_in) < 1e-15 else [(s_out, pw)])
    # path: ray ensemble port to port
    tab = P.trace_rays(cell, pw, n_bounce=n_bounce, s_in=s_in, s_out=s_out, starts=beam_rays(cell, s_in, theta_c, q0, lam_m, n_side))
    ev = P.evaluate(tab, (lambda x: np.full(np.shape(x), R_mirror)) if R_mirror < 1 else None, 0.0, gamma)
    L_rms = float(np.sqrt((ev.W * (ev.L - ev.L_mean) ** 2).sum() / ev.W.sum())) if ev.W.size and ev.W.sum() > 0 else math.nan
    A_ppm = 100 * alpha_ppm * gamma * ev.L_mean if ev.T_det > 0 else 0.0
    sig_A = math.sqrt(2 * Q_E * bandwidth / (P_in * ev.T_det)) if ev.T_det > 0 else math.inf
    # Gaussian-beam model along the centre ray
    cp = centre_path(cell, s_in, theta_c, s_out, pw, n_bounce)
    gb = gaussian_path(cell, cp, q0, lam_m, ports, R_mirror)
    # chaos: twin rays and dither visibility at the exit pass
    sep, fit = P.twin_divergence(cell, s_in, theta_c, n_lyap)
    n_pass = int(min(cp["n_hits"], n_pass_max)) if cp["n_hits"] else 1
    da = RP.dither_analysis(cell, theta_c, dither, n_pass, lam, n_eff, waveform, n_max=n_dither_max, s_in=s_in)
    V = float(da["V"][n_pass - 1])
    nec_shot = sig_A / A_ppm if A_ppm > 0 else math.inf
    nec_fringe = 2 * math.sqrt(f_par) * V / A_ppm if A_ppm > 0 else math.inf
    return dict(T=float(ev.T_det), L_mean=float(ev.L_mean), L_rms=L_rms, S1=float(ev.S1), leaked=tab.leaked_fraction,
                back=float(np.mean(tab.exit_port == 0)), trapped=tab.trapped_fraction, A_ppm=A_ppm, nec_shot=nec_shot,
                nec_fringe=nec_fringe, nec=math.hypot(nec_shot, nec_fringe), lam=float(fit["lam"]) if fit else 0.0,
                lam_fit=fit is not None, V=V, V_pass=da["V"], same_path=float(da["same_path"][n_pass - 1]), n_pass=n_pass,
                exit_pass=cp["n_hits"] if cp["exits"] else -1, L_centre=float(cp["chords"].sum()), T_G=gb["T_G"],
                w_max_ratio=gb["w_max_ratio"], lam_G=gb["lam_G"], m_max=gb["m_max"], theta=float(theta_c), w_hits=gb["w"], share=gb["share"], sep=sep, centre=cp,
                area=float(cell.area), dither_n=int(da["n_used"]))


def retune(cell, theta0, q0, lam_m, span=None, n_theta=41, R_mirror=0.995, gamma=0.2, n_side=8, n_bounce=2000):
    """Launch angle within theta0 ± span (default ± 4 beam divergences) that maximises the sensitivity Γ T ⟨L⟩ of a
    small ray ensemble (n_side²), and that maximum. A deformed mirror shifts the spot pattern off re-entrance; this
    re-aims the input before the cells are compared."""
    w = q_width(q0, lam_m)[0]
    span = 4 * lam_m / (math.pi * w) if span is None else span
    pw = cell.meta["port_w"]
    best = (-1.0, theta0)
    for th in theta0 + span * np.linspace(-1, 1, int(n_theta)):
        tab = P.trace_rays(cell, pw, n_bounce=n_bounce, s_in=cell.s_in_default, s_out=cell.s_out_default,
                           starts=beam_rays(cell, cell.s_in_default, th, q0, lam_m, n_side))
        ev = P.evaluate(tab, (lambda x: np.full(np.shape(x), R_mirror)) if R_mirror < 1 else None, 0.0, gamma)
        if ev.S1 > best[0] * (1 + 1e-12):
            best = (float(ev.S1), float(th))
    return best[1], best[0]


def _common(L, W, N, M, distinct, port_width, eps4, eps3, conic, deform, walls, n_facets, out_frac, smooth_circle,
            wavelength, n_eff, species):
    require_positive(wavelength=wavelength, n_eff=n_eff)
    require_choice("species", species, tuple(SPECIES))
    lam_m = wavelength / n_eff
    q0, c0 = input_q(L, W, N, M, bool(distinct), port_width, lam_m, bool(walls))
    pw = c0.meta["port_w"]
    base = baseline_cell(L, W, N, M, bool(distinct), pw, eps4, eps3, conic, deform, bool(walls), lam_m)
    circ = circle_reference(base.area, n_facets, pw, out_frac, bool(smooth_circle))
    return lam_m, q0, base, circ


_KEYS = ("T", "L_mean", "L_rms", "S1", "A_ppm", "nec_shot", "nec_fringe", "nec", "lam", "V", "same_path", "n_pass",
         "exit_pass", "T_G", "w_max_ratio", "lam_G", "m_max", "leaked", "area", "theta")
_UNITS = dict(theta="rad", L_mean="m", L_rms="m", S1="m", A_ppm="1/ppm", nec_shot="ppm", nec_fringe="ppm", nec="ppm", lam="1/hit",
              lam_G="1/hit", area="m^2")


def compare(L=15e-3, W=1.5e-3, N=14, M=3, distinct=1, port_width=0.0, eps4=0.0, eps3=0.0, conic=0.0, deform="both", walls=0,
            n_facets=24, circle_theta=0.35, out_frac=0.37, smooth_circle=0, wavelength=1.6537e-6, n_eff=2.0, gamma=0.2,
            species="CH4", R_mirror=0.995, dither=2e-3, waveform="sine", f_parasitic=1e-6, P_in=1e-3, bandwidth=1.0,
            n_side=24, n_bounce=2000, retune_launch=1) -> Result:
    """Elongated deformed Herriott cell (h_) against the CMPC of the same area (c_): path, sensitivity and detection
    limits against chaos (Lyapunov exponent, dither visibility) and the Gaussian-beam model, with the ratios.
    retune_launch: re-aim the baseline's launch angle for the largest Γ T ⟨L⟩ (retune) before comparing."""
    require_range("R_mirror", R_mirror, 0.0, 1.0)
    require_nonnegative(dither=dither, f_parasitic=f_parasitic)
    lam_m, q0, base, circ = _common(L, W, N, M, distinct, port_width, eps4, eps3, conic, deform, walls, n_facets, out_frac,
                                    smooth_circle, wavelength, n_eff, species)
    kw = dict(n_eff=n_eff, gamma=gamma, alpha_ppm=SPECIES[species][0], R_mirror=R_mirror, dither=dither, waveform=waveform,
              f_par=f_parasitic, P_in=P_in, bandwidth=bandwidth, n_side=n_side, n_bounce=n_bounce)
    th = retune(base, base.meta["theta_launch"], q0, lam_m, R_mirror=R_mirror, gamma=gamma)[0] if retune_launch else base.meta["theta_launch"]
    h = cell_figures(base, th, q0, lam_m, **kw)
    c = cell_figures(circ, circle_theta, q0, lam_m, **kw)
    v = {f"h_{k}": h[k] for k in _KEYS}
    v.update({f"c_{k}": c[k] for k in _KEYS})
    v.update(path_gain=h["L_mean"] / c["L_mean"] if c["L_mean"] > 0 else math.nan,
             sensitivity_gain=h["S1"] / c["S1"] if c["S1"] > 0 else math.nan,
             chaos_lost=c["lam"] - h["lam"], washout_lost=h["V"] - c["V"],
             nec_ratio=h["nec"] / c["nec"] if c["nec"] > 0 else math.nan,
             R_design=base.meta["R"], port_width=base.meta["port_w"], w_in=q_width(q0, lam_m)[0], circle_radius=circ.meta["r"])
    u = {k: "" for k in v}
    for p_ in ("h_", "c_"):
        u.update({p_ + k: un for k, un in _UNITS.items()})
    u.update(R_design="m", port_width="m", w_in="m", circle_radius="m")
    return Result(values=v, units=u, assumptions=[
        "2D ray optics in the slab (n_eff); mirrors: constant reflectance R_mirror at every hit, smooth profile mirrors "
        "for the baseline, flat facets for the CMPC",
        "Same input beam for both cells: eigenmode of the undeformed baseline at its window, sampled on a quantile grid",
        "Absorbance per ppm from the strongest illustrative line (trace_gas, not HITRAN), linear regime, gas overlap Γ",
        "Fringe floor 2 sqrt(f_parasitic) V from one parasitic copy whose phase follows the launch dither; shot noise "
        "with responsivity 1 A/W",
        "Lyapunov exponent from twin rays at the launch angle (0 when no exponential stretch); λ_G from the linearised "
        "(ABCD) map along the centre ray"])


def deformation_sweep(eps4_values=(0.0, 0.02, 0.05, 0.1, 0.15), L=15e-3, W=1.5e-3, N=14, M=3, distinct=1, port_width=0.0,
                      eps3=0.0, conic=0.0, deform="both", walls=0, n_facets=24, circle_theta=0.35, out_frac=0.37,
                      smooth_circle=0, wavelength=1.6537e-6, n_eff=2.0, gamma=0.2, species="CH4", R_mirror=0.995,
                      dither=2e-3, waveform="sine", f_parasitic=1e-6, P_in=1e-3, bandwidth=1.0, n_side=16,
                      n_bounce=2000, retune_launch=1) -> Result:
    """Path, sensitivity and NEC of the baseline against its chaos (λ, V, λ_G) as the quartic deformation grows, with
    the CMPC of the same area as the reference (c_*)."""
    lam_m, q0, _, circ = _common(L, W, N, M, distinct, port_width, 0.0, eps3, conic, deform, walls, n_facets, out_frac,
                                 smooth_circle, wavelength, n_eff, species)
    kw = dict(n_eff=n_eff, gamma=gamma, alpha_ppm=SPECIES[species][0], R_mirror=R_mirror, dither=dither, waveform=waveform,
              f_par=f_parasitic, P_in=P_in, bandwidth=bandwidth, n_side=n_side, n_bounce=n_bounce)
    pw = circ.meta["port_w"]
    rows = []
    for e in eps4_values:
        b = baseline_cell(L, W, N, M, bool(distinct), pw, float(e), eps3, conic, deform, bool(walls), lam_m)
        th = retune(b, b.meta["theta_launch"], q0, lam_m, R_mirror=R_mirror, gamma=gamma)[0] if retune_launch else b.meta["theta_launch"]
        rows.append(cell_figures(b, th, q0, lam_m, **kw))
    c = cell_figures(circ, circle_theta, q0, lam_m, **kw)
    v = {"eps4": np.asarray(eps4_values, float)}
    v.update({k: np.array([r[k] for r in rows], float) for k in _KEYS})
    v.update({f"c_{k}": c[k] for k in _KEYS})
    u = {k: _UNITS.get(k, "") for k in v}
    u.update({f"c_{k}": _UNITS.get(k, "") for k in _KEYS})
    return Result(values=v, units=u, assumptions=["As compare(); one row per eps4 (quartic deformation, fraction of the edge sag)"])
