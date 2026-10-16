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
  * Dither visibility V at the exit pass (ray_phase.dither_analysis; V_floor = √π/(2√n), the level n random phases
    leave: a V near it is unresolved and the fringe-limited NEC an upper bound): the fringe left when the launch angle is
    dithered with amplitude A. Chaotic cells scramble the path within the dither (V → 0, fringes wash out); a
    regular, re-entrant Herriott cell keeps it (V → 1). With a parasitic copy of power fraction f_par the fringe
    floor is 2 sqrt(f_par) V in absorbance (f_par = 1e-6: a 2e-3 fringe before washout):  NEC_fringe = 2 sqrt(f_par) V / A_ppm;  NEC = sqrt(NEC_shot² + NEC_fringe²).

RIPPLED END MIRRORS AND THE CHAOS MAP. ripple_a sin(2πη/Λ + φ) on the mirror profile is a standard map of stochasticity
K = 8π² a L / Λ² (K ≈ 1 for 0.3 µm at Λ = 0.5 mm, L = 10 mm). chaos_map runs the exact tangent map of the walls along the
launch beam's rays and a grid of departures from M1 (FTLE, MEGNO ⟨Y⟩ > 4: chaotic) and says whether the beam launches into
the chaotic sea or onto a regular island (tangent_chaos, chaos_map).

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
                  walls=False, lam_m=1.6537e-6 / 2.0, n_seg=0, seg_curv=1.0, seg_tilt=0.0, seg_seed=1, ripple_a=0.0,
                  ripple_period=0.5e-3, ripple_phase=0.0, ripple_on="both"):
    """Elongated (deformed, optionally segmented or rippled) Herriott cell L × W; port_w = 0: 2.5 × the mode radius on
    the mirrors. n_seg, seg_curv, seg_tilt, seg_seed: curved segmented mirrors; ripple_a, ripple_period, ripple_phase,
    ripple_on: sinusoidal ripple on the end mirrors, stochasticity K = 8π² a L / Λ² (planar_cell.deformed_herriott_cell)."""
    require_positive(L=L, W=W)
    R = design_radius(L, N, M)
    pw = port_w if port_w > 0 else 2.5 * mirror_radius_w(L, R, lam_m)
    a = W / 2
    A = a - pw
    if A <= 0:
        raise ValueError("the window does not fit: make the cell wider or the window narrower")
    c = P.deformed_herriott_cell(R, N, M, A, pw, a, None, L, math.pi / N if distinct else 0.0, conic, eps3, eps4, deform, walls,
                                 n_seg, seg_curv, seg_tilt, seg_seed, ripple_a, ripple_period, ripple_phase, ripple_on)
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


# ---------------------------------------------------------------- index perturbations inside the cell
_HASH_M = 2147483647


def place_inclusions(cell, n_inc, radius=50e-6, dn=0.01, sign="both", seed=1, gap=None):
    """n_inc circular regions of radius `radius` and index change ±dn (sign 'both' random, 'plus', 'minus') at random,
    non-overlapping positions inside the cell, kept `gap` (default radius/2) clear of each other and of the mirrors
    (mulberry32, seed: the JS port places the same ones). Herriott cells: between the mirror edge chords, |y| < a;
    circular cells: inside the inscribed circle. Returns dict(cx, cy, r, dn) (may hold fewer than n_inc if full)."""
    require_choice("sign", sign, ("both", "plus", "minus"))
    gap = 0.5 * radius if gap is None else gap
    m = cell.meta
    u = P.mulberry32(seed)
    if m.get("kind") == "circle":
        rmax = m["r"] * (math.cos(math.pi / m["n_facets"]) if m.get("n_facets") else 1.0) - radius - gap
        box = (-rmax, rmax, -rmax, rmax)
        inside = lambda x, y: x * x + y * y <= rmax * rmax
    else:
        xe = max(abs(float(cell.A[k][0])) for k in range(cell.n_elements) if cell.labels[k].startswith("M"))
        xe = min(xe, min(abs(float(cell.B[k][0])) for k in range(cell.n_elements) if cell.labels[k].startswith("M")))
        xm, ym = xe - radius - gap, m["aperture"] - radius - gap
        box = (-xm, xm, -ym, ym)
        inside = lambda x, y: True
    cx, cy, dd = [], [], []
    if box[1] <= box[0] or box[3] <= box[2]:
        return dict(cx=cx, cy=cy, r=[], dn=[])
    tries = 0
    while len(cx) < int(n_inc) and tries < 400 * max(int(n_inc), 1):
        tries += 1
        x = box[0] + (box[1] - box[0]) * u()
        y = box[2] + (box[3] - box[2]) * u()
        if not inside(x, y):
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < (2 * radius + gap) ** 2 for a, b in zip(cx, cy)):
            continue
        sg = (-1.0 if u() < 0.5 else 1.0) if sign == "both" else (1.0 if sign == "plus" else -1.0)
        cx.append(x); cy.append(y); dd.append(sg * dn)
    return dict(cx=cx, cy=cy, r=[radius] * len(cx), dn=dd)


class MediumCell:
    """A planar_cell wall with circular index inclusions (Δn_eff on n0) inside: 2D ray optics with refraction
    (Snell, total internal reflection), geometric and optical path. Everything else is delegated to the wall."""

    def __init__(self, cell, inc=None, n0=1.0):
        self.cell = cell
        inc = inc or dict(cx=[], cy=[], r=[], dn=[])
        self.cx, self.cy, self.r, self.dn = (np.asarray(inc[k], float) for k in ("cx", "cy", "r", "dn"))
        self.n0 = float(n0)
        self.n_inc = int(self.cx.size)

    def __getattr__(self, name):
        return getattr(self.__dict__["cell"], name)

    def path_lengths(self, theta, n_pass, s_in=None, return_ids=False):
        """ray_phase interface: optical path / n0 to each of the first n_pass wall hits (NaN once leaked) and path ids
        (hash of the wall elements and the inclusions entered)."""
        s_in = self.cell.s_in_default if s_in is None else s_in
        th = np.atleast_1d(np.asarray(theta, float))
        x, y, dx, dy = P.launch(self.cell, np.full(th.shape, s_in), th)
        r = march(self, x, y, dx, dy, int(n_pass))
        return (r["WL"], r["ID"]) if return_ids else r["WL"]


def as_medium(cell, n0=1.0):
    return cell if isinstance(cell, MediumCell) else MediumCell(cell, None, n0)


def march(mc, x, y, dx, dy, max_walls, s_in=None, s_out=None, port_w=0.0, ports="none", record=False, max_events=None):
    """Trace rays through the medium cell until max_walls wall hits, a port (ports 'both': output checked first, then
    input; 'out': output only) or a leak. Per ray and wall hit: optical path / n0 (WL), geometric path (WG), s, signed
    sin χ, element, point, path id; exit code (1 output, 0 input, -1 still inside, -2 leaked) and the number of hits.
    record=True keeps every event point (wall hits and inclusion crossings) per ray."""
    x, y, dx, dy = (np.array(np.atleast_1d(v), float) for v in (x, y, dx, dy))
    n = x.size
    cell = mc.cell
    Pm = cell.perimeter
    eps = 1e-12 * Pm
    nw_max = int(max_walls)
    shp = (n, nw_max)
    WL, WG, WS, WC, WX, WY = (np.full(shp, np.nan) for _ in range(6))
    WK, ID = np.full(shp, -1), np.full(shp, -1.0)
    inside = np.full(n, -1)
    nw = np.zeros(n, int)
    exit_ = np.full(n, -1)
    Lo, Lg, pid = np.zeros(n), np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    pts = [[(float(x[i]), float(y[i]))] for i in range(n)] if record else None
    max_events = max_events or nw_max * 40 + 200
    for _ in range(max_events):
        ia = np.nonzero(alive)[0]
        if not ia.size:
            break
        xa, ya, dxa, dya, ins = x[ia], y[ia], dx[ia], dy[ia], inside[ia]
        m = ia.size
        out = ins < 0
        tw = np.full(m, np.inf)
        nxw, nyw, sw, kw = np.zeros(m), np.zeros(m), np.zeros(m), np.full(m, -1)
        if out.any():
            t_, nx_, ny_, s_, k_ = cell.hit_k(xa[out], ya[out], dxa[out], dya[out])
            tw[out], nxw[out], nyw[out], sw[out], kw[out] = t_, nx_, ny_, s_, k_
        ti = np.full(m, np.inf)
        ji = np.full(m, -1)
        if mc.n_inc:
            bx, by = xa[:, None] - mc.cx[None, :], ya[:, None] - mc.cy[None, :]
            b = bx * dxa[:, None] + by * dya[:, None]
            c = bx * bx + by * by - (mc.r * mc.r)[None, :]
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0.0))
            t_in = np.where((disc > 0) & out[:, None], -b - sq, np.inf)
            t_in = np.where(t_in > eps, t_in, np.inf)
            jmin = np.argmin(t_in, axis=1)
            tmin = t_in[np.arange(m), jmin]
            ti = np.where(out, tmin, ti)
            ji = np.where(out & np.isfinite(tmin), jmin, ji)
            jj = np.maximum(ins, 0)
            t_ex = -b[np.arange(m), jj] + sq[np.arange(m), jj]
            ti = np.where(~out, t_ex, ti)
            ji = np.where(~out, ins, ji)
        inc_ev = ti < tw
        leak = ~inc_ev & ~np.isfinite(tw)
        t = np.where(inc_ev, ti, np.where(leak, 0.0, tw))
        nmed = mc.n0 + np.where(ins >= 0, mc.dn[np.maximum(ins, 0)] if mc.n_inc else 0.0, 0.0)
        x[ia], y[ia] = xa + t * dxa, ya + t * dya
        Lg[ia] += t
        Lo[ia] += t * nmed / mc.n0
        if record:
            for q, i in enumerate(ia):
                if not leak[q]:
                    pts[i].append((float(x[i]), float(y[i])))
        if leak.any():
            exit_[ia[leak]] = -2
            alive[ia[leak]] = False
            pid[ia[leak]] = -1.0
        sel = np.nonzero(inc_ev)[0]
        if sel.size:                                           # refraction at an inclusion edge (or TIR)
            i = ia[sel]
            j = ji[sel]
            nxn, nyn = (x[i] - mc.cx[j]) / mc.r[j], (y[i] - mc.cy[j]) / mc.r[j]
            ent = inside[i] < 0
            n1 = np.where(ent, mc.n0, mc.n0 + mc.dn[j])
            n2 = np.where(ent, mc.n0 + mc.dn[j], mc.n0)
            Nx, Ny = np.where(ent, nxn, -nxn), np.where(ent, nyn, -nyn)
            cosi = -(dx[i] * Nx + dy[i] * Ny)
            eta = n1 / n2
            kk = 1 - eta * eta * (1 - cosi * cosi)
            tir = kk < 0
            sk = np.sqrt(np.maximum(kk, 0.0))
            rx = np.where(tir, dx[i] + 2 * cosi * Nx, eta * dx[i] + (eta * cosi - sk) * Nx)
            ry = np.where(tir, dy[i] + 2 * cosi * Ny, eta * dy[i] + (eta * cosi - sk) * Ny)
            nn = np.hypot(rx, ry)
            dx[i], dy[i] = rx / nn, ry / nn
            newin = np.where(tir, inside[i], np.where(ent, j, -1))
            entered = ent & ~tir
            pid[i] = np.where(entered & (pid[i] >= 0), np.mod(pid[i] * 1000003 + 100000 + j, _HASH_M), pid[i])
            inside[i] = newin
        wsel = np.nonzero(~inc_ev & ~leak)[0]
        if wsel.size:                                          # mirror (wall) hit
            i = ia[wsel]
            jw = nw[i]
            nxi, nyi = nxw[wsel], nyw[wsel]
            WL[i, jw], WG[i, jw], WS[i, jw] = Lo[i], Lg[i], sw[wsel]
            WC[i, jw] = dx[i] * nyi - dy[i] * nxi
            WK[i, jw], WX[i, jw], WY[i, jw] = kw[wsel], x[i], y[i]
            pid[i] = np.where(pid[i] >= 0, np.mod(pid[i] * 1000003 + kw[wsel] + 1, _HASH_M), pid[i])
            ID[i, jw] = pid[i]
            nw[i] += 1
            done = np.zeros(i.size, bool)
            if ports in ("both", "out"):
                pout = P.in_window(sw[wsel], s_out, port_w, Pm)
                pin = P.in_window(sw[wsel], s_in, port_w, Pm) if ports == "both" else np.zeros(i.size, bool)
                done = pout | pin
                exit_[i[done]] = np.where(pout[done], 1, 0)
                alive[i[done]] = False
            go = ~done
            dx[i[go]], dy[i[go]] = P.reflect(dx[i[go]], dy[i[go]], nxi[go], nyi[go])
            full = go & (nw[i] >= nw_max)
            alive[i[full]] = False
    return dict(WL=WL, WG=WG, WS=WS, WC=WC, WK=WK, WX=WX, WY=WY, ID=ID, n_walls=nw, exit=exit_, L_opt=Lo, L_geo=Lg, pts=pts)


# ---------------------------------------------------------------- Gaussian-beam two-mirror model
def centre_path(cell, s_in, theta, s_out, port_w, n_max=2000):
    """Centre ray from s_in at θ until it hits the output window (kept) or leaves: chords (geometric, between wall
    hits), curvature and cos χ per hit, every event point (pts: inclusion crossings too) and its optical path."""
    mc = as_medium(cell)
    x, y, dx, dy = P.launch(mc.cell, np.array([s_in]), np.array([theta]))
    r = march(mc, x, y, dx, dy, n_max, s_in, s_out, port_w, "out", record=True)
    nh = int(r["n_walls"][0])
    S, SC, K = r["WS"][0, :nh], r["WC"][0, :nh], r["WK"][0, :nh]
    G = r["WG"][0, :nh]
    chords = np.diff(np.r_[0.0, G])
    exits = bool(r["exit"][0] == 1)
    kap = mc.cell.curvature_at(S) if nh else np.zeros(0)
    pts = np.array(r["pts"][0])
    return dict(xs=pts[:, 0], ys=pts[:, 1], hx=np.r_[x[0], r["WX"][0, :nh]], hy=np.r_[y[0], r["WY"][0, :nh]], s=S, sinchi=SC,
                element=K, chords=chords, kappa=kap, cos=np.sqrt(np.clip(1 - SC**2, 1e-12, 1.0)), exits=exits, n_hits=nh,
                L_opt=float(r["WL"][0, nh - 1]) if nh else 0.0)


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



def ensemble(mc, starts, port_w, s_in, s_out, n_bounce, R_mirror, gamma, lam_m):
    """Port-to-port statistics of a ray ensemble (equal launch weights, R per reflection before the exit): T, mean
    geometric path ⟨L⟩, rms, Γ T ⟨L⟩, shares that leak / return / stay, and the M² of the detected light from the
    second moments of its exit offsets (along the wall × cos χ) and angles: M² = (4π/λm) √(⟨y²⟩⟨u²⟩ − ⟨yu⟩²)."""
    x, y, dx, dy = starts
    r = march(mc, x, y, dx, dy, n_bounce, s_in, s_out, port_w, "both")
    n = x.size
    det = np.nonzero(r["exit"] == 1)[0]
    nh = r["n_walls"][det]
    L = r["WG"][det, nh - 1] if det.size else np.zeros(0)
    W = R_mirror ** (nh - 1.0) if R_mirror < 1 else np.ones(det.size)
    out = dict(leaked=float(np.mean(r["exit"] == -2)), back=float(np.mean(r["exit"] == 0)), trapped=float(np.mean(r["exit"] == -1)),
               L=L, W=W)
    sw = W.sum()
    if sw <= 0:
        out.update(T=0.0, L_mean=math.nan, L_rms=math.nan, S1=0.0, M2=math.nan)
        return out
    Lm = float((W * L).sum() / sw)
    sc = r["WC"][det, nh - 1]
    ds = (r["WS"][det, nh - 1] - s_out + mc.perimeter / 2) % mc.perimeter - mc.perimeter / 2
    yv, uv = ds * np.sqrt(np.clip(1 - sc**2, 0, 1)), np.arcsin(np.clip(sc, -1, 1))
    my, mu = (W * yv).sum() / sw, (W * uv).sum() / sw
    cyy, cuu, cyu = (W * (yv - my) ** 2).sum() / sw, (W * (uv - mu) ** 2).sum() / sw, (W * (yv - my) * (uv - mu)).sum() / sw
    out.update(T=float(sw / n), L_mean=Lm, L_rms=float(np.sqrt((W * (L - Lm) ** 2).sum() / sw)), S1=float(gamma * sw / n * Lm),
               M2=float(4 * math.pi / lam_m * math.sqrt(max(cyy * cuu - cyu * cyu, 0.0))))
    return out


def twin_divergence(mc, s_in, theta, n_hits=300, dtheta=1e-9):
    """Hit-point separation of two rays dtheta apart (no ports) and its Lyapunov fit (planar_cell.lyapunov_fit)."""
    x, y, dx, dy = P.launch(mc.cell, np.array([s_in, s_in]), np.array([theta, theta + dtheta]))
    r = march(mc, x, y, dx, dy, n_hits)
    n = int(min(r["n_walls"]))
    sep = np.hypot(r["WX"][0, :n] - r["WX"][1, :n], r["WY"][0, :n] - r["WY"][1, :n])
    return sep, P.lyapunov_fit(sep, scale=math.sqrt(abs(mc.area)))


def tangent_chaos(cell, x, y, dx, dy, n_hits=300):
    """Finite-time chaos indicators of rays (walls only, no ports, inclusions ignored) from the exact tangent map of the
    billiard, unfolded: flight [[1, ℓ], [0, 1]], reflection [[1, 0], [−2κ/cos χ, 1]] with the local wall curvature κ (ripple
    included) and angle χ, ℓ in units of sqrt(area); the tangent vector is renormalised every hit. Per ray: FTLE λ =
    Σ ln g_k / n per hit, MEGNO ⟨Y⟩ = mean over n of (2/n) Σ k ln g_k (→ 0 periodic, → 2 quasi-periodic (island, KAM
    torus), ≈ λ n / 2 chaotic), and the number of hits before the ray left (n_hits: stayed)."""
    mc = as_medium(cell.cell if isinstance(cell, MediumCell) else cell)
    n = int(n_hits)
    r = march(mc, x, y, dx, dy, n)
    nh = r["n_walls"]
    m = nh.size
    ch = np.diff(np.c_[np.zeros(m), np.nan_to_num(r["WG"])], axis=1)
    S = r["WS"]
    fin = np.isfinite(S)
    kap = np.zeros(S.shape)
    if fin.any():
        kap[fin] = mc.cell.curvature_at(S[fin])
    cs = np.sqrt(np.clip(1 - np.nan_to_num(r["WC"]) ** 2, 1e-12, 1.0))
    ell = math.sqrt(abs(mc.area))
    u, v = np.full(m, math.sqrt(0.5)), np.full(m, math.sqrt(0.5))
    acc, Sk, Yb = np.zeros(m), np.zeros(m), np.zeros(m)
    for j in range(n):
        ok = j < nh
        u = u + np.where(ok, ch[:, j], 0.0) / ell * v
        v = v - np.where(ok, 2 * kap[:, j] * ell / cs[:, j], 0.0) * u
        nn = np.hypot(u, v)
        g = np.where(ok, np.log(nn), 0.0)
        acc += g
        Sk += (j + 1) * g
        Yb += np.where(ok, 2 * Sk / (j + 1), 0.0)
        u, v = u / nn, v / nn
    k = np.maximum(nh, 1)
    return dict(lam=acc / k, Y=Yb / k, n_hits=nh)


VERDICTS = {2: "launches into the chaotic sea", 1: "launches into a regular island", 0: "regular phase space (no chaotic sea)",
            -1: "launch escapes the cell"}


def chaos_map(cell, theta_c, q0, lam_m, n_hits=300, n_grid=0, n_side=8, Y_chaos=4.0, min_hits=30):
    """Does the beam launch into a chaotic sea or onto an island? tangent_chaos of the n_side² beam rays at θc (their
    chaotic share: ⟨Y⟩ > Y_chaos with ≥ min_hits hits; escaped: < min_hits hits) and, n_grid > 0, of an n_grid × n_grid grid
    of departures from the input mirror (Herriott: M1 (all its segments) over its length, sin χ within ±2a/d; circle: the whole boundary,
    |sin χ| < 0.95): sea, island, esc_sea = chaotic, regular and escaped shares of that section. Verdict: -1 escapes (≥ ½
    of the beam escaped), 2 sea (≥ ½ of the rest chaotic), 1 island (the launch is regular while ≥ 5 % of the section that
    stays is chaotic), 0 no sea."""
    c = cell.cell if isinstance(cell, MediumCell) else cell
    s_in = c.s_in_default
    tb = tangent_chaos(c, *beam_rays(c, s_in, theta_c, q0, lam_m, n_side), n_hits=n_hits)
    stay = tb["n_hits"] >= min_hits
    chaotic = stay & (tb["Y"] > Y_chaos)
    out = dict(f_launch=float(chaotic.mean()), esc_launch=float((~stay).mean()),
               Y_launch=float(np.median(tb["Y"])), lam_ftle=float(np.median(tb["lam"][stay])) if stay.any() else math.nan,
               sea=math.nan, island=math.nan, esc_sea=math.nan, grid=None)
    if n_grid:
        m = c.meta
        if m.get("kind") == "herriott" or "aperture" in m:
            ks = [k for k, lab in enumerate(c.labels) if lab == "M1" or lab.startswith("M1 ")]
            s_lo, s_hi = c.s0[ks[0]], c.s0[ks[-1]] + c.L[ks[-1]]
            sg = s_lo + (s_hi - s_lo) * (np.arange(n_grid) + 0.5) / n_grid
            pm = 2 * m["aperture"] / m["d"]
        else:
            sg = c.perimeter * (np.arange(n_grid) + 0.5) / n_grid
            pm = 0.95
        pg = pm * (2 * (np.arange(n_grid) + 0.5) / n_grid - 1)
        SS, PP = np.meshgrid(sg, pg)
        tg = tangent_chaos(c, *P.launch(c, SS.ravel(), np.arcsin(PP.ravel())), n_hits=n_hits)
        st = tg["n_hits"] >= min_hits
        ch_ = st & (tg["Y"] > Y_chaos)
        out.update(sea=float(ch_.mean()), island=float((st & ~ch_).mean()), esc_sea=float((~st).mean()),
                   grid=dict(s=sg, p=pg, Y=tg["Y"].reshape(n_grid, n_grid), lam=tg["lam"].reshape(n_grid, n_grid),
                             n_hits=tg["n_hits"].reshape(n_grid, n_grid)))
    if out["esc_launch"] >= 0.5:
        v = -1
    elif chaotic.sum() >= 0.5 * stay.sum():
        v = 2
    elif n_grid and out["sea"] >= 0.05 * (out["sea"] + out["island"]) and out["sea"] > 0:
        v = 1
    else:
        v = 0
    out["verdict"] = v
    return out


def cell_figures(cell, theta_c, q0, lam_m, n_eff=2.0, gamma=0.2, alpha_ppm=SPECIES["CH4"][0], R_mirror=0.995, dither=2e-3,
                 waveform="sine", f_par=1e-6, P_in=1e-3, bandwidth=1.0, n_side=24, n_bounce=2000, n_lyap=300, n_pass_max=60,
                 n_dither_max=12801, chaos_grid=12, n_ftle=300):
    """All figures of one cell (module docstring) as a dict; cell may be a MediumCell (index inclusions)."""
    mc = as_medium(cell, n_eff)
    s_in, s_out = mc.s_in_default, mc.s_out_default
    pw = mc.meta["port_w"]
    lam = lam_m * n_eff
    ports = [(s_in, pw)] + ([] if abs(s_out - s_in) < 1e-15 else [(s_out, pw)])
    ev = ensemble(mc, beam_rays(mc, s_in, theta_c, q0, lam_m, n_side), pw, s_in, s_out, n_bounce, R_mirror, gamma, lam_m)
    A_ppm = 100 * alpha_ppm * gamma * ev["L_mean"] if ev["T"] > 0 else 0.0
    sig_A = math.sqrt(2 * Q_E * bandwidth / (P_in * ev["T"])) if ev["T"] > 0 else math.inf
    cp = centre_path(mc, s_in, theta_c, s_out, pw, n_bounce)
    gb = gaussian_path(mc, cp, q0, lam_m, ports, R_mirror)
    sep, fit = twin_divergence(mc, s_in, theta_c, n_lyap)
    n_pass = int(min(cp["n_hits"], n_pass_max)) if cp["n_hits"] else 1
    da = RP.dither_analysis(mc, theta_c, dither, n_pass, lam, n_eff, waveform, n_max=n_dither_max, s_in=s_in)
    V = float(da["V"][n_pass - 1])
    cm = chaos_map(mc, theta_c, q0, lam_m, n_ftle, chaos_grid)
    nec_shot = sig_A / A_ppm if A_ppm > 0 else math.inf
    nec_fringe = 2 * math.sqrt(f_par) * V / A_ppm if A_ppm > 0 else math.inf
    return dict(V_floor=float(da["noise_floor"]), T=ev["T"], L_mean=ev["L_mean"], L_rms=ev["L_rms"], S1=ev["S1"], M2=ev["M2"], leaked=ev["leaked"], back=ev["back"],
                trapped=ev["trapped"], A_ppm=A_ppm, nec_shot=nec_shot, nec_fringe=nec_fringe, nec=math.hypot(nec_shot, nec_fringe),
                lam=float(fit["lam"]) if fit else 0.0, lam_fit=fit is not None, V=V, V_pass=da["V"],
                same_path=float(da["same_path"][n_pass - 1]), n_pass=n_pass, exit_pass=cp["n_hits"] if cp["exits"] else -1,
                L_centre=float(cp["chords"].sum()), T_G=gb["T_G"], w_max_ratio=gb["w_max_ratio"], lam_G=gb["lam_G"],
                m_max=gb["m_max"], theta=float(theta_c), w_hits=gb["w"], share=gb["share"], sep=sep, centre=cp,
                area=float(mc.area), dither_n=int(da["n_used"]), n_inc=mc.n_inc, ens=ev,
                K_ripple=float(mc.meta.get("K_ripple", 0.0)), f_launch=cm["f_launch"], esc_launch=cm["esc_launch"],
                Y_launch=cm["Y_launch"], lam_ftle=cm["lam_ftle"], sea=cm["sea"], island=cm["island"], verdict=cm["verdict"],
                chaos=cm)


def retune(cell, theta0, q0, lam_m, span=None, n_theta=41, R_mirror=0.995, gamma=0.2, n_side=8, n_bounce=2000, n_eff=2.0):
    """Launch angle within theta0 ± span (default ± 4 beam divergences) that maximises the sensitivity Γ T ⟨L⟩ of a
    small ray ensemble (n_side²), and that maximum. A perturbed cell moves the spot pattern off re-entrance; this
    re-aims the input before the cells are compared."""
    mc = as_medium(cell, n_eff)
    w = q_width(q0, lam_m)[0]
    span = 4 * lam_m / (math.pi * w) if span is None else span
    pw = mc.meta["port_w"]
    best = (-1.0, theta0)
    for th in theta0 + span * np.linspace(-1, 1, int(n_theta)):
        ev = ensemble(mc, beam_rays(mc, mc.s_in_default, th, q0, lam_m, n_side), pw, mc.s_in_default, mc.s_out_default, n_bounce,
                      R_mirror, gamma, lam_m)
        if ev["S1"] > best[0] * (1 + 1e-12):
            best = (float(ev["S1"]), float(th))
    return best[1], best[0]


_PERT = dict(eps4=0.0, eps3=0.0, conic=0.0, deform="both", n_seg=0, seg_curv=1.0, seg_tilt=0.0, seg_seed=1, n_inc=0, inc_radius=50e-6,
             inc_dn=0.01, inc_sign="both", inc_seed=1, inc_circle=0, ripple_a=0.0, ripple_period=0.5e-3, ripple_phase=0.0,
             ripple_on="both")


def _common(L, W, N, M, distinct, port_width, walls, n_facets, out_frac, smooth_circle, wavelength, n_eff, species, pert):
    require_positive(wavelength=wavelength, n_eff=n_eff)
    require_choice("species", species, tuple(SPECIES))
    lam_m = wavelength / n_eff
    q0, c0 = input_q(L, W, N, M, bool(distinct), port_width, lam_m, bool(walls))
    pw = c0.meta["port_w"]
    p = dict(_PERT, **pert)
    base = baseline_cell(L, W, N, M, bool(distinct), pw, p["eps4"], p["eps3"], p["conic"], p["deform"], bool(walls), lam_m,
                         int(p["n_seg"]), p["seg_curv"], p["seg_tilt"], int(p["seg_seed"]), p["ripple_a"], p["ripple_period"],
                         p["ripple_phase"], p["ripple_on"])
    circ = circle_reference(base.area, n_facets, pw, out_frac, bool(smooth_circle))
    inc = place_inclusions(base, int(p["n_inc"]), p["inc_radius"], p["inc_dn"], p["inc_sign"], int(p["inc_seed"])) if p["n_inc"] else None
    incc = place_inclusions(circ, int(p["n_inc"]), p["inc_radius"], p["inc_dn"], p["inc_sign"], int(p["inc_seed"])) if p["n_inc"] and p["inc_circle"] else None
    return lam_m, q0, MediumCell(base, inc, n_eff), MediumCell(circ, incc, n_eff)


_KEYS = ("T", "L_mean", "L_rms", "S1", "M2", "A_ppm", "nec_shot", "nec_fringe", "nec", "lam", "V", "V_floor", "same_path", "n_pass",
         "exit_pass", "T_G", "w_max_ratio", "lam_G", "m_max", "leaked", "area", "theta", "n_inc", "K_ripple", "f_launch",
         "esc_launch", "Y_launch", "lam_ftle", "sea", "island", "verdict")
_UNITS = dict(theta="rad", L_mean="m", L_rms="m", S1="m", A_ppm="1/ppm", nec_shot="ppm", nec_fringe="ppm", nec="ppm", lam="1/hit",
              lam_G="1/hit", area="m^2", lam_ftle="1/hit")


def compare(L=15e-3, W=1.5e-3, N=14, M=3, distinct=1, port_width=0.0, eps4=0.0, eps3=0.0, conic=0.0, deform="both", walls=0,
            n_seg=0, seg_curv=1.0, seg_tilt=0.0, seg_seed=1, n_inc=0, inc_radius=50e-6, inc_dn=0.01, inc_sign="both", inc_seed=1,
            inc_circle=0, ripple_a=0.0, ripple_period=0.5e-3, ripple_phase=0.0, ripple_on="both", n_facets=24, circle_theta=0.35, out_frac=0.37, smooth_circle=0, wavelength=1.6537e-6, n_eff=2.0,
            gamma=0.2, species="CH4", R_mirror=0.995, dither=2e-3, waveform="sine", f_parasitic=1e-6, P_in=1e-3, bandwidth=1.0,
            n_side=24, n_bounce=2000, retune_launch=1, chaos_grid=20, n_ftle=300) -> Result:
    """Elongated Herriott cell (h_; deformed, segmented and/or with index inclusions) against the CMPC of the same area
    (c_): path, sensitivity and detection limits against chaos (Lyapunov exponent, dither visibility), beam quality
    (M² of the detected light) and the Gaussian-beam model, with the ratios. retune_launch: re-aim the baseline's
    launch for the largest Γ T ⟨L⟩ before comparing. ripple_*: rippled end mirrors (K_ripple = 8π² a L / Λ²); the chaos
    map (chaos_map: tangent-map FTLE and MEGNO of the launch beam and of a chaos_grid² section) says whether the beam
    launches into the chaotic sea (verdict 2) or onto a regular island (1)."""
    require_range("R_mirror", R_mirror, 0.0, 1.0)
    require_nonnegative(dither=dither, f_parasitic=f_parasitic, n_inc=n_inc, inc_radius=inc_radius)
    pert = dict(eps4=eps4, eps3=eps3, conic=conic, deform=deform, n_seg=n_seg, seg_curv=seg_curv, seg_tilt=seg_tilt, seg_seed=seg_seed,
                n_inc=n_inc, inc_radius=inc_radius, inc_dn=inc_dn, inc_sign=inc_sign, inc_seed=inc_seed, inc_circle=inc_circle,
                ripple_a=ripple_a, ripple_period=ripple_period, ripple_phase=ripple_phase, ripple_on=ripple_on)
    lam_m, q0, base, circ = _common(L, W, N, M, distinct, port_width, walls, n_facets, out_frac, smooth_circle, wavelength, n_eff,
                                    species, pert)
    kw = dict(n_eff=n_eff, gamma=gamma, alpha_ppm=SPECIES[species][0], R_mirror=R_mirror, dither=dither, waveform=waveform,
              f_par=f_parasitic, P_in=P_in, bandwidth=bandwidth, n_side=n_side, n_bounce=n_bounce, chaos_grid=chaos_grid, n_ftle=n_ftle)
    th = retune(base, base.meta["theta_launch"], q0, lam_m, R_mirror=R_mirror, gamma=gamma, n_eff=n_eff)[0] if retune_launch else base.meta["theta_launch"]
    h = cell_figures(base, th, q0, lam_m, **kw)
    c = cell_figures(circ, circle_theta, q0, lam_m, **kw)
    v = {f"h_{k}": h[k] for k in _KEYS}
    v.update({f"c_{k}": c[k] for k in _KEYS})
    v.update(path_gain=h["L_mean"] / c["L_mean"] if c["L_mean"] > 0 else math.nan,
             sensitivity_gain=h["S1"] / c["S1"] if c["S1"] > 0 else math.nan,
             chaos_lost=c["lam"] - h["lam"], washout_lost=h["V"] - c["V"],
             nec_ratio=h["nec"] / c["nec"] if c["nec"] > 0 else math.nan,
             R_design=base.meta["R"], port_width=base.meta["port_w"], w_in=q_width(q0, lam_m)[0], circle_radius=circ.meta["r"],
             verdict_text=VERDICTS[h["verdict"]])
    u = {k: "" for k in v}
    for p_ in ("h_", "c_"):
        u.update({p_ + k: un for k, un in _UNITS.items()})
    u.update(R_design="m", port_width="m", w_in="m", circle_radius="m")
    return Result(values=v, units=u, assumptions=[
        "2D ray optics in the slab (n_eff); mirrors: constant reflectance R_mirror at every hit; smooth profile, curved "
        "segmented or faceted walls; circular index inclusions refract (Snell, TIR), Fresnel reflection at them neglected",
        "Same input beam for both cells: eigenmode of the undeformed baseline at its window, sampled on a quantile grid",
        "Absorbance per ppm from the strongest illustrative line (trace_gas, not HITRAN), geometric path, gas overlap Γ",
        "Fringe floor 2 sqrt(f_parasitic) V from one parasitic copy whose phase follows the launch dither (optical path); "
        "shot noise with responsivity 1 A/W",
        "Lyapunov exponent from twin rays at the launch angle (0 when no exponential stretch); Gaussian (ABCD) model "
        "along the centre ray with the walls only (inclusions not included), M² from the detected rays",
        "Chaos map from the exact tangent map of the walls (inclusions and ports ignored): MEGNO ⟨Y⟩ > 4 over n_ftle hits "
        "counts as chaotic; flat facets have κ = 0, so polygon cells read regular there (their mixing is ray splitting at "
        "corners, which the twin-ray λ sees)"])


SWEEPABLE = ("eps4", "eps3", "conic", "n_seg", "seg_curv", "seg_tilt", "n_inc", "inc_dn", "inc_radius", "ripple_a", "ripple_period",
             "ripple_phase")


def perturbation_sweep(param="eps4", values=(0.0, 0.02, 0.05, 0.1, 0.15), L=15e-3, W=1.5e-3, N=14, M=3, distinct=1,
                       port_width=0.0, walls=0, n_facets=24, circle_theta=0.35, out_frac=0.37, smooth_circle=0,
                       wavelength=1.6537e-6, n_eff=2.0, gamma=0.2, species="CH4", R_mirror=0.995, dither=2e-3,
                       waveform="sine", f_parasitic=1e-6, P_in=1e-3, bandwidth=1.0, n_side=16, n_bounce=2000, retune_launch=1, chaos_grid=10,
                       **pert) -> Result:
    """The baseline's path, sensitivity, NEC, chaos (λ, V) and beam quality (M²) against one perturbation parameter
    (SWEEPABLE; the others as given in pert), with the CMPC of the same area as the reference (c_*)."""
    require_choice("param", param, SWEEPABLE)
    lam_m, q0, _, circ = _common(L, W, N, M, distinct, port_width, walls, n_facets, out_frac, smooth_circle, wavelength, n_eff,
                                 species, dict(pert, n_inc=0))
    kw = dict(n_eff=n_eff, gamma=gamma, alpha_ppm=SPECIES[species][0], R_mirror=R_mirror, dither=dither, waveform=waveform,
              f_par=f_parasitic, P_in=P_in, bandwidth=bandwidth, n_side=n_side, n_bounce=n_bounce, chaos_grid=chaos_grid)
    rows = []
    for val in values:
        p = dict(pert, **{param: val})
        _, _, b, _ = _common(L, W, N, M, distinct, port_width, walls, n_facets, out_frac, smooth_circle, wavelength, n_eff, species, p)
        th = retune(b, b.meta["theta_launch"], q0, lam_m, R_mirror=R_mirror, gamma=gamma, n_eff=n_eff)[0] if retune_launch else b.meta["theta_launch"]
        rows.append(cell_figures(b, th, q0, lam_m, **kw))
    c = cell_figures(circ, circle_theta, q0, lam_m, **kw)
    v = {param: np.asarray(values, float)}
    v.update({k: np.array([r[k] for r in rows], float) for k in _KEYS})
    v.update({f"c_{k}": c[k] for k in _KEYS})
    u = {k: _UNITS.get(k, "") for k in v}
    u.update({f"c_{k}": _UNITS.get(k, "") for k in _KEYS})
    return Result(values=v, units=u, assumptions=[f"As compare(); one row per value of {param}"])


def deformation_sweep(eps4_values=(0.0, 0.02, 0.05, 0.1, 0.15), **kw) -> Result:
    """perturbation_sweep over the quartic deformation eps4."""
    return perturbation_sweep("eps4", eps4_values, **kw)
