"""2D billiard ray tracing for chip-scale multipass cells: path-length statistics between an input and an output port.

The guided mode of a membrane (or any planar waveguide) propagates in-plane as a ray and reflects specularly at
the cell boundary. Two cell families:

    Stadium         two half-discs of radius r joined by straights of length a (a = 0: circle, integrable)
    SegmentedCell   regular N-gon of facet mirrors (pseudo-integrable), with designed perturbations
                    (random facet tilt, radial offset, facet curvature) that make it mixing

The tracer is geometry only: it stores every chord length and angle of incidence (RayTable). evaluate() then
re-weights that one table for any mirror reflectance R(sin χ), background loss and evanescent factor Γ, so
mirror, loss and wavelength sweeps cost no extra tracing.

Boundary coordinate s is arclength (Stadium) or facet index × facet length (SegmentedCell); ports are windows
in s. sin χ is the signed sine of the angle of incidence (conserved in a circle). SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

_SHAPES = ("stadium", "polygon")


# ---------------------------------------------------------------- geometries
@dataclass
class Stadium:
    """Stadium billiard; a = 0 is the circle. Footprint 2(r + a/2) × 2r. s = 0 at the left end of the bottom straight."""

    r: float
    a: float = 0.0

    @property
    def area(self):
        return 2 * self.r * self.a + np.pi * self.r**2

    @property
    def perimeter(self):
        return 2 * self.a + 2 * np.pi * self.r

    @property
    def mean_chord(self):
        return np.pi * self.area / self.perimeter      # Cauchy formula, 2D

    @property
    def s_in_default(self):
        return self.a / 2

    def point_at(self, s):
        """Boundary point and outward normal at arclength s (counter-clockwise)."""
        s = np.mod(np.asarray(s, float), self.perimeter)
        r, a = self.r, self.a
        x, y, nx, ny = (np.empty_like(s) for _ in range(4))
        m1 = s < a
        m2 = (s >= a) & (s < a + np.pi * r)
        m3 = (s >= a + np.pi * r) & (s < 2 * a + np.pi * r)
        m4 = s >= 2 * a + np.pi * r
        x[m1], y[m1], nx[m1], ny[m1] = s[m1] - a / 2, -r, 0, -1
        ph = (s[m2] - a) / r - np.pi / 2
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
        x, y, _, _ = self.point_at(np.linspace(0, self.perimeter, n))
        return x, y

    def hit(self, px, py, dx, dy):
        """First boundary hit from interior/boundary points along unit directions: t, outward normal, s."""
        r, a = self.r, self.a
        eps = 1e-9 * r
        n = px.size
        T = np.full((6, n), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (r - py) / dy
            T[0] = np.where((dy > 0) & (tt > eps) & (np.abs(px + tt * dx) <= a / 2 + 1e-12), tt, np.inf)
            tb = (-r - py) / dy
            T[1] = np.where((dy < 0) & (tb > eps) & (np.abs(px + tb * dx) <= a / 2 + 1e-12), tb, np.inf)
        for idx, cx, sign in ((2, a / 2, +1), (4, -a / 2, -1)):
            ox, oy = px - cx, py
            b = ox * dx + oy * dy
            disc = b * b - (ox * ox + oy * oy - r * r)
            sq = np.sqrt(np.maximum(disc, 0))
            for j, t in enumerate((-b - sq, -b + sq)):
                ok = (disc >= 0) & (t > eps) & (sign * (px + t * dx - cx) >= -1e-12)
                T[idx + j] = np.where(ok, t, np.inf)
        k = np.argmin(T, axis=0)
        t = T[k, np.arange(n)]
        hx, hy = px + t * dx, py + t * dy
        nx, ny, s = np.zeros(n), np.zeros(n), np.zeros(n)
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

    _hit = hit


@dataclass
class SegmentedCell:
    """Regular N-gon of facet mirrors (circumradius r), each facet an in-plane mirror.

    The unperturbed polygon is pseudo-integrable (zero Lyapunov exponent). Designed perturbations break that:
      tilt_rms    rms random tilt of each facet about its midpoint [rad] (fixed pattern, set by lithography)
      offset_rms  rms radial displacement of each facet [m]
      curvature   1/ρ of every facet [1/m]; > 0 convex into the cell (dispersing, strongly mixing), < 0 concave
      curvature_rms  rms random spread of 1/ρ between facets [1/m]
      tilts, curvatures, offsets   optional explicit per-facet arrays (length N) ADDED to the above, e.g. to tilt
                  one chosen facet, or to reproduce a perturbation pattern drawn elsewhere (the JS applet port)
    Corner gaps/overlaps are neglected: facets are extended by 15 % so rays cannot leak.
    s runs along the facets (length 2h each); s_in_default is the middle of facet 0.
    """

    r: float
    n_facets: int = 24
    tilt_rms: float = 0.0
    curvature: float = 0.0
    offset_rms: float = 0.0
    seed: int = 0
    a: float = 0.0          # kept for interface compatibility with Stadium
    curvature_rms: float = 0.0
    tilts: object = None
    curvatures: object = None
    offsets: object = None

    def __post_init__(self):
        N = self.n_facets
        rng = np.random.default_rng(self.seed)
        th = 2 * np.pi * np.arange(N) / N
        self.h = self.r * np.sin(np.pi / N)
        self.apothem = self.r * np.cos(np.pi / N)
        phi = th + self.tilt_rms * rng.standard_normal(N)
        dr = self.offset_rms * rng.standard_normal(N)
        kap = np.full(N, float(self.curvature))
        if self.curvature_rms:
            kap = kap + self.curvature_rms * rng.standard_normal(N)
        extra = {"tilts": self.tilts, "curvatures": self.curvatures, "offsets": self.offsets}
        for name, v in extra.items():
            if v is not None and np.shape(v) != (N,):
                raise ValueError(f"{name} must have n_facets = {N} entries")
        if self.tilts is not None:
            phi = phi + np.asarray(self.tilts, float)
        if self.offsets is not None:
            dr = dr + np.asarray(self.offsets, float)
        if self.curvatures is not None:
            kap = kap + np.asarray(self.curvatures, float)
        self.kappa = kap
        self.n_k = np.stack([np.cos(phi), np.sin(phi)], 1)
        self.t_k = np.stack([-np.sin(phi), np.cos(phi)], 1)
        self.p_k = (self.apothem + dr)[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
        self.s_in_default = self.h

    @property
    def perimeter(self):
        return 2 * self.h * self.n_facets

    @property
    def area(self):
        return 0.5 * self.n_facets * self.r**2 * np.sin(2 * np.pi / self.n_facets)

    @property
    def mean_chord(self):
        return np.pi * self.area / self.perimeter

    def _surface(self, k, u):
        """Point and outward normal on facet k at tangential offset u (arrays)."""
        nk, tk, pk = self.n_k[k], self.t_k[k], self.p_k[k]
        x, y = pk[:, 0] + u * tk[:, 0], pk[:, 1] + u * tk[:, 1]
        nx, ny = nk[:, 0].copy(), nk[:, 1].copy()
        kap = self.kappa[k]
        c = kap != 0
        if c.any():
            rho = 1 / kap[c]
            R, sg, uc = np.abs(rho), np.sign(rho), np.broadcast_to(u, kap.shape)[c]
            sag = R - np.sqrt(np.maximum(R * R - uc * uc, 0))
            x[c] += sg * sag * nk[c, 0]
            y[c] += sg * sag * nk[c, 1]
            cx, cy = pk[c, 0] + rho * nk[c, 0], pk[c, 1] + rho * nk[c, 1]
            nx[c], ny[c] = -sg * (x[c] - cx) / R, -sg * (y[c] - cy) / R
        return x, y, nx, ny

    def point_at(self, s):
        s = np.mod(np.asarray(s, float), self.perimeter)
        k = np.minimum((s // (2 * self.h)).astype(int), self.n_facets - 1)
        return self._surface(k, s - k * 2 * self.h - self.h)

    def outline(self, n=400):
        per = max(n // self.n_facets, 4)
        xs, ys = [], []
        for k in range(self.n_facets):
            x, y, _, _ = self._surface(np.full(per, k), np.linspace(-self.h, self.h, per))
            xs.append(x)
            ys.append(y)
        xs.append(xs[0][:1])
        ys.append(ys[0][:1])
        return np.concatenate(xs), np.concatenate(ys)

    def inside(self, x, y):
        """Inside the (flat-facet) half-planes; ignores facet sag."""
        x, y = np.asarray(x), np.asarray(y)
        ok = np.ones(x.shape, bool)
        for k in range(self.n_facets):
            ok &= (x - self.p_k[k, 0]) * self.n_k[k, 0] + (y - self.p_k[k, 1]) * self.n_k[k, 1] <= 0
        return ok

    def hit(self, px, py, dx, dy):
        n = px.size
        eps = 1e-9 * self.r
        hh = 1.15 * self.h
        bt, bnx, bny, bs = np.full(n, np.inf), np.zeros(n), np.zeros(n), np.zeros(n)
        for k in range(self.n_facets):
            nk, tk, pk = self.n_k[k], self.t_k[k], self.p_k[k]
            rho = 1 / self.kappa[k] if self.kappa[k] != 0 else None
            with np.errstate(divide="ignore", invalid="ignore"):
                if rho is None:
                    dn = dx * nk[0] + dy * nk[1]
                    t = ((pk[0] - px) * nk[0] + (pk[1] - py) * nk[1]) / dn
                    u = (px + t * dx - pk[0]) * tk[0] + (py + t * dy - pk[1]) * tk[1]
                    ok = (dn > 1e-12) & (t > eps) & (np.abs(u) <= hh)
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
            bnx, bny = np.where(upd, nx_, bnx), np.where(upd, ny_, bny)
            bs = np.where(upd, k * 2 * self.h + np.clip(u, -self.h, self.h) + self.h, bs)
        return bt, bnx, bny, bs

    _hit = hit


def make_cell(shape="stadium", radius=5e-3, straight=0.0, n_facets=24, tilt_rms=0.0, curvature=0.0, offset_rms=0.0, seed=0):
    """Stadium (shape='stadium', uses straight) or SegmentedCell (shape='polygon', uses the facet parameters)."""
    require_choice("shape", shape, _SHAPES)
    if shape == "stadium":
        return Stadium(radius, straight)
    return SegmentedCell(radius, int(n_facets), tilt_rms, curvature, offset_rms, int(seed))


# ---------------------------------------------------------------- tracing
def _reflect(dx, dy, nx, ny):
    dn = dx * nx + dy * ny
    rx, ry = dx - 2 * dn * nx, dy - 2 * dn * ny
    nn = np.hypot(rx, ry)
    return rx / nn, ry / nn


def poincare(cell, n_rays=40, n_hits=300, seed=0):
    """Boundary phase-space samples (s, signed sin χ) of long trajectories without ports. Initial conditions are
    uniform in (s, sin χ), the invariant measure of billiards. Returns two (n_rays, n_hits) arrays."""
    rng = np.random.default_rng(seed)
    x, y, nx, ny = cell.point_at(rng.random(n_rays) * cell.perimeter)
    ch = np.arcsin(rng.uniform(-0.98, 0.98, n_rays))
    ux, uy = -nx, -ny
    dx, dy = ux * np.cos(ch) - uy * np.sin(ch), ux * np.sin(ch) + uy * np.cos(ch)
    S, SC = np.full((n_rays, n_hits), np.nan), np.full((n_rays, n_hits), np.nan)
    ok = np.ones(n_rays, bool)
    for j in range(n_hits):
        t, hnx, hny, hs = cell.hit(x, y, dx, dy)
        ok &= np.isfinite(t)
        t = np.where(ok, t, 0)
        x, y = x + t * dx, y + t * dy
        S[:, j] = np.where(ok, hs, np.nan)
        SC[:, j] = np.where(ok, dx * hny - dy * hnx, np.nan)
        dx, dy = _reflect(dx, dy, hnx, hny)
    return S, SC


@dataclass
class RayTable:
    """Geometry-only ray data, reusable for any mirror R(sin χ), loss and wavelength."""

    cell: object
    port_w: float
    s_in: float
    s_out: float
    n_rays: int
    n_bounce: int
    chord: np.ndarray      # (n_rays, n_bounce) float32, chord j ends at hit j
    sinchi: np.ndarray     # (n_rays, n_bounce) float32, signed sin of the incidence angle at hit j
    exit_idx: np.ndarray   # (n_rays,) hit index that lands in a port, -1: trapped (n_bounce or max_path reached)
    exit_port: np.ndarray  # (n_rays,) 1 output (detected), 0 input (lost), -1 trapped, -2 leaked through a corner
    theta0: float
    start: np.ndarray = field(default=None, repr=False)   # (n_rays, 4) x, y, dx, dy
    hits: np.ndarray = field(default=None, repr=False)    # (n_rays, n_bounce, 2) hit positions (keep_hits only)
    itin: np.ndarray = field(default=None, repr=False)    # (n_rays,) uint64 hash of the facet sequence (path identity)

    @property
    def trapped_fraction(self):
        return float(np.mean(self.exit_port == -1))


def in_window(s, s0, w, P):
    """Is boundary coordinate s inside the window of width w centred on s0 (periodic in P)?"""
    d = np.mod(s - s0 + P / 2, P) - P / 2
    return np.abs(d) <= w / 2


def trace_rays(cell, port_w, n_rays=3000, n_bounce=2500, theta0=np.radians(20), s_in=None, s_out_frac=0.37,
               seed=1, keep_start=True, keep_hits=False, max_path=None, theta_c=0.0) -> RayTable:
    """Launch rays from the input port (angle uniform in theta_c ± theta0 from the inward normal, position uniform
    across the port) and trace specular reflections until they hit the input or the output port window
    (centred s_out_frac × perimeter further along), n_bounce is exhausted or the path exceeds max_path."""
    rng = np.random.default_rng(seed)
    P = cell.perimeter
    s_in = getattr(cell, "s_in_default", 0.0) if s_in is None else s_in
    s_out = s_in + s_out_frac * P
    x, y, nx, ny = cell.point_at(s_in + (rng.random(n_rays) - 0.5) * port_w)
    ang = theta_c + (rng.random(n_rays) * 2 - 1) * theta0
    ux, uy = -nx, -ny
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
        t, hnx, hny, hs = cell.hit(x[ia], y[ia], dx[ia], dy[ia])
        bad = ~np.isfinite(t)
        if bad.any():
            exit_port[ia[bad]] = -2
            alive[ia[bad]] = False
            t, hs = np.where(bad, 0.0, t), np.where(bad, -1e9, hs)
        hx, hy = x[ia] + t * dx[ia], y[ia] + t * dy[ia]
        fac = np.clip(np.floor(np.nan_to_num(hs) / (P / nseg)), 0, nseg - 1).astype(np.uint64) + np.uint64(1)
        itin[ia] = itin[ia] * np.uint64(1000003) + fac          # wraps modulo 2^64 by design
        chord[ia, j] = t
        if keep_hits:
            hits[ia, j, 0], hits[ia, j, 1] = hx, hy
        sinchi[ia, j] = dx[ia] * hny - dy[ia] * hnx
        pin, pout = in_window(hs, s_in, port_w, P), in_window(hs, s_out, port_w, P)
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
        dx[ik], dy[ik] = _reflect(dx[ia][keep], dy[ia][keep], hnx[keep], hny[keep])
    return RayTable(cell, port_w, s_in, s_out, n_rays, n_bounce, chord, sinchi, exit_idx, exit_port, theta0, start, hits, itin)


def trace_path(cell, x0, y0, dx0, dy0, n_hits=40):
    """Polyline of one trajectory (no ports), for plotting."""
    x, y, dx, dy = (np.array([v], float) for v in (x0, y0, dx0, dy0))
    xs, ys = [x[0]], [y[0]]
    for _ in range(n_hits):
        t, nx, ny, _ = cell.hit(x, y, dx, dy)
        x, y = x + t * dx, y + t * dy
        dx, dy = _reflect(dx, dy, nx, ny)
        xs.append(x[0])
        ys.append(y[0])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------- path-length statistics
@dataclass
class CellResult:
    T_det: float          # fraction of launched power reaching the output port
    L_mean: float         # power-weighted mean geometric path of detected light [m]
    L_eff_gas: float      # Γ L_mean: equivalent bulk-gas path [m]
    S1: float             # Γ T_det L_mean: dP_det/P_in per unit α_gas [m]
    L: np.ndarray         # path of each detected ray [m]
    W: np.ndarray         # its power weight
    bounces: np.ndarray   # number of chords of each detected ray
    n_det: int
    L_mean_err: float     # standard error of L_mean (Kish effective sample size)


def _lnR_table(R_of_sinchi, n=513):
    sg = np.linspace(0, 1, n)
    return sg, np.log(np.clip(R_of_sinchi(sg), 1e-12, 1.0))


def evaluate(table: RayTable, R_of_sinchi, alpha_bg=0.0, Gamma=1.0) -> CellResult:
    """Re-weight a ray table. R_of_sinchi(|sin χ|) -> mirror power reflectance (vectorised); alpha_bg: background
    power attenuation of the guided mode [1/m] (scattering, absorption; not the gas). Reflections at the ports
    themselves do not count; trapped rays are dropped (check table.trapped_fraction against the loss)."""
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    if sel.size == 0:
        return CellResult(0.0, np.nan, np.nan, 0.0, np.array([]), np.array([]), np.array([]), 0, np.nan)
    e = table.exit_idx[sel]
    ncol = int(e.max()) + 1                      # columns past the last exit carry nothing
    j = np.arange(ncol)[None, :]
    L = (table.chord[sel, :ncol].astype(float) * (j <= e[:, None])).sum(1)
    sg, lnR = _lnR_table(R_of_sinchi)
    lnW = (np.interp(np.abs(table.sinchi[sel, :ncol]), sg, lnR) * (j < e[:, None])).sum(1) - alpha_bg * L
    W = np.exp(lnW)
    if W.sum() == 0:
        return CellResult(0.0, np.nan, np.nan, 0.0, L, W, e + 1, sel.size, np.nan)
    T = W.sum() / table.n_rays
    Lm = (W * L).sum() / W.sum()
    var = (W * (L - Lm) ** 2).sum() / W.sum()
    n_kish = W.sum() ** 2 / (W**2).sum()
    return CellResult(float(T), float(Lm), float(Gamma * Lm), float(Gamma * T * Lm), L, W, e + 1, sel.size,
                      float(np.sqrt(var / max(n_kish, 1))))


def occupancy_map(cell, port_w, R_of_sinchi, alpha_bg=0.0, n_rays=500, n_bounce=1500, nb=160, theta0=np.radians(20),
                  seed=4, samples=8, theta_c=0.0):
    """Power-weighted, time-averaged ray density over the cell (all launched rays until absorbed or exited).
    Returns x edges, y edges, density normalised to its mean over the interior (NaN outside), and the coverage
    = share of the interior where the density exceeds 10 % of that mean."""
    tab = trace_rays(cell, port_w, n_rays, n_bounce, theta0, seed=seed, keep_hits=True, theta_c=theta_c)
    sg, lnR = _lnR_table(R_of_sinchi)
    j = np.arange(n_bounce)[None, :]
    last = np.where(tab.exit_idx >= 0, tab.exit_idx, n_bounce - 1)[:, None]
    valid = j <= last
    ch = tab.chord.astype(float) * valid
    lnr = np.interp(np.abs(tab.sinchi), sg, lnR) * (j < last)
    surv = np.exp(np.cumsum(lnr, 1) - lnr - alpha_bg * (np.cumsum(ch, 1) - ch))
    x0 = np.concatenate([tab.start[:, :1], tab.hits[:, :-1, 0]], 1)
    y0 = np.concatenate([tab.start[:, 1:2], tab.hits[:, :-1, 1]], 1)
    x1, y1 = tab.hits[..., 0], tab.hits[..., 1]
    wch = surv * ch
    xo, yo = cell.outline(2000)
    xr = max(cell.a / 2 + cell.r, float(np.max(np.abs(xo))))
    yr = max(cell.r, float(np.max(np.abs(yo))))
    xe = np.linspace(-xr, xr, nb + 1)
    ye = np.linspace(-yr, yr, max(int(nb * yr / xr), 8) + 1)
    H = np.zeros((len(xe) - 1, len(ye) - 1))
    m = valid & (wch > 0)
    rng = np.random.default_rng(seed + 100)
    for k in range(samples):
        f = (k + rng.random(x0.shape)) / samples        # stratified positions along every chord
        H += np.histogram2d((x0 + f * (x1 - x0))[m], (y0 + f * (y1 - y0))[m], bins=(xe, ye), weights=wch[m] / samples)[0]
    XX, YY = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")
    ins = cell.inside(XX, YY)
    dens = np.where(ins, H / max(H[ins].mean(), 1e-300), np.nan)
    return xe, ye, dens, float((dens[ins] > 0.1).mean())


def mean_field_estimate(cell, port_w, R_mean, alpha_bg=0.0, Gamma=1.0):
    """Ergodic estimate (uniform boundary flux, every chord = mean chord ℓ = πA/P). Per hit: output with
    probability η = w/P, lost through the input with η, else survives with R_mean exp(-α ℓ)."""
    ell = cell.mean_chord
    eta = port_w / cell.perimeter
    surv = (1 - 2 * eta) * R_mean * np.exp(-alpha_bg * ell)
    T = eta * np.exp(-alpha_bg * ell) / (1 - surv)
    L = ell / (1 - surv)
    return dict(T_det=T, L_mean=L, n_bounce=1 / (1 - surv), L_eff_gas=Gamma * L, S1=Gamma * T * L)


def dB_per_cm_to_alpha(db_cm):
    """dB/cm power loss -> power attenuation coefficient α [1/m]."""
    return np.asarray(db_cm) * 100 * np.log(10) / 10


# ---------------------------------------------------------------- Result front ends
def _check_geometry(shape, radius, straight, n_facets, port_width):
    require_choice("shape", shape, _SHAPES)
    require_positive(radius=radius, port_width=port_width)
    require_nonnegative(straight=straight)
    if shape == "polygon" and int(n_facets) < 3:
        raise ValueError("n_facets must be >= 3")


def mean_field(shape="stadium", radius=5e-3, straight=5e-3, n_facets=24, port_width=150e-6, R_mean=0.999,
               alpha_bg=0.0, Gamma=1.0) -> Result:
    """Ergodic (mean-field) transmission and mean path of a multipass cell with two ports of width w."""
    _check_geometry(shape, radius, straight, n_facets, port_width)
    require_range("R_mean", R_mean, 0.0, 1.0)
    require_nonnegative(alpha_bg=alpha_bg, Gamma=Gamma)
    cell = make_cell(shape, radius, straight, n_facets)
    if 2 * port_width >= cell.perimeter:
        raise ValueError("port_width must be below half the perimeter")
    r = mean_field_estimate(cell, port_width, R_mean, alpha_bg, Gamma)
    return Result(
        values={"T_det": r["T_det"], "L_mean": r["L_mean"], "n_bounce": r["n_bounce"], "L_eff_gas": r["L_eff_gas"],
                "S1": r["S1"], "mean_chord": cell.mean_chord, "area": cell.area, "perimeter": cell.perimeter},
        units={"T_det": "", "L_mean": "m", "n_bounce": "", "L_eff_gas": "m", "S1": "m", "mean_chord": "m", "area": "m^2", "perimeter": "m"},
        assumptions=["Fully chaotic (ergodic) cell, uniform boundary flux", "Every chord replaced by the mean chord πA/P",
                     "Angle-independent mirror reflectance R_mean", "Not valid for regular cells (circle, unperturbed polygon)"],
    )


def ray_statistics(shape="stadium", radius=5e-3, straight=5e-3, n_facets=24, tilt_rms=0.0, curvature=0.0,
                   port_width=150e-6, theta0=0.349066, R_mean=0.999, alpha_bg=0.0, Gamma=1.0,
                   n_rays=1000, n_bounce=1500, seed=1) -> Result:
    """Monte-Carlo path statistics: rays launched from the input port, power-weighted over their exits through the
    output port, with an angle-independent mirror R_mean (use evaluate() with a RayTable for R(sin χ))."""
    _check_geometry(shape, radius, straight, n_facets, port_width)
    require_positive(theta0=theta0)
    require_range("R_mean", R_mean, 0.0, 1.0)
    require_nonnegative(alpha_bg=alpha_bg, Gamma=Gamma, tilt_rms=tilt_rms)
    n_rays, n_bounce = int(n_rays), int(n_bounce)
    if n_rays < 1 or n_bounce < 1:
        raise ValueError("n_rays and n_bounce must be >= 1")
    cell = make_cell(shape, radius, straight, n_facets, tilt_rms, curvature)
    tab = trace_rays(cell, port_width, n_rays, n_bounce, theta0, seed=int(seed), keep_start=False)
    r = evaluate(tab, lambda s: np.full_like(s, R_mean), alpha_bg, Gamma)
    return Result(
        values={"T_det": r.T_det, "L_mean": r.L_mean, "L_mean_err": r.L_mean_err, "L_eff_gas": r.L_eff_gas, "S1": r.S1,
                "n_det": r.n_det, "trapped_fraction": tab.trapped_fraction, "mean_chord": cell.mean_chord},
        units={"T_det": "", "L_mean": "m", "L_mean_err": "m", "L_eff_gas": "m", "S1": "m", "n_det": "",
               "trapped_fraction": "", "mean_chord": "m"},
        assumptions=["2D ray optics, specular reflection, no diffraction beyond the launch spread",
                     "Ideal port windows; light hitting the input port is lost",
                     "Rays still inside after n_bounce are dropped: keep trapped_fraction × R_mean^n_bounce negligible"],
    )
