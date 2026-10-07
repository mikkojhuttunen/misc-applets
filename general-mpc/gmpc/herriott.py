"""Herriott-type multipass cells: exact 3D ray tracing between real mirror surfaces.

Mirror surface (local frame, vertex at the origin, local +z pointing into the cell):

    z = f(x, y) = (cx x² + cy y²) / (1 + sqrt(1 - (1 + kx) cx² x² - (1 + ky) cy² y²)) + Σ a_ij x^i y^j

cx = 1/Rx, cy = 1/Ry (concave mirror: centre of curvature at +R on local z), kx, ky conic constants (0 = sphere or
toroid-like biconic; -1 = paraboloid), a_ij small polynomial deformations (astigmatism, coma, trefoil, ...). Rx = Ry,
k = 0, no a_ij is an exact sphere. Intersections are found by Newton iteration from the vertex plane (exact to
rounding, no paraxial approximation), so spherical aberration, astigmatism, tilt and decentre are all in the trace.
Each mirror has a clear aperture (rays outside are lost) and holes (rays inside leave the cell through them).

Herriott design (paraxial, Herriott, Kogelnik & Kompfner, Appl. Opt. 3, 523, 1964): two mirrors of radius R at
spacing d, cos θ = 1 - d/R; spot n (counting every mirror hit) at x_n = x0 cos nθ + sqrt(d/(4f - d)) (x0 + 2f x0') sin nθ
with f = R/2. Re-entrant after N hits when N θ = 2π M: d = R (1 - cos 2πM/N). Injection at (A, 0) with slopes
x0' = -A/R, y0' = (B/R) sqrt((2R - d)/d) gives the ellipse x_n = A cos nθ, y_n = B sin nθ (a circle for A = B).
An astigmatic pair (Rx ≠ Ry) has θx ≠ θy: a Lissajous pattern, re-entrant when N θx = 2π Mx and N θy = 2π My.
SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .stats import effective_path, lyapunov_fit, min_pairwise_distance


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Mirror3D:
    vertex: np.ndarray
    rot: np.ndarray                    # columns: local x, y, z axes in global coordinates
    Rx: float = np.inf
    Ry: float = np.inf
    kx: float = 0.0
    ky: float = 0.0
    poly: dict = field(default_factory=dict)   # {(i, j): a_ij}, sag term a_ij x^i y^j
    aperture: float = 0.0254
    holes: list = field(default_factory=list)  # [(x, y, radius)] in local coordinates
    label: str = ""

    def __post_init__(self):
        self.vertex = np.asarray(self.vertex, float)
        self.rot = np.asarray(self.rot, float)

    @property
    def cx(self):
        return 0.0 if np.isinf(self.Rx) else 1.0 / self.Rx

    @property
    def cy(self):
        return 0.0 if np.isinf(self.Ry) else 1.0 / self.Ry

    def sag(self, x, y):
        """f(x, y) and its partial derivatives (fx, fy); NaN outside the biconic's domain."""
        cx, cy = self.cx, self.cy
        u = cx * x * x + cy * y * y
        w = 1 - (1 + self.kx) * cx * cx * x * x - (1 + self.ky) * cy * cy * y * y
        sw = np.sqrt(np.where(w >= 0, w, np.nan))
        den = 1 + sw
        f = u / den
        dsw_dx = np.where(sw > 0, -(1 + self.kx) * cx * cx * x / sw, 0.0)
        dsw_dy = np.where(sw > 0, -(1 + self.ky) * cy * cy * y / sw, 0.0)
        fx = 2 * cx * x / den - u * dsw_dx / den**2
        fy = 2 * cy * y / den - u * dsw_dy / den**2
        for (i, j), a in self.poly.items():
            f = f + a * x**i * y**j
            if i:
                fx = fx + a * i * x ** (i - 1) * y**j
            if j:
                fy = fy + a * j * x**i * y ** (j - 1)
        return f, fx, fy

    def intersect(self, p, d, eps=1e-8, iters=60):
        """Ray (global points p, unit directions d; arrays (n, 3)) to surface: t, global hit point, global unit normal
        (pointing into the cell), local (x, y) of the hit. t = inf where there is no forward hit. Hits closer than eps
        (10 nm) are ignored, so a start point a hair off a deformed surface does not reflect on itself."""
        o = (p - self.vertex) @ self.rot
        dl = d @ self.rot
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dl[:, 2]) > 1e-15, -o[:, 2] / dl[:, 2], np.inf)
            for _ in range(iters):
                x, y, z = o[:, 0] + t * dl[:, 0], o[:, 1] + t * dl[:, 1], o[:, 2] + t * dl[:, 2]
                f, fx, fy = self.sag(x, y)
                F = z - f
                dF = dl[:, 2] - fx * dl[:, 0] - fy * dl[:, 1]
                step = F / dF
                t = t - np.where(np.isfinite(step), step, 0.0)
                if np.nanmax(np.abs(np.where(np.isfinite(step), step, 0.0)), initial=0.0) < 1e-15 * (1 + np.nanmax(np.abs(np.where(np.isfinite(t), t, 0.0)), initial=0.0)):
                    break
            x, y = o[:, 0] + t * dl[:, 0], o[:, 1] + t * dl[:, 1]
            f, fx, fy = self.sag(x, y)
        ok = np.isfinite(t) & (t > eps) & np.isfinite(f) & (np.abs(o[:, 2] + t * dl[:, 2] - f) < 1e-9 * (1 + np.abs(f)))
        nl = np.stack([-fx, -fy, np.ones_like(fx)], 1)
        nl /= np.linalg.norm(nl, axis=1, keepdims=True)
        t = np.where(ok, t, np.inf)
        return t, p + t[:, None] * d, nl @ self.rot.T, np.stack([x, y], 1)


def two_mirror_cell(d, R1=(1.0, 1.0), R2=(1.0, 1.0), aperture=0.0254, holes1=(), holes2=(), k1=(0.0, 0.0), k2=(0.0, 0.0),
                    poly1=None, poly2=None):
    """Mirror 1 at z = 0 facing +z, mirror 2 at z = d facing -z (local frame rotated π about y, so its local x is -x).
    R1, R2: (Rx, Ry) of each mirror; holes in local coordinates."""
    m1 = Mirror3D(np.zeros(3), np.eye(3), R1[0], R1[1], k1[0], k1[1], dict(poly1 or {}), aperture, list(holes1), "M1")
    m2 = Mirror3D(np.array([0.0, 0.0, d]), rot_y(np.pi), R2[0], R2[1], k2[0], k2[1], dict(poly2 or {}), aperture, list(holes2), "M2")
    return [m1, m2]


def perturb_mirror(m: Mirror3D, tilt_x=0.0, tilt_y=0.0, rot_about_axis=0.0, shift=(0.0, 0.0, 0.0), dRx=0.0, dRy=0.0,
                   poly=None, dk=(0.0, 0.0)):
    """Copy of a mirror tilted about its own local x and y axes (rad, about the vertex), rotated about its axis, shifted
    (global m), with radius changes and extra polynomial sag terms (added to existing ones)."""
    R = m.rot @ rot_x(tilt_x) @ rot_y(tilt_y) @ rot_z(rot_about_axis)
    p = dict(m.poly)
    for k, v in (poly or {}).items():
        p[k] = p.get(k, 0.0) + v
    return replace(m, vertex=m.vertex + np.asarray(shift, float), rot=R, Rx=m.Rx + dRx, Ry=m.Ry + dRy, poly=p,
                   kx=m.kx + dk[0], ky=m.ky + dk[1], holes=list(m.holes))


# ---------------------------------------------------------------- design helpers (paraxial)
def theta_of(R, d):
    """Phase advance per hit, cos θ = 1 - d/R (two identical mirrors); NaN if unstable."""
    c = 1 - d / R
    return float(np.arccos(c)) if -1 < c < 1 else np.nan


def reentrant_spacing(R, N, M):
    """Spacing for re-entrance after N hits with M turns: d = R (1 - cos 2πM/N)."""
    return float(R * (1 - np.cos(2 * np.pi * M / N)))


def injection(Rx, Ry, d, A, B):
    """Start point on mirror 1 (local (A, 0)) and direction for the paraxial pattern x_n = A cos nθx, y_n = B sin nθy."""
    xp = -A / Rx
    yp = B / Ry * np.sqrt((2 * Ry - d) / d)
    v = np.array([xp, yp, 1.0])
    return np.array([A, 0.0]), v / np.linalg.norm(v)


def paraxial_spots(n, d, A, B, Rx, Ry):
    """Paraxial spot positions (global x, y) for hit numbers n (1, 2, ...) of the injection() beam."""
    n = np.asarray(n, float)
    return A * np.cos(n * theta_of(Rx, d)), B * np.sin(n * theta_of(Ry, d))


def mode_radius(R, d, wavelength):
    """1/e² intensity radius of the cell's fundamental mode on the mirrors (symmetric two-mirror resonator,
    g = 1 - d/R): w² = (λ d / π) / sqrt(1 - g²)."""
    g = 1 - d / R
    return float(np.sqrt(wavelength * d / np.pi / np.sqrt(1 - g * g)))


# ---------------------------------------------------------------- tracing
def start_on_mirror(m: Mirror3D, xy, direction_local):
    """Global start point on mirror m at local (x, y) and the global direction for a local direction vector."""
    f, _, _ = m.sag(np.array([xy[0]]), np.array([xy[1]]))
    p = m.vertex + m.rot @ np.array([xy[0], xy[1], f[0]])
    v = m.rot @ np.asarray(direction_local, float)
    return p, v / np.linalg.norm(v)


@dataclass
class Trace3D:
    hits: np.ndarray        # (n_rays, n_max, 3) global hit points (NaN unused)
    local: np.ndarray       # (n_rays, n_max, 2) local (x, y) on the mirror hit
    mirror: np.ndarray      # (n_rays, n_max) mirror index (-1 unused)
    cos_inc: np.ndarray     # cosine of the angle of incidence at each hit
    chord: np.ndarray       # (n_rays, n_max) chord ending at each hit
    n_hits: np.ndarray      # reflections + the exit hit
    exit: np.ndarray        # per ray: 'hole', 'miss', 'trapped'
    exit_mirror: np.ndarray
    exit_hole: np.ndarray
    exit_dir: np.ndarray    # global direction when leaving (for 'hole')
    exit_normal: np.ndarray # surface normal at the exit point (as if the hole were mirror)
    start: np.ndarray
    start_dir: np.ndarray

    def path(self, r=0):
        return float(np.nansum(self.chord[r]))

    def reflections(self, r=0):
        return int(self.n_hits[r] - (1 if self.exit[r] == "hole" else 0))


def trace3d(mirrors, p0, d0, n_max=500):
    """Trace rays (p0, d0: (n, 3) or (3,)) through a list of mirrors until they pass a hole, miss an aperture or reach
    n_max hits. Every hit inside an aperture and outside the holes reflects."""
    p = np.atleast_2d(np.asarray(p0, float)).copy()
    d = np.atleast_2d(np.asarray(d0, float)).copy()
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = p.shape[0]
    hits = np.full((n, n_max, 3), np.nan)
    loc = np.full((n, n_max, 2), np.nan)
    mir = np.full((n, n_max), -1)
    cosi = np.full((n, n_max), np.nan)
    chord = np.full((n, n_max), np.nan)
    nh = np.zeros(n, int)
    exit_kind = np.array(["trapped"] * n, dtype=object)
    exit_m, exit_h = np.full(n, -1), np.full(n, -1)
    exit_dir = np.full((n, 3), np.nan)
    exit_nrm = np.full((n, 3), np.nan)
    start, start_dir = p.copy(), d.copy()
    alive = np.ones(n, bool)
    for j in range(n_max):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        best_t = np.full(ia.size, np.inf)
        best = [None] * 4
        best_m = np.full(ia.size, -1)
        for k, m in enumerate(mirrors):
            t, P, N, xy = m.intersect(p[ia], d[ia])
            upd = t < best_t
            best_t = np.where(upd, t, best_t)
            best_m = np.where(upd, k, best_m)
            for q, arr in enumerate((P, N, xy)):
                best[q] = arr.copy() if best[q] is None else np.where(upd[:, None], arr, best[q])
        P, N, xy = best[0], best[1], best[2]
        none = ~np.isfinite(best_t)
        exit_kind[ia[none]] = "miss"
        alive[ia[none]] = False
        ok = ~none
        hits[ia[ok], j] = P[ok]
        loc[ia[ok], j] = xy[ok]
        mir[ia[ok], j] = best_m[ok]
        chord[ia[ok], j] = best_t[ok]
        cosi[ia[ok], j] = np.abs((d[ia] * N).sum(1))[ok]
        nh[ia[ok]] += 1
        for q in np.nonzero(ok)[0]:
            r = ia[q]
            m = mirrors[best_m[q]]
            x, y = xy[q]
            if x * x + y * y > m.aperture**2:
                exit_kind[r] = "miss"; exit_m[r] = best_m[q]; alive[r] = False
                continue
            for hk, (hx, hy, hr) in enumerate(m.holes):
                if (x - hx) ** 2 + (y - hy) ** 2 <= hr * hr:
                    exit_kind[r] = "hole"; exit_m[r] = best_m[q]; exit_h[r] = hk; exit_dir[r] = d[r]; exit_nrm[r] = N[q]
                    alive[r] = False
                    break
        go = ia[ok & alive[ia]]
        sel = ok & alive[ia]
        dn = (d[go] * N[sel]).sum(1)
        d[go] = d[go] - 2 * dn[:, None] * N[sel]
        d[go] /= np.linalg.norm(d[go], axis=1, keepdims=True)
        p[go] = P[sel]
    return Trace3D(hits, loc, mir, cosi, chord, nh, exit_kind, exit_m, exit_h, exit_dir, exit_nrm, start, start_dir)


def herriott_cell(R, N, M, A, B=None, hole_radius=None, aperture=0.0254, wavelength=1.55e-6, Rx2=None, Ry2=None):
    """Ideal re-entrant Herriott design: spacing d = R (1 - cos 2πM/N), injection at local (A, 0) of mirror 1 through a
    hole of radius hole_radius (default: 3 mode radii, at most 0.4 × the nearest spot spacing). Returns
    dict(mirrors, d, p0, d0, theta, w_mode, hole_radius)."""
    B = A if B is None else B
    d = reentrant_spacing(R, N, M)
    w = mode_radius(R, d, wavelength)
    if hole_radius is None:
        th = theta_of(R, d)
        n = np.arange(2, N + 1, 2)
        xs, ys = A * np.cos(n * th), B * np.sin(n * th)
        sp = np.hypot(xs[:-1] - A, ys[:-1])                       # other mirror-1 spots to the hole centre
        hole_radius = float(min(3 * w, 0.4 * sp.min())) if sp.size else 3 * w
    mirrors = two_mirror_cell(d, (R, R), (Rx2 or R, Ry2 or R), aperture, holes1=[(A, 0.0, hole_radius)])
    xy, v = injection(R, R, d, A, B)
    p0, d0 = start_on_mirror(mirrors[0], xy, v)
    return dict(mirrors=mirrors, d=d, p0=p0, d0=d0, theta=theta_of(R, d), w_mode=w, hole_radius=hole_radius)


def astigmatic_cell(Rx, Ry, d, A, B, hole_radius, aperture=0.0254):
    """Two identical biconic mirrors (Rx, Ry) at spacing d, injected for x_n = A cos nθx, y_n = B sin nθy."""
    mirrors = two_mirror_cell(d, (Rx, Ry), (Rx, Ry), aperture, holes1=[(A, 0.0, hole_radius)])
    xy, v = injection(Rx, Ry, d, A, B)
    p0, d0 = start_on_mirror(mirrors[0], xy, v)
    return dict(mirrors=mirrors, d=d, p0=p0, d0=d0, theta_x=theta_of(Rx, d), theta_y=theta_of(Ry, d))


def astigmatic_reentrant(R_mean, N, Mx, My):
    """(Rx, Ry, d) for a re-entrant astigmatic cell: N θx = 2π Mx, N θy = 2π My, with d chosen so that the mean of
    Rx and Ry is R_mean."""
    cx, cy = 1 - np.cos(2 * np.pi * Mx / N), 1 - np.cos(2 * np.pi * My / N)      # d = R (1 - cos θ)
    d = 2 * R_mean / (1 / cx + 1 / cy)
    return float(d / cx), float(d / cy), float(d)


# ---------------------------------------------------------------- figures of merit
def reentrance(tr: Trace3D, r=0):
    """Exit summary of ray r: passes, path, exit kind, distance of the exit hit from the start point (exit_offset), and
    for a hole exit the re-entry angle: angle between the start direction and the exit ray reflected as if the hole
    were mirror (0 for a perfectly re-entrant cell, where the ray would repeat its first pass)."""
    k = tr.n_hits[r] - 1
    out = dict(exit=tr.exit[r], n_hits=int(tr.n_hits[r]), reflections=tr.reflections(r), path=tr.path(r))
    if k >= 0:
        out["exit_offset"] = float(np.linalg.norm(tr.hits[r, k] - tr.start[r]))
    if tr.exit[r] == "hole":
        v, nrm = tr.exit_dir[r], tr.exit_normal[r]
        refl = v - 2 * (v @ nrm) * nrm
        out["reentry_angle"] = float(np.arccos(np.clip(refl @ tr.start_dir[r] / np.linalg.norm(refl), -1, 1)))
    return out


def spot_metrics(tr: Trace3D, mirrors, r=0, w=None):
    """Per mirror: number of reflections, smallest spot-to-spot distance, and (mirrors with holes) the smallest
    clearance from a reflected spot to the hole edge; also in units of the beam radius w when given."""
    out = []
    last = tr.n_hits[r] - (1 if tr.exit[r] == "hole" else 0)
    for k, m in enumerate(mirrors):
        sel = np.nonzero(tr.mirror[r, :last] == k)[0]
        P = tr.local[r, sel]
        e = dict(mirror=m.label, reflections=int(sel.size), min_spacing=min_pairwise_distance(P))
        if m.holes and sel.size:
            hx, hy, hr = m.holes[0]
            e["hole_clearance"] = float((np.hypot(P[:, 0] - hx, P[:, 1] - hy) - hr).min())
        if w:
            e["min_spacing_in_w"] = e["min_spacing"] / w
            if "hole_clearance" in e:
                e["hole_clearance_in_w"] = e["hole_clearance"] / w
        out.append(e)
    return out


def herriott_effective_path(tr: Trace3D, R=1.0, r=0):
    """(path, Σ I_j ℓ_j, intensity left) of ray r with mirror reflectance R (scalar or callable of cos_inc)."""
    k = int(tr.n_hits[r])
    ch = tr.chord[r, :k]
    Rv = np.asarray(R(tr.cos_inc[r, :k]), float) if callable(R) else np.full(k, float(R))
    if tr.exit[r] == "hole":
        Rv[-1] = 1.0                                   # the exit hit passes through the hole: no reflection
    return effective_path(ch, Rv)


def twin_divergence_3d(mirrors, p0, d0, n_max=300, dslope=1e-9):
    """Separation of the hit points of two rays whose directions differ by dslope (rad, about local y), per hit,
    with a Lyapunov fit. Stable Herriott cells stay bounded or grow linearly."""
    d1 = rot_y(dslope) @ np.asarray(d0, float)
    tr = trace3d(mirrors, np.stack([p0, p0]), np.stack([d0, d1]), n_max)
    n = int(min(tr.n_hits))
    sep = np.linalg.norm(tr.hits[0, :n] - tr.hits[1, :n], axis=1)
    return sep, lyapunov_fit(sep, scale=float(np.linalg.norm(mirrors[1].vertex - mirrors[0].vertex)))
