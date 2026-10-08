"""Planar (2D) multipass cells whose wall is a chain of mirror elements: circle, segmented polygon (CMPC), smooth or
faceted stadium, and the integrated (on-chip, in-plane) Herriott cell; ray tracing with ports and the CMPC path
statistics. Grown out of general-mpc (gmpc.planar, gmpc.trace2d), which now imports this engine.

Every element is a chord A -> B (counter-clockwise around the cell) with a signed curvature κ:

    κ = 0   flat facet
    κ < 0   concave as seen from inside: the element bulges outward (a piece of a circular wall, radius 1/|κ|)
    κ > 0   convex as seen from inside: bulges inward (dispersing, strongly mixing)

so one element type covers smooth circular walls, flat facets and curved facets. The outward normal of a chord is
n = (t_y, -t_x), t = (B - A)/|B - A|. The wall does not have to be closed: the integrated Herriott cell is two
facing concave arcs, and rays that pass the mirror ends leave the cell (exit -2, "leaked").

Builders:  polygon_cell (regular N-gon, the CMPC; same convention as billiard_cell.SegmentedCell), circle_cell,
stadium_cell (smooth or faceted caps, single or segmented straights; s as billiard_cell.Stadium), herriott_planar_cell
(two concave in-plane mirrors, re-entrant design, input and output through one window at the mirror edge).
perturb() tilts each element about its chord midpoint, shifts it along its normal and changes its curvature,
randomly (rms values, seed, optional label selection) and/or with explicit per-element arrays. Endpoints move with
the element, so perturbed walls have tiny gaps or overlaps at the joints, which `ext` (the fractional extension of an
element's hit window past its chord ends, 0.15 for facets as in the CMPC) covers for faceted walls.

Boundary coordinate s: arclength along the elements from the start of element 0 (chord projection for flat facets,
arc angle × radius for curved ones). Ports are windows in s. sin χ is the signed sine of the angle of incidence.

Integrated Herriott design (paraxial, one transverse dimension y, mirrors of radius R at spacing d): cos θ = 1 - d/R,
spot n at y_n = y0 cos nθ + sqrt(d/(4f - d)) (y0 + 2f y0') sin nθ with f = R/2 (Herriott, Kogelnik & Kompfner,
Appl. Opt. 3, 523 (1964)). Launching from the mirror point y0 = A along its normal (through the centre of curvature,
y0' = -A/R) gives y_n = A cos nθ: A is the outermost spot, so the input/output window sits at the mirror edge.
Re-entrant after N hits when N θ = 2π M, d = R (1 - cos 2πM/N); the ray then leaves through the window it came in by.
In a slab the in-plane mode radius on the mirrors is w² = (λ/n_eff) d / (π sqrt(1 - g²)), g = 1 - d/R. SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

SHAPES = ("circle", "polygon", "stadium", "faceted_stadium")


@dataclass
class Element:
    A: tuple
    B: tuple
    kappa: float = 0.0
    ext: float = 0.0
    label: str = ""


class Cell2D:
    """A planar mirror wall (elements in counter-clockwise order) with vectorised ray hits."""

    def __init__(self, elements, name="cell", s_in=None, s_out=None, meta=None):
        if len(elements) < 2:
            raise ValueError("a cell needs at least two elements")
        self.elements = list(elements)
        self.name = name
        E = len(elements)
        self.A = np.array([e.A for e in elements], float)
        self.B = np.array([e.B for e in elements], float)
        self.kappa = np.array([e.kappa for e in elements], float)
        self.ext = np.array([e.ext for e in elements], float)
        self.labels = [e.label for e in elements]
        ch = self.B - self.A
        self.h = 0.5 * np.hypot(ch[:, 0], ch[:, 1])
        self.t = ch / (2 * self.h[:, None])
        self.n = np.stack([self.t[:, 1], -self.t[:, 0]], 1)
        self.M = 0.5 * (self.A + self.B)
        self.R = np.full(E, np.inf)
        self.C = np.full((E, 2), np.nan)
        self.L = 2 * self.h.copy()
        self.sense = np.ones(E)
        for k in range(E):
            if self.kappa[k] == 0:
                continue
            R = 1 / abs(self.kappa[k])
            if R < self.h[k] * (1 - 1e-12):
                raise ValueError(f"element {k}: radius {R:g} smaller than half its chord {self.h[k]:g}")
            d0 = np.sqrt(max(R * R - self.h[k] ** 2, 0.0))
            self.R[k] = R
            self.C[k] = self.M[k] - d0 * self.n[k] if self.kappa[k] < 0 else self.M[k] + d0 * self.n[k]
            self.L[k] = 2 * R * np.arcsin(min(self.h[k] / R, 1.0))
            vA, vB = self.A[k] - self.C[k], self.B[k] - self.C[k]
            self.sense[k] = np.sign(vA[0] * vB[1] - vA[1] * vB[0]) or 1.0
        self.s0 = np.concatenate([[0.0], np.cumsum(self.L)[:-1]])
        self.perimeter = float(self.L.sum())
        self.n_elements = E
        self.s_in_default = float(self.L[0] / 2) if s_in is None else float(s_in)
        self.s_out_default = s_out
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- geometry
    @property
    def closed(self):
        gap = np.hypot(*(self.A[np.r_[1:self.n_elements, 0]] - self.B).T)
        return bool(np.all(gap <= 1e-6 * self.perimeter))

    @property
    def area(self):
        """Area enclosed by the chords plus the circular segments of curved elements (shoelace + segments).
        For an open wall (integrated Herriott) the chords are joined end to end."""
        x, y = self.A[:, 0], self.A[:, 1]
        xn, yn = self.B[:, 0], self.B[:, 1]
        a = 0.5 * np.sum(x * yn - xn * y)
        nxt = np.r_[1:self.n_elements, 0]
        a += 0.5 * np.sum(self.B[:, 0] * self.A[nxt, 1] - self.A[nxt, 0] * self.B[:, 1])   # joints (0 when closed)
        for k in np.nonzero(self.kappa)[0]:
            R, al = self.R[k], np.arcsin(min(self.h[k] / self.R[k], 1.0))
            seg = R * R * (al - np.sin(al) * np.cos(al))
            a += seg if self.kappa[k] < 0 else -seg
        return float(a)

    @property
    def mean_chord(self):
        return np.pi * self.area / self.perimeter

    @property
    def extent(self):
        """Half widths (x, y) of a box centred on the origin that holds the wall."""
        ox, oy = self.outline(16)
        return float(np.max(np.abs(ox))), float(np.max(np.abs(oy)))

    def segment_of(self, s):
        s = np.mod(np.asarray(s, float), self.perimeter)
        return np.clip(np.searchsorted(self.s0, s, side="right") - 1, 0, self.n_elements - 1)

    def point_at(self, s):
        """Boundary point and outward normal at boundary coordinate s (arrays)."""
        s = np.mod(np.asarray(s, float), self.perimeter)
        k = self.segment_of(s)
        loc = s - self.s0[k]
        x, y, nx, ny = (np.empty_like(s) for _ in range(4))
        flat = self.kappa[k] == 0
        kf = k[flat]
        f = loc[flat] / self.L[kf]
        x[flat] = self.A[kf, 0] + f * (self.B[kf, 0] - self.A[kf, 0])
        y[flat] = self.A[kf, 1] + f * (self.B[kf, 1] - self.A[kf, 1])
        nx[flat], ny[flat] = self.n[kf, 0], self.n[kf, 1]
        kc = k[~flat]
        if kc.size:
            ang = loc[~flat] / self.R[kc] * self.sense[kc]
            vx, vy = self.A[kc, 0] - self.C[kc, 0], self.A[kc, 1] - self.C[kc, 1]
            c, sn = np.cos(ang), np.sin(ang)
            px, py = vx * c - vy * sn, vx * sn + vy * c
            x[~flat], y[~flat] = self.C[kc, 0] + px, self.C[kc, 1] + py
            sg = np.where(self.kappa[kc] < 0, 1.0, -1.0)
            nx[~flat], ny[~flat] = sg * px / self.R[kc], sg * py / self.R[kc]
        return x, y, nx, ny

    def outline(self, per_element=24):
        """Wall points element by element (closed walls: back to the start; open walls: NaN between elements)."""
        pts = []
        closed = self.closed
        for k in range(self.n_elements):
            s = self.s0[k] + np.linspace(0, self.L[k], per_element)
            x, y, _, _ = self.point_at(np.minimum(s, self.s0[k] + self.L[k] * (1 - 1e-12)))
            pts.append(np.stack([x, y], 1))
            if not closed:
                pts.append(np.full((1, 2), np.nan))
        P = np.concatenate(pts + ([pts[0][:1]] if closed else []))
        return P[:, 0], P[:, 1]

    def inside(self, x, y):
        """Point-in-cell test (even-odd rule) on the outline polygon, curved elements sampled; open walls closed by
        joining the element ends."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ox, oy = [], []
        for k in range(self.n_elements):
            s = self.s0[k] + np.linspace(0, self.L[k] * (1 - 1e-12), 16)
            px, py, _, _ = self.point_at(s)
            ox.append(px)
            oy.append(py)
        ox, oy = np.concatenate(ox), np.concatenate(oy)
        inside = np.zeros(x.shape, bool)
        j = len(ox) - 1
        for i in range(len(ox)):
            cond = ((oy[i] > y) != (oy[j] > y)) & (x < (ox[j] - ox[i]) * (y - oy[i]) / (oy[j] - oy[i] + 1e-300) + ox[i])
            inside ^= cond
            j = i
        return inside

    def hit_k(self, px, py, dx, dy):
        """First wall hit of rays from interior points along unit directions: t, outward normal, s, element index
        (t = inf and element -1 where a ray leaves through a gap or past an open wall)."""
        px, py, dx, dy = (np.atleast_1d(np.asarray(v, float)) for v in (px, py, dx, dy))
        n = px.size
        eps = 1e-12 * self.perimeter
        bt = np.full(n, np.inf)
        bnx, bny, bs = np.zeros(n), np.zeros(n), np.zeros(n)
        bk = np.full(n, -1)
        for k in range(self.n_elements):
            hh = self.h[k] * (1 + self.ext[k])
            Mx, My = self.M[k]
            tx, ty = self.t[k]
            nkx, nky = self.n[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                if self.kappa[k] == 0:
                    dn = dx * nkx + dy * nky
                    t = ((Mx - px) * nkx + (My - py) * nky) / dn
                    u = (px + t * dx - Mx) * tx + (py + t * dy - My) * ty
                    ok = (dn > 1e-12) & (t > eps) & (np.abs(u) <= hh)
                    nx_, ny_ = np.full(n, nkx), np.full(n, nky)
                    sl = np.clip(u, -self.h[k], self.h[k]) + self.h[k]
                else:
                    R = self.R[k]
                    cx, cy = self.C[k]
                    ox, oy = px - cx, py - cy
                    b = ox * dx + oy * dy
                    disc = b * b - (ox * ox + oy * oy - R * R)
                    sq = np.sqrt(np.maximum(disc, 0))
                    t = -b + sq if self.kappa[k] < 0 else -b - sq
                    hx, hy = px + t * dx, py + t * dy
                    side = ((hx - cx) * nkx + (hy - cy) * nky) * (-np.sign(self.kappa[k])) > 0
                    u = (hx - Mx) * tx + (hy - My) * ty
                    ok = (disc >= 0) & (t > eps) & side & (np.abs(u) <= hh)
                    sg = 1.0 if self.kappa[k] < 0 else -1.0
                    nx_, ny_ = sg * (hx - cx) / R, sg * (hy - cy) / R
                    vAx, vAy = self.A[k, 0] - cx, self.A[k, 1] - cy
                    vHx, vHy = hx - cx, hy - cy
                    ang = np.arctan2(vAx * vHy - vAy * vHx, vAx * vHx + vAy * vHy) * self.sense[k]
                    sl = np.clip(R * ang, 0.0, self.L[k])
            upd = ok & (t < bt)
            bt = np.where(upd, t, bt)
            bnx, bny = np.where(upd, nx_, bnx), np.where(upd, ny_, bny)
            bs = np.where(upd, self.s0[k] + sl, bs)
            bk = np.where(upd, k, bk)
        return bt, bnx, bny, bs, bk

    def hit(self, px, py, dx, dy):
        """billiard_cell interface: t, outward normal, s."""
        return self.hit_k(px, py, dx, dy)[:4]


# ---------------------------------------------------------------- builders
def _arc_elements(c, r, phi0, phi1, n_pieces, faceted, ext, label):
    """Elements along a circle (centre c, radius r) from angle phi0 to phi1 (counter-clockwise)."""
    ph = np.linspace(phi0, phi1, n_pieces + 1)
    out = []
    for a, b in zip(ph[:-1], ph[1:]):
        A = (c[0] + r * np.cos(a), c[1] + r * np.sin(a))
        B = (c[0] + r * np.cos(b), c[1] + r * np.sin(b))
        out.append(Element(A, B, 0.0 if faceted else -1.0 / r, ext if faceted else 0.0, label))
    return out


def polygon_cell(r, n_facets=24, ext=0.15):
    """Regular N-gon of flat facets, circumradius r, facet k centred at angle 2πk/N (same convention as
    billiard_cell.SegmentedCell, so s_in_default is the middle of facet 0)."""
    N = int(n_facets)
    els = []
    for k in range(N):
        a, b = 2 * np.pi * k / N - np.pi / N, 2 * np.pi * k / N + np.pi / N
        els.append(Element((r * np.cos(a), r * np.sin(a)), (r * np.cos(b), r * np.sin(b)), 0.0, ext, f"facet{k}"))
    return Cell2D(els, f"polygon N={N}")


def circle_cell(r, n_arcs=4):
    """Smooth circular wall made of n_arcs concave arcs (no corners)."""
    els = _arc_elements((0.0, 0.0), r, -np.pi / n_arcs, 2 * np.pi - np.pi / n_arcs, int(n_arcs), False, 0.0, "circle")
    return Cell2D(els, "circle")


def stadium_cell(r, a, cap_facets=None, straight_segments=1, ext=0.15):
    """Stadium: straights of length a at y = ±r joined by half-circles of radius r centred at (±a/2, 0).
    cap_facets=None: smooth caps (two concave arcs each); an integer: each cap is that many flat facets inscribed in
    the half-circle. straight_segments: number of flat facets per straight. s = 0 at the left end of the bottom
    straight (as billiard_cell.Stadium); s_in_default is the middle of the bottom straight."""
    els = []
    ns = int(straight_segments)
    xs = np.linspace(-a / 2, a / 2, ns + 1)
    seg_ext = ext if ns > 1 or cap_facets else 0.0
    if a > 0:
        for i in range(ns):
            els.append(Element((xs[i], -r), (xs[i + 1], -r), 0.0, seg_ext, "straight_bottom"))
    faceted = cap_facets is not None
    npc = int(cap_facets) if faceted else 2
    els += _arc_elements((a / 2, 0.0), r, -np.pi / 2, np.pi / 2, npc, faceted, ext, "cap_right")
    if a > 0:
        for i in range(ns):
            els.append(Element((xs[ns - i], r), (xs[ns - i - 1], r), 0.0, seg_ext, "straight_top"))
    els += _arc_elements((-a / 2, 0.0), r, np.pi / 2, 3 * np.pi / 2, npc, faceted, ext, "cap_left")
    name = f"stadium a={a:g} r={r:g}" + (f", {npc} facets per cap" if faceted else ", smooth caps")
    return Cell2D(els, name, s_in=a / 2 if a > 0 else None)


def herriott_planar_cell(R, N, M, A, port_w=None, aperture=None, R2=None, d=None, phase=0.0):
    """Integrated (in-plane) Herriott cell: mirror M1 (vertex at x = -d/2, centre of curvature at x = -d/2 + R) and
    M2 (vertex at x = +d/2, radius R2, default R) facing each other, transverse coordinate y. Re-entrant design
    d = R (1 - cos 2πM/N) unless d is given (N even: the ray comes back to M1 after N hits).
    Spot pattern y_n = A cos(nθ + phase): the input/output window (width port_w, default A/10) is centred on the point
    y0 = A cos(phase) of M1 and meta['theta_launch'] is the paraxial launch angle from the mirror normal there.
    phase = 0: window at the mirror edge, launch along the normal (exactly through the centre of curvature); the
    pattern runs out and back, so spots coincide in pairs. phase = π/N: all N/2 spots per mirror distinct, the window
    becomes a slot just inside the edge. aperture: half width of both mirrors in y (default A + port_w).
    Element 0 = M1 (running from y = +a to -a), element 1 = M2 (y = -a to +a)."""
    require_positive(R=R, A=A)
    N, M = int(N), int(M)
    if N < 4 or N % 2:
        raise ValueError("N must be even and >= 4 (the ray returns to the window on M1 after N hits)")
    if not 0 < M < N / 2:
        raise ValueError("need 0 < M < N/2 for a stable re-entrant design")
    d = reentrant_spacing(R, N, M) if d is None else float(d)
    require_positive(d=d)
    R2 = R if R2 is None else float(R2)
    w = A / 10 if port_w is None else float(port_w)
    a = A + w if aperture is None else float(aperture)
    y0 = A * np.cos(phase)
    if a >= min(R, R2):
        raise ValueError("aperture must be smaller than the mirror radii")
    if a < max(A, abs(y0) + w / 2):
        raise ValueError("aperture must hold the spot pattern and the window")
    c1, c2 = (-d / 2 + R, 0.0), (d / 2 - R2, 0.0)
    p1, p2 = np.arcsin(a / R), np.arcsin(a / R2)
    m1 = _arc_elements(c1, R, np.pi - p1, np.pi + p1, 1, False, 0.0, "M1")
    m2 = _arc_elements(c2, R2, -p2, p2, 1, False, 0.0, "M2")
    s_in = R * (p1 - np.arcsin(y0 / R))
    th = theta_of(R, d)
    th_launch = float(np.arctan(-A * np.sin(phase) * np.sqrt((2 * R - d) / d) / R)) if phase else 0.0
    meta = dict(R=R, R2=R2, d=d, N=N, M=M, A=A, aperture=a, port_w=w, theta=th, phase=float(phase), y0=float(y0),
                theta_launch=th_launch)
    return Cell2D(m1 + m2, f"integrated Herriott R={R:g} d={d:g}", s_in=s_in, s_out=s_in, meta=meta)


def herriott_planar_spots(meta):
    """Paraxial design spots: (y on M1 for even hits 2..N-2, y on M2 for odd hits 1..N-1), the minimum spot-to-spot
    distance on each mirror, and the window clearance (closest M1 spot to the window centre minus half the window)."""
    N, A, ph, th = meta["N"], meta["A"], meta["phase"], meta["theta"]
    n = np.arange(1, N)
    y = A * np.cos(n * th + ph)
    y1, y2 = y[n % 2 == 0], y[n % 2 == 1]
    sp = min(min_pairwise_distance(np.r_[y1, meta["y0"]][:, None]), min_pairwise_distance(y2[:, None]))
    clear = float(np.min(np.abs(y1 - meta["y0"])) - meta["port_w"] / 2) if y1.size else np.inf
    return y1, y2, sp, clear


def perturb(cell, tilt_rms=0.0, offset_rms=0.0, curvature_rms=0.0, seed=0, select=None,
            tilts=None, offsets=None, curvatures=None):
    """New cell with each element tilted about its chord midpoint (rad, counter-clockwise), shifted along its outward
    normal (m) and its curvature changed (1/m). Random rms values use one normal draw per element (seed), applied only
    to elements whose label contains `select` (None = all); explicit per-element arrays are added on top. The port
    positions (s_in, s_out) and meta data are kept."""
    rng = np.random.default_rng(seed)
    E = cell.n_elements
    mask = np.array([select is None or select in lab for lab in cell.labels], float)
    tl = tilt_rms * rng.standard_normal(E) * mask
    of = offset_rms * rng.standard_normal(E) * mask
    cu = curvature_rms * rng.standard_normal(E) * mask
    for name, v in (("tilts", tilts), ("offsets", offsets), ("curvatures", curvatures)):
        if v is not None and np.shape(v) != (E,):
            raise ValueError(f"{name} must have one value per element ({E})")
    tl = tl + (0 if tilts is None else np.asarray(tilts, float))
    of = of + (0 if offsets is None else np.asarray(offsets, float))
    cu = cu + (0 if curvatures is None else np.asarray(curvatures, float))
    els = []
    for k, e in enumerate(cell.elements):
        M0 = cell.M[k]
        M = M0 + of[k] * cell.n[k]
        c, s = np.cos(tl[k]), np.sin(tl[k])
        rot = lambda P: (M[0] + c * (P[0] - M0[0]) - s * (P[1] - M0[1]), M[1] + s * (P[0] - M0[0]) + c * (P[1] - M0[1]))
        els.append(Element(rot(e.A), rot(e.B), e.kappa + cu[k], e.ext, e.label))
    return Cell2D(els, cell.name + " (perturbed)", s_in=cell.s_in_default, s_out=cell.s_out_default, meta=cell.meta)


def make_cell(shape="stadium", radius=5e-3, straight=5e-3, n_facets=24, tilt_rms=0.0, curvature_rms=0.0,
              offset_rms=0.0, seed=1, select=None, straight_segments=1):
    """circle (radius), polygon (circumradius, n_facets), stadium (cap radius, straight), faceted_stadium (n_facets
    flat facets per cap, straight_segments per straight), with random element perturbations."""
    require_choice("shape", shape, SHAPES)
    require_positive(radius=radius)
    require_nonnegative(straight=straight, tilt_rms=tilt_rms, curvature_rms=curvature_rms, offset_rms=offset_rms)
    if shape in ("polygon", "faceted_stadium") and int(n_facets) < (3 if shape == "polygon" else 1):
        raise ValueError("too few facets")
    if shape == "circle":
        cell = circle_cell(radius)
    elif shape == "polygon":
        cell = polygon_cell(radius, int(n_facets))
    elif shape == "stadium":
        cell = stadium_cell(radius, straight)
    else:
        cell = stadium_cell(radius, straight, cap_facets=int(n_facets), straight_segments=int(straight_segments))
    if tilt_rms or curvature_rms or offset_rms:
        cell = perturb(cell, tilt_rms, offset_rms, curvature_rms, int(seed), select)
    return cell


# ---------------------------------------------------------------- integrated Herriott design (paraxial)
def theta_of(R, d):
    """Phase advance per hit, cos θ = 1 - d/R (two identical mirrors); NaN if unstable."""
    c = 1 - d / R
    return float(np.arccos(c)) if -1 < c < 1 else np.nan


def reentrant_spacing(R, N, M):
    """Spacing for re-entrance after N hits with M turns: d = R (1 - cos 2πM/N)."""
    return float(R * (1 - np.cos(2 * np.pi * M / N)))


def paraxial_spots(n, R, d, y0, slope0):
    """Paraxial transverse spot positions for hit numbers n (1, 2, ...): y_n = y0 cos nθ + sqrt(d/(4f - d)) (y0 + 2f y0')
    sin nθ, f = R/2."""
    n = np.asarray(n, float)
    th, f = theta_of(R, d), R / 2
    return y0 * np.cos(n * th) + np.sqrt(d / (4 * f - d)) * (y0 + 2 * f * slope0) * np.sin(n * th)


def mode_radius(R, d, wavelength, n_index=1.0):
    """1/e² radius of the cell's fundamental mode on the mirrors (symmetric two-mirror resonator, g = 1 - d/R), for a
    medium of index n (slab n_eff in a membrane): w² = (λ/n) d / (π sqrt(1 - g²))."""
    g = 1 - d / R
    return float(np.sqrt(wavelength / n_index * d / np.pi / np.sqrt(1 - g * g)))


# ---------------------------------------------------------------- tracing
def reflect(dx, dy, nx, ny):
    dn = dx * nx + dy * ny
    rx, ry = dx - 2 * dn * nx, dy - 2 * dn * ny
    nn = np.hypot(rx, ry)
    return rx / nn, ry / nn


def launch(cell, s, theta):
    """Points on the wall at s and unit directions at angle theta from the inward normal (counter-clockwise +)."""
    s, theta = np.broadcast_arrays(np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(theta, float)))
    x, y, nx, ny = cell.point_at(s)
    ux, uy = -nx, -ny
    return x, y, ux * np.cos(theta) - uy * np.sin(theta), ux * np.sin(theta) + uy * np.cos(theta)


def in_window(s, s0, w, P):
    d = np.mod(s - s0 + P / 2, P) - P / 2
    return np.abs(d) <= w / 2


@dataclass
class RayTable:
    cell: object
    port_w: float
    s_in: float
    s_out: float
    n_rays: int
    n_bounce: int
    chord: np.ndarray        # (n_rays, n_bounce): chord j ends at hit j
    sinchi: np.ndarray       # signed sin of the incidence angle at hit j
    element: np.ndarray      # element index of hit j (-1 unused)
    exit_idx: np.ndarray     # hit index in a port, -1 still inside
    exit_port: np.ndarray    # 1 output, 0 input, -1 inside (n_bounce / max_path reached), -2 leaked through a gap
    start: np.ndarray = field(default=None, repr=False)
    hits: np.ndarray = field(default=None, repr=False)

    @property
    def trapped_fraction(self):
        return float(np.mean(self.exit_port == -1))

    @property
    def leaked_fraction(self):
        return float(np.mean(self.exit_port == -2))


def trace_rays(cell, port_w, n_rays=1000, n_bounce=1500, theta0=0.35, s_in=None, s_out_frac=0.37, seed=1,
               theta_c=0.0, max_path=None, keep_hits=False, ports=True, s_out=None) -> RayTable:
    """Rays launched across the input port window (position uniform, angle uniform in theta_c ± theta0) and traced
    until they hit the output window (checked first) or the input window, n_bounce or max_path. The output window is
    centred on s_out, else cell.s_out_default (the integrated Herriott cell: the input window itself), else
    s_in + s_out_frac × perimeter. ports=False traces a closed cell (no exits)."""
    rng = np.random.default_rng(seed)
    P = cell.perimeter
    s_in = cell.s_in_default if s_in is None else s_in
    if s_out is None:
        s_out = cell.s_out_default if getattr(cell, "s_out_default", None) is not None else s_in + s_out_frac * P
    x, y, dx, dy = launch(cell, s_in + (rng.random(n_rays) - 0.5) * port_w, theta_c + (rng.random(n_rays) * 2 - 1) * theta0)
    start = np.stack([x, y, dx, dy], 1).copy()
    chord = np.zeros((n_rays, n_bounce))
    sinchi = np.zeros((n_rays, n_bounce))
    elem = np.full((n_rays, n_bounce), -1)
    hits = np.full((n_rays, n_bounce, 2), np.nan) if keep_hits else None
    exit_idx = np.full(n_rays, -1)
    exit_port = np.full(n_rays, -1)
    alive = np.ones(n_rays, bool)
    Lc = np.zeros(n_rays)
    for j in range(n_bounce):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        t, nx, ny, hs, hk = cell.hit_k(x[ia], y[ia], dx[ia], dy[ia])
        bad = ~np.isfinite(t)
        if bad.any():
            exit_port[ia[bad]] = -2
            alive[ia[bad]] = False
            t = np.where(bad, 0.0, t)
        hx, hy = x[ia] + t * dx[ia], y[ia] + t * dy[ia]
        chord[ia, j] = t
        sinchi[ia, j] = dx[ia] * ny - dy[ia] * nx
        elem[ia, j] = hk
        if keep_hits:
            hits[ia, j, 0], hits[ia, j, 1] = hx, hy
        keep = ~bad
        if ports:
            pin, pout = in_window(hs, s_in, port_w, P), in_window(hs, s_out, port_w, P)
            done = (pin | pout) & ~bad
            exit_idx[ia[done]] = j
            exit_port[ia[done]] = np.where(pout[done], 1, 0)
            alive[ia[done]] = False
            keep &= ~done
        if max_path is not None:
            Lc[ia] += t
            over = keep & (Lc[ia] >= max_path)
            alive[ia[over]] = False
            keep &= ~over
        ik = ia[keep]
        x[ik], y[ik] = hx[keep], hy[keep]
        dx[ik], dy[ik] = reflect(dx[ia][keep], dy[ia][keep], nx[keep], ny[keep])
    return RayTable(cell, port_w, s_in, s_out, n_rays, n_bounce, chord, sinchi, elem, exit_idx, exit_port, start, hits)


@dataclass
class CellResult:
    T_det: float
    L_mean: float
    L_eff_gas: float
    S1: float
    L: np.ndarray
    W: np.ndarray
    n_det: int
    L_mean_err: float


def evaluate(table: RayTable, R_of_sinchi=None, alpha_bg=0.0, Gamma=1.0) -> CellResult:
    """Power-weighted statistics of the rays that reach the output port. R_of_sinchi(|sin χ|) -> reflectance
    (None = 1); every reflection before the exit counts, the exit hit itself does not."""
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    if sel.size == 0:
        return CellResult(0.0, np.nan, np.nan, 0.0, np.array([]), np.array([]), 0, np.nan)
    e = table.exit_idx[sel]
    ncol = int(e.max()) + 1
    j = np.arange(ncol)[None, :]
    L = (table.chord[sel, :ncol] * (j <= e[:, None])).sum(1)
    lnR = 0.0
    if R_of_sinchi is not None:
        R = np.asarray(R_of_sinchi(np.abs(table.sinchi[sel, :ncol])), float)
        lnR = (np.log(np.clip(R, 1e-300, 1.0)) * (j < e[:, None])).sum(1)
    W = np.exp(lnR - alpha_bg * L)
    if W.sum() == 0:
        return CellResult(0.0, np.nan, np.nan, 0.0, L, W, int(sel.size), np.nan)
    T = W.sum() / table.n_rays
    Lm = (W * L).sum() / W.sum()
    var = (W * (L - Lm) ** 2).sum() / W.sum()
    n_k = W.sum() ** 2 / (W**2).sum()
    return CellResult(float(T), float(Lm), float(Gamma * Lm), float(Gamma * T * Lm), L, W, int(sel.size), float(np.sqrt(var / max(n_k, 1))))


def mean_field_estimate(cell, port_w, R_mean=1.0, alpha_bg=0.0, Gamma=1.0):
    """Ergodic (fully chaotic) estimate with two ports of width w: mean chord πA/P, exit probability w/P per hit."""
    ell = cell.mean_chord
    eta = port_w / cell.perimeter
    surv = (1 - 2 * eta) * R_mean * np.exp(-alpha_bg * ell)
    T = eta * np.exp(-alpha_bg * ell) / (1 - surv)
    L = ell / (1 - surv)
    return dict(T_det=T, L_mean=L, n_bounce=1 / (1 - surv), L_eff_gas=Gamma * L, S1=Gamma * T * L)


def trace_path(cell, s, theta, n_hits=50, port=None):
    """Polyline (x, y) of one trajectory from wall coordinate s at angle theta, and the hit table (s, signed sin χ,
    element). Stops when the ray leaves through a gap; port = (s0, w) also stops it at the first hit in that window
    (the hit is kept)."""
    x, y, dx, dy = launch(cell, s, theta)
    xs, ys, S, SC, K = [x[0]], [y[0]], [], [], []
    for _ in range(n_hits):
        t, nx, ny, hs, hk = cell.hit_k(x, y, dx, dy)
        if not np.isfinite(t[0]):
            break
        x, y = x + t * dx, y + t * dy
        S.append(hs[0]); SC.append(float(dx[0] * ny[0] - dy[0] * nx[0])); K.append(int(hk[0]))
        xs.append(x[0]); ys.append(y[0])
        if port is not None and in_window(hs[0], port[0], port[1], cell.perimeter):
            break
        dx, dy = reflect(dx, dy, nx, ny)
    return np.array(xs), np.array(ys), np.array(S), np.array(SC), np.array(K)


def poincare(cell, n_rays=40, n_hits=300, seed=0):
    """(s, signed sin χ) of trajectories started uniformly in boundary phase space; arrays (n_rays, n_hits)."""
    rng = np.random.default_rng(seed)
    x, y, dx, dy = launch(cell, rng.random(n_rays) * cell.perimeter, np.arcsin(rng.uniform(-0.98, 0.98, n_rays)))
    S, SC = np.full((n_rays, n_hits), np.nan), np.full((n_rays, n_hits), np.nan)
    ok = np.ones(n_rays, bool)
    for j in range(n_hits):
        t, nx, ny, hs, _ = cell.hit_k(x, y, dx, dy)
        ok &= np.isfinite(t)
        t = np.where(ok, t, 0.0)
        x, y = x + t * dx, y + t * dy
        S[:, j] = np.where(ok, hs, np.nan)
        SC[:, j] = np.where(ok, dx * ny - dy * nx, np.nan)
        dx, dy = reflect(dx, dy, np.where(ok, nx, 1.0), np.where(ok, ny, 0.0))
    return S, SC


def lyapunov_fit(sep, scale, lo_rel=1e-10, hi_rel=1e-2, min_points=6):
    """Least-squares slope of ln(separation) against hit number over the stretch between lo_rel·scale and hi_rel·scale.
    Returns dict(lam = exponent per hit, intercept, first, last) or None. Exponential growth (lam clearly > 0 over a
    long stretch) means chaos; regular and pseudo-integrable cells grow linearly (lam -> small and falling)."""
    sep = np.asarray(sep, float)
    j = np.nonzero((sep > lo_rel * scale) & (sep < hi_rel * scale))[0]
    if j.size < min_points:
        return None
    x, y = j + 1.0, np.log(sep[j])
    A = np.vstack([x, np.ones_like(x)]).T
    (lam, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(lam=float(lam), intercept=float(c), first=int(x[0]), last=int(x[-1]))


def twin_divergence(cell, s, theta, n_hits=300, dtheta=1e-9):
    """Distance between the hit points of two rays launched dtheta apart, per hit, and the Lyapunov fit
    (exponent per reflection; None when there is no clean exponential stretch)."""
    a = trace_path(cell, s, theta, n_hits)
    b = trace_path(cell, s, theta + dtheta, n_hits)
    n = min(len(a[0]), len(b[0]))
    sep = np.hypot(a[0][1:n] - b[0][1:n], a[1][1:n] - b[1][1:n])
    return sep, lyapunov_fit(sep, scale=np.sqrt(abs(cell.area)))


def effective_path(chords, R):
    """(geometric path, absorption-weighted path Σ I_j ℓ_j, intensity left) of one ray: chord j carries the product of
    the reflectances of the hits before it. R: scalar or one value per reflection (len(chords) values)."""
    ch = np.asarray(chords, float)
    Rj = np.broadcast_to(np.asarray(R, float), ch.shape)
    I = np.concatenate([[1.0], np.cumprod(Rj)])
    return float(ch.sum()), float((I[:-1] * ch).sum()), float(I[-1])


def min_pairwise_distance(P):
    """Smallest distance between distinct points of an (n, d) array (O(n²), fine for spot patterns)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.inf
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def herriott_planar_trace(cell, dtheta=0.0, n_max=2000):
    """Trace the centre ray of an integrated Herriott cell from its window (launch angle meta['theta_launch'] + dtheta)
    until it leaves (window or past a mirror end). Returns dict(exit = 'port' | 'leak' | 'trapped', n_hits (reflections + the exit hit), path, spots per
    mirror (transverse y of the reflections), exit_offset (distance along the wall from the window centre, m),
    exit_angle (angle between the exit ray and the reversed launch direction mirrored about the normal, rad))."""
    w = cell.meta.get("port_w", 0.0)
    theta = cell.meta.get("theta_launch", 0.0) + dtheta
    xs, ys, S, SC, K = trace_path(cell, cell.s_in_default, theta, n_max, port=(cell.s_in_default, w))
    P = cell.perimeter
    nh = len(S)
    out = dict(n_hits=nh, path=float(np.hypot(np.diff(xs), np.diff(ys)).sum()))
    if nh and in_window(S[-1], cell.s_in_default, w, P):
        out["exit"] = "port"
        out["exit_offset"] = float(np.mod(S[-1] - cell.s_in_default + P / 2, P) - P / 2)
        # the launch leaves with sin χ = sin θ; a ray arriving with the same sin χ would repeat its first pass
        out["exit_angle"] = float(np.arcsin(np.clip(SC[-1], -1, 1)) - theta)
    else:
        out["exit"] = "leak" if nh < n_max else "trapped"
    refl = slice(0, nh - (1 if out["exit"] == "port" else 0))
    out["reflections"] = refl.stop
    out["spots"] = [ys[1:][refl][K[refl] == k] for k in range(cell.n_elements)]
    out["chords"] = np.hypot(np.diff(xs), np.diff(ys))
    out["xs"], out["ys"], out["s"], out["sinchi"], out["element"] = xs, ys, S, SC, K
    return out


# ---------------------------------------------------------------- Result front ends
def _shape_inputs(shape, radius, straight, n_facets, port_width=None):
    require_choice("shape", shape, SHAPES)
    require_positive(radius=radius)
    if port_width is not None:
        require_positive(port_width=port_width)
    require_nonnegative(straight=straight)


def ray_statistics(shape="faceted_stadium", radius=5e-3, straight=5e-3, n_facets=12, tilt_rms=0.0, curvature_rms=0.0,
                   seed=1, port_width=150e-6, theta0=0.349066, R_mean=0.999, alpha_bg=0.0, Gamma=1.0,
                   n_rays=1000, n_bounce=1500) -> Result:
    """Monte-Carlo path statistics of a planar cell with an input and an output port (output 0.37 P along the wall),
    with the ergodic mean-field values for comparison."""
    _shape_inputs(shape, radius, straight, n_facets, port_width)
    require_positive(theta0=theta0)
    require_range("R_mean", R_mean, 0.0, 1.0)
    require_nonnegative(alpha_bg=alpha_bg, Gamma=Gamma)
    n_rays, n_bounce = int(n_rays), int(n_bounce)
    if n_rays < 1 or n_bounce < 1:
        raise ValueError("n_rays and n_bounce must be >= 1")
    cell = make_cell(shape, radius, straight, n_facets, tilt_rms, curvature_rms, seed=seed)
    if 2 * port_width >= cell.perimeter:
        raise ValueError("port_width must be below half the perimeter")
    tab = trace_rays(cell, port_width, n_rays, n_bounce, theta0, seed=int(seed))
    r = evaluate(tab, lambda s: np.full_like(s, R_mean), alpha_bg, Gamma)
    mf = mean_field_estimate(cell, port_width, R_mean, alpha_bg, Gamma)
    return Result(
        values={"T_det": r.T_det, "L_mean": r.L_mean, "L_mean_err": r.L_mean_err, "L_eff_gas": r.L_eff_gas, "S1": r.S1,
                "n_det": r.n_det, "trapped_fraction": tab.trapped_fraction, "leaked_fraction": tab.leaked_fraction,
                "mean_chord": cell.mean_chord, "area": cell.area, "perimeter": cell.perimeter,
                "T_mean_field": mf["T_det"], "L_mean_field": mf["L_mean"]},
        units={"T_det": "", "L_mean": "m", "L_mean_err": "m", "L_eff_gas": "m", "S1": "m", "n_det": "", "trapped_fraction": "",
               "leaked_fraction": "", "mean_chord": "m", "area": "m^2", "perimeter": "m", "T_mean_field": "", "L_mean_field": "m"},
        assumptions=["2D ray optics, specular reflection", "Ideal port windows; light hitting the input port is lost",
                     "Angle-independent mirror reflectance R_mean", "Random element perturbations: one normal draw per element (seed)",
                     "Mean field assumes a fully chaotic cell; regular and faceted cells deviate"],
    )


def chaos(shape="stadium", radius=5e-3, straight=5e-3, n_facets=12, tilt_rms=0.0, curvature_rms=0.0, seed=1,
          theta=0.5, n_hits=300) -> Result:
    """Twin-ray Lyapunov exponent per reflection: two rays launched 1 nrad apart from the default input point."""
    _shape_inputs(shape, radius, straight, n_facets)
    require_range("theta", theta, -1.55, 1.55)
    cell = make_cell(shape, radius, straight, n_facets, tilt_rms, curvature_rms, seed=seed)
    sep, fit = twin_divergence(cell, cell.s_in_default, theta, int(n_hits))
    lam = fit["lam"] if fit else 0.0
    return Result(
        values={"lyapunov": lam, "chaotic": bool(fit is not None and lam > 0.05), "final_separation": float(sep[-1]) if sep.size else np.nan,
                "fit_first": fit["first"] if fit else 0, "fit_last": fit["last"] if fit else 0},
        units={"lyapunov": "1/reflection", "chaotic": "", "final_separation": "m", "fit_first": "", "fit_last": ""},
        assumptions=["Exponent from ln(separation) between 1e-10 and 1e-2 of the cell size; 0 when no exponential stretch",
                     "chaotic: exponent > 0.05 per reflection"],
    )


def planar_herriott(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, port_width=100e-6, wavelength=1.55e-6, n_index=2.479,
                    R_mirror=0.999, dR2=0.0, tilt2=0.0, spacing_error=0.0, dtheta=0.0) -> Result:
    """Integrated (in-plane) Herriott cell: re-entrant design (d, θ, path N d, mode radius, spot spacing, window
    clearance) and the traced centre ray, with a radius error and a tilt of mirror M2, a spacing error and a
    launch-angle offset dtheta."""
    require_positive(R=R, A=A, port_width=port_width, wavelength=wavelength, n_index=n_index)
    require_range("R_mirror", R_mirror, 0.0, 1.0)
    require_range("dtheta", dtheta, -0.5, 0.5)
    N, M = int(N), int(M)
    cell0 = herriott_planar_cell(R, N, M, A, port_width, phase=phase)
    d = cell0.meta["d"]
    cell = herriott_planar_cell(R, N, M, A, port_width, R2=R + dR2, d=d + spacing_error, phase=phase)
    cell.meta["theta_launch"] = cell0.meta["theta_launch"]
    if tilt2:
        cell = perturb(cell, tilts=[0.0, tilt2])
    tr = herriott_planar_trace(cell, dtheta, n_max=10 * N + 10)
    w = mode_radius(R, d, wavelength, n_index)
    _, _, spacing, clear = herriott_planar_spots(cell0.meta)
    nref = tr["reflections"]
    ch = tr["chords"]
    L, Leff, I = effective_path(ch, np.r_[np.full(nref, R_mirror), np.ones(len(ch) - nref)])
    sag = R - np.sqrt(R * R - cell0.meta["aperture"] ** 2)
    return Result(
        values={"d": d, "theta": cell0.meta["theta"], "theta_launch": cell0.meta["theta_launch"], "path_design": N * d,
                "mode_radius": w, "spot_spacing_design": spacing, "spot_spacing_in_w": spacing / w,
                "window_clearance_in_w": clear / w, "exits_through_port": tr["exit"] == "port", "reflections": nref,
                "path": tr["path"], "exit_offset": tr.get("exit_offset", np.nan), "exit_angle": tr.get("exit_angle", np.nan),
                "L_eff": Leff, "I_end": I, "footprint_length": d + 2 * sag, "footprint_width": 2 * cell0.meta["aperture"]},
        units={"d": "m", "theta": "rad", "theta_launch": "rad", "path_design": "m", "mode_radius": "m", "spot_spacing_design": "m",
               "spot_spacing_in_w": "", "window_clearance_in_w": "", "exits_through_port": "", "reflections": "", "path": "m",
               "exit_offset": "m", "exit_angle": "rad", "L_eff": "m", "I_end": "", "footprint_length": "m", "footprint_width": "m"},
        assumptions=["2D ray optics between two concave in-plane mirrors (exact circular arcs)",
                     "Launch from the window centre on M1 at theta_launch + dtheta from the mirror normal; pattern y_n = A cos(nθ + phase)",
                     "phase = 0 puts spots on top of each other in pairs (spacing 0); phase = π/N separates them",
                     "Paraxial design values (d, θ, spot spacing, window clearance, mode radius with λ/n_eff)",
                     "exit_offset: distance along M1 between the exit hit and the window centre; exit_angle: 0 = repeats its first pass",
                     "Effective path with a constant mirror reflectance R_mirror"],
    )


def _phase_front_end(cell, theta_c, amplitude, n_pass, wavelength, n_index, waveform):
    from ..ray_phase.engine import dither_analysis
    r = dither_analysis(cell, theta_c, amplitude, int(n_pass), wavelength, n_index, waveform, n_max=6401)
    i = int(n_pass) - 1
    return Result(
        values={"V_first": float(r["V"][0]), "V_last": float(r["V"][i]), "V_model_last": float(r["V_model"][i]),
                "x_last": float(r["x"][i]), "chirp_last": float(r["q"][i]), "same_path_last": float(r["same_path"][i]),
                "A_half_last": float(r["A_05"][i]), "L_last": float(r["L0"][i]), "noise_floor": r["noise_floor"],
                "resolved": bool(r["resolved"][i])},
        units={"V_first": "", "V_last": "", "V_model_last": "", "x_last": "rad", "chirp_last": "rad", "same_path_last": "",
               "A_half_last": "rad", "L_last": "m", "noise_floor": "", "resolved": ""},
        assumptions=["ray_phase.dither_analysis on the planar cell: ray optics, reflection phases ignored",
                     "Model V: linear phase, |J0(aA)| (sine) or |sinc(aA)| (triangle)", "Path identity = element sequence"],
    )


def dither_visibility(shape="faceted_stadium", radius=5e-3, straight=5e-3, n_facets=12, tilt_rms=0.0, curvature_rms=0.0,
                      seed=1, theta_c=0.5, wavelength=1.55e-6, n_index=1.0, amplitude=1e-5, n_pass=20, waveform="sine") -> Result:
    """Fringe visibility at pass 1 and pass n_pass under launch-angle dither in a circle, polygon or (faceted) stadium."""
    _shape_inputs(shape, radius, straight, n_facets)
    require_positive(wavelength=wavelength, n_index=n_index, amplitude=amplitude)
    require_range("theta_c", theta_c, -1.55, 1.55)
    require_choice("waveform", waveform, ("sine", "triangle"))
    if int(n_pass) < 1:
        raise ValueError("n_pass must be >= 1")
    cell = make_cell(shape, radius, straight, n_facets, tilt_rms, curvature_rms, seed=seed)
    return _phase_front_end(cell, theta_c, amplitude, n_pass, wavelength, n_index, waveform)


def planar_herriott_dither(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, dtheta=0.0, wavelength=1.55e-6, n_index=2.479,
                           amplitude=1e-4, n_pass=19, waveform="sine") -> Result:
    """Fringe visibility under launch-angle dither in an integrated Herriott cell (launch from the window at the design
    angle + dtheta), pass 1 and pass n_pass (n_pass < N keeps the ray inside)."""
    require_positive(wavelength=wavelength, n_index=n_index, amplitude=amplitude)
    require_choice("waveform", waveform, ("sine", "triangle"))
    if int(n_pass) < 1:
        raise ValueError("n_pass must be >= 1")
    cell = herriott_planar_cell(R, N, M, A, phase=phase)
    return _phase_front_end(cell, cell.meta["theta_launch"] + dtheta, amplitude, n_pass, wavelength, n_index, waveform)


def _mirror_front_end(cell, cell_ref, theta_c, fan, n_beams, n_hits, n_eff, polarization, periods, sin_design, wavelength):
    from ..bragg_grating.engine import TrenchDBR
    from ..cell_mirror.engine import cell_mirror_stats
    require_positive(n_eff=n_eff, wavelength=wavelength)
    require_choice("polarization", polarization, ("TE", "TM"))
    if int(n_beams) < 1 or int(n_hits) < 1 or int(periods) < 1:
        raise ValueError("n_beams, n_hits and periods must be >= 1")
    dbr = TrenchDBR(n_tooth=n_eff, lam_design=wavelength, N=int(periods), m_gap=1, m_tooth=1, bounce_loss=0.0,
                    slab_pol=polarization, sin_design=sin_design)
    a = cell_mirror_stats(cell, theta_c, fan, int(n_beams), int(n_hits), dbr)
    b = cell_mirror_stats(cell_ref, theta_c, fan, int(n_beams), int(n_hits), dbr)
    vals = {k: a[k] for k in ("R_mean", "R_eff", "I_end", "L_eff", "L_geom", "chi_50", "chi_95", "chi_max")}
    vals.update({f"{k}_regular": b[k] for k in ("R_eff", "L_eff", "chi_95")})
    vals.update(R_design=a["R_design"], R_uniform=a["R_uniform"], loss_per_bounce=1 - a["R_eff"])
    units = {k: "" for k in vals}
    units.update(L_eff="m", L_geom="m", L_eff_regular="m", chi_50="deg", chi_95="deg", chi_max="deg", chi_95_regular="deg")
    return Result(values=vals, units=units, assumptions=[
        "cell_mirror statistics on the planar cell; mirror = first-order etched-trench DBR (bragg_grating.TrenchDBR, no extra loss) at every hit",
        "Self-consistent weights: each ray loses R(χ) at every hit; NaN hits after a ray leaves the cell",
        "_regular: same launch, unperturbed cell"])


def mirror_reflectance(shape="faceted_stadium", radius=5e-3, straight=5e-3, n_facets=12, tilt_rms=0.0, curvature_rms=0.0,
                       seed=1, theta_c=0.5, fan=0.01745, n_beams=7, n_hits=150, n_eff=2.479, polarization="TM", periods=6,
                       sin_design=0.0, wavelength=1.55e-6) -> Result:
    """Etched-trench DBR reflectance averaged over the angles the light meets in a circle, polygon or (faceted) stadium,
    with the same launch in the unperturbed cell."""
    _shape_inputs(shape, radius, straight, n_facets)
    require_range("theta_c", theta_c, -1.55, 1.55)
    require_nonnegative(fan=fan)
    cell = make_cell(shape, radius, straight, n_facets, tilt_rms, curvature_rms, seed=seed)
    ref = make_cell(shape, radius, straight, n_facets)
    return _mirror_front_end(cell, ref, theta_c, fan, n_beams, n_hits, n_eff, polarization, periods, sin_design, wavelength)


def planar_herriott_mirror(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, fan=0.0, n_beams=1, n_hits=19, n_eff=2.479,
                           polarization="TM", periods=4, sin_design=0.0, wavelength=1.55e-6, tilt2=0.0) -> Result:
    """Etched-trench DBR reflectance over the angles of an integrated Herriott cell (small angles of incidence, the
    best case for a DBR designed at normal incidence), with an optional tilt of M2 against the ideal cell."""
    cell0 = herriott_planar_cell(R, N, M, A, phase=phase)
    cell = perturb(cell0, tilts=[0.0, tilt2]) if tilt2 else cell0
    return _mirror_front_end(cell, cell0, cell0.meta["theta_launch"], fan, n_beams, n_hits, n_eff, polarization, periods, sin_design, wavelength)
