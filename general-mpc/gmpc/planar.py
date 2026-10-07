"""Planar (2D) multipass cells whose wall is a closed chain of mirror elements.

Every element is a chord A -> B (counter-clockwise around the cell) with a signed curvature κ:

    κ = 0   flat facet
    κ < 0   concave as seen from inside: the element bulges outward (a piece of a circular wall, radius 1/|κ|)
    κ > 0   convex as seen from inside: bulges inward (dispersing, strongly mixing)

so one element type covers smooth circular walls (one arc per quarter or half circle), flat facets (the segmented
CMPC) and curved facets. The outward normal of a chord is n = (t_y, -t_x), t = (B - A)/|B - A|.

Builders:  polygon_cell (regular N-gon, the CMPC), circle_cell (smooth), stadium_cell (smooth or faceted caps,
single or segmented straights). perturb() tilts each element about its chord midpoint, shifts it along its normal and
changes its curvature, randomly (rms values, seed) and/or with explicit per-element arrays; endpoints stay attached
to the element, so perturbed walls have tiny gaps or overlaps at the joints, which `ext` (the fractional extension of
an element's hit window past its chord ends) covers for faceted walls.

Boundary coordinate s: arclength along the elements from the start of element 0 (chord projection for flat facets,
arc angle × radius for curved ones). SI units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Element:
    A: tuple
    B: tuple
    kappa: float = 0.0
    ext: float = 0.0
    label: str = ""


class Cell2D:
    """A closed planar mirror wall (elements in counter-clockwise order) with vectorised ray hits."""

    def __init__(self, elements, name="cell"):
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
        self.s_in_default = float(self.L[0] / 2)

    # ---------------------------------------------------------------- geometry
    @property
    def area(self):
        """Area enclosed by the chords plus the circular segments of curved elements (shoelace + segments)."""
        x, y = self.A[:, 0], self.A[:, 1]
        xn, yn = self.B[:, 0], self.B[:, 1]
        a = 0.5 * np.sum(x * yn - xn * y)
        for k in np.nonzero(self.kappa)[0]:
            R, al = self.R[k], np.arcsin(min(self.h[k] / self.R[k], 1.0))
            seg = R * R * (al - np.sin(al) * np.cos(al))
            a += seg if self.kappa[k] < 0 else -seg
        return float(a)

    @property
    def mean_chord(self):
        return np.pi * self.area / self.perimeter

    def point_at(self, s):
        """Boundary point and outward normal at boundary coordinate s (arrays)."""
        s = np.mod(np.asarray(s, float), self.perimeter)
        k = np.clip(np.searchsorted(self.s0, s, side="right") - 1, 0, self.n_elements - 1)
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
        pts = []
        for k in range(self.n_elements):
            s = self.s0[k] + np.linspace(0, self.L[k], per_element)
            x, y, _, _ = self.point_at(s)
            pts.append(np.stack([x, y], 1))
        P = np.concatenate(pts + [pts[0][:1]])
        return P[:, 0], P[:, 1]

    def inside(self, x, y):
        """Point-in-cell test on the outline polygon (curved elements sampled)."""
        from matplotlib.path import Path as _P  # optional, only for plotting helpers
        ox, oy = self.outline(16)
        return _P(np.stack([ox, oy], 1)).contains_points(np.stack([np.ravel(x), np.ravel(y)], 1)).reshape(np.shape(x))

    def hit(self, px, py, dx, dy):
        """First wall hit of rays from interior points along unit directions: t, outward normal, s, element index."""
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
    math-engines billiard_cell.SegmentedCell, so s_in_default is the middle of facet 0)."""
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
    straight (as billiard_cell.Stadium)."""
    els = []
    ns = int(straight_segments)
    xs = np.linspace(-a / 2, a / 2, ns + 1)
    if a > 0:
        for i in range(ns):
            els.append(Element((xs[i], -r), (xs[i + 1], -r), 0.0, ext if ns > 1 or cap_facets else 0.0, "straight_bottom"))
    faceted = cap_facets is not None
    npc = int(cap_facets) if faceted else 2
    els += _arc_elements((a / 2, 0.0), r, -np.pi / 2, np.pi / 2, npc, faceted, ext, "cap_right")
    if a > 0:
        for i in range(ns):
            els.append(Element((xs[ns - i], r), (xs[ns - i - 1], r), 0.0, ext if ns > 1 or cap_facets else 0.0, "straight_top"))
    els += _arc_elements((-a / 2, 0.0), r, np.pi / 2, 3 * np.pi / 2, npc, faceted, ext, "cap_left")
    return Cell2D(els, f"stadium a={a:g} r={r:g}" + (f", {npc} facets per cap" if faceted else ", smooth caps"))


def perturb(cell, tilt_rms=0.0, offset_rms=0.0, curvature_rms=0.0, seed=0, select=None,
            tilts=None, offsets=None, curvatures=None):
    """New cell with each element tilted about its chord midpoint (rad), shifted along its outward normal (m) and its
    curvature changed (1/m). Random rms values use one normal draw per element (seed), applied only to elements whose
    label contains `select` (None = all); explicit per-element arrays are added on top."""
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
    return Cell2D(els, cell.name + " (perturbed)")
