"""Effective-index perturbations Δn_eff(x, y).

`Shape`: a round, slightly deformed dot r(θ) = R [1 + Σ_m (a_m cos mθ + b_m sin mθ)] about (x0, y0), filled with
Δn and a smooth tanh edge of width `edge`; value, gradient and Hessian are analytic.
`GridIndex` (index_map.py) holds an image. `Perturbation` sums components and lists their bounding circles;
outside every bounding circle Δn is zero to < 1e-6 of the peak, and the ray tracer only integrates inside them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TAIL_U = 7.0          # tanh tail: 0.5 (1 + tanh(-7)) ≈ 8e-7
STEP_U = 4.5          # fine RK4 steps only where |u| < STEP_U (|∇Δn| > 2e-4 of its peak)


@dataclass
class Shape:
    x0: float
    y0: float
    R: float
    dn: float
    edge: float = 3e-6
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))   # a[m-1] multiplies cos mθ, m = 1..M
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rotation: float = 0.0

    def __post_init__(self):
        self.a = np.asarray(self.a, float)
        self.b = np.asarray(self.b, float)
        M = max(len(self.a), len(self.b))
        self.a = np.pad(self.a, (0, M - len(self.a)))
        self.b = np.pad(self.b, (0, M - len(self.b)))

    @property
    def m(self):
        return np.arange(1, len(self.a) + 1)

    def radius(self, theta, deriv=0):
        """r(θ) and its θ-derivatives (deriv = 0, 1, 2)."""
        th = np.asarray(theta, float)[..., None] - self.rotation
        m = self.m
        c, s = np.cos(m * th), np.sin(m * th)
        if deriv == 0:
            return self.R * (1 + (self.a * c + self.b * s).sum(-1))
        if deriv == 1:
            return self.R * (m * (-self.a * s + self.b * c)).sum(-1)
        return self.R * (-(m**2) * (self.a * c + self.b * s)).sum(-1)

    def coefficients(self):
        """(a_m, b_m) of the contour in the cell frame, i.e. including the rotation."""
        ph = self.m * self.rotation
        return self.a * np.cos(ph) - self.b * np.sin(ph), self.a * np.sin(ph) + self.b * np.cos(ph)

    @property
    def bound(self):
        return self.R * (1 + np.abs(self.a).sum() + np.abs(self.b).sum()) + TAIL_U * self.edge

    def _radius_all(self, theta):
        th = np.asarray(theta, float)[..., None] - self.rotation
        m = self.m
        c, s = np.cos(m * th), np.sin(m * th)
        ac, bs = self.a * c, self.b * s
        r = self.R * (1 + (ac + bs).sum(-1))
        r1 = self.R * (m * (self.b * c - self.a * s)).sum(-1)
        r2 = self.R * (-(m**2) * (ac + bs)).sum(-1)
        return r, r1, r2

    def chord_integral(self, x, y, tx, ty, ln, n_newton=6, n_gl=12):
        """∫ Δn ds along chords r = (x, y) + s (tx, ty), 0 ≤ s ≤ ln. The two edge crossings come from the circle of
        radius R, refined by Newton steps on r(θ(s)) − ρ(s); the tanh edge zones (|u| < 7.5) around them are
        integrated with Gauss–Legendre, the flat interior contributes Δn × its length. Returns (values, ok); ok is
        False where the chord grazes the dot, edge windows overlap or Newton does not settle, and the caller
        integrates those chords numerically."""
        x, y, tx, ty, ln = (np.asarray(v, float) for v in (x, y, tx, ty, ln))
        dx, dy = x - self.x0, y - self.y0
        b = dx * tx + dy * ty
        c = dx * dx + dy * dy
        rmax = self.R * (1 + np.abs(self.a).sum() + np.abs(self.b).sum()) + TAIL_U * self.edge
        disc0 = b * b - (c - self.R**2)
        discm = b * b - (c - rmax**2)
        val = np.zeros(len(x))
        ok = discm <= 0                                       # misses the bounding circle: 0
        cand = ~ok & (disc0 > (0.1 * self.R) ** 2)
        if not cand.any():
            return val, ok
        idx = np.nonzero(cand)[0]
        X0, Y0, TX, TY = dx[idx], dy[idx], tx[idx], ty[idx]
        sq = np.sqrt(disc0[idx])
        out = []
        for s0 in (-b[idx] - sq, -b[idx] + sq):
            s_ = s0.copy()
            for _ in range(n_newton):
                px, py = X0 + s_ * TX, Y0 + s_ * TY
                rho = np.hypot(px, py)
                r, r1, _ = self._radius_xy(px, py, rho)
                fp = r1 * (px * TY - py * TX) / rho**2 - (px * TX + py * TY) / rho
                s_ = s_ - (r - rho) / np.where(np.abs(fp) > 1e-12, fp, 1e-12)
            px, py = X0 + s_ * TX, Y0 + s_ * TY
            rho = np.hypot(px, py)
            r, r1, _ = self._radius_xy(px, py, rho)
            fp = r1 * (px * TY - py * TX) / rho**2 - (px * TX + py * TY) / rho
            out.append((s_, np.abs(r - rho), 7.5 * self.edge / np.maximum(np.abs(fp), 1e-12)))
        (s1, e1, w1), (s2, e2, w2) = out
        good = ((e1 < 1e-9 * self.R) & (e2 < 1e-9 * self.R) & (s1 + w1 < s2 - w2) & (s1 - w1 > 0)
                & (s2 + w2 < ln[idx]))
        if good.any():
            g, wg = np.polynomial.legendre.leggauss(n_gl)
            j = idx[good]
            tot = self.dn * ((s2 - w2) - (s1 + w1))[good]
            for sc, wc in ((s1[good], w1[good]), (s2[good], w2[good])):
                S = sc[:, None] + wc[:, None] * g[None, :]
                v = self.eval(x[j][:, None] + S * tx[j][:, None], y[j][:, None] + S * ty[j][:, None], hess=False)[0]
                tot = tot + (v * wg[None, :]).sum(1) * wc
            val[j] = tot
            ok[j] = True
        return val, ok

    def _radius_xy(self, dx, dy, rho):
        """r, r', r'' at the direction of (dx, dy): cos mθ, sin mθ by recurrence (no trig calls)."""
        c0, s0 = dx / rho, dy / rho
        cr, sr = np.cos(self.rotation), np.sin(self.rotation)
        c1, s1 = c0 * cr + s0 * sr, s0 * cr - c0 * sr
        c, sn = c1, s1
        r = np.zeros(np.shape(dx)); r1 = np.zeros(np.shape(dx)); r2 = np.zeros(np.shape(dx))
        for k in range(len(self.a)):
            m = k + 1
            ac, bs = self.a[k] * c, self.b[k] * sn
            r = r + ac + bs
            r1 = r1 + m * (self.b[k] * c - self.a[k] * sn)
            r2 = r2 - m * m * (ac + bs)
            c, sn = c * c1 - sn * s1, sn * c1 + c * s1
        return self.R * (1 + r), self.R * r1, self.R * r2

    def radon(self, n_p=None, n_a=None):
        """Radon table of the dot about its own centre: T[i, j] = ∫ Δn along the line with direction angle
        α_i = π i / n_a (unit t = (cos α, sin α)) at signed offset p_j = (r − c) · (−sin α, cos α), p_j uniform on
        [−bound, bound]. Line integrals by chord_integral (32-point Gauss–Legendre for grazing lines)."""
        n_p = int(np.ceil(2 * self.bound / (self.edge / 3))) + 1 if n_p is None else n_p
        mdeg = max(1, len(self.a))
        n_a = int(np.ceil(np.pi / min(0.012 * 6 / mdeg, 0.05))) if n_a is None else n_a
        al = np.pi * np.arange(n_a) / n_a
        pp = np.linspace(-self.bound, self.bound, n_p)
        A, Pp = np.meshgrid(al, pp, indexing="ij")
        tx, ty = np.cos(A).ravel(), np.sin(A).ravel()
        nx, ny = -ty, tx
        L = self.bound
        x = self.x0 + Pp.ravel() * nx - 1.001 * L * tx
        y = self.y0 + Pp.ravel() * ny - 1.001 * L * ty
        ln = np.full(len(x), 2.002 * L)
        v, ok = self.chord_integral(x, y, tx, ty, ln)
        if (~ok).any():
            j = np.nonzero(~ok)[0]
            dx, dy = x[j] - self.x0, y[j] - self.y0
            b = dx * tx[j] + dy * ty[j]
            disc = np.maximum(b * b - (dx * dx + dy * dy - L * L), 0)
            s1, s2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
            g, wg = np.polynomial.legendre.leggauss(32)
            S = 0.5 * (s1 + s2)[:, None] + 0.5 * (s2 - s1)[:, None] * g[None, :]
            vv = self.eval(x[j][:, None] + S * tx[j][:, None], y[j][:, None] + S * ty[j][:, None], hess=False)[0]
            v[j] = (vv * wg[None, :]).sum(1) * 0.5 * (s2 - s1)
        return dict(T=v.reshape(n_a, n_p), n_a=n_a, p0=-self.bound, dp=pp[1] - pp[0], n_p=n_p)

    @staticmethod
    def radon_lookup(tab, cx, cy, x, y, tx, ty):
        """Line integrals for lines through (x, y) along (tx, ty) of a dot centred at (cx, cy), from its table."""
        al = np.arctan2(ty, tx)
        flip = al < 0
        al = np.where(flip, al + np.pi, al)
        al = np.where(al >= np.pi, al - np.pi, al)
        sgn = np.where(flip, -1.0, 1.0)
        p = sgn * ((x - cx) * (-ty) + (y - cy) * tx)
        fa = al / np.pi * tab["n_a"]
        ia = np.floor(fa).astype(int)
        wa = fa - ia
        ia0, ia1 = ia % tab["n_a"], (ia + 1) % tab["n_a"]
        fp = (p - tab["p0"]) / tab["dp"]
        ip = np.clip(np.floor(fp).astype(int), 0, tab["n_p"] - 2)
        wp = np.clip(fp - ip, 0, 1)
        T = tab["T"]
        # wrapping α: α + π is the same line with p → −p
        def val(iaa, wrap):
            pp = np.where(wrap, (tab["n_p"] - 1) - ip - 1, ip)
            ww = np.where(wrap, 1 - wp, wp)
            return T[iaa, pp] * (1 - ww) + T[iaa, pp + 1] * ww
        wrap1 = (ia + 1) >= tab["n_a"]
        inside = (p > tab["p0"]) & (p < -tab["p0"])
        out = (1 - wa) * val(ia0, np.zeros_like(wrap1)) + wa * val(ia1, wrap1)
        return np.where(inside, out, 0.0)

    def safe_step(self, x, y):
        """Distance a ray can move without entering the edge zone (|u| < STEP_U); ≥ 0."""
        dx, dy = np.asarray(x) - self.x0, np.asarray(y) - self.y0
        rho = np.maximum(np.hypot(dx, dy), 1e-4 * self.R)
        r = self._radius_xy(dx, dy, rho)[0]
        return np.maximum(0.5 * (np.abs(r - np.hypot(dx, dy)) - STEP_U * self.edge), 0.0)

    def eval(self, x, y, hess=True):
        """Δn, ∂x, ∂y and (if hess) ∂xx, ∂xy, ∂yy at (x, y)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx, dy = x - self.x0, y - self.y0
        rho = np.maximum(np.hypot(dx, dy), 1e-4 * self.R)
        r, r1, r2 = self._radius_xy(dx, dy, rho)
        w = self.edge
        u = (r - rho) / w
        t = np.tanh(u)
        S = 0.5 * (1 + t)
        S1 = 0.5 * (1 - t * t)
        rx, ry = dx / rho, dy / rho
        thx, thy = -dy / rho**2, dx / rho**2
        ux, uy = (r1 * thx - rx) / w, (r1 * thy - ry) / w
        out = [self.dn * S, self.dn * S1 * ux, self.dn * S1 * uy]
        if not hess:
            return out
        S2 = -2 * t * S1
        rho4 = rho**4
        thxx, thxy, thyy = 2 * dx * dy / rho4, (dy * dy - dx * dx) / rho4, -2 * dx * dy / rho4
        pxx, pxy, pyy = (1 - rx * rx) / rho, -rx * ry / rho, (1 - ry * ry) / rho
        uxx = (r2 * thx * thx + r1 * thxx - pxx) / w
        uxy = (r2 * thx * thy + r1 * thxy - pxy) / w
        uyy = (r2 * thy * thy + r1 * thyy - pyy) / w
        out += [self.dn * (S2 * ux * ux + S1 * uxx), self.dn * (S2 * ux * uy + S1 * uxy),
                self.dn * (S2 * uy * uy + S1 * uyy)]
        return out

    def to_dict(self):
        return dict(kind="shape", x0=self.x0, y0=self.y0, R=self.R, dn=self.dn, edge=self.edge,
                    a=self.a.tolist(), b=self.b.tolist(), rotation=self.rotation)


class Perturbation:
    """Sum of components (Shape, GridIndex, ...) with .eval(x, y, hess) and .bound / centre."""

    def __init__(self, components=()):
        self.components = list(components)

    def __len__(self):
        return len(self.components)

    @property
    def regions(self):
        """(K, 3) array of bounding circles (cx, cy, radius)."""
        if not self.components:
            return np.zeros((0, 3))
        return np.array([[c.x0, c.y0, c.bound] for c in self.components], float)

    def eval(self, x, y, hess=True):
        x, y = np.asarray(x, float), np.asarray(y, float)
        acc = [np.zeros(np.broadcast(x, y).shape) for _ in range(6 if hess else 3)]
        for c in self.components:
            for k, v in enumerate(c.eval(x, y, hess)):
                acc[k] = acc[k] + v
        return acc

    def safe_step(self, x, y, ds):
        """Per-point RK4 step: the smallest component safe step, at least ds (components without safe_step: ds)."""
        h = np.full(np.shape(x), np.inf)
        for c in self.components:
            f = getattr(c, "safe_step", None)
            h = np.minimum(h, f(x, y) if f else 0.0)
        return np.maximum(h, ds)

    def value(self, x, y):
        return self.eval(x, y, hess=False)[0]

    def check_inside(self, cell_radius, margin=0.0):
        reg = self.regions
        if len(reg) and np.any(np.hypot(reg[:, 0], reg[:, 1]) + reg[:, 2] > cell_radius - margin):
            raise ValueError("a perturbation's bounding circle reaches the cell wall")

    def rasterize(self, n=256, extent=None):
        """Δn on an n×n grid covering the cell (or `extent` = (xmin, xmax, ymin, ymax)); returns (img, extent)."""
        if extent is None:
            reg = self.regions
            r = np.max(np.hypot(reg[:, 0], reg[:, 1]) + reg[:, 2])
            extent = (-r, r, -r, r)
        xs = np.linspace(extent[0], extent[1], n)
        ys = np.linspace(extent[2], extent[3], n)
        X, Y = np.meshgrid(xs, ys)
        return self.value(X, Y), extent

    def to_dict(self):
        return [c.to_dict() for c in self.components]


def random_shape(rng, cell_radius, m_max=6, R_range=(40e-6, 120e-6), dn_range=(1e-3, 1e-3), edge=3e-6,
                 sigma=0.06, decay=1.0, centre=None, place_radius=None, margin=40e-6, rotation=False):
    """Random deformed dot: a_m, b_m ~ N(0, (sigma / (m-1)^decay)²) for m = 2..m_max (m = 1 left at 0, it is a
    shift), R and Δn uniform in their ranges, centre uniform in the disk that keeps the bounding circle `margin`
    away from the wall (or fixed)."""
    m = np.arange(1, m_max + 1)
    sd = np.where(m >= 2, sigma / np.maximum(m - 1, 1) ** decay, 0.0)
    a, b = rng.normal(0, 1, m_max) * sd, rng.normal(0, 1, m_max) * sd
    R = rng.uniform(*R_range)
    dn = rng.uniform(*dn_range)
    rot = rng.uniform(0, 2 * np.pi) if rotation else 0.0
    s = Shape(0.0, 0.0, R, dn, edge, a, b, rot)
    if centre is None:
        rmax = (cell_radius - margin - s.bound) if place_radius is None else place_radius
        rr, ph = rmax * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        centre = (rr * np.cos(ph), rr * np.sin(ph))
    s.x0, s.y0 = float(centre[0]), float(centre[1])
    return s
