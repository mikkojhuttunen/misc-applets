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

    def safe_step(self, x, y):
        """Distance a ray can move without entering the edge zone (|u| < STEP_U); ≥ 0."""
        dx, dy = np.asarray(x) - self.x0, np.asarray(y) - self.y0
        rho = np.hypot(dx, dy)
        r = self._radius_all(np.arctan2(dy, dx))[0]
        return np.maximum(0.5 * (np.abs(r - rho) - STEP_U * self.edge), 0.0)

    def eval(self, x, y, hess=True):
        """Δn, ∂x, ∂y and (if hess) ∂xx, ∂xy, ∂yy at (x, y)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx, dy = x - self.x0, y - self.y0
        rho = np.maximum(np.hypot(dx, dy), 1e-4 * self.R)
        th = np.arctan2(dy, dx)
        r, r1, r2 = self._radius_all(th)
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
