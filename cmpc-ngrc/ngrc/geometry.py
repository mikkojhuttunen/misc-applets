"""CMPC geometry: wall, slab index, ports (input / output apertures in the wall) and detector lines.

`CircularCell`: analytic circular wall; a port sits at wall angle `angle` (from +x, counter-clockwise).
`WallCell`: any closed wall built from flat / curved mirror elements (`gmpc.planar.Cell2D`: stadium, polygon, ...);
a port sits at boundary arclength `s`.
Any ray that hits the wall inside a port leaves the cell there (input ports are holes too). Input ports launch a fan
of rays: positions across the aperture, directions = inward normal rotated by `launch` ± fan/2.

Both cells give the tracer the same interface: wall_hit (distance, outward normal, mirror focusing power, boundary
coordinate), port_at (port index of a hit), port_frame, check_regions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Port:
    angle: float                 # wall angle of the aperture centre (rad)
    width: float = 20e-6         # aperture chord (m)
    role: str = "inout"          # "in", "out" or "inout"
    launch: float = 0.0          # launch direction relative to the inward normal (rad, + = CCW)
    fan: float = 0.6             # full launch-angle spread of the ray fan (rad)
    label: str = ""
    s: float = None              # WallCell only: boundary arclength of the aperture centre (m)

    @property
    def is_input(self) -> bool:
        return self.role in ("in", "inout")

    @property
    def is_output(self) -> bool:
        return self.role in ("out", "inout")


@dataclass
class Detector:
    """Line of point detectors outside an output port, perpendicular to its outward normal."""
    distance: float = 100e-6     # from the port centre along the outward normal (m)
    width: float = 400e-6        # length of the line (m)
    n_pix: int = 64


@dataclass
class CircularCell:
    radius: float = 1.0e-3
    n_eff: float = 1.80          # slab-mode effective index of the unperturbed membrane
    wavelength: float = 1.55e-6
    ports: list = field(default_factory=list)
    reflectance: float = 0.99    # wall power reflectance (constant), or set reflectance_fn(cos_chi)
    reflection_phase: float = np.pi
    detector: Detector = field(default_factory=Detector)
    reflectance_fn: object = None
    model: str = "fga"            # "fga": frozen Gaussians (Herman–Kluk) with smooth leakage through the port
                                  # openings; "gbs": evolving Gaussian beamlets, hard ports (legacy)

    @property
    def k0(self) -> float:
        return 2 * np.pi / self.wavelength

    @property
    def inputs(self) -> list:
        return [i for i, p in enumerate(self.ports) if p.is_input]

    @property
    def outputs(self) -> list:
        return [i for i, p in enumerate(self.ports) if p.is_output]

    def R(self, cos_chi):
        if self.reflectance_fn is not None:
            return np.asarray(self.reflectance_fn(cos_chi), float)
        return np.full(np.shape(cos_chi), self.reflectance, float)

    def port_index(self, phi):
        """Index of the port that contains wall angle(s) phi, -1 for the mirror wall."""
        phi = np.atleast_1d(np.asarray(phi, float))
        out = np.full(phi.shape, -1, int)
        for i, p in enumerate(self.ports):
            half = np.arcsin(min(1.0, p.width / (2 * self.radius)))
            d = np.angle(np.exp(1j * (phi - p.angle)))
            out[np.abs(d) <= half] = i
        return out

    def port_frame(self, i):
        """Centre point, outward normal and tangent (CCW) of port i."""
        a = self.ports[i].angle
        nrm = np.array([np.cos(a), np.sin(a)])
        return self.radius * nrm, nrm, np.array([-nrm[1], nrm[0]])

    def detector_points(self, i):
        """(n_pix, 2) detector coordinates for output port i and the coordinate along the line (m)."""
        c, nrm, tan = self.port_frame(i)
        u = np.linspace(-0.5, 0.5, self.detector.n_pix) * self.detector.width
        return c + self.detector.distance * nrm + u[:, None] * tan, u

    # ---- tracer interface ----
    def wall_hit(self, x, y, tx, ty):
        """Distance to the wall along unit directions, outward normal, focusing power of the mirror (1/R for a
        concave circle; the beamlet sees P → P − 2 n0 Q · power / cos χ) and the boundary coordinate of the hit."""
        Rc = self.radius
        b = x * tx + y * ty
        d = -b + np.sqrt(np.maximum(b * b - (x * x + y * y) + Rc * Rc, 0))
        hx, hy = x + d * tx, y + d * ty
        rr = np.hypot(hx, hy)
        return d, hx / rr, hy / rr, np.full(np.shape(d), 1 / Rc), np.mod(np.arctan2(hy, hx), 2 * np.pi) * Rc

    def port_at(self, s_wall):
        return self.port_index(np.asarray(s_wall) / self.radius)

    def _port_centres(self):
        return np.array([p.angle * self.radius for p in self.ports]), 2 * np.pi * self.radius

    def port_offsets(self, s_wall):
        """(n_ports, n) signed boundary distance of hits from each port centre."""
        c, per = self._port_centres()
        s_wall = np.atleast_1d(np.asarray(s_wall, float))
        return np.mod(s_wall[None, :] - c[:, None] + per / 2, per) - per / 2

    def aperture_points(self, i, n=None):
        """Sample points across the opening of port i (on the wall), spacing ≤ λ/(3 n0); returns (points, du)."""
        p = self.ports[i]
        n = int(np.ceil(p.width / (self.wavelength / self.n_eff / 3))) + 1 if n is None else n
        u = (np.arange(n) + 0.5) / n * p.width - p.width / 2
        return self._wall_points(i, u), p.width / n

    def _wall_points(self, i, u):
        a = self.ports[i].angle + u / self.radius
        return self.radius * np.stack([np.cos(a), np.sin(a)], 1)

    def contains(self, x, y):
        return x * x + y * y < self.radius**2

    def check_regions(self, regions, margin=0.0):
        reg = np.asarray(regions)
        if len(reg) and np.any(np.hypot(reg[:, 0], reg[:, 1]) + reg[:, 2] > self.radius - margin):
            raise ValueError("a perturbation's bounding circle reaches the cell wall")

    def check(self):
        for i, p in enumerate(self.ports):
            if p.role not in ("in", "out", "inout"):
                raise ValueError(f"port {i}: role must be in/out/inout")
            if not 0 < p.width < self.radius:
                raise ValueError(f"port {i}: width must be in (0, radius)")
        idx = self.port_index(np.linspace(-np.pi, np.pi, 20001))
        if len(self.ports) > 1:
            centres = self.port_index(np.array([p.angle for p in self.ports]))
            if len(set(centres)) != len(self.ports):
                raise ValueError("ports overlap")
        return idx

    def variant(self, wavelength=None, launch_offset=0.0, launch=None):
        """Copy with another wavelength and/or launch angle on the input ports (offset added, or set): one
        "virtual node" set of a multiplexed reservoir."""
        import copy
        c = copy.deepcopy(self)
        if wavelength is not None:
            c.wavelength = wavelength
        for p in c.ports:
            if p.is_input:
                p.launch = (p.launch if launch is None else launch) + launch_offset
        return c

    def to_dict(self):
        d = asdict(self)
        d.pop("reflectance_fn")
        d.pop("wall", None)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["ports"] = [Port(**p) for p in d.get("ports", [])]
        d["detector"] = Detector(**d.get("detector", {}))
        return cls(**d)


@dataclass
class WallCell(CircularCell):
    """Cell with any closed mirror wall (`gmpc.planar.Cell2D`); ports at boundary arclength Port.s.
    radius is unused except as a length scale (set it to the half-width of the cell)."""
    wall: object = None

    def __post_init__(self):
        if self.wall is None:
            raise ValueError("WallCell needs a gmpc.planar.Cell2D wall")
        self._ox, self._oy = self.wall.outline(48)
        self.radius = 0.5 * max(np.ptp(self._ox), np.ptp(self._oy))

    def wall_hit(self, x, y, tx, ty):
        d, nx, ny, sb, k = self.wall.hit(np.asarray(x, float), np.asarray(y, float), np.asarray(tx, float),
                                          np.asarray(ty, float))
        power = np.where(k >= 0, -self.wall.kappa[np.maximum(k, 0)], 0.0)
        nn = np.hypot(nx, ny)
        nn = np.where(nn > 0, nn, 1.0)
        return d, nx / nn, ny / nn, power, sb

    def _port_centres(self):
        return np.array([p.s for p in self.ports], float), self.wall.perimeter

    def _wall_points(self, i, u):
        x, y, _, _ = self.wall.point_at(self.ports[i].s + np.asarray(u))
        return np.stack([x, y], 1)

    def port_at(self, s_wall):
        s_wall = np.atleast_1d(np.asarray(s_wall, float))
        out = np.full(s_wall.shape, -1, int)
        P = self.wall.perimeter
        for i, p in enumerate(self.ports):
            d = np.mod(s_wall - p.s + P / 2, P) - P / 2
            out[np.abs(d) <= p.width / 2] = i
        return out

    def port_frame(self, i):
        x, y, nx, ny = self.wall.point_at(np.array([self.ports[i].s]))
        nrm = np.array([nx[0], ny[0]])
        return np.array([x[0], y[0]]), nrm, np.array([-nrm[1], nrm[0]])

    def contains(self, x, y):
        return np.ones(np.shape(x), bool)

    def _inside_polygon(self, x, y):
        X, Y = self._ox, self._oy
        inside = False
        for i in range(len(X) - 1):
            if (Y[i] > y) != (Y[i + 1] > y) and x < X[i] + (y - Y[i]) * (X[i + 1] - X[i]) / (Y[i + 1] - Y[i]):
                inside = not inside
        return inside

    def check_regions(self, regions, margin=0.0):
        for cx, cy, r in np.asarray(regions).reshape(-1, 3):
            d = np.min(np.hypot(self._ox - cx, self._oy - cy))
            if not self._inside_polygon(cx, cy) or d < r + margin:
                raise ValueError("a perturbation's bounding circle reaches the cell wall")

    def check(self):
        for i, p in enumerate(self.ports):
            if p.s is None:
                raise ValueError(f"port {i}: WallCell ports need a boundary coordinate s")

    def to_dict(self):
        raise NotImplementedError("WallCell holds a Cell2D wall; rebuild it from its builder arguments")


def wall_ports(wall, positions, inputs=(0,), fractions=True, **kw):
    """Ports on a Cell2D wall at boundary coordinates (fractions of the perimeter by default)."""
    P = wall.perimeter
    return [Port(angle=0.0, s=(f * P if fractions else f), role="inout" if k in inputs else "out", label=f"P{k}", **kw)
            for k, f in enumerate(positions)]


def stadium(radius=0.5e-3, straight=1.0e-3, **kw):
    """Smooth stadium wall (gmpc.planar.stadium_cell): half-circles of `radius` joined by straights."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "general-mpc"))
    from gmpc.planar import stadium_cell
    return stadium_cell(radius, straight, **kw)


def symmetric_ports(n, offset=0.0, inputs=(0,), **kw):
    """n ports equally spaced on the wall starting at `offset` (rad); ports listed in `inputs` are "inout", the rest
    "out"."""
    return [Port(angle=offset + 2 * np.pi * k / n, role="inout" if k in inputs else "out", label=f"P{k}", **kw)
            for k in range(n)]


def ports_at(angles_deg, inputs=(0,), **kw):
    """Ports at arbitrary wall angles (degrees), e.g. an asymmetric layout."""
    return [Port(angle=np.deg2rad(a), role="inout" if k in inputs else "out", label=f"P{k}", **kw)
            for k, a in enumerate(angles_deg)]


def membrane_neff(wavelength=1.55e-6, thickness=300e-9, n_core=2.0, n_clad=1.0, polarization="TE"):
    """TE0 effective index of a free-standing membrane (math-engines membrane_mode)."""
    from engines.membrane_mode.engine import solve_mode
    return solve_mode(wavelength, thickness, n_core=n_core, n_clad=n_clad, polarization=polarization).neff


def dneff_dthickness(wavelength=1.55e-6, thickness=300e-9, n_core=2.0, n_clad=1.0, polarization="TE", h=1e-9):
    """∂n_eff/∂t (1/m): converts a local thickness change Δt of an etched / deposited dot to Δn_eff."""
    f = lambda t: membrane_neff(wavelength, t, n_core, n_clad, polarization)
    return (f(thickness + h) - f(thickness - h)) / (2 * h)
