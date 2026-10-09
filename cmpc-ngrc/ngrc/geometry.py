"""Circular CMPC: wall, slab index, ports (input / output apertures in the wall) and detector lines.

Angles are measured from +x, counter-clockwise. A port is an aperture of chord width `width` centred at wall angle
`angle`; any ray that hits the wall inside a port leaves the cell there (input ports are holes too). Input ports
launch a fan of rays: positions across the aperture, directions = inward normal rotated by `launch` ± fan/2.
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

    def to_dict(self):
        d = asdict(self)
        d.pop("reflectance_fn")
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["ports"] = [Port(**p) for p in d.get("ports", [])]
        d["detector"] = Detector(**d.get("detector", {}))
        return cls(**d)


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
