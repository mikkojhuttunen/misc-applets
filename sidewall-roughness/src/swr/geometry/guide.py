from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Cfg
from .. import materials as mat
from ..orientation import diagonal_eps, oriented_medium, rotation_for
from ..slab import Layer


@dataclass(frozen=True)
class Segment:
    """A sidewall in the cross-section, from the top corner (y0, z0) to the bottom corner (y1, z1)."""
    y0: float
    z0: float
    y1: float
    z1: float

    def distance(self, y, z):
        p = np.stack([np.asarray(y) - self.y0, np.asarray(z) - self.z0], axis=-1)
        d = np.array([self.y1 - self.y0, self.z1 - self.z0])
        t = np.clip(p @ d / (d @ d), 0.0, 1.0)
        return np.linalg.norm(p - t[..., None] * d, axis=-1)


@dataclass
class WallProfile:
    """Top-view outward offsets of the +y (left) and -y (right) walls, sampled at x."""
    x: np.ndarray
    h_left: np.ndarray
    h_right: np.ndarray

    @staticmethod
    def smooth(x0, x1):
        x = np.array([x0, x1])
        return WallProfile(x, np.zeros(2), np.zeros(2))


@dataclass
class Guide:
    platform: str
    kind: str
    lam_um: float
    structures: list
    background: object
    film_material: str
    film_thickness: float
    top_z: float                      # highest point of the guide
    core_half_width: float            # widest lateral half-extent of the core (bottom of walls)
    sidewalls: list[Segment]
    interface: tuple[str, str]        # (inside, outside) materials at the rough surfaces
    eps_interface: tuple              # diagonal sim-frame eps of (inside, outside) at lam
    core_stack: list[Layer]           # vertical layers through the guide centre
    outside_stack: list[Layer]        # vertical layers beside the guide (lateral leakage reference)
    eps_below: tuple
    eps_above: tuple
    info: dict = field(default_factory=dict)

    @property
    def n_cladding_max(self) -> float:
        return float(np.sqrt(max(max(self.eps_below), max(self.eps_above))))


def _angle_tidy3d(angle_from_plane_deg):
    return np.deg2rad(90.0 - angle_from_plane_deg)


def _rib_polyslab(td, medium, profile: WallProfile, half_width, z_bot, z_top, angle_deg):
    x = np.asarray(profile.x, dtype=float)
    right = np.column_stack([x, -(half_width + profile.h_right)])
    left = np.column_stack([x[::-1], (half_width + profile.h_left)[::-1]])
    verts = np.vstack([right, left])
    return td.Structure(
        geometry=td.PolySlab(vertices=verts.tolist(), axis=2, slab_bounds=(z_bot, z_top),
                             sidewall_angle=float(_angle_tidy3d(angle_deg)), reference_plane="top"),
        medium=medium)


def _walls(half_width_top, z_top, z_bot, angle_deg):
    run = (z_top - z_bot) / np.tan(np.deg2rad(angle_deg))
    hw_bot = half_width_top + run
    return [Segment(half_width_top, z_top, hw_bot, z_bot), Segment(-half_width_top, z_top, -hw_bot, z_bot)], hw_bot


def build_guide(cfg: Cfg, platform: str, lam_um: float, profile: WallProfile | None = None,
                x_extent: tuple[float, float] = (-1e3, 1e3), substrate: str | None = None) -> Guide:
    """Structures of one platform at one wavelength (needed for off-axis anisotropic media)."""
    import tidy3d as td

    p = cfg.platforms[platform]
    above_name = cfg.claddings.above
    below_name = cfg.claddings.below
    above = mat.build(above_name, cfg)
    below = mat.build(below_name, cfg)
    film = mat.build(p.film, cfg)
    rot = rotation_for(p.orientation, cfg) if film.uniaxial else None
    film_med = oriented_medium(film, rot, cfg, lam_um)
    eps_film = tuple(float(v) for v in diagonal_eps(film, rot, lam_um))
    eps_above = tuple(float(v) for v in above.eps_principal(lam_um))
    eps_below = tuple(float(v) for v in below.eps_principal(lam_um))
    t = float(p.film_thickness_um)
    profile = profile or WallProfile.smooth(*x_extent)
    inf = td.inf
    structures = []

    sub_name = substrate or cfg.substrate
    z = 0.0
    sub_layers = list(cfg.substrates[sub_name])
    if not sub_layers and below_name != above_name:
        structures.append(td.Structure(geometry=td.Box.from_bounds((-inf, -inf, -inf), (inf, inf, 0.0)),
                                       medium=below.td_medium()))
    for i, layer in enumerate(sub_layers):
        m = mat.build(layer["material"], cfg)
        z_lo = -inf if layer["thickness_um"] is None else z - float(layer["thickness_um"])
        structures.append(td.Structure(geometry=td.Box.from_bounds((-inf, -inf, z_lo), (inf, inf, z)),
                                       medium=m.td_medium()))
        if i == 0:      # guided/cutoff reference is the layer right under the film (BOX)
            eps_below = tuple(float(v) for v in m.eps_principal(lam_um))
        z = z_lo
    info_sub = sub_name if sub_layers else "membrane"

    kind = p.type
    sidewalls: list[Segment] = []
    if kind == "rib":
        t_etch = float(p.etch_fraction) * t
        t_slab = t - t_etch
        if t_slab > 0:
            structures.append(td.Structure(geometry=td.Box.from_bounds((-inf, -inf, 0.0), (inf, inf, t_slab)),
                                           medium=film_med))
        hw = 0.5 * float(p.top_width_um)
        structures.append(_rib_polyslab(td, film_med, profile, hw, t_slab, t, float(p.sidewall_angle_deg)))
        sidewalls, hw_bot = _walls(hw, t, t_slab, float(p.sidewall_angle_deg))
        interface = (p.film, above_name)
        eps_if = (eps_film, eps_above)
        core = [Layer(eps_film, t)]
        outside = [Layer(eps_film, t_slab)] if t_slab > 0 else []
        top, half = t, hw_bot
        info = {"remaining_slab_um": t_slab, "etch_depth_um": t_etch}
    elif kind == "strip_loaded":
        strip = mat.build(p.strip, cfg)
        eps_strip = tuple(float(v) for v in strip.eps_principal(lam_um))
        structures.append(td.Structure(geometry=td.Box.from_bounds((-inf, -inf, 0.0), (inf, inf, t)),
                                       medium=film_med))
        h = float(p.strip_height_um)
        hw = 0.5 * float(p.strip_width_um)
        structures.append(_rib_polyslab(td, strip.td_medium(), profile, hw, t, t + h, float(p.sidewall_angle_deg)))
        sidewalls, hw_bot = _walls(hw, t + h, t, float(p.sidewall_angle_deg))
        interface = (p.strip, above_name)
        eps_if = (eps_strip, eps_above)
        core = [Layer(eps_film, t), Layer(eps_strip, h)]
        outside = [Layer(eps_film, t)]
        top, half = t + h, hw_bot
        info = {"strip_height_um": h}
    elif kind == "slab":
        structures.append(td.Structure(geometry=td.Box.from_bounds((-inf, -inf, 0.0), (inf, inf, t)),
                                       medium=film_med))
        interface = (p.film, above_name)
        eps_if = (eps_film, eps_above)
        core = [Layer(eps_film, t)]
        outside = core
        top, half = t, 0.0
        info = {}
    else:
        raise ValueError(f"platform {platform}: unknown type {kind!r}")

    info.update({"substrate": info_sub, "orientation": p["orientation"] if "orientation" in p else None,
                 "film_source": film.source})
    return Guide(platform=platform, kind=kind, lam_um=lam_um, structures=structures,
                 background=above.td_medium(), film_material=p.film, film_thickness=t, top_z=top,
                 core_half_width=half, sidewalls=sidewalls, interface=interface, eps_interface=eps_if,
                 core_stack=core, outside_stack=outside, eps_below=eps_below, eps_above=eps_above,
                 info=info)
