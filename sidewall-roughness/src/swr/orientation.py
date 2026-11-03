"""Crystal orientation: the single place where crystal axes are mapped to the simulation frame.

Simulation frame: x = propagation direction, y = lateral (in the film plane), z = film normal.
Crystal frame: principal axes a, b, c (= crystal X, Y, Z); optic axis of uniaxial media is c.

An orientation is (cut, propagation angle):
* cut: the crystal axis along the film normal (sim z): "x", "y", "z" or a 3-vector,
* reference_axis: a crystal axis in the film plane (from config orientation_defaults),
* angle theta: propagation direction = cos(theta) * reference + sin(theta) * (cut x reference).

The same function serves a future multipass (Herriott-type) cell where theta changes from
pass to pass: call it once per propagation direction.
"""

from __future__ import annotations

import numpy as np

from .config import Cfg
from .materials import MaterialModel, freq

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def _vec(axis):
    v = AXES[axis] if isinstance(axis, str) else np.asarray(axis, dtype=float)
    return v / np.linalg.norm(v)


def crystal_to_sim(cut, propagation_angle_deg: float, reference_axis) -> np.ndarray:
    """Rotation matrix R with rows = sim x, y, z unit vectors in crystal coordinates.

    A crystal-frame tensor eps_c becomes eps_sim = R @ eps_c @ R.T.
    """
    n = _vec(cut)
    u = _vec(reference_axis)
    u = u - np.dot(u, n) * n
    if np.linalg.norm(u) < 1e-12:
        raise ValueError("reference_axis must not be parallel to the cut axis")
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    th = np.deg2rad(propagation_angle_deg)
    x = np.cos(th) * u + np.sin(th) * v
    y = np.cross(n, x)
    return np.vstack([x, y, n])


def rotate_tensor(eps_principal, rotation: np.ndarray) -> np.ndarray:
    """eps_principal: (..., 3) diagonal entries in the crystal frame -> (..., 3, 3) sim-frame tensor."""
    eps = np.asarray(eps_principal)
    diag = eps[..., :, None] * np.eye(3)
    return rotation @ diag @ rotation.T


def rotation_for(orientation_name: str, cfg: Cfg) -> np.ndarray:
    o = cfg.orientations[orientation_name]
    ref = cfg.orientation_defaults[o.cut].reference_axis if "reference_axis" not in o else o.reference_axis
    return crystal_to_sim(o.cut, o.propagation_angle_deg, ref)


def is_axis_permutation(rotation: np.ndarray, tol: float) -> bool:
    return bool(np.all(np.minimum(np.abs(rotation), np.abs(np.abs(rotation) - 1.0)) < tol))


def sim_eps_tensor(material: MaterialModel, rotation: np.ndarray, lam_um) -> np.ndarray:
    return rotate_tensor(material.eps_principal(lam_um), rotation)


def oriented_medium(material: MaterialModel, rotation: np.ndarray | None, cfg: Cfg,
                    lam_um: float | None = None):
    """Tidy3D medium for `material` in the sim frame.

    Isotropic -> its own (dispersive) medium. Uniaxial with the crystal axes along the sim axes
    -> dispersive td.AnisotropicMedium. Any other orientation -> td.FullyAnisotropicMedium,
    which Tidy3D only supports non-dispersive, so it is evaluated at `lam_um` and must be
    used with a narrowband source (README, limitations).
    """
    import tidy3d as td

    if not material.uniaxial:
        return material.td_medium()
    if rotation is None:
        raise ValueError(f"{material.name} is uniaxial: an orientation is required")
    tol = cfg.orientation_tol
    if is_axis_permutation(rotation, tol):
        comps = []
        for i in range(3):
            j = int(np.argmax(np.abs(rotation[i])))
            comps.append(material.td_e if j == 2 else material.td_o)
        return td.AnisotropicMedium(xx=comps[0], yy=comps[1], zz=comps[2], name=material.name)
    if lam_um is None:
        raise ValueError("off-axis orientation gives a non-dispersive medium: pass lam_um")
    eps = sim_eps_tensor(material, rotation, lam_um)
    eps = 0.5 * (eps + eps.T)
    return td.FullyAnisotropicMedium(permittivity=eps.tolist(), name=f"{material.name}@{lam_um}um")


def diagonal_eps(material: MaterialModel, rotation: np.ndarray | None, lam_um) -> np.ndarray:
    """(eps_xx, eps_yy, eps_zz) in the sim frame; off-diagonal parts are dropped (slab solver)."""
    eps_p = material.eps_principal(lam_um)
    if not material.uniaxial or rotation is None:
        return eps_p
    return np.diagonal(rotate_tensor(eps_p, rotation), axis1=-2, axis2=-1)


__all__ = ["crystal_to_sim", "rotate_tensor", "rotation_for", "oriented_medium", "diagonal_eps",
           "sim_eps_tensor", "is_axis_permutation", "freq"]
