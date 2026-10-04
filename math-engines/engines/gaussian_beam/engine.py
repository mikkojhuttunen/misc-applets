"""Gaussian beam (TEM00) propagation and the ABCD law.

Conventions: SI units; wavelength is the vacuum wavelength, the medium enters
through n; w is the 1/e^2 intensity radius; z - z0 > 0 is downstream of the
waist; R > 0 for a beam diverging in +z; M2 >= 1 scales the far-field spread.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_positive

_ASSUME = ["Paraxial TEM00", "w is the 1/e^2 intensity radius"]


def _radius_and_roc(z, w0, wavelength, z0, n, M2):
    require_positive(w0=w0, wavelength=wavelength, n=n, M2=M2)
    zr = np.pi * w0**2 * n / (M2 * wavelength)
    dz = np.asarray(z, dtype=float) - z0
    w = w0 * np.sqrt(1 + (dz / zr) ** 2)
    with np.errstate(divide="ignore"):
        roc = np.where(dz == 0, np.inf, dz * (1 + (zr / np.where(dz == 0, 1, dz)) ** 2))
    return zr, dz, w, roc


def beam_parameters(w0, wavelength, n=1.0, M2=1.0) -> Result:
    """Rayleigh range, divergence half-angle and confocal parameter of a waist w0."""
    require_positive(w0=w0, wavelength=wavelength, n=n, M2=M2)
    zr = np.pi * w0**2 * n / (M2 * wavelength)
    theta = M2 * wavelength / (np.pi * n * w0)
    return Result(
        values={"z_R": zr, "theta": theta, "b": 2 * zr},
        units={"z_R": "m", "theta": "rad", "b": "m"},
        assumptions=_ASSUME + ["theta is the far-field half-angle in the medium"],
    )


def beam_at(z, w0, wavelength, z0=0.0, n=1.0, M2=1.0) -> Result:
    """Beam radius, wavefront radius of curvature and Gouy phase at position z."""
    zr, dz, w, roc = _radius_and_roc(z, w0, wavelength, z0, n, M2)
    return Result(
        values={"w": w, "R": roc, "gouy_phase": np.arctan(dz / zr), "z_R": zr},
        units={"w": "m", "R": "m", "gouy_phase": "rad", "z_R": "m"},
        assumptions=list(_ASSUME),
    )


def q_parameter(z, w0, wavelength, z0=0.0, n=1.0):
    """Complex beam parameter q = (z - z0) + i z_R (helper, returns complex)."""
    require_positive(w0=w0, wavelength=wavelength, n=n)
    zr = np.pi * w0**2 * n / wavelength
    return (np.asarray(z, dtype=float) - z0) + 1j * zr


def free_space(d, n=1.0):
    """ABCD matrix of a distance d (helper). With q in the medium, use d as is."""
    return np.array([[1.0, float(d)], [0.0, 1.0]])


def thin_lens(f):
    """ABCD matrix of a thin lens of focal length f (f > 0 converging; helper)."""
    if f == 0:
        raise ValueError("focal length f must be non-zero")
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def abcd_propagate(q, M):
    """ABCD law q' = (A q + B) / (C q + D) (helper)."""
    (A, B), (C, D) = np.asarray(M, dtype=float)
    return (A * q + B) / (C * q + D)


def beam_from_q(q, wavelength, n=1.0) -> Result:
    """Beam radius, curvature and waist position from a complex beam parameter."""
    require_positive(wavelength=wavelength, n=n)
    inv = 1 / np.asarray(q, dtype=complex)
    w = np.sqrt(-wavelength / (np.pi * n * np.imag(inv)))
    with np.errstate(divide="ignore"):
        roc = np.where(np.real(inv) == 0, np.inf, 1 / np.where(np.real(inv) == 0, 1, np.real(inv)))
    zr = np.imag(q)
    return Result(
        values={"w": w, "R": roc, "z_R": zr, "w0": np.sqrt(zr * wavelength / (np.pi * n)), "distance_to_waist": -np.real(q)},
        units={"w": "m", "R": "m", "z_R": "m", "w0": "m", "distance_to_waist": "m"},
        assumptions=_ASSUME + ["distance_to_waist > 0 means the waist lies downstream"],
    )


def thin_lens_focus(w0, s, f, wavelength, n=1.0) -> Result:
    """Waist produced by a thin lens of focal length f placed s after an input waist w0."""
    require_positive(w0=w0, s=s, wavelength=wavelength, n=n)
    q = abcd_propagate(q_parameter(s, w0, wavelength, 0.0, n), thin_lens(f))
    out = beam_from_q(q, wavelength, n)
    return Result(
        values={"w0_out": out["w0"], "z_out": out["distance_to_waist"], "w_at_lens": out["w"]},
        units={"w0_out": "m", "z_out": "m", "w_at_lens": "m"},
        assumptions=_ASSUME + ["Thin lens", "z_out is measured from the lens, > 0 downstream"],
    )
