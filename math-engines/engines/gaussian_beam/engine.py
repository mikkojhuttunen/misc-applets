"""Gaussian (TEM00) beam propagation and the ABCD law.

Conventions
-----------
* SI units throughout. ``wavelength`` is the vacuum wavelength; the medium
  enters through the refractive index ``n``.
* ``w`` is the 1/e^2 intensity radius. ``M2`` scales the beam quality
  (embedded-Gaussian model: lambda -> M2 * lambda).
* q = (z - z0) + i z_R. ``distance_to_waist`` > 0 means the waist lies
  downstream of the current plane.
* R > 0 for a beam diverging after its waist; R = inf at the waist.
"""
from __future__ import annotations

import numpy as np

from engines.common import Result, require_positive

_ASSUME = ["Paraxial TEM00 (embedded Gaussian for M2 > 1)", "w is the 1/e^2 intensity radius"]


def rayleigh_range(w0, wavelength, n=1.0, M2=1.0):
    """z_R = pi n w0^2 / (M2 lambda) in metres (helper, returns array/float)."""
    return np.pi * n * np.asarray(w0, float) ** 2 / (M2 * np.asarray(wavelength, float))


def _radius_and_roc(z, w0, wavelength, z0=0.0, n=1.0, M2=1.0):
    require_positive("w0", w0)
    require_positive("wavelength", wavelength)
    require_positive("n", n)
    require_positive("M2", M2)
    zr = rayleigh_range(w0, wavelength, n, M2)
    dz = np.asarray(z, float) - z0
    w = w0 * np.sqrt(1 + (dz / zr) ** 2)
    with np.errstate(divide="ignore"):
        roc = np.where(dz == 0, np.inf, dz + zr ** 2 / np.where(dz == 0, 1, dz))
    return zr, dz, w, roc


def beam_parameters(w0, wavelength, n=1.0, M2=1.0) -> Result:
    """Rayleigh range, far-field half-angle divergence and confocal parameter."""
    require_positive("w0", w0)
    require_positive("wavelength", wavelength)
    zr = rayleigh_range(w0, wavelength, n, M2)
    theta = M2 * wavelength / (np.pi * n * w0)
    return Result(
        values={"z_R": zr, "divergence": theta, "confocal_parameter": 2 * zr, "w0": w0},
        units={"z_R": "m", "divergence": "rad", "confocal_parameter": "m", "w0": "m"},
        assumptions=_ASSUME + ["divergence is the far-field half-angle w0/z_R"],
    )


def beam_at(z, w0, wavelength, z0=0.0, n=1.0, M2=1.0) -> Result:
    """Beam radius, wavefront radius of curvature and Gouy phase at position z."""
    zr, dz, w, roc = _radius_and_roc(z, w0, wavelength, z0, n, M2)
    return Result(
        values={"w": w, "R": roc, "gouy_phase": np.arctan(dz / zr), "z_R": zr},
        units={"w": "m", "R": "m", "gouy_phase": "rad", "z_R": "m"},
        assumptions=_ASSUME,
    )


# ---------- complex beam parameter and ABCD matrices ----------
def q_parameter(z, w0, wavelength, z0=0.0, n=1.0, M2=1.0):
    """Complex beam parameter q = (z - z0) + i z_R (helper)."""
    return (np.asarray(z, float) - z0) + 1j * rayleigh_range(w0, wavelength, n, M2)


def free_space(d):
    """ABCD matrix of propagation over a physical length d (q is kept in physical length)."""
    return np.array([[1.0, float(d)], [0.0, 1.0]])


def thin_lens(f):
    """ABCD matrix of a thin lens of focal length f (f > 0 converging)."""
    if f == 0:
        raise ValueError("f must be non-zero")
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def abcd_propagate(q, M):
    """ABCD law q' = (A q + B) / (C q + D)."""
    (A, B), (C, D) = np.asarray(M, float)
    return (A * q + B) / (C * q + D)


def beam_from_q(q, wavelength, n=1.0, M2=1.0) -> Result:
    """Beam radius, curvature, waist size and distance to the waist from q."""
    require_positive("wavelength", wavelength)
    q = np.asarray(q, complex)
    zr = q.imag
    require_positive("Im(q) = z_R", zr)
    dz = q.real
    w0 = np.sqrt(zr * M2 * wavelength / (np.pi * n))
    w = w0 * np.sqrt(1 + (dz / zr) ** 2)
    with np.errstate(divide="ignore"):
        roc = np.where(dz == 0, np.inf, dz + zr ** 2 / np.where(dz == 0, 1, dz))
    return Result(
        values={"w": w, "R": roc, "w0": w0, "z_R": zr, "distance_to_waist": -dz},
        units={"w": "m", "R": "m", "w0": "m", "z_R": "m", "distance_to_waist": "m"},
        assumptions=_ASSUME + ["distance_to_waist > 0: waist downstream"],
    )


def thin_lens_focus(w0, s, f, wavelength, n=1.0, M2=1.0) -> Result:
    """Waist after a thin lens placed a distance s after an input waist w0.

    Uses the ABCD law; the result agrees with Self's formula (see tests).
    """
    require_positive("f", f)
    q_in = q_parameter(s, w0, wavelength, 0.0, n, M2)
    b = beam_from_q(abcd_propagate(q_in, thin_lens(f)), wavelength, n, M2)
    w0n = b["w0"]
    return Result(
        values={"w0_out": w0n, "waist_distance": b["distance_to_waist"], "z_R_out": b["z_R"],
                "magnification": w0n / w0},
        units={"w0_out": "m", "waist_distance": "m", "z_R_out": "m", "magnification": "1"},
        assumptions=_ASSUME + ["Thin lens, no aberrations", "waist_distance measured from the lens, > 0 downstream"],
    )
