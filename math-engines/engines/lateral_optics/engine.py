"""Lateral (in-plane) optics of slab-waveguide gratings: ridge effective index with lateral confinement, ridge-mode
overlap with a Gaussian beam, and the round-trip efficiency of a lateral Gaussian beam bouncing off a curved or finite
DBR. SI units; y is the lateral coordinate, the beam propagates along z in a medium of effective index n.

Python reference for ridgeRef, neffFull (ridge), ridgeLateral, gaussOverlap, bounce and bestBounce in
dbr-structures/dbr-engine.js.
"""
from __future__ import annotations

from math import erf

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive
from ..grating_coupler.engine import slab_mode
from ..slab_waveguide.engine import neff_three_layer

_TAU = 2 * np.pi


def _flip(pol):
    """Polarisation of the lateral effective-index step: a vertical TE mode has E across the ridge (lateral TM)."""
    return "TM" if pol == "TE" else "TE"


def side_index(wavelength, n_sub, n_core, n_clad, side_thickness, polarization="TE"):
    """Effective index of the side regions: the slab of thickness side_thickness (rib), or max(n_sub, n_clad) for a
    strip or below cut-off."""
    floor = max(n_sub, n_clad)
    if not side_thickness > 1e-12:
        return floor
    v = neff_three_layer(wavelength, n_sub, n_core, n_clad, side_thickness, polarization, 0)
    return v if np.isfinite(v) else floor


def ridge_reference(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness, polarization="TE", order=0):
    """Lateral effective-index step of a ridge of width W on a slab of thickness t (sides: side_thickness).

    Returns the vertical slab index v_t, the side index n_O, the ridge index N_t (lateral mode of the symmetric
    n_O | v_t | n_O slab) and the lateral confinement Γ = ∫_ridge φ² / ∫ φ²; None if the ridge is laterally cut off.
    Grating slices of local thickness d then get N(d) = N_t + Γ (v(d) - v_t) (see ridge_neff)."""
    vt = neff_three_layer(wavelength, n_sub, n_core, n_clad, thickness, polarization, order)
    nO = side_index(wavelength, n_sub, n_core, n_clad, side_thickness, polarization)
    if not (np.isfinite(vt) and vt > nO):
        return None
    m = slab_mode(wavelength, nO, vt, nO, width, _flip(polarization), 0)
    if m is None:
        return None
    rs = (vt / nO) ** 2 if _flip(polarization) == "TM" else 1.0
    ps = np.arctan2(rs * m["gs"], m["kx"])                       # phase of cos(kx y - ps) in the ridge
    kx = m["kx"]
    core = width / 2 + (np.sin(2 * (kx * width - ps)) + np.sin(2 * ps)) / (4 * kx)
    return {"v_t": vt, "n_side": nO, "N_t": m["neff"], "gamma": float(min(1.0, max(0.0, core / m["norm"]))), "mode": m}


def ridge_neff(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness, local_thickness, polarization="TE",
               order=0):
    """Effective index of a ridge section whose core is locally local_thickness thick (an etched grating slice)."""
    ref = ridge_reference(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness, polarization, order)
    if ref is None:
        return np.nan
    v = neff_three_layer(wavelength, n_sub, n_core, n_clad, local_thickness, polarization, order)
    if not np.isfinite(v):
        v = max(n_sub, n_clad)
    return ref["N_t"] + ref["gamma"] * (v - ref["v_t"])


def ridge_beam_overlap(wavelength, n_side, n_ridge, width, w0, polarization="TE", samples=800):
    """Lateral ridge mode (slab n_side | n_ridge | n_side of width W, lateral polarisation flipped) against a Gaussian
    exp(-y²/w0²): power overlap Oy² = (∫φ g)² / (∫g² ∫φ²) and the SHG width 1/∫φ⁴ (φ normalised, ∫φ² = 1).
    Trapezoid on `samples` intervals over |y| ≤ W/2 + max(3/γ, 3 w0) (the JS port's grid)."""
    m = slab_mode(wavelength, n_side, n_ridge, n_side, width, _flip(polarization), 0)
    if m is None:
        return None
    ext = width / 2 + max(3 / m["gs"], 3 * w0)
    y = np.linspace(-ext, ext, samples + 1)
    w = np.full(y.size, 2 * ext / samples)
    w[0] = w[-1] = ext / samples
    phi = m["field"](y + width / 2) / np.sqrt(m["norm"])
    g = np.exp(-((y / w0) ** 2))
    ov, g2, p4 = np.sum(w * phi * g), np.sum(w * g * g), np.sum(w * phi**4)
    return {"Oy2": float(ov * ov / g2), "w_eff": float(1 / p4), "neff": m["neff"]}


# ---------------------------------------------------------------- lateral Gaussian beams and curved DBRs
def gauss_overlap(q1, q2, k):
    """Power overlap of two 1D Gaussian beams u = exp(-i k y² / (2 q)) at the same plane (k in the medium)."""
    A1, A2 = 1j * k / (2 * q1), 1j * k / (2 * q2)                # u = exp(-A y²)
    return float(2 * np.sqrt(A1.real * A2.real) / abs(A1 + np.conj(A2)))


def bounce(wavelength_medium, w, d, Rc, aperture_half=np.inf):
    """Round trip of a lateral Gaussian (waist w at the launch plane, flat phase) over a distance d to a mirror of
    radius Rc (concave > 0, inf = flat) with aperture half-width a, and back: power coupled back into the launched
    mode, eta = overlap × clip, clip = erf(√2 a / w_m) with w_m the beam radius at the mirror."""
    lam = wavelength_medium
    k = _TAU / lam
    zR = np.pi * w * w / lam
    q0, qm = 1j * zR, d + 1j * zR
    inv = 1 / qm
    wm = np.sqrt(lam / (np.pi * max(-inv.imag, 1e-30)))
    iq2 = inv - (2 / Rc if np.isfinite(Rc) else 0.0)
    qb = 1 / iq2 + d
    ov = gauss_overlap(q0, qb, k)
    clip = erf(np.sqrt(2) * aperture_half / wm) if np.isfinite(aperture_half) else 1.0
    return {"eta": ov * clip, "overlap": ov, "clip": clip, "w_mirror": float(wm)}


def best_bounce(wavelength_medium, d, Rc, aperture_half=np.inf):
    """Launch waist maximising the bounce efficiency (the resonator eigenmode for a stable half-cavity d, Rc):
    log grid of 121 waists, then 40 ternary steps (as the JS port)."""
    lam = wavelength_medium
    wlo = max(0.3 * lam, 0.05e-6)
    whi = max(2 * aperture_half if np.isfinite(aperture_half) else 50e-6, 2 * np.sqrt(lam * d) + 1e-6)
    best = None
    for i in range(121):
        w = wlo * (whi / wlo) ** (i / 120)
        b = bounce(lam, w, d, Rc, aperture_half)
        if best is None or b["eta"] > best[1]["eta"]:
            best = (w, b)
    lo, hi = best[0] / 1.1, best[0] * 1.1
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if bounce(lam, m1, d, Rc, aperture_half)["eta"] > bounce(lam, m2, d, Rc, aperture_half)["eta"]:
            hi = m2
        else:
            lo = m1
    w = (lo + hi) / 2
    return {"w": w, **bounce(lam, w, d, Rc, aperture_half)}


# ---------------------------------------------------------------- Result front ends (spec.yaml)
def ridge_effective_index(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness=0.0, etch_depth=0.0,
                          polarization="TE", beam_waist=2e-6) -> Result:
    """Ridge (rib, strip) effective index by the effective-index method, its lateral confinement, the index of a
    grating slice etched by etch_depth, the overlap of the lateral mode with a Gaussian beam and the SHG width."""
    require_positive(wavelength=wavelength, thickness=thickness, width=width, beam_waist=beam_waist)
    require_nonnegative(side_thickness=side_thickness, etch_depth=etch_depth)
    require_choice("polarization", polarization, ("TE", "TM"))
    ref = ridge_reference(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness, polarization)
    if ref is None:
        raise ValueError("the ridge is laterally cut off")
    ov = ridge_beam_overlap(wavelength, ref["n_side"], ref["v_t"], width, beam_waist, polarization)
    return Result(
        values={"n_slab": ref["v_t"], "n_side": ref["n_side"], "n_ridge": ref["N_t"], "gamma_lateral": ref["gamma"],
                "n_etched": ridge_neff(wavelength, n_sub, n_core, n_clad, thickness, width, side_thickness,
                                       thickness - etch_depth, polarization),
                "beam_overlap": ov["Oy2"], "shg_width": ov["w_eff"]},
        units={"n_slab": "", "n_side": "", "n_ridge": "", "gamma_lateral": "", "n_etched": "", "beam_overlap": "", "shg_width": "m"},
        assumptions=["Effective-index method in two steps (vertical slab, then lateral slab)",
                     "Etched slices as perturbations weighted by the lateral confinement Γ",
                     "Gaussian beam exp(-y²/w0²) centred on the ridge"],
    )


def curved_dbr_bounce(wavelength, n_eff, distance, radius, aperture=np.inf) -> Result:
    """Best lateral waist and round-trip efficiency of a Gaussian bouncing off a curved (or flat) DBR at `distance`
    (half-cavity: launch plane = cavity centre), aperture = full DBR width. radius = inf for a flat DBR."""
    require_positive(wavelength=wavelength, n_eff=n_eff, distance=distance)
    lam = wavelength / n_eff
    a = aperture / 2 if np.isfinite(aperture) else np.inf
    b = best_bounce(lam, distance, radius, a)
    g = 1 - 2 * distance / radius if np.isfinite(radius) else 1.0
    return Result(
        values={"waist": b["w"], "w_mirror": b["w_mirror"], "eta": b["eta"], "overlap": b["overlap"], "clip": b["clip"],
                "g": g},
        units={"waist": "m", "w_mirror": "m", "eta": "", "overlap": "", "clip": "", "g": ""},
        assumptions=["Paraxial 1D Gaussian beam in a medium of index n_eff", "Mirror: thin curved reflector at the DBR penetration plane",
                     "g = 1 - L/R_c for the full cavity L = 2 distance"],
    )
