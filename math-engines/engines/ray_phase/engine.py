"""Optical phase of rays launched at slightly different angles into a multipass cell, and the fringe visibility
left when the launch angle is dithered.

A ray leaves the input point at angle θ from the inward normal and is traced through the cell (billiard_cell
geometry). After p mirror hits its geometric path is L_p(θ); its optical phase relative to the centre ray θc is

    Δφ_p(θ) = k0 n (L_p(θ) - L_p(θc)),      k0 = 2π / λ,  n = index of the medium (slab n_eff for a membrane)

Reflection phases are common to all rays with the same facet sequence and are ignored (rays that change facet
sequence pick up the same number of reflections per pass, so only the path term differs).

Angle dithering θ(t) = θc + δ(t) makes the phase of pass-p light wander by Δφ_p(θc + δ). Any interference term
between that light and light whose phase does not follow the dither (a reference, or a pass with a very
different dφ/dθ) is averaged over one dither period to

    V_p = | ⟨ exp(i Δφ_p(θc + δ)) ⟩_t |           (fringe visibility left; 1 = untouched, 0 = washed out)

For a linear phase Δφ = a δ: sine dither δ = A sin ωt gives V = |J0(a A)|, triangle dither (δ uniform in ±A)
gives V = |sin(a A)/(a A)|. V² is the coherence (Strehl-like) factor of an angular fan of rays with those
weights. Ray picture only: rays at different angles end at different points after p passes, so V_p describes
the phase stability of each ray's own path, not the overlap of neighbouring rays. SI units.
"""
from __future__ import annotations

import numpy as np

from ..billiard_cell.engine import SegmentedCell, _reflect
from ..common import Result, require_choice, require_positive, require_range

_WAVEFORMS = ("sine", "triangle")


def launch(cell, theta, s_in=None):
    """Start points and unit directions for rays leaving boundary coordinate s_in at angles theta (array)."""
    s_in = cell.s_in_default if s_in is None else s_in
    th = np.atleast_1d(np.asarray(theta, float))
    x, y, nx, ny = cell.point_at(np.full(th.shape, s_in))
    ux, uy = -nx, -ny
    return x, y, ux * np.cos(th) - uy * np.sin(th), ux * np.sin(th) + uy * np.cos(th)


def path_lengths(cell, theta, n_pass, s_in=None):
    """Cumulative geometric path [m] to each of the first n_pass mirror hits: array (len(theta), n_pass).
    Rays that leak through a corner gap get NaN from that hit on."""
    x, y, dx, dy = launch(cell, theta, s_in)
    L = np.full((x.size, int(n_pass)), np.nan)
    acc = np.zeros(x.size)
    for j in range(int(n_pass)):
        t, nx, ny, _ = cell.hit(x, y, dx, dy)
        ok = np.isfinite(t)
        acc = np.where(ok, acc + np.where(ok, t, 0.0), np.nan)
        L[:, j] = acc
        t = np.where(ok, t, 0.0)
        x, y = x + t * dx, y + t * dy
        dx, dy = _reflect(dx, dy, np.where(ok, nx, 1.0), np.where(ok, ny, 0.0))
    return L


def relative_phase(L, L_ref, wavelength, n_index=1.0):
    """Δφ = 2π n (L - L_ref) / λ [rad]."""
    return 2 * np.pi * n_index * (np.asarray(L, float) - L_ref) / wavelength


def dither_offsets(amplitude, n=401, waveform="sine"):
    """Angle offsets δ sampled uniformly in time over one dither period: A sin(2π t) or a triangle wave (δ
    uniform in [-A, A]). Equal weights; the midpoint rule makes the sine case exact for band-limited phases."""
    require_choice("waveform", waveform, _WAVEFORMS)
    t = (np.arange(n) + 0.5) / n
    return amplitude * np.sin(2 * np.pi * t) if waveform == "sine" else amplitude * (2 * t - 1)


def visibility(dphi, weights=None):
    """|Σ w exp(iΔφ)| / Σ w over the samples (last axis), ignoring NaN samples."""
    dphi = np.asarray(dphi, float)
    w = np.ones(dphi.shape[-1]) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(dphi)
    W = np.where(ok, w, 0.0)
    z = (W * np.exp(1j * np.where(ok, dphi, 0.0))).sum(-1)
    return np.abs(z) / W.sum(-1)


def dither_scan(cell, theta_c, amplitude, n_pass, wavelength, n_index=1.0, waveform="sine", n_samples=401, s_in=None):
    """Phases and visibilities of passes 1..n_pass under angle dither. Returns dict with
    offsets δ, L (samples × passes), dphi (samples × passes, relative to the undithered ray),
    V (per pass), max_step (largest phase jump between neighbouring samples per pass; > π means the sampling
    no longer resolves the phase and V is only an upper-bound-like estimate)."""
    d = dither_offsets(amplitude, n_samples, waveform)
    L0 = path_lengths(cell, [theta_c], n_pass, s_in)[0]
    L = path_lengths(cell, theta_c + d, n_pass, s_in)
    dphi = relative_phase(L, L0[None, :], wavelength, n_index)
    order = np.argsort(d, kind="stable")
    steps = np.abs(np.diff(dphi[order], axis=0))
    max_step = np.nanmax(steps, axis=0) if len(d) > 1 else np.zeros(int(n_pass))
    return dict(offsets=d, L=L, L0=L0, dphi=dphi, V=visibility(dphi.T), max_step=max_step)


def _cell(radius, n_facets, curvature):
    require_positive(radius=radius)
    n = int(n_facets)
    if n < 3:
        raise ValueError("n_facets must be >= 3")
    return SegmentedCell(radius, n, curvature=curvature)


def first_mirror_phase(radius=5e-3, n_facets=24, theta_c=0.6545, wavelength=1.55e-6, n_index=1.0, curvature=0.0) -> Result:
    """Path from the input point to the first mirror and how fast its phase turns with launch angle."""
    require_positive(wavelength=wavelength, n_index=n_index)
    require_range("theta_c", theta_c, -1.55, 1.55)
    cell = _cell(radius, n_facets, curvature)
    h = 1e-6
    L = path_lengths(cell, [theta_c - h, theta_c, theta_c + h], 1)[:, 0]
    dL = (L[2] - L[0]) / (2 * h)
    dphi = 2 * np.pi * n_index * dL / wavelength
    return Result(
        values={"L1": L[1], "dL_dtheta": dL, "dphi_dtheta": dphi, "dtheta_2pi": abs(2 * np.pi / dphi) if dphi else np.inf},
        units={"L1": "m", "dL_dtheta": "m/rad", "dphi_dtheta": "rad/rad", "dtheta_2pi": "rad"},
        assumptions=["Ray optics in a regular segmented cell", "Phase = k0 n L, reflection phases ignored",
                     "Derivative by central difference (1 µrad)"],
    )


def dither_visibility(radius=5e-3, n_facets=24, theta_c=0.6545, wavelength=1.55e-6, n_index=1.0, amplitude=1e-4,
                      n_pass=24, waveform="sine", curvature=0.0) -> Result:
    """Fringe visibility left at pass 1 and pass n_pass when the launch angle is dithered with amplitude A."""
    require_positive(wavelength=wavelength, n_index=n_index, amplitude=amplitude)
    require_range("theta_c", theta_c, -1.55, 1.55)
    require_choice("waveform", waveform, _WAVEFORMS)
    n_pass = int(n_pass)
    if n_pass < 1:
        raise ValueError("n_pass must be >= 1")
    r = dither_scan(_cell(radius, n_facets, curvature), theta_c, amplitude, n_pass, wavelength, n_index, waveform)
    resolved = bool(r["max_step"][-1] <= np.pi)
    return Result(
        values={"V_first": float(r["V"][0]), "V_last": float(r["V"][-1]), "phase_span_last": float(np.nanmax(r["dphi"][:, -1]) - np.nanmin(r["dphi"][:, -1])),
                "L_last": float(r["L0"][-1]), "resolved": resolved},
        units={"V_first": "", "V_last": "", "phase_span_last": "rad", "L_last": "m", "resolved": ""},
        assumptions=["Ray optics, reflection phases ignored", "V = |time average of exp(iΔφ)| over one dither period",
                     "resolved = False: phase changes by more than π between dither samples, V is not reliable"],
    )
