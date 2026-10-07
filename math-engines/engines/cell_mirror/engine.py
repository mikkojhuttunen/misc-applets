"""Mirror reflectance actually experienced by the light in a multipass cell, from the cell's own angle-of-incidence
distribution and an angle-dependent mirror (etched-trench DBR, bragg_grating.TrenchDBR).

Rays are launched from the input point at θc + fan offsets and traced through a billiard_cell.SegmentedCell (facet
tilts, curvatures and offsets included). At hit j a ray meets the mirror at |sin χ_j| and keeps R(χ_j) of its power,
so its intensity arriving at hit j is I_j = Π_{i<j} R(χ_i) (I_0 = 1) and chord j (which ends at hit j) carries I_j.
From that, self-consistently (no assumed reflectance):

    R_mean   = Σ I_j R(χ_j) / Σ I_j          power-weighted mean reflectance over all hits of all rays
    I_end    = mean over rays of Π_j R(χ_j)   power left after the traced hits
    R_eff    = I_end^(1/n̄)                   per-bounce reflectance of the total power (n̄ = mean hits per ray).
               Not the geometric mean of R over hits: that weights hits of rays already killed (e.g. in a
               Brewster dip) as much as live ones and overstates the loss.
    L_eff    = mean over rays of Σ_j I_j ℓ_j  absorption-weighted path (a weak absorber α removes α L_eff)
    |χ| distribution weighted by I_j: histogram and percentiles (median, 95 %, max)

References for comparison: R at the DBR design angle, and the uniform-sin χ average (the invariant measure of a
fully chaotic billiard, i.e. what an ergodic cell would see). Comparing a perturbed cell with the same launch in the
unperturbed cell shows what the perturbation costs in mirror reflectance and in path length.

R(χ) is evaluated exactly (transfer matrix) at every hit: a table interpolated in |sin χ| is off by up to 0.15 in R
next to the total-internal-reflection edge of a p-polarised mirror, where R rises like a square root. The uniform
average uses the exact R on N_TABLE points. SI units; angles in rad except the chi_* outputs (degrees).
"""
from __future__ import annotations

import numpy as np

from ..billiard_cell.engine import SegmentedCell, _reflect
from ..bragg_grating.engine import TrenchDBR
from ..common import Result, require_choice, require_nonnegative, require_positive, require_range
from ..ray_phase.engine import launch

N_TABLE = 4097


def hit_angles(cell, theta, n_hits, s_in=None):
    """|sin χ| at each mirror hit and the chord ending there: two arrays (n_rays, n_hits); NaN after a leak."""
    x, y, dx, dy = launch(cell, theta, s_in)
    n_hits = int(n_hits)
    S = np.full((x.size, n_hits), np.nan)
    C = np.full((x.size, n_hits), np.nan)
    alive = np.ones(x.size, bool)
    for j in range(n_hits):
        t, nx, ny, _ = cell.hit(x, y, dx, dy)
        ok = alive & np.isfinite(t)
        S[:, j] = np.where(ok, np.abs(dx * ny - dy * nx), np.nan)
        C[:, j] = np.where(ok, t, np.nan)
        t = np.where(ok, t, 0.0)
        x, y = x + t * dx, y + t * dy
        dx, dy = _reflect(dx, dy, np.where(ok, nx, 1.0), np.where(ok, ny, 0.0))
        alive = ok
    return S, C


def uniform_average(R_of_sinchi, n=N_TABLE):
    """⟨R⟩ over sin χ uniform in [0, 1] (fully chaotic cell), trapezoid rule on n points."""
    sg = np.linspace(0.0, 1.0, n)
    R = np.asarray(R_of_sinchi(sg), float)
    return float(((R[1:] + R[:-1]) * 0.5 * np.diff(sg)).sum())


def weighted_quantile(values, weights, q):
    """Smallest value v with (weight of values <= v) >= q × total weight."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    k = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(v[o][min(k, v.size - 1)])


def weighted_stats(S, C, R_of_sinchi, bin_deg=1.0):
    """Self-consistent power-weighted mirror statistics of traced rays (module docstring).
    R_of_sinchi(|sin χ| array) -> power reflectance, evaluated at every hit."""
    valid = np.isfinite(S)
    Rv = np.asarray(R_of_sinchi(np.where(valid, S, 0.0)), float)
    lr = np.where(valid, np.log(np.clip(Rv, 1e-300, 1.0)), 0.0)
    cum = np.cumsum(lr, axis=1)
    I = np.exp(cum - lr)                                   # intensity arriving at each hit
    w = np.where(valid, I, 0.0)
    Rhit = np.exp(lr)
    n_hits = valid.sum(1)
    chi = np.degrees(np.arcsin(np.clip(np.where(valid, S, 0.0), 0, 1)))
    edges = np.arange(0.0, 90.0 + bin_deg, bin_deg)
    hist = np.histogram(chi[valid], bins=edges, weights=w[valid])[0]
    W = w.sum()
    return dict(
        R_mean=float((w * Rhit).sum() / W),
        I_end=float(np.exp(cum[:, -1]).mean()) if S.shape[1] else 1.0,
        L_eff=float(np.where(valid, I * C, 0.0).sum(1).mean()), L_geom=float(np.where(valid, C, 0.0).sum(1).mean()),
        chi_50=weighted_quantile(chi[valid], w[valid], 0.5), chi_95=weighted_quantile(chi[valid], w[valid], 0.95),
        chi_max=float(chi[valid].max()), hist=hist / W, edges=edges, n_hits=int(n_hits.sum()),
        R_eff=float(np.exp(cum[:, -1]).mean() ** (S.shape[0] / max(n_hits.sum(), 1))) if S.shape[1] else 1.0)


def launch_angles(theta_c, fan, n_beams):
    n = int(n_beams)
    return np.array([theta_c]) if n == 1 else theta_c + fan * (np.arange(n) / (n - 1) - 0.5)


def cell_mirror_stats(cell, theta_c, fan, n_beams, n_hits, dbr: TrenchDBR, wavelength=None, s_in=None):
    """weighted_stats for beams fanned around θc in `cell`, with the DBR's R(χ) at `wavelength` (default: design)."""
    lam = dbr.lam_design if wavelength is None else wavelength
    Rf = lambda s: dbr.R(lam, s)
    S, C = hit_angles(cell, launch_angles(theta_c, fan, n_beams), n_hits, s_in)
    out = weighted_stats(S, C, Rf)
    out.update(R_design=float(dbr.R(lam, dbr.sin_design)), R_uniform=uniform_average(Rf))
    return out


def cell_mirror_reflectance(radius=5e-3, n_facets=24, theta_c=0.6545, fan=0.01745, n_beams=7, n_hits=150,
                            tilt_rms=0.0, curvature=0.0, seed=1, n_eff=2.479, polarization="TM", periods=6,
                            m_gap=1, m_tooth=1, sin_design=0.0, wavelength=1.55e-6, bounce_loss=0.0) -> Result:
    """Mirror reflectance averaged over the angles the light meets in a segmented cell, for an etched-trench DBR,
    with the same launch in the unperturbed cell for comparison."""
    require_positive(radius=radius, n_eff=n_eff, wavelength=wavelength)
    require_nonnegative(fan=fan, tilt_rms=tilt_rms)
    require_range("theta_c", theta_c, -1.55, 1.55)
    require_range("bounce_loss", bounce_loss, 0.0, 1.0)
    require_choice("polarization", polarization, ("TE", "TM"))
    if int(n_facets) < 3 or int(n_beams) < 1 or int(n_hits) < 1 or int(periods) < 1:
        raise ValueError("n_facets >= 3, n_beams >= 1, n_hits >= 1 and periods >= 1 required")
    dbr = TrenchDBR(n_tooth=n_eff, lam_design=wavelength, N=int(periods), m_gap=int(m_gap), m_tooth=int(m_tooth),
                    bounce_loss=bounce_loss, slab_pol=polarization, sin_design=sin_design)
    pert = SegmentedCell(radius, int(n_facets), tilt_rms=tilt_rms, curvature=curvature, seed=int(seed))
    reg = SegmentedCell(radius, int(n_facets), curvature=curvature)
    a = cell_mirror_stats(pert, theta_c, fan, n_beams, n_hits, dbr)
    b = cell_mirror_stats(reg, theta_c, fan, n_beams, n_hits, dbr)
    keys = ("R_mean", "R_eff", "I_end", "L_eff", "chi_50", "chi_95", "chi_max")
    vals = {k: a[k] for k in keys}
    vals.update({f"{k}_regular": b[k] for k in ("R_mean", "R_eff", "L_eff", "chi_95")})
    vals.update(R_design=a["R_design"], R_uniform=a["R_uniform"], loss_per_bounce=1 - a["R_eff"])
    units = {k: "" for k in vals}
    units.update(L_eff="m", L_eff_regular="m", chi_50="deg", chi_95="deg", chi_max="deg", chi_95_regular="deg")
    return Result(values=vals, units=units, assumptions=[
        "2D ray optics in a segmented cell; mirror = 1D effective-index trench DBR (bragg_grating.TrenchDBR)",
        "Slab TE -> p, slab TM -> s on the trench walls", "Self-consistent weights: each ray loses R(χ) at every hit",
        "Exact transfer-matrix R(χ) at every hit", "Closed cell (no ports): n_hits reflections per ray",
        "_regular: same launch, unperturbed cell"])
