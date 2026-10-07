"""Ray tracing in planar cells (gmpc.planar.Cell2D): rays from an input port, specular reflection, exits through the
input or output port window, and the CMPC simulator's path statistics.

Same semantics as math-engines billiard_cell (trace_rays / evaluate), generalised to any element wall:
    RayTable    geometry only (chord, signed sin χ and element index per hit, exits), reusable for any mirror
    evaluate    re-weight a table for a mirror reflectance R(|sin χ|), background loss and evanescent factor Γ
    poincare    boundary phase-space samples (s, sin χ)
    twin_divergence   separation of two rays launched δθ apart: exponential growth = chaos (Lyapunov exponent)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stats import lyapunov_fit


def reflect(dx, dy, nx, ny):
    dn = dx * nx + dy * ny
    rx, ry = dx - 2 * dn * nx, dy - 2 * dn * ny
    nn = np.hypot(rx, ry)
    return rx / nn, ry / nn


def launch(cell, s, theta):
    """Points on the wall at s and unit directions at angle theta from the inward normal (counter-clockwise +)."""
    s, theta = np.broadcast_arrays(np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(theta, float)))
    x, y, nx, ny = cell.point_at(s)
    ux, uy = -nx, -ny
    return x, y, ux * np.cos(theta) - uy * np.sin(theta), ux * np.sin(theta) + uy * np.cos(theta)


def in_window(s, s0, w, P):
    d = np.mod(s - s0 + P / 2, P) - P / 2
    return np.abs(d) <= w / 2


@dataclass
class RayTable:
    cell: object
    port_w: float
    s_in: float
    s_out: float
    n_rays: int
    n_bounce: int
    chord: np.ndarray        # (n_rays, n_bounce): chord j ends at hit j
    sinchi: np.ndarray       # signed sin of the incidence angle at hit j
    element: np.ndarray      # element index of hit j (-1 unused)
    exit_idx: np.ndarray     # hit index in a port, -1 still inside
    exit_port: np.ndarray    # 1 output, 0 input, -1 inside (n_bounce / max_path reached), -2 leaked through a gap
    start: np.ndarray = field(default=None, repr=False)
    hits: np.ndarray = field(default=None, repr=False)

    @property
    def trapped_fraction(self):
        return float(np.mean(self.exit_port == -1))

    @property
    def leaked_fraction(self):
        return float(np.mean(self.exit_port == -2))


def trace_rays(cell, port_w, n_rays=1000, n_bounce=1500, theta0=0.35, s_in=None, s_out_frac=0.37, seed=1,
               theta_c=0.0, max_path=None, keep_hits=False, ports=True) -> RayTable:
    """Rays launched across the input port window (position uniform, angle uniform in theta_c ± theta0) and traced
    until they hit the input or the output window (s_out = s_in + s_out_frac × perimeter), n_bounce or max_path.
    ports=False traces a closed cell (no exits)."""
    rng = np.random.default_rng(seed)
    P = cell.perimeter
    s_in = cell.s_in_default if s_in is None else s_in
    s_out = s_in + s_out_frac * P
    x, y, dx, dy = launch(cell, s_in + (rng.random(n_rays) - 0.5) * port_w, theta_c + (rng.random(n_rays) * 2 - 1) * theta0)
    start = np.stack([x, y, dx, dy], 1).copy()
    chord = np.zeros((n_rays, n_bounce))
    sinchi = np.zeros((n_rays, n_bounce))
    elem = np.full((n_rays, n_bounce), -1)
    hits = np.full((n_rays, n_bounce, 2), np.nan) if keep_hits else None
    exit_idx = np.full(n_rays, -1)
    exit_port = np.full(n_rays, -1)
    alive = np.ones(n_rays, bool)
    Lc = np.zeros(n_rays)
    for j in range(n_bounce):
        ia = np.nonzero(alive)[0]
        if ia.size == 0:
            break
        t, nx, ny, hs, hk = cell.hit(x[ia], y[ia], dx[ia], dy[ia])
        bad = ~np.isfinite(t)
        if bad.any():
            exit_port[ia[bad]] = -2
            alive[ia[bad]] = False
            t = np.where(bad, 0.0, t)
        hx, hy = x[ia] + t * dx[ia], y[ia] + t * dy[ia]
        chord[ia, j] = t
        sinchi[ia, j] = dx[ia] * ny - dy[ia] * nx
        elem[ia, j] = hk
        if keep_hits:
            hits[ia, j, 0], hits[ia, j, 1] = hx, hy
        keep = ~bad
        if ports:
            pin, pout = in_window(hs, s_in, port_w, P), in_window(hs, s_out, port_w, P)
            done = (pin | pout) & ~bad
            exit_idx[ia[done]] = j
            exit_port[ia[done]] = np.where(pout[done], 1, 0)
            alive[ia[done]] = False
            keep &= ~done
        if max_path is not None:
            Lc[ia] += t
            over = keep & (Lc[ia] >= max_path)
            alive[ia[over]] = False
            keep &= ~over
        ik = ia[keep]
        x[ik], y[ik] = hx[keep], hy[keep]
        dx[ik], dy[ik] = reflect(dx[ia][keep], dy[ia][keep], nx[keep], ny[keep])
    return RayTable(cell, port_w, s_in, s_out, n_rays, n_bounce, chord, sinchi, elem, exit_idx, exit_port, start, hits)


@dataclass
class CellResult:
    T_det: float
    L_mean: float
    L_eff_gas: float
    S1: float
    L: np.ndarray
    W: np.ndarray
    n_det: int
    L_mean_err: float


def evaluate(table: RayTable, R_of_sinchi=None, alpha_bg=0.0, Gamma=1.0) -> CellResult:
    """Power-weighted statistics of the rays that reach the output port. R_of_sinchi(|sin χ|) -> reflectance
    (None = 1); every reflection before the exit counts, the exit hit itself does not."""
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    if sel.size == 0:
        return CellResult(0.0, np.nan, np.nan, 0.0, np.array([]), np.array([]), 0, np.nan)
    e = table.exit_idx[sel]
    ncol = int(e.max()) + 1
    j = np.arange(ncol)[None, :]
    L = (table.chord[sel, :ncol] * (j <= e[:, None])).sum(1)
    lnR = 0.0
    if R_of_sinchi is not None:
        R = np.asarray(R_of_sinchi(np.abs(table.sinchi[sel, :ncol])), float)
        lnR = (np.log(np.clip(R, 1e-300, 1.0)) * (j < e[:, None])).sum(1)
    W = np.exp(lnR - alpha_bg * L)
    T = W.sum() / table.n_rays
    Lm = (W * L).sum() / W.sum()
    var = (W * (L - Lm) ** 2).sum() / W.sum()
    n_k = W.sum() ** 2 / (W**2).sum()
    return CellResult(float(T), float(Lm), float(Gamma * Lm), float(Gamma * T * Lm), L, W, int(sel.size), float(np.sqrt(var / max(n_k, 1))))


def mean_field_estimate(cell, port_w, R_mean=1.0, alpha_bg=0.0, Gamma=1.0):
    """Ergodic (fully chaotic) estimate with two ports of width w: mean chord πA/P, exit probability w/P per hit."""
    ell = cell.mean_chord
    eta = port_w / cell.perimeter
    surv = (1 - 2 * eta) * R_mean * np.exp(-alpha_bg * ell)
    T = eta * np.exp(-alpha_bg * ell) / (1 - surv)
    L = ell / (1 - surv)
    return dict(T_det=T, L_mean=L, n_bounce=1 / (1 - surv), L_eff_gas=Gamma * L, S1=Gamma * T * L)


def trace_path(cell, s, theta, n_hits=50):
    """Polyline (x, y) of one trajectory from wall coordinate s at angle theta (no ports), and the hit table
    (s, signed sin χ, element)."""
    x, y, dx, dy = launch(cell, s, theta)
    xs, ys, S, SC, K = [x[0]], [y[0]], [], [], []
    for _ in range(n_hits):
        t, nx, ny, hs, hk = cell.hit(x, y, dx, dy)
        if not np.isfinite(t[0]):
            break
        x, y = x + t * dx, y + t * dy
        S.append(hs[0]); SC.append(float(dx[0] * ny[0] - dy[0] * nx[0])); K.append(int(hk[0]))
        dx, dy = reflect(dx, dy, nx, ny)
        xs.append(x[0]); ys.append(y[0])
    return np.array(xs), np.array(ys), np.array(S), np.array(SC), np.array(K)


def poincare(cell, n_rays=40, n_hits=300, seed=0):
    """(s, signed sin χ) of trajectories started uniformly in boundary phase space; arrays (n_rays, n_hits)."""
    rng = np.random.default_rng(seed)
    x, y, dx, dy = launch(cell, rng.random(n_rays) * cell.perimeter, np.arcsin(rng.uniform(-0.98, 0.98, n_rays)))
    S, SC = np.full((n_rays, n_hits), np.nan), np.full((n_rays, n_hits), np.nan)
    ok = np.ones(n_rays, bool)
    for j in range(n_hits):
        t, nx, ny, hs, _ = cell.hit(x, y, dx, dy)
        ok &= np.isfinite(t)
        t = np.where(ok, t, 0.0)
        x, y = x + t * dx, y + t * dy
        S[:, j] = np.where(ok, hs, np.nan)
        SC[:, j] = np.where(ok, dx * ny - dy * nx, np.nan)
        dx, dy = reflect(dx, dy, np.where(ok, nx, 1.0), np.where(ok, ny, 0.0))
    return S, SC


def twin_divergence(cell, s, theta, n_hits=300, dtheta=1e-9):
    """Distance between the hit points of two rays launched dtheta apart, per hit, and the Lyapunov fit
    (exponent per reflection; None when there is no clean exponential stretch)."""
    a = trace_path(cell, s, theta, n_hits)
    b = trace_path(cell, s, theta + dtheta, n_hits)
    n = min(len(a[0]), len(b[0]))
    sep = np.hypot(a[0][1:n] - b[0][1:n], a[1][1:n] - b[1][1:n])
    return sep, lyapunov_fit(sep, scale=np.sqrt(cell.area))
