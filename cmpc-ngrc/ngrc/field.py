"""Coherent beamlet sum at the detector line of an output port → complex speckle field."""
from __future__ import annotations

import numpy as np


def beamlet_matrix(cell, exits, port, points=None):
    """G[pixel, j] = field of exit record j (of `port`) at the detector pixels; E = G @ 1 is the speckle field, and
    E = G @ exp(i k0 ΔL_j) adds per-record optical-path changes (phase-screen model). Returns (G, ray ids).
    With aperture-model exits (and no explicit points) G = K @ B: B samples each beamlet over the port opening,
    K propagates the aperture field to the detector pixels (2D Rayleigh–Sommerfeld, kR ≫ 1)."""
    if getattr(exits, "model", "gbs") == "fga":
        e = exits.select(port)
        if points is None:
            ap, du = cell.aperture_points(port)
            return propagation_matrix(cell, port, ap, du) @ _frozen_at(cell, exits, e, ap, port), e["ray"]
        return _frozen_at(cell, exits, e, points, port), e["ray"]
    if points is None:
        points, _ = cell.detector_points(port)
    return _beamlets_at(cell, exits.select(port), points)


def propagation_matrix(cell, port, ap, du, points=None):
    """K[pixel, k]: field at the detector pixels from aperture samples ap (spacing du), 2D Rayleigh–Sommerfeld in
    the asymptotic form sqrt(k/(2π i R)) e^{ikR} cos θ du, k = k0 n0, θ from the port's outward normal.
    Far-field detector (default): sqrt(k/(2π i)) e^{−ik d̂·r} cos θ du per direction d̂ (the 1/sqrt(R) e^{ikR} of a
    common distance dropped)."""
    _, nrm, tan = cell.port_frame(port)
    k = cell.k0 * cell.n_eff
    if points is None and cell.detector.farfield:
        sn = cell.detector_sines()
        cs = np.sqrt(1 - sn**2)
        dx, dy = cs * nrm[0] + sn * tan[0], cs * nrm[1] + sn * tan[1]
        ph = -k * (dx[:, None] * ap[None, :, 0] + dy[:, None] * ap[None, :, 1])
        return np.sqrt(k / (2j * np.pi)) * np.exp(1j * ph) * cs[:, None] * du
    if points is None:
        points, _ = cell.detector_points(port)
    dx = points[:, None, 0] - ap[None, :, 0]
    dy = points[:, None, 1] - ap[None, :, 1]
    R = np.hypot(dx, dy)
    cos = np.maximum((dx * nrm[0] + dy * nrm[1]) / R, 0.0)
    return np.sqrt(k / (2j * np.pi * R)) * np.exp(1j * k * R) * cos * du


def _frozen_at(cell, exits, e, points, port):
    """Frozen-Gaussian (Herman–Kluk) contributions of the exit records e of `port` at points of its opening.
    Coherent states live on the port opening (coordinate x along its tangent): W R (γ/π)^¼ exp(−γ (x − x_t)²/2)
    exp(i k0 [L + n0 t·(r − r_hit)] + iφ) (on a flat opening the last phase is k0 p_t (x − x_t)), with R from the
    stability matrix taken from the ray frame to the opening (A/cos χ, B/cos χ, C cos χ, D cos χ)."""
    from .rays import zfun
    n0, k0 = cell.n_eff, cell.k0
    c, nrm, tan = cell.port_frame(port)
    cosc = np.maximum(np.abs(e["tx"] * nrm[0] + e["ty"] * nrm[1]), 1e-3)
    Q = e["Q"].real / cosc + 1j * e["Q"].imag / cosc
    P = e["P"].real * cosc + 1j * e["P"].imag * cosc
    z = zfun(Q, P, exits.beta, exits.gamma, k0)
    zr = zfun(e["Q"], e["P"], exits.beta, exits.gamma, k0)
    argz = e["argZ"] + np.angle(z / zr)
    R = np.sqrt(np.abs(z)) * np.exp(0.5j * argz)
    xt = (e["x"] - c[0]) * tan[0] + (e["y"] - c[1]) * tan[1]
    xp = (points[:, 0] - c[0]) * tan[0] + (points[:, 1] - c[1]) * tan[1]
    d = xp[:, None] - xt[None, :]
    # exact local plane-wave phase k0 n0 t·(r − r_hit) on the (curved) opening; envelope along the opening
    s = (points[:, 0:1] - e["x"][None, :]) * e["tx"] + (points[:, 1:2] - e["y"][None, :]) * e["ty"]
    g = (exits.gamma / np.pi) ** 0.25 * np.exp(-exits.gamma * d * d / 2 + 1j * k0 * n0 * s)
    return (e["amp"] * e["W"] * R * np.exp(1j * (k0 * e["L"] + e["phase"]))) * g


def _beamlets_at(cell, e, points):
    n0, k0 = cell.n_eff, cell.k0
    dx = points[:, 0:1] - e["x"][None, :]
    dy = points[:, 1:2] - e["y"][None, :]
    s = dx * e["tx"] + dy * e["ty"]
    q = -dx * e["ty"] + dy * e["tx"]
    Qs = e["Q"] + e["P"] * s / n0
    M = e["P"] / Qs
    argQs = e["argQ"] + np.angle(Qs / e["Q"])
    amp = e["amp"] * np.abs(Qs) ** -0.5 * np.exp(-0.5j * argQs)
    G = amp * np.exp(1j * (k0 * (e["L"] + n0 * s + 0.5 * M * q * q) + e["phase"]))
    return G, e["ray"]


def detector_field(cell, exits, port, points=None):
    G, _ = beamlet_matrix(cell, exits, port, points)
    return G.sum(axis=1)


def speckle_contrast(I):
    I = np.asarray(I, float)
    return float(I.std() / I.mean()) if I.mean() > 0 else np.nan


def field_correlation(E1, E2):
    """|<E1, E2>| / (|E1| |E2|): 1 for identical speckle, ~1/sqrt(N) for independent speckle."""
    E1, E2 = np.ravel(E1), np.ravel(E2)
    return float(np.abs(np.vdot(E1, E2)) / (np.linalg.norm(E1) * np.linalg.norm(E2)))


def intensity_correlation(I1, I2):
    return float(np.corrcoef(np.ravel(I1), np.ravel(I2))[0, 1])
