"""Coherent beamlet sum at the detector line of an output port → complex speckle field."""
from __future__ import annotations

import numpy as np


def beamlet_matrix(cell, exits, port, points=None):
    """G[pixel, j] = field of exit ray j (of `port`) at the detector pixels; E = G @ 1 is the speckle field, and
    E = G @ exp(i k0 ΔL_j) adds per-ray optical-path changes (phase-screen model). Returns (G, ray ids)."""
    if points is None:
        points, _ = cell.detector_points(port)
    e = exits.select(port)
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
