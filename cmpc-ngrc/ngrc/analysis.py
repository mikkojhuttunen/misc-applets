"""Sensitivity, coverage and validity analyses."""
from __future__ import annotations

import copy

import numpy as np

from .field import detector_field, field_correlation
from .rays import trace
from .shapes import Perturbation


def _intensity(fields, keys):
    return np.concatenate([np.abs(fields[k]) ** 2 for k in keys])


def sensitivity(model, base_shapes, m_max=6, h=0.01):
    """Finite-difference Jacobian of the stacked detector intensities with respect to a_m, b_m (m = 2..m_max),
    normalised by the mean unperturbed intensity, averaged over base shapes.

    Returns S[m-2] = rms over pixels of sqrt((|∂I/∂a_m|² + |∂I/∂b_m|²)/2) / Ī0 and the singular values of the
    stacked Jacobian (how many independent shape directions the detectors see)."""
    ref = model.fields(None)
    keys = sorted(ref)
    I0 = _intensity(ref, keys).mean()
    S, svals = [], []
    for base in base_shapes:
        cols = []
        for m in range(2, m_max + 1):
            for coef in ("a", "b"):
                d = []
                for sgn in (1, -1):
                    s = copy.deepcopy(base)
                    arr = getattr(s, coef).copy()
                    if len(arr) < m:
                        arr = np.pad(arr, (0, m - len(arr)))
                        other = "b" if coef == "a" else "a"
                        setattr(s, other, np.pad(getattr(s, other), (0, m - len(getattr(s, other)))))
                    arr[m - 1] += sgn * h
                    setattr(s, coef, arr)
                    d.append(_intensity(model.fields(Perturbation([s])), keys))
                cols.append((d[0] - d[1]) / (2 * h) / I0)
        J = np.array(cols).T
        Sm = np.sqrt((J[:, 0::2] ** 2 + J[:, 1::2] ** 2).mean(axis=0) / 2)
        S.append(Sm)
        svals.append(np.linalg.svd(J, compute_uv=False))
    return np.mean(S, axis=0), np.mean(svals, axis=0)


def occupancy(model, k=0, n=96, samples_per_chord=64):
    """Power-weighted ray density over the cell from the unperturbed chords of source k (n×n, extent ±Rc)."""
    t = model.tables[k]
    ch, ex = t["chords"], t["exits"]
    Rc = model.cell.radius
    # amplitude of each chord: amplitude of its ray at exit is a lower bound; use the per-ray exit amplitude
    # spread back with R^(bounces) is not stored per chord, so weight chords equally per ray (geometric density)
    g = (np.arange(samples_per_chord) + 0.5) / samples_per_chord
    X = ch["x"][:, None] + g[None, :] * ch["len"][:, None] * ch["tx"][:, None]
    Y = ch["y"][:, None] + g[None, :] * ch["len"][:, None] * ch["ty"][:, None]
    W = np.repeat(ch["len"][:, None] / samples_per_chord, samples_per_chord, axis=1)
    H, _, _ = np.histogram2d(Y.ravel(), X.ravel(), bins=n, range=[[-Rc, Rc], [-Rc, Rc]], weights=W.ravel())
    return H / H.max()


def validity_scan(cell, src, pert, scales, model=None):
    """Mean field correlation between the phase-screen and the full curved-ray speckle when Δn of `pert` is
    multiplied by each scale."""
    from .perturbative import PhaseScreenModel
    model = PhaseScreenModel(cell, [src]) if model is None else model
    out = []
    for f in scales:
        p = copy.deepcopy(pert)
        for c in p.components:
            c.dn *= f
        F = model.fields(p)
        ex = trace(cell, src, p, "curved")
        out.append(np.mean([field_correlation(F[(src.port, o)], detector_field(cell, ex, o)) for o in cell.outputs]))
    return np.array(out)


def decorrelation(model, pert, scales):
    """Field correlation of the perturbed speckle with the unperturbed one vs Δn scale (phase-screen model)."""
    ref = model.fields(None)
    out = []
    for f in scales:
        p = copy.deepcopy(pert)
        for c in p.components:
            c.dn *= f
        F = model.fields(p)
        out.append(np.mean([field_correlation(ref[k], F[k]) for k in ref]))
    return np.array(out)
