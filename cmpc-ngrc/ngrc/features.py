"""Speckle → feature vectors (the NGRC layer).

The K input ports (or launch conditions) × output ports × detector pixels intensities are the reservoir "nodes".
NGRC feature vector = [1, linear terms, unique quadratic monomials], the quadratic part optionally a random subset.
"""
from __future__ import annotations

import numpy as np


def stack_intensities(fields, keys=None, reference=None, mode="relative", eps=1e-12):
    """Concatenate |E|² over (input, output) keys. mode: "raw", "relative" (I/I_ref − 1), "log" (log I/I_ref),
    "delta" (I − I_ref)."""
    keys = sorted(fields) if keys is None else keys
    I = np.concatenate([np.abs(fields[k]) ** 2 for k in keys])
    if mode == "raw" or reference is None:
        return I
    Ir = np.concatenate([np.abs(reference[k]) ** 2 for k in keys])
    if mode == "relative":
        return I / (Ir + eps * Ir.mean()) - 1
    if mode == "log":
        return np.log((I + eps * Ir.mean()) / (Ir + eps * Ir.mean()))
    if mode == "delta":
        return I - Ir
    raise ValueError(mode)


def ngrc_features(X, quadratic=True, n_quad=None, rng=None, constant=True):
    """X (n_samples, d) → [1, X, X_i X_j (i ≤ j)]; n_quad: keep a random subset of that many monomials."""
    X = np.asarray(X, float)
    parts = [np.ones((len(X), 1))] if constant else []
    parts.append(X)
    if quadratic:
        i, j = np.triu_indices(X.shape[1])
        if n_quad is not None and n_quad < len(i):
            rng = np.random.default_rng(0) if rng is None else rng
            sel = rng.choice(len(i), n_quad, replace=False)
            i, j = i[sel], j[sel]
        parts.append(X[:, i] * X[:, j])
    return np.hstack(parts)


def add_noise(I, rng, shot=0.0, rel=0.0, floor=0.0):
    """Shot-like noise σ = shot·sqrt(I·Ī), relative noise σ = rel·I, additive floor σ = floor·Ī."""
    I = np.asarray(I, float)
    m = I.mean()
    sig = np.sqrt((shot**2) * np.abs(I) * m + (rel * I) ** 2 + (floor * m) ** 2)
    return I + rng.normal(0, 1, I.shape) * sig
