"""Spectral metrics: FWHM, band statistics, ripple, speckle contrast; efficiency units."""
from __future__ import annotations

import numpy as np


def fwhm(x, y) -> float:
    """Full width at half maximum around the global maximum, with linear interpolation of both edges.
    NaN when the curve does not fall below half maximum inside the grid."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    half = y[i] / 2
    left = np.nonzero(y[:i] < half)[0]
    right = np.nonzero(y[i:] < half)[0]
    if left.size == 0 or right.size == 0:
        return float("nan")
    a = left[-1]
    xl = x[a] + (half - y[a]) * (x[a + 1] - x[a]) / (y[a + 1] - y[a])
    b = i + right[0]
    xr = x[b - 1] + (half - y[b - 1]) * (x[b] - x[b - 1]) / (y[b] - y[b - 1])
    return float(abs(xr - xl))


def band_stats(x, y, lo, hi) -> dict:
    """Mean, min, max, ripple (dB, max/min) and contrast σ/mean of y over lo ≤ x ≤ hi (last axis)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = (x >= min(lo, hi)) & (x <= max(lo, hi))
    if not np.any(sel):
        raise ValueError("band contains no grid points")
    yb = y[..., sel]
    mean, mn, mx = yb.mean(axis=-1), yb.min(axis=-1), yb.max(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ripple = 10 * np.log10(mx / mn)
        contrast = yb.std(axis=-1) / mean
    return {"mean": mean, "min": mn, "max": mx, "ripple_db": ripple, "contrast": contrast, "points": int(sel.sum())}


def ensemble_stats(y) -> dict:
    """Per-point statistics over realizations (axis 0): mean, std, contrast σ/mean, 10/50/90 % quantiles."""
    y = np.asarray(y, dtype=float)
    mean, std = y.mean(axis=0), y.std(axis=0, ddof=1) if y.shape[0] > 1 else np.zeros(y.shape[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = std / mean
    q10, q50, q90 = np.quantile(y, [0.1, 0.5, 0.9], axis=0)
    return {"mean": mean, "std": std, "contrast": c, "q10": q10, "median": q50, "q90": q90, "n": int(y.shape[0])}


def percent_per_w_cm2(eta_norm):
    """1/(W m²) -> %/(W cm²)."""
    return np.asarray(eta_norm) * 1e-2


def percent_per_w(eta):
    """1/W -> %/W."""
    return np.asarray(eta) * 1e2
