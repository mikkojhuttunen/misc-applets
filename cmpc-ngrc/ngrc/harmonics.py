"""Circular-harmonic decomposition (CHD) of 2D shapes: r(θ) = R0 [1 + Σ_m (a_m cos mθ + b_m sin mθ)].

`shape_chd`  exact coefficients of an analytic Shape (the ground-truth labels).
`contour`    boundary r(θ) of any Δn map at a level (default half the peak) around its centroid.
`chd`        Fourier coefficients of r(θ).
`map_chd`    contour + chd: what an ideal "digital" CHD of the image gives (the baseline the optics is compared with).
`power`      rotation-invariant spectrum p_m = a_m² + b_m².
"""
from __future__ import annotations

import numpy as np


def shape_chd(shape, m_max=None):
    a, b = shape.coefficients()
    m_max = len(a) if m_max is None else m_max
    a = np.pad(a, (0, max(0, m_max - len(a))))[:m_max]
    b = np.pad(b, (0, max(0, m_max - len(b))))[:m_max]
    return shape.R, a, b


def chd(r, m_max=8):
    """R0, a[m-1], b[m-1] (m = 1..m_max) of r sampled at θ_k = 2πk/N."""
    c = np.fft.rfft(np.asarray(r, float)) / len(r)
    R0 = c[0].real
    cm = c[1:m_max + 1]
    cm = np.pad(cm, (0, m_max - len(cm)))
    return R0, 2 * cm.real / R0, -2 * cm.imag / R0


def power(a, b):
    return np.asarray(a) ** 2 + np.asarray(b) ** 2


def centroid(value_fn, x0, y0, half, n=257):
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x0 + u, y0 + u)
    w = np.abs(value_fn(X, Y))
    s = w.sum()
    return (float((w * X).sum() / s), float((w * Y).sum() / s)) if s > 0 else (x0, y0)


def contour(value_fn, x0, y0, r_max, n_theta=256, n_r=600, level=0.5, peak=None, use_centroid=True):
    """θ_k = 2πk/N and the outermost radius where |Δn| crosses level·peak along each ray from the centroid."""
    if use_centroid:
        x0, y0 = centroid(value_fn, x0, y0, r_max)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    rr = np.linspace(0, r_max, n_r)
    X = x0 + rr[None, :] * np.cos(th)[:, None]
    Y = y0 + rr[None, :] * np.sin(th)[:, None]
    v = np.abs(value_fn(X, Y))
    pk = v.max() if peak is None else abs(peak)
    above = v >= level * pk
    r = np.zeros(n_theta)
    for k in range(n_theta):
        idx = np.nonzero(above[k])[0]
        if len(idx) == 0:
            continue
        j = idx[-1]
        if j + 1 >= n_r:
            r[k] = rr[-1]
            continue
        v0, v1 = v[k, j], v[k, j + 1]
        f = (v0 - level * pk) / (v0 - v1) if v0 != v1 else 0.0
        r[k] = rr[j] + f * (rr[j + 1] - rr[j])
    return th, r, (x0, y0)


def map_chd(component, m_max=8, n_theta=256, level=0.5):
    """CHD of one perturbation component (Shape or GridIndex) from its Δn map."""
    fn = lambda X, Y: component.eval(X, Y, hess=False)[0]
    th, r, c = contour(fn, component.x0, component.y0, component.bound, n_theta=n_theta, level=level)
    R0, a, b = chd(r, m_max)
    return R0, a, b, c
