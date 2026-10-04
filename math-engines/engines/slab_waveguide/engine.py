"""Guided modes of planar (slab) waveguides.

Geometry: substrate x < 0, core 0..d, cladding x > d (three-layer), or any
stack listed bottom to top (multilayer). TE = E along y, TM = H along y.
Wavelength is the vacuum wavelength; all lengths in m.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive

_POL = ("TE", "TM")


def _dispersion(ne, k, ns, nf, nc, d, pol, m):
    kx = k * np.sqrt(max(nf * nf - ne * ne, 0.0))
    gs = k * np.sqrt(max(ne * ne - ns * ns, 0.0))
    gc = k * np.sqrt(max(ne * ne - nc * nc, 0.0))
    rs = nf * nf / (ns * ns) if pol == "TM" else 1.0
    rc = nf * nf / (nc * nc) if pol == "TM" else 1.0
    return kx * d - m * np.pi - np.arctan2(rs * gs, kx) - np.arctan2(rc * gc, kx)


def neff_three_layer(wavelength, n_sub, n_core, n_clad, thickness, polarization="TE", order=0):
    """Effective index of mode `order`; NaN below cut-off (helper, scalar)."""
    k = 2 * np.pi / wavelength
    lo, hi = max(n_sub, n_clad) + 1e-12, n_core - 1e-12
    if n_core <= max(n_sub, n_clad) or thickness <= 0:
        return np.nan
    if _dispersion(lo, k, n_sub, n_core, n_clad, thickness, polarization, order) <= 0:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _dispersion(mid, k, n_sub, n_core, n_clad, thickness, polarization, order) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def slab_neff(wavelength, n_sub, n_core, n_clad, thickness, polarization="TE", order=0) -> Result:
    """Effective index, normalised frequency V and normalised index b of a three-layer slab mode."""
    require_positive(wavelength=wavelength, n_sub=n_sub, n_core=n_core, n_clad=n_clad, thickness=thickness)
    require_choice("polarization", polarization, _POL)
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    k = 2 * np.pi / wavelength
    neff = neff_three_layer(wavelength, n_sub, n_core, n_clad, thickness, polarization, order)
    V = k * thickness * np.sqrt(n_core**2 - n_sub**2) if n_core > n_sub else np.nan
    b = (neff**2 - n_sub**2) / (n_core**2 - n_sub**2) if np.isfinite(neff) else np.nan
    n_modes = 0
    while n_modes < 200 and np.isfinite(neff_three_layer(wavelength, n_sub, n_core, n_clad, thickness, polarization, n_modes)):
        n_modes += 1
    return Result(
        values={"neff": neff, "V": V, "b": b, "guided": bool(np.isfinite(neff)), "n_modes": n_modes},
        units={"neff": "", "V": "", "b": "", "guided": "", "n_modes": ""},
        assumptions=["Lossless isotropic layers", "neff is NaN below cut-off"],
    )


def multilayer_neff(wavelength, indices, thicknesses, polarization="TE") -> Result:
    """All guided modes of a stack. indices: bottom to top, first and last semi-infinite;
    thicknesses: the interior layers (len(indices) - 2 values). Returns neff (descending)
    and the fraction of each mode's power-like integral ∫E² in every interior layer."""
    require_positive(wavelength=wavelength)
    n = [float(v) for v in indices]
    t = [float(v) for v in thicknesses]
    require_positive(indices=n)
    require_nonnegative(thicknesses=t)
    require_choice("polarization", polarization, _POL)
    if len(t) != len(n) - 2:
        raise ValueError("thicknesses must have len(indices) - 2 entries")
    k = 2 * np.pi / wavelength
    p = (lambda x: x * x) if polarization == "TM" else (lambda x: 1.0)

    def shoot(ne, keep=False):
        b2 = (k * ne) ** 2
        gs = np.sqrt(max(b2 - (k * n[0]) ** 2, 0.0))
        E, D = 1.0, gs / p(n[0])
        samples = []
        x = 0.0
        for i, d in enumerate(t, start=1):
            q = (k * n[i]) ** 2 - b2
            P = p(n[i])

            def step(xx, E=E, D=D, q=q, P=P):
                if q > 0:
                    kk = np.sqrt(q)
                    return E * np.cos(kk * xx) + P * D / kk * np.sin(kk * xx), (-E * kk * np.sin(kk * xx) + P * D * np.cos(kk * xx)) / P
                g = np.sqrt(-q) or 1e-30
                return E * np.cosh(g * xx) + P * D / g * np.sinh(g * xx), (E * g * np.sinh(g * xx) + P * D * np.cosh(g * xx)) / P

            if keep:
                for j in range(60):
                    xx = d * (j + 0.5) / 60
                    samples.append((i, step(xx)[0], d / 60))
            E, D = step(d)
            s = abs(E) + abs(D)
            if s > 1e100:
                E, D = E / s, D / s
                samples = [(a, b / s, c) for a, b, c in samples]
            x += d
        gc = np.sqrt(max(b2 - (k * n[-1]) ** 2, 0.0))
        return p(n[-1]) * D + gc * E, E, gs, gc, samples

    nlo, nmax = max(n[0], n[-1]), max(n)
    grid = nlo + (nmax - nlo) * np.arange(1, 3000) / 3000
    vals = [shoot(v)[0] for v in grid]
    roots = []
    for i in range(1, len(grid)):
        if np.sign(vals[i]) != np.sign(vals[i - 1]) and np.isfinite(vals[i]):
            lo, hi, flo = grid[i - 1], grid[i], vals[i - 1]
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                fm = shoot(mid)[0]
                if np.sign(fm) == np.sign(flo):
                    lo, flo = mid, fm
                else:
                    hi = mid
            r = 0.5 * (lo + hi)
            f, E, *_ = shoot(r)
            if abs(f) < 1e-3 * (abs(E) + 1) * k:
                roots.append(r)
    roots.sort(reverse=True)
    fractions = []
    for r in roots:
        _, E, gs, gc, smp = shoot(r, keep=True)
        tot = 1 / (2 * gs) + E * E / (2 * gc) + sum(e * e * w for _, e, w in smp)
        fr = [0.0] * len(t)
        for i, e, w in smp:
            fr[i - 1] += e * e * w / tot
        fractions.append(fr)
    return Result(
        values={"neff": np.array(roots), "layer_fraction": np.array(fractions).reshape(len(roots), len(t))},
        units={"neff": "", "layer_fraction": ""},
        assumptions=["Lossless isotropic layers", "layer_fraction uses ∫E² (TE power for TE modes)"],
    )
