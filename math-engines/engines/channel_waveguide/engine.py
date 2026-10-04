"""Channel waveguides: strip-loaded, rib, ridge and buried geometries.

Two solvers:
- effective index method (EIM): vertical slab modes under and beside the core,
  then a lateral three-layer slab with the other polarisation;
- semi-vectorial finite differences (FD) on a rectangular, possibly non-uniform
  grid, for the dominant field E_x (quasi-TE) or E_y (quasi-TM).

Coordinates: x lateral (core centred on x = 0), y vertical. For strip, rib and
ridge the film bottom is y = 0; for the buried channel the substrate surface is
y = 0 and the core spans -(d + t) < y < -d. Region codes: 0 substrate, 1 film or
core, 2 loading strip, 3 superstrate. All lengths in m; wavelength in vacuum.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive
from ..slab_waveguide import engine as slab

GEOMETRIES = ("strip", "rib", "ridge", "buried")
_POL = ("TE", "TM")
_THIN = 5e-9  # layers thinner than this are treated as absent


def _check_geometry(geometry, width, film_thickness, strip_height, slab_thickness, depth):
    require_choice("geometry", geometry, GEOMETRIES)
    require_positive(width=width, film_thickness=film_thickness)
    require_nonnegative(strip_height=strip_height, slab_thickness=slab_thickness, depth=depth)


def laterally_confined(geometry, film_thickness, strip_height, slab_thickness, n_strip, n_sup) -> bool:
    if geometry == "strip":
        return strip_height > _THIN and abs(n_strip - n_sup) > 1e-6
    if geometry == "rib":
        return slab_thickness < film_thickness - _THIN
    return True


def region_at(geometry, x, y, width, film_thickness, strip_height=0.0, slab_thickness=0.0, depth=0.0):
    """Region code at (x, y) (helper, vectorised)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    a = np.abs(x) < width / 2
    t, h, ts, d = film_thickness, strip_height, slab_thickness, depth
    r = np.full(x.shape, 3, dtype=int)
    if geometry == "buried":
        r[y < 0] = 0
        r[(y < 0) & a & (y >= -(d + t)) & (y < -d)] = 1
        return r
    r[y < 0] = 0
    if geometry == "strip":
        r[(y >= 0) & (y < t)] = 1
        if h > _THIN:
            r[a & (y >= t) & (y < t + h)] = 2
    elif geometry == "rib":
        r[(y >= 0) & (y < ts)] = 1
        r[a & (y >= 0) & (y < t)] = 1
    else:
        r[a & (y >= 0) & (y < t)] = 1
    return r


def _stacks(geometry, t, h, ts, d, n_sup, n_strip, n_film, n_sub):
    floor = max(n_sub, n_sup)
    if geometry == "strip":
        inner = [(n_film, t)] + ([(n_strip, h)] if h > _THIN else [])
        return inner, [(n_film, t)], floor
    if geometry == "rib":
        return [(n_film, t)], ([(n_film, min(ts, t))] if ts > _THIN else []), floor
    if geometry == "ridge":
        return [(n_film, t)], [], floor
    return [(n_film, t)] + ([(n_sub, d)] if d > 1e-9 else []), [], n_sub


def _vertical_modes(wavelength, stack, n_sub, n_sup, pol):
    if not stack:
        return []
    r = slab.multilayer_neff(wavelength, [n_sub] + [n for n, _ in stack] + [n_sup], [d for _, d in stack], pol)
    return [float(v) for v in r["neff"]]


def _lateral_modes(wavelength, n_out, n_in, width, pol):
    out, m = [], 0
    while m < 100:
        v = slab.neff_three_layer(wavelength, n_out, n_in, n_out, width, pol, m)
        if not np.isfinite(v):
            break
        out.append(float(v)); m += 1
    return out


def eim_modes(geometry, width, film_thickness, n_sup, n_film, n_sub, wavelength, polarization="TE",
              strip_height=0.0, n_strip=1.0, slab_thickness=0.0, depth=0.0) -> Result:
    """Guided modes of one polarisation family from the effective index method.

    Quasi-TE: TE vertical slab, then a TM lateral slab of width w; quasi-TM the other way round.
    For vertical order m the outside index is the m-th mode of the outside stack, or the
    cladding index when that stack has no such mode. A mode is flagged leaky when its index
    lies below the fundamental outside slab mode of the same polarisation.
    """
    _check_geometry(geometry, width, film_thickness, strip_height, slab_thickness, depth)
    require_positive(n_sup=n_sup, n_film=n_film, n_sub=n_sub, n_strip=n_strip, wavelength=wavelength)
    require_choice("polarization", polarization, _POL)
    inner, outer, floor = _stacks(geometry, film_thickness, strip_height, slab_thickness, depth, n_sup, n_strip, n_film, n_sub)
    conf = laterally_confined(geometry, film_thickness, strip_height, slab_thickness, n_strip, n_sup)
    vin = _vertical_modes(wavelength, inner, n_sub, n_sup, polarization)
    rows = []
    out_fund = None
    if not conf:
        rows = [(v, m, -1, v, v, False) for m, v in enumerate(vin)]
    else:
        vout = _vertical_modes(wavelength, outer, n_sub, n_sup, polarization)
        out_fund = vout[0] if vout else None
        lat = "TM" if polarization == "TE" else "TE"
        fallback = n_sub if geometry == "buried" else floor
        for m, n_in in enumerate(vin):
            n_out = vout[m] if m < len(vout) else fallback
            if n_in <= n_out + 1e-9:
                continue
            for n, ne in enumerate(_lateral_modes(wavelength, n_out, n_in, width, lat)):
                if ne > floor + 1e-9:
                    rows.append((ne, m, n, n_in, n_out, out_fund is not None and ne < out_fund))
    rows.sort(key=lambda r: -r[0])
    col = lambda i, dt=float: np.array([r[i] for r in rows], dtype=dt)
    return Result(
        values={"neff": col(0), "m": col(1, int), "n": col(2, int), "n_in": col(3), "n_out": col(4), "leaky": col(5, bool),
                "n_modes": len(rows), "laterally_confined": conf, "outside_slab_neff": np.nan if out_fund is None else out_fund},
        units={"neff": "", "m": "", "n": "", "n_in": "", "n_out": "", "leaky": "", "n_modes": "", "laterally_confined": "",
               "outside_slab_neff": ""},
        assumptions=["Effective index method: separable field, least accurate near cut-off and for tall cores",
                     "n = -1 marks a slab mode (no lateral confinement)"],
    )


def _operator(x_edges, y_edges, index_map, wavelength, polarization):
    """Sparse semi-vectorial operator A with A E = β² E (row-major cells, x fastest)."""
    from scipy import sparse

    xe, ye = np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)
    dx, dy = np.diff(xe), np.diff(ye)
    nx, ny = dx.size, dy.size
    n2 = np.asarray(index_map, dtype=float) ** 2
    if n2.shape != (ny, nx):
        raise ValueError(f"index_map must have shape (len(y_edges)-1, len(x_edges)-1) = {(ny, nx)}, got {n2.shape}")
    k0 = 2 * np.pi / wavelength
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny)
    wx, wy = polarization == "TE", polarization == "TM"
    for j in range(ny):
        for i in range(nx):
            r = i + nx * j
            c = n2[j, i]
            for s in (-1, 1):
                ii = i + s
                if ii < 0 or ii >= nx:
                    diag[r] -= 1 / (dx[i] * dx[i])
                    continue
                cf = 1 / (0.5 * (dx[i] + dx[ii]) * dx[i])
                nn = n2[j, ii]
                if wx:
                    a = 2 / (c + nn)
                    rows.append(r); cols.append(ii + nx * j); vals.append(cf * a * nn); diag[r] -= cf * a * c
                else:
                    rows.append(r); cols.append(ii + nx * j); vals.append(cf); diag[r] -= cf
            for s in (-1, 1):
                jj = j + s
                if jj < 0 or jj >= ny:
                    diag[r] -= 1 / (dy[j] * dy[j])
                    continue
                cf = 1 / (0.5 * (dy[j] + dy[jj]) * dy[j])
                nn = n2[jj, i]
                if wy:
                    a = 2 / (c + nn)
                    rows.append(r); cols.append(i + nx * jj); vals.append(cf * a * nn); diag[r] -= cf * a * c
                else:
                    rows.append(r); cols.append(i + nx * jj); vals.append(cf); diag[r] -= cf
            diag[r] += k0 * k0 * c
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr() + sparse.diags(diag)
    return A, k0, n2


def fd_modes(x_edges, y_edges, index_map, wavelength, polarization="TE", n_modes=4, n_floor=None) -> Result:
    """Guided modes of the semi-vectorial wave equation on a rectangular grid.

    index_map[j, i] is the refractive index of cell (x_i, y_j). Quasi-TE solves for E_x with
    ∂x[(1/n²) ∂x(n² E_x)] + ∂y² E_x + k0² n² E_x = β² E_x, quasi-TM the same with x and y
    exchanged. Cell-centred nodes, n² averaged arithmetically across each face, the field
    vanishing one cell beyond the window. Modes with neff <= n_floor (default: smallest index
    on the window boundary) are discarded. Fields are normalised to a positive maximum of 1.
    """
    from scipy.sparse.linalg import eigs

    require_positive(wavelength=wavelength)
    require_choice("polarization", polarization, _POL)
    A, k0, n2 = _operator(x_edges, y_edges, index_map, wavelength, polarization)
    if n_floor is None:
        n_floor = float(np.sqrt(min(n2[0].min(), n2[-1].min(), n2[:, 0].min(), n2[:, -1].min())))
    sigma = k0 * k0 * n2.max() * 1.0001
    k = int(min(max(1, n_modes + 2), A.shape[0] - 2))
    vals, vecs = eigs(A, k=k, sigma=sigma, which="LM", v0=np.ones(A.shape[0]))  # fixed start vector: reproducible
    order = np.argsort(-vals.real)
    neffs, fields = [], []
    for o in order:
        b2 = vals[o].real
        if b2 <= 0 or abs(vals[o].imag) > 1e-9 * abs(b2):
            continue
        ne = np.sqrt(b2) / k0
        if ne <= n_floor + 1e-6:
            continue
        f = vecs[:, o].real.reshape(n2.shape)
        f = f / f.flat[np.argmax(np.abs(f))]
        neffs.append(ne); fields.append(f)
        if len(neffs) == n_modes:
            break
    return Result(
        values={"neff": np.array(neffs), "field": np.array(fields)},
        units={"neff": "", "field": ""},
        assumptions=["Semi-vectorial: polarisation coupling at corners neglected", "Field zero one cell outside the window",
                     "E_x (quasi-TE) or E_y (quasi-TM), normalised to a maximum of +1"],
    )


def region_fractions(field, x_edges, y_edges, regions, n_regions=4):
    """Share of ∫|E|² dA in each region code 0..n_regions-1 (helper)."""
    dx, dy = np.diff(np.asarray(x_edges, dtype=float)), np.diff(np.asarray(y_edges, dtype=float))
    w = np.asarray(field, dtype=float) ** 2 * dy[:, None] * dx[None, :]
    tot = w.sum()
    return np.array([w[np.asarray(regions) == c].sum() / tot for c in range(n_regions)])


def fd_fundamental(geometry, width, film_thickness, n_sup, n_film, n_sub, wavelength, polarization="TE",
                   strip_height=0.0, n_strip=1.0, slab_thickness=0.0, depth=0.0, cell=20e-9, margin=1.5e-6) -> Result:
    """Fundamental mode on a uniform grid of square cells, with the field share per region."""
    _check_geometry(geometry, width, film_thickness, strip_height, slab_thickness, depth)
    require_positive(cell=cell, margin=margin, wavelength=wavelength)
    if geometry == "buried":
        y0, y1 = -(depth + film_thickness), max(0.0, -depth)
    else:
        y0, y1 = 0.0, film_thickness + (strip_height if geometry == "strip" else 0.0)
    xe = np.arange(-(width / 2 + margin), width / 2 + margin + cell / 2, cell)
    ye = np.arange(y0 - margin, y1 + margin + cell / 2, cell)
    if (xe.size - 1) * (ye.size - 1) > 60000:
        raise ValueError("grid too large: increase cell or reduce margin")
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    reg = region_at(geometry, xc[None, :], yc[:, None], width, film_thickness, strip_height, slab_thickness, depth)
    nmap = np.choose(reg, [n_sub, n_film, n_strip, n_sup])
    r = fd_modes(xe, ye, nmap, wavelength, polarization, 1, n_floor=max(n_sub, n_sup))
    if not r["neff"].size:
        return Result(values={"neff": np.nan, "share_sub": np.nan, "share_film": np.nan, "share_strip": np.nan, "share_sup": np.nan},
                      units={k: "" for k in ("neff", "share_sub", "share_film", "share_strip", "share_sup")},
                      assumptions=["No guided mode found in the window"])
    fr = region_fractions(r["field"][0], xe, ye, reg)
    return Result(
        values={"neff": float(r["neff"][0]), "share_sub": fr[0], "share_film": fr[1], "share_strip": fr[2], "share_sup": fr[3]},
        units={"neff": "", "share_sub": "", "share_film": "", "share_strip": "", "share_sup": ""},
        assumptions=r.assumptions + [f"Uniform {cell * 1e9:.0f} nm cells, {margin * 1e6:.2f} µm margin"],
    )
