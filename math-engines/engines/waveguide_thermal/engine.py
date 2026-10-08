"""Steady-state 2D heat conduction in a waveguide cross-section (finite volumes on a graded rectilinear grid).

    ∇·(k ∇T) = -q'''        q''' = heat per volume (W/m^3), uniform in the heated region

Boundary conditions: isothermal substrate bottom (heat sink, ΔT = 0), convective top surface (h_top, ΔT_amb = 0),
adiabatic sides. The waveguide is long and uniformly heated along z (2D). Output temperatures are rises above the
heat sink. Materials and their thermal conductivities come from the thermo_optic LUT.

ridge_heating builds the common stacks: substrate / buried layer (BOX) / optional film (slab) / ridge or strip,
with cladding around and above. Only the half x >= 0 is solved (mirror symmetry). The energy time constant
τ_E = (∫ ρ c_p ΔT dA) / q' is the stored heat over the heating power: the area under 1 - ΔT(t)/ΔT_∞ for a heat
step, weighted over the cross-section (a mean response time, not a single-exponential fit).
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive
from ..thermo_optic import engine as _to

MATERIALS = _to.MATERIALS


def graded_axis(breaks, fine_lo, fine_hi, h_fine, growth=0.15, min_cells=2):
    """Cell edges through every break point; spacing h_fine inside [fine_lo, fine_hi], growing linearly
    (h = h_fine + growth * distance) away from it."""
    b = np.unique(np.asarray(breaks, dtype=float))
    edges = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        s = np.linspace(a, c, 801)
        dist = np.maximum(np.maximum(fine_lo - s, s - fine_hi), 0.0)
        inv = 1.0 / (h_fine + growth * dist)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
        n = max(min_cells, int(math.ceil(cum[-1])))
        edges.extend(np.interp(np.linspace(0, cum[-1], n + 1)[1:], cum, s))
    edges = np.array(edges)
    edges[-1] = b[-1]
    return edges


def solve_heat(x_edges, y_edges, k, q, h_top=10.0):
    """Finite-volume solve. k, q: (ny, nx) arrays of conductivity (W/(m K)) and heat per volume (W/m^3).
    Bottom edge isothermal (ΔT = 0), top edge convective (h_top, W/(m^2 K)), left and right edges adiabatic.
    Returns ΔT (ny, nx) and the heat flow into the sink per unit length (W/m)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    dx, dy = np.diff(x_edges), np.diff(y_edges)
    ny, nx = k.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx))
    # horizontal faces
    Gx = dy[:, None] / (dx[None, :-1] / (2 * k[:, :-1]) + dx[None, 1:] / (2 * k[:, 1:]))
    # vertical faces
    Gy = dx[None, :] / (dy[:-1, None] / (2 * k[:-1, :]) + dy[1:, None] / (2 * k[1:, :]))
    for G, a, b in ((Gx, idx[:, :-1], idx[:, 1:]), (Gy, idx[:-1, :], idx[1:, :])):
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
        vals += [-G.ravel(), -G.ravel()]
    diag[:, :-1] += Gx
    diag[:, 1:] += Gx
    diag[:-1, :] += Gy
    diag[1:, :] += Gy
    G_bot = dx / (dy[0] / (2 * k[0, :]))
    diag[0, :] += G_bot
    if h_top > 0:
        diag[-1, :] += dx / (dy[-1] / (2 * k[-1, :]) + 1.0 / h_top)
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny)).tocsr()
    rhs = (q * dy[:, None] * dx[None, :]).ravel()
    T = spsolve(A, rhs).reshape(ny, nx)
    return T, float(np.sum(G_bot * T[0, :]))


def _k_of(material, override=None):
    return float(override) if override is not None else _to.entry(material)["k"]


def ridge_heating(heat_per_length=1.0, core_material="si", core_width=0.5e-6, core_height=0.22e-6,
                  slab_material="si", slab_thickness=0.0, box_material="sio2", box_thickness=2e-6,
                  substrate_material="si", substrate_thickness=500e-6, clad_material="sio2", clad_thickness=2e-6,
                  domain_half_width=500e-6, h_top=10.0, k_core=None, k_slab=None, k_box=None, k_substrate=None,
                  k_clad=None, resolution=1.0, return_field=False) -> Result:
    """Temperature rise of a heated ridge/strip waveguide on a layered substrate.

    Stack, bottom to top: substrate (H), buried layer (BOX), film of slab_material (slab_thickness, may be 0),
    ridge/strip of core_material (core_width x core_height) on the film; cladding fills the rest up to
    clad_thickness above the BOX (raised to the device height if smaller). heat_per_length q' (W/m) is
    deposited uniformly in the ridge. k_* override the LUT conductivities (thin-film values, worst cases)."""
    require_positive(heat_per_length=heat_per_length, core_width=core_width, core_height=core_height,
                     box_thickness=box_thickness, substrate_thickness=substrate_thickness,
                     domain_half_width=domain_half_width, resolution=resolution)
    require_nonnegative(slab_thickness=slab_thickness, clad_thickness=clad_thickness, h_top=h_top)
    for name, m in (("core_material", core_material), ("slab_material", slab_material), ("box_material", box_material),
                    ("substrate_material", substrate_material), ("clad_material", clad_material)):
        require_choice(name, m, MATERIALS)
    if domain_half_width <= core_width / 2:
        raise ValueError("domain_half_width must exceed core_width / 2")
    kc, ks, kb, ksub, kcl = (_k_of(core_material, k_core), _k_of(slab_material, k_slab), _k_of(box_material, k_box),
                             _k_of(substrate_material, k_substrate), _k_of(clad_material, k_clad))
    require_positive(k_core=kc, k_slab=ks, k_box=kb, k_substrate=ksub, k_clad=kcl)

    H, tb, ts, hc = substrate_thickness, box_thickness, slab_thickness, core_height
    y_box, y_film = H + tb, H + tb + ts
    y_core = y_film + hc
    y_top = max(H + tb + clad_thickness, y_core)
    w2 = core_width / 2
    feats = [w2, hc, tb] + ([ts] if ts > 0 else []) + ([y_top - y_core] if y_top > y_core else [])
    h_f = min(feats) / (6 * resolution)
    h_f = max(h_f, min(w2, hc) / (12 * resolution))
    growth = 0.15 / resolution
    xe = graded_axis([0.0, w2, domain_half_width], 0.0, w2, min(h_f, w2 / 4), growth)
    ybreaks = [0.0, H, y_box, y_film, y_core, y_top]
    ye = graded_axis(ybreaks, y_box, y_core, h_f, growth)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc)
    mat = np.full(X.shape, clad_material, dtype=object)
    mat[Y < y_box] = box_material
    mat[Y < H] = substrate_material
    if ts > 0:
        mat[(Y > y_box) & (Y < y_film)] = slab_material
    core = (Y > y_film) & (Y < y_core) & (X < w2)
    mat[core] = core_material
    # by region, so that a material used in two roles (Si core and Si substrate) keeps each region's k
    k = np.empty(X.shape)
    k[:] = kcl
    k[Y < y_box] = kb
    k[Y < H] = ksub
    if ts > 0:
        k[(Y > y_box) & (Y < y_film)] = ks
    k[core] = kc
    dA = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    A_half = float(np.sum(dA[core]))
    q = np.zeros(X.shape)
    q[core] = (heat_per_length / 2) / A_half
    T, q_sink = solve_heat(xe, ye, k, q, h_top)
    rc = np.zeros(X.shape)
    for m in set(mat.ravel()):
        e = _to.entry(m)
        rc[mat == m] = e["rho"] * e["cp"]
    T_core = float(np.sum(T[core] * dA[core]) / A_half)
    film = (Y > y_box) & (Y < y_film) & (X < w2) if ts > 0 else core
    vals = {
        "dT_core": T_core, "dT_max": float(T.max()), "R_th": T_core / heat_per_length,
        "dT_film_under_core": float(np.sum(T[film] * dA[film]) / np.sum(dA[film])),
        "dT_box_top": float(np.interp(y_box, yc, T[:, 0])), "dT_substrate_top": float(np.interp(H, yc, T[:, 0])),
        "tau_E": float(np.sum(rc * T * dA) / (heat_per_length / 2)),
        "heat_balance": (q_sink + (float(np.sum(T[-1, :] * np.diff(xe) / (np.diff(ye)[-1] / (2 * k[-1, :]) + 1 / h_top))) if h_top > 0 else 0.0)) / (heat_per_length / 2),
        "n_cells": int(T.size),
    }
    units = {"dT_core": "K", "dT_max": "K", "R_th": "K m/W", "dT_film_under_core": "K", "dT_box_top": "K",
             "dT_substrate_top": "K", "tau_E": "s", "heat_balance": "", "n_cells": ""}
    if return_field:
        vals.update(x_edges=xe, y_edges=ye, dT=T)
        units.update(x_edges="m", y_edges="m", dT="K")
    return Result(values=vals, units=units, assumptions=[
        "2D steady state, waveguide long and uniformly heated along z; heat deposited uniformly in the ridge",
        "Isothermal substrate bottom, convective top (h_top), adiabatic sides at ±domain_half_width",
        "Thermal conductivities from the thermo_optic LUT (room temperature, temperature independent) unless overridden",
        f"k used (W/m/K): core {kc:g}, film {ks:g}, BOX {kb:g}, substrate {ksub:g}, cladding {kcl:g}",
        "heat_balance: (heat into sink + through top) / heat generated; should be 1",
    ])


def thermal_shift(heat_per_length=1.0, dneff_dT=1.8e-4, wavelength=1.55e-6, n_group=4.2, length=1e-3, **geometry) -> Result:
    """ridge_heating plus the optical consequences: Δn_eff = dn_eff/dT ΔT_core, the phase over `length`
    and a ring/grating resonance shift Δλ = λ Δn_eff / n_g (expansion neglected)."""
    require_positive(wavelength=wavelength, n_group=n_group, length=length)
    r = ridge_heating(heat_per_length=heat_per_length, **geometry)
    dn = dneff_dT * r["dT_core"]
    v = dict(r.values)
    v.update(dneff=dn, phase=2 * np.pi * dn * length / wavelength, delta_lambda=wavelength * dn / n_group)
    u = dict(r.units)
    u.update(dneff="", phase="rad", delta_lambda="m")
    return Result(values=v, units=u, assumptions=r.assumptions + ["Mode temperature taken as the mean over the ridge"])
