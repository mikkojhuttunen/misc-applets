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


def conduction_matrix(x_edges, y_edges, k, h_top=10.0):
    """Finite-volume conductance matrix A (W/(m K), sparse CSR) with A ΔT = heat per cell per length, and the
    conductances of the bottom faces to the sink. Bottom isothermal, top convective (h_top), sides adiabatic."""
    from scipy.sparse import coo_matrix

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
    return A, G_bot


def solve_heat(x_edges, y_edges, k, q, h_top=10.0):
    """Finite-volume solve. k, q: (ny, nx) arrays of conductivity (W/(m K)) and heat per volume (W/m^3).
    Bottom edge isothermal (ΔT = 0), top edge convective (h_top, W/(m^2 K)), left and right edges adiabatic.
    Returns ΔT (ny, nx) and the heat flow into the sink per unit length (W/m)."""
    from scipy.sparse.linalg import spsolve

    A, G_bot = conduction_matrix(x_edges, y_edges, k, h_top)
    dx, dy = np.diff(x_edges), np.diff(y_edges)
    T = spsolve(A, (q * dy[:, None] * dx[None, :]).ravel()).reshape(k.shape)
    return T, float(np.sum(G_bot * T[0, :]))


def _k_of(material, override=None):
    return float(override) if override is not None else _to.entry(material)["k"]


REGIONS = ("substrate", "box", "slab", "core", "clad")


def build_ridge(core_material="si", core_width=0.5e-6, core_height=0.22e-6, slab_material="si", slab_thickness=0.0,
                box_material="sio2", box_thickness=2e-6, substrate_material="si", substrate_thickness=500e-6,
                clad_material="sio2", clad_thickness=2e-6, domain_half_width=500e-6, k_core=None, k_slab=None, k_box=None,
                k_substrate=None, k_clad=None, resolution=1.0) -> dict:
    """Graded grid (half domain x >= 0), region map (index into REGIONS), materials, k and ρc_p per cell."""
    require_positive(core_width=core_width, core_height=core_height, box_thickness=box_thickness,
                     substrate_thickness=substrate_thickness, domain_half_width=domain_half_width, resolution=resolution)
    require_nonnegative(slab_thickness=slab_thickness, clad_thickness=clad_thickness)
    mats = dict(substrate=substrate_material, box=box_material, slab=slab_material, core=core_material, clad=clad_material)
    for name, m in mats.items():
        require_choice(f"{name} material", m, MATERIALS)
    if domain_half_width <= core_width / 2:
        raise ValueError("domain_half_width must exceed core_width / 2")
    kk = dict(substrate=_k_of(substrate_material, k_substrate), box=_k_of(box_material, k_box), slab=_k_of(slab_material, k_slab),
              core=_k_of(core_material, k_core), clad=_k_of(clad_material, k_clad))
    require_positive(**{f"k_{r}": v for r, v in kk.items()})
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
    ye = graded_axis([0.0, H, y_box, y_film, y_core, y_top], y_box, y_core, h_f, growth)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc)
    reg = np.full(X.shape, REGIONS.index("clad"))
    reg[Y < y_box] = REGIONS.index("box")
    reg[Y < H] = REGIONS.index("substrate")
    if ts > 0:
        reg[(Y > y_box) & (Y < y_film)] = REGIONS.index("slab")
    reg[(Y > y_film) & (Y < y_core) & (X < w2)] = REGIONS.index("core")
    # by region, so that a material used in two roles (Si core and Si substrate) keeps each region's k
    k = np.array([kk[r] for r in REGIONS])[reg]
    rc = np.array([_to.entry(mats[r])["rho"] * _to.entry(mats[r])["cp"] for r in REGIONS])[reg]
    return dict(xe=xe, ye=ye, xc=xc, yc=yc, X=X, Y=Y, reg=reg, k=k, rc=rc, mats=mats, kk=kk,
                dA=np.diff(ye)[:, None] * np.diff(xe)[None, :], core=reg == REGIONS.index("core"),
                levels=dict(H=H, y_box=y_box, y_film=y_film, y_core=y_core, y_top=y_top, w2=w2))


def _geometry(kwargs):
    keys = ("core_material", "core_width", "core_height", "slab_material", "slab_thickness", "box_material", "box_thickness",
            "substrate_material", "substrate_thickness", "clad_material", "clad_thickness", "domain_half_width", "k_core",
            "k_slab", "k_box", "k_substrate", "k_clad", "resolution")
    return {k: v for k, v in kwargs.items() if k in keys}


def _heat_map(g, heat_per_length, weight=None):
    """Heat per volume (W/m^3) on the half domain: uniform in the core, or ∝ weight inside the core."""
    w = np.where(g["core"], 1.0 if weight is None else weight, 0.0)
    return w * (heat_per_length / 2) / float(np.sum(w * g["dA"]))


def ridge_heating(heat_per_length=1.0, core_material="si", core_width=0.5e-6, core_height=0.22e-6,
                  slab_material="si", slab_thickness=0.0, box_material="sio2", box_thickness=2e-6,
                  substrate_material="si", substrate_thickness=500e-6, clad_material="sio2", clad_thickness=2e-6,
                  domain_half_width=500e-6, h_top=10.0, k_core=None, k_slab=None, k_box=None, k_substrate=None,
                  k_clad=None, resolution=1.0, return_field=False, heat_weight=None) -> Result:
    """Temperature rise of a heated ridge/strip waveguide on a layered substrate.

    Stack, bottom to top: substrate (H), buried layer (BOX), film of slab_material (slab_thickness, may be 0),
    ridge/strip of core_material (core_width x core_height) on the film; cladding fills the rest up to
    clad_thickness above the BOX (raised to the device height if smaller). heat_per_length q' (W/m) is
    deposited uniformly in the ridge (or ∝ heat_weight, an array on the grid, inside it). k_* override the LUT
    conductivities (thin-film values, worst cases)."""
    require_positive(heat_per_length=heat_per_length)
    require_nonnegative(h_top=h_top)
    g = build_ridge(**_geometry(locals()))
    q = _heat_map(g, heat_per_length, heat_weight)
    T, q_sink = solve_heat(g["xe"], g["ye"], g["k"], q, h_top)
    xe, ye, yc, dA, k, core = g["xe"], g["ye"], g["yc"], g["dA"], g["k"], g["core"]
    lv = g["levels"]
    A_half = float(np.sum(dA[core]))
    T_core = float(np.sum(T[core] * dA[core]) / A_half)
    film = (g["reg"] == REGIONS.index("slab")) & (g["X"] < lv["w2"]) if slab_thickness > 0 else core
    q_top = float(np.sum(T[-1, :] * np.diff(xe) / (np.diff(ye)[-1] / (2 * k[-1, :]) + 1 / h_top))) if h_top > 0 else 0.0
    vals = {
        "dT_core": T_core, "dT_max": float(T.max()), "R_th": T_core / heat_per_length,
        "dT_film_under_core": float(np.sum(T[film] * dA[film]) / np.sum(dA[film])),
        "dT_box_top": float(np.interp(lv["y_box"], yc, T[:, 0])), "dT_substrate_top": float(np.interp(lv["H"], yc, T[:, 0])),
        "tau_E": float(np.sum(g["rc"] * T * dA) / (heat_per_length / 2)),
        "heat_balance": (q_sink + q_top) / (heat_per_length / 2),
        "n_cells": int(T.size),
    }
    units = {"dT_core": "K", "dT_max": "K", "R_th": "K m/W", "dT_film_under_core": "K", "dT_box_top": "K",
             "dT_substrate_top": "K", "tau_E": "s", "heat_balance": "", "n_cells": ""}
    if return_field:
        vals.update(x_edges=xe, y_edges=ye, dT=T)
        units.update(x_edges="m", y_edges="m", dT="K")
    kk = g["kk"]
    return Result(values=vals, units=units, assumptions=[
        "2D steady state, waveguide long and uniformly heated along z; heat deposited in the ridge",
        "Isothermal substrate bottom, convective top (h_top), adiabatic sides at ±domain_half_width",
        "Thermal conductivities from the thermo_optic LUT (room temperature, temperature independent) unless overridden",
        f"k used (W/m/K): core {kk['core']:g}, film {kk['slab']:g}, BOX {kk['box']:g}, substrate {kk['substrate']:g}, cladding {kk['clad']:g}",
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


def ridge_dynamics(core_material="si", core_width=0.5e-6, core_height=0.22e-6, slab_material="si", slab_thickness=0.0,
                   box_material="sio2", box_thickness=2e-6, substrate_material="si", substrate_thickness=500e-6,
                   clad_material="sio2", clad_thickness=2e-6, domain_half_width=500e-6, h_top=10.0, k_core=None,
                   k_slab=None, k_box=None, k_substrate=None, k_clad=None, resolution=1.0, steps_per_doubling=8) -> Result:
    """Thermal dynamics of the ridge temperature (mean over the ridge) for heat switched on at t = 0:

        C dT/dt + A T = Q      (C: ρc_p per cell; A: conduction_matrix)

    Step response by backward Euler, time step doubling every steps_per_doubling steps (0.1 ns .. 10 τ_E), times to 10/50/90 % of the
    steady rise; frequency response H(f) = T̂_core(f)/T_core(0) from (A + i2πf C) T̂ = Q, and its -3 dB frequency.
    The response is not single-exponential: a fast part (the BOX, µs) and a slow tail (the substrate, ms)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve, splu

    require_nonnegative(h_top=h_top)
    g = build_ridge(**_geometry(locals()))
    A, _ = conduction_matrix(g["xe"], g["ye"], g["k"], h_top)
    A = A.tocsc()
    dA, core = g["dA"], g["core"]
    C = (g["rc"] * dA).ravel()
    Q = (_heat_map(g, 1.0) * dA).ravel()
    w = (np.where(core, dA, 0.0) / float(np.sum(dA[core]))).ravel()
    T0 = spsolve(A, Q)
    Tc0 = float(w @ T0)
    tau_E = float(C @ T0 / 0.5)

    def H(f):
        return float(abs(w @ spsolve((A + diags(2j * np.pi * f * C)).tocsc(), Q.astype(complex))) / Tc0)

    lo, hi = 1e-2, 1e9
    for _ in range(36):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if H(mid) > 1 / math.sqrt(2) else (lo, mid)
    f3 = math.sqrt(lo * hi)
    t_end = 10 * tau_E
    T = np.zeros_like(Q)
    t, dt, ts, out = 0.0, 1e-10, [], []
    while t < t_end:                       # blocks of steps_per_doubling steps, dt doubling between blocks
        lu = splu((A + diags(C / dt)).tocsc())
        for _ in range(int(steps_per_doubling)):
            T = lu.solve(Q + C / dt * T)
            t += dt
            ts.append(t)
            out.append(float(w @ T) / Tc0)
        dt *= 2
    ts = np.array(ts)
    out = np.array(out)

    def t_at(level):
        i = int(np.argmax(out >= level))
        if out[i] < level:
            return float("nan")
        if i == 0:
            return float(ts[0])
        return float(np.exp(np.interp(level, out[i - 1:i + 1], np.log(ts[i - 1:i + 1]))))

    return Result(
        values={"t_10": t_at(0.1), "t_50": t_at(0.5), "t_90": t_at(0.9), "f_3dB": f3, "tau_E": tau_E,
                "H_1kHz": H(1e3), "H_1MHz": H(1e6), "t": ts, "step": out},
        units={"t_10": "s", "t_50": "s", "t_90": "s", "f_3dB": "Hz", "tau_E": "s", "H_1kHz": "", "H_1MHz": "", "t": "s", "step": ""},
        assumptions=["2D (uniform along z), linear, temperature-independent properties; heat uniform in the ridge",
                     "Backward-Euler step response on geometric time steps (first-order accurate; t_x within a few %)",
                     "Frequency response exact for the discretised model"],
    )


def ridge_mode(wavelength=1.55e-6, core_material="si", core_width=0.5e-6, core_height=0.22e-6, slab_material="si",
               slab_thickness=0.0, box_material="sio2", box_thickness=2e-6, clad_material="sio2", clad_thickness=2e-6,
               margin=1.5e-6, cells_per_wavelength=24, al_fraction=0.20, indices=None, h=None):
    """Fundamental scalar (Helmholtz) mode of the ridge on a uniform grid in a window around it (field 0 at the
    window edge; the substrate is outside the window, i.e. leakage into it is neglected).

    Returns dict with x, y (cell centres, full width), E (normalised field), n map, region map, n_eff.
    indices: optional {region: n} override (otherwise thermo_optic.index_at for each region's material).
    h: grid step (default from cells_per_wavelength); keep it fixed when differencing over wavelength."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh

    mats = dict(box=box_material, slab=slab_material, core=core_material, clad=clad_material)
    n_of = {r: (indices[r] if indices and r in indices else _to.index_at(m, wavelength, al_fraction)) for r, m in mats.items()}
    w2, ts, hc = core_width / 2, slab_thickness, core_height
    y_film, y_core = ts, ts + hc                      # y = 0 at the BOX top
    n_max = max(n_of.values())
    if h is None:
        h = min(wavelength / (n_max * cells_per_wavelength), hc / 6, (ts / 3 if ts > 0 else hc / 6), w2 / 6)

    def axis(breaks):
        # uniform within each segment, every break on a cell edge
        e = [breaks[0]]
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            m_ = max(1, int(math.ceil((hi - lo) / h - 1e-9)))
            e.extend(lo + (hi - lo) * np.arange(1, m_ + 1) / m_)
        return np.array(e)

    xe = axis([-(w2 + margin), -w2, w2, w2 + margin])
    ye = axis([-min(margin, box_thickness)] + ([0.0, y_film] if ts > 0 else [0.0]) + [y_core, y_core + margin])
    x, y = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    dx, dy = np.diff(xe), np.diff(ye)
    hx, hy = float(dx.min()), float(dy.min())
    nx, ny = x.size, y.size
    X, Y = np.meshgrid(x, y)
    reg = np.full(X.shape, REGIONS.index("clad"))
    reg[Y < 0] = REGIONS.index("box")
    if ts > 0:
        reg[(Y > 0) & (Y < y_film)] = REGIONS.index("slab")
    reg[(Y > y_film) & (Y < y_core) & (np.abs(X) < w2)] = REGIONS.index("core")
    n = np.array([n_of.get(r, 1.0) for r in REGIONS])[reg]
    k0 = 2 * np.pi / wavelength
    # finite volumes: Σ_faces (face/dist)(E_j - E_i) + k² n² E_i V_i = β² V_i E_i, E = 0 half a cell outside the window
    idx = np.arange(nx * ny).reshape(ny, nx)
    V = (dy[:, None] * dx[None, :])
    Gx = dy[:, None] / (0.5 * (dx[None, :-1] + dx[None, 1:])) * np.ones((ny, 1))
    Gy = dx[None, :] / (0.5 * (dy[:-1, None] + dy[1:, None])) * np.ones((1, nx))
    diag = (k0 * n) ** 2 * V
    diag[:, :-1] -= Gx
    diag[:, 1:] -= Gx
    diag[:-1, :] -= Gy
    diag[1:, :] -= Gy
    diag[:, 0] -= dy / (0.5 * dx[0])
    diag[:, -1] -= dy / (0.5 * dx[-1])
    diag[0, :] -= dx / (0.5 * dy[0])
    diag[-1, :] -= dx / (0.5 * dy[-1])
    rows = [idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(), idx.ravel()]
    cols = [idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(), idx.ravel()]
    vals = [Gx.ravel(), Gx.ravel(), Gy.ravel(), Gy.ravel(), diag.ravel()]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2).tocsc()
    M = diags(V.ravel()).tocsc()
    ev, vecs = eigsh(A, k=2, M=M, sigma=(k0 * n_max) ** 2, which="LM")
    i = int(np.argmax(ev))
    E = vecs[:, i].reshape(ny, nx)
    E = E / np.sqrt(np.sum(E**2 * V))
    neff = math.sqrt(ev[i]) / k0
    return dict(x=x, y=y, X=X, Y=Y, E=E, n=n, reg=reg, n_of=n_of, neff=neff, hx=hx, hy=hy, h=h, V=V, mats=mats)


def mode_weighted_heating(heat_per_length=1.0, wavelength=1.55e-6, heat_in_mode=False, core_material="si",
                          core_width=0.5e-6, core_height=0.22e-6, slab_material="si", slab_thickness=0.0,
                          box_material="sio2", box_thickness=2e-6, substrate_material="si", substrate_thickness=500e-6,
                          clad_material="sio2", clad_thickness=2e-6, domain_half_width=500e-6, h_top=10.0,
                          margin=1.5e-6, al_fraction=0.20, cells_per_wavelength=24) -> Result:
    """Thermal index shift of the guided mode with its real field profile instead of a ridge-mean temperature.

    Scalar mode (ridge_mode): Γ_r = (n_r/n_eff) ∫_r E² / ∫E² = ∂n_eff/∂n_r (exact first-order perturbation of the
    scalar Helmholtz equation), uniform-temperature dn_eff/dT = Σ Γ_r dn_r/dT, and under the computed heating

        Δn_eff = ∫ n (dn/dT) ΔT E² dA / (n_eff ∫ E² dA),    ΔT_mode = Δn_eff / (dn_eff/dT)

    heat_in_mode: deposit the heat ∝ E² inside the ridge (absorption in the core) instead of uniformly."""
    from scipy.interpolate import RegularGridInterpolator

    require_positive(heat_per_length=heat_per_length, wavelength=wavelength, margin=margin)
    geo = dict(core_material=core_material, core_width=core_width, core_height=core_height, slab_material=slab_material,
               slab_thickness=slab_thickness, box_material=box_material, box_thickness=box_thickness, clad_material=clad_material,
               clad_thickness=clad_thickness)
    m = ridge_mode(wavelength, margin=margin, al_fraction=al_fraction, cells_per_wavelength=cells_per_wavelength, **geo)
    g = build_ridge(substrate_material=substrate_material, substrate_thickness=substrate_thickness,
                    domain_half_width=domain_half_width, **geo)
    y0 = g["levels"]["y_box"]
    xm = np.concatenate([-g["xc"][::-1], g["xc"]])
    weight = None
    E2 = m["E"] ** 2 * m["V"]          # E² dA per cell
    if heat_in_mode:
        fE = RegularGridInterpolator((m["y"] + y0, m["x"]), m["E"] ** 2, bounds_error=False, fill_value=0.0)
        weight = fE(np.stack([g["Y"].ravel(), g["X"].ravel()], axis=-1)).reshape(g["X"].shape)
    q = _heat_map(g, heat_per_length, weight)
    T, _ = solve_heat(g["xe"], g["ye"], g["k"], q, h_top)
    Tm = np.concatenate([T[:, ::-1], T], axis=1)
    fT = RegularGridInterpolator((g["yc"], xm), Tm, bounds_error=False, fill_value=None)
    Tmode = fT(np.stack([(m["Y"] + y0).ravel(), m["X"].ravel()], axis=-1)).reshape(m["X"].shape)
    dndT = np.array([_to.entry(m["mats"].get(r, "air"), al_fraction)["dn_dT"] if r in m["mats"] else 0.0 for r in REGIONS])[m["reg"]]
    neff, n = m["neff"], m["n"]
    tot = float(np.sum(E2))
    gam = {r: float(np.sum(np.where(m["reg"] == i, n * E2, 0.0)) / (neff * tot)) for i, r in enumerate(REGIONS)}
    dneff_dT = sum(gam[r] * _to.entry(m["mats"][r], al_fraction)["dn_dT"] for r in m["mats"])
    dneff = float(np.sum(n * dndT * Tmode * E2) / (neff * tot))
    core = g["core"]
    T_core = float(np.sum(T[core] * g["dA"][core]) / np.sum(g["dA"][core]))
    dl = wavelength * 0.01
    n_p = ridge_mode(wavelength + dl, margin=margin, al_fraction=al_fraction, h=m["h"], **geo)["neff"]
    n_m = ridge_mode(wavelength - dl, margin=margin, al_fraction=al_fraction, h=m["h"], **geo)["neff"]
    ng = neff - wavelength * (n_p - n_m) / (2 * dl)
    return Result(
        values={"neff": neff, "n_group": ng, "Gamma_core": gam["core"], "Gamma_slab": gam["slab"], "Gamma_box": gam["box"],
                "Gamma_clad": gam["clad"], "dneff_dT": dneff_dT, "dT_core": T_core, "dT_mode": dneff / dneff_dT if dneff_dT else float("nan"),
                "dneff": dneff, "R_th_mode": (dneff / dneff_dT) / heat_per_length if dneff_dT else float("nan"),
                "dlambda_dT": wavelength * dneff_dT / ng, "delta_lambda": wavelength * dneff / ng},
        units={"neff": "", "n_group": "", "Gamma_core": "", "Gamma_slab": "", "Gamma_box": "", "Gamma_clad": "", "dneff_dT": "1/K",
               "dT_core": "K", "dT_mode": "K", "dneff": "", "R_th_mode": "K m/W", "dlambda_dT": "m/K", "delta_lambda": "m"},
        assumptions=["Scalar (polarisation-averaged) fundamental mode; for high-contrast Si wires TE/TM Γ differ by up to ~10 %",
                     "Mode window excludes the substrate (no leakage) and is filled with the cladding above the BOX; n from thermo_optic.index_at",
                     "First-order perturbation in ΔT (no thermal lensing feedback on the mode shape)",
                     "dlambda_dT without substrate expansion; n_group from scalar-mode dispersion incl. material dispersion"],
    )
