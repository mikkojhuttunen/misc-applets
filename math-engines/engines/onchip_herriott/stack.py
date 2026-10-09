"""Vertical layer stack of an on-chip cell and its fundamental slab mode.

Stack, bottom to top: substrate (semi-infinite: SiO2, or air for a membrane) | platform film (Si, Si3N4, amorphous
Al2O3, LiNbO3 or LiTaO3 thin film) | optional thin Er:Al2O3 layer | cladding (semi-infinite: air or SiO2). The light
is guided in this slab and travels in plane, so the in-plane cell sees the slab's n_eff.

slab_mode solves the TE or TM fundamental mode by transfer through the layers (field and its scaled derivative are
continuous; for TM the derivative is divided by n²), roots bracketed on a fine n_eff grid and refined by bisection.
From the field f (E_y for TE, H_y for TM) it returns the power-weighted profile S = f² (TE) or f²/n² (TM, Poynting
∝ H_y²/n²), the share Γ_i of S in every layer (claddings included), and the effective height h_eff = ∫S dz / max S.
Materials: the Sellmeier / Cauchy tables of waveguide-core (λ in µm). SI units outside.
"""
from __future__ import annotations

import math

import numpy as np

# n² = A + Σ B λ²/(λ² − C) (λ, √C in µm), or Cauchy n = c0 + c1/λ² + c2/λ⁴; same tables as waveguide-core.js
_SELL = {
    "sio2": [1, 0.6961663, 0.0684043**2, 0.4079426, 0.1162414**2, 0.8974794, 9.896161**2],
    "si3n4": [1, 3.0249, 0.1353406**2, 40314, 1239.842**2, 0, 0],
    "ln_e": [1, 2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08],
    "ln_o": [1, 2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6],
    "lt_e": [1, 3.49474, 0.02789, 0, 0, 1.5, 150],
    "lt_o": [1, 3.47785, 0.02797, 0, 0, 1.5, 150],
    "si": [1, 10.6684293, 0.301516485**2, 0.0030434748, 1.13475115**2, 1.54133408, 1104**2],
}
_CAUCHY = {"al2o3": [1.646, 0.00962, 0.0], "eral2o3": [1.646, 0.00962, 0.0]}
PLATFORMS = {"al2o3": {"TE": "al2o3", "TM": "al2o3"}, "si3n4": {"TE": "si3n4", "TM": "si3n4"},
             "tfln": {"TE": "ln_e", "TM": "ln_o"}, "tflt": {"TE": "lt_e", "TM": "lt_o"}, "si": {"TE": "si", "TM": "si"}}
N_SAMPLE = 120        # field samples per layer


def n_material(mat, wavelength):
    """Refractive index of a named material at the vacuum wavelength [m]; 'air' = 1."""
    if mat == "air":
        return 1.0
    lam = wavelength * 1e6
    if mat in _CAUCHY:
        c0, c1, c2 = _CAUCHY[mat]
        l2 = lam * lam
        return c0 + c1 / l2 + c2 / (l2 * l2)
    s = _SELL[mat]
    l2, n2 = lam * lam, s[0]
    for k in (1, 3, 5):
        if s[k]:
            n2 += s[k] * l2 / (l2 - s[k + 1])
    return math.sqrt(n2)


def layer_stack(platform="al2o3", film_t=0.4e-6, er_t=0.0, substrate="sio2", cladding="air", wavelength=1.55e-6,
                polarization="TE"):
    """(indices bottom to top incl. the two semi-infinite media, interior thicknesses, layer names)."""
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {tuple(PLATFORMS)}")
    names = [substrate, PLATFORMS[platform][polarization]]
    t = [film_t]
    if er_t > 0:
        names.append("eral2o3")
        t.append(er_t)
    names.append(cladding)
    return [n_material(m, wavelength) for m in names], t, ["substrate", "film"] + (["er"] if er_t > 0 else []) + ["cladding"]


def _transfer(n, t, ne, k, tm):
    """Field f and D = f'/p (p = n² for TM, 1 for TE) at the top of the stack for a decaying start in the substrate,
    the mismatch with a decaying cladding field, and the per-layer start values (for sampling)."""
    p = (lambda x: x * x) if tm else (lambda x: 1.0)
    b2 = (k * ne) ** 2
    gs = math.sqrt(max(b2 - (k * n[0]) ** 2, 0.0))
    E, D = 1.0, gs / p(n[0])
    starts = []
    for i, d in enumerate(t, start=1):
        starts.append((E, D))
        q, P = (k * n[i]) ** 2 - b2, p(n[i])
        if q > 0:
            kk = math.sqrt(q)
            E, D = E * math.cos(kk * d) + P * D / kk * math.sin(kk * d), (-E * kk * math.sin(kk * d) + P * D * math.cos(kk * d)) / P
        else:
            g = math.sqrt(-q) or 1e-30
            E, D = E * math.cosh(g * d) + P * D / g * math.sinh(g * d), (E * g * math.sinh(g * d) + P * D * math.cosh(g * d)) / P
        s = abs(E) + abs(D)
        if s > 1e100:
            E, D = E / s, D / s
            starts = [(a / s, b / s) for a, b in starts]
    gc = math.sqrt(max(b2 - (k * n[-1]) ** 2, 0.0))
    return p(n[-1]) * D + gc * E, E, starts, gs, gc


def slab_mode(indices, thicknesses, wavelength, polarization="TE", n_grid=4000):
    """Fundamental guided mode: dict(neff, z, S (power profile, max 1), gamma (share per layer, bottom to top incl.
    the semi-infinite media), h_eff, gs, gc (decay constants)). NaN neff below cut-off."""
    n, t = [float(v) for v in indices], [float(v) for v in thicknesses]
    tm = polarization == "TM"
    k = 2 * math.pi / wavelength
    lo, hi = max(n[0], n[-1]), max(n)
    root = math.nan
    prev_ne, prev_f = None, None
    for i in range(n_grid - 1, 0, -1):                         # from the top: the first root is the fundamental mode
        ne = lo + (hi - lo) * i / n_grid
        f = _transfer(n, t, ne, k, tm)[0]
        if prev_f is not None and math.isfinite(f) and (f > 0) != (prev_f > 0):
            a, b, fa = ne, prev_ne, f
            for _ in range(80):
                mid = 0.5 * (a + b)
                fm = _transfer(n, t, mid, k, tm)[0]
                if (fm > 0) == (fa > 0):
                    a, fa = mid, fm
                else:
                    b = mid
            root = 0.5 * (a + b)
            break
        prev_ne, prev_f = ne, f
    if not math.isfinite(root):
        return dict(neff=math.nan, z=np.array([]), S=np.array([]), gamma=[math.nan] * (len(t) + 2), h_eff=math.nan, gs=math.nan, gc=math.nan)
    _, Etop, starts, gs, gc = _transfer(n, t, root, k, tm)
    p = (lambda x: x * x) if tm else (lambda x: 1.0)
    b2 = (k * root) ** 2
    zs, fs, ws, lay = [], [], [], []
    tail = 6.0
    # substrate tail: f = e^{gs z}, z < 0
    zz = -tail / gs * (1 - (np.arange(N_SAMPLE) + 0.5) / N_SAMPLE)
    zs.append(zz); fs.append(np.exp(gs * zz)); ws.append(np.full(N_SAMPLE, tail / gs / N_SAMPLE)); lay.append(np.zeros(N_SAMPLE, int))
    z0 = 0.0
    for i, d in enumerate(t, start=1):
        E, D = starts[i - 1]
        q, P = (k * n[i]) ** 2 - b2, p(n[i])
        x = d * (np.arange(N_SAMPLE) + 0.5) / N_SAMPLE
        if q > 0:
            kk = math.sqrt(q)
            f = E * np.cos(kk * x) + P * D / kk * np.sin(kk * x)
        else:
            g = math.sqrt(-q) or 1e-30
            f = E * np.cosh(g * x) + P * D / g * np.sinh(g * x)
        zs.append(z0 + x); fs.append(f); ws.append(np.full(N_SAMPLE, d / N_SAMPLE)); lay.append(np.full(N_SAMPLE, i))
        z0 += d
    zz = tail / gc * (np.arange(N_SAMPLE) + 0.5) / N_SAMPLE
    zs.append(z0 + zz); fs.append(Etop * np.exp(-gc * zz)); ws.append(np.full(N_SAMPLE, tail / gc / N_SAMPLE)); lay.append(np.full(N_SAMPLE, len(t) + 1))
    z, f, w, L = (np.concatenate(a) for a in (zs, fs, ws, lay))
    nz = np.array(n)[L]
    S = f * f / (nz * nz if tm else 1.0)
    # exact tails: ∫_{-∞}^0 e^{2gs z} dz = 1/(2gs), cladding Etop²/(2gc), divided by n² for TM
    tot_core = [(S * w)[L == i].sum() for i in range(1, len(t) + 1)]
    sub = 1 / (2 * gs) / (n[0] ** 2 if tm else 1.0)
    clad = Etop * Etop / (2 * gc) / (n[-1] ** 2 if tm else 1.0)
    tot = sub + clad + sum(tot_core)
    gamma = [sub / tot] + [c / tot for c in tot_core] + [clad / tot]
    smax = S.max()
    return dict(neff=root, z=z, S=S / smax, gamma=gamma, h_eff=tot / smax, gs=gs, gc=gc)


def stack_mode(platform="al2o3", film_t=0.4e-6, er_t=0.0, substrate="sio2", cladding="air", wavelength=1.55e-6,
               polarization="TE"):
    """slab_mode of layer_stack(...), with named overlaps: gamma_substrate, gamma_film, gamma_er, gamma_cladding."""
    n, t, names = layer_stack(platform, film_t, er_t, substrate, cladding, wavelength, polarization)
    m = slab_mode(n, t, wavelength, polarization)
    for nm, g in zip(names, m["gamma"]):
        m[f"gamma_{nm}"] = g
    m.setdefault("gamma_er", 0.0)
    m["indices"], m["thicknesses"], m["names"] = n, t, names
    return m
