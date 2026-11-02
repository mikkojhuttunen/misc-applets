"""Trench-DBR mirrors of a free-standing membrane cell, in the effective-index picture (design_calcs/tmm.py,
vectorised and extended).

The guided TM0 slab mode (E mostly normal to the membrane) meets the etched mirror in the plane; for this in-plane
reflection its field is s-polarised (E perpendicular to the plane of incidence). Layers: slab (n_eff) | gap, tooth,
gap, tooth ... | air, gaps of index 1 (etched through), teeth of index n_eff (unetched membrane). Angles: chi from the
wall normal inside the slab, s = sin chi; the transverse wavenumber n_eff s is conserved, so the gaps are evanescent
above the critical angle s > 1/n_eff and the mirror reflects totally, apart from tunnelling through the first gap
(frustrated TIR; the transfer matrix includes it).

Extras, both optional and per bounce:
  * roughness: R ← R exp(−(4π n_eff σ cos chi / λ)²) (Rayleigh total integrated scatter of the sidewalls, σ rms);
  * vertical diffraction in the air gaps (estimate, eta=...): the mode is unguided across a gap and couples back into
    the next tooth with the overlap η(z) of the TM0 profile with itself propagated over z = d / cos chi_air in free
    space (exact scalar angular spectrum, gap_coupling; no paraxial approximation: the membrane modes are only
    ~λ/5 wide). Assumption: every traversal of a propagating gap multiplies the field by √η (the diffracted part is
    lost, never re-coupled), built into the transfer matrix as an imaginary phase i ln(1/√η) of each gap, so multiple
    reflections are counted consistently; evanescent gaps (above the critical angle) are not charged. This treats each
    gap as isolated and is a pessimistic bound: a first-order mirror (period < λ/2) carries a stop-band Bloch mode
    below the light line, which does not radiate inside the mirror, so the real loss sits at the slab-to-mirror
    transition and needs a vertical 2D (FDTD / mode-matching) calculation.
Units: thicknesses in wavelengths (vacuum λ = 1)."""
from __future__ import annotations

import math

import numpy as np


def _cz(n, kx):
    v = n * n - kx * kx + 0j
    c = np.sqrt(v)
    return np.where(c.imag < 0, -c, c)


def stack(n_slab, d_gap, d_tooth, N):
    """Layer list [(n, d), ...] for N periods (gap, tooth) and a closing gap; d_gap / d_tooth scalars or per-period
    arrays (chirp, two-band designs)."""
    dg = np.broadcast_to(np.asarray(d_gap, float), (N,))
    dt = np.broadcast_to(np.asarray(d_tooth, float), (N,))
    lay = []
    for i in range(N):
        lay += [(1.0, float(dg[i])), (n_slab, float(dt[i]))]
    lay.append((1.0, float(dg[-1]) if N else 0.25))
    return lay


def refl(n_slab, layers, s, lam=1.0, fields=False, eta=None, n_exit=1.0):
    """|r|² of the stack for s = sin chi in the slab (array). eta: callable η(z) (z in the units of lam) to charge the
    vertical diffraction of the propagating air gaps (module docstring). fields=True also returns the forward and
    backward amplitudes (a, b) and c = n cos(theta) in every layer, normalised to unit incident amplitude. n_exit: the
    medium behind the stack (air for an etched-through mirror; n_slab if the membrane continued behind it)."""
    s = np.atleast_1d(np.asarray(s, float))
    kx = n_slab * s
    k0 = 2 * np.pi / lam
    m11, m12, m21, m22 = (np.ones_like(kx, complex), np.zeros_like(kx, complex), np.zeros_like(kx, complex), np.ones_like(kx, complex))
    mats = []
    for n, d in layers:
        c = _cz(n, kx)
        ph = k0 * c * d
        if eta is not None and n == 1.0:
            prop = np.abs(c.imag) < 1e-12
            z = d / np.maximum(np.where(prop, c.real, 1.0), 1e-3)
            et = np.clip(eta(z), 1e-12, 1.0)
            ph = ph + np.where(prop, 1j * 0.5 * np.log(1 / et), 0.0)
        cs, sn = np.cos(ph), np.sin(ph)
        cc = np.where(np.abs(c) < 1e-14, 1e-14, c)
        a11, a12, a21, a22 = cs, -1j * sn / cc, -1j * cc * sn, cs
        mats.append((c, a11, a12, a21, a22))
        m11, m12, m21, m22 = m11 * a11 + m12 * a21, m11 * a12 + m12 * a22, m21 * a11 + m22 * a21, m21 * a12 + m22 * a22
        sc = np.maximum.reduce([np.abs(m11), np.abs(m12), np.abs(m21), np.abs(m22)])   # r is homogeneous in M:
        sc = np.where(sc > 1e100, sc, 1.0)                                             # rescale before overflow
        m11, m12, m21, m22 = m11 / sc, m12 / sc, m21 / sc, m22 / sc
    c0, ce = _cz(n_slab, kx), _cz(n_exit, kx)
    B = m11 + m12 * ce
    C = m21 + m22 * ce
    r = (c0 * B - C) / (c0 * B + C)
    R = np.clip(np.abs(r) ** 2, 0.0, 1.0)
    if not fields:
        return R
    E, H = 1 + r, c0 * (1 - r)                     # tangential E and H at the slab interface (s-pol, H in units of c)
    out = []
    for (c, a11, a12, a21, a22) in mats:
        cc = np.where(np.abs(c) < 1e-14, 1e-14, c)
        out.append((c, 0.5 * (E + H / cc), 0.5 * (E - H / cc)))
        E, H = a22 * E - a12 * H, -a21 * E + a11 * H     # inverse of a unimodular characteristic matrix
    return R, out, c0


def quarter(n_slab, s_design, m_gap=1, m_tooth=1):
    """Gap and tooth thicknesses (in λ) of odd quarter-wave orders at the design angle (sin chi = s_design)."""
    kx = n_slab * s_design
    cg = math.sqrt(max(1 - kx * kx, 1e-6))
    ct = math.sqrt(n_slab**2 - kx * kx)
    return m_gap / (4 * cg), m_tooth / (4 * ct)


def roughness_factor(n_slab, s, sigma, lam):
    cosc = np.sqrt(np.clip(1 - np.asarray(s, float) ** 2, 0, 1))
    return np.exp(-(4 * np.pi * n_slab * sigma * cosc / lam) ** 2)


def gaussian_waist(n_core, t, lam, pol="TM"):
    """1/e² intensity half-width w0 = 2 σ of the Gaussian with the same rms width σ as the slab mode's power flux
    (S_z ∝ H_y²/ε for TM0, E_y² for TE0): the TM0 mode of a thin membrane carries most of its power in the air, where
    E_x is ε times larger than inside. Returns (w0, σ, n_eff)."""
    from design_calcs import slab
    ne = slab.tm0(n_core, 1.0, t, lam) if pol == "TM" else slab.te0(n_core, 1.0, t, lam)
    k0 = 2 * np.pi / lam
    h, g = k0 * math.sqrt(n_core**2 - ne**2), k0 * math.sqrt(ne**2 - 1)
    x = np.linspace(-t / 2 - 15 / g, t / 2 + 15 / g, 20001)
    inside = np.abs(x) <= t / 2
    f = np.where(inside, np.cos(h * x), np.cos(h * t / 2) * np.exp(-g * (np.abs(x) - t / 2)))
    S = f * f / np.where(inside, n_core**2, 1.0) if pol == "TM" else f * f
    sig = math.sqrt(np.trapezoid(S * x * x, x) / np.trapezoid(S, x))
    return 2 * sig, sig, float(ne)


def gap_coupling(n_core, t, lam, pol="TM", npts=2**14):
    """η(z): power overlap of the slab mode with itself after free-space propagation over z (scalar angular spectrum of
    the transverse E profile: H_y/ε for TM0, E_y for TE0). Returns a callable of z (same units as lam), tabulated."""
    from design_calcs import slab
    ne = slab.tm0(n_core, 1.0, t, lam) if pol == "TM" else slab.te0(n_core, 1.0, t, lam)
    k0 = 2 * np.pi / lam
    h, g = k0 * math.sqrt(n_core**2 - ne**2), k0 * math.sqrt(ne**2 - 1)
    L = t / 2 + 40 / g
    x = np.linspace(-L, L, npts, endpoint=False)
    ins = np.abs(x) <= t / 2
    f = np.where(ins, np.cos(h * x), math.cos(h * t / 2) * np.exp(-g * (np.abs(x) - t / 2)))
    if pol == "TM":
        f = f / np.where(ins, n_core**2, 1.0)
    F = np.fft.fft(f)
    kx = 2 * np.pi * np.fft.fftfreq(x.size, x[1] - x[0])
    kz = np.sqrt(k0**2 - kx**2 + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)
    nf = np.vdot(f, f).real
    zs = np.linspace(0, 6 * lam, 241)
    et = np.array([abs(np.vdot(f, np.fft.ifft(F * np.exp(1j * kz * z)))) ** 2 / nf**2 for z in zs])
    return lambda z: np.interp(np.asarray(z, float), zs, et)


def design_R(n_slab, s, N, s1, m_gap=1, m_tooth=1, chirp=0.0, s2=None, sigma=0.0, lam=1.0, eta=None):
    """R(s) of a design: N periods quarter-wave at s1 (the second half at s2 for a two-band design), thicknesses
    scaled linearly by 1 + chirp (i/(N−1) − ½) along the stack; roughness sigma and the gap diffraction eta(z) optional (sigma, z in
    the units of lam)."""
    dg1, dt1 = quarter(n_slab, s1, m_gap, m_tooth)
    dg, dt = np.full(N, dg1), np.full(N, dt1)
    if s2 is not None and N > 1:
        dg2, dt2 = quarter(n_slab, s2, m_gap, m_tooth)
        dg[N // 2:], dt[N // 2:] = dg2, dt2
    if chirp and N > 1:
        f = 1 + chirp * (np.arange(N) / (N - 1) - 0.5)
        dg, dt = dg * f, dt * f
    lay = stack(n_slab, dg * lam, dt * lam, N)
    R = refl(n_slab, lay, s, lam, eta=eta)
    if sigma:
        R = R * roughness_factor(n_slab, s, sigma, lam)
    return R, dg, dt


def bounce_average(R, p):
    """Bounce-averaged R and the effective (geometric-mean) R that sets the transmission, for weights p."""
    p = np.asarray(p, float) / np.sum(p)
    return float(np.sum(p * R)), float(math.exp(np.sum(p * np.log(np.maximum(R, 1e-300)))))


def period(n_slab, d_gap, d_tooth):
    """Mean period of the stack in λ; below ½ (first order) the stop-band Bloch mode lies under the light line."""
    return float(np.mean(np.asarray(d_gap)) + np.mean(np.asarray(d_tooth)))


def optimise(n_slab, s, p, N, sigma=0.0, lam=1.0, eta=None, chirps=(0.0, -0.1, 0.1, -0.2, 0.2), two_band=True, n_angle=10,
             orders=(1, 3, 5)):
    """Best design for N periods: odd quarter-wave orders of gap and tooth (1, 3, 5), design angle on a grid below the
    critical angle, optional linear chirp and two-band (half the periods at a second design angle) variants, scored by
    the bounce-averaged R over the weights p(s)."""
    smax = 0.95 / n_slab
    angles = np.linspace(0, smax, n_angle)
    best = None
    for mg in orders:
        for mt in orders:
            for s1 in angles:
                for ch in chirps:
                    R, dg, dt = design_R(n_slab, s, N, s1, mg, mt, ch, None, sigma, lam, eta)
                    v = bounce_average(R, p)
                    if best is None or v[0] > best["R_avg"]:
                        best = dict(R_avg=v[0], R_eff=v[1], N=N, m_gap=mg, m_tooth=mt, s1=float(s1), s2=None, chirp=ch,
                                    d_gap=dg.tolist(), d_tooth=dt.tolist())
    if two_band and N >= 4:
        mg, mt, ch = best["m_gap"], best["m_tooth"], best["chirp"]
        for s1 in angles:
            for s2 in angles:
                if s2 <= s1:
                    continue
                R, dg, dt = design_R(n_slab, s, N, s1, mg, mt, ch, s2, sigma, lam, eta)
                v = bounce_average(R, p)
                if v[0] > best["R_avg"]:
                    best = dict(R_avg=v[0], R_eff=v[1], N=N, m_gap=mg, m_tooth=mt, s1=float(s1), s2=float(s2), chirp=ch,
                                d_gap=dg.tolist(), d_tooth=dt.tolist())
    return best
