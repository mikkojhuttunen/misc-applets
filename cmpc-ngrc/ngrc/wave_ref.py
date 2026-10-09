"""Wave-optics reference for the ray/beamlet engine: one pass of a Gaussian beam through a Δn dot.

`bpm`         split-step Fourier beam propagation (paraxial scalar wave equation, exact for the small angles and
              Δn here): ∂E/∂z = i/(2k) ∂²E/∂y² + i k0 Δn(z, y) E, k = k0 n0, carrier exp(i k z) removed.
`beamlet_sum` the same beam as a Gaussian-beam summation: parallel beamlets on a grid across the beam, traced
              through the dot with the engine's RK4 ray + Q/P system (mode "curved"), or with straight rays that
              only pick up ∫Δn ds (mode "straight" = the phase-screen model), summed at the observation plane.
Coordinates: z along the beam = engine x, transverse y = engine y.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .field import beamlet_matrix
from .rays import Exits, _rk4


def bpm(dn_fn, n0, wavelength, y, z_end, E0, dz=1e-6, absorb=0.08):
    """Field envelope at z_end. dn_fn(z, y) → Δn on the y grid; absorbing super-Gaussian edges (fraction absorb
    of the window on each side)."""
    k0 = 2 * np.pi / wavelength
    k = k0 * n0
    ny, dy = len(y), y[1] - y[0]
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    half = np.exp(-1j * ky**2 * dz / (4 * k))
    L = y[-1] - y[0]
    edge = absorb * L
    d = np.minimum(y - y[0], y[-1] - y)
    mask = np.where(d < edge, np.exp(-((edge - d) / (0.4 * edge)) ** 4), 1.0)
    E = np.asarray(E0, complex).copy()
    nz = int(round(z_end / dz))
    for i in range(nz):
        E = np.fft.ifft(half * np.fft.fft(E))
        E = E * np.exp(1j * k0 * dn_fn((i + 0.5) * dz, y) * dz)
        E = np.fft.ifft(half * np.fft.fft(E)) * mask
    return E


def beamlet_sum(pert, n0, wavelength, beam_w, z_end, y_obs, mode="curved", spacing=None, waist=None, ds=1.5e-6,
                span=2.5):
    """Field at (z_end, y_obs) of a Gaussian beam (1/e field half-width beam_w, waist at z = 0) built from parallel
    beamlets (waist `waist`, grid `spacing`) traced through `pert` (dots between z = 0 and z_end)."""
    k0 = 2 * np.pi / wavelength
    waist = beam_w / 6 if waist is None else waist
    spacing = waist / 1.5 if spacing is None else spacing
    y0 = np.arange(-span * beam_w, span * beam_w + spacing / 2, spacing)
    N = len(y0)
    amp = np.exp(-(y0 / beam_w) ** 2) * spacing / (np.sqrt(np.pi) * waist)
    zR = k0 * n0 * waist**2 / 2
    x, y = np.zeros(N), y0.copy()
    px, py = np.full(N, n0), np.zeros(N)
    L, Q, P = np.zeros(N), np.ones(N, complex), np.full(N, 1j * n0 / zR)
    argQ = np.zeros(N)

    def free(m, s):
        nonlocal Q, argQ
        x[m] += s * px[m] / np.hypot(px[m], py[m])
        y[m] += s * py[m] / np.hypot(px[m], py[m])
        L[m] += n0 * s
        Qn = Q[m] + P[m] * s / n0
        argQ[m] += np.angle(Qn / Q[m])
        Q[m] = Qn

    reg = pert.regions
    if len(reg):
        x_in = np.min(reg[:, 0] - reg[:, 2])
        x_out = np.max(reg[:, 0] + reg[:, 2])
        free(np.arange(N), np.full(N, x_in))
        m = mode if mode in ("curved", "straight") else "none"
        while np.any(x < x_out):
            act = np.nonzero(x < x_out)[0]
            s = _rk4(pert, n0, m, ds, x[act], y[act], px[act], py[act], L[act], Q[act], P[act])
            x[act], y[act], px[act], py[act], L[act] = s[0], s[1], s[2], s[3], s[4]
            argQ[act] += np.angle(s[5] / Q[act])
            Q[act], P[act] = s[5], s[6]
    pn = np.hypot(px, py)
    tx, ty = px / pn, py / pn
    free(np.arange(N), (z_end - x) / tx)
    ex = Exits(port=np.zeros(N, int), ray=np.arange(N), x=x, y=y, tx=tx, ty=ty, L=L, Q=Q, P=P, argQ=argQ,
               amp=amp, phase=np.zeros(N), nb=np.zeros(N, int))
    cell = SimpleNamespace(n_eff=n0, k0=k0)
    pts = np.c_[np.full_like(y_obs, z_end), y_obs]
    G, _ = beamlet_matrix(cell, ex, 0, pts)
    return G.sum(axis=1) * np.exp(-1j * k0 * n0 * z_end)


def compare(pert, n0=1.8, wavelength=1.55e-6, beam_w=60e-6, z_end=5e-3, y_half=None, ny=4096, window=1.2e-3,
            dz=1e-6):
    """Curved-beamlet and phase-screen (straight) sums against BPM at z_end, over |y| < y_half (default
    2.5 beam_w). corr_*: field correlation of the total field; scat_*: of the scattered field
    ΔE = E(dots) − E(no dots), the part that carries the dot's information; err_*: |ΔE_model − ΔE_bpm| / |ΔE_bpm|."""
    from .field import field_correlation
    from .shapes import Perturbation
    y = (np.arange(ny) - ny // 2) * (window / ny)
    E0 = np.exp(-(y / beam_w) ** 2)
    y_half = 2.5 * beam_w if y_half is None else y_half
    sel = np.abs(y) < y_half
    out = dict(y=y[sel])
    for tag, p in (("free", Perturbation([])), ("dot", pert)):
        Eb = bpm(lambda z, yy: p.value(np.full_like(yy, z), yy), n0, wavelength, y, z_end, E0, dz=dz)[sel]
        out[tag] = dict(bpm=Eb, curved=beamlet_sum(p, n0, wavelength, beam_w, z_end, y[sel], "curved"),
                        straight=beamlet_sum(p, n0, wavelength, beam_w, z_end, y[sel], "straight"))
    f, d = out["free"], out["dot"]
    dB = d["bpm"] - f["bpm"]
    for m in ("curved", "straight"):
        out[f"corr_{m}"] = field_correlation(d["bpm"], d[m])
        dM = d[m] - f[m]
        out[f"scat_{m}"] = field_correlation(dB, dM)
        out[f"err_{m}"] = float(np.linalg.norm(dM - dB) / np.linalg.norm(dB))
    out["scat_fraction"] = float(np.linalg.norm(dB) / np.linalg.norm(d["bpm"]))
    return out
