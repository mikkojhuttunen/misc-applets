"""Parametric amplification in fiber: chi2 three-wave and chi3 degenerate four-wave mixing.

Conventions
-----------
* SI units. Powers in W, wavelengths are vacuum wavelengths, losses are power
  attenuation coefficients in 1/m (not dB), lumped dumps are power transmissions.
* chi2: w_p = w_s + w_i. Amplitudes are photon-flux normalised; the coupling is
  K = 2 d_eff sqrt(hbar w_p w_s w_i / (2 eps0 c^3 n^3)) * Theta with the Gaussian
  three-mode overlap Theta, and the gain coefficient is Gamma = K sqrt(Phi_p).
  ``phase_mismatch`` is dk = k_p - k_s - k_i (including any QPM grating term).
* chi3: 2 w_p = w_s + w_i. gamma_j = n2 w_j / (c A_eff), A_eff from
  f_p^2 f_s f_i. ``phase_mismatch`` is the net kappa = dbeta + 2 gamma_p P_p(0);
  Gamma = gamma_p P_p sqrt(w_s w_i) / w_p. SPM and XPM are included in the full solver.
* With an undepleted pump both processes reduce to the same 2x2 signal-idler system,
  solved exactly by ``small_signal_gain``; gain is symmetric in the sign of the mismatch.
* Idler loss (and optionally signal/pump loss from the same spectral filter) can be
  continuous or lumped at N equally spaced dumps (z = k L/(N+1)).
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive
from ..stimulated_scattering import engine as ss

C0 = 299_792_458.0          # m/s
HBAR = 1.054_571_817e-34     # J s
EPS0 = 8.854_187_8128e-12    # F/m

_PROCESSES = ("chi2", "chi3")


def idler_wavelength(lambda_signal, lambda_pump, process="chi2"):
    """Idler wavelength from energy conservation (helper)."""
    lam_sum = lambda_pump if process == "chi2" else lambda_pump / 2
    return 1 / (1 / lam_sum - 1 / np.asarray(lambda_signal, float))


def mode_overlap(w_pump, w_signal, w_idler, process="chi2"):
    """chi2: Theta (1/m) of three Gaussian fields; chi3: 1/A_eff (1/m^2) of f_p^2 f_s f_i (helper)."""
    wp, ws, wi = float(w_pump), float(w_signal), float(w_idler)
    if process == "chi3":
        return (math.pi / (2 / wp ** 2 + 1 / ws ** 2 + 1 / wi ** 2)) / ((math.pi / 2) ** 2 * wp * wp * ws * wi)
    s = 1 / wp ** 2 + 1 / ws ** 2 + 1 / wi ** 2
    return (math.pi / s) / math.sqrt((math.pi / 2) ** 3 * wp * wp * ws * ws * wi * wi)


def coupling(process, lambda_pump, lambda_signal, w_pump, w_signal, w_idler, pump_power,
             d_eff=0.0, n2=0.0, n=1.45) -> Result:
    """Small-signal gain coefficient Gamma and the quantities behind it."""
    require_choice("process", process, _PROCESSES)
    require_positive(lambda_pump=lambda_pump, lambda_signal=lambda_signal, w_pump=w_pump, w_signal=w_signal,
                     w_idler=w_idler, pump_power=pump_power)
    li = idler_wavelength(lambda_signal, lambda_pump, process)
    wP, wS, wI = (2 * math.pi * C0 / x for x in (lambda_pump, lambda_signal, li))
    th = mode_overlap(w_pump, w_signal, w_idler, process)
    phi_p = pump_power / (HBAR * wP)
    if process == "chi2":
        require_positive(d_eff=d_eff)
        K = 2 * d_eff * math.sqrt(HBAR * wP * wS * wI / (2 * EPS0 * C0 ** 3 * n ** 3)) * th
        gamma_p, gP, gamma_c = float("nan"), float("nan"), K * math.sqrt(phi_p)
        a_eff = 1 / th ** 2
    else:
        require_positive(n2=n2)
        gamma_p = n2 * wP / C0 * th
        gP = gamma_p * pump_power
        gamma_c = gP * math.sqrt(wS * wI) / wP
        K = gamma_c / math.sqrt(phi_p)
        a_eff = 1 / th
    return Result(
        values={"gain_coefficient": gamma_c, "K": K, "A_eff": a_eff, "lambda_idler": li, "gamma": gamma_p,
                "nonlinear_phase": 2 * gP if process == "chi3" else 0.0, "pump_photon_flux": phi_p},
        units={"gain_coefficient": "1/m", "K": "1/(m s^-1/2)", "A_eff": "m^2", "lambda_idler": "m", "gamma": "1/(W m)",
               "nonlinear_phase": "rad/m", "pump_photon_flux": "1/s"},
        assumptions=["Single transverse mode per wave, Gaussian profiles exp(-r^2/w^2)",
                     "chi2: n = %.2f for all three waves" % n, "chi3: gamma_j proportional to omega_j"],
    )


# ---------- undepleted-pump analytic solution ----------
def _expm2(kap, beta, z, a_s):
    """exp(M z) for M = [[a_s, i kap], [-i kap, beta]] (complex beta, real a_s); None on overflow."""
    tr, df = beta + a_s, beta - a_s
    disc = cmath.sqrt(df * df + 4 * kap * kap)
    if disc.real < 0:
        disc = -disc
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    if max(l1.real, l2.real) * z > 600:
        return None
    M = ((complex(a_s), 1j * kap), (-1j * kap, beta))
    if abs(disc) * z < 1e-7:
        l = tr / 2
        el = cmath.exp(l * z)
        return ((el * (1 + (M[0][0] - l) * z), el * M[0][1] * z), (el * M[1][0] * z, el * (1 + (M[1][1] - l) * z)))
    e1, e2 = cmath.exp(l1 * z), cmath.exp(l2 * z)
    E = [[0j, 0j], [0j, 0j]]
    for r in range(2):
        for q in range(2):
            a = M[r][q] - (l2 if r == q else 0)
            b = M[r][q] - (l1 if r == q else 0)
            E[r][q] = (e1 * a - e2 * b) / disc
    return E


def _gain_one(kap, dk, L, idler_loss, signal_loss, n_dumps, t_idler, t_signal):
    nseg = n_dumps + 1
    dz = L / nseg
    E = _expm2(kap, complex(-idler_loss / 2, dk), dz, -signal_loss / 2)
    if E is None:
        return math.inf, math.inf
    s, c = 1 + 0j, 0j
    for sg in range(nseg):
        s, c = E[0][0] * s + E[0][1] * c, E[1][0] * s + E[1][1] * c
        if sg < nseg - 1:
            c *= t_idler
            s *= t_signal
    return abs(s) ** 2, abs(c) ** 2


def small_signal_gain(gain_coefficient, phase_mismatch, length, idler_loss=0.0, signal_loss=0.0,
                      n_dumps=0, idler_dump_transmission=1.0, signal_dump_transmission=1.0) -> Result:
    """Exact undepleted-pump signal gain and idler photon conversion; vectorised over phase_mismatch."""
    require_nonnegative(gain_coefficient=gain_coefficient)
    require_positive(length=length)
    require_nonnegative(idler_loss=idler_loss)
    require_nonnegative(signal_loss=signal_loss)
    dk = np.asarray(phase_mismatch, float)
    tI, tS = math.sqrt(idler_dump_transmission), math.sqrt(signal_dump_transmission)
    pairs = [_gain_one(gain_coefficient, float(d), length, idler_loss, signal_loss, int(n_dumps), tI, tS) for d in dk.ravel()]
    G = np.array([p[0] for p in pairs]).reshape(dk.shape)
    I = np.array([p[1] for p in pairs]).reshape(dk.shape)
    if G.ndim == 0:
        G, I = float(G), float(I)
    with np.errstate(divide="ignore"):
        gdb = 10 * np.log10(G)
    return Result({"gain": G, "gain_db": gdb, "idler_conversion": I},
                  {"gain": "", "gain_db": "dB", "idler_conversion": ""},
                  ["Undepleted, lossless pump", "no idler seed (phase-insensitive gain)",
                   "idler_conversion = idler photons out / signal photons in"])


# ---------- full coupled-wave solution (pump depletion, SPM/XPM, scattering) ----------
def _scattering_setup(process, P0, L, wP, wS, wI, radii, sbs, srs, pump_linewidth):
    if not sbs and not srs:
        return None
    w = [wP, wS, wI]
    rad = list(radii)
    Ap = math.pi * rad[0] ** 2
    sc = {"P0": P0, "wt": [1, 1, 1] if process == "chi3" else [1, wS / wP, wI / wP], "pairs": [],
          "trackR": False, "rate": 0.0, "sol": None, "gA": 0.0}
    if srs:
        for a, b in ((0, 1), (0, 2), (1, 2)):
            j = a if w[a] > w[b] else b
            k = b if j == a else a
            g = float(ss.raman_peak_gain(2 * math.pi * C0 / w[j]) * ss.raman_shape(w[j] - w[k])) / (math.pi * (rad[j] ** 2 + rad[k] ** 2) / 2)
            sc["pairs"].append((j, k, g, w[j] / w[k]))
            sc["rate"] = max(sc["rate"], abs(g) * P0 * w[j] / w[k])
        wR = w[0] - ss.RAMAN_PEAK_SHIFT
        sc["gRp"] = float(ss.raman_peak_gain(2 * math.pi * C0 / w[0])) / Ap
        sc["ratioPR"] = w[0] / wR
        sc["trackR"] = True
        sc["x0R"] = (6.62607015e-34 * wR / (2 * math.pi) * 1e12) / P0
        sc["rate"] = max(sc["rate"], sc["gRp"] * P0)
    if sbs:
        br = ss.brillouin_parameters(2 * math.pi * C0 / w[0], pump_linewidth)
        gA = br["g_B"] / Ap
        sc["sol"] = ss.SbsSolution(P0, gA, L, br["seed_power"])
        sc["gA"] = gA
        sc["rate"] = max(sc["rate"], gA * sc["sol"].D)
    return sc


def _add_scattering(sc, z, Y, o):
    P0, wt = sc["P0"], sc["wt"]
    Pw = [P0 * wt[0] * (Y[0] ** 2 + Y[1] ** 2), P0 * wt[1] * (Y[2] ** 2 + Y[3] ** 2), P0 * wt[2] * (Y[4] ** 2 + Y[5] ** 2)]
    for j, k, c, r in sc["pairs"]:
        gk, gj = 0.5 * c * Pw[j], -0.5 * c * r * Pw[k]
        o[2 * k] += gk * Y[2 * k]
        o[2 * k + 1] += gk * Y[2 * k + 1]
        o[2 * j] += gj * Y[2 * j]
        o[2 * j + 1] += gj * Y[2 * j + 1]
    loss = sc["gA"] * sc["sol"].PB(z) if sc["sol"] else 0.0
    if sc["trackR"]:
        o[6] = sc["gRp"] * Pw[0] * Y[6]
        loss += sc["ratioPR"] * sc["gRp"] * P0 * Y[6]
    else:
        o[6] = 0.0
    o[0] -= 0.5 * loss * Y[0]
    o[1] -= 0.5 * loss * Y[1]


def amplify(process, lambda_pump, lambda_signal, pump_power, signal_power, length, gain_coefficient,
            phase_mismatch=0.0, idler_loss=0.0, signal_loss=0.0, pump_loss=0.0, n_dumps=0,
            idler_dump_transmission=1.0, signal_dump_transmission=1.0, pump_dump_transmission=1.0,
            sbs=False, srs=False, pump_linewidth=0.0, mode_radii=None, record=False, max_steps=400000) -> Result:
    """Integrate the coupled-wave equations with RK4 (pump depletion, SPM/XPM for chi3, optional SBS/SRS).

    ``mode_radii`` = (w_pump, w_signal, w_idler) in m is needed only when sbs or srs is on.
    """
    require_choice("process", process, _PROCESSES)
    require_positive(lambda_pump=lambda_pump, lambda_signal=lambda_signal, pump_power=pump_power,
                     signal_power=signal_power, length=length)
    require_nonnegative(gain_coefficient=gain_coefficient)
    if (sbs or srs) and mode_radii is None:
        raise ValueError("mode_radii (w_pump, w_signal, w_idler) is required when sbs or srs is enabled")
    li = idler_wavelength(lambda_signal, lambda_pump, process)
    wP, wS, wI = (2 * math.pi * C0 / x for x in (lambda_pump, lambda_signal, li))
    phi_p0 = pump_power / (HBAR * wP)
    r0 = signal_power / (HBAR * wS) / phi_p0
    kap, dk, a2 = gain_coefficient, float(phase_mismatch), idler_loss / 2
    a2s, a2p = signal_loss / 2, pump_loss / 2
    tI, tS, tP = math.sqrt(idler_dump_transmission), math.sqrt(signal_dump_transmission), math.sqrt(pump_dump_transmission)
    sc = _scattering_setup(process, pump_power, length, wP, wS, wI, mode_radii or (1, 1, 1), sbs, srs, pump_linewidth)
    srate = sc["rate"] if sc else 0.0
    chi3 = process == "chi3"
    if chi3:
        npump = kap * wP / math.sqrt(wS * wI)
        ns, ni = npump * wS / wP, npump * wI / wP
        db = dk - 2 * npump
        rate = max(kap * math.sqrt(1 + r0), abs(db), 3 * npump * (1 + r0), a2, srate, 1e-6)
        y = [1.0, 0.0, math.sqrt(r0 * wS / wP), 0.0, 0.0, 0.0, sc["x0R"] if sc and sc["trackR"] else 0.0]
        cs_, ci_ = wP / wS, wP / wI
    else:
        rate = max(kap * math.sqrt(1 + r0), abs(dk), a2, srate, 1e-6)
        y = [1.0, 0.0, math.sqrt(r0), 0.0, 0.0, 0.0, sc["x0R"] if sc and sc["trackR"] else 0.0]
        cs_ = ci_ = 1.0
    N = math.ceil(length * rate / 0.04)
    N = max(2000 if record else 300, min(max_steps, N))
    nseg = int(n_dumps) + 1
    nper = math.ceil(N / nseg)
    h = length / (nseg * nper)

    def f2(z, Y):
        c, s = math.cos(dk * z), math.sin(dk * z)
        pr, pi, sr, si, ir, ii = Y[:6]
        o = [0.0] * 7
        xr, xi = sr * ir - si * ii, sr * ii + si * ir
        yr, yi = xr * c + xi * s, xi * c - xr * s
        o[0], o[1] = -kap * yi, kap * yr
        xr, xi = pr * ir + pi * ii, pi * ir - pr * ii
        yr, yi = xr * c - xi * s, xr * s + xi * c
        o[2], o[3] = -kap * yi, kap * yr
        xr, xi = pr * sr + pi * si, pi * sr - pr * si
        yr, yi = xr * c - xi * s, xr * s + xi * c
        o[4], o[5] = -kap * yi - a2 * ir, kap * yr - a2 * ii
        return o

    def f3(z, Y):
        c, sn = math.cos(db * z), math.sin(db * z)
        pr, pi, sr, si, ir, ii = Y[:6]
        P2, S2, I2 = pr * pr + pi * pi, sr * sr + si * si, ir * ir + ii * ii
        o = [0.0] * 7
        xr, xi = sr * ir - si * ii, sr * ii + si * ir
        yr, yi = xr * pr + xi * pi, xi * pr - xr * pi
        er, ei = yr * c - yi * sn, yr * sn + yi * c
        o[0], o[1] = -npump * ((P2 + 2 * S2 + 2 * I2) * pi + 2 * ei), npump * ((P2 + 2 * S2 + 2 * I2) * pr + 2 * er)
        qr, qi = pr * pr - pi * pi, 2 * pr * pi
        gr, gi = qr * c + qi * sn, qi * c - qr * sn
        Br, Bi = (S2 + 2 * P2 + 2 * I2) * sr + (gr * ir + gi * ii), (S2 + 2 * P2 + 2 * I2) * si + (gi * ir - gr * ii)
        o[2], o[3] = -ns * Bi, ns * Br
        Cr, Ci = (I2 + 2 * P2 + 2 * S2) * ir + (gr * sr + gi * si), (I2 + 2 * P2 + 2 * S2) * ii + (gi * sr - gr * si)
        o[4], o[5] = -ni * Ci - a2 * ir, ni * Cr - a2 * ii
        return o

    base = f3 if chi3 else f2

    def f(z, Y):
        o = base(z, Y)
        if a2s:
            o[2] -= a2s * Y[2]
            o[3] -= a2s * Y[3]
        if a2p:
            o[0] -= a2p * Y[0]
            o[1] -= a2p * Y[1]
        if sc:
            _add_scattering(sc, z, Y, o)
        return o

    rec = {"z": [], "P_pump": [], "P_signal": [], "P_idler": [], "P_raman": []} if record else None
    every = max(1, (nseg * nper) // 900)

    def push(z):
        rec["z"].append(z)
        rec["P_pump"].append((y[0] ** 2 + y[1] ** 2) * pump_power)
        rec["P_signal"].append((y[2] ** 2 + y[3] ** 2) * cs_ * phi_p0 * HBAR * wS)
        rec["P_idler"].append((y[4] ** 2 + y[5] ** 2) * ci_ * phi_p0 * HBAR * wI)
        rec["P_raman"].append(y[6] * pump_power)

    z, cnt = 0.0, 0
    if record:
        push(0.0)
    for sg in range(nseg):
        for nn in range(nper):
            k1 = f(z, y)
            t = [y[j] + 0.5 * h * k1[j] for j in range(7)]
            k2 = f(z + 0.5 * h, t)
            t = [y[j] + 0.5 * h * k2[j] for j in range(7)]
            k3 = f(z + 0.5 * h, t)
            t = [y[j] + h * k3[j] for j in range(7)]
            k4 = f(z + h, t)
            y = [y[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(7)]
            cnt += 1
            z = (sg * nper + nn + 1) * h
            if record and (cnt % every == 0 or nn == nper - 1):
                push(z)
        if sg < nseg - 1:
            y[4] *= tI; y[5] *= tI; y[2] *= tS; y[3] *= tS; y[0] *= tP; y[1] *= tP
            if record:
                push(z)
    fp = y[0] ** 2 + y[1] ** 2
    fs = (y[2] ** 2 + y[3] ** 2) * cs_
    fi = (y[4] ** 2 + y[5] ** 2) * ci_
    resid = abs(fp + 2 * fs - (1 + 2 * r0)) / (1 + 2 * r0) if chi3 else abs(fp + fs - (1 + r0)) / (1 + r0)
    values = {"P_pump_out": fp * pump_power, "P_signal_out": fs * phi_p0 * HBAR * wS, "P_idler_out": fi * phi_p0 * HBAR * wI,
              "gain_db": 10 * math.log10(fs / r0), "pump_depletion": 1 - fp, "photon_residual": resid,
              "P_raman_out": y[6] * pump_power, "steps": nseg * nper,
              "P_sbs_reflected": sc["sol"].D if sc and sc["sol"] else 0.0,
              "pump_fraction": fp, "signal_fraction": fs, "idler_fraction": fi}
    units = {"P_pump_out": "W", "P_signal_out": "W", "P_idler_out": "W", "gain_db": "dB", "pump_depletion": "",
             "photon_residual": "", "P_raman_out": "W", "steps": "", "P_sbs_reflected": "W",
             "pump_fraction": "", "signal_fraction": "", "idler_fraction": ""}
    if record:
        for k_, v_ in rec.items():
            values[k_] = np.array(v_)
            units[k_] = "m" if k_ == "z" else "W"
    assumptions = ["CW, slowly varying envelopes, single transverse mode per wave", "no idler seed",
                   "photon fractions are relative to the input pump photon flux",
                   "photon_residual checks Phi_p + Phi_s (chi2) or Phi_p + 2 Phi_s (chi3); not conserved with scattering or pump/signal loss"]
    return Result(values, units, assumptions)


# ---------- bandwidth ----------
def gain_bandwidth(gain_coefficient, length, lambda_signal, lambda_pump, gvm, beta2_sum=0.0, phase_mismatch=0.0,
                   idler_loss=0.0, process="chi2", drop_db=3.0) -> Result:
    """Signal gain bandwidth from the undepleted gain and a Taylor phase mismatch.

    dk(Omega) = dk0 + s (gvm Omega - beta2_sum Omega^2 / 2) with s = +1 (chi2) or -1 (chi3).
    The band is the contiguous region around the peak within ``drop_db`` of it.
    """
    require_choice("process", process, _PROCESSES)
    require_positive(gain_coefficient=gain_coefficient)
    require_positive(length=length)
    sgn = 1.0 if process == "chi2" else -1.0
    dkf = lambda O: phase_mismatch + sgn * (gvm * O - 0.5 * beta2_sum * O * O)
    gdb = lambda O: 10 * math.log10(max(_gain_one(gain_coefficient, dkf(O), length, idler_loss, 0.0, 0, 1.0, 1.0)[0], 1e-300))
    # bracket: |dk| = 6 Gamma or 8 pi / L, whichever larger, through the linear term
    R = max(6 * gain_coefficient, 8 * math.pi / length, 1.5 * idler_loss) + abs(phase_mismatch)
    span = R / abs(gvm) if gvm else (math.sqrt(2 * R / abs(beta2_sum)) if beta2_sum else 2 * math.pi * 15e12)
    span = min(span, 2 * math.pi * 15e12)
    O = np.linspace(-span, span, 2001)
    g = np.array([gdb(x) for x in O])
    k0 = int(np.argmax(g))
    thr = g[k0] - drop_db

    def edge(direction):
        k = k0
        while 0 < k < len(O) - 1 and g[k + direction] >= thr:
            k += direction
        if not (0 < k < len(O) - 1):
            return O[k]
        a, b = O[k], O[k + direction]
        for _ in range(60):
            m = 0.5 * (a + b)
            if gdb(m) >= thr:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    lo, hi = edge(-1), edge(1)
    W = hi - lo
    li = float(idler_wavelength(lambda_signal, lambda_pump, process))
    to_nm = lambda lam: W / (2 * math.pi) * lam * lam / C0
    est = 4 * math.sqrt(gain_coefficient * math.log(2) / length) / abs(gvm) if gvm else float("inf")
    return Result(
        values={"width": W, "width_hz": W / (2 * math.pi), "width_nm": to_nm(lambda_signal),
                "idler_width_nm": to_nm(li), "peak_gain_db": float(g[k0]), "peak_offset": float(O[k0]),
                "high_gain_estimate": est},
        units={"width": "rad/s", "width_hz": "Hz", "width_nm": "m", "idler_width_nm": "m", "peak_gain_db": "dB",
               "peak_offset": "rad/s", "high_gain_estimate": "rad/s"},
        assumptions=["Undepleted pump", "Taylor phase mismatch to second order", "flat idler loss",
                     "width_nm is Delta-lambda = lambda^2 Delta-nu / c (metres despite the name)"],
    )
