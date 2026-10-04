"""Dissipative idler channel for parametric amplifiers: continuous and lumped idler loss.

Shared by opa_chi2 and opa_chi3. Under an undepleted pump the signal and the conjugated idler
c = a_i* e^{iΔk z} (photon-flux amplitudes) obey a linear 2×2 system

    d/dz [s, c] = M [s, c],  M = [[-α_s/2, iΓ], [-iΓ, iΔk - α_i/2]]

with Γ the parametric gain coefficient and Δk the total mismatch (χ2: Γ = K sqrt(F_p), Δk;
χ3: Γ = sqrt(γ_s γ_i) P, Δk → κ = Δβ + 2γ_p P). Losses:

  continuous  power attenuation α_i (1/m) along the whole length;
  lumped      N idler dumps at z = kL/(N+1), each removing dump_loss_db of idler power, with
              lossless segments in between (the χ(2)/χ(3) "idler dump" or filter scheme).

Both may act together. A spectral weight w(λ_i) ∈ [0, 1] (flat, band-pass, notch or custom curve)
scales α_i and the dump attenuation in dB at each idler wavelength, as in parametric-amplifier.html.
SI units; dB only where named.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive

PROFILES = ("flat", "pass", "stop", "custom")
DB = 10 / np.log(10)          # dB per neper of power


def db_to_alpha(db_per_m):
    """Power attenuation in dB/m → α in 1/m (helper)."""
    return np.asarray(db_per_m, dtype=float) / DB


def _pchip(xs, ys):
    """Monotone cubic (Fritsch–Carlson) interpolant, constant outside the points; the same
    construction as the applet's custom loss curve."""
    o = np.argsort(xs)
    xs, ys = np.asarray(xs, dtype=float)[o], np.asarray(ys, dtype=float)[o]
    n = len(xs)
    d = [(ys[k + 1] - ys[k]) / max(1e-30, xs[k + 1] - xs[k]) for k in range(n - 1)]
    m = []
    for k in range(n):
        if k == 0:
            m.append(d[0] if n > 1 else 0.0)
        elif k == n - 1:
            m.append(d[n - 2])
        else:
            m.append(0.0 if d[k - 1] * d[k] <= 0 else (d[k - 1] + d[k]) / 2)
    for k in range(n - 1):
        if d[k] == 0:
            m[k] = m[k + 1] = 0.0
            continue
        a, b = m[k] / d[k], m[k + 1] / d[k]
        h = a * a + b * b
        if h > 9:
            t = 3 / np.sqrt(h)
            m[k], m[k + 1] = t * a * d[k], t * b * d[k]

    def f(x):
        x = np.asarray(x, dtype=float)
        if n == 0:
            return np.zeros_like(x)
        j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, max(n - 2, 0))
        if n == 1:
            return np.full_like(x, ys[0])
        hh = xs[j + 1] - xs[j]
        t = (x - xs[j]) / hh
        t2, t3 = t * t, t * t * t
        mm = np.asarray(m)
        y = ((2 * t3 - 3 * t2 + 1) * ys[j] + (t3 - 2 * t2 + t) * hh * mm[j]
             + (-2 * t3 + 3 * t2) * ys[j + 1] + (t3 - t2) * hh * mm[j + 1])
        return np.where(x <= xs[0], ys[0], np.where(x >= xs[-1], ys[-1], y))

    return f


def loss_weight(profile, wavelength, band_center=None, band_width=None, band_edge=None, points=None):
    """Fraction w ∈ [0, 1] of the peak idler loss applied at `wavelength` (helper, vectorised).

    flat: 1 everywhere. pass: loss only inside [c - W/2, c + W/2] with tanh edges of width e.
    stop: lossless notch (1 - pass). custom: monotone cubic through `points` = [(λ, w), ...]."""
    require_choice("profile", profile, PROFILES)
    lam = np.asarray(wavelength, dtype=float)
    if profile == "flat":
        return np.ones_like(lam)
    if profile == "custom":
        if not points:
            raise ValueError("points must list (wavelength, weight) pairs for the custom profile")
        xs, ys = zip(*points)
        return np.clip(_pchip(xs, ys)(lam), 0.0, 1.0)
    require_positive(band_center=band_center, band_width=band_width, band_edge=band_edge)
    a, b = band_center - band_width / 2, band_center + band_width / 2
    box = np.clip(0.5 * (np.tanh((lam - a) / band_edge) - np.tanh((lam - b) / band_edge)), 0.0, 1.0)
    return box if profile == "pass" else 1 - box


def loss_spectrum(profile, wavelength, band_center, band_width, band_edge, alpha_idler=0.0, dump_loss_db=0.0) -> Result:
    """Idler loss at one wavelength: profile weight, continuous α_i·w and dump attenuation·w."""
    require_nonnegative(alpha_idler=alpha_idler, dump_loss_db=dump_loss_db)
    w = loss_weight(profile, wavelength, band_center, band_width, band_edge)
    return Result(
        values={"weight": w, "alpha_idler": alpha_idler * w, "alpha_idler_db": alpha_idler * w * DB, "dump_loss_db": dump_loss_db * w},
        units={"weight": "", "alpha_idler": "1/m", "alpha_idler_db": "dB/m", "dump_loss_db": "dB"},
        assumptions=["tanh band edges: the weight goes from 12 % to 88 % over twice band_edge",
                     "The same weight scales the continuous loss and the dB attenuation of each dump"],
    )


def _expm2(m11, m12, m21, m22, z):
    """exp(M z) of a 2×2 complex matrix (arrays broadcast), via
    e^{tz/2}[cosh(Dz/2) I + (M - t/2 I) sinh(Dz/2)/(D/2)], t = tr M, D² = (m11 - m22)² + 4 m12 m21."""
    t = m11 + m22
    D = np.sqrt((m11 - m22) ** 2 + 4 * m12 * m21 + 0j)
    x = D * z / 2
    small = np.abs(x) < 1e-6
    # eigen-exponentials kept apart so a strong loss (e^{tz/2} → 0) never multiplies cosh → ∞
    e1, e2 = np.exp((t + D) * z / 2), np.exp((t - D) * z / 2)
    Ds = np.where(small, 1.0, D)
    A = np.where(small, np.exp(t * z / 2) * (1 + x * x / 2), (e1 + e2) / 2)                  # e^{tz/2} cosh(Dz/2)
    B = np.where(small, np.exp(t * z / 2) * z * (1 + x * x / 6), (e1 - e2) / Ds)             # e^{tz/2} sinh(Dz/2)/(D/2)
    return (A + (m11 - t / 2) * B, m12 * B, m21 * B, A + (m22 - t / 2) * B)


def propagate_linear(gamma, delta_k, length, alpha_idler=0.0, dumps=0, dump_loss_db=0.0,
                     alpha_signal=0.0, dump_loss_signal_db=0.0):
    """Exact undepleted-pump solution with continuous and lumped losses (helper, vectorised over
    gamma, delta_k and the losses). Starts from s = 1, c = 0; returns complex (s(L), c(L)), so
    |s|² is the signal gain and |c|² the idler photons out per signal photon in."""
    require_positive(length=length)
    require_nonnegative(gamma=gamma, alpha_idler=alpha_idler, dump_loss_db=dump_loss_db,
                        alpha_signal=alpha_signal, dump_loss_signal_db=dump_loss_signal_db)
    dumps = int(dumps)
    if dumps < 0:
        raise ValueError(f"dumps must be >= 0, got {dumps!r}")
    g = np.asarray(gamma, dtype=float)
    m11 = -np.asarray(alpha_signal, dtype=float) / 2 + 0j
    m22 = 1j * np.asarray(delta_k, dtype=float) - np.asarray(alpha_idler, dtype=float) / 2
    dz = length / (dumps + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        E11, E12, E21, E22 = _expm2(m11, 1j * g, -1j * g, m22, dz)
        ti = 10 ** (-np.asarray(dump_loss_db, dtype=float) / 20)
        ts = 10 ** (-np.asarray(dump_loss_signal_db, dtype=float) / 20)
        s, c = np.ones_like(E11), np.zeros_like(E11)
        for k in range(dumps + 1):
            s, c = E11 * s + E12 * c, E21 * s + E22 * c
            if k < dumps:
                s, c = s * ts, c * ti
    return s, c


def linear_gain(gamma, delta_k, length, alpha_idler=0.0, dumps=0, dump_loss_db=0.0,
                alpha_signal=0.0, dump_loss_signal_db=0.0) -> Result:
    """Small-signal signal gain and idler output with a dissipative idler (continuous α_i and/or N
    lumped dumps), compared with the lossless amplifier of the same Γ, Δk and L."""
    s, c = propagate_linear(gamma, delta_k, length, alpha_idler, dumps, dump_loss_db, alpha_signal, dump_loss_signal_db)
    s0, _ = propagate_linear(gamma, delta_k, length)
    G, G0 = np.abs(s) ** 2, np.abs(s0) ** 2
    a = np.asarray(alpha_idler, dtype=float)
    dk = np.asarray(delta_k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_ad = np.where(a > 0, np.asarray(gamma, dtype=float) ** 2 * a / ((a / 2) ** 2 + dk**2), 0.0)
    return Result(
        values={"signal_gain": G, "signal_gain_db": 10 * np.log10(G), "idler_photon_conversion": np.abs(c) ** 2,
                "lossless_gain_db": 10 * np.log10(G0),
                "idler_attenuation_db": a * length * DB + int(dumps) * np.asarray(dump_loss_db, dtype=float),
                "adiabatic_gain_coefficient": g_ad},
        units={"signal_gain": "", "signal_gain_db": "dB", "idler_photon_conversion": "", "lossless_gain_db": "dB",
               "idler_attenuation_db": "dB", "adiabatic_gain_coefficient": "1/m"},
        assumptions=[
            "Undepleted, lossless pump; no idler seed; exact 2×2 matrix exponential per lossless or lossy segment",
            "Lumped dumps at z = kL/(N+1), k = 1..N, act on the idler amplitude as 10^(-dB/20)",
            "χ2: Γ = K sqrt(F_p), Δk; χ3: Γ = sqrt(γ_s γ_i) P and Δk → κ = Δβ + 2γ_p P",
            "adiabatic_gain_coefficient Γ² α_i / ((α_i/2)² + Δk²) is the signal power gain rate for α_i ≫ Γ",
        ],
    )


def amplifier_with_loss(gamma, delta_k, length, wavelength_signal, wavelength_idler, wavelength_pump,
                        alpha_idler=0.0, dumps=0, dump_loss_db=0.0, profile="flat", band_center=None,
                        band_width=None, band_edge=None, points=None, loss_acts_on="idler"):
    """Gain arrays for the OPA gain spectra (helper, vectorised over the signal grid).

    The peak idler loss (α_i, dump dB) is weighted by the profile at each idler wavelength; with
    loss_acts_on = "all" the same curve also attenuates the signal at its own wavelength (a real
    filter). Returns dict(gain, lossless_gain, idler_photons, weight_idler, weight_signal, weight_pump);
    the pump weight is reported only: pump attenuation is outside the undepleted-pump model."""
    require_choice("loss_acts_on", loss_acts_on, ("idler", "all"))
    require_nonnegative(alpha_idler=alpha_idler, dump_loss_db=dump_loss_db)
    w_i = loss_weight(profile, wavelength_idler, band_center, band_width, band_edge, points)
    lossy = (alpha_idler > 0 or (int(dumps) > 0 and dump_loss_db > 0))
    on_all = loss_acts_on == "all" and profile != "flat"
    w_s = loss_weight(profile, wavelength_signal, band_center, band_width, band_edge, points) if on_all else np.zeros_like(w_i)
    w_p = float(loss_weight(profile, wavelength_pump, band_center, band_width, band_edge, points)) if on_all else 0.0
    s0, c0 = propagate_linear(gamma, delta_k, length)
    if lossy:
        s, c = propagate_linear(gamma, delta_k, length, alpha_idler * w_i, dumps, dump_loss_db * w_i,
                                alpha_idler * w_s, dump_loss_db * w_s)
    else:
        s, c = s0, c0
    return {"gain": np.abs(s) ** 2, "lossless_gain": np.abs(s0) ** 2, "idler_photons": np.abs(c) ** 2,
            "weight_idler": w_i, "weight_signal": w_s, "weight_pump": w_p}
