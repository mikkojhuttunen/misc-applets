"""χ(2) optical parametric amplification ω_p = ω_s + ω_i in guided modes (CW, collinear).

Fields are photon-flux amplitudes a_j with |a_j|² = F_j = P_j / (ħ ω_j) (photons/s). With
Δk = k_p - k_s - k_i (minus any QPM grating vector, the phase_matching convention):

    da_p/dz = i K a_s a_i e^{-iΔk z}
    da_s/dz = i K a_p a_i* e^{+iΔk z}
    da_i/dz = i K a_p a_s* e^{+iΔk z}

K = 2 d_eff θ sqrt(ħ ω_p ω_s ω_i / (2 ε0 c³ n_p n_s n_i)), θ = ∫ e_p e_s e_i dA the overlap of
power-normalised Gaussian modes (1/e² intensity radii w_j). The parametric gain coefficient
Γ = K sqrt(F_p0) reduces to the textbook Γ² = 2 ω_s ω_i d_eff² I_p / (n_p n_s n_i ε0 c³) with
I_p = P_p θ². The integrator works in u_j = a_j / sqrt(F_p0), like parametric-amplifier.html.
SI units in and out.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive
from ..materials.engine import MATERIALS, index as material_index
from ..phase_matching.engine import three_wave_mismatch

HBAR = 1.054571817e-34
C0 = 299792458.0
EPS0 = 8.8541878128e-12


def idler_wavelength(wavelength_pump, wavelength_signal):
    """λ_i from 1/λ_i = 1/λ_p - 1/λ_s (helper, vectorised); requires λ_s > λ_p."""
    lp = np.asarray(wavelength_pump, dtype=float)
    ls = np.asarray(wavelength_signal, dtype=float)
    if np.any(ls <= lp):
        raise ValueError(f"wavelength_signal must be longer than wavelength_pump, got {wavelength_signal!r} <= {wavelength_pump!r}")
    return 1.0 / (1.0 / lp - 1.0 / ls)


def mode_overlap(w_pump, w_signal, w_idler):
    """θ = ∫ e_p e_s e_i dA for Gaussian modes normalised to unit power (1/m, helper)."""
    require_positive(w_pump=w_pump, w_signal=w_signal, w_idler=w_idler)
    wp, ws, wi = (np.asarray(x, dtype=float) for x in (w_pump, w_signal, w_idler))
    inv = 1 / wp**2 + 1 / ws**2 + 1 / wi**2
    return (np.pi / inv) * (2 / np.pi) ** 1.5 / (wp * ws * wi)


def _coupling_k(wavelength_pump, wavelength_signal, d_eff, n_pump, n_signal, n_idler, theta):
    li = idler_wavelength(wavelength_pump, wavelength_signal)
    wp, ws, wi = (2 * np.pi * C0 / np.asarray(l, dtype=float) for l in (wavelength_pump, wavelength_signal, li))
    return 2 * d_eff * theta * np.sqrt(HBAR * wp * ws * wi / (2 * EPS0 * C0**3 * n_pump * n_signal * n_idler))


def coupling(wavelength_pump, wavelength_signal, d_eff, pump_power, n_pump, n_signal, n_idler,
             w_pump, w_signal, w_idler) -> Result:
    """Idler wavelength, mode overlap and the parametric gain coefficient Γ = K sqrt(F_p0)."""
    require_positive(wavelength_pump=wavelength_pump, wavelength_signal=wavelength_signal, d_eff=d_eff,
                     pump_power=pump_power, n_pump=n_pump, n_signal=n_signal, n_idler=n_idler)
    li = idler_wavelength(wavelength_pump, wavelength_signal)
    theta = mode_overlap(w_pump, w_signal, w_idler)
    K = _coupling_k(wavelength_pump, wavelength_signal, d_eff, n_pump, n_signal, n_idler, theta)
    Fp0 = pump_power * wavelength_pump / (2 * np.pi * HBAR * C0)
    gamma = K * np.sqrt(Fp0)
    return Result(
        values={"wavelength_idler": li, "overlap": theta, "area_eff": 1 / theta**2, "coupling_k": K,
                "gamma": gamma, "gain_length": 1 / gamma},
        units={"wavelength_idler": "m", "overlap": "1/m", "area_eff": "m^2", "coupling_k": "s^(1/2)/m",
               "gamma": "1/m", "gain_length": "m"},
        assumptions=[
            "Collinear CW three-wave mixing ω_p = ω_s + ω_i, slowly varying envelopes",
            "Gaussian mode profiles (1/e² intensity radii), no diffraction (guided waves)",
            "area_eff = 1/θ²; Γ is the small-signal amplitude gain rate at Δk = 0",
        ],
    )


def _gain_closed_form(gamma, delta_k, length):
    """G_s = 1 + Γ² |sinh(gL)/g|², g² = Γ² - (Δk/2)² (undepleted pump, vectorised)."""
    gamma = np.asarray(gamma, dtype=float)
    g2 = gamma**2 - (np.asarray(delta_k, dtype=float) / 2) ** 2
    g = np.sqrt(np.abs(g2))
    x = g * length
    small = x < 1e-8
    xs = np.where(small, 1.0, x)
    shape = np.where(g2 >= 0, np.sinh(xs) / xs, np.sin(xs) / xs)  # sinh(gL)/(gL) or its analytic continuation
    shape = np.where(small, 1.0, shape)
    return 1 + (gamma * length * shape) ** 2, g2


def small_signal_gain(gamma, delta_k, length, wavelength_pump, wavelength_signal) -> Result:
    """Undepleted-pump signal gain and idler conversion for a phase mismatch Δk (closed form)."""
    require_positive(gamma=gamma, length=length)
    li = idler_wavelength(wavelength_pump, wavelength_signal)
    G, g2 = _gain_closed_form(gamma, delta_k, length)
    peak = np.cosh(np.asarray(gamma, dtype=float) * length) ** 2
    return Result(
        values={"signal_gain": G, "signal_gain_db": 10 * np.log10(G),
                "idler_conversion": (np.asarray(wavelength_signal, dtype=float) / li) * (G - 1),
                "growth_rate": np.sqrt(np.maximum(g2, 0.0)), "g_squared": g2,
                "peak_gain_db": 10 * np.log10(peak)},
        units={"signal_gain": "", "signal_gain_db": "dB", "idler_conversion": "", "growth_rate": "1/m",
               "g_squared": "1/m^2", "peak_gain_db": "dB"},
        assumptions=[
            "Undepleted, lossless pump; no idler seed",
            "G_s = 1 + (Γ/g)² sinh²(gL), g² = Γ² - (Δk/2)²; sinh → sin where g² < 0",
            "idler_conversion = P_i(L)/P_s(0) = (ω_i/ω_s)(G_s - 1); peak_gain_db is cosh²(ΓL) at Δk = 0",
        ],
    )


def propagate_normalised(kappa, delta_k, length, flux_ratio, alpha_pump=0.0, alpha_signal=0.0, alpha_idler=0.0,
                         max_steps=200000, record=False) -> Result:
    """Fixed-step RK4 of the coupled equations in u_j = a_j / sqrt(F_p0); u_p(0) = 1, u_s(0) = sqrt(r0),
    u_i(0) = 0. kappa = Γ (1/m); alphas are power loss coefficients (1/m). Same step rule as the
    applet: N = max(300, min(max_steps, ceil(L·rate/0.04)))."""
    require_positive(length=length)
    require_nonnegative(kappa=kappa, flux_ratio=flux_ratio, alpha_pump=alpha_pump, alpha_signal=alpha_signal,
                        alpha_idler=alpha_idler)
    kap, dk, r0 = float(kappa), float(delta_k), float(flux_ratio)
    ap, as_, ai = alpha_pump / 2, alpha_signal / 2, alpha_idler / 2
    rate = max(kap * math.sqrt(1 + r0), abs(dk), ap, as_, ai, 1e-6)
    N = max(300, min(int(max_steps), math.ceil(length * rate / 0.04)))
    h = length / N
    ik = 1j * kap

    def f(z, p, s, i):
        e = complex(math.cos(dk * z), math.sin(dk * z))
        return (ik * s * i * e.conjugate() - ap * p,
                ik * p * i.conjugate() * e - as_ * s,
                ik * p * s.conjugate() * e - ai * i)

    p, s, i = 1 + 0j, complex(math.sqrt(r0)), 0j
    every = max(1, N // 500)
    rec = {"z": [0.0], "fp": [1.0], "fs": [r0], "fi": [0.0]} if record else None
    for n in range(N):
        z = n * h
        k1 = f(z, p, s, i)
        k2 = f(z + h / 2, p + h / 2 * k1[0], s + h / 2 * k1[1], i + h / 2 * k1[2])
        k3 = f(z + h / 2, p + h / 2 * k2[0], s + h / 2 * k2[1], i + h / 2 * k2[2])
        k4 = f(z + h, p + h * k3[0], s + h * k3[1], i + h * k3[2])
        p += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        s += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        i += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if record and ((n + 1) % every == 0 or n == N - 1):
            rec["z"].append((n + 1) * h)
            rec["fp"].append(abs(p) ** 2)
            rec["fs"].append(abs(s) ** 2)
            rec["fi"].append(abs(i) ** 2)
    fp, fs, fi = abs(p) ** 2, abs(s) ** 2, abs(i) ** 2
    vals = {"fp": fp, "fs": fs, "fi": fi, "steps": N, "capped": N == int(max_steps),
            "manley_rowe_residual": abs(fp + fs - (1 + r0)) / (1 + r0)}
    units = {"fp": "", "fs": "", "fi": "", "steps": "", "capped": "", "manley_rowe_residual": ""}
    if record:
        for k, v in rec.items():
            vals[f"{k}_z"] = np.array(v)
            units[f"{k}_z"] = "m" if k == "z" else ""
    return Result(values=vals, units=units, assumptions=[
        "Photon fluxes normalised to the input pump flux", "Fixed-step RK4, step ≤ 0.04 / rate",
        "manley_rowe_residual |f_p + f_s - (1 + r0)|/(1 + r0) is zero unless pump or signal is lossy (idler loss keeps it)"])


def coupled_wave(wavelength_pump, wavelength_signal, d_eff, pump_power, signal_power, length,
                 n_pump, n_signal, n_idler, w_pump, w_signal, w_idler, delta_k=0.0,
                 alpha_pump=0.0, alpha_signal=0.0, alpha_idler=0.0, max_steps=200000) -> Result:
    """Full nonlinear propagation with pump depletion and linear losses: output powers, gain,
    depletion and power profiles along z (arrays z, pump_power_z, signal_power_z, idler_power_z)."""
    require_positive(signal_power=signal_power)
    c = coupling(wavelength_pump, wavelength_signal, d_eff, pump_power, n_pump, n_signal, n_idler,
                 w_pump, w_signal, w_idler)
    lp, ls, li = wavelength_pump, wavelength_signal, float(c["wavelength_idler"])
    r0 = (signal_power * ls) / (pump_power * lp)          # F_s0 / F_p0
    r = propagate_normalised(c["gamma"], delta_k, length, r0, alpha_pump, alpha_signal, alpha_idler,
                             max_steps=max_steps, record=True)
    sp, ss, si = pump_power, pump_power * lp / ls, pump_power * lp / li   # flux fraction → W
    Pp, Ps, Pi = r["fp"] * sp, r["fs"] * ss, r["fi"] * si
    return Result(
        values={"wavelength_idler": li, "gamma": c["gamma"], "pump_out": Pp, "signal_out": Ps, "idler_out": Pi,
                "signal_gain_db": 10 * math.log10(Ps / signal_power), "pump_depletion": 1 - Pp / pump_power,
                "manley_rowe_residual": r["manley_rowe_residual"], "steps": r["steps"],
                "z": r["z_z"], "pump_power_z": r["fp_z"] * sp, "signal_power_z": r["fs_z"] * ss,
                "idler_power_z": r["fi_z"] * si},
        units={"wavelength_idler": "m", "gamma": "1/m", "pump_out": "W", "signal_out": "W", "idler_out": "W",
               "signal_gain_db": "dB", "pump_depletion": "", "manley_rowe_residual": "", "steps": "",
               "z": "m", "pump_power_z": "W", "signal_power_z": "W", "idler_power_z": "W"},
        assumptions=c.assumptions + r.assumptions[1:] + [
            "No idler seed; constant Δk (uniform grating or waveguide)",
            "Losses are power attenuation coefficients α_j (dP/dz = -α P)",
        ] + (["Step count hit max_steps: accuracy not guaranteed"] if r["capped"] else []),
    )


def gain_spectrum(material, wavelength_pump, wavelength_signal, span, length, d_eff, pump_power,
                  w_pump, w_signal, w_idler, period=None, order=1, points=401) -> Result:
    """Small-signal gain vs signal wavelength for a QPM grating in a bulk-index material (type 0).
    Without `period`, the grating is chosen to phase-match at `wavelength_signal`. Arrays:
    signal_wavelengths, idler_wavelengths, delta_k, gain, gain_db."""
    require_choice("material", material, MATERIALS)
    require_positive(span=span, length=length, d_eff=d_eff, pump_power=pump_power)
    points = int(points)
    if points < 3:
        raise ValueError(f"points must be >= 3, got {points!r}")
    lp = float(wavelength_pump)
    np_ = float(material_index(material, lp))

    def mismatch(ls, per):
        li = idler_wavelength(lp, ls)
        return three_wave_mismatch(ls, li, material_index(material, ls), material_index(material, li), np_,
                                   period=per, order=order)

    if period is None:
        period = float(mismatch(wavelength_signal, None)["period_required"])
    ls = np.linspace(wavelength_signal - span / 2, wavelength_signal + span / 2, points)
    if ls[0] <= lp:
        raise ValueError("span reaches the pump wavelength; reduce span")
    li = idler_wavelength(lp, ls)
    dk = mismatch(ls, period)["delta_k"]
    n_s, n_i = material_index(material, ls), material_index(material, li)
    theta = mode_overlap(w_pump, w_signal, w_idler)
    gamma = _coupling_k(lp, ls, d_eff, np_, n_s, n_i, theta) * np.sqrt(pump_power * lp / (2 * np.pi * HBAR * C0))
    G, _ = _gain_closed_form(gamma, dk, length)
    k = int(np.argmax(G))
    half = G[k] / 2
    lo, hi = k, k
    while lo > 0 and G[lo - 1] >= half:
        lo -= 1
    while hi < points - 1 and G[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == points - 1:
        fwhm = float("nan")
    else:
        edge = lambda a, b: ls[a] + (half - G[a]) * (ls[b] - ls[a]) / (G[b] - G[a])
        fwhm = edge(hi, hi + 1) - edge(lo - 1, lo)
    return Result(
        values={"period": period, "peak_gain_db": 10 * np.log10(G[k]), "peak_signal_wavelength": ls[k],
                "bandwidth_3db": fwhm, "gamma_center": gamma[points // 2],
                "signal_wavelengths": ls, "idler_wavelengths": li, "delta_k": dk, "gain": G, "gain_db": 10 * np.log10(G)},
        units={"period": "m", "peak_gain_db": "dB", "peak_signal_wavelength": "m", "bandwidth_3db": "m",
               "gamma_center": "1/m", "signal_wavelengths": "m", "idler_wavelengths": "m", "delta_k": "1/m",
               "gain": "", "gain_db": "dB"},
        assumptions=[
            "Type-0 interaction: all three waves see the bulk Sellmeier index of the chosen material",
            "d_eff already includes the QPM factor (2/π d33 for first order, 50 % duty)",
            "Undepleted pump, CW, Gaussian guided-mode overlap without waveguide dispersion",
            "bandwidth_3db: full width where G ≥ G_max/2 around the peak; None if it reaches the span edge",
        ],
    )
