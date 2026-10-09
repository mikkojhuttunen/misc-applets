"""What heating does to phase matching and to resonators.

QPM (type-0, all waves extraordinary in MgO:LiNbO₃, Gayer et al. temperature-dependent Sellmeier):

    Δk(T) = 2π (n_p/λ_p - n_s/λ_s - n_i/λ_i) - 2π/Λ(T),   1/λ_i = 1/λ_p - 1/λ_s,   Λ(T) = Λ₀ (1 + α_L ΔT)

The low-gain (SHG/DFG/SFG) efficiency for a mismatch profile δk(z) = (dΔk/dT) ΔT(z) is

    η/η₀ = | (1/L) ∫₀^L exp(i ∫₀^z δk dz') dz |²

(sinc² for uniform ΔT, FWHM ΔT = 5.566 / (L |dΔk/dT|)). A non-uniform ΔT(z), e.g. the absorbed-pump profile, chirps
Δk; retuning the chip temperature removes only its mean.

Ring/disk resonator with absorption heating (coupled-mode theory, intracavity energy U):

    U = κ_e P / [(Δ + g U)² + κ²/4],   g = ω (dn_eff/dT / n_g) (R'/L_ring) κ_abs

Bistable (thermal triangle, hysteresis) above P_th = κ³ / (3√3 g κ_e). SI units.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range
from ..materials import engine as _mat

C0 = 299792458.0
SINC2_HALF = 1.391557378  # sinc²(x) = 1/2 at x = 1.3916 (x = ΔkL/2)


def _dk(lp, ls, temperature):
    li = 1.0 / (1.0 / lp - 1.0 / ls)
    n = lambda lam: float(_mat.thermal_index("ln_e_gayer", lam, temperature)["n"])
    return 2 * np.pi * (n(lp) / lp - n(ls) / ls - n(li) / li), li


def qpm_period_and_slope(pump_wavelength=0.775e-6, signal_wavelength=1.55e-6, temperature=297.65, alpha_L=15.4e-6):
    """(Λ, dΔk/dT, λ_idler) for type-0 QPM in 5 % MgO:LiNbO₃ (bulk indices)."""
    dk0, li = _dk(pump_wavelength, signal_wavelength, temperature)
    dkp, _ = _dk(pump_wavelength, signal_wavelength, temperature + 0.5)
    dkm, _ = _dk(pump_wavelength, signal_wavelength, temperature - 0.5)
    period = 2 * np.pi / dk0
    # -2π/Λ(T): d/dT = +(2π/Λ) α_L
    slope = (dkp - dkm) + dk0 * alpha_L
    return period, slope, li


def phase_matching_factor(z, dk):
    """η/η₀ = |(1/L) ∫ exp(i φ(z)) dz|², φ(z) = ∫₀^z δk, for δk sampled on z (trapezoid rule)."""
    z = np.asarray(z, dtype=float)
    dk = np.asarray(dk, dtype=float) * np.ones_like(z)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (dk[1:] + dk[:-1]) * np.diff(z))])
    e = np.exp(1j * phi)
    integral = np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(z))
    return float(abs(integral / (z[-1] - z[0])) ** 2)


def best_retuned_factor(z, dk, span=None):
    """Largest phase_matching_factor over a uniform offset δk₀ added to δk(z) (retuning the chip temperature).
    Returns (factor, offset)."""
    z = np.asarray(z, dtype=float)
    L = z[-1] - z[0]
    dk = np.asarray(dk, dtype=float) * np.ones_like(z)
    mean = float(np.sum(0.5 * (dk[1:] + dk[:-1]) * np.diff(z)) / L)
    span = span if span is not None else 4 * np.pi / L + float(np.ptp(dk))
    offs = -mean + np.linspace(-span, span, 401)
    vals = [phase_matching_factor(z, dk + o) for o in offs]
    i = int(np.argmax(vals))
    lo, hi = offs[max(i - 1, 0)], offs[min(i + 1, len(offs) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if phase_matching_factor(z, dk + a) > phase_matching_factor(z, dk + b):
            hi = b
        else:
            lo = a
    o = 0.5 * (lo + hi)
    return phase_matching_factor(z, dk + o), o


def qpm_thermal(pump_wavelength=0.775e-6, signal_wavelength=1.55e-6, length=0.01, dT_in=1.0, decay_length=1.0,
                temperature=297.65, alpha_L=15.4e-6, n_z=2001) -> Result:
    """Temperature acceptance of a type-0 PPLN interaction and the efficiency left when the waveguide is heated
    non-uniformly, ΔT(z) = ΔT_in exp(-z/ℓ) (ℓ: decay length of the heat, e.g. the pump absorption length;
    a large ℓ is uniform heating). pump_wavelength = signal_wavelength/2 is SHG (signal = idler)."""
    require_positive(pump_wavelength=pump_wavelength, signal_wavelength=signal_wavelength, length=length,
                     decay_length=decay_length, temperature=temperature)
    require_nonnegative(dT_in=dT_in)
    if not pump_wavelength < signal_wavelength:
        raise ValueError("pump_wavelength must be shorter than signal_wavelength")
    period, slope, li = qpm_period_and_slope(pump_wavelength, signal_wavelength, temperature, alpha_L)
    z = np.linspace(0.0, length, int(n_z))
    dT = dT_in * np.exp(-z / decay_length)
    dk = slope * dT
    eta = phase_matching_factor(z, dk)
    eta_rt, off = best_retuned_factor(z, dk)
    phase_spread = float(abs(slope) * dT_in * (decay_length * (1 - math.exp(-length / decay_length))))
    return Result(
        values={"period": period, "idler_wavelength": li, "dDk_dT": slope, "dT_FWHM": 4 * SINC2_HALF / (length * abs(slope)),
                "eta_heated": eta, "eta_retuned": eta_rt, "retune_dT": -off / slope if slope != 0 else 0.0,
                "phase_mismatch": phase_spread},
        units={"period": "m", "idler_wavelength": "m", "dDk_dT": "1/(m K)", "dT_FWHM": "K", "eta_heated": "", "eta_retuned": "",
               "retune_dT": "K", "phase_mismatch": "rad"},
        assumptions=[
            "Type-0 (eee) QPM in 5 % MgO:LiNbO₃, Gayer et al. (2008) temperature Sellmeier; bulk indices: the period of a thin-film "
            "waveguide is shorter (waveguide dispersion) but dΔk/dT is close to the bulk value",
            "Low-gain (undepleted) interaction; poling period expands with α_L (clamped TFLN on Si: use about 2.6e-6)",
            "phase_mismatch: accumulated ∫δk dz without retuning",
        ],
    )


def ring_thermal_bistability(wavelength=1.55e-6, ring_length=2 * np.pi * 100e-6, n_group=2.3, Q_intrinsic=1e6,
                             Q_coupling=1e6, absorbing_fraction=0.5, R_th=0.5, dneff_dT=3e-5, power=0.01) -> Result:
    """Thermal bistability threshold and on-resonance heating of a ring resonator.

    κ_i = ω/Q_i (κ_abs = f_abs κ_i heats the ring), κ_e = ω/Q_c, κ = κ_i + κ_e. Heat spreads along the whole ring,
    so the ring's thermal resistance is R'/L_ring (K/W)."""
    require_positive(wavelength=wavelength, ring_length=ring_length, n_group=n_group, Q_intrinsic=Q_intrinsic,
                     Q_coupling=Q_coupling, R_th=R_th)
    require_range("absorbing_fraction", absorbing_fraction, 0.0, 1.0)
    require_nonnegative(power=power)
    w = 2 * np.pi * C0 / wavelength
    ki, ke = w / Q_intrinsic, w / Q_coupling
    k = ki + ke
    kabs = absorbing_fraction * ki
    R_ring = R_th / ring_length
    g = w * abs(dneff_dT) / n_group * R_ring * kabs
    P_th = k**3 / (3 * math.sqrt(3) * g * ke) if g > 0 else math.inf
    U_res = 4 * ke * power / k**2          # thermally locked on resonance
    dT_res = R_ring * kabs * U_res
    shift = g * U_res
    return Result(
        values={"Q_loaded": w / k, "linewidth": wavelength * k / w, "P_threshold": P_th, "U_resonance": U_res,
                "P_abs_resonance": kabs * U_res, "dT_resonance": dT_res, "shift_linewidths": shift / k,
                "delta_lambda_max": wavelength * shift / w, "buildup": U_res * C0 / (n_group * ring_length) / power if power > 0 else 4 * ke * C0 / (k**2 * n_group * ring_length),
                "bistable": bool(power > P_th)},
        units={"Q_loaded": "", "linewidth": "m", "P_threshold": "W", "U_resonance": "J", "P_abs_resonance": "W", "dT_resonance": "K",
               "shift_linewidths": "", "delta_lambda_max": "m", "buildup": "", "bistable": ""},
        assumptions=[
            "Single-mode coupled-mode theory, steady state; heat spread uniformly along the ring (R'/L_ring)",
            "Only the absorbing share of the intrinsic loss heats; scattering does not",
            "On resonance = laser swept from the blue side so the thermally shifted resonance follows it (dn/dT > 0)",
            "buildup: circulating power / input power on resonance",
        ],
    )


def ring_energy_roots(detuning, power, kappa, kappa_e, g):
    """Real positive steady-state energies U of U[(Δ + gU)² + κ²/4] = κ_e P (one or three)."""
    c = [g * g, 2 * detuning * g, detuning**2 + kappa**2 / 4, -kappa_e * power]
    r = np.roots(c) if g != 0 else np.array([kappa_e * power / (detuning**2 + kappa**2 / 4)])
    return np.sort(np.real(r[(np.abs(np.imag(r)) < 1e-9 * np.max(np.abs(r))) & (np.real(r) > 0)]))
