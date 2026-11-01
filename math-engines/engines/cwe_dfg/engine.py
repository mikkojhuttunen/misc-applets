"""Coupled-wave DFG and OPA (CW) in a χ² waveguide with an arbitrary poling profile g(z).

Three waves ω_p = ω_s + ω_i in the convention of ``phase_matching`` (ω3 = ω1 + ω2): 1 = signal, 2 = idler,
3 = pump. Amplitudes normalised to power (|A|² in W):

    dA_s/dz = -α_s/2 A_s + i κ_s g(z) A_p A_i* e^{-iφ}
    dA_i/dz = -α_i/2 A_i + i κ_i g(z) A_p A_s* e^{-iφ}
    dA_p/dz = -α_p/2 A_p + i κ_p g(z) A_s A_i  e^{+iφ},     φ(z) = ∫ Δk dz,  Δk = β_p - β_s - β_i

κ_j = ω_j C (so κ_p = κ_s + κ_i: power and the Manley-Rowe relations are conserved), with
κ_p² = η_SFG = 8π² d² / (ε0 c n_s n_i n_p λ_p² A_eff), the low-conversion SFG efficiency P_p = η P_s P_i L².
Small-signal parametric gain coefficient Γ² = κ_s κ_i P_p = 8π² d² P_p / (ε0 c n_s n_i n_p λ_s λ_i A_eff).
SHG (cwe_shg) is the degenerate limit: κ_SHG² = η_SFG / 4 at λ_pump,SHG = 2 λ_p.

Domains, Δk tables and grating helpers are those of ``cwe_shg``. Solvers:
  opa_transfer     undepleted pump, exact per domain (2×2 matrix exponential), signal/idler loss allowed
  three_wave       full depleted RK4, steps never straddle a domain wall, loss on all three waves
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range
from ..cwe_shg.engine import (DB_PER_M, _check_domains, _dk_table, domains_from_lengths,  # noqa: F401
                              periodic_domains)

_C = 299792458.0
_EPS0 = 8.8541878128e-12
DEPLETION_FLAG = 0.03


def idler_wavelength(wavelength_pump, wavelength_signal):
    lp, ls = np.asarray(wavelength_pump, float), np.asarray(wavelength_signal, float)
    if np.any(ls <= lp):
        raise ValueError("signal wavelength must be longer than the pump wavelength")
    return 1.0 / (1.0 / lp - 1.0 / ls)


# ----------------------------------------------------------------------------- coupling

def coupling_three_wave(wavelength_signal, wavelength_idler, n_signal, n_idler, n_pump, d_coeff, a_eff) -> Result:
    """κ_s, κ_i, κ_p and the SFG efficiency from d and the three-wave overlap area.

    A_eff = ∫|E_s|² ∫|E_i|² ∫|E_p|² / |∫_χ E_s E_i E_p* dA|²; d is the full-strength coefficient (QPM factor
    from g(z))."""
    require_positive(wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, n_signal=n_signal,
                     n_idler=n_idler, n_pump=n_pump, d_coeff=d_coeff, a_eff=a_eff)
    lp = 1.0 / (1.0 / wavelength_signal + 1.0 / wavelength_idler)
    eta = 8 * np.pi**2 * d_coeff**2 / (_EPS0 * _C * n_signal * n_idler * n_pump * lp**2 * a_eff)
    kp = np.sqrt(eta)
    ks, ki = kp * lp / wavelength_signal, kp * lp / wavelength_idler
    return Result(
        values={"kappa_signal": ks, "kappa_idler": ki, "kappa_pump": kp, "eta_sfg": eta, "wavelength_pump": lp,
                "gain_coeff": np.sqrt(ks * ki)},
        units={"kappa_signal": "1/(m W^0.5)", "kappa_idler": "1/(m W^0.5)", "kappa_pump": "1/(m W^0.5)",
               "eta_sfg": "1/(W m^2)", "wavelength_pump": "m", "gain_coeff": "1/(m W^0.5)"},
        assumptions=["CW, one spatial mode per wave; Kleinman symmetry; full-strength d (QPM factor from g(z))",
                     "gain_coeff = √(κ_s κ_i): Γ = gain_coeff · √P_pump"],
    )


def opa_gain_uniform(gamma, delta_k, length):
    """Closed-form signal gain of a uniform medium, unseeded idler, undepleted lossless pump:
    G = 1 + (Γ/g)² sinh²(gL), g = √(Γ² - Δk²/4) (sinh → sin above the gain band edge)."""
    g = np.sqrt(np.asarray(gamma**2 - np.asarray(delta_k, float) ** 2 / 4, dtype=complex))
    gl = g * length
    small = np.abs(gl) < 1e-8
    shc = np.where(small, length, np.sinh(np.where(small, 1.0, gl)) / np.where(small, 1.0, g))
    return 1 + gamma**2 * np.abs(shc) ** 2


# ----------------------------------------------------------------------------- undepleted transfer matrix

def _expm2(m11, m12, m21, m22, ell):
    """exp(M ℓ) of 2×2 complex matrices (arrays), stable at degenerate eigenvalues."""
    mean = 0.5 * (m11 + m22)
    half = 0.5 * (m11 - m22)
    s = np.sqrt(half * half + m12 * m21 + 0j)
    sl = s * ell
    small = np.abs(sl) < 1e-6
    ch = np.cosh(sl)
    sh_s = np.where(small, ell * (1 + sl * sl / 6), np.sinh(np.where(small, 1.0, sl)) / np.where(small, 1.0, s))
    e = np.exp(mean * ell)
    return e * (ch + sh_s * half), e * sh_s * m12, e * sh_s * m21, e * (ch - sh_s * half)


def opa_transfer(boundaries, g, delta_k, kappa_signal, kappa_idler, pump_power, signal_power, idler_power=0.0,
                 alpha_signal=0.0, alpha_idler=0.0, idler_phase=0.0):
    """Undepleted-pump OPA/DFG, exact per domain, vectorised over the rows of Δk (helper).

    κ_s, κ_i: scalars or arrays with one value per Δk row (dispersion of the coupling across wavelengths).
    Pump amplitude real, √P_p, constant (no pump loss: use three_wave for that). Variables u = A_s,
    v = A_i* e^{-iφ}: du/dz = -α_s/2 u + iκ_s g √P_p v,  dv/dz = -(α_i/2 + iΔk) v - iκ_i g √P_p u.
    Returns P_signal, P_idler, gain (P_s(L)/P_s(0)) and the output amplitudes."""
    require_positive(kappa_signal=kappa_signal, kappa_idler=kappa_idler, pump_power=pump_power)
    require_nonnegative(signal_power=signal_power, idler_power=idler_power, alpha_signal=alpha_signal, alpha_idler=alpha_idler)
    b, gg = _check_domains(boundaries, g)
    ell = np.diff(b)
    dk = _dk_table(delta_k, ell.size)
    ap = math.sqrt(pump_power)
    m = dk.shape[0]
    kappa_signal = np.broadcast_to(np.asarray(kappa_signal, dtype=float), (m,))
    kappa_idler = np.broadcast_to(np.asarray(kappa_idler, dtype=float), (m,))
    u = np.full(m, math.sqrt(signal_power), dtype=complex)
    v = np.full(m, math.sqrt(idler_power) * np.exp(-1j * idler_phase), dtype=complex)
    for k in range(ell.size):
        e11, e12, e21, e22 = _expm2(-0.5 * alpha_signal + 0 * dk[:, k], 1j * kappa_signal * gg[k] * ap,
                                    -1j * kappa_idler * gg[k] * ap, -(0.5 * alpha_idler + 1j * dk[:, k]), ell[k])
        u, v = e11 * u + e12 * v, e21 * u + e22 * v
    ps, pi = np.abs(u) ** 2, np.abs(v) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = ps / signal_power if signal_power > 0 else np.full(m, np.nan)
    return {"A_signal": u, "A_idler_conj": v, "P_signal": ps, "P_idler": pi, "gain": gain}


# ----------------------------------------------------------------------------- full solver

def _rhs(a1, a2, a3, gk, ph, k1, k2, k3, al1, al2, al3):
    e = np.exp(1j * ph)
    return (-0.5 * al1 * a1 + 1j * k1 * gk * a3 * np.conj(a2) * np.conj(e),
            -0.5 * al2 * a2 + 1j * k2 * gk * a3 * np.conj(a1) * np.conj(e),
            -0.5 * al3 * a3 + 1j * k3 * gk * a1 * a2 * e)


def _run(b, gg, dk, kap, a, alphas, h_max, min_steps):
    ell = np.diff(b)
    ph = np.zeros(dk.shape[0])
    a1, a2, a3 = a
    steps = 0
    for k in range(ell.size):
        n = max(min_steps, math.ceil(ell[k] / h_max - 1e-9))
        h = ell[k] / n
        dkk = dk[:, k]
        for j in range(n):
            p0 = ph + dkk * j * h
            pm, p1 = p0 + dkk * h / 2, p0 + dkk * h
            k1 = _rhs(a1, a2, a3, gg[k], p0, *kap, *alphas)
            k2 = _rhs(a1 + h / 2 * k1[0], a2 + h / 2 * k1[1], a3 + h / 2 * k1[2], gg[k], pm, *kap, *alphas)
            k3 = _rhs(a1 + h / 2 * k2[0], a2 + h / 2 * k2[1], a3 + h / 2 * k2[2], gg[k], pm, *kap, *alphas)
            k4 = _rhs(a1 + h * k3[0], a2 + h * k3[1], a3 + h * k3[2], gg[k], p1, *kap, *alphas)
            a1 = a1 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            a2 = a2 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            a3 = a3 + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        steps += n
        ph = ph + dkk * ell[k]
    return a1, a2, a3, steps


def three_wave(boundaries, g, delta_k, kappa_signal, kappa_idler, kappa_pump, pump_power, signal_power,
               idler_power=0.0, alpha_signal=0.0, alpha_idler=0.0, alpha_pump=0.0, max_step=None, min_steps=4,
               convergence_check=False):
    """Full three-wave coupled equations (pump depletion, loss on every wave) by wall-aware RK4 (helper).

    Diagnostics (lossless only): power_residual = ΣP(L)/ΣP(0) - 1 and manley_rowe = change of
    P_s/ω_s - P_i/ω_i relative to the input photon flux (both should be ~0)."""
    require_positive(kappa_signal=kappa_signal, kappa_idler=kappa_idler, kappa_pump=kappa_pump, pump_power=pump_power)
    require_nonnegative(signal_power=signal_power, idler_power=idler_power, alpha_signal=alpha_signal,
                        alpha_idler=alpha_idler, alpha_pump=alpha_pump)
    b, gg = _check_domains(boundaries, g)
    dk = _dk_table(delta_k, b.size - 1)
    ptot = pump_power + signal_power + idler_power
    kmax = max(kappa_signal, kappa_idler, kappa_pump)
    dkmax = float(np.max(np.abs(dk)))
    h = min(np.pi / 8 / dkmax if dkmax > 0 else np.inf, 0.02 / (kmax * math.sqrt(ptot)))
    h = float(max_step) if max_step is not None else h
    m = dk.shape[0]
    a0 = (np.full(m, math.sqrt(signal_power), complex), np.full(m, math.sqrt(idler_power), complex),
          np.full(m, math.sqrt(pump_power), complex))
    kap = (kappa_signal, kappa_idler, kappa_pump)
    al = (alpha_signal, alpha_idler, alpha_pump)
    a1, a2, a3, steps = _run(b, gg, dk, kap, a0, al, h, min_steps)
    out = {"A_signal": a1, "A_idler": a2, "A_pump": a3, "P_signal": np.abs(a1) ** 2, "P_idler": np.abs(a2) ** 2,
           "P_pump": np.abs(a3) ** 2, "steps": steps}
    if signal_power > 0:
        out["gain"] = out["P_signal"] / signal_power
    if alpha_signal == alpha_idler == alpha_pump == 0:
        out["power_residual"] = (out["P_signal"] + out["P_idler"] + out["P_pump"]) / ptot - 1
        # photon fluxes ∝ P/ω ∝ P/κ (κ_j ∝ ω_j)
        f = lambda ps, pi: ps / kappa_signal - pi / kappa_idler
        ref = (signal_power / kappa_signal + idler_power / kappa_idler + pump_power / kappa_pump)
        out["manley_rowe"] = (f(out["P_signal"], out["P_idler"]) - f(signal_power, idler_power)) / ref
    if convergence_check:
        a1h = _run(b, gg, dk, kap, a0, al, h / 2, 2 * min_steps)[0]
        out["step_error"] = np.abs(np.abs(a1h) ** 2 - out["P_signal"]) / np.maximum(np.abs(a1h) ** 2, 1e-300)
    return out


# ----------------------------------------------------------------------------- spec function

def ordered_opa(wavelength_pump, wavelength_signal, n_pump, n_signal, n_idler, d_coeff, a_eff, length, pump_power,
                signal_power, period_offset=0.0, duty_cycle=0.5, loss_signal=0.0, loss_idler=0.0, loss_pump=0.0) -> Result:
    """Ordered first-order QPM OPA/DFG at the matched period: closed form, exact undepleted, depleted RK4.

    Losses in dB/m (power). Gains are signal gains P_s(L)/P_s(0); dfg_efficiency = P_i/(P_p P_s)."""
    require_positive(wavelength_pump=wavelength_pump, wavelength_signal=wavelength_signal, length=length,
                     pump_power=pump_power, signal_power=signal_power)
    require_range("duty_cycle", duty_cycle, 1e-3, 1 - 1e-3)
    li = float(idler_wavelength(wavelength_pump, wavelength_signal))
    dk = 2 * np.pi * (n_pump / wavelength_pump - n_signal / wavelength_signal - n_idler / li)
    if dk == 0:
        raise ValueError("Δk = 0 without a grating: no QPM period is needed")
    period = 2 * np.pi / abs(dk) + period_offset
    require_positive(period=period)
    cp = coupling_three_wave(wavelength_signal, li, n_signal, n_idler, n_pump, d_coeff, a_eff)
    ks, ki, kp = float(cp["kappa_signal"]), float(cp["kappa_idler"]), float(cp["kappa_pump"])
    a_s, a_i, a_p = DB_PER_M * loss_signal, DB_PER_M * loss_idler, DB_PER_M * loss_pump
    b, g = periodic_domains(length, period, duty_cycle)
    tm = opa_transfer(b, g, abs(dk), ks, ki, pump_power, signal_power, alpha_signal=a_s, alpha_idler=a_i)
    full = three_wave(b, g, abs(dk), ks, ki, kp, pump_power, signal_power, alpha_signal=a_s, alpha_idler=a_i,
                      alpha_pump=a_p, convergence_check=True)
    strength = abs(2 * np.sin(np.pi * duty_cycle) / np.pi)
    gamma = math.sqrt(ks * ki * pump_power)
    dres = abs(dk) - 2 * np.pi / period
    closed = float(opa_gain_uniform(strength * gamma, dres, length))
    db = lambda x: 10 * math.log10(x)
    gain_full = float(full["gain"][0])
    assumptions = cp.assumptions + [
        "Undepleted value: exact per-domain matrix exponentials, no pump loss",
        "Coupled-wave value: RK4 with steps that never straddle a domain wall, pump depletion and all losses",
        "Closed form: uniform medium with Γ_eff = (2 sin πD/π) Γ and the residual mismatch; no loss",
    ]
    if abs(wavelength_signal - li) / wavelength_signal < 2e-3:
        assumptions.append("WARNING: near degeneracy (λ_s ≈ λ_i) the type-0 OPA becomes phase sensitive; "
                           "the non-degenerate gain formulas do not apply")
    depl = 1 - float(full["P_pump"][0]) / (pump_power * math.exp(-a_p * length))
    if depl > DEPLETION_FLAG:
        assumptions.append(f"Pump depletion {100 * depl:.1f} %: the undepleted gain overestimates the output")
    return Result(
        values={"wavelength_idler": li, "delta_k": dk, "period": period, "gamma": gamma, "gain_closed_db": db(closed),
                "gain_undepleted_db": db(float(tm["gain"][0])), "gain_db": db(gain_full),
                "p_idler": float(full["P_idler"][0]), "p_signal": float(full["P_signal"][0]),
                "p_pump_out": float(full["P_pump"][0]),
                "dfg_efficiency": float(full["P_idler"][0]) / (pump_power * signal_power),
                "power_residual": float(full.get("power_residual", [np.nan])[0]),
                "manley_rowe": float(full.get("manley_rowe", [np.nan])[0]), "step_error": float(full["step_error"][0])},
        units={"wavelength_idler": "m", "delta_k": "1/m", "period": "m", "gamma": "1/m", "gain_closed_db": "dB",
               "gain_undepleted_db": "dB", "gain_db": "dB", "p_idler": "W", "p_signal": "W", "p_pump_out": "W",
               "dfg_efficiency": "1/W", "power_residual": "", "manley_rowe": "", "step_error": ""},
        assumptions=assumptions,
    )
