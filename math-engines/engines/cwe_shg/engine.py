"""Coupled-wave SHG in a χ² waveguide with an arbitrary poling profile g(z).

Fields: pump at vacuum wavelength λ (ω) and second harmonic at λ/2 (2ω). Slowly varying CW
amplitudes normalised so that |A|² is the guided power in W:

    dA1/dz = -α1/2 A1 + i κ g(z) A2 A1* e^{-iφ(z)}
    dA2/dz = -α2/2 A2 + i κ g(z) A1²    e^{+iφ(z)},     φ(z) = ∫0^z Δk dz'

Δk = β_SH - 2 β_pump (the convention of ``phase_matching.three_wave_mismatch`` with
ω1 = ω2). The grating vector is NOT subtracted: the poling function g(z) carries it.
g = ±1 for hard domains; any value in [-1, 1] is accepted (effective-medium / apodised
profiles). With this normalisation the same κ appears in both equations and
|A1|² + |A2|² is conserved without loss. κ² is the normalised efficiency η0 of a perfectly
phase-matched, full-strength (unpoled, Δk = 0) device: P_SH = η0 P² L² at low conversion.

Poling description (shared by all solvers here and by the later cwe_* engines): domain
boundaries z_0 = 0 < z_1 < ... < z_N = L and one value g_k per domain. Δk may be a scalar,
an array over wavelengths (M,), or (M, N) with one value per domain (Δk(z) piecewise
constant on the domain grid, e.g. width or temperature variation along the device).

SI units throughout: m, W, 1/m, m/V. κ is in 1/(m √W), η0 in 1/(W m²).
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

_C = 299792458.0
_EPS0 = 8.8541878128e-12
DB_PER_M = math.log(10) / 10          # α [1/m] = DB_PER_M · loss [dB/m] (power)
DEPLETION_FLAG = 0.03                  # undepleted results are flagged above this conversion


# ----------------------------------------------------------------------------- coupling

def coupling_shg(wavelength, n_pump, n_sh, d_coeff, a_eff) -> Result:
    """Coupling κ and normalised efficiency η0 = κ² from the effective area of the mode overlap.

    η0 = 8π² d² / (ε0 c n_p² n_SH λ² A_eff), with A_eff = 1/|Γ|² from the power-normalised
    overlap Γ = ∫ χ(x,y) E_p² E_SH* dA over the nonlinear region. d is the full-strength
    coefficient of the poled crystal for the chosen cut and polarisation (d33 for x-cut TE or
    z-cut TM), not the QPM-reduced 2d/π: the poling function g(z) supplies that factor."""
    require_positive(wavelength=wavelength, n_pump=n_pump, n_sh=n_sh, d_coeff=d_coeff, a_eff=a_eff)
    eta0 = 8 * np.pi**2 * np.asarray(d_coeff, float) ** 2 / (_EPS0 * _C * n_pump**2 * n_sh * wavelength**2 * a_eff)
    return Result(
        values={"kappa": np.sqrt(eta0), "eta_norm": eta0},
        units={"kappa": "1/(m W^0.5)", "eta_norm": "1/(W m^2)"},
        assumptions=[
            "CW plane-wave-in-mode approximation; A_eff = 1/|Γ|² from the overlap of the normalised mode fields",
            "Kleinman symmetry; d is the full-strength coefficient, the QPM factor comes from g(z)",
        ],
    )


# ----------------------------------------------------------------------------- poling

def periodic_domains(length, period, duty_cycle=0.5, offset=0.0):
    """Boundaries and signs of an ordered ±1 grating on [0, L] (helper, plain arrays).

    Each period starts with the +1 domain of length D·Λ; `offset` shifts the grating by that
    distance along z (the first, partial domain is kept). Returns (boundaries (N+1,), g (N,))."""
    require_positive(length=length, period=period)
    require_range("duty_cycle", duty_cycle, 1e-9, 1 - 1e-9)
    nmin = math.floor((0.0 - offset) / period) - 1
    nmax = math.ceil((length - offset) / period) + 1
    n = np.arange(nmin, nmax + 1)
    z_up = offset + n * period
    z_dn = z_up + duty_cycle * period
    walls = np.sort(np.concatenate([z_up, z_dn]))
    sign_after = np.where(np.isin(walls, z_up), 1.0, -1.0)
    inside = (walls > 0) & (walls < length)
    first = sign_after[walls <= 0][-1]
    b = np.concatenate([[0.0], walls[inside], [float(length)]])
    g = np.concatenate([[first], sign_after[inside]])
    return b, g


def domains_from_lengths(lengths, first_sign=1.0):
    """Boundaries and alternating signs from a list of domain lengths (helper)."""
    ell = np.asarray(lengths, dtype=float)
    require_positive(lengths=ell)
    b = np.concatenate([[0.0], np.cumsum(ell)])
    g = float(first_sign) * (-1.0) ** np.arange(ell.size)
    return b, g


def _check_domains(boundaries, g):
    b = np.asarray(boundaries, dtype=float)
    gg = np.asarray(g, dtype=float)
    if b.ndim != 1 or b.size < 2 or b[0] != 0.0:
        raise ValueError("boundaries must be a 1D array starting at 0 with at least two entries")
    if np.any(np.diff(b) <= 0) or not np.all(np.isfinite(b)):
        raise ValueError("boundaries must be strictly increasing and finite")
    if gg.shape != (b.size - 1,):
        raise ValueError(f"g must have one value per domain ({b.size - 1}), got shape {gg.shape}")
    require_range("g", gg, -1.0, 1.0)
    return b, gg


def _dk_table(delta_k, n_domains):
    """Δk as a (M, N) table: per wavelength, per domain."""
    dk = np.asarray(delta_k, dtype=float)
    if dk.ndim == 0:
        dk = dk.reshape(1)
    if dk.ndim == 1:
        dk = np.repeat(dk[:, None], n_domains, axis=1)
    if dk.ndim != 2 or dk.shape[1] != n_domains:
        raise ValueError(f"delta_k must be scalar, (M,) or (M, {n_domains}), got shape {np.shape(delta_k)}")
    if not np.all(np.isfinite(dk)):
        raise ValueError("delta_k must be finite")
    return dk


def _expm1_over(x):
    """(e^x - 1)/x for complex x, accurate near 0."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 1e-5
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 + x / 2 + x * x / 6, (np.exp(xs) - 1) / xs)


# ----------------------------------------------------------------------------- undepleted

def poling_spectrum(boundaries, g, delta_k, gain=0.0):
    """G(Δk) = ∫0^L g(z) e^{iφ(z) + gain·z} dz, exact per domain (helper).

    Returns shape (M,). `gain` is a real rate (1/m) used for loss: α2/2 - α1."""
    b, gg = _check_domains(boundaries, g)
    ell = np.diff(b)
    dk = _dk_table(delta_k, ell.size)
    q = 1j * dk + gain
    expo = np.concatenate([np.zeros((dk.shape[0], 1), complex), np.cumsum(q * ell, axis=1)[:, :-1]], axis=1)
    return np.sum(gg * np.exp(expo) * ell * _expm1_over(q * ell), axis=1)


def shg_undepleted(boundaries, g, delta_k, kappa, power, alpha_pump=0.0, alpha_sh=0.0):
    """Undepleted-pump SH amplitude and power at z = L, exact per domain (helper, vectorised over Δk).

    A2(L) = i κ A1(0)² e^{-α2 L/2} ∫ g e^{iφ - (α1 - α2/2) z} dz. Returns dict of arrays."""
    require_positive(kappa=kappa, power=power)
    require_nonnegative(alpha_pump=alpha_pump, alpha_sh=alpha_sh)
    b = np.asarray(boundaries, dtype=float)
    L = b[-1]
    G = poling_spectrum(b, g, delta_k, gain=alpha_sh / 2 - alpha_pump)
    a2 = 1j * kappa * power * np.exp(-alpha_sh * L / 2) * G
    return {"A_sh": a2, "P_sh": np.abs(a2) ** 2, "P_pump": power * np.exp(-alpha_pump * L) * np.ones(G.shape), "G": G}


def shg_depleted_matched(kappa, power, length):
    """Closed form for a uniform, lossless, phase-matched medium: P_SH = P tanh²(κ √P L) (helper)."""
    return power * np.tanh(kappa * np.sqrt(power) * length) ** 2


# ----------------------------------------------------------------------------- full solver

def _rhs(a1, a2, gk, phase, kappa, al1, al2):
    e = np.exp(1j * phase)
    return (-0.5 * al1 * a1 + 1j * kappa * gk * a2 * np.conj(a1) * np.conj(e),
            -0.5 * al2 * a2 + 1j * kappa * gk * a1 * a1 * e)


def _rk4(a1, a2, h, gk, ph0, dkk, kappa, al1, al2):
    k1 = _rhs(a1, a2, gk, ph0, kappa, al1, al2)
    pm = ph0 + dkk * h / 2
    k2 = _rhs(a1 + h / 2 * k1[0], a2 + h / 2 * k1[1], gk, pm, kappa, al1, al2)
    k3 = _rhs(a1 + h / 2 * k2[0], a2 + h / 2 * k2[1], gk, pm, kappa, al1, al2)
    k4 = _rhs(a1 + h * k3[0], a2 + h * k3[1], gk, ph0 + dkk * h, kappa, al1, al2)
    return (a1 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            a2 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def _default_step(dk, kappa, power, sh_power):
    dkmax = float(np.max(np.abs(dk))) if dk.size else 0.0
    h_phase = np.pi / 8 / dkmax if dkmax > 0 else np.inf
    h_nl = 0.02 / (kappa * math.sqrt(power + sh_power))
    return min(h_phase, h_nl)


def _integrate(b, gg, dk, kappa, a1, a2, al1, al2, h_max, min_steps, stepping):
    ell = np.diff(b)
    if stepping == "walls":
        ph = np.zeros(dk.shape[0])
        for k in range(ell.size):
            n = max(min_steps, math.ceil(ell[k] / h_max - 1e-9))
            h = ell[k] / n
            dkk = dk[:, k]
            for j in range(n):
                a1, a2 = _rk4(a1, a2, h, gg[k], ph + dkk * j * h, dkk, kappa, al1, al2)
            ph = ph + dkk * ell[k]
        return a1, a2, int(sum(max(min_steps, math.ceil(x / h_max - 1e-9)) for x in ell))
    # uniform grid that ignores the walls: g is sampled once per step at the step centre (the
    # usual piecewise-constant discretisation), so a step that straddles a wall uses one sign
    L = b[-1]
    n = max(min_steps, math.ceil(L / h_max - 1e-9))
    h = L / n
    phi_b = np.concatenate([np.zeros((dk.shape[0], 1)), np.cumsum(dk * ell, axis=1)], axis=1)

    def phase(z):
        k = min(int(np.searchsorted(b, z, side="right")) - 1, ell.size - 1)
        return phi_b[:, k] + dk[:, k] * (z - b[k])

    for j in range(n):
        z0 = j * h
        gm = gg[min(int(np.searchsorted(b, z0 + h / 2, side="right")) - 1, ell.size - 1)]
        p0, pm, p1 = phase(z0), phase(z0 + h / 2), phase(z0 + h)
        k1 = _rhs(a1, a2, gm, p0, kappa, al1, al2)
        k2 = _rhs(a1 + h / 2 * k1[0], a2 + h / 2 * k1[1], gm, pm, kappa, al1, al2)
        k3 = _rhs(a1 + h / 2 * k2[0], a2 + h / 2 * k2[1], gm, pm, kappa, al1, al2)
        k4 = _rhs(a1 + h * k3[0], a2 + h * k3[1], gm, p1, kappa, al1, al2)
        a1 = a1 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        a2 = a2 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return a1, a2, n


def shg_coupled(boundaries, g, delta_k, kappa, power, alpha_pump=0.0, alpha_sh=0.0, sh_power=0.0,
                max_step=None, min_steps=4, stepping="walls", convergence_check=False):
    """Full (depleted) coupled-wave SHG by fixed-step RK4, vectorised over the rows of Δk (helper).

    stepping="walls": every domain is integrated separately, so no step straddles a domain wall
    (exact treatment of discontinuous g). stepping="uniform": one grid over the whole device that
    ignores the walls (reference for the naive scheme). The default step resolves both the phase
    (|Δk| h ≤ π/8) and the nonlinear length (κ √P h ≤ 0.02).

    Returns dict: A_pump, A_sh, P_pump, P_sh (arrays over M), power_residual (lossless only:
    (P_pump + P_sh)/(P_in + P_sh,in) - 1), steps, and with convergence_check the relative change
    of P_sh when the step is halved."""
    require_positive(kappa=kappa, power=power)
    require_nonnegative(alpha_pump=alpha_pump, alpha_sh=alpha_sh, sh_power=sh_power)
    require_choice("stepping", stepping, ("walls", "uniform"))
    b, gg = _check_domains(boundaries, g)
    dk = _dk_table(delta_k, b.size - 1)
    h_max = float(max_step) if max_step is not None else _default_step(dk, kappa, power, sh_power)
    require_positive(max_step=h_max)
    m = dk.shape[0]
    a10 = np.full(m, math.sqrt(power), dtype=complex)
    a20 = np.full(m, math.sqrt(sh_power), dtype=complex)
    a1, a2, steps = _integrate(b, gg, dk, kappa, a10, a20, alpha_pump, alpha_sh, h_max, min_steps, stepping)
    out = {"A_pump": a1, "A_sh": a2, "P_pump": np.abs(a1) ** 2, "P_sh": np.abs(a2) ** 2, "steps": steps}
    if alpha_pump == 0 and alpha_sh == 0:
        out["power_residual"] = (out["P_pump"] + out["P_sh"]) / (power + sh_power) - 1.0
    else:
        out["power_residual"] = np.full(m, np.nan)
    if convergence_check:
        _, a2h, _ = _integrate(b, gg, dk, kappa, a10, a20, alpha_pump, alpha_sh, h_max / 2, 2 * min_steps, stepping)
        ph = np.abs(a2h) ** 2
        scale = np.maximum(ph, 1e-300)
        out["step_error"] = np.abs(ph - out["P_sh"]) / scale
    return out


# ----------------------------------------------------------------------------- spec function

def ordered_shg(wavelength, n_pump, n_sh, d_coeff, a_eff, length, power, period_offset=0.0,
                duty_cycle=0.5, loss_pump=0.0, loss_sh=0.0) -> Result:
    """Ordered first-order QPM SHG: matched period, closed-form and coupled-wave SH power.

    The period is Λ = 2π/Δk + period_offset; loss_pump and loss_sh are power losses in dB/m."""
    require_positive(wavelength=wavelength, n_pump=n_pump, n_sh=n_sh, length=length, power=power)
    require_nonnegative(loss_pump=loss_pump, loss_sh=loss_sh)
    dk = 4 * np.pi * (n_sh - n_pump) / wavelength
    if dk == 0:
        raise ValueError("n_sh equals n_pump: no QPM period is needed (birefringent or modal matching)")
    period = 2 * np.pi / abs(dk) + period_offset
    require_positive(period=period)
    cp = coupling_shg(wavelength, n_pump, n_sh, d_coeff, a_eff)
    kappa, eta0 = float(cp["kappa"]), float(cp["eta_norm"])
    al1, al2 = DB_PER_M * loss_pump, DB_PER_M * loss_sh
    b, g = periodic_domains(length, period, duty_cycle)
    und = shg_undepleted(b, g, abs(dk), kappa, power, al1, al2)
    full = shg_coupled(b, g, abs(dk), kappa, power, al1, al2, convergence_check=True)
    p_und, p_sh = float(und["P_sh"][0]), float(full["P_sh"][0])
    dk_res = abs(dk) - 2 * np.pi / period
    strength = abs(2 * np.sin(np.pi * duty_cycle) / np.pi)
    p_closed = eta0 * strength**2 * power**2 * length**2 * float(np.sinc(dk_res * length / 2 / np.pi)) ** 2
    assumptions = cp.assumptions + [
        "CW, single spatial mode per wave, slowly varying amplitudes",
        "Undepleted value exact per domain; coupled-wave value by RK4 with steps that never straddle a domain wall",
        "Closed form (2 sin πD / π)² η0 P² L² sinc²(Δk' L/2) ignores loss and the partial last period",
    ]
    if p_sh / power > DEPLETION_FLAG:
        assumptions.append(f"Pump depletion {100 * p_sh / power:.1f} %: the undepleted value overestimates P_SH")
    res = float(full["power_residual"][0])
    return Result(
        values={
            "delta_k": dk, "period": period, "domains": int(g.size), "eta_norm": eta0, "kappa": kappa,
            "p_sh_closed": p_closed, "p_sh_undepleted": p_und, "p_sh": p_sh, "p_pump_out": float(full["P_pump"][0]),
            "conversion": p_sh / power, "efficiency": p_sh / power**2,
            "power_residual": res, "step_error": float(full["step_error"][0]),
        },
        units={
            "delta_k": "1/m", "period": "m", "domains": "", "eta_norm": "1/(W m^2)", "kappa": "1/(m W^0.5)",
            "p_sh_closed": "W", "p_sh_undepleted": "W", "p_sh": "W", "p_pump_out": "W", "conversion": "",
            "efficiency": "1/W", "power_residual": "", "step_error": "",
        },
        assumptions=assumptions,
    )
