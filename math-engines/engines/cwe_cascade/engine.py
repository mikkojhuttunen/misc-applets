"""Cascaded third-harmonic generation: ω + ω → 2ω (SHG, "a") then ω + 2ω → 3ω (SFG, "b") on one poling g(z).

Fields 1 = ω, 2 = 2ω, 3 = 3ω, |A|² in W. The SHG terms are those of ``cwe_shg``, the SFG terms those of
``cwe_dfg`` with (signal, idler, pump) = (ω, 2ω, 3ω):

    dA1/dz = -α1/2 A1 + i κ_a g A2 A1* e^{-iφa} + i κ_1 g A3 A2* e^{-iφb}
    dA2/dz = -α2/2 A2 + i κ_a g A1²   e^{+iφa} + i κ_2 g A3 A1* e^{-iφb}
    dA3/dz = -α3/2 A3 + i κ_3 g A1 A2  e^{+iφb}
    Δk_a = β2 - 2β1,  Δk_b = β3 - β2 - β1,  φ = ∫ Δk dz  (grating vectors not subtracted: g carries them)

κ_a² = η0 of SHG (cwe_shg.coupling_shg), κ_3² = η_SFG of ω + 2ω (cwe_dfg.coupling_three_wave), κ_1 : κ_2 : κ_3
= 1 : 2 : 3. Each process has its own d (Miller scaling) and overlap area (the 3ω mode may be a higher order).
Both processes conserve power separately, so the lossless sum P1 + P2 + P3 is conserved.

Undepleted ω and lowest order in the cascade:
    A2(z) = i κ_a A1² ∫0^z g e^{iφa} dz',   A3(L) = -κ_3 κ_a A1³ ∫0^L g(z) e^{iφb(z)} ∫0^z g(z') e^{iφa(z')} dz' dz
evaluated exactly per domain (``cascade_undepleted``); loss enters as complex rates.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range
from ..cwe_dfg.engine import coupling_three_wave
from ..cwe_shg.engine import (DB_PER_M, _check_domains, _dk_table, _expm1_over, coupling_shg,  # noqa: F401
                              domains_from_lengths, periodic_domains)


# ----------------------------------------------------------------------------- coupling

def coupling_thg(wavelength, n1, n2, n3, d_shg, d_sfg, a_eff_shg, a_eff_sfg) -> Result:
    """κ_a (SHG ω+ω→2ω) and κ_1, κ_2, κ_3 (SFG ω+2ω→3ω) at fundamental vacuum wavelength λ."""
    require_positive(wavelength=wavelength, n1=n1, n2=n2, n3=n3, d_shg=d_shg, d_sfg=d_sfg,
                     a_eff_shg=a_eff_shg, a_eff_sfg=a_eff_sfg)
    a = coupling_shg(wavelength, n1, n2, d_shg, a_eff_shg)
    b = coupling_three_wave(wavelength, wavelength / 2, n1, n2, n3, d_sfg, a_eff_sfg)
    ka, k1, k2, k3 = float(a["kappa"]), float(b["kappa_signal"]), float(b["kappa_idler"]), float(b["kappa_pump"])
    return Result(
        values={"kappa_a": ka, "kappa_1": k1, "kappa_2": k2, "kappa_3": k3, "eta_shg": ka**2, "eta_sfg": k3**2,
                "eta_thg_matched": (k3 * ka) ** 2 / 4},
        units={"kappa_a": "1/(m W^0.5)", "kappa_1": "1/(m W^0.5)", "kappa_2": "1/(m W^0.5)", "kappa_3": "1/(m W^0.5)",
               "eta_shg": "1/(W m^2)", "eta_sfg": "1/(W m^2)", "eta_thg_matched": "1/(W^2 m^4)"},
        assumptions=["CW, one spatial mode per wave; full-strength d per process (QPM factor from g(z))",
                     "eta_thg_matched: P3 = η P³ L⁴ for a uniform medium with both processes matched (Δk_a = Δk_b = 0), undepleted"],
    )


# ----------------------------------------------------------------------------- undepleted cascade

def _m1(x):
    """∫0^1 t e^{xt} dt for complex x, accurate near 0."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(small, 0.5 + x / 3 + x * x / 8 + x**3 / 30, ((xs - 1) * np.exp(xs) + 1) / xs**2)


def _m2(x):
    """∫0^1 t² e^{xt} dt."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 1e-2
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 / 3 + x / 4 + x * x / 10 + x**3 / 36,
                    (np.exp(xs) * (xs * xs - 2 * xs + 2) - 2) / xs**3)


def _inner_outer(qa, qb, ell):
    """F = ∫0^ℓ e^{i qb s} (e^{i qa s} - 1)/(i qa) ds and E_b = ∫0^ℓ e^{i qb s} ds (complex q allowed)."""
    eb = ell * _expm1_over(1j * qb * ell)
    small = np.abs(qa * ell) < 1e-3
    qa_s = np.where(small, 1.0, qa)
    big = (ell * _expm1_over(1j * (qa_s + qb) * ell) - eb) / (1j * qa_s)
    series = ell**2 * _m1(1j * qb * ell) + 0.5j * qa * ell**3 * _m2(1j * qb * ell)
    return np.where(small, series, big), eb


def cascade_integral(boundaries, g, delta_k_a, delta_k_b, gain_a=0.0, gain_b=0.0):
    """Nested integral J = ∫0^L g e^{iφb + γb z} ∫0^z g e^{iφa + γa z'} dz' dz, exact per domain (helper).

    Returns (J, G_a) with G_a = ∫0^L g e^{iφa + γa z} dz, arrays over the rows of the Δk tables. γ are real
    rates (loss): γa = α2/2 - α1, γb = α3/2 - α1/2 - α2/2."""
    b, gg = _check_domains(boundaries, g)
    ell = np.diff(b)
    n = ell.size
    qa = _dk_table(delta_k_a, n) - 1j * gain_a
    qb = _dk_table(delta_k_b, n) - 1j * gain_b
    if qa.shape != qb.shape:
        qa, qb = np.broadcast_arrays(qa, qb)
    m = qa.shape[0]
    pa = np.zeros(m, complex)          # accumulated exponent of the inner integrand at the domain start
    pb = np.zeros(m, complex)
    inner = np.zeros(m, complex)       # ∫0^{z_k} ...
    J = np.zeros(m, complex)
    for k in range(n):
        F, eb = _inner_outer(qa[:, k], qb[:, k], ell[k])
        ea = ell[k] * _expm1_over(1j * qa[:, k] * ell[k])
        J += gg[k] * np.exp(1j * pb) * (inner * eb + gg[k] * np.exp(1j * pa) * F)
        inner += gg[k] * np.exp(1j * pa) * ea
        pa += qa[:, k] * ell[k]
        pb += qb[:, k] * ell[k]
    return J, inner


def cascade_undepleted(boundaries, g, delta_k_a, delta_k_b, kappa_a, kappa_3, power, alpha=(0.0, 0.0, 0.0)):
    """Undepleted ω, lowest-order cascade: A2(L), A3(L), P2, P3 (helper; κ may be arrays per row)."""
    require_positive(power=power)
    a1, a2, a3 = alpha
    require_nonnegative(alpha=list(alpha))
    J, Ga = cascade_integral(boundaries, g, delta_k_a, delta_k_b, a2 / 2 - a1, a3 / 2 - a1 / 2 - a2 / 2)
    L = float(np.asarray(boundaries)[-1])
    A2 = 1j * np.asarray(kappa_a) * power * np.exp(-a2 * L / 2) * Ga
    A3 = -np.asarray(kappa_3) * np.asarray(kappa_a) * power**1.5 * np.exp(-a3 * L / 2) * J
    return {"A_sh": A2, "A_th": A3, "P_sh": np.abs(A2) ** 2, "P_th": np.abs(A3) ** 2, "J": J, "G_a": Ga}


# ----------------------------------------------------------------------------- full solver

def _rhs(a1, a2, a3, gk, pha, phb, ka, k1, k2, k3, al):
    ea, eb = np.exp(1j * pha), np.exp(1j * phb)
    c1, c2, c3 = np.conj(a1), np.conj(a2), np.conj(eb)
    return (-0.5 * al[0] * a1 + 1j * gk * (ka * a2 * c1 * np.conj(ea) + k1 * a3 * c2 * c3),
            -0.5 * al[1] * a2 + 1j * gk * (ka * a1 * a1 * ea + k2 * a3 * c1 * c3),
            -0.5 * al[2] * a3 + 1j * gk * k3 * a1 * a2 * eb)


def _run(b, gg, dka, dkb, kap, a, al, h_max, min_steps):
    ell = np.diff(b)
    pa = np.zeros(dka.shape[0])
    pb = np.zeros(dkb.shape[0])
    a1, a2, a3 = a
    steps = 0
    for k in range(ell.size):
        n = max(min_steps, math.ceil(ell[k] / h_max - 1e-9))
        h = ell[k] / n
        qa, qb = dka[:, k], dkb[:, k]
        for j in range(n):
            s0 = j * h
            r1 = _rhs(a1, a2, a3, gg[k], pa + qa * s0, pb + qb * s0, *kap, al)
            sm = s0 + h / 2
            r2 = _rhs(a1 + h / 2 * r1[0], a2 + h / 2 * r1[1], a3 + h / 2 * r1[2], gg[k], pa + qa * sm, pb + qb * sm, *kap, al)
            r3 = _rhs(a1 + h / 2 * r2[0], a2 + h / 2 * r2[1], a3 + h / 2 * r2[2], gg[k], pa + qa * sm, pb + qb * sm, *kap, al)
            s1 = s0 + h
            r4 = _rhs(a1 + h * r3[0], a2 + h * r3[1], a3 + h * r3[2], gg[k], pa + qa * s1, pb + qb * s1, *kap, al)
            a1 = a1 + h / 6 * (r1[0] + 2 * r2[0] + 2 * r3[0] + r4[0])
            a2 = a2 + h / 6 * (r1[1] + 2 * r2[1] + 2 * r3[1] + r4[1])
            a3 = a3 + h / 6 * (r1[2] + 2 * r2[2] + 2 * r3[2] + r4[2])
        steps += n
        pa = pa + qa * ell[k]
        pb = pb + qb * ell[k]
    return a1, a2, a3, steps


def cascade_coupled(boundaries, g, delta_k_a, delta_k_b, kappa_a, kappa_1, kappa_2, kappa_3, power,
                    alpha=(0.0, 0.0, 0.0), max_step=None, min_steps=4, convergence_check=False):
    """Full three-field cascade (depletion, back-conversion, loss) by wall-aware RK4 (helper)."""
    require_positive(kappa_a=kappa_a, kappa_1=kappa_1, kappa_2=kappa_2, kappa_3=kappa_3, power=power)
    require_nonnegative(alpha=list(alpha))
    b, gg = _check_domains(boundaries, g)
    dka, dkb = _dk_table(delta_k_a, b.size - 1), _dk_table(delta_k_b, b.size - 1)
    dka, dkb = np.broadcast_arrays(dka, dkb)
    dkmax = float(max(np.max(np.abs(dka)), np.max(np.abs(dkb))))
    kmax = max(kappa_a, kappa_1, kappa_2, kappa_3)
    h = min(np.pi / 8 / dkmax if dkmax > 0 else np.inf, 0.02 / (kmax * math.sqrt(power)))
    h = float(max_step) if max_step is not None else h
    m = dka.shape[0]
    a0 = (np.full(m, math.sqrt(power), complex), np.zeros(m, complex), np.zeros(m, complex))
    kap = (kappa_a, kappa_1, kappa_2, kappa_3)
    a1, a2, a3, steps = _run(b, gg, dka, dkb, kap, a0, alpha, h, min_steps)
    out = {"A1": a1, "A2": a2, "A3": a3, "P1": np.abs(a1) ** 2, "P2": np.abs(a2) ** 2, "P3": np.abs(a3) ** 2, "steps": steps}
    if not any(alpha):
        out["power_residual"] = (out["P1"] + out["P2"] + out["P3"]) / power - 1
    if convergence_check:
        a3h = _run(b, gg, dka, dkb, kap, a0, alpha, h / 2, 2 * min_steps)[2]
        p = np.abs(a3h) ** 2
        out["step_error"] = np.abs(p - out["P3"]) / np.maximum(p, 1e-300)
    return out


# ----------------------------------------------------------------------------- design helpers

def two_section_domains(length, period_a, period_b, split=0.5, duty_cycle=0.5):
    """Ordered SHG grating (period Λa) on [0, split·L] followed by an SFG grating (Λb) on the rest."""
    require_range("split", split, 1e-6, 1 - 1e-6)
    la = split * length
    ba, ga = periodic_domains(la, period_a, duty_cycle)
    bb, gb = periodic_domains(length - la, period_b, duty_cycle)
    return np.concatenate([ba, la + bb[1:]]), np.concatenate([ga, gb])


def commensurate_orders(delta_k_a, delta_k_b, max_order=9, duty_cycles=np.linspace(0.05, 0.95, 91)) -> list[dict]:
    """Single periods that match both processes with integer orders: Λ = 2π m_a/Δk_a, residual Δk_b - 2π m_b/Λ.

    For each (m_a, m_b) the duty cycle maximising |c_ma c_mb| (c_m = 2 sin(mπD)/(mπ)) is chosen; sorted by
    |residual|. A residual below ~2.78/L keeps the SFG step inside its sinc² FWHM."""
    out = []
    for ma in range(1, max_order + 1):
        lam_ = 2 * np.pi * ma / abs(delta_k_a)
        for mb in range(1, max_order + 1):
            res = abs(delta_k_b) - 2 * np.pi * mb / lam_
            prod = np.abs(2 * np.sin(ma * np.pi * duty_cycles) / (ma * np.pi) * 2 * np.sin(mb * np.pi * duty_cycles) / (mb * np.pi))
            i = int(np.argmax(prod))
            out.append({"m_a": ma, "m_b": mb, "period": lam_, "residual": float(res), "duty_cycle": float(duty_cycles[i]),
                        "strength": float(prod[i])})
    return sorted(out, key=lambda r: abs(r["residual"]))


# ----------------------------------------------------------------------------- spec function

def two_section_thg(wavelength, n1, n2, n3, d_shg, d_sfg, a_eff_shg, a_eff_sfg, length, power, split=0.5,
                    loss_1=0.0, loss_2=0.0, loss_3=0.0) -> Result:
    """Two ordered sections (SHG then SFG) at their matched periods: closed form, exact undepleted, coupled waves.

    Losses in dB/m. Closed form P3 = (2/π)⁴ κ3² κa² P³ La² Lb² (each section matches only its own process)."""
    require_positive(wavelength=wavelength, length=length, power=power)
    dka = 4 * np.pi * (n2 - n1) / wavelength
    dkb = 2 * np.pi * (3 * n3 - 2 * n2 - n1) / wavelength
    if dka == 0 or dkb == 0:
        raise ValueError("a process is already phase matched without a grating")
    pa, pb = 2 * np.pi / abs(dka), 2 * np.pi / abs(dkb)
    c = coupling_thg(wavelength, n1, n2, n3, d_shg, d_sfg, a_eff_shg, a_eff_sfg)
    ka, k1, k2, k3 = (float(c[k]) for k in ("kappa_a", "kappa_1", "kappa_2", "kappa_3"))
    b, g = two_section_domains(length, pa, pb, split)
    al = (DB_PER_M * loss_1, DB_PER_M * loss_2, DB_PER_M * loss_3)
    und = cascade_undepleted(b, g, abs(dka), abs(dkb), ka, k3, power, al)
    full = cascade_coupled(b, g, abs(dka), abs(dkb), ka, k1, k2, k3, power, al, convergence_check=True)
    la, lb = split * length, (1 - split) * length
    closed = (2 / np.pi) ** 4 * k3**2 * ka**2 * power**3 * la**2 * lb**2
    p3 = float(full["P3"][0])
    res = full.get("power_residual")
    assumptions = c.assumptions + [
        "Undepleted: exact nested per-domain integral; coupled waves: RK4 with depletion, back-conversion and loss",
        "Closed form ignores loss, the cross terms of each section on the other process and partial periods",
    ]
    if float(full["P2"][0] + p3) / power > 0.03:
        assumptions.append("Fundamental depletion above 3 %: the undepleted value overestimates the output")
    return Result(
        values={"delta_k_a": dka, "delta_k_b": dkb, "period_a": pa, "period_b": pb, "p_th_closed": closed,
                "p_th_undepleted": float(und["P_th"][0]), "p_th": p3, "p_sh": float(full["P2"][0]),
                "efficiency": p3 / power**3, "eta_thg_matched": float(c["eta_thg_matched"]),
                "power_residual": float(res[0]) if res is not None else float("nan"),
                "step_error": float(full["step_error"][0])},
        units={"delta_k_a": "1/m", "delta_k_b": "1/m", "period_a": "m", "period_b": "m", "p_th_closed": "W",
               "p_th_undepleted": "W", "p_th": "W", "p_sh": "W", "efficiency": "1/W^2", "eta_thg_matched": "1/(W^2 m^4)",
               "power_residual": "", "step_error": ""},
        assumptions=assumptions,
    )
