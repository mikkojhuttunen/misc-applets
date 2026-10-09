"""Heating along an Er-doped waveguide amplifier: pump depletion, the heat it leaves behind, ΔT(z) and Δn_eff(z).

Er³⁺ as an effective two-level system (the model of er-waveguide-amplifier.html, with overlap-averaged intensities
I = Γ P / A_d over the doped area A_d instead of the per-cell mode profile):

    0 = (R_p,a + R_s,a) N₁ - (R_p,e + R_s,e) N₂ - N₂/τ - C_up N₂²,   R = σ Γ P / (A_d hν),   N₁ + N₂ = N_active
    dP_p/dz = -[Γ_p (σ_a,p n₁ - σ_e,p N₂) + α] P_p,   dP_s/dz = [Γ_s (σ_e,s N₂ - σ_a,s n₁) - α] P_s

n₁ = N₁ + N_q includes the quenched ions (fraction f_q), which only absorb. 980 nm: σ_e,p = 0 (⁴I₁₁/₂ empties fast).

Heat per unit length is everything the ions absorb minus what they re-radiate spontaneously:

    q'_Er = Γ_p (σ_a,p n₁ - σ_e,p N₂) P_p + Γ_s (σ_a,s n₁ - σ_e,s N₂) P_s - η_rad (N₂/τ) hν_s A_d
          = A_d [ R_p,net (hν_p - hν_s) + (1 - η_rad) N₂/τ hν_s + C_up N₂² hν_s ] + (quenched-ion absorption)

(quantum defect, non-radiative decay, upconversion, quenching), plus f_abs α (P_p + P_s) from the absorbing part of
the background loss. Spontaneous emission (and ASE) is assumed to leave the waveguide. ΔT(z) = R' q'(z): the local
2D thermal resistance, valid where q' changes slowly over a few substrate thicknesses (no axial conduction).
SI units: m, W, s, m², m³.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range

H = 6.62607015e-34
C0 = 299792458.0
DB = math.log(10) / 10


def _populations(Pp, Ps, p):
    Ip, Is = p["Gp"] * Pp / p["Ad"], p["Gs"] * Ps / p["Ad"]
    Ra = p["sap"] * Ip / p["hvp"] + p["sas"] * Is / p["hvs"]
    Re = p["sep"] * Ip / p["hvp"] + p["ses"] * Is / p["hvs"]
    N = p["N"]
    b = Ra + Re + 1 / p["tau"]
    n2 = 2 * Ra * N / (b + math.sqrt(b * b + 4 * p["Cup"] * Ra * N)) if Ra > 0 else 0.0
    return N - n2 + p["Nq"], n2


def _rhs(Pp, Ps, p):
    n1, n2 = _populations(Pp, Ps, p)
    gp = p["Gp"] * (p["sap"] * n1 - p["sep"] * n2)
    gs = p["Gs"] * (p["ses"] * n2 - p["sas"] * n1)
    return -(gp + p["al"]) * Pp, (gs - p["al"]) * Ps


def _heat(Pp, Ps, p):
    n1, n2 = _populations(Pp, Ps, p)
    absorbed = p["Gp"] * (p["sap"] * n1 - p["sep"] * n2) * Pp + p["Gs"] * (p["sas"] * n1 - p["ses"] * n2) * Ps
    radiated = p["eta"] * n2 / p["tau"] * p["hvs"] * p["Ad"]
    return absorbed - radiated + p["fabs"] * p["al"] * (Pp + Ps), n1, n2


def _params(pump_wavelength, signal_wavelength, n_er, quenched_fraction, tau, c_up, sigma_a_pump, sigma_e_pump,
            sigma_a_signal, sigma_e_signal, gamma_pump, gamma_signal, doped_area, loss_db_per_cm, absorbing_fraction, eta_rad):
    return dict(hvp=H * C0 / pump_wavelength, hvs=H * C0 / signal_wavelength, N=n_er * (1 - quenched_fraction),
                Nq=n_er * quenched_fraction, tau=tau, Cup=c_up, sap=sigma_a_pump, sep=sigma_e_pump, sas=sigma_a_signal,
                ses=sigma_e_signal, Gp=gamma_pump, Gs=gamma_signal, Ad=doped_area, al=loss_db_per_cm * 100 * DB,
                fabs=absorbing_fraction, eta=eta_rad)


def propagate(pump_power, signal_power, length, n_z, p):
    """RK4 along z; returns z, P_p, P_s, q', N₂/N_total arrays."""
    dz = length / n_z
    z = np.linspace(0.0, length, n_z + 1)
    Pp, Ps = np.empty(n_z + 1), np.empty(n_z + 1)
    Pp[0], Ps[0] = pump_power, signal_power
    for i in range(n_z):
        a, b = Pp[i], Ps[i]
        k1 = _rhs(a, b, p)
        k2 = _rhs(a + 0.5 * dz * k1[0], b + 0.5 * dz * k1[1], p)
        k3 = _rhs(a + 0.5 * dz * k2[0], b + 0.5 * dz * k2[1], p)
        k4 = _rhs(a + dz * k3[0], b + dz * k3[1], p)
        Pp[i + 1] = max(0.0, a + dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
        Ps[i + 1] = max(0.0, b + dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
    hq = [_heat(a, b, p) for a, b in zip(Pp, Ps)]
    q = np.array([h[0] for h in hq])
    inv = np.array([h[2] for h in hq]) / (p["N"] + p["Nq"])
    return z, Pp, Ps, q, inv


_DEFAULTS = dict(pump_wavelength=0.98e-6, signal_wavelength=1.532e-6, n_er=1.5e26, quenched_fraction=0.0, tau=7.5e-3,
                 c_up=4e-24, sigma_a_pump=1.7e-25, sigma_e_pump=0.0, sigma_a_signal=5.7e-25, sigma_e_signal=5.7e-25,
                 gamma_pump=0.5, gamma_signal=0.4, doped_area=0.8e-12, loss_db_per_cm=0.25, absorbing_fraction=0.5,
                 eta_rad=0.8)


def er_amplifier_heating(pump_power=0.2, signal_power=1e-6, length=0.03, R_th=0.5, dneff_dT=3e-5,
                         pump_wavelength=0.98e-6, signal_wavelength=1.532e-6, n_er=1.5e26, quenched_fraction=0.0,
                         tau=7.5e-3, c_up=4e-24, sigma_a_pump=1.7e-25, sigma_e_pump=0.0, sigma_a_signal=5.7e-25,
                         sigma_e_signal=5.7e-25, gamma_pump=0.5, gamma_signal=0.4, doped_area=0.8e-12,
                         loss_db_per_cm=0.25, absorbing_fraction=0.5, eta_rad=0.8, n_z=300) -> Result:
    """Pump and signal along the amplifier, heat per length q'(z), ΔT(z) = R' q'(z), Δn_eff(z), signal gain.

    Defaults: sputtered Al₂O₃:Er (as in er-waveguide-amplifier.html) in a 2 x 0.4 µm strip on TFLN; R' from
    waveguide_thermal.ridge_heating for that cross-section (≈ 0.5 K m/W). Lengths in m, cross-sections in m²,
    n_er in 1/m³, c_up in m³/s."""
    require_positive(length=length, R_th=R_th, pump_wavelength=pump_wavelength, signal_wavelength=signal_wavelength,
                     n_er=n_er, tau=tau, gamma_pump=gamma_pump, gamma_signal=gamma_signal, doped_area=doped_area)
    require_nonnegative(pump_power=pump_power, signal_power=signal_power, c_up=c_up, sigma_a_pump=sigma_a_pump,
                        sigma_e_pump=sigma_e_pump, sigma_a_signal=sigma_a_signal, sigma_e_signal=sigma_e_signal,
                        loss_db_per_cm=loss_db_per_cm)
    for name, v in (("quenched_fraction", quenched_fraction), ("absorbing_fraction", absorbing_fraction), ("eta_rad", eta_rad),
                    ("gamma_pump", gamma_pump), ("gamma_signal", gamma_signal)):
        require_range(name, v, 0.0, 1.0)
    n_z = int(n_z)
    p = _params(pump_wavelength, signal_wavelength, n_er, quenched_fraction, tau, c_up, sigma_a_pump, sigma_e_pump,
                sigma_a_signal, sigma_e_signal, gamma_pump, gamma_signal, doped_area, loss_db_per_cm, absorbing_fraction, eta_rad)
    z, Pp, Ps, q, inv = propagate(pump_power, signal_power, length, n_z, p)
    dT = R_th * q
    heat = float(np.trapezoid(q, z)) if hasattr(np, "trapezoid") else float(np.trapz(q, z))
    p_abs = pump_power - Pp[-1]
    i_hot = int(np.argmax(q))
    return Result(
        values={"gain_dB": 10 * math.log10(Ps[-1] / signal_power) if signal_power > 0 and Ps[-1] > 0 else float("nan"),
                "pump_out": float(Pp[-1]), "pump_absorbed": float(p_abs), "heat_total": heat,
                "heat_fraction": heat / p_abs if p_abs > 0 else float("nan"),
                "q_max": float(q[i_hot]), "z_hot": float(z[i_hot]), "dT_max": float(dT[i_hot]),
                "dT_mean": float(np.mean(dT)), "dneff_max": float(dneff_dT * dT[i_hot]),
                "phase_total": float(2 * math.pi / signal_wavelength * dneff_dT * heat * R_th),
                "inversion_in": float(inv[0]), "inversion_out": float(inv[-1]),
                "z": z, "P_pump": Pp, "P_signal": Ps, "q": q, "dT": dT, "inversion": inv},
        units={"gain_dB": "dB", "pump_out": "W", "pump_absorbed": "W", "heat_total": "W", "heat_fraction": "", "q_max": "W/m",
               "z_hot": "m", "dT_max": "K", "dT_mean": "K", "dneff_max": "", "phase_total": "rad", "inversion_in": "",
               "inversion_out": "", "z": "m", "P_pump": "W", "P_signal": "W", "q": "W/m", "dT": "K", "inversion": ""},
        assumptions=[
            "Effective two-level Er model with overlap-averaged intensities; no ASE, no excited-state absorption",
            "Heat = absorbed power - spontaneous emission radiated out (η_rad of the ⁴I₁₃/₂ decay); upconversion and quenching all heat",
            "ΔT(z) = R' q'(z): local 2D cross-section resistance, no axial heat flow; R' and dn_eff/dT temperature independent",
            "phase_total: thermal phase accumulated by the signal over the whole length, (2π/λ_s) dn_eff/dT ∫ΔT dz",
        ],
    )


def er_pump_limit(dT_max=10.0, dneff_max=1e-3, P_search=20.0, signal_power=1e-6, length=0.03, R_th=0.5, dneff_dT=3e-5,
                  pump_wavelength=0.98e-6, signal_wavelength=1.532e-6, n_er=1.5e26, quenched_fraction=0.0, tau=7.5e-3,
                  c_up=4e-24, sigma_a_pump=1.7e-25, sigma_e_pump=0.0, sigma_a_signal=5.7e-25, sigma_e_signal=5.7e-25,
                  gamma_pump=0.5, gamma_signal=0.4, doped_area=0.8e-12, loss_db_per_cm=0.25, absorbing_fraction=0.5,
                  eta_rad=0.8, n_z=200) -> Result:
    """Largest launched pump power for which the hottest point stays within ΔT_max and |Δn_eff| ≤ Δn_max,
    and the signal gain there. The other arguments are those of er_amplifier_heating."""
    require_positive(dT_max=dT_max, dneff_max=dneff_max, P_search=P_search)
    kwargs = dict(signal_power=signal_power, length=length, R_th=R_th, dneff_dT=dneff_dT, pump_wavelength=pump_wavelength,
                  signal_wavelength=signal_wavelength, n_er=n_er, quenched_fraction=quenched_fraction, tau=tau, c_up=c_up,
                  sigma_a_pump=sigma_a_pump, sigma_e_pump=sigma_e_pump, sigma_a_signal=sigma_a_signal,
                  sigma_e_signal=sigma_e_signal, gamma_pump=gamma_pump, gamma_signal=gamma_signal, doped_area=doped_area,
                  loss_db_per_cm=loss_db_per_cm, absorbing_fraction=absorbing_fraction, eta_rad=eta_rad, n_z=n_z)
    dndT = abs(dneff_dT)

    def over(P):
        r = er_amplifier_heating(pump_power=P, **kwargs)
        return r["dT_max"] > dT_max or abs(r["dneff_max"]) > dneff_max

    if not over(P_search):
        P = P_search
        limited = False
    else:
        lo, hi = 0.0, P_search
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if over(mid) else (mid, hi)
        P, limited = lo, True
    r = er_amplifier_heating(pump_power=P, **kwargs)
    by = "dn_eff" if dndT * dT_max > dneff_max else "dT"
    return Result(
        values={"P_limit": P, "limited": limited, "gain_dB": r["gain_dB"], "dT_max": r["dT_max"], "dneff_max": r["dneff_max"],
                "heat_total": r["heat_total"], "limited_by_dn": by == "dn_eff"},
        units={"P_limit": "W", "limited": "", "gain_dB": "dB", "dT_max": "K", "dneff_max": "", "heat_total": "W", "limited_by_dn": ""},
        assumptions=r.assumptions + [f"Searched 0..{P_search:g} W; limited = False means even P_search stays within budget",
                                     "The hottest point is usually the input facet (undepleted pump)"],
    )
