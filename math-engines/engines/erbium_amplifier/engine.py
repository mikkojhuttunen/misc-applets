"""Erbium-doped waveguide amplifier: cross-sections, inversion and propagation.

Model: Er3+ as an effective two-level system (4I15/2 ground state N1,
4I13/2 metastable level N2). At a 980 nm pump the 4I11/2 level empties fast
into 4I13/2, so the pump only absorbs; in-band pumping (e.g. 1480 nm) also
causes stimulated emission. Energy-transfer upconversion removes excitations
at the rate C_up N2^2. A fraction f_q of the ions is quenched: these decay too
fast to be inverted, so they stay in the ground state and only absorb.

Units: SI throughout. Wavelengths are vacuum wavelengths in m, cross-sections
in m^2, concentrations in m^-3 (1e20 cm^-3 = 1e26 m^-3), intensities in W/m^2,
lengths in m, the background loss alpha is the power attenuation coefficient in
1/m (dB/m = 4.343 alpha). Gains are returned as natural-log power ratios
ln(P_out/P_in); dB = 4.343 ln.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

H = 6.62607015e-34          # Planck constant, J s
C0 = 2.99792458e8           # speed of light, m/s
C2 = H * C0 / 1.380649e-23  # second radiation constant hc/k, m K
N_REF = 1e26                # reference concentration for concentration effects, 1e20 cm^-3 in m^-3
LAMBDA_ZERO = 1.532e-6      # McCumber zero-line wavelength (ε = hc/λ0)

# Smooth model of the Er:Al2O3 4I15/2 -> 4I13/2 absorption band:
# Gaussians (centre, relative height, FWHM), normalised to the peak height sigma_a_peak.
_BANDS = ((1533e-9, 2.9, 11e-9), (1512e-9, 2.35, 58e-9), (1553e-9, 1.15, 30e-9), (1588e-9, 0.4, 50e-9), (1468e-9, 0.7, 40e-9))
_PEAK_GRID = 1400e-9 + 0.25e-9 * np.arange(1201)  # 1400..1700 nm in 0.25 nm steps


def _band_shape(wavelength):
    lam = np.asarray(wavelength, dtype=float)
    s = np.zeros_like(lam)
    for c, a, w in _BANDS:
        q = (lam - c) / w
        s = s + a * np.exp(-4 * np.log(2) * q * q)
    return s


_SHAPE_MAX = float(np.max(_band_shape(_PEAK_GRID)))


def sigma_absorption(wavelength, sigma_a_peak=5.7e-25):
    """Absorption cross-section σ_a(λ) in m^2 (helper, vectorised)."""
    return sigma_a_peak * _band_shape(wavelength) / _SHAPE_MAX


def sigma_emission(wavelength, sigma_a_peak=5.7e-25, temperature=295.0):
    """Emission cross-section from McCumber theory, σ_e = σ_a exp[(hc/λ0 - hc/λ)/kT] (helper)."""
    lam = np.asarray(wavelength, dtype=float)
    return sigma_absorption(lam, sigma_a_peak) * np.exp(C2 / temperature * (1 / LAMBDA_ZERO - 1 / lam))


def pump_sigmas(pump_wavelength, sigma_a_peak=5.7e-25, sigma_a_980=1.7e-25, temperature=295.0):
    """(σ_a, σ_e) at the pump. Within ±30 nm of 980 nm: (sigma_a_980, 0); otherwise the 1.5 µm band model."""
    if abs(pump_wavelength - 980e-9) <= 30e-9:
        return float(sigma_a_980), 0.0
    return float(sigma_absorption(pump_wavelength, sigma_a_peak)), float(sigma_emission(pump_wavelength, sigma_a_peak, temperature))


def cross_sections(wavelength, sigma_a_peak=5.7e-25, temperature=295.0) -> Result:
    """Model absorption and emission cross-sections of Er:Al2O3 in the 1.5 µm band."""
    require_positive(wavelength=wavelength, sigma_a_peak=sigma_a_peak, temperature=temperature)
    return Result(
        values={"sigma_a": sigma_absorption(wavelength, sigma_a_peak), "sigma_e": sigma_emission(wavelength, sigma_a_peak, temperature)},
        units={"sigma_a": "m^2", "sigma_e": "m^2"},
        assumptions=["Smooth five-Gaussian model of the Er:Al2O3 absorption band, peak at 1533 nm, not measured data",
                     "Emission from McCumber theory with λ0 = 1532 nm"],
    )


def concentration_effects(n_er, c_up, upconversion="proportional", quenching="proportional", k_q=0.05, f_q=0.10) -> Result:
    """Effective upconversion coefficient and quenched fraction at concentration n_er.

    upconversion: "fixed" (C_up as given) or "proportional" (C_up · n_er / 1e26 m^-3).
    quenching: "none", "proportional" (f_q = k_q · n_er / 1e26 m^-3) or "fixed" (f_q as given).
    The quenched fraction is capped at 0.95.
    """
    require_positive(n_er=n_er)
    require_nonnegative(c_up=c_up, k_q=k_q, f_q=f_q)
    require_choice("upconversion", upconversion, ("fixed", "proportional"))
    require_choice("quenching", quenching, ("none", "proportional", "fixed"))
    cup = c_up * (n_er / N_REF if upconversion == "proportional" else 1.0)
    fq = k_q * n_er / N_REF if quenching == "proportional" else (f_q if quenching == "fixed" else 0.0)
    fq = min(0.95, fq)
    return Result(
        values={"c_up_eff": cup, "f_q": fq, "n_quenched": fq * n_er, "n_active": (1 - fq) * n_er},
        units={"c_up_eff": "m^3/s", "f_q": "", "n_quenched": "m^-3", "n_active": "m^-3"},
        assumptions=["Linear growth with concentration is a model choice; fit k_q and C_up to lifetime and saturation data"],
    )


def upper_population(rate_up, rate_down, n_active, tau, c_up):
    """Steady-state N2 of 0 = R_up N1 - R_down N2 - N2/τ - C_up N2^2 with N1 = n_active - N2 (helper).

    rate_up = Σ σ_a I/(hν), rate_down = Σ σ_e I/(hν), in 1/s. Uses the root form that stays
    accurate for C_up -> 0, where it reduces to N2 = R_up n_active / (R_up + R_down + 1/τ).
    """
    ru = np.asarray(rate_up, dtype=float)
    b = ru + np.asarray(rate_down, dtype=float) + 1 / tau
    with np.errstate(invalid="ignore", divide="ignore"):
        n2 = 2 * ru * n_active / (b + np.sqrt(b * b + 4 * c_up * ru * n_active))
    return np.where(ru > 0, n2, 0.0)


def _derivatives(pp, ps, cells, par):
    wp, ws, da = cells
    ip, is_ = pp * wp / da, ps * ws / da
    ru = par["sap"] * ip / par["hvp"] + par["sas"] * is_ / par["hvs"]
    rd = par["sep"] * ip / par["hvp"] + par["ses"] * is_ / par["hvs"]
    n2 = upper_population(ru, rd, par["n_act"], par["tau"], par["cup"])
    n1 = par["n_act"] - n2 + par["n_q"]
    gp = np.sum((par["sap"] * n1 - par["sep"] * n2) * wp)
    gs = np.sum((par["ses"] * n2 - par["sas"] * n1) * ws)
    return -(gp + par["alpha"]) * pp, (gs - par["alpha"]) * ps, float(np.sum(n1 * ws)), float(np.sum(n2 * ws))


def propagate(pump_power, signal_power, length, weight_pump, weight_signal, cell_area, n_er, tau, c_up_eff, f_q,
              sigma_ap, sigma_ep, sigma_as, sigma_es, pump_wavelength, signal_wavelength, alpha=0.0, steps=200) -> Result:
    """Co-propagating pump and signal through the doped region, RK4 in z.

    The doped region is a set of cells with areas cell_area (m^2). weight_pump and
    weight_signal are |E|^2 dA / ∫|E|^2 dA of each mode in each doped cell (their sums
    are the overlaps Γ). The local intensity is I = P · weight / cell_area.
    Returns P_p(z), P_s(z), the signal-weighted inversion and the integrals
    I1 = ∫ Σ N1 w_s dz and I2 = ∫ Σ N2 w_s dz (in m^-2) that give the probe gain.
    """
    require_nonnegative(pump_power=pump_power, signal_power=signal_power, alpha=alpha, c_up_eff=c_up_eff, sigma_ep=sigma_ep)
    require_positive(length=length, n_er=n_er, tau=tau, sigma_ap=sigma_ap, sigma_as=sigma_as, sigma_es=sigma_es,
                     pump_wavelength=pump_wavelength, signal_wavelength=signal_wavelength)
    require_range("f_q", f_q, 0.0, 0.95)
    wp, ws, da = (np.asarray(v, dtype=float) for v in (weight_pump, weight_signal, cell_area))
    require_positive(cell_area=da)
    steps = int(steps)
    par = {"sap": sigma_ap, "sep": sigma_ep, "sas": sigma_as, "ses": sigma_es, "hvp": H * C0 / pump_wavelength,
           "hvs": H * C0 / signal_wavelength, "n_act": n_er * (1 - f_q), "n_q": n_er * f_q, "tau": tau, "cup": c_up_eff, "alpha": alpha}
    cells = (wp, ws, da)
    gam_s = float(np.sum(ws))
    dz = length / steps
    z, P, S, inv = [0.0], [float(pump_power)], [float(signal_power)], []
    i1 = i2 = 0.0
    k = _derivatives(P[0], S[0], cells, par)
    inv.append(k[3] / (n_er * gam_s) if gam_s > 0 else 0.0)
    for s in range(steps):
        p, q = P[s], S[s]
        k2 = _derivatives(p + 0.5 * dz * k[0], q + 0.5 * dz * k[1], cells, par)
        k3 = _derivatives(p + 0.5 * dz * k2[0], q + 0.5 * dz * k2[1], cells, par)
        k4 = _derivatives(p + dz * k3[0], q + dz * k3[1], cells, par)
        pn = max(0.0, p + dz / 6 * (k[0] + 2 * k2[0] + 2 * k3[0] + k4[0]))
        qn = max(0.0, q + dz / 6 * (k[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        kn = _derivatives(pn, qn, cells, par)
        i1 += 0.5 * dz * (k[2] + kn[2])
        i2 += 0.5 * dz * (k[3] + kn[3])
        z.append((s + 1) * dz); P.append(pn); S.append(qn)
        inv.append(kn[3] / (n_er * gam_s) if gam_s > 0 else 0.0)
        k = kn
    z, P, S, inv = (np.array(v) for v in (z, P, S, inv))
    with np.errstate(divide="ignore"):
        gain_ln = float(np.log(S[-1] / S[0])) if S[0] > 0 else np.nan
    return Result(
        values={"z": z, "pump_power": P, "signal_power": S, "inversion": inv, "I1": i1, "I2": i2, "gain_ln": gain_ln,
                "gamma_pump": float(np.sum(wp)), "gamma_signal": gam_s},
        units={"z": "m", "pump_power": "W", "signal_power": "W", "inversion": "", "I1": "m^-2", "I2": "m^-2", "gain_ln": "",
               "gamma_pump": "", "gamma_signal": ""},
        assumptions=["Steady state, co-propagating pump and signal", "No ASE, no excited-state absorption",
                     "Er only in the listed cells; intensity ∝ |E|^2 of the dominant field", "Same background loss at pump and signal"],
    )


def probe_gain_ln(wavelength, I1, I2, length, alpha=0.0, gamma_ratio=1.0, sigma_a_peak=5.7e-25, temperature=295.0):
    """Small-signal probe gain ln(P_out/P_in) at wavelength from the integrals of propagate (helper)."""
    return gamma_ratio * (sigma_emission(wavelength, sigma_a_peak, temperature) * I2 - sigma_absorption(wavelength, sigma_a_peak) * I1) - alpha * length


def uniform_amplifier(pump_wavelength, signal_wavelength, pump_power, signal_power, length, n_er, gamma_pump, gamma_signal,
                      doped_area, tau=7.5e-3, c_up=4e-24, upconversion="proportional", quenching="proportional", k_q=0.05,
                      f_q=0.10, alpha=0.0, sigma_a_peak=5.7e-25, sigma_a_980=1.7e-25, temperature=295.0, steps=200) -> Result:
    """Amplifier with uniform pump and signal intensity over the doped area (overlaps Γ_p, Γ_s)."""
    require_range("gamma_pump", gamma_pump, 0.0, 1.0)
    require_range("gamma_signal", gamma_signal, 0.0, 1.0)
    require_positive(doped_area=doped_area, signal_power=signal_power)
    ce = concentration_effects(n_er, c_up, upconversion, quenching, k_q, f_q)
    sap, sep = pump_sigmas(pump_wavelength, sigma_a_peak, sigma_a_980, temperature)
    sas = float(sigma_absorption(signal_wavelength, sigma_a_peak))
    ses = float(sigma_emission(signal_wavelength, sigma_a_peak, temperature))
    r = propagate(pump_power, signal_power, length, [gamma_pump], [gamma_signal], [doped_area], n_er, tau, ce["c_up_eff"], ce["f_q"],
                  sap, sep, sas, ses, pump_wavelength, signal_wavelength, alpha, steps)
    return Result(
        values={"gain_ln": r["gain_ln"], "gain": float(np.exp(r["gain_ln"])), "signal_out": float(r["signal_power"][-1]),
                "pump_out": float(r["pump_power"][-1]), "inversion_in": float(r["inversion"][0]), "f_q": ce["f_q"],
                "c_up_eff": ce["c_up_eff"]},
        units={"gain_ln": "", "gain": "", "signal_out": "W", "pump_out": "W", "inversion_in": "", "f_q": "", "c_up_eff": "m^3/s"},
        assumptions=r.assumptions + ce.assumptions + ["Uniform intensity P Γ / A over the doped area"],
    )
