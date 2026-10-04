"""χ(3) parametric amplification by degenerate-pump four-wave mixing 2ω_p = ω_s + ω_i (fibre OPA).

Fields are power amplitudes A_j (|A_j|² = P_j, W). With Δβ = β_s + β_i - 2β_p and
γ_j = γ ω_j/ω_p (γ = n2 ω_p / (c A_eff) at the pump):

    dA_p/dz = iγ_p [(|A_p|² + 2|A_s|² + 2|A_i|²) A_p + 2 A_s A_i A_p* e^{+iΔβz}]
    dA_s/dz = iγ_s [(|A_s|² + 2|A_p|² + 2|A_i|²) A_s + A_p² A_i* e^{-iΔβz}]
    dA_i/dz = iγ_i [(|A_i|² + 2|A_p|² + 2|A_s|²) A_i + A_p² A_s* e^{-iΔβz}]

The ω_j/ω_p scaling makes the mixing terms conserve photon number exactly. Small-signal
gain: g² = (rP)² - (κ/2)², r = sqrt(γ_s γ_i), κ = Δβ + 2γ_p P. The integrator works in
amplitudes normalised to sqrt(P_p0), as the χ3 mode of parametric-amplifier.html. SI units.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import Result, half_max_width, require_nonnegative, require_positive
from ..idler_loss.engine import amplifier_with_loss, propagate_linear

C0 = 299792458.0
N2_SILICA = 2.6e-20   # m²/W


def idler_wavelength(wavelength_pump, wavelength_signal):
    """λ_i from 1/λ_i = 2/λ_p - 1/λ_s (helper, vectorised); requires λ_s > λ_p/2."""
    lp = np.asarray(wavelength_pump, dtype=float)
    ls = np.asarray(wavelength_signal, dtype=float)
    if np.any(ls <= lp / 2):
        raise ValueError(f"wavelength_signal must exceed wavelength_pump/2, got {wavelength_signal!r}")
    return 1.0 / (2.0 / lp - 1.0 / ls)


def effective_area(w_pump, w_signal, w_idler):
    """A_eff = 1/θ, θ = ∫ e_p² e_s e_i dA for unit-power Gaussian modes (m², helper)."""
    require_positive(w_pump=w_pump, w_signal=w_signal, w_idler=w_idler)
    wp, ws, wi = (np.asarray(x, dtype=float) for x in (w_pump, w_signal, w_idler))
    theta = (np.pi / (2 / wp**2 + 1 / ws**2 + 1 / wi**2)) * (2 / np.pi) ** 2 / (wp**2 * ws * wi)
    return 1 / theta


def _rates(gamma_nl, pump_power, wavelength_pump, wavelength_signal):
    """(γ_p P, r P = sqrt(γ_s γ_i) P, idler wavelength) for frequency-scaled γ_j."""
    li = idler_wavelength(wavelength_pump, wavelength_signal)
    gp = np.asarray(gamma_nl, dtype=float) * pump_power
    return gp, gp * np.sqrt(wavelength_pump**2 / (np.asarray(wavelength_signal, dtype=float) * li)), li


def _gain_closed_form(r, kappa, length):
    """G_s = 1 + r² |sinh(gL)/g|², g² = r² - (κ/2)² (vectorised; sin where g² < 0)."""
    r = np.asarray(r, dtype=float)
    g2 = r**2 - (np.asarray(kappa, dtype=float) / 2) ** 2
    x = np.sqrt(np.abs(g2)) * length
    small = x < 1e-8
    xs = np.where(small, 1.0, x)
    shape = np.where(small, 1.0, np.where(g2 >= 0, np.sinh(xs) / xs, np.sin(xs) / xs))
    return 1 + (r * length * shape) ** 2, g2


def coupling(wavelength_pump, wavelength_signal, n2, pump_power, w_pump, w_signal, w_idler) -> Result:
    """Idler wavelength, effective area, nonlinear coefficient γ and the parametric rates."""
    require_positive(wavelength_pump=wavelength_pump, wavelength_signal=wavelength_signal, n2=n2, pump_power=pump_power)
    aeff = effective_area(w_pump, w_signal, w_idler)
    gamma = n2 * 2 * np.pi / (wavelength_pump * aeff)
    gp, r, li = _rates(gamma, pump_power, wavelength_pump, wavelength_signal)
    return Result(
        values={"wavelength_idler": li, "area_eff": aeff, "gamma_nl": gamma, "nonlinear_rate": gp,
                "nonlinear_length": 1 / gp, "gain_max": r},
        units={"wavelength_idler": "m", "area_eff": "m^2", "gamma_nl": "1/(W m)", "nonlinear_rate": "1/m",
               "nonlinear_length": "m", "gain_max": "1/m"},
        assumptions=[
            "Degenerate pump, co-polarised CW waves, slowly varying envelopes",
            "Gaussian mode profiles; A_eff = 1/∫e_p² e_s e_i dA (πw² for equal radii)",
            "γ_j = γ ω_j/ω_p; gain_max = sqrt(γ_s γ_i) P is the gain coefficient at κ = 0",
        ],
    )


def small_signal_gain(gamma_nl, pump_power, delta_beta, length, wavelength_pump, wavelength_signal,
                      alpha_idler=0.0, dumps=0, dump_loss_db=0.0) -> Result:
    """Undepleted-pump FWM gain including pump SPM and signal/idler XPM: closed form when lossless,
    exact 2×2 solution with a dissipative idler (continuous α_i and/or N lumped dumps)."""
    require_positive(gamma_nl=gamma_nl, pump_power=pump_power, length=length)
    gp, r, li = _rates(gamma_nl, pump_power, wavelength_pump, wavelength_signal)
    kappa = np.asarray(delta_beta, dtype=float) + 2 * gp
    G0, g2 = _gain_closed_form(r, kappa, length)
    lossy = alpha_idler > 0 or (int(dumps) > 0 and dump_loss_db > 0)
    if lossy:
        s, c = propagate_linear(r, kappa, length, alpha_idler, dumps, dump_loss_db)
        G, conv = np.abs(s) ** 2, np.abs(c) ** 2
    else:
        G, conv = G0, G0 - 1
    return Result(
        values={"signal_gain": G, "signal_gain_db": 10 * np.log10(G),
                "idler_conversion": (np.asarray(wavelength_signal, dtype=float) / li) * conv,
                "lossless_gain_db": 10 * np.log10(G0),
                "kappa": kappa, "growth_rate": np.sqrt(np.maximum(g2, 0.0)), "g_squared": g2,
                "peak_gain_db": 10 * np.log10(np.cosh(r * length) ** 2)},
        units={"signal_gain": "", "signal_gain_db": "dB", "idler_conversion": "", "lossless_gain_db": "dB",
               "kappa": "1/m", "growth_rate": "1/m", "g_squared": "1/m^2", "peak_gain_db": "dB"},
        assumptions=[
            "Undepleted, lossless pump; no idler seed",
            "κ = Δβ + 2γ_p P; g² = (rP)² - (κ/2)²; G_s = 1 + (rP/g)² sinh²(gL) (lossless)",
            "peak_gain_db is cosh²(rPL), lossless at κ = 0",
        ] + (["Dissipative idler: exact matrix exponential with Γ = rP and Δk → κ, dumps at z = kL/(N+1); "
              "growth_rate and g² refer to the lossless case"] if lossy else []),
    )


def propagate_normalised(gamma_power, delta_beta, length, flux_ratio, ratio_signal, ratio_idler,
                         alpha_pump=0.0, alpha_signal=0.0, alpha_idler=0.0, max_steps=200000, record=False,
                         dumps=0, dump_loss_db=0.0, dump_loss_signal_db=0.0, dump_loss_pump_db=0.0) -> Result:
    """Fixed-step RK4 of the χ3 equations in A_j / sqrt(P_p0). gamma_power = γ_p P_p0 (1/m),
    ratio_signal = ω_s/ω_p, ratio_idler = ω_i/ω_p, flux_ratio = F_s0/F_p0; alphas are power loss
    coefficients (1/m). fp, fs, fi are photon fluxes / F_p0. Same step rule as the applet."""
    require_positive(length=length, ratio_signal=ratio_signal, ratio_idler=ratio_idler)
    require_nonnegative(gamma_power=gamma_power, flux_ratio=flux_ratio, alpha_pump=alpha_pump,
                        alpha_signal=alpha_signal, alpha_idler=alpha_idler)
    npp, db, r0 = float(gamma_power), float(delta_beta), float(flux_ratio)
    nps, npi = npp * ratio_signal, npp * ratio_idler
    ap, as_, ai = alpha_pump / 2, alpha_signal / 2, alpha_idler / 2
    rate = max(npp * math.sqrt(ratio_signal * ratio_idler) * math.sqrt(1 + r0), abs(db), 3 * npp * (1 + r0),
               ap, as_, ai, 1e-6)
    N = max(300, min(int(max_steps), math.ceil(length * rate / 0.04)))
    dumps = int(dumps)
    if dumps < 0:
        raise ValueError(f"dumps must be >= 0, got {dumps!r}")
    require_nonnegative(dump_loss_db=dump_loss_db, dump_loss_signal_db=dump_loss_signal_db, dump_loss_pump_db=dump_loss_pump_db)
    n_seg = dumps + 1
    n_per = math.ceil(N / n_seg)
    h = length / (n_seg * n_per)
    t_i, t_s, t_p = (10 ** (-x / 20) for x in (dump_loss_db, dump_loss_signal_db, dump_loss_pump_db))

    def f(z, p, s, i):
        e = complex(math.cos(db * z), math.sin(db * z))
        Pp, Ps, Pi = p.real**2 + p.imag**2, s.real**2 + s.imag**2, i.real**2 + i.imag**2
        p2e = p * p * e.conjugate()
        return (1j * npp * ((Pp + 2 * Ps + 2 * Pi) * p + 2 * s * i * p.conjugate() * e) - ap * p,
                1j * nps * ((Ps + 2 * Pp + 2 * Pi) * s + p2e * i.conjugate()) - as_ * s,
                1j * npi * ((Pi + 2 * Pp + 2 * Ps) * i + p2e * s.conjugate()) - ai * i)

    cs, ci = 1 / ratio_signal, 1 / ratio_idler   # power fraction → photon-flux fraction
    p, s, i = 1 + 0j, complex(math.sqrt(r0 * ratio_signal)), 0j
    every = max(1, (n_seg * n_per) // 500)
    rec = {"z": [0.0], "fp": [1.0], "fs": [r0], "fi": [0.0]} if record else None
    for seg in range(n_seg):
        for n in range(n_per):
            z = (seg * n_per + n) * h
            k1 = f(z, p, s, i)
            k2 = f(z + h / 2, p + h / 2 * k1[0], s + h / 2 * k1[1], i + h / 2 * k1[2])
            k3 = f(z + h / 2, p + h / 2 * k2[0], s + h / 2 * k2[1], i + h / 2 * k2[2])
            k4 = f(z + h, p + h * k3[0], s + h * k3[1], i + h * k3[2])
            p += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            s += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            i += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            if record and ((seg * n_per + n + 1) % every == 0 or n == n_per - 1):
                z = (seg * n_per + n + 1) * h
                rec["z"].append(z)
                rec["fp"].append(abs(p) ** 2)
                rec["fs"].append(abs(s) ** 2 * cs)
                rec["fi"].append(abs(i) ** 2 * ci)
        if seg < n_seg - 1:      # lumped dump at z = (seg + 1) L / (N + 1)
            p, s, i = p * t_p, s * t_s, i * t_i
            if record:
                z = (seg + 1) * n_per * h
                rec["z"].append(z)
                rec["fp"].append(abs(p) ** 2)
                rec["fs"].append(abs(s) ** 2 * cs)
                rec["fi"].append(abs(i) ** 2 * ci)
    fp, fs, fi = abs(p) ** 2, abs(s) ** 2 * cs, abs(i) ** 2 * ci
    vals = {"fp": fp, "fs": fs, "fi": fi, "steps": n_seg * n_per, "capped": N == int(max_steps),
            "manley_rowe_residual": abs(fp + 2 * fs - (1 + 2 * r0)) / (1 + 2 * r0)}
    units = {"fp": "", "fs": "", "fi": "", "steps": "", "capped": "", "manley_rowe_residual": ""}
    if record:
        for k, v in rec.items():
            vals[f"{k}_z"] = np.array(v)
            units[f"{k}_z"] = "m" if k == "z" else ""
    return Result(values=vals, units=units, assumptions=[
        "Photon fluxes normalised to the input pump flux", "Fixed-step RK4, step ≤ 0.04 / rate; lumped dumps at z = kL/(N+1) split the length into equal segments",
        "manley_rowe_residual |f_p + 2f_s - (1 + 2r0)|/(1 + 2r0) is zero unless pump or signal is lossy"])


def coupled_wave(wavelength_pump, wavelength_signal, gamma_nl, pump_power, signal_power, length, delta_beta=0.0,
                 alpha_pump=0.0, alpha_signal=0.0, alpha_idler=0.0, dumps=0, dump_loss_db=0.0,
                 dump_loss_signal_db=0.0, dump_loss_pump_db=0.0, max_steps=200000) -> Result:
    """Full nonlinear FWM propagation with pump depletion, SPM/XPM, continuous losses α_j and N lumped
    dumps at z = kL/(N+1) (idler attenuation dump_loss_db; the signal and pump ones model a real
    filter): output powers, gain, depletion and power profiles along z (arrays z, pump_power_z, ...)."""
    require_positive(gamma_nl=gamma_nl, pump_power=pump_power, signal_power=signal_power)
    lp, ls = float(wavelength_pump), float(wavelength_signal)
    li = float(idler_wavelength(lp, ls))
    gp = gamma_nl * pump_power
    r0 = (signal_power * ls) / (pump_power * lp)
    r = propagate_normalised(gp, delta_beta, length, r0, lp / ls, lp / li, alpha_pump, alpha_signal, alpha_idler,
                             max_steps=max_steps, record=True, dumps=dumps, dump_loss_db=dump_loss_db,
                             dump_loss_signal_db=dump_loss_signal_db, dump_loss_pump_db=dump_loss_pump_db)
    sp, ss, si = pump_power, pump_power * lp / ls, pump_power * lp / li
    Pp, Ps, Pi = r["fp"] * sp, r["fs"] * ss, r["fi"] * si
    return Result(
        values={"wavelength_idler": li, "nonlinear_rate": gp, "kappa": delta_beta + 2 * gp,
                "pump_out": Pp, "signal_out": Ps, "idler_out": Pi,
                "signal_gain_db": 10 * math.log10(Ps / signal_power), "pump_depletion": 1 - Pp / pump_power,
                "manley_rowe_residual": r["manley_rowe_residual"], "steps": r["steps"],
                "z": r["z_z"], "pump_power_z": r["fp_z"] * sp, "signal_power_z": r["fs_z"] * ss,
                "idler_power_z": r["fi_z"] * si},
        units={"wavelength_idler": "m", "nonlinear_rate": "1/m", "kappa": "1/m", "pump_out": "W", "signal_out": "W",
               "idler_out": "W", "signal_gain_db": "dB", "pump_depletion": "", "manley_rowe_residual": "",
               "steps": "", "z": "m", "pump_power_z": "W", "signal_power_z": "W", "idler_power_z": "W"},
        assumptions=[
            "Degenerate CW pump, co-polarised, no idler seed; constant Δβ along z",
            "γ_j = γ ω_j/ω_p (photon-number-conserving mixing terms), SPM and XPM included",
            "Losses are power attenuation coefficients α_j (dP/dz = -α P); no Raman or Brillouin",
        ] + ([f"{int(dumps)} lumped dumps at z = kL/(N+1)"] if int(dumps) > 0 else []) + r.assumptions[1:] + (["Step count hit max_steps: accuracy not guaranteed"] if r["capped"] else []),
    )


LOSS_ARGS = ("alpha_idler", "dumps", "dump_loss_db", "loss_profile", "band_center", "band_width", "band_edge",
             "loss_points", "loss_acts_on")


def _spectrum_result(lp, ls, li, dbeta, r, kappa, length, loss, extra_vals, extra_units, assumptions):
    A = amplifier_with_loss(r, kappa, length, ls, li, lp, loss["alpha_idler"], loss["dumps"], loss["dump_loss_db"],
                            loss["loss_profile"], loss["band_center"], loss["band_width"], loss["band_edge"],
                            loss["loss_points"], loss["loss_acts_on"])
    G, G0 = A["gain"], A["lossless_gain"]
    long = ls > lp
    k = int(np.argmax(np.where(long, G, -np.inf)))
    vals = {"peak_gain_db": 10 * np.log10(G[k]), "peak_signal_wavelength": ls[k],
            "lobe_width_3db": half_max_width(ls[long], G[long]),
            "lossless_peak_gain_db": 10 * np.log10(G0[long].max()), "lossless_lobe_width_3db": half_max_width(ls[long], G0[long]),
            "pump_loss_weight": A["weight_pump"],
            "signal_wavelengths": ls, "idler_wavelengths": li, "delta_beta": dbeta, "gain": G, "gain_db": 10 * np.log10(G),
            "lossless_gain_db": 10 * np.log10(G0), "loss_weight": A["weight_idler"]}
    units = {"peak_gain_db": "dB", "peak_signal_wavelength": "m", "lobe_width_3db": "m", "lossless_peak_gain_db": "dB",
             "lossless_lobe_width_3db": "m", "pump_loss_weight": "", "signal_wavelengths": "m",
             "idler_wavelengths": "m", "delta_beta": "1/m", "gain": "", "gain_db": "dB", "lossless_gain_db": "dB",
             "loss_weight": ""}
    vals.update(extra_vals)
    units.update(extra_units)
    return Result(values=vals, units=units, assumptions=assumptions + [
        "Peak and lobe_width_3db refer to the long-wavelength gain lobe (λ_s > λ_p); "
        "lobe_width_3db is the full width where G ≥ G_peak/2, None if it reaches the grid edge",
        "Idler loss (if any): exact undepleted-pump solution; pump attenuation by the loss curve is not modelled"])


def _signal_grid(wavelength_pump, span, points):
    points = int(points)
    if points < 3:
        raise ValueError(f"points must be >= 3, got {points!r}")
    ls = np.linspace(wavelength_pump - span / 2, wavelength_pump + span / 2, points)
    if ls[0] <= wavelength_pump / 2:
        raise ValueError("span reaches half the pump wavelength; reduce span")
    return ls


def gain_spectrum(wavelength_pump, beta2, beta4, gamma_nl, pump_power, length, span, points=401,
                  alpha_idler=0.0, dumps=0, dump_loss_db=0.0, loss_profile="flat",
                  band_center=None, band_width=None, band_edge=None, loss_points=None, loss_acts_on="idler") -> Result:
    """Small-signal gain vs signal wavelength from the even Taylor expansion of Δβ about the pump:
    Δβ = β2 Ω² + β4 Ω⁴/12, Ω = ω_s - ω_p (β3 cancels for a degenerate pump). Optional dissipative
    idler (see idler_loss): peak α_i and/or N dumps weighted by the loss profile at each λ_i."""
    require_positive(gamma_nl=gamma_nl, pump_power=pump_power, length=length, span=span)
    lp = float(wavelength_pump)
    ls = _signal_grid(lp, span, points)
    li = idler_wavelength(lp, ls)
    O = 2 * np.pi * C0 * (1 / ls - 1 / lp)
    db = beta2 * O**2 + beta4 * O**4 / 12
    gp, r, _ = _rates(gamma_nl, pump_power, lp, ls)
    loss = {k: v for k, v in locals().items() if k in LOSS_ARGS}
    disc = beta2**2 - 4 * (beta4 / 12) * (2 * gp)      # (β4/12) x² + β2 x + 2γP = 0, x = Ω²
    roots = [] if disc < 0 or (beta2 == 0 and beta4 == 0) else (
        [(-beta2 + sg * math.sqrt(disc)) / (2 * beta4 / 12) for sg in (1, -1)] if beta4 != 0 else [-2 * gp / beta2])
    roots = [x for x in roots if x > 0]
    om = math.sqrt(min(roots)) if roots else float("nan")
    return _spectrum_result(lp, ls, li, db, r, db + 2 * gp, length, loss, {"omega_phase_matched": om}, {"omega_phase_matched": "rad/s"}, [
        "Undepleted pump, CW, constant γ (frequency-scaled per wave); Δβ from β2 and β4 only",
        "omega_phase_matched: smallest Ω > 0 with κ = Δβ + 2γP = 0 (None if none exists)",
    ])


def _lp01_beta(wavelength, core_radius, delta_n):
    from ..step_index_fiber.engine import lp01     # needs SciPy; imported on first use
    r = lp01(wavelength, core_radius, delta_n)
    return 2 * np.pi / wavelength * float(r["neff"]), float(r["mode_radius"])


def fiber_parameters(wavelength_pump, wavelength_signal, core_radius, delta_n, n2=N2_SILICA) -> Result:
    """Exact LP01 phase mismatch Δβ = β_s + β_i - 2β_p of a step-index silica fibre, pump β2/β3/β4
    (polynomial fit of β(ω) over ±50 Trad/s), effective area and γ. Requires SciPy."""
    require_positive(wavelength_pump=wavelength_pump, wavelength_signal=wavelength_signal, core_radius=core_radius,
                     delta_n=delta_n, n2=n2)
    lp, ls = float(wavelength_pump), float(wavelength_signal)
    li = float(idler_wavelength(lp, ls))
    bp, wp = _lp01_beta(lp, core_radius, delta_n)
    bs, ws = _lp01_beta(ls, core_radius, delta_n)
    bi, wi = _lp01_beta(li, core_radius, delta_n)
    w0, H = 2 * np.pi * C0 / lp, 5e13
    x = np.linspace(-1, 1, 17)
    beta = np.array([_lp01_beta(2 * np.pi * C0 / (w0 + H * t), core_radius, delta_n)[0] for t in x])
    c = np.polynomial.polynomial.polyfit(x, beta - bp, 8)
    b2, b3, b4 = 2 * c[2] / H**2, 6 * c[3] / H**3, 24 * c[4] / H**4
    aeff = float(effective_area(wp, ws, wi))
    gamma = n2 * 2 * np.pi / (lp * aeff)
    return Result(
        values={"wavelength_idler": li, "delta_beta": bs + bi - 2 * bp, "beta2": b2, "beta3": b3, "beta4": b4,
                "mode_radius_pump": wp, "area_eff": aeff, "gamma_nl": gamma},
        units={"wavelength_idler": "m", "delta_beta": "1/m", "beta2": "s^2/m", "beta3": "s^3/m", "beta4": "s^4/m",
               "mode_radius_pump": "m", "area_eff": "m^2", "gamma_nl": "1/(W m)"},
        assumptions=[
            "Weakly guiding step-index fibre, fused-silica cladding, core index = cladding + Δn at every λ",
            "Marcuse mode-field radii; A_eff from the Gaussian four-wave overlap",
            "β2, β3, β4 from an 8th-degree fit of β(ω) over ±50 Trad/s around the pump",
        ],
    )


def fiber_gain_spectrum(wavelength_pump, core_radius, delta_n, pump_power, length, span, n2=N2_SILICA,
                        points=201, alpha_idler=0.0, dumps=0, dump_loss_db=0.0, loss_profile="flat",
                        band_center=None, band_width=None, band_edge=None, loss_points=None, loss_acts_on="idler") -> Result:
    """Small-signal gain vs signal wavelength with the exact LP01 Δβ at every point and γ from the
    pump-signal-idler overlap at each point; optional dissipative idler as in gain_spectrum. Requires SciPy."""
    require_positive(core_radius=core_radius, delta_n=delta_n, pump_power=pump_power, length=length, span=span, n2=n2)
    lp = float(wavelength_pump)
    ls = _signal_grid(lp, span, points)
    li = idler_wavelength(lp, ls)
    bp, wp = _lp01_beta(lp, core_radius, delta_n)
    S = [_lp01_beta(l, core_radius, delta_n) for l in ls]
    I = [_lp01_beta(l, core_radius, delta_n) for l in li]
    db = np.array([s[0] + i[0] - 2 * bp for s, i in zip(S, I)])
    aeff = effective_area(wp, np.array([s[1] for s in S]), np.array([i[1] for i in I]))
    gamma = n2 * 2 * np.pi / (lp * aeff)
    gp, r, _ = _rates(gamma, pump_power, lp, ls)
    loss = {k: v for k, v in locals().items() if k in LOSS_ARGS}
    return _spectrum_result(lp, ls, li, db, r, db + 2 * gp, length, loss, {"gamma_pump": float(n2 * 2 * np.pi / (lp * effective_area(wp, wp, wp)))},
                            {"gamma_pump": "1/(W m)"}, [
        "Undepleted pump, CW, LP01 of a weakly guiding silica step-index fibre (exact Δβ, no Taylor expansion)",
        "gamma_pump is γ for equal pump-mode radii (reference value)",
    ])
