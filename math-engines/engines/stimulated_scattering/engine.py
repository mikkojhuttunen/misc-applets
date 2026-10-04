"""Stimulated Brillouin and Raman scattering in fused-silica fiber.

Conventions
-----------
* SI units; ``mode_area`` is the effective area (m^2), ``pump_linewidth`` in Hz.
* SBS: exact CW two-wave solution for a forward pump and a backward Stokes wave,
  P_p(z) - P_B(z) = C, seeded at z = L by thermal phonons
  P_seed = h nu_p (kT / h nu_B) dnu_B. The gain is reduced by a Lorentzian pump
  linewidth: g_B,eff = g_B dnu_B / (dnu_B + dnu_p).
* SRS: single-oscillator Raman response h(t) ~ exp(-t/tau2) sin(t/tau1)
  (Blow & Wood 1989) normalised to a peak of 1e-13 m/W at 1 um, scaling 1/lambda.
  The gain is odd in the frequency shift (Stokes gain, anti-Stokes loss) and is
  rolled off smoothly above 32 THz, where silica has essentially no Raman gain.
"""
from __future__ import annotations

import math

import numpy as np

from engines.common import C0, H_PLANCK, K_B, Result, require_nonnegative, require_positive

N_SILICA = 1.45
V_ACOUSTIC = 5960.0       # m/s, longitudinal acoustic velocity in silica
G_B_PEAK = 5e-11          # m/W, bulk silica Brillouin gain
TAU1, TAU2 = 12.2e-15, 32e-15


def _raman_im(O):
    a, b = 1 / TAU2, 1 / TAU1
    X = a * a + b * b - O * O
    Y = 2 * a * O
    return 2 * a * b * O / (X * X + Y * Y)


def _raman_peak():
    best, ob = 0.0, 0.0
    for k in range(1, 4000):
        O = 2 * math.pi * 40e12 * k / 4000
        v = _raman_im(O)
        if v > best:
            best, ob = v, O
    return best, ob


RAMAN_PEAK_VALUE, RAMAN_PEAK_SHIFT = _raman_peak()   # (arbitrary units, rad/s)


def raman_shape(shift):
    """Normalised, signed Raman gain shape for an angular frequency shift (rad/s) (helper)."""
    O = np.asarray(shift, float)
    nu = np.abs(O) / 2 / math.pi / 1e12
    win = np.where(nu > 32, np.exp(-((nu - 32) / 5) ** 2), 1.0)
    return _raman_im(O) / RAMAN_PEAK_VALUE * win


def raman_peak_gain(lambda_pump):
    """Peak Raman gain coefficient g_R (m/W) for a pump wavelength (helper)."""
    return 1e-13 * (1e-6 / np.asarray(lambda_pump, float))


def raman_gain(frequency_shift, lambda_pump) -> Result:
    """Raman gain coefficient for a pump-to-Stokes frequency shift (Hz, > 0 Stokes)."""
    require_positive("lambda_pump", lambda_pump)
    g = raman_peak_gain(lambda_pump) * raman_shape(2 * math.pi * np.asarray(frequency_shift, float))
    return Result({"g_R": g, "peak_shift": RAMAN_PEAK_SHIFT / 2 / math.pi},
                  {"g_R": "m/W", "peak_shift": "Hz"},
                  ["Blow-Wood single-oscillator Raman response", "co-polarised waves", "roll-off above 32 THz"])


def srs_threshold(length, mode_area, lambda_pump) -> Result:
    """Smith's forward-SRS critical power P_cr = 16 A_eff / (g_R L) and the first Stokes wavelength."""
    for nm, v in (("length", length), ("mode_area", mode_area), ("lambda_pump", lambda_pump)):
        require_positive(nm, v)
    wR = 2 * math.pi * C0 / lambda_pump - RAMAN_PEAK_SHIFT
    return Result({"threshold": 16 * mode_area / (raman_peak_gain(lambda_pump) * length), "lambda_stokes": 2 * math.pi * C0 / wR},
                  {"threshold": "W", "lambda_stokes": "m"},
                  ["Smith (1972) criterion, lossless fiber of length L", "co-polarised pump and Stokes"])


def brillouin_parameters(lambda_pump, pump_linewidth=0.0, temperature=293.0) -> Result:
    """Brillouin shift, gain linewidth, effective gain and thermal seed for a pump wavelength."""
    require_positive("lambda_pump", lambda_pump)
    require_nonnegative("pump_linewidth", pump_linewidth)
    nuB = 2 * N_SILICA * V_ACOUSTIC / lambda_pump
    dnuB = 20e6 * (1550e-9 / lambda_pump) ** 2
    g = G_B_PEAK * dnuB / (dnuB + pump_linewidth)
    nth = K_B * temperature / (H_PLANCK * nuB)
    seed = H_PLANCK * C0 / lambda_pump * nth * dnuB
    return Result({"shift": nuB, "linewidth": dnuB, "g_B": g, "seed_power": seed},
                  {"shift": "Hz", "linewidth": "Hz", "g_B": "m/W", "seed_power": "W"},
                  ["nu_B = 2 n v_A / lambda", "dnu_B = 20 MHz (1550 nm / lambda)^2", "Lorentzian pump spectrum"])


class SbsSolution:
    """Exact two-wave CW SBS profile (helper object used by the amplifier engine)."""

    def __init__(self, P0, gA, L, seed):
        def pbL(D):
            C, q = P0 - D, D / P0
            e = q * math.exp(-gA * C * L)
            return C * e / (1 - e)

        lo, hi = math.log(seed * 1e-6), math.log(P0 * (1 - 1e-9))
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if pbL(math.exp(m)) > seed:
                hi = m
            else:
                lo = m
        self.D = math.exp(0.5 * (lo + hi))
        self.C = P0 - self.D
        self.q = self.D / P0
        self.gA = gA
        self.PL = self.C / (1 - self.q * math.exp(-gA * self.C * L))

    def PB(self, z):
        e = self.q * math.exp(-self.gA * self.C * z)
        return self.C * e / (1 - e)


def sbs_two_wave(pump_power, length, mode_area, lambda_pump, pump_linewidth=0.0) -> Result:
    """Backward Stokes power at the input, transmitted pump and Smith threshold 21 A_eff/(g_B L)."""
    for nm, v in (("pump_power", pump_power), ("length", length), ("mode_area", mode_area)):
        require_positive(nm, v)
    br = brillouin_parameters(lambda_pump, pump_linewidth)
    sol = SbsSolution(pump_power, br["g_B"] / mode_area, length, br["seed_power"])
    return Result({"reflected_power": sol.D, "transmitted_pump": sol.PL, "reflectivity": sol.D / pump_power,
                   "threshold": 21 * mode_area / (br["g_B"] * length)},
                  {"reflected_power": "W", "transmitted_pump": "W", "reflectivity": "1", "threshold": "W"},
                  ["CW, undepleted by other processes", "no fiber loss", "co-polarised", "thermal seed at z = L"])
