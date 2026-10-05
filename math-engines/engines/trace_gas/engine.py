"""Voigt absorption spectra of trace gases and their passage through a cell with a distribution of path lengths.

UNITS. Line lists and the spectral helpers (voigt, alpha_species, alpha_mixture, peak_alpha_per_ppm) work in
HITRAN units: wavenumber ν in cm^-1, line strength S in cm^-1/(molecule cm^-2) at 296 K, γ_air in cm^-1/atm
(HWHM), absorption coefficients in 1/cm. The Result front ends (line_peak, shot_noise) are SI.

LINE DATA. ILLUSTRATIVE_LINES are hand-entered approximate parameters of a few representative lines of CH4,
NH3, CO2 and H2O near 1.5-1.7 µm: good for the shape of a spectrum and orders of magnitude (about ±30 % on
strengths, strongest positions to ~0.1 cm^-1), NOT HITRAN, not to be quoted. Run tools/fetch_hitran.py (needs
`pip install hitran-api` and access to hitran.org) to write a hitran_lines.json; load_lines() uses it when
it is passed explicitly, named by $TRACE_GAS_LINES, or saved next to this file.

Model: Voigt profile (Faddeeva function), air broadening only (no self-broadening, no pressure shift),
partition function Q(T) ∝ T^q.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..common import Result, require_choice, require_positive, require_range

KB = 1.380649e-23
Q_E = 1.602176634e-19
C_CM = 29979245800.0          # speed of light, cm/s
C2 = 1.4387769                # hc/k, cm K
T_REF, P_REF = 296.0, 101325.0

# name: (HITRAN molecule id, molar mass amu, q with Q(T)/Q(296) ≈ (T/296)^q)
SPECIES = {"CH4": (6, 16.04, 1.5), "NH3": (11, 17.03, 1.5), "CO2": (2, 44.01, 1.0), "H2O": (1, 18.02, 1.5),
           "CO": (5, 28.01, 1.0), "C2H2": (26, 26.04, 1.0)}

# (ν cm^-1, S cm/molecule, γ_air cm^-1/atm, n_air, E'' cm^-1) -- ILLUSTRATIVE, see module docstring
ILLUSTRATIVE_LINES = {
    "NH3": [(6528.76, 1.1e-20, 0.095, 0.7, 120.), (6531.40, 5.0e-21, 0.090, 0.7, 200.),
            (6524.55, 4.0e-21, 0.095, 0.7, 150.), (6544.40, 3.5e-21, 0.090, 0.7, 250.),
            (6596.60, 6.0e-21, 0.095, 0.7, 100.)],
    "CH4": [(6046.95, 1.5e-21, 0.060, 0.7, 62.), (6057.10, 1.2e-21, 0.060, 0.7, 105.),
            (6065.58, 8.0e-22, 0.058, 0.7, 210.), (6074.90, 9.0e-22, 0.058, 0.7, 315.),
            (6082.02, 4.0e-22, 0.058, 0.7, 440.), (6005.65, 5.5e-22, 0.062, 0.7, 10.)],
    "CO2": [(6359.97, 1.7e-23, 0.072, 0.7, 150.), (6357.31, 1.1e-23, 0.072, 0.7, 260.),
            (6362.45, 1.2e-23, 0.072, 0.7, 330.), (6347.80, 1.0e-23, 0.072, 0.7, 450.)],
    "H2O": [(6540.40, 3.0e-23, 0.080, 0.7, 700.), (6573.20, 2.5e-23, 0.080, 0.7, 600.),
            (6338.10, 2.0e-23, 0.080, 0.7, 500.)],
}
ILLUSTRATIVE_SOURCE = "ILLUSTRATIVE line parameters (not HITRAN)"

# breath-like mixture (mole fractions); NH3 and CH4 are typical orders of magnitude, not clinical references
BREATH_EXAMPLE = {"CO2": 0.04, "H2O": 0.05, "CH4": 2e-6, "NH3": 0.5e-6}

_COLUMNS = ("nu", "sw", "gamma_air", "n_air", "elower")


def load_lines(path=None):
    """(lines, source): lines = {species: (ν, S, γ_air, n_air, E'') arrays}. Looks for a HITRAN json at `path`,
    then $TRACE_GAS_LINES, then hitran_lines.json next to this file; falls back to ILLUSTRATIVE_LINES."""
    for cand in (path, os.environ.get("TRACE_GAS_LINES"), Path(__file__).with_name("hitran_lines.json")):
        if cand and Path(cand).is_file():
            d = json.loads(Path(cand).read_text())
            return {k: tuple(np.array(v[c], float) for c in _COLUMNS) for k, v in d.items()}, f"HITRAN ({Path(cand).name})"
    return {k: tuple(np.array(c, float) for c in zip(*v)) for k, v in ILLUSTRATIVE_LINES.items()}, ILLUSTRATIVE_SOURCE


def number_density_cm3(p_pa=P_REF, T=T_REF):
    """Total number density [1/cm^3] of an ideal gas."""
    return p_pa / (KB * T) * 1e-6


def doppler_hwhm(nu0, T, mass_amu):
    """Doppler HWHM [cm^-1] = ν0 sqrt(2 ln2 k T / m c²)."""
    return 3.5812e-7 * np.asarray(nu0) * np.sqrt(T / mass_amu)


def voigt(dnu, gamma_L, gamma_D_hwhm):
    """Area-normalised Voigt profile [cm]; Lorentz and Doppler HWHM in cm^-1."""
    from scipy.special import wofz
    sig = gamma_D_hwhm / np.sqrt(2 * np.log(2))
    z = (dnu + 1j * gamma_L) / (sig * np.sqrt(2))
    return np.real(wofz(z)) / (sig * np.sqrt(2 * np.pi))


def line_strength(S0, nu0, elower, T, qexp):
    """Line strength at T from its 296 K value: partition function, Boltzmann factor, stimulated emission."""
    return (S0 * (T_REF / T) ** qexp * np.exp(-C2 * elower * (1 / T - 1 / T_REF))
            * (1 - np.exp(-C2 * nu0 / T)) / (1 - np.exp(-C2 * nu0 / T_REF)))


def lorentz_hwhm(gamma_air, n_air, p_pa=P_REF, T=T_REF):
    """Air-broadened Lorentz HWHM [cm^-1]."""
    return gamma_air * (p_pa / P_REF) * (T_REF / T) ** n_air


def alpha_species(nu, lines, mass_amu, qexp, x, p_pa=P_REF, T=T_REF):
    """Absorption coefficient [1/cm] on the wavenumber grid nu [cm^-1] of one species at mole fraction x."""
    nu0, S0, gam, nair, el = lines
    N = number_density_cm3(p_pa, T) * x
    S = line_strength(S0, nu0, el, T, qexp)
    gL = lorentz_hwhm(gam, nair, p_pa, T)
    gD = doppler_hwhm(nu0, T, mass_amu)
    nu = np.asarray(nu, float)
    out = np.zeros_like(nu)
    for k in range(nu0.size):
        out += S[k] * N * voigt(nu - nu0[k], gL[k], gD[k])
    return out


def alpha_mixture(nu, mix=None, lines=None, p_pa=P_REF, T=T_REF):
    """Total absorption coefficient [1/cm] of a mixture {species: mole fraction}, and the per-species dict.
    Species without lines in the list are skipped (absent from the per-species dict)."""
    mix = BREATH_EXAMPLE if mix is None else mix
    lines = load_lines()[0] if lines is None else lines
    parts = {sp: alpha_species(nu, lines[sp], SPECIES[sp][1], SPECIES[sp][2], x, p_pa, T) for sp, x in mix.items() if sp in lines}
    return sum(parts.values()) if parts else np.zeros_like(np.asarray(nu, float)), parts


def strongest_line(species, lines=None):
    """Index and parameters of the strongest line (largest S at 296 K) of a species."""
    lines = load_lines()[0] if lines is None else lines
    ln = lines[species]
    k = int(np.argmax(ln[1]))
    return k, tuple(np.array([c[k]]) for c in ln)


def peak_alpha_per_ppm(species, p_pa=P_REF, T=T_REF, lines=None):
    """(peak absorption coefficient [1/cm] per 1 ppm of the strongest listed line, its ν0 [cm^-1]).
    Neighbouring lines are not included."""
    _, one = strongest_line(species, lines)
    nu0 = float(one[0][0])
    a = alpha_species(np.array([nu0]), one, SPECIES[species][1], SPECIES[species][2], 1e-6, p_pa, T)[0]
    return float(a), nu0


def through_cell(alpha_cm, L_m, W, gamma_eff):
    """Transmitted fraction when light takes paths L_i [m] with power weights W_i and only Γ of the modal energy
    samples the gas:  T(ν) = Σ W_i exp(-Γ α(ν) L_i) / Σ W_i. Not exp(-Γ α ⟨L⟩): the path-length distribution
    matters for strong lines."""
    a = np.atleast_1d(np.asarray(alpha_cm, float))[:, None] * 100.0 * gamma_eff   # 1/m
    w = np.asarray(W, float) / np.sum(W)
    return np.exp(-a * np.asarray(L_m, float)[None, :]) @ w


def shot_noise_A(P_in=1e-3, T_det=0.05, bandwidth=1.0, responsivity=1.0):
    """Shot-noise-limited fractional intensity noise (= absorbance noise) sqrt(2 q B / (ℜ P_in T_det))."""
    return float(np.sqrt(2 * Q_E * bandwidth / (responsivity * P_in * T_det)))


# ---------------------------------------------------------------- Result front ends (SI)
def line_peak(species="NH3", mole_fraction=1e-6, pressure=101325.0, temperature=296.0, path_length=1.0) -> Result:
    """Strongest listed line of a species: centre, widths, peak absorption coefficient and the peak absorbance
    α_peak L over an (effective) path length L."""
    require_choice("species", species, tuple(ILLUSTRATIVE_LINES))
    require_positive(pressure=pressure, temperature=temperature, path_length=path_length)
    require_range("mole_fraction", mole_fraction, 0.0, 1.0)
    lines, src = load_lines()
    if species not in lines:
        raise ValueError(f"no lines for {species} in {src}")
    _, one = strongest_line(species, lines)
    nu0, S0, gam, nair, el = (float(c[0]) for c in one)
    mass, q = SPECIES[species][1], SPECIES[species][2]
    a_cm = alpha_species(np.array([nu0]), one, mass, q, mole_fraction, pressure, temperature)[0]
    gL = float(lorentz_hwhm(gam, nair, pressure, temperature))
    gD = float(doppler_hwhm(nu0, temperature, mass))
    alpha = a_cm * 100.0
    return Result(
        values={"wavenumber": nu0 * 100.0, "wavelength": 1e-2 / nu0, "S": float(line_strength(S0, nu0, el, temperature, q)) * 1e-2,
                "lorentz_hwhm": gL * C_CM, "doppler_hwhm": gD * C_CM, "alpha_peak": alpha, "absorbance": alpha * path_length},
        units={"wavenumber": "1/m", "wavelength": "m", "S": "m/molecule", "lorentz_hwhm": "Hz", "doppler_hwhm": "Hz",
               "alpha_peak": "1/m", "absorbance": ""},
        assumptions=[f"Line data: {src}", "Single line, neighbours not included", "Voigt profile, air broadening only, no pressure shift",
                     "Partition function Q ∝ T^q", "Absorbance = α_peak L (weak absorption; multiply L by Γ for an evanescent mode)"],
    )


def shot_noise(P_in=1e-3, T_det=0.05, bandwidth=1.0, responsivity=1.0) -> Result:
    """Shot-noise-limited absorbance noise at the detector."""
    require_positive(P_in=P_in, T_det=T_det, bandwidth=bandwidth, responsivity=responsivity)
    return Result(
        values={"sigma_A": shot_noise_A(P_in, T_det, bandwidth, responsivity), "P_det": P_in * T_det},
        units={"sigma_A": "", "P_det": "W"},
        assumptions=["Shot noise of the photocurrent only (no RIN, detector or etalon noise)", "Noise bandwidth B (1 Hz ↔ 0.5 s averaging)"],
    )
