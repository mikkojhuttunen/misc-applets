"""gas_spectra — Voigt absorption spectra of trace gases near 1.5-1.7 um, and their passage through a
multipass cell with a distribution of path lengths.

DATA STATUS (read this):
  * ILLUSTRATIVE_LINES below are hand-entered approximate line parameters (positions, strengths and
    air-broadening of a few representative lines of CH4, NH3, CO2, H2O near 1.5-1.7 um). They are good
    enough to show WHAT a spectrum looks like and to scale orders of magnitude (about +-30 % on strengths,
    positions of the strongest lines to ~0.1 cm^-1) but they are NOT HITRAN and must not be quoted.
  * hitran.org was not reachable from the development sandbox. Run `python fetch_hitran.py` on a machine with
    classic HAPI (hitran.org/hapi) to write `hitran_lines.json`; load_lines() then uses it automatically.

Units: wavenumber cm^-1, line strength S cm^-1/(molecule cm^-2) at 296 K, gamma_air cm^-1/atm (HWHM).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import wofz

KB = 1.380649e-23
C_CM = 29979245800.0
C2 = 1.4387769            # cm K  (hc/k)
T_REF, P_REF = 296.0, 101325.0

# name: (HITRAN molecule id, molar mass amu, exponent of partition-function scaling Q(T)/Q(296) ~ (T/296)^q)
SPECIES = {"CH4": (6, 16.04, 1.5), "NH3": (11, 17.03, 1.5), "CO2": (2, 44.01, 1.0), "H2O": (1, 18.02, 1.5),
           "CO": (5, 28.01, 1.0), "C2H2": (26, 26.04, 1.0), "NO": (8, 30.01, 1.0), "N2O": (4, 44.01, 1.0)}

# (nu cm^-1, S cm/molecule, gamma_air cm^-1/atm, n_air, E'' cm^-1)  -- ILLUSTRATIVE, see module docstring
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
            (6338.10, 2.0e-23, 0.080, 0.7, 500.),
            # mid-IR (nu2 band wing near 1900 cm^-1): order-of-magnitude placeholders
            (1900.35, 2.0e-22, 0.090, 0.7, 1000.), (1899.30, 3.0e-22, 0.090, 0.7, 900.)],
    # mid-IR fundamentals (4.5-5.3 um). Strengths are ORDER-OF-MAGNITUDE placeholders (factor ~2); positions of the
    # CO and NO lines are close to the known ones, the rest are indicative. Replace with HITRAN.
    "NO":  [(1900.08, 1.6e-19, 0.058, 0.7, 150.), (1900.52, 1.5e-19, 0.058, 0.7, 150.), (1893.50, 1.2e-19, 0.058, 0.7, 120.)],
    "CO":  [(2147.08, 2.0e-19, 0.065, 0.7, 3.8), (2150.86, 3.5e-19, 0.065, 0.7, 11.5), (2154.60, 4.5e-19, 0.065, 0.7, 23.)],
    "N2O": [(2223.76, 6.0e-19, 0.075, 0.7, 50.), (2227.60, 6.5e-19, 0.075, 0.7, 70.)],
}

# a breath-like gas mixture (mole fractions). NH3 and CH4 values are typical orders of magnitude, not clinical references.
BREATH_EXAMPLE = {"CO2": 0.04, "H2O": 0.05, "CH4": 2e-6, "NH3": 0.5e-6}
# mid-IR breath-like mixture: FeNO-scale NO (clinical cut-offs are 25-50 ppb), exhaled CO of order 1-5 ppm, ambient N2O ~0.3 ppm
MIDIR_BREATH = {"H2O": 0.05, "NO": 25e-9, "CO": 2e-6, "N2O": 0.3e-6}


def load_lines(path: str | Path | None = None):
    """Line list dict species -> (nu, S, gamma_air, n_air, Elow). Merges every data/hitran_lines*.json in the repository
    (near-IR and mid-IR windows); falls back to the illustrative list."""
    files = [Path(path)] if path else sorted((Path(__file__).resolve().parents[1] / "data").glob("hitran_lines*.json"))
    files = [f for f in files if f.is_file()]
    if files:
        out = {}
        for f in files:
            for k, v in json.loads(f.read_text()).items():
                cols = tuple(np.array(v[c]) for c in ("nu", "sw", "gamma_air", "n_air", "elower"))
                out[k] = cols if k not in out else tuple(np.concatenate([a, b]) for a, b in zip(out[k], cols))
        return out, "HITRAN (" + ", ".join(f.name for f in files) + ")"
    out = {k: tuple(np.array(c) for c in zip(*v)) for k, v in ILLUSTRATIVE_LINES.items()}
    return out, "ILLUSTRATIVE line parameters (not HITRAN)"


def number_density_cm3(p_pa=P_REF, T=T_REF):
    return p_pa / (KB * T) * 1e-6


def voigt(dnu, gamma_L, gamma_D_hwhm):
    """Area-normalised Voigt profile [cm]; gamma_L, gamma_D are HWHM [cm^-1]."""
    sig = gamma_D_hwhm / np.sqrt(2 * np.log(2))
    z = (dnu + 1j * gamma_L) / (sig * np.sqrt(2))
    return np.real(wofz(z)) / (sig * np.sqrt(2 * np.pi))


_HITRAN_ID = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "NO": 8, "NH3": 11, "C2H2": 26}
_QCACHE = {}


def q_ratio(species, T):
    """Q(296 K)/Q(T) from the HITRAN partition sums (TIPS-2021, via classic HAPI) when HAPI is installed, else the
    power-law approximation (T/296)^-q stored in SPECIES."""
    if T == T_REF:
        return 1.0
    key = (species, round(T, 3))
    if key not in _QCACHE:
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                from hapi import partitionSum
                q = partitionSum(_HITRAN_ID[species], 1, [T_REF, T])
            q = q[1] if isinstance(q, tuple) else q
            _QCACHE[key] = float(q[0] / q[1])
        except Exception:
            _QCACHE[key] = (T_REF / T) ** SPECIES[species][2]
    return _QCACHE[key]


def alpha_species(nu, lines, mass_amu, qexp, x, p_pa=P_REF, T=T_REF, qratio=None):
    """Absorption coefficient [1/cm] of one species with mole fraction x."""
    nu0, S0, gam, nair, el = lines
    N = number_density_cm3(p_pa, T) * x
    S = S0 * ((T_REF / T) ** qexp if qratio is None else qratio) * np.exp(-C2 * el * (1 / T - 1 / T_REF)) * (1 - np.exp(-C2 * nu0 / T)) / (1 - np.exp(-C2 * nu0 / T_REF))
    gL = gam * (p_pa / P_REF) * (T_REF / T) ** nair
    out = np.zeros_like(nu, dtype=float)
    for k in range(nu0.size):
        gD = 3.5812e-7 * nu0[k] * np.sqrt(T / mass_amu)
        out += S[k] * N * voigt(nu - nu0[k], gL[k], gD)
    return out


def alpha_mixture(nu, mix=None, lines=None, p_pa=P_REF, T=T_REF):
    """Total gas absorption coefficient [1/cm] of a mixture {species: mole fraction}; also per-species dict."""
    mix = BREATH_EXAMPLE if mix is None else mix
    lines = load_lines()[0] if lines is None else lines
    parts = {sp: alpha_species(nu, lines[sp], SPECIES[sp][1], SPECIES[sp][2], x, p_pa, T, q_ratio(sp, T)) for sp, x in mix.items() if sp in lines}
    return sum(parts.values()), parts


def peak_alpha_per_ppm(species, p_pa=P_REF, T=T_REF, lines=None):
    """Peak absorption coefficient [1/cm] per 1 ppm of the strongest listed line of `species`."""
    lines = load_lines()[0] if lines is None else lines
    ln = lines[species]
    k = int(np.argmax(ln[1]))
    one = tuple(np.array([c[k]]) for c in ln)
    nu0 = one[0][0]
    a = alpha_species(np.array([nu0]), one, SPECIES[species][1], SPECIES[species][2], 1e-6, p_pa, T)[0]
    return a, nu0


def through_cell(alpha_cm, L_m, W, gamma_eff):
    """Fraction of power transmitted when light takes paths L_i (metres, weights W_i) and only the fraction
    gamma_eff of the modal energy samples the gas:  T(nu) = sum_i W_i exp(-gamma_eff alpha(nu) L_i) / sum W_i.
    alpha_cm: array over nu [1/cm]. (Not exp(-alpha <L>): the path-length distribution matters for strong lines.)"""
    a = np.asarray(alpha_cm, float)[:, None] * 100.0 * gamma_eff             # 1/m
    w = W / W.sum()
    return np.exp(-a * L_m[None, :]) @ w


def shot_noise_A(P_in=1e-3, T_det=0.05, bandwidth=1.0, responsivity=1.0):
    """Shot-noise-limited fractional intensity noise (= absorbance noise) at the output, for input power P_in [W]."""
    P = P_in * T_det
    return float(np.sqrt(2 * 1.602e-19 * bandwidth / (responsivity * P)))
