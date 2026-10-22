#!/usr/bin/env python3
"""
hitran_lbl.py -- line-by-line Voigt spectra straight from the cached HITRAN files in ./hitran_data/.

No HAPI and no internet needed: the 160-character HITRAN records (*.data, written by HAPI's fetch in
get_spectra.py) are parsed here and summed with a Voigt profile (Faddeeva function). Reproduces HAPI's
absorptionCoefficient_Voigt (same line-shape, broadening, wing cut-off) to better than 1 % at the line peaks;
run this file to check against spectra_out/spec_*.csv.

    from hitran_lbl import alpha, Mixture
    a = alpha(nu, "NH3", x=1e-6)                      # absorption coefficient [1/cm] of 1 ppm NH3 in air
    a_tot, parts = Mixture({"NH3": 1e-6, "H2O": 0.06, "CO2": 0.045}).alpha(nu)

Physics (HITRAN conventions, reference T0 = 296 K, p0 = 1 atm):
    S(T)   = S0 Q(T0)/Q(T) exp(-c2 E''(1/T - 1/T0)) (1 - exp(-c2 nu0/T)) / (1 - exp(-c2 nu0/T0))
    gL     = (p/p0) (T0/T)^n_air [(1 - x) gamma_air + x gamma_self]          Lorentz HWHM
    gD     = nu0/c sqrt(2 ln2 k T / m)                                         Doppler HWHM
    nu0'   = nu0 + delta_air p/p0                                              pressure shift
    alpha  = x N(p, T) sum_k S_k V(nu - nu0'_k; gL_k, gD_k)                    [1/cm]
Each line is cut at 50 half-widths (HAPI's default OmegaWingHW), or with wing_cm at a fixed distance with the
profile value there subtracted (the 25 cm^-1 convention of LBLRTM / MT_CKD: no steps where a strong line is cut, and
no Lorentzian far wings, which overestimate the real sub-Lorentzian wings). S0 already carries the natural isotopic
abundance, so the result is for the natural gas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import wofz

DATA = Path(__file__).resolve().parent / "hitran_data"
T0, P0_PA = 296.0, 101325.0
KB = 1.380649e-23
C2 = 1.4387769                    # hc/k [cm K]
AMU = 1.66053906660e-27
C = 299792458.0

# HITRAN molecule id, molar mass of the main isotopologue [amu]
MOLECULES = {"H2O": (1, 18.0106), "CO2": (2, 43.9898), "CH4": (6, 16.0313), "NO": (8, 29.9980),
             "NH3": (11, 17.0265), "C2H6": (27, 30.0470)}
# Q(296 K)/Q(T) of the main isotopologue at 310 K from TIPS-2021 (HAPI partitionSum); used when HAPI is absent
Q_RATIO_310 = {"H2O": 0.933202, "CO2": 0.942189, "CH4": 0.931095, "NO": 0.946015, "NH3": 0.930635, "C2H6": 0.901863}
WING_HW = 50.0


def read_par(path):
    """HITRAN 160-character records -> dict of arrays (nu, sw, gamma_air, gamma_self, elower, n_air, delta_air)."""
    cols = {"nu": (3, 15), "sw": (15, 25), "gamma_air": (35, 40), "gamma_self": (40, 45), "elower": (45, 55),
            "n_air": (55, 59), "delta_air": (59, 67)}
    rows = [ln for ln in Path(path).read_text().splitlines() if len(ln) >= 67]
    return {k: np.array([float(r[a:b]) for r in rows]) for k, (a, b) in cols.items()}


def find_tables(species, nu_lo, nu_hi):
    """Cached files of `species` whose window overlaps [nu_lo, nu_hi]."""
    out = []
    for f in sorted(DATA.glob(f"{species}_*_*.data")):
        lo, hi = (float(v) for v in f.stem.split("_")[-2:])
        if lo < nu_hi and hi > nu_lo:
            out.append(f)
    return out


_CACHE: dict = {}


def lines(species, nu_lo, nu_hi):
    """Merged line list of `species` from every cached window overlapping [nu_lo, nu_hi]."""
    fs = find_tables(species, nu_lo, nu_hi)
    if not fs:
        raise FileNotFoundError(f"no cached HITRAN data for {species} in {nu_lo}-{nu_hi} cm^-1 (run get_spectra.py)")
    key = tuple(fs)
    if key not in _CACHE:
        parts = [read_par(f) for f in fs]
        _CACHE[key] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return _CACHE[key]


def q_ratio(species, T):
    """Q(T0)/Q(T): HAPI's TIPS-2021 when installed, else the tabulated 310 K value or a T^1.5 power law."""
    if T == T0:
        return 1.0
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            from hapi import partitionSum
            q = partitionSum(MOLECULES[species][0], 1, [T0, T])
        return float(q[0] / q[1])
    except Exception:
        return Q_RATIO_310[species] if abs(T - 310.0) < 1e-9 else (T0 / T) ** 1.5


def voigt(dnu, gL, gD):
    """Area-normalised Voigt profile [cm]; gL, gD are HWHM [cm^-1]."""
    sig = gD / np.sqrt(2 * np.log(2))
    return np.real(wofz((dnu + 1j * gL) / (sig * np.sqrt(2)))) / (sig * np.sqrt(2 * np.pi))


def alpha(nu, species, x, T=310.0, p_atm=1.0, wing_hw=WING_HW, wing_cm=None):
    """Absorption coefficient [1/cm] of `species` at mole fraction x in air, on the wavenumber grid nu [cm^-1].
    wing_cm: cut every line at this distance [cm^-1] and subtract its value there (else cut at wing_hw half-widths)."""
    nu = np.asarray(nu, float)
    L = lines(species, nu.min() - (wing_cm or 5), nu.max() + (wing_cm or 5))
    mass = MOLECULES[species][1] * AMU
    N = x * p_atm * P0_PA / (KB * T) * 1e-6                               # absorbers per cm^3
    S = L["sw"] * q_ratio(species, T) * np.exp(-C2 * L["elower"] * (1 / T - 1 / T0)) \
        * (1 - np.exp(-C2 * L["nu"] / T)) / (1 - np.exp(-C2 * L["nu"] / T0))
    gL = p_atm * (T0 / T) ** L["n_air"] * ((1 - x) * L["gamma_air"] + x * L["gamma_self"])
    gD = L["nu"] / C * np.sqrt(2 * np.log(2) * KB * T / mass)
    nu0 = L["nu"] + L["delta_air"] * p_atm
    cut = np.full_like(gL, wing_cm) if wing_cm else wing_hw * np.maximum(gL, gD)
    keep = (nu0 + cut > nu.min()) & (nu0 - cut < nu.max())
    out = np.zeros_like(nu)
    for k in np.flatnonzero(keep):
        lo, hi = np.searchsorted(nu, [nu0[k] - cut[k], nu0[k] + cut[k]])
        if hi > lo:
            ped = voigt(np.array([cut[k]]), gL[k], gD[k])[0] if wing_cm else 0.0
            out[lo:hi] += S[k] * np.maximum(voigt(nu[lo:hi] - nu0[k], gL[k], gD[k]) - ped, 0.0)
    return N * out


class Mixture:
    """A gas mixture {species: mole fraction} at temperature T [K] and pressure p [atm]."""

    def __init__(self, mix, T=310.0, p_atm=1.0, wing_cm=None):
        self.mix, self.T, self.p, self.wing_cm = dict(mix), T, p_atm, wing_cm

    def alpha(self, nu):
        parts = {sp: alpha(nu, sp, x, self.T, self.p, wing_cm=self.wing_cm) for sp, x in self.mix.items()}
        return sum(parts.values()), parts


def strongest_line(species, nu_lo, nu_hi, T=310.0, x=1e-6, step=0.002):
    """(nu, alpha) of the highest absorption peak of `species` at mole fraction x inside [nu_lo, nu_hi]."""
    nu = np.arange(nu_lo, nu_hi, step)
    a = alpha(nu, species, x, T)
    i = int(np.argmax(a))
    return float(nu[i]), float(a[i])


def _check():
    """Compare with the HAPI spectra in spectra_out/ (1 ppm analyte, 310 K, 1 atm, 0.01 cm^-1 grid)."""
    out = Path(__file__).resolve().parent / "spectra_out"
    for tag, sp in [("NH3_1.5um", "NH3"), ("CH4_1.6um", "CH4"), ("CH4_3.3um", "CH4"), ("C2H6_3.3um", "C2H6"), ("NO_5.3um", "NO")]:
        d = np.genfromtxt(out / f"spec_{tag}.csv", delimiter=",", skip_header=1)
        i = int(np.argmax(d[:, 1]))
        m = np.abs(d[:, 0] - d[i, 0]) < 1.0
        a = alpha(d[m, 0], sp, 1e-6)
        print(f"{tag:11s} peak {d[i, 0]:.2f} cm-1  HAPI {d[i, 1]:.4e}  here {alpha(d[i:i + 1, 0], sp, 1e-6)[0]:.4e}  "
              f"max rel. dev. within +-1 cm-1 (above 1 % of peak): {np.max(np.abs(a / d[m, 1] - 1)[d[m, 1] > 0.01 * d[i, 1]]):.2%}")


if __name__ == "__main__":
    _check()
