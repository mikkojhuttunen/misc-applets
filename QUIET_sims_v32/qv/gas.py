"""Gas absorption from HITRAN line lists with HAPI (offline, on the line files of ../HITRAN/hitran_data in this repo:
hitran.org itself is not reachable from the simulation host, and the files are the ones HITRAN/get_spectra.py fetched).

Absorption coefficient alpha(nu) [1/cm] of 1 ppm of an analyte in a breath background: total pressure 1 atm,
35 C (308.15 K), Voigt profiles with air broadening (HAPI Diluent air; the self-broadening of the 4-6 % H2O and CO2 of
breath is not separated), the analyte's main isotopologue as in the line files. Background: H2O at saturation at
35 C (5.56 kPa: 5.5 % at 1 atm) and CO2 4.5 %. Windows and targets: CH4 at 3.31 um (2950-3100 /cm) and 1.65 um
(6040-6180), C2H6 at 3.34 um (2880-3000), NH3 at 1.53 um (6480-6620)."""
from __future__ import annotations

import contextlib
import io
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

HITRAN_DIR = Path(__file__).resolve().parent.parent.parent / "HITRAN" / "hitran_data"
T_BREATH = 308.15
P_ATM = 1.0
X_H2O = 5.56e3 / 101325.0          # saturation vapour pressure at 35 C over 1 atm
X_CO2 = 0.045
MOL = {"H2O": 1, "CO2": 2, "CH4": 6, "C2H6": 27, "NH3": 11}
TARGETS = {  # key: (molecule, table, target wavelength in um, half-width of the search window in /cm)
    "CH4_3.31": ("CH4", "CH4_2950_3100", 3.31, 30.0),
    "CH4_1.65": ("CH4", "CH4_6040_6180", 1.65, 30.0),
    "C2H6_3.34": ("C2H6", "C2H6_2880_3000", 3.34, 30.0),
    "NH3_1.53": ("NH3", "NH3_6480_6620", 1.53, 30.0),
}
BACKGROUND_TABLES = {"CH4_2950_3100": ("H2O_2950_3100", "CO2_2950_3100"), "CH4_6040_6180": ("H2O_6040_6180", "CO2_6040_6180"),
                     "C2H6_2880_3000": ("H2O_2880_3000", "CO2_2880_3000"), "NH3_6480_6620": ("H2O_6480_6620", "CO2_6480_6620")}


@lru_cache(maxsize=1)
def _hapi():
    import hapi
    with contextlib.redirect_stdout(io.StringIO()):
        hapi.db_begin(str(HITRAN_DIR))
    return hapi


def alpha(table, x, nu_min, nu_max, step=0.002, T=T_BREATH, p=P_ATM):
    """alpha(nu) [1/cm] of the gas in `table` at mole fraction x."""
    h = _hapi()
    with contextlib.redirect_stdout(io.StringIO()):
        nu, coef = h.absorptionCoefficient_Voigt(SourceTables=table, Environment={"T": T, "p": p}, Diluent={"air": 1.0},
                                                 WavenumberRange=[nu_min, nu_max], WavenumberStep=step, HITRAN_units=False)
    # HITRAN_units=False: coefficient in 1/cm for the total number density at (T, p); scale by the mole fraction
    return np.asarray(nu), np.asarray(coef) * x


def target_line(key, step=0.002, span=1.5, nu_center=None, half_width=None):
    """The strongest absorption of 1 ppm analyte within ±half_width (default: the target's window) of the target
    wavelength (or of nu_center): peak alpha, its position, the breath background (H2O, CO2) there and a line-resolved
    strip of ±span /cm."""
    mol, table, lam_um, hw = TARGETS[key]
    nu_t = 1e4 / lam_um if nu_center is None else nu_center
    hw = hw if half_width is None else half_width
    nu, a = alpha(table, 1e-6, nu_t - hw, nu_t + hw, step)
    i = int(np.argmax(a))
    nu0 = float(nu[i])
    lo, hi = nu0 - span, nu0 + span
    nus, a_s = alpha(table, 1e-6, lo, hi, step)
    bg = {}
    for btab, x, name in zip(BACKGROUND_TABLES[table], (X_H2O, X_CO2), ("H2O", "CO2")):
        nb, ab = alpha(btab, x, lo, hi, step)
        bg[name] = np.interp(nus, nb, ab)
    return {"key": key, "molecule": mol, "table": table, "nu0_cm": nu0, "lambda0_um": 1e4 / nu0, "alpha_peak_per_ppm_cm": float(a[i]),
            "bg_at_peak_per_cm": {k: float(np.interp(nu0, nus, v)) for k, v in bg.items()},
            "strip": {"nu": nus.tolist(), "alpha_analyte_1ppm": a_s.tolist(), **{f"alpha_{k}": v.tolist() for k, v in bg.items()}},
            "conditions": {"T_K": T_BREATH, "p_atm": P_ATM, "x_H2O": X_H2O, "x_CO2": X_CO2}}
