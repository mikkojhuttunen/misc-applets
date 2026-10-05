"""gas_spectra — Voigt absorption spectra of trace gases near 1.5-1.7 um and their passage through a multipass cell.
Thin re-export of math-engines/engines/trace_gas (HITRAN units: cm^-1, 1/cm). Line data are ILLUSTRATIVE unless a
HITRAN export is found: hitran_lines.json next to this file, $TRACE_GAS_LINES, or engines/trace_gas/hitran_lines.json
(write one with python math-engines/tools/fetch_hitran.py)."""
from pathlib import Path

import _engines  # noqa: F401
from engines.trace_gas import engine as _tg
from engines.trace_gas.engine import (BREATH_EXAMPLE, C2, C_CM, ILLUSTRATIVE_LINES, KB, P_REF, SPECIES, T_REF,  # noqa: F401
                                      alpha_species, number_density_cm3, shot_noise_A, through_cell, voigt)

_LOCAL = Path(__file__).with_name("hitran_lines.json")


def load_lines(path=None):
    return _tg.load_lines(path or (_LOCAL if _LOCAL.is_file() else None))


def alpha_mixture(nu, mix=None, lines=None, p_pa=P_REF, T=T_REF):
    return _tg.alpha_mixture(nu, mix, load_lines()[0] if lines is None else lines, p_pa, T)


def peak_alpha_per_ppm(species, p_pa=P_REF, T=T_REF, lines=None):
    return _tg.peak_alpha_per_ppm(species, p_pa, T, load_lines()[0] if lines is None else lines)
