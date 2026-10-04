"""Second-harmonic generation with quasi-phase-matching (QPM).

Pump at vacuum wavelength λ, SH at λ/2. Δk = β_SH - 2 β_pump = 4π (n_SH - n_pump) / λ,
with effective (or bulk) indices of the interacting modes. First-order QPM period
Λ = 2π / |Δk|. SI units.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_positive

SINC2_HALF = 1.391557  # x with sinc^2(x) = 1/2, sinc(x) = sin(x)/x


def phase_matching(wavelength, n_pump, n_sh, ng_pump=None, ng_sh=None, length=None) -> Result:
    """Phase mismatch, QPM period and coherence length; with group indices and a
    length also the pump-wavelength FWHM of the sinc² QPM response."""
    require_positive(wavelength=wavelength, n_pump=n_pump, n_sh=n_sh)
    dk = 4 * np.pi * (n_sh - n_pump) / wavelength
    vals = {"delta_k": dk, "period": 2 * np.pi / abs(dk), "coherence_length": np.pi / abs(dk)}
    units = {"delta_k": "1/m", "period": "m", "coherence_length": "m"}
    assumptions = ["Undepleted pump", "First-order QPM, 50 % duty cycle"]
    if ng_pump is not None and ng_sh is not None and length is not None:
        require_positive(ng_pump=ng_pump, ng_sh=ng_sh, length=length)
        ddk = -4 * np.pi / wavelength**2 * (ng_sh - ng_pump)        # dΔk/dλ
        vals["fwhm"] = 4 * SINC2_HALF / (length * abs(ddk))
        units["fwhm"] = "m"
        assumptions.append("Bandwidth from group-velocity mismatch only (first order in λ)")
    return Result(values=vals, units=units, assumptions=assumptions)


def sinc2_response(delta_k_residual, length):
    """Relative SHG efficiency sinc²(Δk' L / 2) for a residual mismatch Δk' (helper, vectorised)."""
    x = np.asarray(delta_k_residual, dtype=float) * length / 2
    return np.sinc(x / np.pi) ** 2
