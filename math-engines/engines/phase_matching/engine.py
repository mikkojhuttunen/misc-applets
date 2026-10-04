"""Three-wave phase matching with an optional quasi-phase-matching (QPM) grating.

Convention: sum-frequency process ω3 = ω1 + ω2 (SHG: ω1 = ω2; DFG and OPA are the
same process with the roles renamed). Wavelengths are vacuum wavelengths in m;
n1, n2, n3 are effective (or bulk) phase indices of the interacting modes, so the
same functions serve bulk birefringent, modal and poled phase matching: only the
indices that are passed in differ.

Δk = β3 - β1 - β2 - K_m,  β_i = 2π n_i / λ_i,  K_m = 2π m / Λ  (grating order m).
Positive Δk means the generated wave runs ahead of the driven polarisation's
phase; only |K_m| matters physically, so periods are returned as positive numbers.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_positive

SINC2_HALF = 1.391557  # x with sinc^2(x) = 1/2


def _lambda3(lambda_1, lambda_2):
    return 1.0 / (1.0 / np.asarray(lambda_1, dtype=float) + 1.0 / np.asarray(lambda_2, dtype=float))


def grating_strength(order, duty_cycle) -> float:
    """Relative effective nonlinearity |d_eff/d| = |2 sin(m π D) / (m π)| of a binary
    ±d grating of order m ≥ 1 and duty cycle D (fraction of the period with +d)."""
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")
    if not (0.0 < float(duty_cycle) < 1.0):
        raise ValueError(f"duty_cycle must lie in (0, 1), got {duty_cycle!r}")
    return abs(2.0 * np.sin(order * np.pi * float(duty_cycle)) / (order * np.pi))


def three_wave_mismatch(wavelength_1, wavelength_2, n_1, n_2, n_3, period=None, order=1) -> Result:
    """Phase mismatch of ω3 = ω1 + ω2 and the grating period that cancels it.

    Without `period`: bare Δk, required period for the given order, coherence length.
    With `period`: Δk with the grating vector K_m = 2π m/Λ subtracted (residual mismatch).
    Arrays are allowed for wavelengths and indices (broadcast together)."""
    require_positive(wavelength_1=wavelength_1, wavelength_2=wavelength_2, n_1=n_1, n_2=n_2, n_3=n_3)
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")
    l3 = _lambda3(wavelength_1, wavelength_2)
    dk0 = 2 * np.pi * (n_3 / l3 - n_1 / np.asarray(wavelength_1, dtype=float) - n_2 / np.asarray(wavelength_2, dtype=float))
    with np.errstate(divide="ignore"):
        req = order * 2 * np.pi / np.abs(dk0)
        lc = np.pi / np.abs(dk0)
    vals = {"wavelength_3": l3, "delta_k_bare": dk0, "period_required": req, "coherence_length": lc}
    units = {"wavelength_3": "m", "delta_k_bare": "1/m", "period_required": "m", "coherence_length": "m"}
    if period is not None:
        require_positive(period=period)
        vals["delta_k"] = dk0 - 2 * np.pi * order / np.asarray(period, dtype=float)
        units["delta_k"] = "1/m"
    else:
        vals["delta_k"] = dk0
        units["delta_k"] = "1/m"
    return Result(
        values=vals,
        units=units,
        assumptions=[
            "Sum-frequency convention ω3 = ω1 + ω2; indices are phase indices of the interacting modes",
            "Collinear propagation; period_required = m 2π/|Δk_bare|",
            "delta_k equals delta_k_bare when no period is given",
        ],
    )


def acceptance_response(delta_k, length):
    """Normalised conversion efficiency sinc²(Δk L / 2) (helper, vectorised)."""
    x = np.asarray(delta_k, dtype=float) * length / 2
    return np.sinc(x / np.pi) ** 2


def qpm_efficiency(wavelength_1, wavelength_2, n_1, n_2, n_3, period, length, order=1, duty_cycle=0.5) -> Result:
    """Normalised efficiency of a uniform grating relative to perfect full-strength matching:
    η = (d_eff/d)² sinc²(Δk L / 2), with Δk including the grating vector."""
    require_positive(length=length)
    mm = three_wave_mismatch(wavelength_1, wavelength_2, n_1, n_2, n_3, period=period, order=order)
    g = grating_strength(order, duty_cycle)
    eta = g**2 * acceptance_response(mm["delta_k"], length)
    return Result(
        values={"delta_k": mm["delta_k"], "grating_strength": g, "efficiency_rel": eta},
        units={"delta_k": "1/m", "grating_strength": "", "efficiency_rel": ""},
        assumptions=mm.assumptions + ["Undepleted pump, uniform grating, overlap integral supplied by the caller"],
    )
