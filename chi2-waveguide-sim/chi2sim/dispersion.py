"""Phase-mismatch models Δk(λ) for SHG until the waveguide layer (milestone 3) supplies modal indices.

Each model is a callable lam -> Δk = β_SH - 2 β_pump (1/m) at pump vacuum wavelength lam, with the sign
convention of math-engines `phase_matching`. All of them can be replaced by any user function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engines.materials import engine as mat


@dataclass(frozen=True)
class LinearIndexSHG:
    """Effective indices linear in λ around λ0 from phase and group index at pump and SH.

    n(λ) = n0 + (n0 - n_g)(λ - λ_ref)/λ_ref for each wave (λ_ref = λ0 for the pump, λ0/2 for the SH).
    The default numbers are the TFLN-ridge-like values used in the math-engines `qpm_shg` spec."""

    lambda0: float = 1.55e-6
    n_pump: float = 2.0272
    ng_pump: float = 2.33
    n_sh: float = 2.1924
    ng_sh: float = 2.40

    def n_p(self, lam):
        return self.n_pump + (self.n_pump - self.ng_pump) * (lam - self.lambda0) / self.lambda0

    def n_s(self, lam):
        l2, r = lam / 2, self.lambda0 / 2
        return self.n_sh + (self.n_sh - self.ng_sh) * (l2 - r) / r

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=float)
        return 4 * np.pi * (self.n_s(lam) - self.n_p(lam)) / lam


@dataclass(frozen=True)
class MaterialSHG:
    """Bulk Sellmeier indices (math-engines `materials`), e.g. "ln_e_cong" for a type-0 d33 interaction."""

    pump: str = "ln_e_cong"
    sh: str | None = None

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=float)
        return 4 * np.pi * (mat.index(self.sh or self.pump, lam / 2) - mat.index(self.pump, lam)) / lam


def dk_slope(dk_of_lambda, lam, rel=1e-4):
    """dΔk/dλ by central difference."""
    h = lam * rel
    return (dk_of_lambda(lam + h) - dk_of_lambda(lam - h)) / (2 * h)
