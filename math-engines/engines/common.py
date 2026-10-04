"""Shared conventions for all engines.

Every engine works in SI units (m, s, rad, Hz, W). Functions that a person or
a bot calls return a ``Result`` carrying values, units and assumptions.
Unit conversion for display belongs in spec.yaml (``scale``) or the front end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# CODATA 2018 exact / recommended values, SI
C0 = 299_792_458.0            # m/s
HBAR = 1.054_571_817e-34      # J s
H_PLANCK = 6.626_070_15e-34   # J s
EPS0 = 8.854_187_8128e-12     # F/m
K_B = 1.380_649e-23           # J/K


@dataclass
class Result:
    """Values with their units and the assumptions they rest on."""

    values: dict[str, Any]
    units: dict[str, str]
    assumptions: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def to_dict(self) -> dict:
        """JSON-safe dictionary: arrays become lists, non-finite numbers become None."""
        return {
            "values": {k: json_safe(v) for k, v in self.values.items()},
            "units": dict(self.units),
            "assumptions": list(self.assumptions),
        }


def json_safe(v: Any) -> Any:
    if isinstance(v, Result):
        return v.to_dict()
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (complex, np.complexfloating)):
        return {"re": json_safe(float(v.real)), "im": json_safe(float(v.imag))}
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def require_positive(name: str, value: Any) -> None:
    """Raise ValueError unless every element of ``value`` is finite and > 0."""
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def require_nonnegative(name: str, value: Any) -> None:
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


def require_in_range(name: str, value: Any, lo: float, hi: float) -> None:
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < lo) or np.any(a > hi):
        raise ValueError(f"{name} must lie in [{lo}, {hi}], got {value!r}")


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def db_to_np(db: Any) -> Any:
    """Power ratio in dB to a natural-log power coefficient factor (dB * ln10/10)."""
    return np.asarray(db, dtype=float) * math.log(10) / 10


def angular_frequency(wavelength: Any) -> Any:
    """Vacuum wavelength (m) to angular frequency (rad/s)."""
    return 2 * math.pi * C0 / np.asarray(wavelength, dtype=float)
