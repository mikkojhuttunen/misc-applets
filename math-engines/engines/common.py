"""Shared conventions for all engines: the Result container and input validation.

Every engine works in SI units (m, s, rad, Hz, W). Unit conversion for display
belongs in spec.yaml (scale) or in the front end, never in engine code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Result:
    """Values with their SI units and the assumptions behind them."""

    values: dict[str, Any]
    units: dict[str, str]
    assumptions: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary: arrays become lists, non-finite numbers become None."""
        return {
            "values": {k: jsonable(v) for k, v in self.values.items()},
            "units": dict(self.units),
            "assumptions": list(self.assumptions),
        }


def jsonable(v: Any) -> Any:
    if isinstance(v, Result):
        return v.to_dict()
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (complex, np.complexfloating)):
        return {"re": jsonable(float(np.real(v))), "im": jsonable(float(np.imag(v)))}
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def require_positive(**kwargs: Any) -> None:
    """Raise ValueError naming the quantity if any value is not strictly positive."""
    for name, value in kwargs.items():
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")


def require_nonnegative(**kwargs: Any) -> None:
    for name, value in kwargs.items():
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


def require_range(name: str, value: Any, lo: float, hi: float) -> None:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr < lo) or np.any(arr > hi):
        raise ValueError(f"{name} must lie in [{lo}, {hi}], got {value!r}")


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
