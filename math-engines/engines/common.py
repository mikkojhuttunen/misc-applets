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
        if v.ndim == 0:
            return jsonable(v[()])
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


def half_max_width(x, y) -> float:
    """Full width of the contiguous region around argmax(y) where y >= max(y)/2, with linear
    interpolation at both edges; NaN if the region reaches either end of the grid."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    k = int(np.argmax(y))
    half = y[k] / 2
    lo, hi = k, k
    while lo > 0 and y[lo - 1] >= half:
        lo -= 1
    while hi < len(y) - 1 and y[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == len(y) - 1:
        return float("nan")
    edge = lambda a, b: x[a] + (half - y[a]) * (x[b] - x[a]) / (y[b] - y[a])
    return float(abs(edge(hi, hi + 1) - edge(lo - 1, lo)))
