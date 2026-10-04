"""Uniform entry point for bots, web pages and scripts.

call("gaussian_beam", "beam_parameters", w0=1e-3, wavelength=633e-9)
returns a JSON-safe dictionary (infinities and NaN become None).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .common import Result, jsonable

ENGINE_DIR = Path(__file__).resolve().parent


def available() -> list[str]:
    return sorted(p.name for p in ENGINE_DIR.iterdir() if (p / "engine.py").is_file())


def load(engine: str):
    if engine not in available():
        raise KeyError(f"unknown engine {engine!r}; available: {available()}")
    return importlib.import_module(f"{__package__}.{engine}.engine")


def call(engine: str, function: str, **kwargs: Any) -> dict[str, Any]:
    mod = load(engine)
    fn = getattr(mod, function, None)
    if fn is None or function.startswith("_"):
        raise KeyError(f"engine {engine!r} has no public function {function!r}")
    out = fn(**kwargs)
    if isinstance(out, Result):
        return out.to_dict()
    return {"values": {"result": jsonable(out)}, "units": {}, "assumptions": []}
