"""Uniform entry point for bots and web front ends.

    call("gaussian_beam", "beam_parameters", w0=1e-3, wavelength=633e-9)

returns a JSON-safe dictionary (infinities become None).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from .common import Result, json_safe

_ROOT = Path(__file__).resolve().parent


def list_engines() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules([str(_ROOT)]) if m.ispkg)


def load(engine: str):
    if engine not in list_engines():
        raise KeyError(f"unknown engine {engine!r}; available: {list_engines()}")
    return importlib.import_module(f"engines.{engine}.engine")


def call(engine: str, function: str, **kwargs) -> dict:
    mod = load(engine)
    fn = getattr(mod, function, None)
    if fn is None or function.startswith("_") or not inspect.isfunction(fn):
        raise KeyError(f"engine {engine!r} has no public function {function!r}")
    out = fn(**kwargs)
    if isinstance(out, Result):
        return out.to_dict()
    return {"values": {"value": json_safe(out)}, "units": {}, "assumptions": []}
