"""Configuration: one YAML file with every parameter, optional override files on top.

Overrides may only change keys that exist in the base file (typos raise instead of being
silently ignored). Values are read-only after loading.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"


class Cfg:
    """Read-only nested view: cfg.fdtd.pml_layers or cfg["fdtd"]["pml_layers"]."""

    def __init__(self, data: dict, path: str = ""):
        self._data = MappingProxyType(data)
        self._path = path

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except KeyError:
            raise KeyError(f"config key {self._path + key!r} is missing") from None
        return Cfg(value, f"{self._path}{key}.") if isinstance(value, dict) else value

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, self[k]) for k in self._data)

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(_thaw(self._data)))


def _thaw(m):
    return {k: _thaw(v) if isinstance(v, (dict, MappingProxyType)) else v for k, v in m.items()}


def _merge(base: dict, override: dict, path: str = "", open_sections=("platforms",)) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        here = path + str(key)
        if key not in out and path.rstrip(".") not in open_sections:
            raise KeyError(f"override sets unknown config key {here!r}")
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value, here + ".", open_sections)
        else:
            out[key] = value
    return out


def load(*overrides: str | Path, base: str | Path = DEFAULT_CONFIG) -> Cfg:
    with open(base) as fh:
        data = yaml.safe_load(fh)
    for path in overrides:
        with open(path) as fh:
            data = _merge(data, yaml.safe_load(fh) or {})
    return Cfg(data)


def resolve_path(relative: str) -> Path:
    """Data paths in the config are relative to the package root."""
    return PACKAGE_ROOT / relative
