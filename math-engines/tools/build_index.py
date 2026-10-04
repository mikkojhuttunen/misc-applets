"""Collect every engines/*/spec.yaml into web/engines_index.json.

Common inputs are resolved, so the web page needs no YAML parser. The index also
lists the Python files Pyodide must load. Run from math-engines/:

    python tools/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "engines_index.json"


def _check_numbers(d, where):
    # PyYAML reads 1.0e6 (no exponent sign) as a string; write 1.0e+6
    for key in ("scale", "default", "min", "max"):
        if key in d and "choices" not in d and not isinstance(d[key], (int, float)):
            raise TypeError(f"{where}: {key}={d[key]!r} is not a number (use 1.0e+6, not 1.0e6)")
    return d


def _resolve(item, common, where):
    if isinstance(item, str):
        if item not in common:
            raise KeyError(f"{where}: input {item!r} is not defined under common_inputs")
        return _check_numbers({"name": item, **common[item]}, f"{where}.{item}")
    return _check_numbers(dict(item), f"{where}.{item.get('name')}")


def build() -> dict:
    engines, files = [], ["engines/__init__.py", "engines/common.py", "engines/registry.py"]
    for spec_path in sorted((ROOT / "engines").glob("*/spec.yaml")):
        name = spec_path.parent.name
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        common = spec.get("common_inputs", {}) or {}
        funcs = []
        for fname, f in spec["functions"].items():
            funcs.append({
                "name": fname,
                "title": f.get("title", fname),
                "equations": f.get("equations", []),
                "inputs": [_resolve(i, common, f"{name}.{fname}") for i in f.get("inputs", [])],
                "outputs": [_check_numbers(dict(o), f"{name}.{fname}.{o.get('name')}") for o in f.get("outputs", [])],
            })
        engines.append({"engine": spec.get("engine", name), "version": spec.get("version"), "title": spec.get("title", name),
                        "description": spec.get("description", ""), "references": spec.get("references", []), "functions": funcs})
        files += [f"engines/{name}/__init__.py", f"engines/{name}/engine.py"]
    return {"generated_by": "tools/build_index.py", "files": files, "engines": engines}


def dumps(index: dict) -> str:
    return json.dumps(index, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    INDEX.parent.mkdir(exist_ok=True)
    INDEX.write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
