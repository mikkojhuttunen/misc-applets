"""Collect every engines/*/spec.yaml into web/engines_index.json for the Pyodide calculator.

Run from math-engines/:  python tools/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ROOT / "engines"
OUT = ROOT / "web" / "engines_index.json"
SHARED = ["engines/__init__.py", "engines/common.py", "engines/registry.py"]


def resolve_inputs(spec: dict, fn: dict) -> list[dict]:
    common = spec.get("common_inputs", {}) or {}
    out = []
    for item in fn.get("inputs", []):
        if isinstance(item, str):
            if item not in common:
                raise KeyError(f"{spec['engine']}: input {item!r} not in common_inputs")
            out.append({"name": item, **common[item]})
        else:
            out.append(dict(item))
    return out


def build() -> dict:
    engines = []
    for d in sorted(p for p in ENGINES.iterdir() if (p / "spec.yaml").is_file()):
        spec = yaml.safe_load((d / "spec.yaml").read_text(encoding="utf-8"))
        files = sorted(f"engines/{d.name}/{f.name}" for f in d.glob("*.py") if not f.name.startswith("test_"))
        engines.append({
            "engine": spec["engine"],
            "version": spec.get("version", 1),
            "title": spec.get("title", spec["engine"]),
            "description": spec.get("description", ""),
            "requires": spec.get("requires", []),
            "references": spec.get("references", []),
            "files": files,
            "functions": {
                name: {
                    "title": fn.get("title", name),
                    "equations": fn.get("equations", []),
                    "inputs": resolve_inputs(spec, fn),
                    "outputs": fn.get("outputs", []),
                }
                for name, fn in spec["functions"].items()
            },
        })
    # engines that import other engines need their files too
    deps = {"step_index_fiber": ["materials"], "membrane_mode": ["materials", "slab_waveguide"], "path_coherence": ["billiard_cell"], "ray_phase": ["billiard_cell"], "cell_mirror": ["billiard_cell", "bragg_grating", "ray_phase"],
            "thermo_optic": ["materials", "slab_waveguide"], "waveguide_thermal": ["thermo_optic", "materials", "slab_waveguide"], "thermal_detuning": ["materials"]}
    for e in engines:
        for dep in deps.get(e["engine"], []):
            e["files"] = sorted(set(e["files"]) | {f"engines/{dep}/__init__.py", f"engines/{dep}/engine.py"})
    return {"shared": SHARED, "engines": engines}


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(build()['engines'])} engines)", file=sys.stderr)
