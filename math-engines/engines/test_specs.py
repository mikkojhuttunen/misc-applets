"""Generic checks: every spec.yaml must match its engine, and the web index must be current."""
from __future__ import annotations

import inspect
import json
import math

import pytest
import yaml

from engines import registry
from engines.common import Result
from tools.build_index import INDEX, ROOT, build, dumps

SPECS = sorted((ROOT / "engines").glob("*/spec.yaml"))
CASES = []
for _p in SPECS:
    _s = yaml.safe_load(_p.read_text(encoding="utf-8"))
    for _f in _s["functions"]:
        CASES.append((_p.parent.name, _f))
INDEXED = {e["engine"]: {f["name"]: f for f in e["functions"]} for e in build()["engines"]}


def test_every_engine_has_a_spec_and_version():
    names = registry.list_engines()
    assert names, "no engines found"
    for name in names:
        spec = ROOT / "engines" / name / "spec.yaml"
        assert spec.exists(), f"{name} has no spec.yaml"
        assert isinstance(yaml.safe_load(spec.read_text(encoding="utf-8")).get("version"), int), f"{name}: spec needs an integer version"


@pytest.mark.parametrize("engine,function", CASES)
def test_spec_matches_signature(engine, function):
    fn = getattr(registry.load(engine), function)
    params = inspect.signature(fn).parameters
    spec = INDEXED[engine][function]
    names = [i["name"] for i in spec["inputs"]]
    unknown = set(names) - set(params)
    assert not unknown, f"spec inputs not in {function}(): {unknown}"
    required = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
    assert required <= set(names), f"required arguments missing from spec: {required - set(names)}"
    for i in spec["inputs"]:
        assert "label" in i and "default" in i, f"{engine}.{function}.{i['name']}: label and default are required"
        if "choices" in i:
            assert i["default"] in i["choices"]
        else:
            assert "scale" in i and "ui_unit" in i, f"{engine}.{function}.{i['name']}: numeric inputs need ui_unit and scale"


@pytest.mark.parametrize("engine,function", CASES)
def test_defaults_run_and_outputs_exist(engine, function):
    spec = INDEXED[engine][function]
    kwargs = {i["name"]: (i["default"] if "choices" in i else (int(i["default"]) if i.get("integer") else i["default"] * i["scale"]))
              for i in spec["inputs"]}
    out = getattr(registry.load(engine), function)(**kwargs)
    assert isinstance(out, Result)
    for o in spec["outputs"]:
        assert o["name"] in out.values, f"output {o['name']!r} not returned by {engine}.{function}"
        assert o["name"] in out.units, f"output {o['name']!r} has no unit"
        assert "scale" in o and "ui_unit" in o
    d = registry.call(engine, function, **kwargs)
    json.dumps(d)  # JSON-safe for bots and the web page
    assert all(v is None or not isinstance(v, float) or math.isfinite(v) for v in d["values"].values())


def test_web_index_is_current():
    assert INDEX.exists(), "web/engines_index.json missing: run python tools/build_index.py"
    assert INDEX.read_text(encoding="utf-8") == dumps(build()), \
        "web/engines_index.json is stale: you changed a spec without rebuilding the index (python tools/build_index.py)"
