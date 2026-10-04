"""Every spec.yaml must match its engine.py, and the web index must be current."""
import importlib
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_index  # noqa: E402

from engines.common import Result  # noqa: E402

INDEX = build_index.build()
CASES = [(e, name, fn) for e in INDEX["engines"] for name, fn in e["functions"].items()]


@pytest.mark.parametrize("eng,name,fn", CASES, ids=[f"{e['engine']}.{n}" for e, n, _ in CASES])
def test_spec_matches_engine(eng, name, fn):
    mod = importlib.import_module(f"engines.{eng['engine']}.engine")
    func = getattr(mod, name, None)
    assert func is not None, f"{eng['engine']}.{name} missing in engine.py"
    params = inspect.signature(func).parameters
    names = [i["name"] for i in fn["inputs"]]
    for n in names:
        assert n in params, f"spec input {n!r} is not an argument of {name}"
    for p in params.values():
        if p.default is inspect.Parameter.empty:
            assert p.name in names, f"required argument {p.name!r} of {name} missing from spec"
    kwargs = {i["name"]: (i["default"] if i.get("type") == "choice" else i["default"] * i.get("scale", 1.0)) for i in fn["inputs"]}
    for i in fn["inputs"]:
        if i.get("step") == 1:
            kwargs[i["name"]] = int(round(kwargs[i["name"]]))
    out = func(**kwargs)
    assert isinstance(out, Result), f"{name} must return a Result"
    for o in fn["outputs"]:
        assert o["name"] in out.values, f"spec output {o['name']!r} not returned by {name}"
        assert o["name"] in out.units, f"output {o['name']!r} has no unit in the Result"


def test_spec_has_version_and_references():
    for e in INDEX["engines"]:
        assert isinstance(e["version"], int)
        assert e["references"], f"{e['engine']} lists no references"


def test_web_index_is_current():
    path = ROOT / "web" / "engines_index.json"
    assert path.exists() and path.read_text(encoding="utf-8") == build_index.render(), \
        "web/engines_index.json is stale: run python tools/build_index.py"
