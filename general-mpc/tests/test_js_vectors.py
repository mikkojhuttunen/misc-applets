import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_js_vectors  # noqa: E402


def _same(a, b, where="v"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12), f"{where}: {a} != {b}"
    else:
        assert a == b, where


def test_js_vectors_are_current():
    path = ROOT / "tests" / "js_vectors.json"
    assert path.exists(), "run python tools/make_js_vectors.py"
    _same(json.loads(path.read_text()), json.loads(make_js_vectors.render()))
