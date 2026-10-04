"""The shared JSON test vectors must match the current Python engines.

Compared structurally with a small numerical tolerance, not as exact text: near-total
reflectors (the 800-period Bragg stacks) amplify last-digit differences between BLAS/libm
builds (macOS vs Linux) to ~1e-8 relative in T = 1 - R. Structure, keys and inputs still
have to match exactly; only floats get the tolerance.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_vectors  # noqa: E402

REL, ABS = 1e-6, 1e-12


def _same(a, b, where="vectors"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{where}: keys differ"
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{where}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=REL, abs_tol=ABS), f"{where}: {a!r} != {b!r}"
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def test_vectors_are_current():
    path = ROOT / "test_vectors" / "vectors.json"
    assert path.exists(), "test_vectors/vectors.json missing: run python tools/make_vectors.py"
    try:
        _same(json.loads(path.read_text(encoding="utf-8")), make_vectors.build())
    except AssertionError as e:
        raise AssertionError(f"test_vectors/vectors.json is stale ({e}): run python tools/make_vectors.py") from None
