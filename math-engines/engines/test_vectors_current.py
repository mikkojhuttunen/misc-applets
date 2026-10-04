"""The shared JSON test vectors must match the current Python engines."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_vectors  # noqa: E402


def test_vectors_are_current():
    path = ROOT / "test_vectors" / "vectors.json"
    assert path.exists() and path.read_text(encoding="utf-8") == make_vectors.render(), \
        "test_vectors/vectors.json is stale: run python tools/make_vectors.py"
