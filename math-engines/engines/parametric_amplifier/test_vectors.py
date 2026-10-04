"""The shared JSON vectors used to verify the JavaScript port must match this reference implementation."""
from tools.make_test_vectors import OUT, build, dumps


def test_vectors_are_current():
    assert OUT.exists(), "test_vectors.json missing: run python tools/make_test_vectors.py"
    assert OUT.read_text(encoding="utf-8") == dumps(build()), \
        "test_vectors.json is stale: the Python engine changed; rerun python tools/make_test_vectors.py and the JavaScript tests"
