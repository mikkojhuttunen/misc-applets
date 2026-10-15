"""dbr-structures-standalone.html must be the current build of dbr-structures/index.html + dbr-engine.js."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _strip(text):
    return "\n".join(l for l in text.splitlines() if "Source: github.com/" not in l)    # the commit hash changes


def test_dbr_standalone_is_current():
    spec = importlib.util.spec_from_file_location("build_standalone", REPO / "tools" / "build_standalone.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    name = "dbr-structures/index.html"
    built, saved = bs.build(name), bs.output_path(name).read_text(encoding="utf-8")
    assert _strip(saved) == _strip(built), "dbr-structures-standalone.html is stale: run python tools/build_standalone.py dbr-structures/index.html"
