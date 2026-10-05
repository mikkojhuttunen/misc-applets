"""Put misc-applets/math-engines on sys.path: the sibling folder first, then $MISC_APPLETS, then common clone locations."""
import os
import sys
from pathlib import Path


def locate() -> Path:
    here = Path(__file__).resolve().parent
    cands = [here.parent, os.environ.get("MISC_APPLETS"), "./misc-applets", "../misc-applets", Path.home() / "misc-applets"]
    for c in cands:
        if c and (Path(c) / "math-engines" / "engines").is_dir():
            return Path(c) / "math-engines"
    raise ImportError("math-engines not found: run from a misc-applets checkout or set MISC_APPLETS=/path/to/misc-applets")


ROOT = locate()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
