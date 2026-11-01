"""chi2sim: χ² waveguide frequency conversion with arbitrary poling, ensemble statistics and inverse design.

The general solvers live in the misc-applets math-engines (engines/cwe_shg, engines/phase_matching,
engines/materials); this package builds poling structures, batched fast solvers, ensemble statistics and
synthesis on top of them. It must sit inside a misc-applets checkout (next to math-engines/), or set
MISC_APPLETS=/path/to/misc-applets. See README.md."""
import os
import sys
from pathlib import Path


def _locate_engines() -> Path:
    root = Path(__file__).resolve().parents[1]
    for c in (os.environ.get("MISC_APPLETS"), root.parent, Path.home() / "misc-applets"):
        if c and (Path(c) / "math-engines" / "engines").is_dir():
            return Path(c) / "math-engines"
    raise ImportError("math-engines not found: chi2-waveguide-sim must sit inside a misc-applets checkout "
                      "(next to math-engines/), or set MISC_APPLETS=/path/to/misc-applets")


ENGINES = _locate_engines()
if str(ENGINES) not in sys.path:
    sys.path.insert(0, str(ENGINES))

from . import dispersion, ensemble, fast, materials, metrics, optimize, poling, synthesis, thg, wafer, waveguide  # noqa: E402,F401

__all__ = ["poling", "fast", "ensemble", "metrics", "synthesis", "dispersion", "materials", "waveguide", "wafer", "thg", "optimize", "ENGINES"]
