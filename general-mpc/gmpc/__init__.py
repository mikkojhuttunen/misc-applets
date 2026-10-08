"""general-mpc: planar (stadium, circle, faceted, perturbed, integrated Herriott) and Herriott-type (spherical,
astigmatic, deformed) multipass-cell ray tracing with the CMPC simulator's path statistics. See README.md.

The physics lives in the misc-applets math-engines (engines/planar_cell, engines/herriott_cell, and ray_phase /
cell_mirror for phase, dither and mirror analyses); these modules re-export it under the gmpc names."""
import os
import sys
from pathlib import Path


def _locate_engines() -> Path:
    """math-engines of misc-applets: normally ../math-engines; override with $MISC_APPLETS=/path/to/misc-applets."""
    root = Path(__file__).resolve().parents[1]
    for c in (os.environ.get("MISC_APPLETS"), root.parent, Path.home() / "misc-applets"):
        if c and (Path(c) / "math-engines" / "engines").is_dir():
            return Path(c) / "math-engines"
    raise ImportError("math-engines not found: general-mpc must sit inside a misc-applets checkout (next to math-engines/), "
                      "or set MISC_APPLETS=/path/to/misc-applets")


ENGINES = _locate_engines()
if str(ENGINES) not in sys.path:
    sys.path.insert(0, str(ENGINES))

from . import herriott, planar, stats, trace2d  # noqa: E402,F401

__all__ = ["planar", "trace2d", "herriott", "stats", "ENGINES"]
