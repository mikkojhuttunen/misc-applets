"""general-mpc: planar (stadium, circle, faceted, perturbed) and Herriott-type (spherical, astigmatic, deformed)
multipass-cell ray tracing with the CMPC simulator's path statistics. See README.md."""
from . import herriott, planar, stats, trace2d  # noqa: F401

__all__ = ["planar", "trace2d", "herriott", "stats"]
