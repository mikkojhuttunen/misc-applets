"""Waveguide geometry builders.

Coordinates (um): x = propagation, y = lateral, z = film normal; the film bottom is z = 0.
Sidewall angles in the config are measured from the film plane (90 = vertical); Tidy3D's
PolySlab.sidewall_angle is measured from the vertical, so it is (90 - angle) in radians.

Sidewall perturbations are given as top-view offsets h(x) of each wall, positive outwards
(wider guide). The offset applies to the whole wall height (columnar); z-dependent defects
are stacked PolySlabs (defects module, later milestone).
"""

from .guide import Guide, Segment, build_guide, WallProfile

__all__ = ["Guide", "Segment", "build_guide", "WallProfile"]
