"""Schematic gas molecules (ball and stick) for the 3D figures, and the evanescent-field cloud that senses them.

Molecules: CH4, CO2, H2O, NH3 with textbook geometries (bond angles, not bond lengths: every bond is drawn the same
length b) and CPK colours. A molecule whose centre lies inside the evanescent layer (within `reach` of the membrane
surface) is drawn with a red halo: it overlaps the guided mode's evanescent field and absorbs; molecules further
out do not interact.
"""
from __future__ import annotations

import numpy as np

CPK = {"C": "#3d3d3d", "H": "#f4f4f4", "O": "#d33a2c", "N": "#3a5bd9"}
RAD = {"C": 0.40, "H": 0.27, "O": 0.38, "N": 0.39}        # atom radius / bond length

_T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
_W = np.radians(104.5 / 2)
_A = np.radians(68.0)                                    # NH3: H–N–H ≈ 107°
SHAPES = {
    "CH4": [("C", (0, 0, 0))] + [("H", tuple(v)) for v in _T],
    "CO2": [("C", (0, 0, 0)), ("O", (1, 0, 0)), ("O", (-1, 0, 0))],
    "H2O": [("O", (0, 0, 0)), ("H", (np.cos(_W), np.sin(_W), 0)), ("H", (np.cos(_W), -np.sin(_W), 0))],
    "NH3": [("N", (0, 0, 0))] + [("H", (np.sin(_A) * np.cos(p), np.sin(_A) * np.sin(p), -np.cos(_A)))
                                  for p in np.radians([0, 120, 240])],
}


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def add_molecule(scene, kind, centre, b, rng, sensed=False, halo="#ff2a1a"):
    """Draw one molecule (bond length b, random orientation) into a render3d.Scene."""
    Rm = _rotation(rng)
    atoms = [(el, np.asarray(centre) + b * Rm @ np.asarray(p, float)) for el, p in SHAPES[kind]]
    for el, p in atoms:
        scene.sphere(p, RAD[el] * b, CPK[el])
    hub = atoms[0]
    for el, p in atoms[1:]:
        u = (p - hub[1]) / np.linalg.norm(p - hub[1])
        scene.line([hub[1] + u * RAD[hub[0]] * b * 0.9, p - u * RAD[el] * b * 0.9], "#6b7378", width=1.6)
    if sensed:                                           # soft halo: camera-facing discs behind the molecule
        rmax = b * 1.55
        for f, al in ((1.0, 0.16), (0.78, 0.18), (0.55, 0.2)):
            ang = np.linspace(0, 2 * np.pi, 28, endpoint=False)
            c = np.asarray(centre) - scene.d * rmax * 1.05
            ring = c + f * rmax * (np.cos(ang)[:, None] * scene.right + np.sin(ang)[:, None] * scene.up)
            scene.sheet(ring, al, halo)


def scatter(scene, region, z_bands, n, b, rng, sensed_band, kinds=("CH4", "CO2", "H2O", "NH3")):
    """Place n molecules uniformly over a shapely region, heights drawn from z_bands [(z0, z1, weight), ...];
    those whose height falls in sensed_band (z0, z1) get the halo. Returns the list of (kind, centre, sensed)."""
    from shapely.geometry import Point
    minx, miny, maxx, maxy = region.bounds
    w = np.array([zb[2] for zb in z_bands], float)
    w /= w.sum()
    out, tries = [], 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not region.contains(Point(x, y)):
            continue
        z0, z1, _ = z_bands[rng.choice(len(z_bands), p=w)]
        z = rng.uniform(z0, z1)
        if any(np.hypot(x - c[0], y - c[1]) < 2.6 * b and abs(z - c[2]) < 2.6 * b for _, c, _ in out):
            continue
        k = kinds[rng.integers(len(kinds))]
        sensed = sensed_band[0] <= z <= sensed_band[1]
        add_molecule(scene, k, (x, y, z), b, rng, sensed)
        out.append((k, (x, y, z), sensed))
    return out


def cloud(scene, region, z_surface, reach, direction=1, layers=5, alpha=0.035, color="#ff2a1a"):
    """Evanescent-field cloud over a shapely polygon: stacked translucent sheets from the membrane surface out to
    `reach`, opacity falling like the intensity exp(-2γ h) (reach = 3 decay lengths of the intensity)."""
    geoms = getattr(region, "geoms", [region])
    for k in range(layers):
        h = (k + 0.5) / layers * reach
        a = alpha * np.exp(-3 * h / reach) * 1.6
        for g in geoms:
            xy = np.asarray(g.exterior.coords)[:-1]
            scene.sheet(np.c_[xy, np.full(len(xy), z_surface + direction * h)], a, color)
