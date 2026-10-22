"""Optical component graphics for the 2D schematics.

If a component SVG exists in the components folder (default mpc-schematics/components/, or $GWOPTICS_DIR), it is
used: exported from Alexander Franzen's ComponentLibrary (https://www.gwoptics.org/ComponentLibrary/, CC BY-NC 3.0,
attribution required, non-commercial). Expected file names: laser.svg, isolator.svg, lens.svg, photodiode.svg
(see components/README.md for how to export them). Otherwise a vector stand-in in the same style is drawn: shaded
3D laser head, cylindrical Faraday isolator with its arrow, biconvex lens, photodiode can.

    place(ax, "laser", x, y, w, angle=0)     centre (x, y), width w in data units, rotation in degrees
    used_library()                           True if any library SVG was drawn (add the attribution then)
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from matplotlib import transforms
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

DIR = Path(os.environ.get("GWOPTICS_DIR", Path(__file__).resolve().parent / "components"))
ATTRIBUTION = "Optical component graphics: A. Franzen, ComponentLibrary (gwoptics.org/ComponentLibrary), CC BY-NC 3.0"
_cache: dict = {}
_used = {"library": False}


def used_library():
    return _used["library"]


def _svg(name, px=900):
    if name in _cache:
        return _cache[name]
    f = DIR / f"{name}.svg"
    img = None
    if f.exists():
        try:
            import cairosvg
            from PIL import Image
            png = cairosvg.svg2png(url=str(f), output_width=px)
            img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        except Exception as e:                          # missing cairo, bad file: fall back to the stand-in
            print(f"components: could not render {f}: {e}")
    _cache[name] = img
    return img


def place(ax, name, x, y, w, angle=0.0, z=5):
    img = _svg(name)
    if img is not None:
        h = w * img.shape[0] / img.shape[1]
        tr = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
        ax.imshow(img, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2), transform=tr, zorder=z, interpolation="lanczos")
        _used["library"] = True
        return h
    return STANDINS[name](ax, x, y, w, angle, z)


# ---------------------------------------------------------------- stand-ins (vector, same visual language)
def _rot(pts, x, y, angle):
    a = np.radians(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(pts, float) - [x, y]) @ R.T + [x, y]


def _laser(ax, x, y, w, angle, z):
    """Shaded box laser head with an output aperture on the +x face."""
    h, dep = 0.42 * w, 0.18 * w
    x0, y0 = x - w / 2, y - h / 2
    front = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]
    top = [[x0, y0 + h], [x0 + w, y0 + h], [x0 + w + dep * 0.6, y0 + h + dep * 0.5], [x0 + dep * 0.6, y0 + h + dep * 0.5]]
    side = [[x0 + w, y0], [x0 + w + dep * 0.6, y0 + dep * 0.5], [x0 + w + dep * 0.6, y0 + h + dep * 0.5], [x0 + w, y0 + h]]
    for P, fc in ((front, "#4a5359"), (top, "#7d878d"), (side, "#343b40")):
        ax.add_patch(Polygon(_rot(P, x, y, angle), fc=fc, ec="#1b2328", lw=0.8, zorder=z))
    ax.add_patch(Rectangle((x0 + 0.08 * w, y0 + 0.62 * h), 0.3 * w, 0.18 * h, fc="#c9d1d6", ec="none", zorder=z + 0.1,
                           transform=transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData))
    ap = _rot([[x0 + w + dep * 0.3, y0 + h / 2 + dep * 0.25]], x, y, angle)[0]
    ax.add_patch(Ellipse(ap, 0.07 * w, 0.16 * w, angle=angle, fc="#d0352b", ec="#1b2328", lw=0.6, zorder=z + 0.2))
    return h


def _isolator(ax, x, y, w, angle, z):
    """Faraday isolator: cylinder with end caps and a direction arrow."""
    h = 0.48 * w
    tr = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="#9aa6ad", ec="#1b2328", lw=0.8, zorder=z, transform=tr))
    for xe, fc in ((x - w / 2, "#7d878d"), (x + w / 2, "#c3ccd1")):
        ax.add_patch(Ellipse((xe, y), 0.22 * h, h, fc=fc, ec="#1b2328", lw=0.8, zorder=z + 0.1, transform=tr))
    ax.add_patch(Rectangle((x - w / 2, y + 0.18 * h), w, 0.12 * h, fc="#c3ccd1", ec="none", zorder=z + 0.05, transform=tr))
    a0, a1 = _rot([[x - 0.25 * w, y - 0.12 * h], [x + 0.25 * w, y - 0.12 * h]], x, y, angle)
    ax.add_patch(FancyArrowPatch(a0, a1, arrowstyle="-|>", mutation_scale=10, color="#1b2328", lw=1.2, zorder=z + 0.2))
    return h


def _lens(ax, x, y, w, angle, z):
    """Biconvex lens; w is the lens height (aperture), thickness 0.28 w."""
    t = 0.28 * w
    s = np.linspace(-1, 1, 30)
    right = np.c_[t / 2 * np.sqrt(1 - s**2) + 0.0, s * w / 2]
    left = np.c_[-t / 2 * np.sqrt(1 - s[::-1] ** 2), s[::-1] * w / 2]
    P = _rot(np.vstack([right, left]) + [x, y], x, y, angle)
    ax.add_patch(Polygon(P, closed=True, fc="#bfe1f2", ec="#3a6f8f", lw=0.9, alpha=0.95, zorder=z))
    hl = _rot(np.c_[-0.15 * t * np.ones(8), np.linspace(0.1, 0.35, 8) * w] + [x, y], x, y, angle)
    ax.plot(hl[:, 0], hl[:, 1], color="white", lw=1.2, alpha=0.9, zorder=z + 0.1)
    return w


def _photodiode(ax, x, y, w, angle, z):
    """Photodiode can seen from the side: window face (towards -x), body, two leads."""
    h = 0.62 * w
    tr = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    ax.add_patch(Rectangle((x - 0.25 * w, y - h / 2), 0.55 * w, h, fc="#2f363b", ec="#1b2328", lw=0.8, zorder=z, transform=tr))
    ax.add_patch(Rectangle((x - 0.25 * w, y + 0.18 * h), 0.55 * w, 0.1 * h, fc="#5d676e", ec="none", zorder=z + 0.05, transform=tr))
    ax.add_patch(Ellipse((x - 0.25 * w, y), 0.2 * w, h, fc="#4d6f8a", ec="#1b2328", lw=0.8, zorder=z + 0.1, transform=tr))
    ax.add_patch(Ellipse((x - 0.25 * w, y), 0.09 * w, 0.42 * h, fc="#9ec3dd", ec="none", zorder=z + 0.2, transform=tr))
    for dy in (-0.18, 0.18):
        p0, p1 = _rot([[x + 0.3 * w, y + dy * h], [x + 0.5 * w, y + dy * h]], x, y, angle)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#8a949b", lw=1.4, zorder=z)
    return h


STANDINS = {"laser": _laser, "isolator": _isolator, "lens": _lens, "photodiode": _photodiode}
