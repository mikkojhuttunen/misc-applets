"""Δn_eff images: a pixel map placed in the cell, interpolated with a bicubic spline (C², so the ray tracer gets a
continuous ∇n and Hessian). Load from PNG/TIFF/.npy, or rasterise analytic shapes."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


class GridIndex:
    """img[iy, ix] = Δn on a square-pixel grid of pitch `pitch` centred at (x0, y0). Zero outside the image.

    `smooth` (m) is a Gaussian blur applied before interpolation; it sets the edge width the rays see. Binary
    images need smooth ≳ λ/n_eff (ray-optics validity) and ≳ 2 pixels (no staircase in ∇n)."""

    def __init__(self, img, pitch, x0=0.0, y0=0.0, smooth=0.0, pad=None):
        img = np.asarray(img, float)
        if smooth > 0:
            pad = int(np.ceil(4 * smooth / pitch)) + 4 if pad is None else pad
            img = np.pad(img, pad)
            img = gaussian_filter(img, smooth / pitch)
        else:
            pad = 4 if pad is None else pad
            img = np.pad(img, pad)
        ny, nx = img.shape
        self.img, self.pitch, self.x0, self.y0, self.smooth = img, float(pitch), float(x0), float(y0), float(smooth)
        self.xs = x0 + (np.arange(nx) - (nx - 1) / 2) * pitch
        self.ys = y0 + (np.arange(ny) - (ny - 1) / 2) * pitch
        self._sp = RectBivariateSpline(self.ys, self.xs, img, kx=3, ky=3, s=0)
        peak = np.max(np.abs(img)) or 1.0
        iy, ix = np.nonzero(np.abs(img) > 1e-6 * peak)
        if len(ix):
            d = np.hypot(self.xs[ix] - x0, self.ys[iy] - y0)
            self._bound = float(d.max() + 2 * pitch)
        else:
            self._bound = 0.0

    @property
    def bound(self):
        return self._bound

    @property
    def extent(self):
        return (self.xs[0], self.xs[-1], self.ys[0], self.ys[-1])

    def eval(self, x, y, hess=True):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        shp = x.shape
        xf, yf = x.ravel(), y.ravel()
        inside = (xf >= self.xs[0]) & (xf <= self.xs[-1]) & (yf >= self.ys[0]) & (yf <= self.ys[-1])
        orders = [(0, 0), (0, 1), (1, 0)] + ([(0, 2), (1, 1), (2, 0)] if hess else [])
        out = []
        for oy, ox in orders:       # spline axes are (y, x)
            v = np.zeros(xf.shape)
            if inside.any():
                v[inside] = self._sp.ev(yf[inside], xf[inside], dx=oy, dy=ox)
            out.append(v.reshape(shp))
        return out

    def to_dict(self):
        return dict(kind="grid", pitch=self.pitch, x0=self.x0, y0=self.y0, smooth=self.smooth, shape=list(self.img.shape))


def load_image(path, invert=False):
    """Grey level in [0, 1] from an image file (PNG, TIFF, ...) or a .npy array; row 0 is the top of the image, so
    it is flipped to have +y up."""
    if str(path).endswith(".npy"):
        g = np.load(path).astype(float)
    else:
        from PIL import Image
        g = np.asarray(Image.open(path).convert("L"), float) / 255.0
    g = g[::-1]
    return 1 - g if invert else g


def from_image(path_or_array, pitch, dn, x0=0.0, y0=0.0, smooth=2e-6, invert=False, threshold=None):
    """GridIndex from an image: Δn = dn · grey (optionally binarised at `threshold`)."""
    g = load_image(path_or_array, invert) if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__") \
        else np.asarray(path_or_array, float)
    if threshold is not None:
        g = (g > threshold).astype(float)
    return GridIndex(dn * g, pitch, x0, y0, smooth=smooth)


def rasterize_shape(shape, pitch, half_width=None, binary=False):
    """Image of an analytic Shape on a grid centred at the shape (for image round trips)."""
    h = shape.bound if half_width is None else half_width
    n = int(np.ceil(2 * h / pitch)) | 1
    u = (np.arange(n) - (n - 1) / 2) * pitch
    X, Y = np.meshgrid(shape.x0 + u, shape.y0 + u)
    v = shape.eval(X, Y, hess=False)[0]
    if binary:
        v = (v > 0.5 * shape.dn).astype(float) * shape.dn
    return v
