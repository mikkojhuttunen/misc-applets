"""Guided modes of planar waveguides with anisotropic layers (principal axes aligned with the slab).

Geometry: slab normal x, propagation z, width direction y; layers listed bottom to top, the first
and last semi-infinite. Every layer is described by the three indices seen along (normal x,
transverse y, propagation z) = (n_normal, n_trans, n_prop); a plain number means an isotropic layer.

With principal axes along x, y, z the two polarisations decouple:
  TE (E along y):  kx² = k0² n_trans² - β²,                      continuity of E_y and dE_y/dx
  TM (H along y):  kx² = k0² n_prop² - β² n_prop²/n_normal²,      continuity of H_y and (1/n_prop²) dH_y/dx
so TE sees only n_trans, and a TM mode in the guided region is limited by n_normal (its cut-off-free
upper bound), while n_prop only weights the interface condition. Propagation directions that are not
a principal axis of the crystal couple TE and TM (hybrid modes): use a 2D solver for those.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..common import Result, require_choice, require_nonnegative, require_positive

_POL = ("TE", "TM")
_AXES = ("x", "y", "z")
_N_SCAN = 6000


def uniaxial_principal(n_o: float, n_e: float) -> tuple[float, float, float]:
    """Principal indices (n_x, n_y, n_z) of a uniaxial crystal with its optic axis along z."""
    return (float(n_o), float(n_o), float(n_e))


def oriented_indices(principal, normal: str, propagation: str) -> tuple[float, float, float]:
    """(n_normal, n_trans, n_prop) of a crystal with principal indices (n_x, n_y, n_z) cut with the
    surface normal along crystal axis `normal` and a mode propagating along crystal axis `propagation`.
    The in-plane transverse direction is the remaining crystal axis."""
    require_choice("normal", normal, _AXES)
    require_choice("propagation", propagation, _AXES)
    if normal == propagation:
        raise ValueError("propagation direction must lie in the surface plane (normal != propagation)")
    n = [float(v) for v in principal]
    if len(n) != 3:
        raise ValueError("principal must hold the three indices (n_x, n_y, n_z)")
    require_positive(principal=n)
    trans = next(a for a in _AXES if a not in (normal, propagation))
    return n[_AXES.index(normal)], n[_AXES.index(trans)], n[_AXES.index(propagation)]


def _triple(layer) -> tuple[float, float, float]:
    if np.ndim(layer) == 0:
        v = float(layer)
        return v, v, v
    t = tuple(float(v) for v in layer)
    if len(t) != 3:
        raise ValueError("a layer is a number or (n_normal, n_trans, n_prop)")
    return t


def _coeffs(layer, pol):
    """kx² = k0² a - β² b ; q weights dF/dx in the interface condition."""
    nn, nt, npr = layer
    if pol == "TE":
        return nt * nt, 1.0, 1.0
    return npr * npr, npr * npr / (nn * nn), npr * npr


def _char(neff, k0, layers, thick, pol):
    """Characteristic function of the stack; its zeros are the guided modes (sign is meaningful)."""
    neff = np.atleast_1d(np.asarray(neff, dtype=float))
    beta2 = (k0 * neff) ** 2
    a, b, q = _coeffs(layers[0], pol)
    F = np.ones_like(neff)
    G = np.sqrt(np.maximum(beta2 * b - k0 * k0 * a, 0.0)) / q / k0
    for lay, d in zip(layers[1:-1], thick):
        a, b, q = _coeffs(lay, pol)
        kx2 = (k0 * k0 * a - beta2 * b) / (k0 * k0)          # dimensionless: lengths in units of 1/k0
        dd = d * k0
        k = np.sqrt(np.abs(kx2))
        kd = np.minimum(k * dd, 700.0)
        small = k < 1e-9
        ks = np.where(small, 1.0, k)
        c = np.where(kx2 > 0, np.cos(kd), np.cosh(kd))
        sk = np.where(small, dd, np.where(kx2 > 0, np.sin(kd) / ks, np.sinh(kd) / ks))
        F, G = F * c + q * G * sk, -kx2 * sk * F / q + G * c
        s = np.hypot(F, G)
        F, G = F / s, G / s
    a, b, q = _coeffs(layers[-1], pol)
    gc = np.sqrt(np.maximum(beta2 * b - k0 * k0 * a, 0.0)) / k0
    return G + gc / q * F


def _limit(layer, pol):
    a, b, _ = _coeffs(layer, pol)
    return float(np.sqrt(a / b))   # n_trans for TE, n_normal for TM


def _modes(wavelength, layers, thick, pol):
    k0 = 2 * np.pi / wavelength
    lo = max(_limit(layers[0], pol), _limit(layers[-1], pol))
    hi = max(_limit(l, pol) for l in layers[1:-1])
    if hi <= lo:
        return np.array([])
    grid = np.concatenate([[lo + 1e-12], lo + (hi - lo) * np.arange(1, _N_SCAN) / _N_SCAN, [hi]])
    f = _char(grid, k0, layers, thick, pol)
    roots = []
    for i in np.nonzero(np.sign(f[1:]) * np.sign(f[:-1]) < 0)[0]:
        roots.append(brentq(lambda v: float(_char(v, k0, layers, thick, pol)[0]), grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14))
    return np.array(sorted(roots, reverse=True))


def slab_modes(wavelength, layers, thicknesses, polarization="TE") -> Result:
    """All guided modes of a stack with anisotropic layers.

    layers: bottom to top, each a number (isotropic) or (n_normal, n_trans, n_prop); first and last are
    semi-infinite. thicknesses: the interior layers, len(layers) - 2 values (m). Returns neff in descending order."""
    require_positive(wavelength=wavelength)
    require_choice("polarization", polarization, _POL)
    lay = [_triple(l) for l in layers]
    t = [float(v) for v in thicknesses]
    if len(lay) < 3 or len(t) != len(lay) - 2:
        raise ValueError("need at least three layers and len(layers) - 2 thicknesses")
    for l in lay:
        require_positive(indices=l)
    require_nonnegative(thicknesses=t)
    neff = _modes(wavelength, lay, t, polarization)
    return Result(
        values={"neff": neff, "n_modes": int(len(neff))},
        units={"neff": "", "n_modes": ""},
        assumptions=[
            "Lossless layers with principal axes along the slab normal, transverse and propagation directions (TE/TM decoupled)",
            "TE: E along the transverse axis, uses n_trans; TM: H along the transverse axis, uses n_prop and n_normal",
            "Modes are found as sign changes of the transfer-matrix characteristic function on a fine grid; two modes closer than ~1e-4 in neff can be missed",
        ],
    )


def neff_uniaxial_film(wavelength, n_o, n_e, thickness, cut="x", propagation="y", polarization="TE", order=0, n_sub=1.444, n_clad=1.0) -> Result:
    """Mode `order` of a uniaxial film (optic axis = crystal z) between isotropic substrate and cladding.

    cut: crystal axis along the surface normal; propagation: crystal axis along the propagation direction
    (must differ from the cut and must be a principal axis). neff is NaN when the mode is not guided."""
    require_positive(wavelength=wavelength, n_o=n_o, n_e=n_e, thickness=thickness, n_sub=n_sub, n_clad=n_clad)
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    film = oriented_indices(uniaxial_principal(n_o, n_e), cut, propagation)
    modes = slab_modes(wavelength, [n_sub, film, n_clad], [thickness], polarization)
    neff = modes["neff"]
    return Result(
        values={"neff": float(neff[order]) if order < len(neff) else float("nan"), "n_modes": modes["n_modes"],
                "n_normal": film[0], "n_trans": film[1], "n_prop": film[2]},
        units={"neff": "", "n_modes": "", "n_normal": "", "n_trans": "", "n_prop": ""},
        assumptions=modes.assumptions + [f"Film indices seen along (normal, transverse, propagation) for a {cut}-cut, {propagation}-propagating crystal: {film}"],
    )
