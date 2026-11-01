"""Waveguide layer: ridge/rib cross-section -> n_eff(λ), n_g, mode fields, SHG overlap A_eff, cached tables.

Frame: x lateral, y vertical (y = 0 at the bottom of the film), z propagation. A `Ridge` is a thin film
(thickness `film`) on a substrate, etched by `etch` (etch = film: isolated ridge) to a trapezoid with top
width `width` and sidewall angle `sidewall` (degrees from the substrate plane, 90 = vertical), under a
cladding (air or SiO2). The film is anisotropic (materials.film_tensor): TE (E along x) sees ε_xx = n_lat²,
TM (E along y) sees ε_yy = n_vert².

Solvers (same inputs):
  eim(...)          effective index method on the math-engines slab engines (fast, approximate)
  solve(...)        2D semi-vectorial finite differences, ported from waveguide-core.js (fdSolve): non-uniform
                    graded grid, interface-weighted stencil for the field component normal to x (TE) or y (TM),
                    Dirichlet box; eigenpairs by sparse shift-invert (scipy) at k0² ε_max. Cells cut by an
                    interface get area-weighted ε (sub-pixel supersampling), which also handles slanted sidewalls.
`SHGModel` tabulates n_p(λ), n_SH(λ/2), group indices and A_eff on a λ grid (one FD grid for all λ) and is a
Δk(λ) callable for chi2sim.synthesis; kappa(λ) uses Miller-scaled d33 (materials.miller_d).

Semi-vectorial: TE/TM hybridisation (mode crossings, strongly slanted walls) is not modelled.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

from engines.anisotropic_slab import engine as aslab
from engines.cwe_shg import engine as cwe
from engines.slab_waveguide import engine as slab

from . import materials


@dataclass(frozen=True)
class Ridge:
    film: float = 0.6e-6
    etch: float = 0.3e-6
    width: float = 1.4e-6
    sidewall: float = 70.0
    material: str = "ln"
    cut: str = "x"
    clad: str = "air"
    substrate: str = "sio2"

    def __post_init__(self):
        if not (0 <= self.etch <= self.film and self.film > 0 and self.width > 0):
            raise ValueError("need film > 0, 0 <= etch <= film, width > 0")
        if not (10 <= self.sidewall <= 90):
            raise ValueError("sidewall angle must lie in [10, 90] degrees")
        if self.material not in materials.CRYSTALS or self.cut not in materials.CUTS:
            raise ValueError("unknown material or cut")

    @property
    def slab(self) -> float:
        return self.film - self.etch

    @property
    def spread(self) -> float:
        """Lateral run of one sidewall."""
        return self.etch / math.tan(math.radians(self.sidewall)) if self.sidewall < 90 else 0.0

    def in_film(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        slab_part = (y >= 0) & (y < self.slab)
        half = self.width / 2 + (self.film - y) / math.tan(math.radians(self.sidewall)) if self.sidewall < 90 else self.width / 2
        ridge = (y >= self.slab) & (y < self.film) & (np.abs(x) < half)
        return slab_part | ridge


# ----------------------------------------------------------------------------- indices

def indices(geom: Ridge, wavelength) -> dict:
    lat, vert, prop = materials.film_tensor(geom.material, geom.cut, wavelength)
    return {"film": (lat, vert, prop), "sub": materials.isotropic(geom.substrate, wavelength),
            "clad": materials.isotropic(geom.clad, wavelength)}


def _slab_neff(ix, thickness, wavelength, pol):
    if thickness <= 0:
        return float("nan")
    lat, vert, prop = ix["film"]
    r = aslab.slab_modes(wavelength, [ix["sub"], (vert, lat, prop), ix["clad"]], [thickness], pol)
    return float(r["neff"][0]) if r["n_modes"] else float("nan")


def eim(geom: Ridge, wavelength, polarization="TE") -> float:
    """Effective index method: vertical anisotropic slab in and outside the ridge, then a lateral slab of
    the mid-height width with the complementary polarisation."""
    ix = indices(geom, wavelength)
    n_in = _slab_neff(ix, geom.film, wavelength, polarization)
    n_out = _slab_neff(ix, geom.slab, wavelength, polarization)
    floor = max(ix["sub"], ix["clad"])
    if not np.isfinite(n_out):
        n_out = floor
    if not np.isfinite(n_in) or n_in <= n_out:
        return float("nan")
    w = geom.width + geom.spread
    lat_pol = "TM" if polarization == "TE" else "TE"
    return float(slab.neff_three_layer(wavelength, n_out, n_in, n_out, w, lat_pol, 0))


# ----------------------------------------------------------------------------- grid

@dataclass
class Grid:
    xe: np.ndarray
    ye: np.ndarray
    f_film: np.ndarray        # (Ny, Nx) film area fraction per cell
    f_sub: np.ndarray         # (Ny, Nx) substrate fraction

    @property
    def xc(self):
        return 0.5 * (self.xe[1:] + self.xe[:-1])

    @property
    def yc(self):
        return 0.5 * (self.ye[1:] + self.ye[:-1])

    @property
    def dx(self):
        return np.diff(self.xe)

    @property
    def dy(self):
        return np.diff(self.ye)

    @property
    def area(self):
        return np.outer(self.dy, self.dx)

    @property
    def shape(self):
        return self.f_film.shape


def _axis(core_lo, core_hi, cuts, h, lo, hi, hmax, ratio):
    pts = sorted({core_lo, core_hi, *[c for c in cuts if core_lo < c < core_hi]})
    edges = [core_lo]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, math.ceil((b - a) / h - 1e-9))
        edges.extend(np.linspace(a, b, n + 1)[1:])

    def grade(start, end, sign):
        out, z, step = [], start, h
        while sign * (end - z) > 1e-15:
            step = min(step * ratio, hmax)
            z = z + sign * step
            if sign * (z - end) > -0.5 * step:
                z = end
            out.append(z)
        return out

    left = grade(core_lo, lo, -1)[::-1]
    right = grade(core_hi, hi, 1)
    return np.array(left + edges + right)


def build_grid(geom: Ridge, wavelength_max, h=20e-9, ratio=1.15, lateral_margin=1.6, sub_margin=1.3,
               clad_margin=1.0, supersample=6) -> Grid:
    """Graded grid: uniform step h over the film and sidewalls (plus 0.2 µm), growing by `ratio` up to
    λ/4 outside; margins in units of the longest wavelength. Area fractions by s×s supersampling."""
    hx, hy = (h, h) if np.ndim(h) == 0 else h
    half = geom.width / 2 + geom.spread
    core_x = half + 0.2e-6
    lam = wavelength_max
    xe = _axis(-core_x, core_x, [-half, -geom.width / 2, geom.width / 2, half], hx,
               -core_x - lateral_margin * lam, core_x + lateral_margin * lam, lam / 4, ratio)
    ye = _axis(-0.1e-6, geom.film + 0.1e-6, [0.0, geom.slab, geom.film], hy,
               -sub_margin * lam, geom.film + clad_margin * lam, lam / 4, ratio)
    s = supersample
    u = (np.arange(s) + 0.5) / s
    xs = (xe[:-1, None] + np.diff(xe)[:, None] * u[None, :]).ravel()
    ys = (ye[:-1, None] + np.diff(ye)[:, None] * u[None, :]).ravel()
    X, Y = np.meshgrid(xs, ys)
    fil = geom.in_film(X, Y).astype(float)
    sub = (Y < 0).astype(float)
    ny, nx = ye.size - 1, xe.size - 1
    f_film = fil.reshape(ny, s, nx, s).mean(axis=(1, 3))
    f_sub = sub.reshape(ny, s, nx, s).mean(axis=(1, 3))
    return Grid(xe, ye, f_film, f_sub)


def _eps(grid: Grid, ix, component):
    n_f = ix["film"][0 if component == "x" else 1]
    f_clad = 1 - grid.f_film - grid.f_sub
    return grid.f_film * n_f**2 + grid.f_sub * ix["sub"] ** 2 + f_clad * ix["clad"] ** 2


# ----------------------------------------------------------------------------- FD solver

@dataclass
class Mode:
    neff: float
    polarization: str
    wavelength: float
    field: np.ndarray            # (Ny, Nx) dominant transverse E component, max = +1
    order: tuple = (0, 0)        # (vertical nodes m, lateral nodes n)
    film_fraction: float = 0.0   # share of ∫|E|² dA inside the film


def _operator(grid: Grid, eps, k0, pol):
    ny, nx = eps.shape
    dx, dy = grid.dx, grid.dy
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = k0**2 * eps.ravel().copy()

    def couple(axis, weighted):
        d = dx if axis == 1 else dy
        for s in (-1, 1):
            src = [slice(None), slice(None)]
            dst = [slice(None), slice(None)]
            if s == 1:
                src[axis], dst[axis] = slice(0, -1), slice(1, None)
            else:
                src[axis], dst[axis] = slice(1, None), slice(0, -1)
            src, dst = tuple(src), tuple(dst)
            di = d[src[axis]] if axis == 1 else d[src[axis]][:, None]
            dj = d[dst[axis]] if axis == 1 else d[dst[axis]][:, None]
            hh = 0.5 * (di + dj)
            cf = np.broadcast_to(1.0 / (hh * di), eps[src].shape)
            ec, en = eps[src], eps[dst]
            if weighted:
                a = 2.0 / (ec + en)
                off, dg = cf * a * en, -cf * a * ec
            else:
                off, dg = cf, -cf
            rows.append(idx[src].ravel())
            cols.append(idx[dst].ravel())
            vals.append(off.ravel())
            np.add.at(diag, idx[src].ravel(), dg.ravel())
            # Dirichlet ghost beyond the box edge
            edge = [slice(None), slice(None)]
            edge[axis] = slice(-1, None) if s == 1 else slice(0, 1)
            de = d[-1] if s == 1 else d[0]
            np.add.at(diag, idx[tuple(edge)].ravel(), -1.0 / de**2)

    couple(1, pol == "TE")
    couple(0, pol == "TM")
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _nodes(v):
    m = np.abs(v).max()
    s = np.sign(v[np.abs(v) >= 0.08 * m])
    return int(np.count_nonzero(s[1:] != s[:-1]))


def solve(geom: Ridge, wavelength, polarization="TE", grid: Grid | None = None, n_modes=4, h=20e-9, n_min=None) -> list[Mode]:
    """Guided modes (n_eff above the substrate/cladding and outer-slab limits, or above n_min), descending n_eff."""
    if polarization not in ("TE", "TM"):
        raise ValueError("polarization must be TE or TM")
    grid = grid or build_grid(geom, wavelength, h)
    ix = indices(geom, wavelength)
    eps = _eps(grid, ix, "x" if polarization == "TE" else "y")
    k0 = 2 * math.pi / wavelength
    A = _operator(grid, eps, k0, polarization)
    sigma = k0**2 * eps.max() * 1.0001
    k = min(n_modes + 2, A.shape[0] - 2)
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM")
    floor = max(ix["sub"], ix["clad"])
    lim = max(floor, np.nan_to_num(_slab_neff(ix, geom.slab, wavelength, polarization), nan=0.0)) if n_min is None else n_min
    out = []
    for lam2, v in sorted(zip(vals, vecs.T), key=lambda t: -t[0].real):
        if abs(lam2.imag) > 1e-8 * abs(lam2.real) or lam2.real <= 0:
            continue
        ne = math.sqrt(lam2.real) / k0
        if ne <= lim + 1e-6:
            continue
        f = np.real(v).reshape(grid.shape)
        if np.abs(np.imag(v)).max() > np.abs(f).max():
            f = np.imag(v).reshape(grid.shape)
        f = f / f.flat[np.argmax(np.abs(f))]
        j, i = np.unravel_index(np.argmax(np.abs(f)), f.shape)
        p = f**2 * grid.area
        out.append(Mode(ne, polarization, wavelength, f, (_nodes(f[:, i]), _nodes(f[j, :])),
                        float((p * grid.f_film).sum() / p.sum())))
        if len(out) == n_modes:
            break
    return out


def fundamental(geom: Ridge, wavelength, polarization="TE", grid: Grid | None = None, h=20e-9) -> Mode:
    for m in solve(geom, wavelength, polarization, grid, n_modes=4, h=h):
        if m.order == (0, 0):
            return m
    raise RuntimeError(f"no guided fundamental {polarization} mode at {wavelength * 1e9:.1f} nm")


def a_eff_shg(pump: np.ndarray, sh: np.ndarray, grid: Grid) -> float:
    """A_eff = (∫|E_p|²)² ∫|E_SH|² / |∫_χ E_p² E_SH|², the overlap restricted to the nonlinear film."""
    dA = grid.area
    num = (np.sum(pump**2 * dA)) ** 2 * np.sum(sh**2 * dA)
    ov = np.sum(grid.f_film * pump**2 * sh * dA)
    return float(num / ov**2)


# ----------------------------------------------------------------------------- SHG table

@dataclass
class SHGModel:
    """n_p(λ), n_SH(λ/2), n_g and A_eff on a pump-λ grid, splined; callable as Δk(λ) = 4π(n_SH - n_p)/λ."""

    geom: Ridge
    lambdas: np.ndarray
    polarization: str = "TE"
    h: float = 20e-9
    n_p: np.ndarray = field(init=False)
    n_sh: np.ndarray = field(init=False)
    a_eff_tab: np.ndarray = field(init=False)
    film_fraction: np.ndarray = field(init=False)

    def __post_init__(self):
        lam = np.asarray(self.lambdas, dtype=float)
        if lam.size < 4:
            raise ValueError("need at least 4 wavelengths for the spline")
        self.lambdas = lam
        self.grid = build_grid(self.geom, lam.max(), self.h)
        n_p, n_s, ae, ff = [], [], [], []
        for l in lam:
            mp = fundamental(self.geom, l, self.polarization, self.grid)
            ms = fundamental(self.geom, l / 2, self.polarization, self.grid)
            n_p.append(mp.neff)
            n_s.append(ms.neff)
            ae.append(a_eff_shg(mp.field, ms.field, self.grid))
            ff.append((mp.film_fraction, ms.film_fraction))
        self.n_p, self.n_sh, self.a_eff_tab = np.array(n_p), np.array(n_s), np.array(ae)
        self.film_fraction = np.array(ff)
        self._splines()

    def _splines(self):
        lam = self.lambdas
        self._sp = CubicSpline(lam, self.n_p)
        self._ss = CubicSpline(lam, self.n_sh)
        self._sa = CubicSpline(lam, self.a_eff_tab)

    def _check(self, lam):
        lam = np.asarray(lam, dtype=float)
        if np.any(lam < self.lambdas.min() * (1 - 1e-9)) or np.any(lam > self.lambdas.max() * (1 + 1e-9)):
            raise ValueError("wavelength outside the tabulated range")
        return lam

    def neff_pump(self, lam):
        return self._sp(self._check(lam))

    def neff_sh(self, lam):
        return self._ss(self._check(lam))

    def ng_pump(self, lam):
        lam = self._check(lam)
        return self._sp(lam) - lam * self._sp(lam, 1)

    def ng_sh(self, lam):
        """SH group index at λ/2: n - (λ/2) dn/d(λ/2) = n - λ dn/dλ for n tabulated against pump λ."""
        lam = self._check(lam)
        return self._ss(lam) - lam * self._ss(lam, 1)

    def a_eff(self, lam):
        return self._sa(self._check(lam))

    def __call__(self, lam):
        lam = self._check(lam)
        return 4 * np.pi * (self._ss(lam) - self._sp(lam)) / lam

    def d_coeff(self, lam):
        return materials.miller_d(self.geom.material, self.geom.cut, self.polarization, float(lam))

    def coupling(self, lam):
        """cwe_shg.coupling_shg Result (κ, η0) at pump λ with Miller-scaled d and the tabulated A_eff."""
        return cwe.coupling_shg(lam, float(self.neff_pump(lam)), float(self.neff_sh(lam)), self.d_coeff(lam),
                                float(self.a_eff(lam)))

    def period(self, lam):
        return 2 * np.pi / np.abs(self(lam))

    def summary(self, lam, length=None) -> dict:
        c = self.coupling(lam)
        out = {"period": float(self.period(lam)), "n_pump": float(self.neff_pump(lam)), "n_sh": float(self.neff_sh(lam)),
               "ng_pump": float(self.ng_pump(lam)), "ng_sh": float(self.ng_sh(lam)), "a_eff": float(self.a_eff(lam)),
               "d": self.d_coeff(lam), "eta_norm": float(c["eta_norm"]), "eta_qpm": float(c["eta_norm"]) * (2 / np.pi) ** 2}
        if length is not None:
            h = lam * 1e-4
            slope = (self(lam + h) - self(lam - h)) / (2 * h)
            out["fwhm"] = float(4 * 1.391557 / (length * abs(slope)))
        return out


    # ----------------------------------------------------------------- cache
    def save(self, path):
        """Store the table (not the grid) as .npz; `load` restores it without solving."""
        np.savez(path, lambdas=self.lambdas, n_p=self.n_p, n_sh=self.n_sh, a_eff=self.a_eff_tab,
                 film_fraction=self.film_fraction, polarization=self.polarization, h=self.h,
                 geom=np.array(list(asdict(self.geom).items()), dtype=object))

    @classmethod
    def load(cls, path) -> "SHGModel":
        z = np.load(path, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.geom = Ridge(**{k: (float(v) if k in _FLOATS else str(v)) for k, v in z["geom"]})
        obj.lambdas, obj.polarization, obj.h = z["lambdas"], str(z["polarization"]), float(z["h"])
        obj.n_p, obj.n_sh, obj.a_eff_tab, obj.film_fraction = z["n_p"], z["n_sh"], z["a_eff"], z["film_fraction"]
        obj.grid = None
        obj._splines()
        return obj

    @classmethod
    def cached(cls, geom: Ridge, lambdas, polarization="TE", h=20e-9, cache_dir=None) -> "SHGModel":
        """Load from cache_dir if this exact (geometry, λ grid, polarisation, h) was solved before, else solve and store."""
        lam = np.asarray(lambdas, dtype=float)
        if cache_dir is None:
            return cls(geom, lam, polarization, h)
        key = hashlib.sha1(repr((sorted(asdict(geom).items()), lam.round(15).tolist(), polarization, h)).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"shg_{key}.npz"
        if path.exists():
            return cls.load(path)
        m = cls(geom, lam, polarization, h)
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save(path)
        return m


_FLOATS = {"film", "etch", "width", "sidewall"}


def with_width(geom: Ridge, width) -> Ridge:
    return replace(geom, width=width)


def width_scan(geom: Ridge, widths, wavelength, polarization="TE", h=20e-9) -> dict:
    """Period, Δk and A_eff of SHG at one pump λ against top width (fabrication tolerance; later Δk(z)).
    One grid per width. dΔk/dw from a spline gives the width tolerance of a QPM grating:
    |δw| ≲ 2·1.39/(L |dΔk/dw|) keeps a uniform width error within the sinc² FWHM."""
    out = {"width": np.asarray(widths, dtype=float), "delta_k": [], "a_eff": [], "n_pump": [], "n_sh": []}
    for w in out["width"]:
        g = with_width(geom, w)
        gr = build_grid(g, wavelength, h)
        mp, ms = fundamental(g, wavelength, polarization, gr), fundamental(g, wavelength / 2, polarization, gr)
        out["n_pump"].append(mp.neff)
        out["n_sh"].append(ms.neff)
        out["delta_k"].append(4 * np.pi * (ms.neff - mp.neff) / wavelength)
        out["a_eff"].append(a_eff_shg(mp.field, ms.field, gr))
    for k in ("delta_k", "a_eff", "n_pump", "n_sh"):
        out[k] = np.array(out[k])
    out["period"] = 2 * np.pi / np.abs(out["delta_k"])
    if out["width"].size >= 3:
        out["ddk_dw"] = CubicSpline(out["width"], out["delta_k"])(out["width"], 1)
    return out
