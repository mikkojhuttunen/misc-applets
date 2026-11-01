"""Wafer-scale thickness spread: many devices, each with its own film (and deposited-layer) thickness.

Model: the thickness offset is constant along one device (the device is short compared with the wafer-scale
variation) but differs from device to device. Default: thin-film LN/LT and deposited layers (SiNx, SiC, ...)
spread uniformly over ±5 nm, independently.

`ThicknessFamily` solves the mode model (SHGModel or OPAModel, via a factory) on a small stencil of thickness
offsets and interpolates Δk(x; δt_film, δt_load) and the coupling with Lagrange polynomials, so hundreds of
devices cost a handful of mode solves. `shg_devices` / `opa_devices` apply one poling mask to every device and
return per-device spectra and summary statistics (peak shift, value at the target, band mean, yield).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import fast
from .metrics import fwhm
from .poling import Poling
from .waveguide import Ridge, with_thickness


@dataclass(frozen=True)
class ThicknessSpread:
    film: float = 5e-9          # half-range for "uniform", σ for "gaussian"
    load: float = 5e-9
    dist: str = "uniform"

    def __post_init__(self):
        if self.dist not in ("uniform", "gaussian"):
            raise ValueError("dist must be 'uniform' or 'gaussian'")

    def span(self, which) -> float:
        v = self.film if which == "film" else self.load
        return v if self.dist == "uniform" else 3 * v

    def sample(self, n, rng=None, with_load=True):
        rng = np.random.default_rng(rng)
        draw = (lambda s: rng.uniform(-s, s, n)) if self.dist == "uniform" else (lambda s: rng.normal(0.0, s, n))
        df = draw(self.film) if self.film > 0 else np.zeros(n)
        dl = draw(self.load) if (with_load and self.load > 0) else np.zeros(n)
        return df, dl


def lagrange_weights(nodes, t) -> np.ndarray:
    """Weights (len(t), len(nodes)) of the interpolating polynomial through `nodes`."""
    nodes = np.asarray(nodes, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    w = np.ones((t.size, nodes.size))
    for j, xj in enumerate(nodes):
        for m, xm in enumerate(nodes):
            if m != j:
                w[:, j] *= (t - xm) / (xj - xm)
    return w


class ThicknessFamily:
    """Mode models on film offsets (and load offsets when the geometry has a deposited strip)."""

    def __init__(self, factory: Callable[[Ridge], object], geom: Ridge, spread: ThicknessSpread = ThicknessSpread(),
                 film_points=5, load_points=3, etch="depth"):
        self.geom, self.spread = geom, spread
        self.film_nodes = np.linspace(-1, 1, film_points) * spread.span("film") if spread.film > 0 else np.zeros(1)
        has_load = geom.load is not None and spread.load > 0
        self.load_nodes = np.linspace(-1, 1, load_points) * spread.span("load") if has_load else np.zeros(1)
        self.has_load = has_load
        self.models = [[factory(with_thickness(geom, df, dl, etch)) for dl in self.load_nodes] for df in self.film_nodes]

    @property
    def nominal(self):
        i = int(np.argmin(np.abs(self.film_nodes)))
        j = int(np.argmin(np.abs(self.load_nodes)))
        return self.models[i][j]

    def _weights(self, d_film, d_load):
        wf = lagrange_weights(self.film_nodes, d_film) if self.film_nodes.size > 1 else np.ones((np.size(d_film), 1))
        wl = lagrange_weights(self.load_nodes, d_load) if self.load_nodes.size > 1 else np.ones((np.size(d_load), 1))
        return wf, wl

    def evaluate(self, fn: Callable[[object], np.ndarray], d_film, d_load=None) -> np.ndarray:
        """Interpolate fn(model) (array over x) to every device: shape (n_devices, *fn.shape)."""
        d_film = np.atleast_1d(np.asarray(d_film, dtype=float))
        d_load = np.zeros_like(d_film) if d_load is None else np.atleast_1d(np.asarray(d_load, dtype=float))
        vals = np.array([[np.asarray(fn(m)) for m in row] for row in self.models])     # (nf, nl, ...)
        wf, wl = self._weights(d_film, d_load)
        return np.einsum("di,dj,ij...->d...", wf, wl, vals)

    def delta_k(self, x, d_film, d_load=None) -> np.ndarray:
        return self.evaluate(lambda m: m(x), d_film, d_load)

    def sensitivity(self, x0, length=None, h=0.5e-9) -> dict:
        """Analytic thickness sensitivity at x0 (pump λ for SHG, signal λ for OPA), from the interpolant:
        ∂Δk/∂t, ∂Δk/∂x and the phase-matching shift dx/dt = -(∂Δk/∂t)/(∂Δk/∂x) (inf where ∂Δk/∂x ≈ 0,
        e.g. at OPA degeneracy). With `length`, the thickness tolerance of a uniform grating:
        |∂Δk/∂t| δt ≤ 2·1.39/L keeps the device inside the sinc² FWHM at x0."""
        nom = self.nominal
        dx = x0 * 1e-5
        dkdx = float((nom(x0 + dx) - nom(x0 - dx)) / (2 * dx))
        out = {"ddk_dx": dkdx}
        terms = [("film", lambda d: self.delta_k(x0, [d], [0.0])[0])]
        if self.has_load:
            terms.append(("load", lambda d: self.delta_k(x0, [0.0], [d])[0]))
        for name, f in terms:
            dkdt = float((f(h) - f(-h)) / (2 * h))
            out[f"ddk_d{name}"] = dkdt
            shift = -dkdt / dkdx if dkdx != 0 else float("inf")
            out[f"dpeak_d{name}"] = shift if abs(shift) < 1e3 else float("inf")      # degeneracy: no first-order shift
            if length is not None:
                out[f"tolerance_{name}"] = 2 * 1.391557 / (length * abs(dkdt))
        return out


def _peak(x, y):
    i = np.argmax(y, axis=-1)
    return np.take(x, i)


def shg_devices(family: ThicknessFamily, poling: Poling, lam, lambda0, n_devices=200, seed=0, band=None,
                backend="numpy") -> dict:
    """SHG spectra η(λ) = κ²|G|² (1/W) of one mask on n_devices thickness samples (undepleted)."""
    lam = np.asarray(lam, dtype=float)
    df, dl = family.spread.sample(n_devices, seed, with_load=family.has_load)
    dk = family.delta_k(lam, df, dl)                                    # (D, M)
    kap = family.evaluate(lambda m: np.array(float(m.coupling(lambda0)["kappa"])), df, dl)
    eta = np.empty_like(dk)
    for d in range(n_devices):
        eta[d] = kap[d] ** 2 * np.abs(fast.spectrum(poling, np.abs(dk[d]), backend=backend)[0]) ** 2
    nom_k = float(family.nominal.coupling(lambda0)["kappa"])
    nom = nom_k**2 * np.abs(fast.spectrum(poling, np.abs(family.nominal(lam)), backend=backend)[0]) ** 2
    i0 = int(np.argmin(np.abs(lam - lambda0)))
    out = {"d_film": df, "d_load": dl, "lam": lam, "eta": eta, "nominal": nom,
           "peak": _peak(lam, eta), "nominal_peak": float(lam[np.argmax(nom)]),
           "at_target": eta[:, i0], "nominal_at_target": float(nom[i0]),
           "fwhm": np.array([fwhm(lam, e) for e in eta])}
    if band is not None:
        sel = (lam >= band[0]) & (lam <= band[1])
        out["band_mean"] = eta[:, sel].mean(axis=1)
        out["nominal_band_mean"] = float(nom[sel].mean())
    out.update(_sensitivity(out["peak"] - out["nominal_peak"], df, dl, family.has_load))
    return out


def opa_devices(family: ThicknessFamily, poling: Poling, signals, pump_power, n_devices=100, seed=0, band=None,
                alpha_signal=0.0, alpha_idler=0.0) -> dict:
    """Undepleted OPA signal gain spectra of one mask on n_devices thickness samples."""
    ls = np.asarray(signals, dtype=float)
    df, dl = family.spread.sample(n_devices, seed, with_load=family.has_load)
    dk = family.delta_k(ls, df, dl)                                     # (D, M)
    nominal = family.nominal
    gain = nominal.gain_spectrum(poling.boundaries, poling.g, ls, pump_power, alpha_signal, alpha_idler, delta_k=dk)
    nom = nominal.gain_spectrum(poling.boundaries, poling.g, ls, pump_power, alpha_signal, alpha_idler)
    gdb, ndb = 10 * np.log10(gain), 10 * np.log10(nom)
    out = {"d_film": df, "d_load": dl, "signal": ls, "gain_db": gdb, "nominal_db": ndb,
           "peak": _peak(ls, gdb), "nominal_peak": float(ls[np.argmax(ndb)]), "peak_db": gdb.max(axis=1)}
    if band is not None:
        sel = (ls >= band[0]) & (ls <= band[1])
        out["band_min_db"] = gdb[:, sel].min(axis=1)
        out["nominal_band_min_db"] = float(ndb[sel].min())
    out.update(_sensitivity(out["peak"] - out["nominal_peak"], df, dl, family.has_load))
    return out


def _sensitivity(shift, df, dl, has_load):
    """Least-squares slope of the observed peak shift against the thickness offsets (m per m). Biased when
    peaks leave the scan window or the band vanishes; ThicknessFamily.sensitivity gives the analytic value."""
    cols = [df] + ([dl] if has_load else []) + [np.ones_like(df)]
    a = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(a, shift, rcond=None)
    out = {"dpeak_dfilm": float(coef[0]), "peak_shift_rms": float(np.sqrt(np.mean(shift**2)))}
    if has_load:
        out["dpeak_dload"] = float(coef[1])
    return out


def yield_fraction(values, nominal, rel=0.5) -> float:
    """Fraction of devices whose value reaches rel × nominal (efficiency, or gain in dB)."""
    return float(np.mean(np.asarray(values) >= rel * nominal))
