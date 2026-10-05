"""Interference (speckle / etalon) noise of a multipass cell from its discrete optical paths.

Semiclassical random-phase model, a prediction tool to be validated against full-wave numerics:

 * A PATH is a distinct facet sequence (itinerary) from the input port to one OUTPUT MODE CELL. Output mode cells
   have width d(sin χ) = λ / (n_eff w) in the exit direction (w = port width): one cell per port mode. Rays of one
   path form a single diffraction-limited bundle; different paths interfere with unrelated phases.
 * Path j of a cell has power P_j and length L_j (power-weighted mean of its rays). In one cell
        I(ν) = | Σ_j sqrt(P_j) exp(i φ_j(ν)) |²,     φ_j = 2π ν n_g L_j / c + const_j
   Random-phase contrast² of one cell = 1 - Σ P_j² / (Σ P_j)²  (a single path gives no speckle).
 * A Lorentzian source of FWHM Δν_L multiplies every pair term by exp(-2π Δν_L n_g |L_j - L_k| / c):
        Var_c = Σ_{j≠k} P_j P_k exp(-2π Δν_L n_g |L_j - L_k| / c)
   A detector summing all cells adds variances and means: contrast² = Σ_c Var_c / (Σ_c P_c)².
 * Spectral autocovariance g(Δν) = Σ_{j≠k} P_j P_k cos(2π Δν n_g (L_j - L_k) / c) / Σ_{j≠k} P_j P_k.
   A regular cell shows revivals at its free spectral range; a mixing cell a single decay.
 * A temperature step ΔT shifts all phases by k0 (dn_eff/dT) ΔT L_j, equivalent to a frequency shift
   Δν_eq = ν0 (dn_eff/dT) ΔT / n_g, so the pattern decorrelates as g(Δν_eq).

Reflection phases are neglected. SI units (Hz for frequencies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..billiard_cell.engine import RayTable, evaluate
from ..common import Result, require_nonnegative, require_positive

C = 299792458.0
_PAIR_BLOCK = 1024           # rows per block in the O(n²) pair sums
_WORK = 4_000_000            # elements per temporary array in autocovariance


@dataclass
class Paths:
    cells: list                 # [(P_j array, L_j array)] per output mode cell
    lam: float
    neff: float
    n_g: float
    port_w: float
    T_det: float
    Gamma: float
    n_rays_used: int
    L_all: np.ndarray = field(default=None, repr=False)   # every detected ray: path [m]
    W_all: np.ndarray = field(default=None, repr=False)   # and power weight

    @property
    def P_tot(self):
        return float(sum(P.sum() for P, _ in self.cells))

    @property
    def L_mean(self):
        return float(self.W_all @ self.L_all / self.W_all.sum())

    def n_paths_eff(self):
        """(power-weighted mean over cells of (Σ P)² / Σ P², number of cells)."""
        w = np.array([P.sum() for P, _ in self.cells])
        n = np.array([P.sum() ** 2 / (P**2).sum() for P, _ in self.cells])
        return float((w * n).sum() / w.sum()), len(self.cells)

    def M_spat(self):
        """Effective number of output mode cells, (Σ P_c)² / Σ P_c²."""
        w = np.array([P.sum() for P, _ in self.cells])
        return float(w.sum() ** 2 / (w**2).sum())


def prepare(table: RayTable, R_of_sinchi, alpha_bg, Gamma, lam, neff, n_g, port_w=None, min_cell_power=0.0) -> Paths:
    """Group the detected rays of a RayTable (traced with itineraries) into paths per output mode cell.
    Paths whose weight underflows to zero are dropped; so are cells carrying less than `min_cell_power` of the
    detected power (a fraction, 0 keeps all)."""
    port_w = table.port_w if port_w is None else port_w
    res = evaluate(table, R_of_sinchi, alpha_bg, Gamma)
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    if sel.size == 0:
        raise ValueError("no ray reached the output port")
    s_exit = table.sinchi[sel, table.exit_idx[sel]].astype(float)
    cid = np.floor(s_exit / (lam / (neff * port_w))).astype(np.int64)
    W, L = res.W, res.L.astype(float)
    key = np.stack([cid, table.itin[sel].view(np.int64)], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Pj = np.bincount(inv, weights=W)
    ok = Pj > 0
    Lj = np.zeros_like(Pj)
    Lj[ok] = np.bincount(inv, weights=W * L)[ok] / Pj[ok]
    cj = np.zeros(Pj.size, np.int64)
    cj[inv] = cid
    cells = [(Pj[ok & (cj == c)], Lj[ok & (cj == c)]) for c in np.unique(cj[ok])]
    tot = sum(P.sum() for P, _ in cells)
    cells = [(P, Lc) for P, Lc in cells if P.sum() >= min_cell_power * tot]
    return Paths(cells, lam, neff, n_g, port_w, res.T_det, Gamma, sel.size, L, W)


def _pairs(P, L):
    """Blocks of (2 P_j P_k, |L_j - L_k|) over j < k (each unordered pair once, weight doubled)."""
    n = P.size
    for a in range(0, n, _PAIR_BLOCK):
        i = np.arange(a, min(a + _PAIR_BLOCK, n))
        upper = np.arange(n)[None, :] > i[:, None]
        yield (2 * P[i, None] * P[None, :])[upper], np.abs(L[i, None] - L[None, :])[upper]


def _pair_sum(P, L, kernel):
    """Σ_{j≠k} P_j P_k kernel(|L_j - L_k|)."""
    return float(sum((w * kernel(d)).sum() for w, d in _pairs(P, L))) if P.size > 1 else 0.0


def contrast(paths: Paths, laser_fwhm=0.0, spatial=True):
    """Speckle contrast σ_I/⟨I⟩ for a Lorentzian source of FWHM laser_fwhm [Hz]: spatial=True, a detector that
    sums all mode cells; False, one mode cell (contrast² averaged over cells with weights P_c²)."""
    kern = lambda dL: np.exp(-2 * np.pi * laser_fwhm * paths.n_g * dL / C)
    var = sum(_pair_sum(P, L, kern) for P, L in paths.cells)
    cellP = np.array([P.sum() for P, _ in paths.cells])
    if spatial:
        return float(np.sqrt(var) / cellP.sum())
    return float(np.sqrt(var / (cellP**2).sum()))


def autocovariance(paths: Paths, dnu):
    """Normalised spectral autocovariance g(Δν) of the speckle (g(0) = 1), pooled over cells."""
    dnu = np.asarray(dnu, float)
    num = np.zeros_like(dnu)
    den = 0.0
    for P, L in paths.cells:
        if P.size < 2:
            continue
        for w, d in _pairs(P, L):
            step = max(1, _WORK // max(w.size, 1))
            for a in range(0, dnu.size, step):
                sl = slice(a, a + step)
                num[sl] += (w[None, :] * np.cos(2 * np.pi * dnu[sl, None] * paths.n_g * d[None, :] / C)).sum(1)
            den += w.sum()
    return num / den if den > 0 else np.zeros_like(dnu)


def correlation_width(dnu, g):
    """FWHM of g [Hz]: twice the smallest Δν with g < 0.5 (inf if g never drops below 0.5 on the grid)."""
    k = np.nonzero(np.asarray(g) < 0.5)[0]
    return 2 * float(dnu[k[0]]) if k.size else np.inf


def thermal_shift_hz(dT, nu0, dneff_dT, n_g):
    """Frequency shift equivalent to a temperature step: ν0 (dn_eff/dT) ΔT / n_g."""
    return nu0 * dneff_dT * np.asarray(dT) / n_g


def speckle_noise_A(paths: Paths, dnu, g, dT, nu0, dneff_dT, laser_fwhm, line_fwhm_hz, spatial=True):
    """Absorbance noise from the speckle pattern changing between reference and measurement (step dT [K]):
        σ_A = contrast(laser) · sqrt(2 (1 - ρ(dT))) · sqrt(min(1, FWHM(g) / line FWHM)).
    The last factor: a fitted line of width line_fwhm averages ~line_fwhm / FWHM(g) independent speckle grains.
    (dnu, g) is a precomputed autocovariance on a grid starting at 0."""
    C0 = contrast(paths, laser_fwhm, spatial)
    rho = np.interp(thermal_shift_hz(dT, nu0, dneff_dT, paths.n_g), dnu, g)
    fit = np.sqrt(min(1.0, correlation_width(dnu, g) / line_fwhm_hz))
    return C0 * np.sqrt(max(2 * (1 - rho), 0.0)) * fit


def simulate_spectrum(paths: Paths, dnu_grid, nu0, alpha_cm=None, which="all", dT=0.0, dneff_dT=1.5e-4, seed=0):
    """Coherent output intensity (mean 1 without gas) vs frequency offset [Hz] from nu0: one random-phase
    realisation. alpha_cm: gas absorption [1/cm] on dnu_grid (field decays as exp(-Γ α L / 2)).
    which='all' sums all mode cells; 'single' takes the cell with the most paths."""
    rng = np.random.default_rng(seed)
    k0 = 2 * np.pi * nu0 / C
    cells = paths.cells if which == "all" else [max(paths.cells, key=lambda pl: pl[0].size)]
    dnu_grid = np.asarray(dnu_grid, float)
    I = np.zeros_like(dnu_grid)
    norm = 0.0
    for P, L in cells:
        ph = rng.uniform(0, 2 * np.pi, P.size)[None, :] + k0 * dneff_dT * dT * L[None, :] + 2 * np.pi * np.outer(dnu_grid, paths.n_g * L / C)
        amp = np.sqrt(P)[None, :]
        if alpha_cm is not None:
            amp = amp * np.exp(-0.5 * paths.Gamma * np.asarray(alpha_cm)[:, None] * 100.0 * L[None, :])
        I += np.abs((amp * np.exp(1j * ph)).sum(1)) ** 2
        norm += P.sum()
    return I / norm


# ---------------------------------------------------------------- Result front ends
def thermal_shift(wavelength=1.5317e-6, dneff_dT=1.5e-4, delta_T=0.01, n_group=3.5) -> Result:
    """Frequency shift of the interference pattern equivalent to a temperature step ΔT."""
    require_positive(wavelength=wavelength, n_group=n_group)
    nu0 = C / wavelength
    return Result(
        values={"delta_nu": float(thermal_shift_hz(delta_T, nu0, dneff_dT, n_group)), "nu0": nu0},
        units={"delta_nu": "Hz", "nu0": "Hz"},
        assumptions=["Uniform temperature, thermo-optic shift of n_eff only (thermal expansion neglected)"],
    )


def pair_coherence(laser_fwhm=1e6, path_difference=0.1, n_group=3.5) -> Result:
    """Interference visibility factor of two paths for a Lorentzian source, and the coherence length."""
    require_nonnegative(laser_fwhm=laser_fwhm, path_difference=path_difference)
    require_positive(n_group=n_group)
    vis = float(np.exp(-2 * np.pi * laser_fwhm * n_group * path_difference / C))
    Lc = C / (np.pi * laser_fwhm * n_group) if laser_fwhm > 0 else np.inf
    return Result(
        values={"visibility": vis, "coherence_length": Lc},
        units={"visibility": "", "coherence_length": "m"},
        assumptions=["Lorentzian line shape (white frequency noise)", "Lengths are geometric; n_group converts to optical delay",
                     "visibility = exp(-2π Δν n_g ΔL / c) = exp(-2 ΔL / L_c)"],
    )


def random_phase_contrast(n_paths=10.0, n_modes=1.0) -> Result:
    """Speckle contrast of n_paths equal-power paths with random phases in each of n_modes equal mode cells summed
    on one detector (fully coherent source)."""
    require_positive(n_paths=n_paths, n_modes=n_modes)
    if n_paths < 1 or n_modes < 1:
        raise ValueError("n_paths and n_modes must be >= 1")
    c1 = float(np.sqrt(1 - 1 / n_paths))
    return Result(
        values={"contrast_single": c1, "contrast_sum": c1 / float(np.sqrt(n_modes))},
        units={"contrast_single": "", "contrast_sum": ""},
        assumptions=["Equal path powers, uniformly random independent phases", "Monochromatic source", "Independent, equal mode cells"],
    )
