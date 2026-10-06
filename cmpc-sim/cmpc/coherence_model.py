"""coherence_model — how much a multipass cell's interference (etalon / speckle) noise is reduced by
path multiplicity, path-length spread, mode averaging and laser linewidth. A PREDICTION model for the proposal
stage; the full-wave numerics that validate it are a work-package task.

Model (semiclassical, random-phase statistics on discrete paths):

 * A PATH is a distinct facet sequence (itinerary) from the input port to a given OUTPUT MODE CELL. Rays of one path form a
   single diffraction-limited bundle; different paths interfere with unrelated phases. Output mode cells are cells of
   width  d(sin chi) = lambda / (n_eff w)  in the exit direction (w = port width): each cell is one mode of the port.
 * Path j in cell c has power P_j and length L_j (power-weighted mean over its rays). Intensity in the cell is
        I_c(nu) = | sum_j sqrt(P_j) exp(i phi_j(nu)) |^2,      phi_j = 2 pi nu n_g L_j / c + const_j
 * Random-phase result: contrast^2 of one cell  =  1 - sum P_j^2 / (sum P_j)^2   (one path -> no speckle at all).
 * A source of Lorentzian linewidth dnu_L (FWHM) scales every pair term by exp(-2 pi dnu_L |L_j-L_k| n_g / c):
        Var_c = sum_{j!=k} P_j P_k exp(-2 pi dnu_L n_g |L_j - L_k| / c)
   A detector that sums all mode cells adds the variances and the means:  contrast^2 = sum_c Var_c / (sum_c P_c)^2.
 * Autocovariance of the spectral speckle:  g(dnu) = sum_{j!=k} P_j P_k cos(2 pi dnu n_g (L_j-L_k)/c) / sum_{j!=k} P_j P_k.
   A regular cell shows revivals of g at the free spectral range c/(n_g L_orbit); a mixing cell shows one decay.
 * Temperature drift d(neff)/dT shifts all phases by k0 (dneff/dT) dT L_j, equivalent to a frequency shift
   dnu_eq = nu0 (dneff/dT) dT / n_g, so the speckle pattern decorrelates as g(dnu_eq).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import cmpc_sim as cs

C = 299792458.0


@dataclass
class Paths:
    cells: list                 # list of (P_j array, L_j array) per output mode cell
    lam: float
    neff: float
    n_g: float
    port_w: float
    T_det: float
    Gamma: float
    n_rays_used: int
    L_all: np.ndarray = field(default=None, repr=False)
    W_all: np.ndarray = field(default=None, repr=False)
    cids: list = field(default=None, repr=False)

    @property
    def P_tot(self): return float(sum(P.sum() for P, _ in self.cells))

    @property
    def L_mean(self): return float(self.W_all @ self.L_all / self.W_all.sum())

    def n_paths_eff(self):
        """Power-weighted mean over cells of (sum P)^2 / sum P^2, and the number of cells."""
        w = np.array([P.sum() for P, _ in self.cells])
        n = np.array([P.sum() ** 2 / (P**2).sum() for P, _ in self.cells])
        return float((w * n).sum() / w.sum()), len(self.cells)

    def M_spat(self):
        w = np.array([P.sum() for P, _ in self.cells])
        return float(w.sum() ** 2 / (w**2).sum())


def prepare(table, R_of_sinchi, alpha_bg, Gamma, lam, neff, n_g, port_w=None, min_cell_power=0.0) -> Paths:
    port_w = table.port_w if port_w is None else port_w
    res = cs.evaluate(table, R_of_sinchi, alpha_bg, Gamma)
    sel = np.nonzero((table.exit_port == 1) & (table.exit_idx >= 0))[0]
    s_exit = table.sinchi[sel, table.exit_idx[sel]].astype(float)
    cid = np.floor(s_exit / (lam / (neff * port_w))).astype(np.int64)
    it = table.itin[sel]
    W, L = res.W, res.L.astype(float)
    key = np.stack([cid.astype(np.int64), it.view(np.int64)], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Pj = np.bincount(inv, weights=W)
    Lj = np.bincount(inv, weights=W * L) / Pj
    cj = np.zeros(Pj.size, np.int64)
    cj[inv] = cid
    cells, cids = [], []
    for c in np.unique(cj):
        m = cj == c
        cells.append((Pj[m], Lj[m])); cids.append(int(c))
    return Paths(cells, lam, neff, n_g, port_w, res.T_det, Gamma, sel.size, L, W, cids)


def merge_incoherent(plist):
    """K beams that do not interfere (separate lasers, or delays longer than the coherence length): intensities add,
    so every (beam, cell) pair is an independent mode. Detected powers add (T_det = mean over beams)."""
    p0 = plist[0]
    return Paths([c for p in plist for c in p.cells], p0.lam, p0.neff, p0.n_g, p0.port_w, float(np.mean([p.T_det for p in plist])),
                 p0.Gamma, sum(p.n_rays_used for p in plist), np.concatenate([p.L_all for p in plist]),
                 np.concatenate([p.W_all for p in plist]), [c for p in plist for c in (p.cids or [])])


def merge_coherent(plist):
    """K beams from ONE laser without delay: paths that exit into the same mode cell interfere, so the path sets are united."""
    p0 = plist[0]
    bucket = {}
    for p in plist:
        for (P, L), cid in zip(p.cells, p.cids):
            if cid in bucket:
                bucket[cid] = (np.concatenate([bucket[cid][0], P]), np.concatenate([bucket[cid][1], L]))
            else:
                bucket[cid] = (P, L)
    return Paths(list(bucket.values()), p0.lam, p0.neff, p0.n_g, p0.port_w, float(np.mean([p.T_det for p in plist])),
                 p0.Gamma, sum(p.n_rays_used for p in plist), np.concatenate([p.L_all for p in plist]),
                 np.concatenate([p.W_all for p in plist]), list(bucket.keys()))


def _pair_sum(P, L, kernel):
    """sum_{j != k} P_j P_k kernel(|L_j - L_k|)."""
    if P.size < 2:
        return 0.0
    dL = np.abs(L[:, None] - L[None, :])
    PP = P[:, None] * P[None, :]
    np.fill_diagonal(PP, 0.0)
    return float((PP * kernel(dL)).sum())


def contrast(paths: Paths, laser_fwhm=0.0, spatial=True, span_hz=0.0):
    """Speckle contrast sigma_I / <I> of (spatial=True) a detector summing all mode cells, or (False) one typical
    mode (power-weighted over cells), for a Lorentzian source of FWHM `laser_fwhm` [Hz].
    span_hz: phase dither / sweep averaged uniformly over an equivalent frequency span (a path-length or temperature
    modulation of strength dT is equivalent to span = nu (dneff/dT) dT / n_g). Pair factor sinc^2(span tau_jk)."""
    k = lambda dL: np.exp(-2 * np.pi * laser_fwhm * paths.n_g * dL / C) * np.sinc(span_hz * paths.n_g * dL / C) ** 2
    var = [_pair_sum(P, L, k) for P, L in paths.cells]
    cellP = np.array([P.sum() for P, _ in paths.cells])
    if spatial:
        return float(np.sqrt(sum(var)) / cellP.sum())
    return float(np.sqrt(sum(var) / (cellP**2).sum()))


def autocovariance(paths: Paths, dnu):
    """Normalised spectral autocovariance g(dnu) of the speckle (g(0) = 1), pooled over cells."""
    dnu = np.asarray(dnu, float)
    num = np.zeros_like(dnu)
    den = 0.0
    for P, L in paths.cells:
        if P.size < 2:
            continue
        dL = (L[:, None] - L[None, :])
        PP = P[:, None] * P[None, :]
        np.fill_diagonal(PP, 0.0)
        iu = np.triu_indices(P.size, 1)
        w = 2 * PP[iu]
        d = np.abs(dL[iu])
        for a in range(0, dnu.size, 400):                       # chunked to bound memory
            sl = slice(a, a + 400)
            num[sl] += (w[None, :] * np.cos(2 * np.pi * dnu[sl, None] * paths.n_g * d[None, :] / C)).sum(1)
        den += w.sum()
    return num / den if den > 0 else np.zeros_like(dnu)


def correlation_width(dnu, g):
    """FWHM of g (Hz): 2 x smallest dnu with g < 0.5."""
    k = np.nonzero(g < 0.5)[0]
    return 2 * dnu[k[0]] if k.size else np.inf


def thermal_shift_hz(dT, nu0, dneff_dT, n_g):
    return nu0 * dneff_dT * dT / n_g


def speckle_noise_A(paths: Paths, dnu, g, dT, nu0, dneff_dT, laser_fwhm, line_fwhm_hz, spatial=True, span_hz=0.0):
    """Absorbance noise from the speckle pattern changing between reference and measurement (temperature step dT [K]):
        sigma_A = contrast(laser) * sqrt(2 (1 - rho(dT))) * sqrt(min(1, FWHM(g) / line FWHM)).
    The last factor: a fitted line of width line_fwhm averages ~line_fwhm/FWHM(g) independent speckle grains."""
    C0 = contrast(paths, laser_fwhm, spatial, span_hz)
    rho = np.interp(thermal_shift_hz(dT, nu0, dneff_dT, paths.n_g), dnu, g)
    fit = np.sqrt(min(1.0, correlation_width(dnu, g) / line_fwhm_hz))
    return C0 * np.sqrt(max(2 * (1 - rho), 0.0)) * fit


def simulate_spectrum(paths: Paths, dnu_grid, nu0, alpha_cm=None, which="all", dT=0.0, dneff_dT=1.5e-4, seed=0):
    """Coherent intensity at the output (mean 1 without gas) vs frequency offset [Hz] from nu0: one random-phase
    realisation of the path model. which='all' sums all mode cells; 'single' uses the cell with the most paths."""
    rng = np.random.default_rng(seed)
    k0 = 2 * np.pi * nu0 / C
    cells = paths.cells
    if which == "single":
        cells = [max(cells, key=lambda pl: pl[0].size)]
    I = np.zeros_like(dnu_grid, dtype=float)
    norm = 0.0
    for P, L in cells:
        ph = rng.uniform(0, 2 * np.pi, P.size)[None, :] + k0 * dneff_dT * dT * L[None, :] + 2 * np.pi * np.outer(dnu_grid, paths.n_g * L / C)
        amp = np.sqrt(P)[None, :]
        if alpha_cm is not None:
            amp = amp * np.exp(-0.5 * paths.Gamma * np.asarray(alpha_cm)[:, None] * 100.0 * L[None, :])
        I += np.abs((amp * np.exp(1j * ph)).sum(1)) ** 2
        norm += P.sum()
    return I / norm
