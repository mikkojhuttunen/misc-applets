"""Inverse design of QPM structures for a centre wavelength and FWHM bandwidth (plan section 9, stages 0-3).

Undepleted, uniform waveguide: η(λ) = κ² |G(Δk(λ))|², G(q) = ∫ g e^{iqz} dz. A band in λ maps to a band
in Δk through the (nonlinear) dispersion Δk(λ); the design works in Δk space and reports in λ.

Stage 1  `efficiency_ceiling`: Parseval ∫|G|² dq = 2πL for ±1 structures; a locally periodic binary
         structure puts (2/π)² ≈ 0.405 of it into the first-order sideband, so the in-band mean of |G|²
         is at most about 0.405·2πL/B_k (B_k = band width in Δk). Approximate bound: it assumes the
         sideband structure of locally periodic gratings.
Stage 2  `chirp_from_spectrum`: stationary phase, |G|² ≈ 2π a²/|K'(z)| at K(z) = Δk, a = (2/π) sin πD;
         dz/dK = S(K)/(2π a²). A flat target gives a linear chirp, which reaches the ceiling.
Stage 3  `poling.pwm` walls at Φ = 2πn ± πD(z); `parasitic_orders`; fabrication snapping and l_min.
Statistical design (`random_band_design`): i.i.d. domain lengths whose ensemble-mean spectrum
         (ensemble.renewal_mean) is centred on the band with the requested FWHM.
`design(...)` runs either and returns a `Design` with the predicted spectrum and a report: band_stats over
the FWHM band (edges at half power by construction) and core_stats over its central 80 % (flat-top ripple).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import ensemble, fast, poling
from .metrics import band_stats, fwhm
from .poling import FAB_PRESETS, DEFAULT_FAB, LengthDistribution, Poling

A1 = 2 / math.pi                       # first-order amplitude of a D = 0.5 binary grating


# ----------------------------------------------------------------------------- stage 0

@dataclass(frozen=True)
class Band:
    lambda0: float
    fwhm: float
    k0: float
    k_lo: float
    k_hi: float
    slope: float                        # dΔk/dλ at λ0

    @property
    def b_k(self) -> float:
        return self.k_hi - self.k_lo


def band_to_dk(dk_of_lambda: Callable, lambda0, fwhm_lambda, n=2001) -> Band:
    """Map the λ band λ0 ± FWHM/2 to [Δk_lo, Δk_hi] on a dense grid (no linearisation)."""
    lam = lambda0 + np.linspace(-0.5, 0.5, n) * fwhm_lambda
    dk = dk_of_lambda(lam)
    h = lambda0 * 1e-5
    slope = float((dk_of_lambda(lambda0 + h) - dk_of_lambda(lambda0 - h)) / (2 * h))
    return Band(lambda0, fwhm_lambda, float(dk_of_lambda(lambda0)), float(dk.min()), float(dk.max()), slope)


def lambda_grid(band: Band, length, span=3.0, oversample=8, n_max=40001) -> np.ndarray:
    """λ grid covering span × FWHM, resolving the ordered-grating bandwidth 2π/(L |dΔk/dλ|) by `oversample`."""
    width = span * band.fwhm
    res = 2 * math.pi / (length * abs(band.slope)) / oversample
    n = int(min(n_max, max(401, math.ceil(width / res))))
    return band.lambda0 + np.linspace(-width / 2, width / 2, n)


# ----------------------------------------------------------------------------- stage 1

def efficiency_ceiling(length, b_k, amplitude=A1) -> dict:
    """Parseval bound on the in-band mean of |G|² and the ordered-grating peak, both in m²."""
    total = 2 * math.pi * length
    mean_max = amplitude**2 * total / b_k
    peak = (amplitude * length) ** 2
    return {"parseval_total": total, "sideband_fraction": amplitude**2, "mean_G2_max": mean_max,
            "ordered_peak_G2": peak, "ratio_to_ordered": mean_max / peak,
            "valid": bool(b_k * length > 2 * math.pi)}


# ----------------------------------------------------------------------------- stage 2

def chirp_from_spectrum(k_grid, s_target, amplitude=A1, length=None) -> dict:
    """Stationary-phase chirp z(K) = ∫ S dK / (2π a²).

    s_target: target |G|² on k_grid (m²), or a relative shape when `length` is given (then it is scaled
    so the chirp spans exactly `length`). Returns z and K samples (monotonic), the length and the scale."""
    k = np.asarray(k_grid, dtype=float)
    s = np.asarray(s_target, dtype=float)
    z = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(k))]) / (2 * math.pi * amplitude**2)
    scale = 1.0
    if length is not None:
        scale = length / z[-1]
        z = z * scale
    return {"z": z, "k": k, "length": float(z[-1]), "scale": scale, "level": s * scale}


def poling_from_chirp(chirp: dict, duty=0.5, points_per_domain=40) -> Poling:
    """PWM realisation of a chirp z(K) (from chirp_from_spectrum)."""
    L = chirp["length"]
    kmax = float(np.max(np.abs(chirp["k"])))
    zz = np.linspace(0.0, L, int(points_per_domain * L * kmax / math.pi) + 2)
    kz = np.interp(zz, chirp["z"], chirp["k"])
    d = duty(zz) if callable(duty) else duty
    return poling.pwm(zz, poling.phase_from_k(zz, kz), d)


# ----------------------------------------------------------------------------- stage 3

def parasitic_orders(k_lo, k_hi, duty, dk_of_lambda, lam, orders=range(2, 8)) -> list[dict]:
    """Higher grating orders m with amplitude |sin(mπD)/(m sin πD)| relative to the first, and the part of
    the λ grid whose Δk falls inside [m k_lo, m k_hi] (would be phase matched by that order)."""
    dk = dk_of_lambda(np.asarray(lam, dtype=float))
    out = []
    for m in orders:
        rel = abs(math.sin(m * math.pi * duty) / (m * math.sin(math.pi * duty)))
        hit = (dk >= m * k_lo) & (dk <= m * k_hi)
        out.append({"order": m, "rel_amplitude": rel, "k_range": (m * k_lo, m * k_hi),
                    "lambda_hit": (float(lam[hit].min()), float(lam[hit].max())) if hit.any() and rel > 1e-9 else None})
    return out


# ----------------------------------------------------------------------------- statistical design

def random_band_design(band: Band, length, dk_of_lambda: Callable, kind="gaussian", l_min=None, fab=DEFAULT_FAB, tol=2e-3) -> dict:
    """Domain-length distribution whose ensemble-mean spectrum peaks at K0 with FWHM = band FWHM (in λ).

    Mean length ℓ̄ ≈ π/K0 (first order of alternating domains); the spread σ sets the Lorentzian-like
    width, (1 - |φ|) ≈ B_k ℓ̄/2, so σ² ≈ π B_k / K0³ as the start of a bisection on the exact closed form."""
    l_min = FAB_PRESETS[fab]["l_min"] if l_min is None else l_min
    k0 = abs(band.k0)
    lam = lambda_grid(band, length, span=6.0)
    dk = np.abs(np.asarray(dk_of_lambda(lam)))

    def make(mean, sigma):
        if kind == "gamma":
            return LengthDistribution("gamma", mean, sigma, l_min=min(l_min, 0.5 * mean))
        return LengthDistribution(kind, mean, sigma, l_min=l_min)

    def width(mean, sigma):
        return fwhm(lam, ensemble.renewal_mean(make(mean, sigma), dk, length))

    mean = math.pi / k0
    lo, hi = 1e-3 * math.sqrt(math.pi * band.b_k / k0**3), min(0.5 * mean, 30 * math.sqrt(math.pi * band.b_k / k0**3))
    if kind == "uniform":
        hi = min(hi, (mean - l_min) / math.sqrt(3) * 0.999)
    for _ in range(3):                          # re-centre the mean, then re-fit the spread
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            w = width(mean, mid)
            if not np.isfinite(w) or w > band.fwhm:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1 + tol:
                break
        sigma = math.sqrt(lo * hi)
        s = ensemble.renewal_mean(make(mean, sigma), dk, length)
        lam_peak = lam[int(np.argmax(s))]
        k_peak = abs(float(dk_of_lambda(np.array([lam_peak]))[0]))
        mean *= k_peak / k0
        lo, hi = sigma / 3, min(sigma * 3, 0.5 * mean)
    dist = make(mean, sigma)
    return {"dist": dist, "lambda": lam, "mean_G2": ensemble.renewal_mean(dist, dk, length), "sigma": sigma, "mean": mean}


# ----------------------------------------------------------------------------- wrapper

@dataclass
class Design:
    mode: str
    band: Band
    length: float
    lam: np.ndarray
    G2: np.ndarray                         # predicted |G|² (mean over the ensemble for statistical designs)
    poling: Poling | None                  # deterministic mask, or one realization
    generator: Callable | None = None      # rng -> Poling, statistical designs
    report: dict = field(default_factory=dict)

    def efficiency(self, kappa) -> np.ndarray:
        """η(λ) = κ²|G|² in 1/W (undepleted)."""
        return kappa**2 * self.G2


def design(dk_of_lambda: Callable, lambda0, fwhm_lambda, length, mode="chirp", fab=DEFAULT_FAB, edge=0.25, apod="hann",
           kind="gaussian", sigma_wall=0.0, n_real=64, seed=0, backend="numpy") -> Design:
    """QPM structure for λ0 and FWHM (pump wavelength, SHG convention).

    mode "chirp": apodised linear chirp, PWM walls, fabrication constraints; K range tuned so the
                  achieved FWHM matches. With sigma_wall > 0 the ensemble mean under wall jitter is reported too.
    mode "random": statistical design (i.i.d. lengths, `kind` distribution); spectrum = closed-form ensemble
                  mean, checked by Monte-Carlo over n_real realizations; speckle contrast reported.
    mode "hybrid": chirp + random wall jitter σ (statistical ensemble of chirped masks)."""
    band = band_to_dk(dk_of_lambda, lambda0, fwhm_lambda)
    lam = lambda_grid(band, length)
    dk = np.abs(dk_of_lambda(lam))
    sign = 1.0 if band.k0 > 0 else -1.0
    klo, khi = sorted((abs(band.k_lo), abs(band.k_hi)))
    ceil = efficiency_ceiling(length, khi - klo)
    fabp = FAB_PRESETS[fab]
    rep: dict = {"band": band, "ceiling": ceil, "fab": dict(fabp, preset=fab)}

    if mode in ("chirp", "hybrid"):
        def build(stretch):
            kc, half = (klo + khi) / 2, (khi - klo) / 2 * stretch
            amp = lambda z: poling.window(z, length, apod if edge > 0 else "none", edge)
            p = poling.chirped(length, kc - half, kc + half, duty=lambda z: poling.duty_from_amplitude(amp(z)))
            return poling.fabricate(p, fabp["l_min"], fabp["grid"])

        def width(stretch):
            return fwhm(lam, np.abs(fast.spectrum(build(stretch), dk, backend=backend)[0]) ** 2)

        lo, hi = 0.8, 2.0
        while hi < 20 and not (width(hi) > fwhm_lambda):   # narrow bands near the ordered limit need more range
            lo, hi = hi, hi * 1.6
        for _ in range(30):                        # match the achieved FWHM (in λ) to the target
            mid = 0.5 * (lo + hi)
            w = width(mid)
            if not np.isfinite(w) or w > fwhm_lambda:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-3:
                break
        stretch = 0.5 * (lo + hi)
        p = build(stretch)
        G2 = np.abs(fast.spectrum(p, dk, backend=backend)[0]) ** 2
        rep.update(stretch=stretch, k_range=((klo + khi) / 2 - (khi - klo) / 2 * stretch, (klo + khi) / 2 + (khi - klo) / 2 * stretch),
                   parasitic=parasitic_orders(klo * (1 - 0.5 * (stretch - 1)), khi * (1 + 0.5 * (stretch - 1)), 0.5,
                                              lambda x: np.abs(dk_of_lambda(x)), lam))
        gen = None
        if mode == "hybrid" or sigma_wall > 0:
            gen = lambda rng, p=p: poling.jittered(p, sigma_wall, rng)
            mc = ensemble.monte_carlo(gen, n_real, dk, seed=seed, backend=backend)
            rep["jitter_mean_exact"] = ensemble.jitter_mean(p, sigma_wall, dk)
            rep["mc"] = {k: v for k, v in mc.items() if k != "G2"}
            if mode == "hybrid":
                G2 = rep["jitter_mean_exact"]
        d = Design(mode, band, length, lam, G2, p, gen, rep)
    elif mode == "random":
        rd = random_band_design(band, length, dk_of_lambda, kind=kind, l_min=fabp["l_min"])
        dist = rd["dist"]
        G2 = ensemble.renewal_mean(dist, dk, length)

        def gen(rng, dist=dist):
            return poling.fabricate(poling.random_domains(length, dist, rng), fabp["l_min"], fabp["grid"])

        mc = ensemble.monte_carlo(gen, n_real, dk, seed=seed, backend=backend)
        rep.update(dist=dist, mc={k: v for k, v in mc.items() if k != "G2"})
        d = Design(mode, band, length, lam, G2, gen(np.random.default_rng(seed)), gen, rep)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    bs = band_stats(lam, d.G2, lambda0 - fwhm_lambda / 2, lambda0 + fwhm_lambda / 2)
    core = band_stats(lam, d.G2, lambda0 - 0.4 * fwhm_lambda, lambda0 + 0.4 * fwhm_lambda)
    rep.update(fwhm=fwhm(lam, d.G2), band_stats=bs, core_stats=core, fraction_of_ceiling=float(bs["mean"] / ceil["mean_G2_max"]),
               sign=sign)
    if "mc" in rep:
        mcb = band_stats(lam, rep["mc"]["mean"], lambda0 - fwhm_lambda / 2, lambda0 + fwhm_lambda / 2)
        rep["mc_band_mean"] = float(mcb["mean"])
        sel = (lam >= lambda0 - fwhm_lambda / 2) & (lam <= lambda0 + fwhm_lambda / 2)
        rep["speckle_contrast"] = float(np.nanmean(rep["mc"]["contrast"][sel]))
    if d.poling is not None:
        ell = d.poling.lengths
        rep["min_domain"] = float(ell[1:-1].min()) if ell.size > 2 else float(ell.min())
        rep["domains"] = d.poling.n_domains
    return d
