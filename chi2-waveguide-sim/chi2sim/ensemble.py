"""Ensemble statistics of random and fabrication-perturbed poling: Monte-Carlo and closed forms.

Closed forms (undepleted, uniform Δk):

Renewal structure (alternating ±1 domains, i.i.d. lengths ℓ, φ(q) = E e^{iqℓ}, N = L/⟨ℓ⟩ walls):
    ⟨|G(q)|²⟩ ≈ (4N/q²) Re[(1 - φ)/(1 + φ)]                       (large N, end terms dropped)
  derived from G = (1/iq) Σ_j c_j e^{iqz_j}, c_j = ±2 alternating, E[(-1)^d e^{iq(z_{j+d}-z_j)}] = (-φ)^d.
  Linear in N away from the ordered limit; |G|² is then a sum of many random phasors, i.e. fully
  developed speckle with σ/mean → 1 (Baudrier-Raybaut et al., Nature 432, 374 (2004)).

Wall jitter (each wall of a given structure moved by i.i.d. N(0, σ²), no wall crossings):
    ⟨|G|²⟩ = |X + χ Σ a_j|² + (1 - χ²) Σ |a_j|²,   χ = e^{-q²σ²/2},  a_j = c_j e^{iqz_j}/(iq),
  X the end terms. At the ordered peak the coherent part falls by χ² = e^{-K²σ²}; the rest is an
  incoherent background of order N/q² (Fejer et al., IEEE JQE 28, 2631 (1992), sec. on random errors).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import fast
from .metrics import ensemble_stats
from .poling import LengthDistribution, Poling


def monte_carlo(generator: Callable[[np.random.Generator], Poling], n_real, delta_k, seed=0, gain=0.0,
                backend="numpy", chunk=32) -> dict:
    """|G|² for n_real structures from generator(rng); returns {"G2": (n_real, M), **ensemble_stats}."""
    rng = np.random.default_rng(seed)
    q = np.atleast_1d(np.asarray(delta_k, dtype=float))
    out = np.empty((n_real, q.size))
    for i in range(0, n_real, chunk):
        ps = [generator(rng) for _ in range(min(chunk, n_real - i))]
        out[i:i + len(ps)] = np.abs(fast.spectrum(ps, q, gain=gain, backend=backend)) ** 2
    return {"G2": out, **ensemble_stats(out)}


def renewal_mean(dist: LengthDistribution, delta_k, length) -> np.ndarray:
    """Ensemble-mean |G|² of alternating domains with i.i.d. lengths (large-N closed form)."""
    q = np.asarray(delta_k, dtype=float)
    phi = dist.char(q)
    n = length / dist.actual_mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 4 * n / q**2 * np.real((1 - phi) / (1 + phi))


def jitter_mean(base: Poling, sigma, delta_k) -> np.ndarray:
    """Exact ensemble-mean |G|² of `base` with i.i.d. Gaussian wall jitter σ (walls do not cross)."""
    q = np.atleast_1d(np.asarray(delta_k, dtype=float))
    s = 1j * q
    a = np.exp(s[:, None] * base.walls[None, :]) * base.jumps()[None, :] / s[:, None]
    x = (base.g[-1] * np.exp(s * base.length) - base.g[0]) / s
    chi = np.exp(-0.5 * (q * sigma) ** 2)
    return np.abs(x + chi * a.sum(axis=1)) ** 2 + (1 - chi**2) * np.sum(np.abs(a) ** 2, axis=1)
