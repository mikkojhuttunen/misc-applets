"""Poling structures: one canonical representation and the generators that produce it.

Canonical form (`Poling`): domain boundaries z_0 = 0 < z_1 < ... < z_N = L and one value g_k per domain
(±1 for hard domains, any value in [-1, 1] for effective-medium / apodised profiles). This is the input
of math-engines `cwe_shg` and of the batched solvers in `chi2sim.fast`.

Generators (all SI, all return `Poling`):
  ordered            period Λ, duty cycle D, offset
  jittered           Gaussian wall-position noise on any structure (fabrication-error model)
  random_domains     i.i.d. (or AR(1)-correlated) domain lengths from a `LengthDistribution`
  quasi_random       PRBS, Fibonacci, Thue-Morse, Rudin-Shapiro, Barker-13 on a unit cell
  from_phase / pwm   walls at Φ(z) = 2πn ± πD(z): chirped, apodised, adiabatic structures
  chirped            linear or tanh local grating vector K(z), optional duty-cycle apodisation
  from_walls         user wall list
Fabrication: `fabricate` snaps walls to a grid and removes domains shorter than l_min. `FAB_PRESETS`
gives defaults taken from thin-film LiNbO3 poling demonstrations (verify against your own process).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from engines.cwe_shg import engine as cwe

# Minimum domain length and wall grid. "reproducible": 600 nm periods in x-cut TFLN (Zhao et al.,
# Appl. Phys. Lett. 117, 2020); "frontier": 215 nm periods (Sabatti et al., 2025, x-cut TFLN, preprint
# abstract figure); "conventional": the few-µm periods used for 1550 nm SHG and DFG.
FAB_PRESETS = {
    "reproducible": {"l_min": 0.30e-6, "grid": 10e-9},
    "frontier": {"l_min": 0.11e-6, "grid": 5e-9},
    "conventional": {"l_min": 1.0e-6, "grid": 20e-9},
}
DEFAULT_FAB = "reproducible"


@dataclass(frozen=True)
class Poling:
    boundaries: np.ndarray
    g: np.ndarray

    def __post_init__(self):
        b, g = cwe._check_domains(self.boundaries, self.g)
        object.__setattr__(self, "boundaries", b)
        object.__setattr__(self, "g", g)

    @property
    def length(self) -> float:
        return float(self.boundaries[-1])

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.boundaries)

    @property
    def walls(self) -> np.ndarray:
        """Interior wall positions (excluding 0 and L)."""
        return self.boundaries[1:-1]

    @property
    def n_domains(self) -> int:
        return int(self.g.size)

    @property
    def is_binary(self) -> bool:
        return bool(np.all(np.abs(self.g) == 1.0))

    def jumps(self) -> np.ndarray:
        """g_{k-1} - g_k at each interior wall."""
        return self.g[:-1] - self.g[1:]

    def merged(self) -> "Poling":
        """Equal neighbouring domains joined into one (removes walls without a sign change)."""
        keep = np.concatenate([[True], self.g[1:] != self.g[:-1]])
        starts = self.boundaries[:-1][keep]
        return Poling(np.concatenate([starts, [self.length]]), self.g[keep])

    def wall_list(self) -> dict:
        """Mask description: first sign and the interior walls (for export)."""
        return {"first_sign": float(self.g[0]), "walls": self.walls.copy(), "length": self.length}


def from_walls(walls, length, first_sign=1.0) -> Poling:
    """Alternating ±1 domains from interior wall positions (sorted, inside (0, L))."""
    w = np.sort(np.asarray(walls, dtype=float))
    w = w[(w > 0) & (w < length)]
    b = np.concatenate([[0.0], w, [float(length)]])
    b = b[np.concatenate([[True], np.diff(b) > 0])]
    return Poling(b, float(first_sign) * (-1.0) ** np.arange(b.size - 1))


def ordered(length, period, duty_cycle=0.5, offset=0.0) -> Poling:
    return Poling(*cwe.periodic_domains(length, period, duty_cycle, offset))


def jittered(base: Poling, sigma_wall, rng=None) -> Poling:
    """Interior walls displaced by i.i.d. N(0, σ²); crossing walls annihilate in pairs (binary structures)."""
    rng = np.random.default_rng(rng)
    if not base.is_binary:
        raise ValueError("jittered needs a binary (±1) structure")
    w = base.walls + rng.normal(0.0, sigma_wall, base.walls.size)
    # walls are sign flips: their order does not matter, only their positions; flips outside (0, L) drop out
    inside = (w > 0) & (w < base.length)
    flips_before = np.count_nonzero(w <= 0)
    first = base.g[0] * (-1.0) ** flips_before
    return from_walls(w[inside], base.length, first)


# ----------------------------------------------------------------------------- random lengths

@dataclass(frozen=True)
class LengthDistribution:
    """Domain-length distribution with sampling and its characteristic function φ(q) = E[e^{iqℓ}].

    kind: "gaussian" (mean, std; truncated below l_min), "uniform" (mean ± half-width = std·√3),
    "gamma" (l_min + Gamma with the given mean and std of the shifted part), "exponential"
    (l_min + Exp), "lognormal" (mean, std; truncated below l_min). `mean` and `std` refer to the
    untruncated parent for gaussian/lognormal, to the full distribution for the others."""

    kind: str
    mean: float
    std: float = 0.0
    l_min: float = 0.0

    def __post_init__(self):
        if self.kind not in ("gaussian", "uniform", "gamma", "exponential", "lognormal"):
            raise ValueError(f"unknown distribution {self.kind!r}")
        if not (self.mean > self.l_min >= 0):
            raise ValueError("need mean > l_min >= 0")
        if self.kind == "uniform" and self.mean - math.sqrt(3) * self.std < self.l_min - 1e-15:
            raise ValueError("uniform distribution reaches below l_min")

    def sample(self, n, rng) -> np.ndarray:
        rng = np.random.default_rng(rng)
        m, s, lo = self.mean, self.std, self.l_min
        if self.kind == "uniform":
            return rng.uniform(m - math.sqrt(3) * s, m + math.sqrt(3) * s, n)
        if self.kind in ("gamma", "exponential"):
            mu = m - lo
            if self.kind == "exponential" or s <= 0:
                return lo + rng.exponential(mu, n)
            k = (mu / s) ** 2
            return lo + rng.gamma(k, mu / k, n)
        out = np.empty(0)
        while out.size < n:
            need = n - out.size
            if self.kind == "gaussian":
                x = rng.normal(m, s, 2 * need + 16)
            else:
                sig2 = math.log(1 + (s / m) ** 2)
                x = rng.lognormal(math.log(m) - sig2 / 2, math.sqrt(sig2), 2 * need + 16)
            out = np.concatenate([out, x[x >= max(lo, 1e-15)][:need]])
        return out

    def _grid_pdf(self, n=6001):
        m, s, lo = self.mean, self.std, self.l_min
        if self.kind == "gaussian":
            a, b = max(lo, m - 9 * s), m + 9 * s
            x = np.linspace(a, b, n)
            p = np.exp(-0.5 * ((x - m) / s) ** 2)
        else:  # lognormal
            sig2 = math.log(1 + (s / m) ** 2)
            mu = math.log(m) - sig2 / 2
            a, b = max(lo, math.exp(mu - 9 * math.sqrt(sig2)), 1e-15), math.exp(mu + 9 * math.sqrt(sig2))
            x = np.linspace(a, b, n)
            p = np.exp(-((np.log(x) - mu) ** 2) / (2 * sig2)) / x
        w = np.full(n, (x[1] - x[0]))
        w[0] = w[-1] = w[0] / 2
        pw = p * w
        return x, pw / pw.sum()

    def char(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        m, s, lo = self.mean, self.std, self.l_min
        if self.kind == "uniform":
            h = math.sqrt(3) * s
            return np.exp(1j * q * m) * np.sinc(q * h / np.pi)
        if self.kind in ("gamma", "exponential"):
            mu = m - lo
            k = 1.0 if (self.kind == "exponential" or s <= 0) else (mu / s) ** 2
            return np.exp(1j * q * lo) * (1 - 1j * q * mu / k) ** (-k)
        if self.kind == "gaussian" and lo <= m - 9 * s:
            return np.exp(1j * q * m - 0.5 * (q * s) ** 2)
        x, pw = self._grid_pdf()
        out = np.empty(q.shape, dtype=complex)
        flat, o = q.ravel(), out.ravel()
        for i in range(0, flat.size, 256):
            o[i:i + 256] = np.exp(1j * np.outer(flat[i:i + 256], x)) @ pw
        return out

    def actual_mean(self) -> float:
        if self.kind in ("gaussian", "lognormal") and not (self.kind == "gaussian" and self.l_min <= self.mean - 9 * self.std):
            x, pw = self._grid_pdf()
            return float(x @ pw)
        return self.mean


def random_domains(length, dist: LengthDistribution, rng=None, correlation=0.0, first_sign=1.0) -> Poling:
    """Alternating ±1 domains with lengths from `dist`, truncated at L.

    correlation ρ ∈ (-1, 1): AR(1) correlation of neighbouring lengths, applied through a Gaussian
    copula (marginal distribution kept, then the l_min floor re-applied)."""
    rng = np.random.default_rng(rng)
    n = int(math.ceil(1.3 * length / dist.actual_mean())) + 16
    while True:
        ell = dist.sample(n, rng)
        if correlation:
            ell = _ar1_reorder(ell, correlation, rng)
        if ell.sum() >= length:
            break
        n *= 2
    b = np.concatenate([[0.0], np.cumsum(ell)])
    k = int(np.searchsorted(b, length))
    b = np.concatenate([b[:k], [float(length)]])
    b = b[np.concatenate([[True], np.diff(b) > 0])]
    return Poling(b, float(first_sign) * (-1.0) ** np.arange(b.size - 1))


def _ar1_reorder(ell, rho, rng):
    """Rank-reorder samples to follow an AR(1) Gaussian sequence: keeps the marginal, adds correlation."""
    x = np.empty(ell.size)
    x[0] = rng.normal()
    e = rng.normal(size=ell.size) * math.sqrt(1 - rho * rho)
    for i in range(1, ell.size):
        x[i] = rho * x[i - 1] + e[i]
    out = np.empty_like(ell)
    out[np.argsort(x)] = np.sort(ell)
    return out


# ----------------------------------------------------------------------------- quasi-random sequences

def sequence(name, n, seed=None) -> np.ndarray:
    """First n symbols (±1) of a deterministic aperiodic or pseudo-random sequence."""
    k = np.arange(n)
    if name == "thue_morse":
        bits = np.array([bin(int(i)).count("1") & 1 for i in k])
        return 1.0 - 2.0 * bits
    if name == "rudin_shapiro":
        bits = np.array([sum(1 for j in range(int(i).bit_length()) if (i >> j) & 3 == 3) & 1 for i in k])
        return 1.0 - 2.0 * bits
    if name == "fibonacci":
        s = "A"
        while len(s) < n:
            s = s.replace("A", "a").replace("B", "A").replace("a", "AB")
        return np.array([1.0 if c == "A" else -1.0 for c in s[:n]])
    if name == "barker13":
        code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        return np.resize(code, n)
    if name == "prbs":
        # maximal-length LFSR x^15 + x^14 + 1 (PRBS-15), period 32767; seed picks the starting state
        state = (int(seed) if seed is not None else 1) % 32767 or 1
        out = np.empty(n)
        for i in range(n):
            bit = ((state >> 14) ^ (state >> 13)) & 1
            state = ((state << 1) | bit) & 0x7FFF
            out[i] = 1.0 if bit else -1.0
        return out
    raise ValueError(f"unknown sequence {name!r}")


def quasi_random(name, unit, length, seed=None) -> Poling:
    """Sequence symbols on cells of length `unit`; equal neighbours merge into one domain."""
    n = int(math.ceil(length / unit))
    s = sequence(name, n, seed)
    b = np.minimum(np.arange(n + 1) * unit, length)
    b = b[np.concatenate([[True], np.diff(b) > 0])]
    return Poling(b, s[: b.size - 1]).merged()


# ----------------------------------------------------------------------------- phase-defined structures

def pwm(z, phase, duty=0.5) -> Poling:
    """Binary structure with walls where Φ(z) = 2πn ± πD(z) (pulse-width / pulse-position modulation).

    z: fine grid on [0, L] (resolve the shortest domain by ≥ 20 points); phase Φ(z) and duty D(z) on it.
    +1 where cos Φ > cos πD. The first-order Fourier component is a(z) e^{∓iΦ}, a = (2/π) sin πD, so
    Φ sets the local grating phase (K = dΦ/dz) and D the local amplitude."""
    z = np.asarray(z, dtype=float)
    ph = np.asarray(phase, dtype=float)
    d = np.broadcast_to(np.asarray(duty, dtype=float), z.shape)
    if np.any(d <= 0) or np.any(d >= 1):
        raise ValueError("duty must lie in (0, 1)")
    f = np.cos(ph) - np.cos(np.pi * d)
    s = np.sign(f)
    s[s == 0] = 1.0
    idx = np.nonzero(s[1:] != s[:-1])[0]
    t = f[idx] / (f[idx] - f[idx + 1])
    walls = z[idx] + t * (z[idx + 1] - z[idx])
    return from_walls(walls, z[-1], s[0])


def from_sign(z, f) -> Poling:
    """Binary structure g = sign(f(z)) with walls at the linearly interpolated zero crossings of f on grid z."""
    z = np.asarray(z, dtype=float)
    f = np.asarray(f, dtype=float)
    s = np.sign(f)
    s[s == 0] = 1.0
    idx = np.nonzero(s[1:] != s[:-1])[0]
    t = f[idx] / (f[idx] - f[idx + 1])
    return from_walls(z[idx] + t * (z[idx + 1] - z[idx]), z[-1], s[0])


def phase_from_k(z, k_local) -> np.ndarray:
    """Φ(z) = ∫ K dz (trapezoid) on the grid z."""
    k = np.asarray(k_local, dtype=float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(z))])


def chirped(length, k_start, k_end, profile="linear", duty: float | Callable = 0.5, steepness=3.0, points_per_domain=40) -> Poling:
    """Locally periodic structure sweeping the grating vector from k_start to k_end.

    profile "linear": K(z) linear in z. "tanh": K = Kc + (ΔK/2) tanh(s(2z/L - 1))/tanh(s), the slow-middle
    sweep used for adiabatic conversion. duty: constant or callable D(z) (apodisation)."""
    kmax = max(abs(k_start), abs(k_end))
    n = int(points_per_domain * length * kmax / np.pi) + 2
    z = np.linspace(0.0, length, n)
    u = z / length
    if profile == "linear":
        k = k_start + (k_end - k_start) * u
    elif profile == "tanh":
        kc, dk = (k_start + k_end) / 2, k_end - k_start
        k = kc + dk / 2 * np.tanh(steepness * (2 * u - 1)) / np.tanh(steepness)
    else:
        raise ValueError(f"unknown profile {profile!r}")
    d = duty(z) if callable(duty) else duty
    return pwm(z, phase_from_k(z, k), d)


def duty_from_amplitude(a_rel) -> np.ndarray:
    """D with sin(πD) = A (0 ≤ A ≤ 1): first-order amplitude A·(2/π) by duty-cycle apodisation."""
    a = np.clip(np.asarray(a_rel, dtype=float), 1e-6, 1.0)
    return np.arcsin(a) / np.pi


def window(z, length, kind="tanh", edge=0.1) -> np.ndarray:
    """Relative amplitude A(z) ∈ (0, 1]: flat with smooth edges of fractional width `edge`."""
    u = np.asarray(z, dtype=float) / length
    if kind == "none":
        return np.ones_like(u)
    if kind == "tanh":
        return np.clip(0.5 * (np.tanh((u - edge) / (edge / 3)) - np.tanh((u - 1 + edge) / (edge / 3))), 1e-3, 1)
    if kind == "hann":
        r = np.ones_like(u)
        lo, hi = u < edge, u > 1 - edge
        r[lo] = 0.5 - 0.5 * np.cos(np.pi * u[lo] / edge)
        r[hi] = 0.5 - 0.5 * np.cos(np.pi * (1 - u[hi]) / edge)
        return np.clip(r, 1e-3, 1)
    if kind == "gaussian":
        return np.clip(np.exp(-0.5 * ((u - 0.5) / edge) ** 2), 1e-3, 1)
    raise ValueError(f"unknown window {kind!r}")


# ----------------------------------------------------------------------------- fabrication

def fabricate(p: Poling, l_min=None, grid=None, preset=DEFAULT_FAB) -> Poling:
    """Snap walls to the grid, then absorb every domain shorter than l_min into its neighbours.

    A short interior domain takes the sign of its neighbours (removing two walls); a short first or last
    domain joins its neighbour. Repeated until every domain is ≥ l_min (except a device shorter than it)."""
    if not p.is_binary:
        raise ValueError("fabricate needs a binary (±1) structure")
    fab = FAB_PRESETS[preset]
    l_min = fab["l_min"] if l_min is None else l_min
    grid = fab["grid"] if grid is None else grid
    b, g = p.boundaries.copy(), p.g.copy()
    if grid:
        inner = np.round(b[1:-1] / grid) * grid
        b = np.concatenate([[0.0], inner, [b[-1]]])
    while True:
        ell = np.diff(b)
        keep = ell > 0
        b = np.concatenate([b[:-1][keep], [b[-1]]])
        g = g[keep]
        q = Poling(b, g).merged()
        b, g = q.boundaries, q.g
        ell = np.diff(b)
        short = np.nonzero(ell < l_min - 1e-15)[0]
        if short.size == 0 or g.size == 1:
            return q
        # flip the shortest domain of each run of short ones, left to right, without touching neighbours twice
        done = np.zeros(g.size, dtype=bool)
        for k in short[np.argsort(ell[short], kind="stable")]:
            if done[max(k - 1, 0):k + 2].any():
                continue
            g[k] = g[k - 1] if k > 0 else g[k + 1]
            done[k] = True
