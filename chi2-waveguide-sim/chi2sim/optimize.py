"""Mask optimisation: IFTA synthesis, wall-level gradient descent (JAX) and low-dimensional evolution, with
objectives over wavelength bands and over the wafer thickness ensemble.

Targets are evaluated in the undepleted limit (η_SHG = κ²|G(Δk)|², η_THG from the cascade integral); final masks
are snapped to the fabrication grid, re-checked for l_min and re-evaluated with the exact solvers.

Objectives (all maximised)
  mean    Σ w η                          (band average, or expected efficiency over devices)
  flat    mean - α·std over the band     (flat top; per device, then averaged)
  yield   Σ w sigmoid((η/η_ref - r)/s)   (soft fraction of devices above r·η_ref)
  min     soft minimum over devices      (-(s/β) log Σ w e^{-β η/s}, s = mean η; worst-case robust design)

Device ensembles enter through deterministic quadrature nodes of the thickness spread (`device_nodes`): the
expected value over a uniform ±5 nm spread is a smooth integral, so 9-21 nodes replace hundreds of samples while
optimising; the result is then checked on a random ensemble (wafer.shg_devices / thg.thg_devices).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

from engines.cwe_cascade import engine as casc

from . import ensemble, fast, poling
from .metrics import band_stats
from .poling import DEFAULT_FAB, FAB_PRESETS, Poling


# ----------------------------------------------------------------------------- targets

@dataclass
class Target:
    """|Δk| for D devices × M wavelengths, weights (sum 1 per device row, devices weighted by `device_w`)."""

    q: np.ndarray
    kappa2: float = 1.0
    kind: str = "mean"
    device_w: np.ndarray | None = None
    alpha: float = 1.0                  # flatness weight
    ref: float | None = None            # yield reference efficiency (1/W)
    rel: float = 0.5
    soft: float = 0.1
    beta: float = 10.0

    def __post_init__(self):
        self.q = np.atleast_2d(np.abs(np.asarray(self.q, dtype=float)))
        d = self.q.shape[0]
        self.device_w = np.full(d, 1.0 / d) if self.device_w is None else np.asarray(self.device_w, float) / np.sum(self.device_w)
        if self.kind not in ("mean", "flat", "yield", "min"):
            raise ValueError("kind must be mean, flat, yield or min")
        if self.kind == "yield" and self.ref is None:
            raise ValueError("yield needs ref")

    def score_eta(self, eta, xp=np):
        """Objective from η of shape (D, M)."""
        if self.kind == "mean":
            per = eta.mean(axis=1)
        elif self.kind == "flat":
            per = eta.mean(axis=1) - self.alpha * xp.sqrt(((eta - eta.mean(axis=1, keepdims=True)) ** 2).mean(axis=1) + 1e-300)
        elif self.kind == "yield":
            per = (1 / (1 + xp.exp(-(eta.mean(axis=1) / self.ref - self.rel) / self.soft)))
        else:
            dev = eta.mean(axis=1)
            sc = xp.sum(xp.asarray(self.device_w) * dev) + 1e-300
            x = -self.beta * dev / sc
            mx = xp.max(x)
            return -(sc / self.beta) * (mx + xp.log(xp.sum(xp.asarray(self.device_w) * xp.exp(x - mx))))
        return xp.sum(xp.asarray(self.device_w) * per)

    def evaluate(self, p: Poling) -> float:
        G = fast.spectrum(p, self.q.ravel())[0].reshape(self.q.shape)
        return float(self.score_eta(self.kappa2 * np.abs(G) ** 2))


def device_nodes(spread, n=11, with_load=False, n_load=5):
    """Quadrature of the thickness spread: (d_film, d_load, weights). Uniform: equally spaced midpoint nodes;
    Gaussian: Gauss-Hermite."""
    if spread.dist == "uniform":
        x = (np.arange(n) + 0.5) / n * 2 - 1
        f, wf = x * spread.film, np.full(n, 1.0 / n)
        xl = (np.arange(n_load) + 0.5) / n_load * 2 - 1
        l, wl = xl * spread.load, np.full(n_load, 1.0 / n_load)
    else:
        x, w = np.polynomial.hermite_e.hermegauss(n)
        f, wf = x * spread.film, w / w.sum()
        xl, wl_ = np.polynomial.hermite_e.hermegauss(n_load)
        l, wl = xl * spread.load, wl_ / wl_.sum()
    if not with_load:
        return f, np.zeros_like(f), wf
    F, Lg = np.meshgrid(f, l, indexing="ij")
    W = np.outer(wf, wl)
    return F.ravel(), Lg.ravel(), W.ravel()


def band_target(dk_of_lambda, lam_band, kappa2=1.0, kind="flat", alpha=1.0) -> Target:
    return Target(np.abs(dk_of_lambda(np.asarray(lam_band, float)))[None, :], kappa2, kind, alpha=alpha)


def robust_target(family, lam, kappa2=1.0, kind="mean", n=11, ref=None, rel=0.5) -> Target:
    """Expected-efficiency (or soft-yield) target at wavelength(s) lam over the family's thickness spread."""
    lam = np.atleast_1d(np.asarray(lam, float))
    df, dl, w = device_nodes(family.spread, n, family.has_load)
    q = family.delta_k(lam, df, dl)                     # (D, M)
    return Target(q, kappa2, kind, device_w=w, ref=ref, rel=rel)


# ----------------------------------------------------------------------------- IFTA

def envelope_to_poling(z, env, k0, a_max=2 / math.pi, points_per_domain=40, fab=DEFAULT_FAB) -> Poling:
    """PWM realisation of a first-order envelope: g ≈ Σ_m c_m e^{imΦ} with c_1 = env, Φ = k0 z - arg env,
    duty from |env| = a_max sin(πD)."""
    L = z[-1]
    zf = np.linspace(0.0, L, int(points_per_domain * L * k0 / np.pi) + 2)
    amp = np.interp(zf, z, np.abs(env))
    ph = np.interp(zf, z, np.unwrap(np.angle(env)))
    duty = poling.duty_from_amplitude(np.clip(amp / a_max, 1e-3, 1.0))
    p = poling.pwm(zf, k0 * zf - ph, duty)
    if fab is None:
        return p
    f = FAB_PRESETS[fab]
    return poling.fabricate(p, f["l_min"], f["grid"])


def ifta(k0, k_lo, k_hi, length, check_q, iters=200, a_max=2 / math.pi, n_z=4096, pad=16, guard=0.3, relax=1.0,
         edge=0.2, fab=DEFAULT_FAB, seed=None, level=0.8):
    """Gerchberg-Saxton on the first-order envelope e(z) (baseband around k0), then PWM binarisation.

    Δk space: in [k_lo, k_hi] the magnitude is set to the fixed target √(level · 2π a_max² L / B) (the stationary-
    phase / ceiling level times `level` < 1, which leaves room for the edges; relaxation `relax`), guard bands of
    relative width `guard` are cleared, the rest is free. z space: |e| ≤ a_max = 2/π (the most a binary grating
    can provide), support [0, L]. Start: Hann-apodised linear chirp (or random phase with `seed`). The iterate with
    the lowest in-band contrast (exact spectrum of the fabricated binary mask on check_q, every 20 iterations and
    at the end) is returned. The envelope is a smooth field, so its Fourier constraint can move the walls; a
    binary-domain GS on a ±1 grid stalls when the band spans few FFT bins."""
    dz = length / n_z
    z = (np.arange(n_z) + 0.5) * dz
    nfft = pad * n_z
    d = 2 * np.pi * np.fft.fftfreq(nfft, dz)                    # baseband Δ = q - k0
    lo, hi = k_lo - k0, k_hi - k0
    band = (d >= lo) & (d <= hi)
    w = (hi - lo) * guard
    guard_m = ((d >= lo - w) & (d < lo)) | ((d > hi) & (d <= hi + w))
    if seed is None:
        kp = (hi - lo) / length
        env = a_max * poling.window(z, length, "hann", edge) * np.exp(-1j * (0.5 * kp * (z - length / 2) ** 2 + (lo + hi) / 2 * z))
    else:
        env = a_max * np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, n_z))

    def fwd(e):                                                   # E(Δ) = ∫ e e^{iΔz} dz
        return np.fft.ifft(e, nfft) * nfft * dz

    def inv(E):
        return (np.fft.fft(E) / (nfft * dz))[:n_z]

    def realise_score(e):
        p = envelope_to_poling(np.concatenate([[0.0], z, [length]]), np.concatenate([[e[0]], e, [e[-1]]]), k0, a_max, fab=fab)
        G2 = np.abs(fast.spectrum(p, check_q)[0]) ** 2
        return p, G2.mean(), G2.std() / G2.mean()

    target = math.sqrt(level * 2 * math.pi * a_max**2 * length / (hi - lo))
    best = (None, 0.0, np.inf)
    hist = []
    for it in range(iters + 1):
        E = fwd(env)
        inb = np.abs(E[band])
        hist.append((np.mean(inb**2), inb.std() / inb.mean()))
        if it % 20 == 0 or it == iters:
            p, m, c = realise_score(env)
            if c < best[2]:
                best = (p, m, c)
        if it == iters:
            break
        E[band] = E[band] + relax * (target * np.exp(1j * np.angle(E[band])) - E[band])
        E[guard_m] = 0.0
        env = inv(E)
        mag = np.abs(env)
        env = np.where(mag > a_max, env * a_max / np.maximum(mag, 1e-300), env)
    return {"poling": best[0], "mean": best[1], "contrast": best[2], "history": np.array(hist), "envelope": env, "z": z}


# ----------------------------------------------------------------------------- wall-level gradient (JAX)

def _jax():
    from .fast import _jax as j
    return j()


@dataclass
class WallParams:
    """Domain lengths ℓ_k = l_min + softplus(u_k)·(L - n l_min)/Σ softplus(u): every domain ≥ l_min, Σℓ = L."""

    u: np.ndarray
    length: float
    l_min: float
    g0: float = 1.0

    @classmethod
    def from_poling(cls, p: Poling, l_min):
        q = p.merged()
        if not q.is_binary:
            raise ValueError("binary structure needed")
        ell = q.lengths
        x = np.maximum(ell - l_min, 1e-3 * l_min) / l_min
        return cls(x + np.log(-np.expm1(-x)), q.length, l_min, float(q.g[0]))      # stable inverse softplus

    def lengths(self, u=None, xp=np):
        u = self.u if u is None else u
        sp = xp.logaddexp(0.0, u)
        n = u.shape[0]
        return self.l_min + sp * (self.length - n * self.l_min) / xp.sum(sp)

    def poling(self, u=None) -> Poling:
        ell = np.asarray(self.lengths(u))
        b = np.concatenate([[0.0], np.cumsum(ell)])
        b[-1] = self.length
        return Poling(b, self.g0 * (-1.0) ** np.arange(ell.size))


def gradient_optimise(start: Poling, target: Target, l_min=None, steps=300, lr=0.03, fab=DEFAULT_FAB, log_every=0):
    """Adam on the wall parametrisation of `start` (domain count fixed), maximising target; then fabricated."""
    jax, jnp = _jax()
    f = FAB_PRESETS[fab]
    l_min = f["l_min"] if l_min is None else l_min
    wp = WallParams.from_poling(poling.fabricate(start, l_min, f["grid"]), l_min)
    n = wp.u.size
    jumps = jnp.asarray(2 * wp.g0 * (-1.0) ** np.arange(n - 1))
    g_last = wp.g0 * (-1.0) ** (n - 1)
    q = jnp.asarray(target.q.ravel())
    shape = target.q.shape
    L = wp.length

    def objective(u):
        ell = wp.lengths(u, jnp)
        walls = jnp.cumsum(ell)[:-1]
        s = 1j * q
        G = (jnp.exp(s[:, None] * walls[None, :]) @ jumps.astype(complex) + g_last * jnp.exp(s * L) - wp.g0) / s
        eta = target.kappa2 * jnp.abs(G.reshape(shape)) ** 2
        return target.score_eta(eta, jnp)

    vg = jax.jit(jax.value_and_grad(objective))
    u = jnp.asarray(wp.u)
    m, v = jnp.zeros_like(u), jnp.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    best_u, best_val = u, -np.inf
    for t in range(1, steps + 1):
        val, gr = vg(u)
        val = float(val)
        hist.append(val)
        if val > best_val:
            best_val, best_u = val, u
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr * gr
        u = u + lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)     # ascent
        if log_every and t % log_every == 0:
            print(f"  step {t}: objective {val:.4g}")
    p = poling.fabricate(wp.poling(np.asarray(best_u)), l_min, f["grid"])
    return {"poling": p, "objective": target.evaluate(p), "objective_continuous": best_val,
            "start_objective": hist[0], "history": np.array(hist)}


# ----------------------------------------------------------------------------- low-dimensional evolution

def chirp_mask(L, k0, span, edge, center=0.0, fab=DEFAULT_FAB):
    """Hann-apodised linear chirp K = k0(1 + center) ± span·k0/2 (span ≥ 0; span = 0: ordered)."""
    kc = k0 * (1 + center)
    half = max(span, 1e-6) * k0 / 2
    if edge > 0:
        duty = lambda z: poling.duty_from_amplitude(poling.window(z, L, "hann", edge))
    else:
        duty = 0.5
    p = poling.chirped(L, kc - half, kc + half, duty=duty)
    f = FAB_PRESETS[fab]
    return poling.fabricate(p, f["l_min"], f["grid"])


def evolve_chirp(target: Target, L, k0, bounds=None, sigma_max=0.0, maxiter=25, popsize=12, seed=0, fab=DEFAULT_FAB):
    """Differential evolution over (center, span, edge[, σ_wall]) of an apodised chirp for `target`.

    With sigma_max > 0 a Gaussian wall jitter σ is a design variable and the objective uses the exact
    jitter-ensemble mean (ensemble.jitter_mean): statistical design of a perturbed mask."""
    bounds = bounds or [(-0.02, 0.02), (0.0, 0.08), (0.0, 0.45)]
    if sigma_max > 0:
        bounds = list(bounds) + [(0.0, sigma_max)]

    def build(x):
        return chirp_mask(L, k0, x[1], x[2], x[0], fab)

    def value(x):
        p = build(x)
        if sigma_max > 0 and x[3] > 0:
            G2 = ensemble.jitter_mean(p, x[3], target.q.ravel()).reshape(target.q.shape)
            return float(target.score_eta(target.kappa2 * G2))
        return target.evaluate(p)

    res = differential_evolution(lambda x: -value(x), bounds, maxiter=maxiter, popsize=popsize, seed=seed,
                                 polish=False, tol=1e-6)
    return {"x": res.x, "poling": build(res.x), "objective": -res.fun, "nfev": res.nfev}


# ----------------------------------------------------------------------------- THG

def thg_mask(L, ka, kb, w_start, w_end, phase, span_a, span_b, fab=DEFAULT_FAB, points_per_domain=30):
    """Blended chirped superlattice sign(w(z) cos Φa + (1 - w(z)) cos(Φb + ψ)); K_a and K_b chirped linearly by
    relative spans span_a, span_b around ka, kb; w falls linearly from w_start to w_end."""
    z = np.linspace(0.0, L, int(points_per_domain * L * max(ka, kb) * (1 + max(span_a, span_b)) / np.pi) + 2)
    u = z / L
    pa = poling.phase_from_k(z, ka * (1 + span_a * (u - 0.5)))
    pb = poling.phase_from_k(z, kb * (1 + span_b * (u - 0.5)))
    w = w_start + (w_end - w_start) * u
    p = poling.from_sign(z, w * np.cos(pa) + (1 - w) * np.cos(pb + phase))
    f = FAB_PRESETS[fab]
    return poling.fabricate(p, f["l_min"], f["grid"])


@dataclass
class THGTarget:
    """η_THG at λ0 for D devices: q_a, q_b (D,), nominal κ_a, κ_3, device weights."""

    q_a: np.ndarray
    q_b: np.ndarray
    kappa_a: float
    kappa_3: float
    device_w: np.ndarray = field(default=None)

    def __post_init__(self):
        self.q_a, self.q_b = np.abs(np.atleast_1d(self.q_a)), np.abs(np.atleast_1d(self.q_b))
        d = self.q_a.size
        self.device_w = np.full(d, 1.0 / d) if self.device_w is None else np.asarray(self.device_w) / np.sum(self.device_w)

    def eta(self, p: Poling):
        r = casc.cascade_undepleted(p.boundaries, p.g, self.q_a, self.q_b, self.kappa_a, self.kappa_3, 1.0)
        return r["P_th"]

    def evaluate(self, p: Poling) -> float:
        return float(np.sum(self.device_w * self.eta(p)))


def thg_robust_target(family, lam0, n=11) -> THGTarget:
    df, dl, w = device_nodes(family.spread, n, family.has_load)
    dk = family.evaluate(lambda m: m(np.array([lam0]))[:, 0], df, dl)        # (D, 2)
    c = family.nominal.coupling(lam0)
    return THGTarget(dk[:, 0], dk[:, 1], float(c["kappa_a"]), float(c["kappa_3"]), w)


def evolve_thg(target: THGTarget, L, ka, kb, maxiter=12, popsize=8, seed=0, fab=DEFAULT_FAB,
               bounds=((0.5, 1.0), (0.0, 0.5), (0.0, math.pi), (0.0, 0.03), (0.0, 0.03))):
    """Differential evolution over (w_start, w_end, ψ, span_a, span_b) of the blended chirped superlattice."""
    def build(x):
        return thg_mask(L, ka, kb, *x, fab=fab)

    res = differential_evolution(lambda x: -target.evaluate(build(x)), list(bounds), maxiter=maxiter, popsize=popsize,
                                 seed=seed, polish=False, tol=1e-6)
    return {"x": res.x, "poling": build(res.x), "objective": -res.fun, "nfev": res.nfev}


def core_flatness(p: Poling, dk_of_lambda, lam0, fwhm) -> dict:
    """Mean |G|² and ripple over the central 80 % of the band (exact spectrum)."""
    lam = lam0 + np.linspace(-0.4, 0.4, 801) * fwhm
    G2 = np.abs(fast.spectrum(p, np.abs(dk_of_lambda(lam)))[0]) ** 2
    return band_stats(lam, G2, lam[0], lam[-1])
