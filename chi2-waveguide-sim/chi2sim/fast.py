"""Batched undepleted solver: G(Δk) = ∫0^L g(z) e^{iΔk z + γz} dz for many structures and wavelengths.

For piecewise-constant g the integral is a sum over walls,

    G = [ g_last e^{sL} - g_first + Σ_j c_j e^{s z_j} ] / s,   s = iΔk + γ,   c_j = g_{j-1} - g_j,

so one structure costs O(walls) per wavelength. Structures with different wall counts are padded with
c_j = 0. SH power in the undepleted limit (math-engines cwe_shg normalisation):

    P_SH = κ² P² e^{-α2 L} |G|²,   γ = α2/2 - α1.

Backends: "numpy" (chunked) and "jax" (jit, float64, differentiable in the wall positions: use
`jax_spectrum` directly inside jax.grad). Δk must be uniform along z here; for Δk(z) use cwe_shg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .poling import Poling

_SMALL = 1e-6          # |s| L below this: G = Σ g ℓ (wall form loses precision)


@dataclass(frozen=True)
class Batch:
    walls: np.ndarray      # (R, W) padded wall positions
    jumps: np.ndarray      # (R, W) c_j, zero where padded
    g_first: np.ndarray    # (R,)
    g_last: np.ndarray     # (R,)
    length: np.ndarray     # (R,)
    dc: np.ndarray         # (R,) Σ g_k ℓ_k

    @property
    def size(self) -> int:
        return int(self.walls.shape[0])


def batch(polings: Sequence[Poling] | Poling) -> Batch:
    if isinstance(polings, Poling):
        polings = [polings]
    if isinstance(polings, Batch):
        return polings
    r = len(polings)
    w = max(p.n_domains - 1 for p in polings)
    walls = np.zeros((r, max(w, 1)))
    jumps = np.zeros((r, max(w, 1)))
    for i, p in enumerate(polings):
        n = p.n_domains - 1
        walls[i, :n] = p.walls
        jumps[i, :n] = p.jumps()
    return Batch(walls, jumps,
                 np.array([p.g[0] for p in polings]), np.array([p.g[-1] for p in polings]),
                 np.array([p.length for p in polings]), np.array([float(p.g @ p.lengths) for p in polings]))


def spectrum(polings, delta_k, gain=0.0, backend="numpy", chunk_elems=2**23) -> np.ndarray:
    """G for every structure (rows) and Δk (columns): complex array (R, M)."""
    b = batch(polings)
    q = np.atleast_1d(np.asarray(delta_k, dtype=float))
    if backend == "jax":
        return np.asarray(_jax_batched()(b.walls, b.jumps, b.g_first, b.g_last, b.length, b.dc, q, float(gain)))
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    s = 1j * q + gain
    out = np.empty((b.size, q.size), dtype=complex)
    rows = max(1, chunk_elems // max(1, q.size * b.walls.shape[1]))
    small = np.abs(s)[None, :] * b.length[:, None] < _SMALL
    s_safe = np.where(np.abs(s) == 0, 1.0, s)
    for i in range(0, b.size, rows):
        sl = slice(i, i + rows)
        e = np.exp(s[None, :, None] * b.walls[sl, None, :])                     # (r, M, W)
        acc = np.einsum("rmw,rw->rm", e, b.jumps[sl])
        acc += b.g_last[sl, None] * np.exp(s[None, :] * b.length[sl, None]) - b.g_first[sl, None]
        out[sl] = acc / s_safe[None, :]
    if np.any(small):
        out = np.where(small, b.dc[:, None] + 0j, out)
    return out


def shg_power(G, kappa, power, length, alpha_pump=0.0, alpha_sh=0.0) -> np.ndarray:
    """Undepleted SH power from G computed with gain = α2/2 - α1."""
    G = np.asarray(G)
    att = np.exp(-alpha_sh * np.asarray(length, dtype=float))
    if att.ndim == 1 and G.ndim == 2:
        att = att[:, None]
    return kappa**2 * power**2 * att * np.abs(G) ** 2


def loss_gain(alpha_pump=0.0, alpha_sh=0.0) -> float:
    return alpha_sh / 2 - alpha_pump


# ----------------------------------------------------------------------------- JAX backend

def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def jax_spectrum(walls, jumps, g_first, g_last, length, delta_k, gain=0.0):
    """G(Δk) of one structure, pure JAX (differentiable in walls, jumps and Δk). Assumes |s| L > 1e-6."""
    _, jnp = _jax()
    s = 1j * jnp.asarray(delta_k) + gain
    acc = jnp.exp(s[:, None] * walls[None, :]) @ jumps.astype(complex)
    return (acc + g_last * jnp.exp(s * length) - g_first) / s


_JAX_CACHE = {}


def _jax_batched():
    if "f" not in _JAX_CACHE:
        jax, jnp = _jax()

        def one(w, c, gf, gl, L, dc, q, gain):
            s = 1j * q + gain
            small = jnp.abs(s) * L < _SMALL
            ss = jnp.where(jnp.abs(s) == 0, 1.0, s)
            g = (jnp.exp(ss[:, None] * w[None, :]) @ c.astype(complex) + gl * jnp.exp(ss * L) - gf) / ss
            return jnp.where(small, dc + 0j, g)

        f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None)))
        _JAX_CACHE["f"] = f
    return _JAX_CACHE["f"]
