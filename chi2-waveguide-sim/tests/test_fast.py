import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import fast as F
from chi2sim import poling as P
from engines.cwe_shg import engine as cwe

K0 = 4 * np.pi * (2.1924 - 2.0272) / 1.55e-6
PER = 2 * np.pi / K0


def _structures():
    rng = np.random.default_rng(0)
    d = P.LengthDistribution("gaussian", PER / 2, 0.2 * PER, l_min=0.3e-6)
    return [P.ordered(300e-6, PER), P.random_domains(250e-6, d, rng), P.quasi_random("rudin_shapiro", PER / 2, 200e-6),
            P.Poling(np.array([0, 1e-6, 3e-6, 4e-6]), np.array([0.3, -1.0, 0.7]))]


def test_batch_matches_engine_per_domain_sum():
    ps = _structures()
    q = K0 * np.linspace(0.5, 1.5, 41)
    for gain in (0.0, -40.0):
        G = F.spectrum(ps, q, gain=gain)
        for i, p in enumerate(ps):
            ref = cwe.poling_spectrum(p.boundaries, p.g, q, gain=gain)
            assert np.abs(G[i] - ref).max() <= 1e-9 * np.abs(ref).max()


def test_zero_and_tiny_delta_k():
    p = _structures()[1]
    G = F.spectrum([p], [0.0, 1e-3, 1.0])
    ref = cwe.poling_spectrum(p.boundaries, p.g, [0.0, 1e-3, 1.0])
    assert G[0] == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_shg_power_matches_engine_with_loss():
    p = P.ordered(400e-6, PER)
    a1, a2 = 30.0, 70.0
    G = F.spectrum([p], [K0, 1.002 * K0], gain=F.loss_gain(a1, a2))
    P_sh = F.shg_power(G, 900.0, 0.01, p.length, a1, a2)
    ref = cwe.shg_undepleted(p.boundaries, p.g, [K0, 1.002 * K0], 900.0, 0.01, a1, a2)["P_sh"]
    assert P_sh[0] == pytest.approx(ref, rel=1e-9)


def test_jax_backend_and_gradient():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    ps = _structures()
    q = K0 * np.linspace(0.8, 1.2, 17)
    assert np.allclose(F.spectrum(ps, q, backend="jax"), F.spectrum(ps, q), rtol=1e-10, atol=1e-16)
    p = ps[1]
    w, c = jnp.asarray(p.walls), jnp.asarray(p.jumps())

    def obj(w):
        return jnp.sum(jnp.abs(F.jax_spectrum(w, c, p.g[0], p.g[-1], p.length, q)) ** 2)

    gr = np.asarray(jax.grad(obj)(w))
    h = 1e-12
    for j in (0, 10, 50):
        e = np.zeros(w.size)
        e[j] = h
        fd = (float(obj(w + e)) - float(obj(w - e))) / (2 * h)
        assert gr[j] == pytest.approx(fd, rel=1e-4)
