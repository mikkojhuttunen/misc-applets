import numpy as np
import pytest

import chi2sim  # noqa: F401  (puts math-engines on the path)
from chi2sim import poling as P
from engines.cwe_shg import engine as cwe

PER = 4.69e-6


def test_ordered_matches_engine_and_wall_list():
    p = P.ordered(100e-6, PER, 0.4, offset=1e-6)
    b, g = cwe.periodic_domains(100e-6, PER, 0.4, 1e-6)
    assert np.array_equal(p.boundaries, b) and np.array_equal(p.g, g)
    w = p.wall_list()
    q = P.from_walls(w["walls"], w["length"], w["first_sign"])
    assert np.allclose(q.boundaries, p.boundaries) and np.array_equal(q.g, p.g)


def test_merged_removes_walls_without_sign_change():
    p = P.Poling(np.array([0, 1, 2, 3, 4.0]), np.array([1, 1, -1, -1.0]))
    m = p.merged()
    assert list(m.boundaries) == [0, 2, 4] and list(m.g) == [1, -1]


def test_jitter_zero_is_identity_and_keeps_length():
    p = P.ordered(200e-6, PER)
    assert np.allclose(P.jittered(p, 0.0, 1).boundaries, p.boundaries)
    j = P.jittered(p, 50e-9, 2)
    assert j.length == p.length and j.n_domains == p.n_domains
    assert np.std(j.walls - p.walls) == pytest.approx(50e-9, rel=0.15)


@pytest.mark.parametrize("kind,std", [("gaussian", 0.3e-6), ("uniform", 0.3e-6), ("gamma", 0.5e-6),
                                      ("exponential", 0.0), ("lognormal", 0.4e-6)])
def test_length_distribution_char_matches_samples(kind, std):
    d = P.LengthDistribution(kind, 2.3e-6, std, l_min=0.3e-6)
    x = d.sample(200000, 5)
    assert x.min() >= 0.3e-6 - 1e-15
    assert x.mean() == pytest.approx(d.actual_mean(), rel=1e-2)
    q = np.array([0.2, 1.0, 1.36]) * 1e6
    mc = np.exp(1j * np.outer(q, x)).mean(axis=1)
    assert np.abs(d.char(q) - mc).max() < 0.01


def test_truncated_gaussian_char_uses_the_truncated_pdf():
    d = P.LengthDistribution("gaussian", 0.5e-6, 0.4e-6, l_min=0.3e-6)
    x = d.sample(200000, 1)
    assert x.min() >= 0.3e-6
    q = np.array([1e6, 4e6])
    assert np.abs(d.char(q) - np.exp(1j * np.outer(q, x)).mean(axis=1)).max() < 0.01


def test_random_domains_length_and_correlation():
    d = P.LengthDistribution("gaussian", 2.3e-6, 0.5e-6, l_min=0.3e-6)
    p = P.random_domains(5e-3, d, 3)
    assert p.length == 5e-3 and p.is_binary
    assert np.all(p.lengths[:-1] >= 0.3e-6)
    assert np.all(p.g[1:] == -p.g[:-1])
    pc = P.random_domains(5e-3, d, 3, correlation=0.8)
    ell = pc.lengths[:-1]
    assert np.corrcoef(ell[1:], ell[:-1])[0, 1] > 0.6
    assert np.corrcoef(p.lengths[:-1][1:], p.lengths[:-1][:-1])[0, 1] < 0.1


def test_sequences_first_terms():
    assert list(P.sequence("thue_morse", 8)) == [1, -1, -1, 1, -1, 1, 1, -1]
    assert list(P.sequence("rudin_shapiro", 8)) == [1, 1, 1, -1, 1, 1, -1, 1]
    assert list(P.sequence("fibonacci", 8)) == [1, -1, 1, 1, -1, 1, -1, 1]
    b = P.sequence("barker13", 13)
    ac = [abs(np.dot(b[:13 - k], b[k:])) for k in range(1, 13)]
    assert max(ac) <= 1
    s = P.sequence("prbs", 2 * 32767, seed=11)
    assert np.array_equal(s[:32767], s[32767:])
    assert np.count_nonzero(s[:32767] > 0) == 16384


def test_quasi_random_domains_are_unit_multiples():
    p = P.quasi_random("thue_morse", 2e-6, 101e-6)
    ell = p.lengths[:-1] / 2e-6
    assert np.allclose(ell, np.round(ell)) and p.length == 101e-6
    assert np.all(p.g[1:] != p.g[:-1])


def test_fabricate_grid_and_minimum_domain():
    rng = np.random.default_rng(4)
    d = P.LengthDistribution("exponential", 1.0e-6, l_min=0.0)
    p = P.random_domains(300e-6, d, rng)
    assert p.lengths.min() < 0.3e-6
    f = P.fabricate(p, l_min=0.3e-6, grid=10e-9)
    assert f.lengths.min() >= 0.3e-6 - 1e-15
    assert np.allclose(f.walls / 10e-9, np.round(f.walls / 10e-9), atol=1e-6)
    assert np.all(f.g[1:] != f.g[:-1])
    g = P.fabricate(f, l_min=0.3e-6, grid=10e-9)
    assert np.allclose(g.boundaries, f.boundaries)
    assert P.fabricate(p).lengths.min() >= P.FAB_PRESETS[P.DEFAULT_FAB]["l_min"] - 1e-15


def test_chirped_local_period_and_pwm_duty():
    k1, k2 = 2 * np.pi / 5e-6, 2 * np.pi / 4e-6
    p = P.chirped(2e-3, k1, k2)
    per = p.lengths[1:-1][:-1] + p.lengths[1:-1][1:]
    assert per[:20].mean() == pytest.approx(5e-6, rel=0.01)
    assert per[-20:].mean() == pytest.approx(4e-6, rel=0.01)
    z = np.linspace(0, 100 * PER, 200001)
    q = P.pwm(z, 2 * np.pi * z / PER, 0.3)
    ell = q.lengths[1:-1]
    assert ell[q.g[1:-1] > 0].mean() == pytest.approx(0.3 * PER, rel=1e-3)


def test_window_and_duty_from_amplitude():
    z = np.linspace(0, 1, 1001)
    for kind in ("tanh", "hann", "gaussian", "none"):
        w = P.window(z, 1.0, kind, 0.1)
        assert w.max() <= 1 and w.min() > 0 and w[500] == pytest.approx(1.0, abs=1e-3 if kind != "gaussian" else 1)
    d = P.duty_from_amplitude(np.array([1.0, 0.5]))
    assert np.sin(np.pi * d) == pytest.approx([1.0, 0.5])
