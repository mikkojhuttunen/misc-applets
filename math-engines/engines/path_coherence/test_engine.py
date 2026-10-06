import numpy as np
import pytest

from engines.billiard_cell import engine as bc
from engines.path_coherence import engine as pc

C = 299792458.0


def _paths(cells, n_g=3.0):
    L = np.concatenate([l for _, l in cells])
    W = np.concatenate([p for p, _ in cells])
    return pc.Paths(cells, 1.55e-6, 2.8, n_g, 30e-6, 0.1, 1.0, L.size, L, W)


def test_equal_paths_contrast():
    p = _paths([(np.ones(10), np.linspace(0.1, 0.2, 10))])
    assert pc.contrast(p, 0.0, spatial=False) == pytest.approx(np.sqrt(1 - 1 / 10))
    assert pc.contrast(p, 0.0) == pytest.approx(pc.random_phase_contrast(10, 1)["contrast_single"])
    p4 = _paths([(np.ones(10), np.linspace(0.1, 0.2, 10) + k) for k in range(4)])
    assert pc.contrast(p4, 0.0) == pytest.approx(pc.random_phase_contrast(10, 4)["contrast_sum"])


def test_linewidth_washes_out_contrast():
    p = _paths([(np.ones(5), np.array([0.0, 0.3, 0.6, 0.9, 1.2]))])
    assert pc.contrast(p, 1e10) < 1e-3 * pc.contrast(p, 0.0)


def test_two_path_autocovariance_is_a_cosine():
    dL, n_g = 0.05, 3.0
    p = _paths([(np.array([1.0, 0.5]), np.array([0.1, 0.1 + dL]))], n_g)
    dnu = np.linspace(0, 5e9, 2001)
    g = pc.autocovariance(p, dnu)
    assert np.allclose(g, np.cos(2 * np.pi * dnu * n_g * dL / C), atol=1e-12)
    assert pc.correlation_width(dnu, g) == pytest.approx(2 * C / (6 * n_g * dL), rel=1e-2)


def test_chunked_pair_sums_match_brute_force():
    rng = np.random.default_rng(0)
    P, L = rng.random(2500), rng.random(2500)
    kern = lambda d: np.exp(-d)
    PP = P[:, None] * P[None, :]
    np.fill_diagonal(PP, 0)
    assert pc._pair_sum(P, L, kern) == pytest.approx((PP * kern(np.abs(L[:, None] - L[None, :]))).sum(), rel=1e-12)


def test_simulated_spectrum_statistics():
    p = _paths([(np.ones(1), np.array([0.3]))])
    assert np.allclose(pc.simulate_spectrum(p, np.linspace(-1e9, 1e9, 11), 1.95e14), 1.0)
    p = _paths([(np.ones(30), np.linspace(0.1, 1.0, 30))])
    I = pc.simulate_spectrum(p, np.linspace(-2e10, 2e10, 40001), 1.95e14, seed=3)
    assert I.mean() == pytest.approx(1.0, abs=0.05)


def test_prepare_from_ray_table():
    cell = bc.SegmentedCell(5e-3, 24, curvature=20.0)
    tab = bc.trace_rays(cell, 30e-6, 1500, 2000, 0.02, seed=5, max_path=5.0, theta_c=np.pi / 2 - 7 * np.pi / 24)
    R = lambda s: np.full_like(s, 0.995)
    p = pc.prepare(tab, R, 2.0, 0.5, 1.55e-6, 2.8, 3.5)
    res = bc.evaluate(tab, R, 2.0, 0.5)
    assert p.P_tot == pytest.approx(res.W.sum()) and p.L_mean == pytest.approx(res.L_mean)
    pruned = pc.prepare(tab, R, 2.0, 0.5, 1.55e-6, 2.8, 3.5, min_cell_power=0.05)
    assert 0 < len(pruned.cells) < len(p.cells)
    assert all(P.sum() >= 0.05 * p.P_tot for P, _ in pruned.cells)


def test_result_front_ends():
    t = pc.thermal_shift(1.55e-6, 1.5e-4, 0.01, 3.5)
    assert t["delta_nu"] == pytest.approx(C / 1.55e-6 * 1.5e-4 * 0.01 / 3.5)
    v = pc.pair_coherence(1e6, 0.1, 3.5)
    assert v["visibility"] == pytest.approx(np.exp(-0.1 / v["coherence_length"]))
    assert v["visibility"] == pytest.approx(np.exp(-np.pi * 1e6 * 3.5 * 0.1 / C))
    assert v["variance_factor"] == pytest.approx(v["visibility"] ** 2)
