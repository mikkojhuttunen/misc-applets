"""Milestone 6: objectives, IFTA, wall-gradient, evolution, robust SHG and THG designs."""
import math

import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import dispersion as D
from chi2sim import ensemble as E
from chi2sim import fast
from chi2sim import optimize as O
from chi2sim import poling as P
from chi2sim import synthesis as S
from chi2sim import thg as T
from chi2sim import wafer as Wf
from chi2sim import waveguide as W
from engines.cwe_shg import engine as cwe

DKF = D.LinearIndexSHG()
LAM0 = 1.55e-6


def _band(fw):
    b = S.band_to_dk(DKF, LAM0, fw)
    klo, khi = sorted((abs(b.k_lo), abs(b.k_hi)))
    return abs(b.k0), klo, khi


def test_target_kinds_order():
    eta = np.array([[1.0, 1.2], [0.1, 0.1], [2.0, 2.2]])
    t = lambda k, **kw: O.Target(np.ones((3, 2)), kind=k, **kw).score_eta(eta)
    mean = t("mean")
    assert mean == pytest.approx(eta.mean())
    assert t("geo", floor=0.0) < mean and t("min") < mean
    assert t("geo", floor=0.0) == pytest.approx(math.exp(np.mean(np.log(eta.mean(axis=1)))))
    y = t("yield", ref=1.0)
    assert 0 < y < 1
    assert t("flat", alpha=1.0) < mean


def test_device_nodes():
    sp = Wf.ThicknessSpread()
    f, l, w = O.device_nodes(sp, 11)
    assert w.sum() == pytest.approx(1) and np.all(np.abs(f) < 5e-9) and f.mean() == pytest.approx(0, abs=1e-20)
    f, l, w = O.device_nodes(sp, 5, with_load=True, n_load=3)
    assert f.size == 15 and w.sum() == pytest.approx(1)
    g = O.device_nodes(Wf.ThicknessSpread(dist="gaussian", film=2e-9), 9)
    assert np.sum(g[2] * g[0] ** 2) == pytest.approx((2e-9) ** 2, rel=1e-6)


def test_wall_params_roundtrip_and_constraints():
    ell = np.random.default_rng(3).uniform(0.5e-6, 3.5e-6, 100)
    p = P.Poling(*cwe.domains_from_lengths(ell))
    wp = O.WallParams.from_poling(p, 0.3e-6)
    q = wp.poling()
    assert np.allclose(q.boundaries, p.merged().boundaries, atol=1e-15)
    u = np.random.default_rng(1).normal(0, 3, wp.u.size)
    ell = wp.lengths(u)
    assert ell.min() >= 0.3e-6 and ell.sum() == pytest.approx(p.length)


def test_envelope_constant_gives_grating_of_that_strength():
    L, k0 = 400e-6, 1.5e6
    z = np.linspace(0, L, 101)
    for a in (2 / np.pi, 0.4):
        p = O.envelope_to_poling(z, np.full(z.size, a + 0j), k0, fab=None)
        assert abs(fast.spectrum(p, [k0])[0, 0]) == pytest.approx(a * L, rel=3e-3)


def test_ifta_beats_chirp_on_narrow_band():
    fw, L = 10e-9, 5e-3
    k0, klo, khi = _band(fw)
    chk = np.abs(DKF(LAM0 + np.linspace(-0.4, 0.4, 201) * fw))
    chirp = O.core_flatness(S.design(DKF, LAM0, fw, L, mode="chirp").poling, DKF, LAM0, fw)
    r = O.ifta(k0, klo, khi, L, chk, iters=300, level=0.7)
    f = O.core_flatness(r["poling"], DKF, LAM0, fw)
    assert f["ripple_db"] < chirp["ripple_db"] - 0.5 and f["mean"] > chirp["mean"]
    assert f["mean"] < S.efficiency_ceiling(L, khi - klo)["mean_G2_max"]
    assert r["history"][-1, 1] < r["history"][0, 1]
    assert r["poling"].lengths[1:-1].min() >= P.FAB_PRESETS["reproducible"]["l_min"] - 1e-15


def test_gradient_flattens_band_and_keeps_fabrication():
    fw, L = 10e-9, 5e-3
    start = S.design(DKF, LAM0, fw, L, mode="chirp").poling
    tg = O.band_target(DKF, LAM0 + np.linspace(-0.4, 0.4, 81) * fw, kind="flat", alpha=2.0)
    r = O.gradient_optimise(start, tg, steps=150, lr=0.02)
    assert r["objective"] > 1.1 * r["start_objective"]
    before, after = O.core_flatness(start, DKF, LAM0, fw), O.core_flatness(r["poling"], DKF, LAM0, fw)
    assert after["contrast"] < 0.5 * before["contrast"]
    g = P.FAB_PRESETS["reproducible"]["grid"]
    assert np.allclose(r["poling"].walls / g, np.round(r["poling"].walls / g), atol=1e-6)
    assert r["poling"].lengths[1:-1].min() >= P.FAB_PRESETS["reproducible"]["l_min"] - 1e-15


@pytest.fixture(scope="module")
def shg_family():
    return Wf.ThicknessFamily(lambda g: W.SHGModel(g, np.linspace(1.40e-6, 1.70e-6, 7)), W.Ridge(), film_points=5)


def test_robust_shg_worst_case_beats_heuristics(shg_family):
    nom, L = shg_family.nominal, 4e-3
    k2, k0 = float(nom.coupling(LAM0)["kappa"]) ** 2, abs(float(nom(LAM0)))
    tg = O.robust_target(shg_family, LAM0, k2, kind="geo", n=21)
    ordered = P.ordered(L, float(nom.period(LAM0)))
    c100 = S.design(nom, LAM0, 100e-9, L, mode="chirp").poling
    de = O.evolve_chirp(tg, L, k0, maxiter=10, popsize=8, bounds=[(-0.02, 0.02), (0.0, 0.1), (0.0, 0.45)])
    assert de["objective"] > tg.evaluate(c100) and de["objective"] > 10 * tg.evaluate(ordered)
    # the expected value alone is degenerate: an ordered grating already reaches the Parseval cap over the spread
    tm = O.robust_target(shg_family, LAM0, k2, kind="mean", n=81)
    assert tm.evaluate(ordered) > 0.9 * tm.evaluate(de["poling"])


def test_statistical_wall_jitter_lowers_the_robust_objective(shg_family):
    # exact jitter-ensemble mean: randomness moves power out of the coherent band into a broad weak floor
    nom, L = shg_family.nominal, 3e-3
    k2, k0 = float(nom.coupling(LAM0)["kappa"]) ** 2, abs(float(nom(LAM0)))
    tg = O.robust_target(shg_family, LAM0, k2, kind="geo", n=15)
    p = O.chirp_mask(L, k0, 0.018, 0.05)
    vals = [float(tg.score_eta(k2 * (np.abs(fast.spectrum(p, tg.q.ravel())[0]) ** 2 if s == 0 else
                                     E.jitter_mean(p, s, tg.q.ravel())).reshape(tg.q.shape)))
            for s in (0.0, 50e-9, 150e-9, 300e-9)]
    assert all(a > b for a, b in zip(vals[:-1], vals[1:]))


def test_robust_thg_geometric_mean():
    fam = Wf.ThicknessFamily(lambda g: W.THGModel(g, np.linspace(1.5e-6, 1.6e-6, 4)), W.Ridge(), film_points=3)
    nom, L = fam.nominal, 3e-3
    ka, kb = abs(float(nom.dk_a(LAM0))), abs(float(nom.dk_b(LAM0)))
    tg = O.thg_robust_target(fam, LAM0, n=11, kind="geo")
    base = tg.evaluate(T.two_section(nom, LAM0, L))
    de = O.evolve_thg(tg, L, ka, kb, maxiter=5, popsize=5, seed=2,
                      bounds=((0.4, 0.9), (0.1, 0.6), (0.0, np.pi), (0.005, 0.03), (0.005, 0.03)))
    assert de["objective"] > 2 * base
    assert tg.eta(de["poling"]).min() > 10 * tg.eta(T.two_section(nom, LAM0, L)).min()
