"""Milestone 4: deposited loading strips, three-wave (OPA/DFG) mode model, wafer thickness spread."""
import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import poling as P
from chi2sim import synthesis as S
from chi2sim import wafer as Wf
from chi2sim import waveguide as W
from engines.cwe_dfg import engine as dfg

LAMS = np.linspace(1.40e-6, 1.70e-6, 6)


@pytest.fixture(scope="module")
def family():
    return Wf.ThicknessFamily(lambda g: W.SHGModel(g, LAMS), W.Ridge(), film_points=3)


def test_lagrange_weights_are_exact_for_polynomials():
    nodes = np.array([-1.0, -0.3, 0.4, 1.0])
    t = np.linspace(-1, 1, 9)
    w = Wf.lagrange_weights(nodes, t)
    f = lambda x: 2 - x + 3 * x**2 - 0.5 * x**3
    assert w @ f(nodes) == pytest.approx(f(t), rel=1e-12)
    assert w.sum(axis=1) == pytest.approx(np.ones(9))


def test_with_thickness_etch_modes():
    g = W.Ridge()
    a = W.with_thickness(g, 5e-9)
    assert a.film == pytest.approx(605e-9) and a.etch == g.etch and a.slab == pytest.approx(g.slab + 5e-9)
    b = W.with_thickness(g, 5e-9, etch="slab")
    assert b.slab == pytest.approx(g.slab)
    full = W.with_thickness(W.Ridge(etch=0.6e-6), -5e-9)
    assert full.etch == full.film
    ld = W.with_thickness(W.Ridge(etch=0.0, load="sinx", load_thickness=0.2e-6), 2e-9, -3e-9)
    assert ld.load_thickness == pytest.approx(0.197e-6) and ld.film == pytest.approx(0.602e-6)


def test_loading_strip_raises_index_and_matches_eim():
    bare = W.Ridge(film=0.3e-6, etch=0.0, width=1.2e-6)
    loaded = W.Ridge(film=0.3e-6, etch=0.0, width=1.2e-6, load="sinx", load_thickness=0.3e-6)
    with pytest.raises(RuntimeError):
        W.fundamental(bare, 1.55e-6)                       # unetched, unloaded film: no lateral guiding
    m = W.fundamental(loaded, 1.55e-6)
    assert m.neff > W._slab_neff(W.indices(loaded, 1.55e-6), 0.3e-6, 1.55e-6, "TE")
    assert m.neff == pytest.approx(W.eim(loaded, 1.55e-6), abs=0.015)
    sic = W.fundamental(W.Ridge(film=0.3e-6, etch=0.0, width=1.2e-6, load="sic", load_thickness=0.3e-6), 1.55e-6)
    assert sic.neff > m.neff                                # n(SiC) 2.55 > n(SiNx) 1.95
    with pytest.raises(ValueError, match="load"):
        W.Ridge(load="unobtainium", load_thickness=1e-7)


def test_family_interpolation_matches_direct_solve(family):
    for d in (2.37e-9, -4.1e-9):
        direct = W.SHGModel(W.with_thickness(W.Ridge(), d), LAMS)
        lam = np.array([1.52e-6, 1.55e-6, 1.58e-6])
        # 3-node (quadratic) interpolation over ±5 nm: error well below 0.2 nm of phase-matching wavelength
        slope = (direct(lam + 1e-10) - direct(lam - 1e-10)) / 2e-10          # dΔk/dλ
        err_nm = np.abs((family.delta_k(lam, [d])[0] - direct(lam)) / slope) * 1e9
        assert err_nm.max() < 0.2


def test_zero_spread_reproduces_nominal():
    fam = Wf.ThicknessFamily(lambda g: W.SHGModel(g, LAMS), W.Ridge(), Wf.ThicknessSpread(film=0.0, load=0.0))
    p = P.ordered(2e-3, float(fam.nominal.period(1.55e-6)))
    r = Wf.shg_devices(fam, p, np.linspace(1.52e-6, 1.58e-6, 301), 1.55e-6, n_devices=5)
    assert np.allclose(r["eta"], r["nominal"][None, :], rtol=1e-12)


def test_device_peak_shift_matches_slope_ratio(family):
    lam0 = 1.55e-6
    p = P.ordered(3e-3, float(family.nominal.period(lam0)))
    r = Wf.shg_devices(family, p, np.linspace(1.49e-6, 1.61e-6, 1201), lam0, n_devices=60, seed=3)
    h = 1e-9
    ddk_dt = (family.delta_k(lam0, [h])[0] - family.delta_k(lam0, [-h])[0]) / (2 * h)
    ddk_dl = (family.nominal(lam0 + 1e-10) - family.nominal(lam0 - 1e-10)) / 2e-10
    assert r["dpeak_dfilm"] == pytest.approx(-ddk_dt / ddk_dl, rel=0.05)
    s = family.sensitivity(lam0, 3e-3)
    assert s["dpeak_dfilm"] == pytest.approx(-ddk_dt / ddk_dl, rel=1e-3)
    assert s["tolerance_film"] == pytest.approx(2 * 1.391557 / (3e-3 * abs(ddk_dt)), rel=1e-3)
    assert np.all(np.abs(r["d_film"]) <= 5e-9)


def test_broadband_design_yields_more_than_ordered(family):
    lam0, L = 1.55e-6, 3e-3
    lam = np.linspace(1.5e-6, 1.6e-6, 1001)
    ordered = P.ordered(L, float(family.nominal.period(lam0)))
    chirp = S.design(family.nominal, lam0, 60e-9, L, mode="chirp").poling
    ro = Wf.shg_devices(family, ordered, lam, lam0, n_devices=60, seed=1)
    rc = Wf.shg_devices(family, chirp, lam, lam0, n_devices=60, seed=1)
    yo = Wf.yield_fraction(ro["at_target"], ro["nominal_at_target"])
    yc = Wf.yield_fraction(rc["at_target"], rc["nominal_at_target"])
    assert yc > yo + 0.3


def test_opa_model_degenerate_limit_and_overlap():
    shg = W.SHGModel(W.Ridge(), np.linspace(1.5e-6, 1.6e-6, 4))
    opa = W.OPAModel(W.Ridge(), 775e-9, np.linspace(1.5e-6, 1.6e-6, 5))
    assert float(opa.period(1.55e-6)) == pytest.approx(float(shg.period(1.55e-6)), rel=2e-5)
    assert float(opa.a_eff(1.55e-6)) == pytest.approx(float(shg.a_eff(1.55e-6)), rel=1e-4)
    assert float(opa.idler(1.5e-6)) == pytest.approx(1 / (1 / 775e-9 - 1 / 1.5e-6))
    # Γ² = κ_s κ_i P with η_SFG = 4 η0(SHG) in the degenerate limit
    c = opa.coupling(1.55e-6)
    assert float(c["eta_sfg"]) == pytest.approx(4 * float(shg.coupling(1.55e-6)["eta_norm"]), rel=2e-3)


def test_opa_gain_spectrum_and_mid_ir_flag():
    opa = W.OPAModel(W.Ridge(), 532e-9, np.linspace(1.5e-6, 1.6e-6, 5))
    L = 3e-3
    p = P.ordered(L, float(opa.period(1.55e-6)))
    g = opa.gain_spectrum(p.boundaries, p.g, [1.55e-6], 0.3)[0]
    c = opa.coupling(1.55e-6)
    ref = dfg.opa_gain_uniform(2 / np.pi * float(c["gain_coeff"]) * np.sqrt(0.3), 0.0, L)
    assert 10 * np.log10(g) == pytest.approx(10 * np.log10(ref), rel=0.02)
    assert opa.notes == []
    mir = W.OPAModel(W.Ridge(film=0.9e-6, etch=0.45e-6, width=2.4e-6), 1064e-9, np.linspace(1.5e-6, 1.6e-6, 4))
    assert any("SiO2" in n for n in mir.notes)
    assert 3.2e-6 < float(mir.idler(1.55e-6)) < 3.6e-6


def test_opa_devices_spread():
    fam = Wf.ThicknessFamily(lambda g: W.OPAModel(g, 775e-9, np.linspace(1.45e-6, 1.65e-6, 6)), W.Ridge(), film_points=3)
    ls = np.linspace(1.46e-6, 1.64e-6, 91)
    p = P.ordered(2e-3, float(fam.nominal.period(1.50e-6)))
    r = Wf.opa_devices(fam, p, ls, 0.3, n_devices=20, seed=2)
    assert r["gain_db"].shape == (20, 91)
    assert r["peak_db"].max() > 1.0
    assert r["peak_shift_rms"] > 1e-9
