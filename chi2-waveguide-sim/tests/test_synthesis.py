"""Plan section 7 test 9 and section 9 synthesis tests (Parseval, stationary phase, PWM, ceiling, dispersion)."""
import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import dispersion as D
from chi2sim import fast as F
from chi2sim import poling as P
from chi2sim import synthesis as S
from chi2sim.metrics import band_stats, fwhm
from engines.phase_matching import engine as pm

DKF = D.LinearIndexSHG()
K0 = float(DKF(1.55e-6))
PER = 2 * np.pi / K0


def test_parseval_random_binary():
    rng = np.random.default_rng(0)
    d = P.LengthDistribution("uniform", 2e-6, 0.5e-6, l_min=1e-6)
    p = P.random_domains(40e-6, d, rng)
    q = np.linspace(-2e9, 2e9, 2_000_001)            # dq ≪ 2π/L, Q ℓ ≈ 4000
    G2 = np.abs(F.spectrum([p], q, chunk_elems=2**22)[0]) ** 2
    integral = np.sum(G2) * (q[1] - q[0])
    assert integral == pytest.approx(2 * np.pi * p.length, rel=2e-3)


def test_ordered_peak_and_pwm_orders():
    L = 200 * PER
    z = np.linspace(0, L, 400001)
    for D_ in (0.5, 0.3):
        p = P.pwm(z, 2 * np.pi * z / PER, D_)
        G = np.abs(F.spectrum([p], [K0, 2 * K0, 3 * K0])[0])
        for m in (1, 2, 3):
            assert G[m - 1] == pytest.approx(pm.grating_strength(m, D_) * L, rel=2e-3, abs=1e-4 * L)
    assert S.efficiency_ceiling(L, 1.0)["ordered_peak_G2"] == pytest.approx((2 / np.pi * L) ** 2)
    po = S.parasitic_orders(0.9 * K0, 1.1 * K0, 0.5, lambda x: np.abs(DKF(x)), np.linspace(0.5e-6, 2e-6, 101))
    assert po[0]["rel_amplitude"] == pytest.approx(0, abs=1e-12) and po[1]["rel_amplitude"] == pytest.approx(1 / 3)


def test_linear_chirp_bandwidth_and_stationary_phase_level():
    # test 9 (Arbore et al. 1997): bandwidth ≈ chirp range; level 2π a² L / B = the ceiling
    L = 8e-3
    k1, k2 = 0.95 * K0, 1.05 * K0
    p = P.chirped(L, k1, k2)
    q = np.linspace(0.9 * K0, 1.1 * K0, 6001)
    G2 = np.abs(F.spectrum([p], q)[0]) ** 2
    level = 2 * np.pi * S.A1**2 * L / (k2 - k1)
    # the half-power points sit inside the chirp range by a Fresnel edge of order √K' = √(B/L)
    edge = np.sqrt((k2 - k1) / L)
    assert 0 < (k2 - k1) - fwhm(q, G2) < 3 * edge
    mid = band_stats(q, G2, K0 - 0.3 * (k2 - k1), K0 + 0.3 * (k2 - k1))
    assert mid["mean"] == pytest.approx(level, rel=0.03)
    assert level == pytest.approx(S.efficiency_ceiling(L, k2 - k1)["mean_G2_max"], rel=1e-12)
    # the ripple (Fresnel-like) shrinks as the chirp parameter L·B_k grows
    p2 = P.chirped(4 * L, k1, k2)
    G2b = np.abs(F.spectrum([p2], q)[0]) ** 2
    mid2 = band_stats(q, G2b, K0 - 0.3 * (k2 - k1), K0 + 0.3 * (k2 - k1))
    assert mid2["contrast"] < mid["contrast"]
    assert abs(fwhm(q, G2b) - (k2 - k1)) < abs(fwhm(q, G2) - (k2 - k1))


def test_chirp_from_flat_spectrum_is_linear():
    k = np.linspace(0.97 * K0, 1.03 * K0, 101)
    c = S.chirp_from_spectrum(k, np.ones_like(k), length=5e-3)
    assert c["length"] == pytest.approx(5e-3)
    assert np.allclose(np.diff(c["z"]), np.diff(c["z"])[0])
    assert c["level"][0] == pytest.approx(2 * np.pi * S.A1**2 * 5e-3 / (k[-1] - k[0]), rel=1e-12)
    p = S.poling_from_chirp(c)
    assert p.length == pytest.approx(5e-3)


def test_band_to_dk_nonlinear_mapping():
    md = D.MaterialSHG("ln_e_cong")
    b = S.band_to_dk(md, 1.55e-6, 200e-9)
    ends = md(np.array([1.45e-6, 1.65e-6]))
    assert sorted([b.k_lo, b.k_hi]) == pytest.approx(sorted(ends), rel=1e-9)
    # nonuniform map: the two half-bands differ in Δk
    assert abs((b.k0 - b.k_lo) - (b.k_hi - b.k0)) > 0.01 * b.b_k
    lin = S.band_to_dk(DKF, 1.55e-6, 1e-9)
    assert lin.b_k == pytest.approx(abs(lin.slope) * 1e-9, rel=1e-3)


@pytest.mark.parametrize("fw", [10e-9, 50e-9])
def test_chirp_design_meets_fwhm_fab_and_ceiling(fw):
    d = S.design(DKF, 1.55e-6, fw, 5e-3, mode="chirp")
    r = d.report
    assert r["fwhm"] == pytest.approx(fw, rel=0.02)
    assert r["fraction_of_ceiling"] < 1.0
    assert r["min_domain"] >= P.FAB_PRESETS["reproducible"]["l_min"] - 1e-15
    g = P.FAB_PRESETS["reproducible"]["grid"]
    assert np.allclose(d.poling.walls / g, np.round(d.poling.walls / g), atol=1e-6)


def test_random_design_ensemble_fwhm_and_speckle():
    d = S.design(DKF, 1.55e-6, 50e-9, 5e-3, mode="random", n_real=48)
    r = d.report
    assert r["fwhm"] == pytest.approx(50e-9, rel=0.01)
    assert d.lam[np.argmax(d.G2)] == pytest.approx(1.55e-6, abs=2e-9)
    assert r["mc_band_mean"] == pytest.approx(r["band_stats"]["mean"], rel=0.1)
    assert 0.7 < r["speckle_contrast"] < 1.2
    assert r["fraction_of_ceiling"] < 1.0


def test_hybrid_design_reports_jitter_ensemble():
    d = S.design(DKF, 1.55e-6, 20e-9, 4e-3, mode="hybrid", sigma_wall=100e-9, n_real=24)
    r = d.report
    assert np.allclose(d.G2, r["jitter_mean_exact"])
    assert r["mc_band_mean"] == pytest.approx(r["band_stats"]["mean"], rel=0.1)
    assert r["speckle_contrast"] < 0.5
