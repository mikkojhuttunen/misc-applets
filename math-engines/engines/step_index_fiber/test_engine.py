import numpy as np
import pytest

from engines.step_index_fiber import engine as sif


def test_gloge_approximation():
    # Gloge (1971): b ≈ (1.1428 - 0.996/V)^2 within about 0.2 % for 1.5 < V < 2.5
    for a in (3.3e-6, 3.8e-6, 4.4e-6):
        r = sif.lp01(1.55e-6, a, 0.005)
        assert 1.5 < r["V"] < 2.5
        assert r["b"] == pytest.approx((1.1428 - 0.996 / r["V"]) ** 2, rel=5e-3)


def test_neff_between_cladding_and_core():
    r = sif.lp01(1.55e-6, 4.1e-6, 0.005)
    assert r["n_clad"] < r["neff"] < r["n_clad"] + 0.005


def test_marcuse_hand_value_at_single_mode_cutoff():
    # w/a at V = 2.405: 0.65 + 1.619/2.405^1.5 + 2.879/2.405^6 = 1.0990
    lam, dn = 1.55e-6, 0.005
    ncl = sif.lp01(lam, 4e-6, dn)["n_clad"]
    a = 2.405 * lam / (2 * np.pi * np.sqrt((ncl + dn) ** 2 - ncl**2))
    r = sif.lp01(lam, a, dn)
    assert r["mode_radius"] / a == pytest.approx(1.0990, abs=2e-4)


C0 = 299_792_458.0


def test_smf_dispersion_matches_datasheet():
    # SMF-28-like fiber: D(1550) ~ 17 ps/(nm km) -> beta2 = -D lambda^2 / (2 pi c) ~ -21.7 ps^2/km (datasheet)
    a, dn, lam = 4.1e-6, 0.005, 1.55e-6
    w, h = 2 * np.pi * C0 / lam, 1e13
    k = lambda x: sif.propagation_constant(x, a, dn)
    beta2 = (k(w + h) - 2 * k(w) + k(w - h)) / h**2
    assert beta2 == pytest.approx(-17e-6 * lam**2 / (2 * np.pi * C0), rel=0.06)


def test_chi2_and_chi3_share_signal_idler_pair():
    # 2 omega(1064 nm) = omega(532 nm): same idler and GVM; dk_rel has opposite sign by convention
    a, dn, O = 4.1e-6, 0.005, 2 * np.pi * 1e11
    r2 = sif.phase_mismatch(1.55e-6, 0.532e-6, a, dn, "chi2", detuning=O)
    r3 = sif.phase_mismatch(1.55e-6, 1.064e-6, a, dn, "chi3", detuning=O)
    assert r2["wavelength_idler"] == pytest.approx(r3["wavelength_idler"], rel=1e-12)
    assert r2["gvm"] == pytest.approx(r3["gvm"], rel=1e-6)
    assert r2["dk_rel"] == pytest.approx(-r3["dk_rel"], rel=1e-6)


def test_dk_rel_linear_term_is_gvm():
    # Taylor expansion: dk_rel ~ GVM * Omega for small detuning (chi2 sign)
    O = 2 * np.pi * 1e9
    r = sif.phase_mismatch(1.55e-6, 0.532e-6, 4.1e-6, 0.005, "chi2", detuning=O)
    assert r["dk_rel"] == pytest.approx(r["gvm"] * O, rel=1e-3)


def test_qpm_period_against_phase_matching_engine():
    # independent route: phase_matching.three_wave_mismatch with the LP01 indices gives the same period
    from engines.phase_matching import engine as pm
    a, dn = 4.1e-6, 0.005
    li = 1 / (1 / 0.532e-6 - 1 / 1.55e-6)
    n = lambda lam: float(sif.lp01(lam, a, dn)["neff"])
    ref = pm.three_wave_mismatch(1.55e-6, li, n(1.55e-6), n(li), n(0.532e-6))["period_required"]
    assert sif.phase_mismatch(1.55e-6, 0.532e-6, a, dn, "chi2")["qpm_period"] == pytest.approx(float(ref), rel=1e-9)


def test_vectorised_detuning_and_bad_process():
    r = sif.phase_mismatch(1.55e-6, 0.532e-6, 4.1e-6, 0.005, detuning=np.linspace(-1e13, 1e13, 5))
    assert r["dk_rel"].shape == (5,) and r["dk_rel"][2] == 0
    with pytest.raises(ValueError):
        sif.phase_mismatch(1.55e-6, 0.532e-6, 4.1e-6, 0.005, "chi4")
