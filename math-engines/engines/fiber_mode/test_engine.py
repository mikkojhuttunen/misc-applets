"""Tests for fiber_mode. Each expected value names its independent source."""
import math

import numpy as np
import pytest

from engines.common import C0
from engines.fiber_mode import engine as fm


def test_silica_index_malitson_table():
    # Malitson (1965) / refractiveindex.info: n(1550 nm) = 1.44402, n(532 nm) = 1.46071
    assert fm.silica_index(1550e-9)["n"] == pytest.approx(1.44402, abs=2e-5)
    assert fm.silica_index(532e-9)["n"] == pytest.approx(1.46071, abs=2e-5)


@pytest.mark.parametrize("V", [1.6, 2.0, 2.3])
def test_lp01_b_against_rudolph_neumann(V):
    # Rudolph-Neumann approximation b ~ (1.1428 - 0.996/V)^2, ~0.2 % for 1.5 < V < 2.5
    lam, dn = 1550e-9, 0.005
    ncl = fm.silica_index(lam)["n"]
    a = V * lam / (2 * math.pi * math.sqrt((ncl + dn) ** 2 - ncl ** 2))
    r = fm.lp01_mode(lam, a, dn)
    assert r["V"] == pytest.approx(V, rel=1e-12)
    assert r["b"] == pytest.approx((1.1428 - 0.996 / V) ** 2, rel=5e-3)


def test_n_eff_between_cladding_and_core():
    r = fm.lp01_mode(1550e-9, 4.1e-6, 0.005)
    ncl = fm.silica_index(1550e-9)["n"]
    assert ncl < r["n_eff"] < ncl + 0.005


def test_marcuse_radius_formula():
    # w/a = 0.65 + 1.619 V^-3/2 + 2.879 V^-6 (Marcuse 1977), evaluated by hand at the computed V
    r = fm.lp01_mode(1550e-9, 4.1e-6, 0.005)
    V = r["V"]
    assert r["w_mode"] / 4.1e-6 == pytest.approx(0.65 + 1.619 / V ** 1.5 + 2.879 / V ** 6)


def test_smf_dispersion_matches_datasheet():
    # SMF-28-like fiber: D(1550) ~ 17 ps/(nm km) -> beta2 = -D lambda^2/(2 pi c) ~ -21.7 ps^2/km.
    # chi2 with signal and idler both at 1550 nm is not physical, so compare beta2 directly via k(w).
    a, dn, lam = 4.1e-6, 0.005, 1550e-9
    w, h = 2 * math.pi * C0 / lam, 1e13
    k = lambda x: fm.wavenumber(x, a, dn)
    beta2 = (k(w + h) - 2 * k(w) + k(w - h)) / h ** 2
    D_expected = 17e-6  # s/m^2
    assert beta2 == pytest.approx(-D_expected * lam ** 2 / (2 * math.pi * C0), rel=0.06)


def test_chi2_and_chi3_share_signal_idler_pair():
    # 2 omega(1064 nm) = omega(532 nm): same idler, same GVM, opposite sign convention for dk_rel
    a, dn = 4.1e-6, 0.005
    r2 = fm.phase_mismatch(1550e-9, 532e-9, a, dn, "chi2", detuning=2 * math.pi * 1e11)
    r3 = fm.phase_mismatch(1550e-9, 1064e-9, a, dn, "chi3", detuning=2 * math.pi * 1e11)
    assert r2["lambda_idler"] == pytest.approx(r3["lambda_idler"], rel=1e-12)
    assert r2["gvm"] == pytest.approx(r3["gvm"], rel=1e-6)
    assert r2["dk_rel"] == pytest.approx(-r3["dk_rel"], rel=1e-6)


def test_dk_rel_linear_term_is_gvm():
    # small detuning: dk_rel ~ GVM * Omega (Taylor expansion, chi2 sign)
    r = fm.phase_mismatch(1550e-9, 532e-9, 4.1e-6, 0.005, "chi2", detuning=2 * math.pi * 1e9)
    assert r["dk_rel"] == pytest.approx(r["gvm"] * 2 * math.pi * 1e9, rel=1e-3)


def test_qpm_period_is_tens_of_micrometres():
    # sanity: QPM of 532 -> 1550 + 810 in silica needs Lambda = 2 pi / dk0 of order 40-50 um (applet: 44.8 um)
    r = fm.phase_mismatch(1550e-9, 532e-9, 4.1e-6, 0.005, "chi2")
    assert 40e-6 < r["qpm_period"] < 50e-6


def test_vectorised_detuning():
    det = np.linspace(-1e13, 1e13, 5)
    r = fm.phase_mismatch(1550e-9, 532e-9, 4.1e-6, 0.005, "chi2", detuning=det)
    assert r["dk_rel"].shape == (5,) and r["dk_rel"][2] == 0


def test_rejects_bad_process():
    with pytest.raises(ValueError):
        fm.phase_mismatch(1550e-9, 532e-9, 4.1e-6, 0.005, "chi4")
