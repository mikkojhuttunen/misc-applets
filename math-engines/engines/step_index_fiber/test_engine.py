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
