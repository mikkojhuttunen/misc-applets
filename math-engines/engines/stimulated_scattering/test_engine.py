import math

import pytest

from engines.stimulated_scattering import engine as ss


def test_brillouin_shift_at_1550():
    # nu_B = 2 n v_A / lambda = 2 * 1.45 * 5960 / 1.55e-6 = 11.15 GHz (hand); measured SMF ~10.8-11 GHz
    assert ss.brillouin_parameters(1550e-9)["shift"] == pytest.approx(11.151e9, rel=1e-3)


def test_linewidth_broadening_reduces_gain():
    # g_B,eff = g_B dnu_B / (dnu_B + dnu_p): pump linewidth equal to dnu_B halves the gain
    br = ss.brillouin_parameters(1550e-9, pump_linewidth=20e6)
    assert br["g_B"] == pytest.approx(2.5e-11)


def test_sbs_pump_minus_stokes_is_conserved():
    sol = ss.SbsSolution(1.0, 5e-11 / 50e-12, 20.0, 1e-9)
    for z in (0.0, 5.0, 13.0, 20.0):
        pump = sol.C + sol.PB(z)
        assert pump - sol.PB(z) == pytest.approx(sol.C)
    assert sol.PB(20.0) == pytest.approx(1e-9, rel=1e-6)


def test_sbs_threshold_behaviour():
    # Smith: well below P_cr reflectivity is tiny, well above it most of the pump is reflected
    base = dict(length=30.0, mode_area=10e-12, lambda_pump=1064e-9)
    thr = ss.sbs_two_wave(pump_power=1.0, **base)["threshold"]
    assert ss.sbs_two_wave(pump_power=0.3 * thr, **base)["reflectivity"] < 1e-3
    assert ss.sbs_two_wave(pump_power=3.0 * thr, **base)["reflectivity"] > 0.5


def test_raman_peak_near_13_thz_and_odd():
    # silica Raman gain peaks near 13.2 THz; the model's oscillator peak sits at 13.1 THz
    assert ss.RAMAN_PEAK_SHIFT / 2 / math.pi == pytest.approx(13.1e12, rel=0.02)
    assert ss.raman_shape(-2e13) == pytest.approx(-ss.raman_shape(2e13))
    assert ss.raman_gain(ss.RAMAN_PEAK_SHIFT / 2 / math.pi, 1e-6)["g_R"] == pytest.approx(1e-13, rel=1e-6)


def test_raman_negligible_far_outside_band():
    assert abs(ss.raman_shape(2 * math.pi * 88e12)) < 1e-6


def test_srs_threshold_formula():
    # 16 A / (g_R L) with g_R(1064 nm) = 1e-13/1.064 m/W, A = 10 um^2, L = 30 m -> 56.7 W
    assert ss.srs_threshold(30.0, 10e-12, 1064e-9)["threshold"] == pytest.approx(16 * 10e-12 * 1.064 / (1e-13 * 30))
