import numpy as np
import pytest

from engines.bragg_grating import engine as bg


def test_kappa_rectangular_hand_value():
    # κ = 2 Δn sin(π/2) / λ = 2 * 0.01 / 1.55e-6 = 12903 1/m
    assert bg.coupling_coefficient(0.01, 1.55e-6)["kappa"] == pytest.approx(12903.2, rel=1e-4)


def test_even_order_vanishes_at_half_fill():
    assert bg.coupling_coefficient(0.01, 1.55e-6, 0.5, 2)["kappa"] == pytest.approx(0.0, abs=1e-6)


def test_quarter_wave_stack_analytic():
    # (H L)^N between low-index media at the design wavelength:
    # admittance Y = (n_H/n_L)^(2N) n_out, R = ((n_in - Y)/(n_in + Y))^2   (Born & Wolf)
    lam, nH, nL, N = 1.55e-6, 2.0, 1.5, 5
    r = bg.stack_reflectance(lam, nH, nL, lam / (4 * nH), lam / (4 * nL), N)
    Y = (nH / nL) ** (2 * N) * nL
    assert r["R"] == pytest.approx(((nL - Y) / (nL + Y)) ** 2, rel=1e-9)


def test_lossless_energy_conservation():
    lam = np.linspace(1.50e-6, 1.60e-6, 7)
    r = bg.stack_reflectance(lam, 2.03, 2.01, 0.19e-6, 0.19e-6, 300)
    assert np.allclose(r["R"] + r["T"], 1.0, atol=1e-10)


def test_weak_grating_peak_matches_coupled_mode():
    # independent route: exact transfer matrix vs tanh^2(κL) for a weak, half-filled grating
    nH, nL, lamB = 2.0272, 2.0095, 1.55e-6
    nbar = 0.5 * (nH + nL)
    Lam = lamB / (2 * nbar)
    N = 800
    lam = np.linspace(lamB - 2e-9, lamB + 2e-9, 4001)
    R = bg.stack_reflectance(lam, nH, nL, Lam / 2, Lam / 2, N, n_in=nH, n_out=nH)["R"]
    kap = bg.coupling_coefficient(nH - nL, lamB)["kappa"]
    assert R.max() == pytest.approx(np.tanh(kap * N * Lam) ** 2, abs=2e-3)


def test_cmt_peak_equals_tanh2():
    kap, L = 2e4, 300e-6
    assert bg.cmt_reflectance(1.55e-6, 1.55e-6, 2.0, kap, L) == pytest.approx(np.tanh(kap * L) ** 2)


def test_summary_hand_values():
    # κL = 1 -> R = tanh^2(1) = 0.58002
    assert bg.grating_summary(1e4, 1e-4, 1.55e-6, 2.3)["R_peak"] == pytest.approx(0.580026, rel=1e-5)
