"""Tests for parametric_amplifier. Expected values come from closed forms stated in each test."""
import math

import numpy as np
import pytest

from engines.common import C0
from engines.parametric_amplifier import engine as pa


def test_lossless_gain_formula():
    # G = 1 + (Gamma/g)^2 sinh^2(g L), g^2 = Gamma^2 - (dk/2)^2  (Boyd, Nonlinear Optics, sec. 2.9)
    G, dk, L = 0.95, 1.5, 5.0
    g = math.sqrt(G ** 2 - (dk / 2) ** 2)
    assert pa.small_signal_gain(G, dk, L)["gain"] == pytest.approx(1 + (G / g) ** 2 * math.sinh(g * L) ** 2, rel=1e-10)


def test_phase_matched_gain_is_cosh_squared():
    assert pa.small_signal_gain(0.8, 0.0, 4.0)["gain"] == pytest.approx(math.cosh(3.2) ** 2, rel=1e-10)


def test_outside_gain_band_oscillates():
    # |dk| > 2 Gamma: G = 1 + (Gamma/g')^2 sin^2(g' L), g'^2 = (dk/2)^2 - Gamma^2
    G, dk, L = 0.5, 3.0, 7.0
    gp = math.sqrt((dk / 2) ** 2 - G ** 2)
    assert pa.small_signal_gain(G, dk, L)["gain"] == pytest.approx(1 + (G / gp) ** 2 * math.sin(gp * L) ** 2, rel=1e-10)


def test_gain_symmetric_in_mismatch_sign():
    r = pa.small_signal_gain(0.9, np.array([-1.2, 1.2]), 5.0, idler_loss=2.0)
    assert r["gain"][0] == pytest.approx(r["gain"][1], rel=1e-12)


def test_strong_idler_loss_adiabatic_limit():
    # alpha >> Gamma: the idler follows the signal, power gain coefficient -> 4 Gamma^2 / alpha
    G, alpha, L = 0.5, 200.0, 400.0
    g_eff = math.log(pa.small_signal_gain(G, 0.0, L, idler_loss=alpha)["gain"]) / L
    assert g_eff == pytest.approx(4 * G ** 2 / alpha, rel=2e-3)


def test_complete_idler_dumps_multiply_stage_gains():
    # idler fully removed between N+1 equal stages: G = (cosh^2(Gamma L/(N+1)))^(N+1) at dk = 0
    G, L, N = 1.0, 6.0, 3
    r = pa.small_signal_gain(G, 0.0, L, n_dumps=N, idler_dump_transmission=0.0)
    assert r["gain"] == pytest.approx(math.cosh(G * L / (N + 1)) ** (2 * (N + 1)), rel=1e-10)


def test_signal_loss_without_coupling():
    # Gamma = 0: the signal simply decays, G = exp(-alpha_s L)
    assert pa.small_signal_gain(0.0, 0.0, 3.0, signal_loss=0.7)["gain"] == pytest.approx(math.exp(-2.1))


def test_chi2_overlap_equal_waists():
    # three equal Gaussian fields: Theta = 2^(3/2) / (3 sqrt(pi) w)  (hand integral)
    w = 3e-6
    assert pa.mode_overlap(w, w, w, "chi2") == pytest.approx(2 ** 1.5 / (3 * math.sqrt(math.pi) * w))


def test_chi3_effective_area_equal_waists():
    # equal radii: A_eff = pi w^2, gamma = n2 omega / (c A_eff); 10 um^2, 1064 nm -> 15.35 1/(W km)
    w = math.sqrt(10e-12 / math.pi)
    r = pa.coupling("chi3", 1064e-9, 1550e-9, w, w, w, 10.0, n2=2.6e-20)
    assert r["A_eff"] == pytest.approx(10e-12)
    assert r["gamma"] == pytest.approx(2 * math.pi * 2.6e-20 / (1064e-9 * 10e-12), rel=1e-12)
    assert r["nonlinear_phase"] == pytest.approx(2 * r["gamma"] * 10.0)


def test_chi2_gain_coefficient_scaling():
    # Gamma is proportional to d_eff sqrt(P_p)
    args = ("chi2", 532e-9, 1550e-9, 2.5e-6, 5e-6, 3.2e-6)
    a = pa.coupling(*args, pump_power=1.0, d_eff=0.08e-12)["gain_coefficient"]
    b = pa.coupling(*args, pump_power=4.0, d_eff=0.16e-12)["gain_coefficient"]
    assert b == pytest.approx(4 * a)


def test_chi2_plane_wave_limit():
    # very wide equal modes: Gamma^2 -> 8 pi^2 d^2 I / (n^3 eps0 c lambda_s lambda_i) with I = P / A_eff
    # (Boyd eq. 2.9.x); A_eff = 1/Theta^2 = 9 pi w^2 / 8 for three equal Gaussians
    w, d, P, n = 1e-3, 1e-12, 1e3, 1.45
    r = pa.coupling("chi2", 532e-9, 1550e-9, w, w, w, P, d_eff=d, n=n)
    I = P / r["A_eff"]
    li = r["lambda_idler"]
    expected = math.sqrt(8 * math.pi ** 2 * d ** 2 * I / (n ** 3 * 8.8541878128e-12 * C0 * 1550e-9 * li))
    assert r["gain_coefficient"] == pytest.approx(expected, rel=1e-9)


def test_rk4_chi2_matches_analytic_small_signal():
    # 1 nW signal keeps pump depletion below 1e-5 so the undepleted solution applies
    a = pa.amplify("chi2", 532e-9, 1550e-9, 1.0, 1e-9, 5.0, 0.95, 1.5)
    b = pa.small_signal_gain(0.95, 1.5, 5.0)["gain_db"]
    assert a["gain_db"] == pytest.approx(b, abs=1e-3)
    assert a["photon_residual"] < 1e-10


def test_rk4_chi2_depleted_conserves_photons():
    # Manley-Rowe: Phi_p + Phi_s conserved at every z, through full depletion and back-conversion
    a = pa.amplify("chi2", 532e-9, 1550e-9, 1.0, 1e-3, 8.0, 1.2, 0.0, record=True)
    hbar_w = lambda lam: 1.054571817e-34 * 2 * math.pi * C0 / lam
    phi = a["P_pump"] / hbar_w(532e-9) + a["P_signal"] / hbar_w(1550e-9)
    assert 1 - a["P_pump"].min() / 1.0 > 0.9
    assert np.ptp(phi) / phi[0] < 1e-8


def test_rk4_chi3_matches_analytic_small_signal():
    # chi3 with SPM/XPM: the undepleted net-mismatch model must agree at small signal
    gam = 0.147
    a = pa.amplify("chi3", 1064e-9, 1550e-9, 10.0, 1e-9, 30.0, gam, 0.15)
    b = pa.small_signal_gain(gam, 0.15, 30.0)["gain_db"]
    assert a["gain_db"] == pytest.approx(b, abs=1e-3)
    assert a["photon_residual"] < 1e-10


def test_rk4_idler_loss_matches_analytic():
    a = pa.amplify("chi2", 532e-9, 1550e-9, 1.0, 1e-9, 5.0, 0.95, 0.5, idler_loss=2.3)
    b = pa.small_signal_gain(0.95, 0.5, 5.0, idler_loss=2.3)["gain_db"]
    assert a["gain_db"] == pytest.approx(b, abs=1e-3)


def test_bandwidth_high_gain_estimate_and_nm_conversion():
    # high-gain GVM estimate 4 sqrt(Gamma ln2 / L) / |GVM| (applet docs) within 15 %;
    # nm width = lambda^2 dnu / c
    r = pa.gain_bandwidth(0.95, 5.0, 1550e-9, 532e-9, 15.16e-12, 13e-27)
    assert r["width"] == pytest.approx(r["high_gain_estimate"], rel=0.15)
    assert r["width_nm"] == pytest.approx(1550e-9 ** 2 * r["width_hz"] / C0, rel=1e-12)
    assert r["idler_width_nm"] / r["width_nm"] == pytest.approx(pa.idler_wavelength(1550e-9, 532e-9) ** 2 / 1550e-9 ** 2)


def test_requires_mode_radii_for_scattering():
    with pytest.raises(ValueError):
        pa.amplify("chi2", 532e-9, 1550e-9, 1.0, 1e-6, 5.0, 0.95, srs=True)
