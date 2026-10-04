import numpy as np
import pytest
from scipy.special import ellipj

from engines.opa_chi2 import engine as e
from engines.phase_matching import engine as pm

HB, C0, EPS0 = e.HBAR, e.C0, e.EPS0
ARGS = dict(wavelength_pump=780e-9, wavelength_signal=1550e-9, d_eff=14e-12, n_pump=2.18, n_signal=2.14, n_idler=2.14,
            w_pump=2e-6, w_signal=2e-6, w_idler=2e-6)


def test_idler_and_overlap_hand_values():
    assert e.idler_wavelength(1.0e-6, 1.5e-6) == pytest.approx(3.0e-6, rel=1e-14)
    w = 3e-6   # equal radii: θ = (2/π)^{3/2} π w²/3 / w³
    assert e.mode_overlap(w, w, w) == pytest.approx((2 / np.pi) ** 1.5 * np.pi / (3 * w), rel=1e-14)
    # numerical overlap integral of unit-power Gaussians with different radii
    wp, ws, wi = 1.5e-6, 2.5e-6, 3.5e-6
    r = np.linspace(0, 30e-6, 200001)
    ef = lambda w: np.sqrt(2 / (np.pi * w * w)) * np.exp(-r * r / (w * w))
    num = np.trapezoid(ef(wp) * ef(ws) * ef(wi) * 2 * np.pi * r, r)
    assert e.mode_overlap(wp, ws, wi) == pytest.approx(num, rel=1e-8)


def test_gamma_matches_textbook_intensity_formula():
    c = e.coupling(pump_power=0.7, **ARGS)
    ws, wi = 2 * np.pi * C0 / 1550e-9, 2 * np.pi * C0 / c["wavelength_idler"]
    I = 0.7 * c["overlap"] ** 2
    g2 = 2 * ws * wi * ARGS["d_eff"] ** 2 * I / (2.18 * 2.14 * 2.14 * EPS0 * C0**3)
    assert c["gamma"] ** 2 == pytest.approx(g2, rel=1e-12)
    assert c["gain_length"] * c["gamma"] == pytest.approx(1.0)
    # Γ ∝ sqrt(P_p)
    assert e.coupling(pump_power=2.8, **ARGS)["gamma"] == pytest.approx(2 * c["gamma"], rel=1e-12)


def test_small_signal_limits():
    G, L = 100.0, 0.02
    r = e.small_signal_gain(G, 0.0, L, 780e-9, 1550e-9)
    assert r["signal_gain"] == pytest.approx(np.cosh(G * L) ** 2, rel=1e-12)
    assert r["peak_gain_db"] == pytest.approx(r["signal_gain_db"])
    assert e.small_signal_gain(G, 2 * G, L, 780e-9, 1550e-9)["signal_gain"] == pytest.approx(1 + (G * L) ** 2, rel=1e-6)
    dk = 2 * np.sqrt(G**2 + 300.0**2)   # g² = -300²: G = 1 + (Γ/300)² sin²(300 L)
    assert e.small_signal_gain(G, dk, L, 780e-9, 1550e-9)["signal_gain"] == pytest.approx(1 + (G / 300) ** 2 * np.sin(300 * L) ** 2, rel=1e-10)
    li = e.idler_wavelength(780e-9, 1550e-9)
    assert r["idler_conversion"] == pytest.approx(1550e-9 / li * (r["signal_gain"] - 1), rel=1e-14)
    # symmetric in Δk
    assert e.small_signal_gain(G, 150.0, L, 780e-9, 1550e-9)["signal_gain"] == pytest.approx(
        e.small_signal_gain(G, -150.0, L, 780e-9, 1550e-9)["signal_gain"], rel=1e-14)


@pytest.mark.parametrize("dk", [0.0, 80.0, 250.0])
def test_rk4_matches_closed_form_in_undepleted_limit(dk):
    G, L = 120.0, 0.025
    r = e.propagate_normalised(G, dk, L, 1e-12)
    want = e.small_signal_gain(G, dk, L, 780e-9, 1550e-9)["signal_gain"]
    assert r["fs"] / 1e-12 == pytest.approx(want, rel=1e-7)
    assert r["fi"] / 1e-12 == pytest.approx(want - 1, rel=1e-7)


def test_pump_depletion_matches_armstrong_solution():
    # Δk = 0, no idler seed: f_i(z) = r0/(1+r0) sd²(Γ√(1+r0) z | m), m = 1/(1+r0)
    G, r0 = 150.0, 1e-3
    for L in (0.01, 0.02, 0.035):
        r = e.propagate_normalised(G, 0.0, L, r0, max_steps=20000)
        m = 1 / (1 + r0)
        sn, cn, dn, _ = ellipj(G * np.sqrt(1 + r0) * L, m)
        fi = r0 / (1 + r0) * (sn / dn) ** 2
        assert r["fi"] == pytest.approx(fi, rel=1e-6)
        assert r["fp"] == pytest.approx(1 - fi, rel=1e-6, abs=1e-9)
        assert r["fs"] == pytest.approx(r0 + fi, rel=1e-6)
        assert r["manley_rowe_residual"] < 1e-9


def test_coupled_wave_power_bookkeeping():
    r = e.coupled_wave(pump_power=1.0, signal_power=1e-3, length=0.02, **ARGS)
    li = r["wavelength_idler"]
    # energy conservation (lossless): total power constant
    assert r["pump_out"] + r["signal_out"] + r["idler_out"] == pytest.approx(1.001, rel=1e-9)
    # photon numbers: idler photons generated = signal photons added
    assert r["idler_out"] * li == pytest.approx((r["signal_out"] - 1e-3) * 1550e-9, rel=1e-9)
    assert r["z"][0] == 0 and r["z"][-1] == pytest.approx(0.02)
    assert r["signal_power_z"][-1] == pytest.approx(r["signal_out"])
    assert 0 < r["pump_depletion"] < 1
    # depletion lowers the gain below the closed form
    assert r["signal_gain_db"] < e.small_signal_gain(r["gamma"], 0, 0.02, 780e-9, 1550e-9)["signal_gain_db"]


def test_idler_loss_lowers_gain():
    a = e.coupled_wave(pump_power=1.0, signal_power=1e-6, length=0.02, **ARGS)
    b = e.coupled_wave(pump_power=1.0, signal_power=1e-6, length=0.02, alpha_idler=20.0, **ARGS)
    assert b["signal_gain_db"] < a["signal_gain_db"]
    assert b["manley_rowe_residual"] < 1e-12      # f_p + f_s is untouched by idler loss
    c = e.coupled_wave(pump_power=1.0, signal_power=1e-6, length=0.02, alpha_pump=2.0, **ARGS)
    assert c["manley_rowe_residual"] > 1e-2


def test_gain_spectrum_consistent_with_engines():
    g = e.gain_spectrum("ln_e", 780e-9, 1550e-9, 300e-9, 0.02, 14e-12, 1.0, 2e-6, 2e-6, 2e-6, points=301)
    k = 150   # centre point is phase-matched by construction
    assert g["signal_wavelengths"][k] == pytest.approx(1550e-9)
    assert abs(g["delta_k"][k]) < 1e-6
    assert g["gain"][k] == pytest.approx(np.cosh(g["gamma_center"] * 0.02) ** 2, rel=1e-9)
    # every point agrees with phase_matching + small_signal_gain
    j = 40
    ls, li = g["signal_wavelengths"][j], g["idler_wavelengths"][j]
    from engines.materials.engine import index
    dk = pm.three_wave_mismatch(ls, li, index("ln_e", ls), index("ln_e", li), index("ln_e", 780e-9), period=g["period"])["delta_k"]
    assert g["delta_k"][j] == pytest.approx(dk, rel=1e-12)
    gam = e.coupling(780e-9, ls, 14e-12, 1.0, index("ln_e", 780e-9), index("ln_e", ls), index("ln_e", li), 2e-6, 2e-6, 2e-6)["gamma"]
    assert g["gain"][j] == pytest.approx(e.small_signal_gain(gam, dk, 0.02, 780e-9, ls)["signal_gain"], rel=1e-12)
    assert 10e-9 < g["bandwidth_3db"] < 300e-9
    # a narrow span cuts the gain band: bandwidth undefined
    assert np.isnan(e.gain_spectrum("ln_e", 780e-9, 1550e-9, 5e-9, 0.02, 14e-12, 1.0, 2e-6, 2e-6, 2e-6)["bandwidth_3db"])


def test_validation():
    with pytest.raises(ValueError, match="wavelength_signal"):
        e.idler_wavelength(1.0e-6, 0.9e-6)
    with pytest.raises(ValueError, match="pump_power"):
        e.coupling(pump_power=0.0, **ARGS)
    with pytest.raises(ValueError, match="signal_power"):
        e.coupled_wave(pump_power=1.0, signal_power=0.0, length=0.01, **ARGS)
    with pytest.raises(ValueError, match="material"):
        e.gain_spectrum("glass", 780e-9, 1550e-9, 100e-9, 0.02, 14e-12, 1.0, 2e-6, 2e-6, 2e-6)


def test_dissipative_idler_options():
    from engines.idler_loss import engine as il
    G, dk, L = 120.0, 40.0, 0.03
    for kw in (dict(alpha_idler=150.0), dict(dumps=3, dump_loss_db=20.0), dict(alpha_idler=50.0, dumps=2, dump_loss_db=10.0)):
        r = e.small_signal_gain(G, dk, L, 780e-9, 1550e-9, **kw)
        s, c = il.propagate_linear(G, dk, L, **kw)
        assert r["signal_gain"] == pytest.approx(abs(s) ** 2, rel=1e-12)
        assert r["idler_conversion"] == pytest.approx(1550e-9 / e.idler_wavelength(780e-9, 1550e-9) * abs(c) ** 2, rel=1e-12)
        assert r["signal_gain_db"] < r["lossless_gain_db"]
    # nonlinear model: idler dumps leave f_p + f_s untouched and cost gain
    a = e.coupled_wave(pump_power=1.0, signal_power=1e-3, length=0.02, **ARGS)
    b = e.coupled_wave(pump_power=1.0, signal_power=1e-3, length=0.02, dumps=4, dump_loss_db=30.0, **ARGS)
    assert b["signal_gain_db"] < a["signal_gain_db"] and b["manley_rowe_residual"] < 1e-9
    assert len(b["z"]) == len(b["idler_power_z"]) and b["z"][-1] == pytest.approx(0.02)
    k = int(np.argmin(np.abs(b["z"] - 0.02 / 5)))      # first dump: idler drops by 30 dB at the same z
    j = [i for i in range(1, len(b["z"])) if b["z"][i] == b["z"][i - 1]][0]
    assert b["idler_power_z"][j] == pytest.approx(1e-3 * b["idler_power_z"][j - 1], rel=1e-9) and abs(j - k) <= 1


def test_gain_spectrum_with_idler_loss():
    from engines.idler_loss import engine as il
    base = ("ln_e", 780e-9, 1550e-9, 300e-9, 0.03, 14e-12, 1.0, 2e-6, 2e-6, 2e-6)
    a = e.gain_spectrum(*base, points=201)
    b = e.gain_spectrum(*base, points=201, dumps=4, dump_loss_db=30.0)
    assert b["lossless_peak_gain_db"] == pytest.approx(a["peak_gain_db"])
    assert b["peak_gain_db"] < a["peak_gain_db"] and b["bandwidth_3db"] > a["bandwidth_3db"]
    # band-pass loss away from every idler: same as lossless
    c = e.gain_spectrum(*base, points=201, alpha_idler=500.0, loss_profile="pass", band_offset=500e-9, band_width=10e-9, band_edge=0.5e-9)
    assert c["gain"] == pytest.approx(a["gain"], rel=1e-9)
    assert c["loss_weight"].max() < 1e-12
