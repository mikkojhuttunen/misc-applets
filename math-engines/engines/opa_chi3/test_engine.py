import math

import numpy as np
import pytest

from engines.opa_chi3 import engine as e
from engines.step_index_fiber import engine as sif

C0 = e.C0
LP, LS = 1550e-9, 1556e-9


def test_idler_and_area_hand_values():
    assert e.idler_wavelength(1.0e-6, 1.25e-6) == pytest.approx(1 / (2 - 0.8) * 1e-6, rel=1e-14)
    assert e.effective_area(3e-6, 3e-6, 3e-6) == pytest.approx(np.pi * 9e-12, rel=1e-14)
    wp, ws, wi = 2.0e-6, 2.6e-6, 3.1e-6
    r = np.linspace(0, 30e-6, 200001)
    ef = lambda w: np.sqrt(2 / (np.pi * w * w)) * np.exp(-r * r / (w * w))
    theta = np.trapezoid(ef(wp) ** 2 * ef(ws) * ef(wi) * 2 * np.pi * r, r)
    assert e.effective_area(wp, ws, wi) == pytest.approx(1 / theta, rel=1e-8)


def test_gamma_textbook_value():
    c = e.coupling(LP, LS, 2.6e-20, 1.0, 2.6e-6, 2.6e-6, 2.6e-6)
    assert c["gamma_nl"] == pytest.approx(2.6e-20 * 2 * np.pi * C0 / LP / (C0 * np.pi * 2.6e-6**2), rel=1e-12)
    li = c["wavelength_idler"]
    assert c["gain_max"] == pytest.approx(c["gamma_nl"] * np.sqrt(LP / LS * LP / li), rel=1e-12)


def test_small_signal_limits():
    g, P, L = 0.01, 2.0, 500.0
    li = e.idler_wavelength(LP, LS)
    r = g * P * math.sqrt(LP * LP / (LS * li))
    # κ = 0 → cosh²(rPL)
    a = e.small_signal_gain(g, P, -2 * g * P, L, LP, LS)
    assert a["signal_gain"] == pytest.approx(math.cosh(r * L) ** 2, rel=1e-12)
    assert a["peak_gain_db"] == pytest.approx(a["signal_gain_db"])
    # textbook unscaled limit at Δβ = 0: g² = (γP)² - (2γP/2)² = 0 → G = 1 + (γPL)² (with r ≈ γP)
    b = e.small_signal_gain(g, P, 0.0, L, LP, LP * (1 + 1e-9))
    assert b["signal_gain"] == pytest.approx(1 + (g * P * L) ** 2, rel=1e-6)
    assert a["idler_conversion"] == pytest.approx(LS / li * (a["signal_gain"] - 1), rel=1e-14)


@pytest.mark.parametrize("dbeta", [-0.04, -0.02, 0.0, 0.01])
def test_rk4_matches_closed_form_with_spm_xpm(dbeta):
    g, P, L = 0.01, 2.0, 400.0
    li = e.idler_wavelength(LP, LS)
    r0 = 1e-12
    out = e.propagate_normalised(g * P, dbeta, L, r0, LP / LS, LP / li, max_steps=50000)
    want = e.small_signal_gain(g, P, dbeta, L, LP, LS)["signal_gain"]
    # 1e-5: RK4 discretisation at the applet's step rule (h ≈ 0.04 / rate), here up to 61 dB of gain
    assert out["fs"] / r0 == pytest.approx(want, rel=1e-5)
    assert out["fi"] / r0 == pytest.approx(want - 1, rel=1e-5)


def test_conservation_and_depletion():
    c = e.coupled_wave(LP, LS, 0.005, 2.0, 1e-3, 500.0, -0.008)
    li = c["wavelength_idler"]
    assert c["pump_out"] + c["signal_out"] + c["idler_out"] == pytest.approx(2.001, rel=1e-8)
    # photon numbers: idler photons generated = signal photons added
    assert c["idler_out"] * li == pytest.approx((c["signal_out"] - 1e-3) * LS, rel=1e-7)
    assert c["manley_rowe_residual"] < 1e-7
    assert 0.1 < c["pump_depletion"] < 1
    assert c["signal_gain_db"] < e.small_signal_gain(0.005, 2.0, -0.008, 500.0, LP, LS)["signal_gain_db"]
    assert c["signal_power_z"][-1] == pytest.approx(c["signal_out"]) and c["z"][-1] == pytest.approx(500.0)


def test_pump_alone_gets_spm_only():
    # no signal: pump power stays, nothing is generated (photon budget untouched)
    out = e.propagate_normalised(0.02, -0.01, 300.0, 0.0, LP / LS, LP / e.idler_wavelength(LP, LS))
    assert out["fp"] == pytest.approx(1.0, rel=1e-9) and out["fs"] == 0      # RK4 norm drift only and out["fi"] == 0


def test_losses():
    a = e.coupled_wave(LP, LS, 0.005, 2.0, 1e-6, 500.0, -0.008)
    b = e.coupled_wave(LP, LS, 0.005, 2.0, 1e-6, 500.0, -0.008, alpha_idler=1e-3)
    c = e.coupled_wave(LP, LS, 0.005, 2.0, 1e-6, 500.0, -0.008, alpha_pump=1e-3)
    assert b["signal_gain_db"] < a["signal_gain_db"] and b["manley_rowe_residual"] < 1e-9
    assert c["signal_gain_db"] < a["signal_gain_db"] and c["manley_rowe_residual"] > 0.1


def test_taylor_spectrum_peak_and_phase_matching():
    b2, b4, g, P, L = -1e-27, -1e-55, 0.01, 2.0, 400.0
    s = e.gain_spectrum(LP, b2, b4, g, P, L, 60e-9, points=2001)
    om = s["omega_phase_matched"]
    assert b2 * om**2 + b4 * om**4 / 12 == pytest.approx(-2 * g * P, rel=1e-10)
    # at the phase-matched point κ = 0: G = cosh²(rPL)
    ls = 1 / (1 / LP - om / (2 * np.pi * C0))       # Ω < 0 side (long wavelength); Δβ is even in Ω
    want = e.small_signal_gain(g, P, b2 * om**2 + b4 * om**4 / 12, L, LP, ls)["signal_gain"]
    assert want == pytest.approx(e.small_signal_gain(g, P, -2 * g * P, L, LP, ls)["signal_gain"], rel=1e-9)
    assert abs(s["peak_signal_wavelength"] - ls) < 0.1e-9
    assert s["peak_gain_db"] == pytest.approx(10 * np.log10(want), abs=0.01)
    assert 0 < s["lobe_width_3db"] < 60e-9
    # normal dispersion, β4 = 0: no phase matching, gain ≈ quadratic only
    n = e.gain_spectrum(LP, 1e-27, 0.0, g, P, L, 60e-9)
    assert math.isnan(n["omega_phase_matched"]) and n["peak_gain_db"] < s["peak_gain_db"]


def test_fiber_parameters_smf28_and_taylor_consistency():
    smf = e.fiber_parameters(1550e-9, 1560e-9, 4.1e-6, 0.005)
    assert -24e-27 < smf["beta2"] < -19e-27              # standard SMF ≈ -21.7 ps²/km
    assert smf["beta3"] > 0
    f = e.fiber_parameters(LP, LS, 2.0e-6, 0.02)
    # independent β2: central difference of β(ω) from step_index_fiber
    w0, h = 2 * np.pi * C0 / LP, 1e13
    beta = lambda w: w / C0 * float(sif.lp01(2 * np.pi * C0 / w, 2.0e-6, 0.02)["neff"])
    b2fd = (beta(w0 + h) - 2 * beta(w0) + beta(w0 - h)) / h**2
    assert f["beta2"] == pytest.approx(b2fd, rel=1e-3, abs=2e-30)
    # exact Δβ vs Taylor for a small detuning
    O = 2 * np.pi * C0 * (1 / LS - 1 / LP)
    assert f["delta_beta"] == pytest.approx(f["beta2"] * O**2 + f["beta4"] * O**4 / 12, rel=2e-3)


def test_fiber_gain_spectrum_point_consistency():
    s = e.fiber_gain_spectrum(LP, 2.0e-6, 0.02, 2.0, 500.0, 40e-9, points=41)
    j = 30
    ls = s["signal_wavelengths"][j]
    f = e.fiber_parameters(LP, ls, 2.0e-6, 0.02)
    assert s["delta_beta"][j] == pytest.approx(f["delta_beta"], rel=1e-9, abs=1e-9)
    want = e.small_signal_gain(f["gamma_nl"], 2.0, f["delta_beta"], 500.0, LP, ls)["signal_gain"]
    assert s["gain"][j] == pytest.approx(want, rel=1e-9)


def test_validation():
    with pytest.raises(ValueError, match="wavelength_signal"):
        e.idler_wavelength(1.55e-6, 0.7e-6)
    with pytest.raises(ValueError, match="gamma_nl"):
        e.small_signal_gain(0.0, 1.0, 0.0, 100.0, LP, LS)
    with pytest.raises(ValueError, match="signal_power"):
        e.coupled_wave(LP, LS, 0.01, 1.0, 0.0, 100.0)
    with pytest.raises(ValueError, match="points"):
        e.gain_spectrum(LP, -1e-27, 0.0, 0.01, 1.0, 100.0, 20e-9, points=2)
