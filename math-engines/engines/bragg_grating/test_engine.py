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


# ---------------------------------------------------------------- oblique incidence (v2)
def test_oblique_reduces_to_abeles_at_normal_incidence():
    lam = np.linspace(1.45e-6, 1.65e-6, 41)
    for pol in ("s", "p"):
        mine = bg.stack_R_oblique(lam, np.zeros_like(lam), 1.0, [(2.8, 138e-9), (1.0, 387.5e-9)] * 6, 1.0, pol)
        ref = bg.stack_reflectance(lam, 2.8, 1.0, 138e-9, 387.5e-9, 6, n_in=1.0, n_out=1.0)["R"]
        assert np.max(np.abs(mine - ref)) < 1e-12


def test_single_interface_fresnel_brewster_and_tir():
    n1, n2, s = 2.8, 1.0, 0.25
    c1, c2 = np.sqrt(1 - s**2), np.sqrt(1 - (n1 * s / n2) ** 2)
    rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
    rp = ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)) ** 2
    assert bg.stack_R_oblique(1.55e-6, s, n1, [], n2, "s") == pytest.approx(rs, rel=1e-12)
    assert bg.stack_R_oblique(1.55e-6, s, n1, [], n2, "p") == pytest.approx(rp, rel=1e-9)
    sb = n2 / np.hypot(n1, n2)
    assert bg.stack_R_oblique(1.55e-6, sb, n1, [], n2, "p") < 1e-20
    assert bg.stack_R_oblique(1.55e-6, 0.6, n1, [], n2, "s") == pytest.approx(1.0, abs=1e-9)


def test_frustrated_tir_through_an_air_gap():
    # beyond the critical angle a gap between two high-index media leaks; wider gaps reflect more
    R = [float(bg.stack_R_oblique(1.55e-6, 0.6, 2.8, [(1.0, d)], 2.8, "s")) for d in (20e-9, 200e-9, 2e-6)]
    assert R[0] < R[1] < R[2] < 1 and R[2] > 0.999


def test_trench_dbr_tm_beats_te():
    te = bg.TrenchDBR(n_tooth=2.6, N=10, m_tooth=1, slab_pol="TE", bounce_loss=0)
    tm = bg.TrenchDBR(n_tooth=2.6, N=10, m_tooth=1, slab_pol="TM", bounce_loss=0)
    assert 0 < te.angle_average() < tm.angle_average() <= 1
    assert tm.R(1.55e-6, 0.5) == pytest.approx(1.0, abs=1e-6)     # beyond 1/n_eff: TIR


def test_double_resonant_orders_are_consistent():
    best = bg.double_resonant_orders(1.55e-6, 1.65e-6, 2.9, 2.8, top=3)
    for c in best:
        a1, a2 = c["tooth_orders"]
        assert a1 % 2 == 1 and a2 % 2 == 1
        assert c["tooth_mismatch"] == pytest.approx(abs(a1 * 1.55 / 2.9 - a2 * 1.65 / 2.8) / (a1 * 1.55 / 2.9))
        assert c["d_tooth"] == pytest.approx(a1 * 1.55e-6 / (4 * 2.9))
    assert best[0]["tooth_mismatch"] < 0.01


def test_oblique_result():
    r = bg.oblique_reflectance(1.55e-6, 0.0, 2.83, 1.0, 387.5e-9, 2.83, 1.55e-6 / (4 * 2.83), 10, 1.0)
    assert r["R"] > 0.9999 and r["sin_tir"] == pytest.approx(1 / 2.83)


def test_trench_dbr_design_angle_moves_the_stop_band():
    n_t = 2.479
    normal = bg.TrenchDBR(n_tooth=n_t, N=6, m_tooth=1, slab_pol="TM", bounce_loss=0)
    tilted = bg.TrenchDBR(n_tooth=n_t, N=6, m_tooth=1, slab_pol="TM", bounce_loss=0, sin_design=0.3)
    assert tilted.d_gap == pytest.approx(1.55e-6 / (4 * np.sqrt(1 - (n_t * 0.3) ** 2)))
    assert tilted.d_tooth == pytest.approx(1.55e-6 / (4 * n_t * np.sqrt(1 - 0.09)))
    # quarter-wave at its own design angle: each tilted design beats the other one at that angle
    assert tilted.R(1.55e-6, 0.3) > normal.R(1.55e-6, 0.3)
    assert normal.R(1.55e-6, 0.0) > tilted.R(1.55e-6, 0.0)
    # at the design angle every layer is a quarter wave: R = ((Y - 1)/(Y + 1))^2, Y = (η_t/η_g)^(2N+1), η = n cos θ (s)
    for pol, adm in (("TM", lambda n, c: n * c), ("TE", lambda n, c: n / c)):
        d = bg.TrenchDBR(n_tooth=n_t, N=6, m_tooth=1, slab_pol=pol, bounce_loss=0, sin_design=0.3)
        eta_t, eta_g = adm(n_t, np.sqrt(1 - 0.09)), adm(1.0, np.sqrt(1 - (n_t * 0.3) ** 2))
        Y = (eta_t / eta_g) ** 13
        assert d.R(1.55e-6, 0.3) == pytest.approx(((Y - 1) / (Y + 1)) ** 2, rel=1e-9)
    with pytest.raises(ValueError, match="sin_design"):
        bg.TrenchDBR(n_tooth=n_t, sin_design=0.5)
