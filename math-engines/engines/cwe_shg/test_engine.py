import math

import numpy as np
import pytest

from engines.cwe_shg import engine as e
from engines.phase_matching import engine as pm
from engines.qpm_shg import engine as q

LAM, NP, NSH = 1.55e-6, 2.0272, 2.1924
DK = 4 * np.pi * (NSH - NP) / LAM
PERIOD = 2 * np.pi / DK


def test_coupling_hand_value():
    # η0 = 8π² d² / (ε0 c n_p² n_SH λ² A): d = 25 pm/V, A = 1 µm²
    r = e.coupling_shg(LAM, NP, NSH, 25e-12, 1e-12)
    hand = 8 * math.pi**2 * (25e-12) ** 2 / (8.8541878128e-12 * 299792458.0 * NP**2 * NSH * LAM**2 * 1e-12)
    assert r["eta_norm"] == pytest.approx(hand, rel=1e-12)
    assert r["kappa"] ** 2 == pytest.approx(r["eta_norm"], rel=1e-12)
    # with the first-order QPM factor this is a few 1000 %/(W cm²), the range reported for TFLN ridges
    assert 1e3 < r["eta_norm"] * (2 / math.pi) ** 2 / 100 < 1e4


def test_periodic_domains_layout():
    b, g = e.periodic_domains(10e-6, 4e-6, 0.25)
    assert b == pytest.approx([0, 1e-6, 4e-6, 5e-6, 8e-6, 9e-6, 10e-6])
    assert list(g) == [1, -1, 1, -1, 1, -1]
    b, g = e.periodic_domains(10e-6, 4e-6, 0.5, offset=1e-6)
    assert b == pytest.approx([0, 1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 10e-6])
    assert list(g) == [-1, 1, -1, 1, -1, 1]


def test_period_agrees_with_qpm_shg_and_phase_matching():
    r = e.ordered_shg(LAM, NP, NSH, 25e-12, 1e-12, 1e-3, 1e-3)
    assert r["period"] == pytest.approx(q.phase_matching(LAM, NP, NSH)["period"], rel=1e-12)
    tw = pm.three_wave_mismatch(LAM, LAM, NP, NP, NSH)
    assert r["delta_k"] == pytest.approx(tw["delta_k_bare"], rel=1e-12)


def test_undepleted_ordered_matches_sinc2():
    # test 1: η ∝ L² sinc²(Δk' L/2) with the (2/π)² first-order factor; L an integer number of periods
    L = 400 * PERIOD
    b, g = e.periodic_domains(L, PERIOD)
    kappa, P = 900.0, 1e-3
    dres = np.linspace(-12, 12, 49) * np.pi / L
    r = e.shg_undepleted(b, g, DK + dres, kappa, P)
    closed = kappa**2 * P**2 * (2 / np.pi) ** 2 * L**2 * np.sinc(dres * L / 2 / np.pi) ** 2
    # the counter-rotating grating component at -K adds an O(1/N) ripple
    assert r["P_sh"] == pytest.approx(closed, abs=2e-2 * closed.max())
    assert r["P_sh"][24] == pytest.approx(closed[24], rel=1e-12)
    for D in (0.3, 0.5, 0.7):
        b, g = e.periodic_domains(L, PERIOD, D)
        G = e.poling_spectrum(b, g, DK)
        assert abs(G[0]) == pytest.approx(pm.grating_strength(1, D) * L, rel=1e-9)


def test_undepleted_fwhm_matches_qpm_shg():
    L, ngp, ngs = 2e-3, 2.33, 2.40
    b, g = e.periodic_domains(L, PERIOD)
    lam = LAM + np.linspace(-8e-9, 8e-9, 16001)
    nlin = lambda n0, ng: n0 + (n0 - ng) * (lam - LAM) / LAM        # n(λ) with the given group index
    dk = 4 * np.pi * (nlin(NSH, ngs) - nlin(NP, ngp)) / lam
    p = e.shg_undepleted(b, g, dk, 900.0, 1e-3)["P_sh"]
    above = lam[p >= p.max() / 2]
    fwhm = above[-1] - above[0]
    ref = q.phase_matching(LAM, NP, NSH, ngp, ngs, L)["fwhm"]
    assert fwhm == pytest.approx(ref, rel=2e-3)


def test_depleted_uniform_tanh2():
    # test 2: Δk = 0, g = 1: P_SH = P tanh²(κ √P L), into deep depletion
    kappa, P, L = 900.0, 1.0, 3e-3
    b, g = np.array([0.0, L]), np.array([1.0])
    r = e.shg_coupled(b, g, 0.0, kappa, P, convergence_check=True)
    assert r["P_sh"][0] == pytest.approx(e.shg_depleted_matched(kappa, P, L), rel=1e-8)
    assert r["P_sh"][0] > 0.98 * P
    assert abs(r["power_residual"][0]) < 1e-9
    assert r["step_error"][0] < 1e-8


def test_depleted_qpm_approaches_tanh2_with_effective_kappa():
    # first-order QPM ≈ uniform medium with κ_eff = (2/π) κ (up to the fast-oscillating terms, O(L_c/L_NL))
    kappa, P = 900.0, 1.0
    L = 1500 * PERIOD
    b, g = e.periodic_domains(L, PERIOD)
    r = e.shg_coupled(b, g, DK, kappa, P)
    ref = e.shg_depleted_matched(2 * kappa / np.pi, P, L)
    assert r["P_sh"][0] == pytest.approx(ref, rel=2e-3)
    assert abs(r["power_residual"][0]) < 1e-7


def test_power_conservation_random_domains():
    # test 4: lossless runs conserve |A1|² + |A2|² for any domain pattern and detuning
    rng = np.random.default_rng(1)
    ell = rng.uniform(0.3, 1.7, 600) * PERIOD / 2
    b, g = e.domains_from_lengths(ell)
    dk = DK * np.array([0.9, 1.0, 1.1])
    r = e.shg_coupled(b, g, dk, 900.0, 3.0)
    assert np.all(np.abs(r["power_residual"]) < 1e-7)
    fine = e.shg_coupled(b, g, dk, 900.0, 3.0, max_step=np.pi / 32 / dk.max())
    assert np.all(np.abs(fine["power_residual"]) < np.abs(r["power_residual"]) / 10)
    assert r["P_sh"].max() > 1e-4


def test_wall_aware_vs_uniform_stepping():
    # test 5: identical when the walls lie on the uniform grid; different for random walls
    kappa, P = 900.0, 0.5
    h = PERIOD / 32
    b, g = e.periodic_domains(64 * PERIOD, PERIOD)
    rw = e.shg_coupled(b, g, DK, kappa, P, max_step=h, min_steps=1)
    ru = e.shg_coupled(b, g, DK, kappa, P, max_step=h, min_steps=1, stepping="uniform")
    assert ru["P_sh"][0] == pytest.approx(rw["P_sh"][0], rel=1e-9)
    rng = np.random.default_rng(7)
    b, g = e.domains_from_lengths(rng.uniform(0.5, 1.5, 128) * PERIOD / 2)
    exact = e.shg_coupled(b, g, DK, kappa, P, max_step=h / 8)["P_sh"][0]
    rw = e.shg_coupled(b, g, DK, kappa, P, max_step=h)["P_sh"][0]
    ru = e.shg_coupled(b, g, DK, kappa, P, max_step=h, stepping="uniform")["P_sh"][0]
    assert abs(rw - exact) / exact < 1e-6
    assert abs(ru - exact) / exact > 100 * abs(rw - exact) / exact


def test_perturbative_vs_full_at_low_conversion():
    # test 6: agree to the order of the conversion itself when it is < 1 %
    rng = np.random.default_rng(3)
    b, g = e.domains_from_lengths(rng.exponential(PERIOD / 2, 800) + 0.2 * PERIOD)
    dk = DK * np.linspace(0.8, 1.2, 9)
    for P in (1e-3, 1e-2):
        und = e.shg_undepleted(b, g, dk, 900.0, P)["P_sh"]
        full = e.shg_coupled(b, g, dk, 900.0, P)["P_sh"]
        conv = full.max() / P
        assert conv < 0.01
        assert np.all(np.abs(full - und) <= 3 * conv * und + 1e-6 * und.max())


def test_loss_undepleted_matches_full_solver():
    kappa, P = 900.0, 1e-4
    L = 300 * PERIOD
    b, g = e.periodic_domains(L, PERIOD)
    a1, a2 = 50.0, 120.0                                   # 1/m: about 2.2 and 5.2 dB/cm
    und = e.shg_undepleted(b, g, DK, kappa, P, a1, a2)
    full = e.shg_coupled(b, g, DK, kappa, P, a1, a2)
    assert full["P_sh"][0] == pytest.approx(und["P_sh"][0], rel=1e-4)
    assert full["P_pump"][0] == pytest.approx(P * np.exp(-a1 * L), rel=1e-3)        # minus the converted power
    # matched uniform medium with loss: |∫ e^{-(α1 - α2/2) z}|² e^{-α2 L}
    q_ = a2 / 2 - a1
    G = (np.exp(q_ * L) - 1) / q_
    r = e.shg_undepleted(np.array([0.0, L]), np.array([1.0]), 0.0, kappa, P, a1, a2)
    assert r["P_sh"][0] == pytest.approx(kappa**2 * P**2 * np.exp(-a2 * L) * G**2, rel=1e-12)


def test_per_domain_delta_k_table():
    b, g = e.periodic_domains(100 * PERIOD, PERIOD)
    dk = np.array([DK, 1.01 * DK])
    table = np.repeat(dk[:, None], g.size, axis=1)
    assert e.poling_spectrum(b, g, table) == pytest.approx(e.poling_spectrum(b, g, dk), rel=1e-12)
    full_t = e.shg_coupled(b, g, table, 900.0, 0.1)["P_sh"]
    full_v = e.shg_coupled(b, g, dk, 900.0, 0.1)["P_sh"]
    assert full_t == pytest.approx(full_v, rel=1e-12)


def test_ordered_shg_summary():
    r = e.ordered_shg(LAM, NP, NSH, 25e-12, 1e-12, 1e-3, 1e-3)
    assert r["p_sh_undepleted"] == pytest.approx(r["p_sh_closed"], rel=2e-3)
    assert r["p_sh"] == pytest.approx(r["p_sh_undepleted"], rel=3 * r["conversion"])
    assert abs(r["power_residual"]) < 1e-9
    off = e.ordered_shg(LAM, NP, NSH, 25e-12, 1e-12, 1e-3, 1e-3, period_offset=5e-9)
    assert off["p_sh"] < r["p_sh"]
    deep = e.ordered_shg(LAM, NP, NSH, 25e-12, 1e-12, 1e-3, 0.3)
    assert any("depletion" in a for a in deep.assumptions)


def test_rejects_bad_domains():
    with pytest.raises(ValueError, match="increasing"):
        e.shg_undepleted(np.array([0.0, 2e-6, 1e-6]), np.array([1.0, -1.0]), DK, 900.0, 1e-3)
    with pytest.raises(ValueError, match="one value per domain"):
        e.shg_undepleted(np.array([0.0, 1e-6]), np.array([1.0, -1.0]), DK, 900.0, 1e-3)
    with pytest.raises(ValueError, match="g must lie"):
        e.shg_undepleted(np.array([0.0, 1e-6]), np.array([2.0]), DK, 900.0, 1e-3)
