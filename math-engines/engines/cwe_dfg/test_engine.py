import math

import numpy as np
import pytest

from engines.cwe_dfg import engine as e
from engines.cwe_shg import engine as shg

LP, LS = 775e-9, 1500e-9
LI = 1 / (1 / LP - 1 / LS)
NP, NS, NI = 2.10, 1.91, 1.90
DK = 2 * np.pi * (NP / LP - NS / LS - NI / LI)
PER = 2 * np.pi / DK
KS, KI = 800.0, 800.0 * LS / LI
KP = KS * LS / LP


def test_coupling_relations_and_shg_limit():
    c = e.coupling_three_wave(LS, LI, NS, NI, NP, 25e-12, 1e-12)
    assert c["kappa_pump"] == pytest.approx(c["kappa_signal"] + c["kappa_idler"], rel=1e-12)   # κ ∝ ω
    assert c["wavelength_pump"] == pytest.approx(LP)
    # degenerate SFG of two equal waves: η_SFG = 4 η0(SHG at λ = 2 λ_p)
    d = e.coupling_three_wave(1.55e-6, 1.55e-6, 2.0, 2.0, 2.1, 25e-12, 1e-12)
    s = shg.coupling_shg(1.55e-6, 2.0, 2.1, 25e-12, 1e-12)
    assert d["eta_sfg"] == pytest.approx(4 * s["eta_norm"], rel=1e-12)
    # Γ² = 8π² d² P / (ε0 c n_s n_i n_p λ_s λ_i A)
    hand = 8 * math.pi**2 * (25e-12) ** 2 / (8.8541878128e-12 * 299792458.0 * NS * NI * NP * LS * LI * 1e-12)
    assert c["gain_coeff"] ** 2 == pytest.approx(hand, rel=1e-12)


@pytest.mark.parametrize("dk", [0.0, 600.0, 1900.0, 4000.0])
def test_uniform_gain_matches_closed_form(dk):
    # test 3: cosh² at Δk = 0, the g = √(Γ² - Δk²/4) formula otherwise (Γ = 1000/m, ΓL = 3)
    P, L = 1.0, 3e-3
    gamma = math.sqrt(KS * KI * P)
    b, g = np.array([0.0, L]), np.array([1.0])
    ref = e.opa_gain_uniform(gamma, dk, L)
    if dk == 0:
        assert ref == pytest.approx(math.cosh(gamma * L) ** 2, rel=1e-12)
    tm = e.opa_transfer(b, g, dk, KS, KI, P, 1e-9)
    assert tm["gain"][0] == pytest.approx(ref, rel=1e-10)
    full = e.three_wave(b, g, dk, KS, KI, KP, P, 1e-9)
    assert full["gain"][0] == pytest.approx(ref, rel=1e-6)
    # idler photons = signal photons gained
    assert tm["P_idler"][0] / KI == pytest.approx((tm["P_signal"][0] - 1e-9) / KS, rel=1e-9)


def test_qpm_gain_approaches_effective_gamma():
    P = 1.0
    L = 1000 * PER
    b, g = e.periodic_domains(L, PER)
    gamma = math.sqrt(KS * KI * P)
    tm = e.opa_transfer(b, g, DK, KS, KI, P, 1e-9)
    ref = e.opa_gain_uniform(2 / np.pi * gamma, 0.0, L)
    assert 10 * np.log10(tm["gain"][0]) == pytest.approx(10 * np.log10(ref), rel=5e-3)


def test_transfer_matrix_vs_rk4_random_domains_and_loss():
    rng = np.random.default_rng(2)
    b, g = e.domains_from_lengths(rng.uniform(0.6, 1.4, 500) * PER / 2)
    dk = DK * np.array([0.98, 1.0, 1.03])
    for (aS, aI) in ((0.0, 0.0), (20.0, 60.0)):
        tm = e.opa_transfer(b, g, dk, KS, KI, 0.8, 1e-9, alpha_signal=aS, alpha_idler=aI)
        full = e.three_wave(b, g, dk, KS, KI, KP, 0.8, 1e-9, alpha_signal=aS, alpha_idler=aI)
        assert full["P_signal"] == pytest.approx(tm["P_signal"], rel=1e-6)
        assert full["P_idler"] == pytest.approx(tm["P_idler"], rel=1e-6)


def test_depletion_conservation_and_saturation():
    # test 4: power and Manley-Rowe conserved in the lossless depleted regime; strong seed saturates the gain
    b, g = e.periodic_domains(800 * PER, PER)
    weak = e.three_wave(b, g, DK, KS, KI, KP, 1.0, 1e-6)
    strong = e.three_wave(b, g, DK, KS, KI, KP, 1.0, 0.05)
    assert abs(strong["power_residual"][0]) < 1e-8 and abs(strong["manley_rowe"][0]) < 1e-8
    assert strong["P_pump"][0] < 0.7
    assert strong["gain"][0] < 0.9 * weak["gain"][0]


def test_low_gain_dfg_is_the_poling_spectrum():
    # P_i = κ_i² P_p P_s |G|² in the low-gain limit: the same |G|² as SHG design
    b, g = e.periodic_domains(200 * PER, PER)
    dk = DK * np.linspace(0.99, 1.01, 7)
    P, Ps = 1e-3, 1e-3
    tm = e.opa_transfer(b, g, dk, KS, KI, P, Ps)
    G = shg.poling_spectrum(b, g, dk)
    assert tm["P_idler"] == pytest.approx(KI**2 * P * Ps * np.abs(G) ** 2, rel=2e-3)


def test_per_domain_delta_k_table():
    b, g = e.periodic_domains(100 * PER, PER)
    dk = np.array([DK, 1.01 * DK])
    tab = np.repeat(dk[:, None], g.size, axis=1)
    a = e.opa_transfer(b, g, tab, KS, KI, 1.0, 1e-6)["gain"]
    c = e.opa_transfer(b, g, dk, KS, KI, 1.0, 1e-6)["gain"]
    assert a == pytest.approx(c, rel=1e-12)


def test_ordered_opa_summary_and_warnings():
    r = e.ordered_opa(LP, LS, NP, NS, NI, 25e-12, 1e-12, 2e-3, 0.3, 1e-6)
    assert r["period"] == pytest.approx(PER, rel=1e-12)
    assert r["gain_db"] == pytest.approx(r["gain_undepleted_db"], abs=0.01)
    assert r["gain_db"] == pytest.approx(r["gain_closed_db"], abs=0.05)
    assert abs(r["power_residual"]) < 1e-9 and abs(r["manley_rowe"]) < 1e-9
    deg = e.ordered_opa(775e-9, 1549.5e-9, 2.10, 1.91, 1.91, 25e-12, 1e-12, 1e-3, 0.1, 1e-6)
    assert any("degeneracy" in a for a in deg.assumptions)
    with pytest.raises(ValueError, match="longer than the pump"):
        e.idler_wavelength(1.55e-6, 1.0e-6)


def test_per_row_kappa_equals_separate_runs():
    b, g = e.periodic_domains(150 * PER, PER)
    dk = DK * np.array([0.995, 1.0, 1.004])
    ks = np.array([700.0, 800.0, 900.0])
    ki = ks * LS / LI
    joint = e.opa_transfer(b, g, dk, ks, ki, 1.0, 1e-6)["gain"]
    for j in range(3):
        assert joint[j] == pytest.approx(e.opa_transfer(b, g, dk[j], ks[j], ki[j], 1.0, 1e-6)["gain"][0], rel=1e-12)
