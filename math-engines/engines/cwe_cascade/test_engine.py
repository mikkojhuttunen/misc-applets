import math

import numpy as np
import pytest

from engines.cwe_cascade import engine as e
from engines.cwe_dfg import engine as dfg
from engines.cwe_shg import engine as shg

LAM, N1, N2, N3 = 1.55e-6, 1.9109, 2.1027, 2.20
DKA = 4 * np.pi * (N2 - N1) / LAM
DKB = 2 * np.pi * (3 * N3 - 2 * N2 - N1) / LAM
PA, PB = 2 * np.pi / DKA, 2 * np.pi / DKB
KA, K3 = 900.0, 1500.0
K1, K2 = K3 / 3, 2 * K3 / 3


def test_coupling_is_shg_plus_sfg():
    c = e.coupling_thg(LAM, N1, N2, N3, 22e-12, 20e-12, 0.9e-12, 1.2e-12)
    s = shg.coupling_shg(LAM, N1, N2, 22e-12, 0.9e-12)
    d = dfg.coupling_three_wave(LAM, LAM / 2, N1, N2, N3, 20e-12, 1.2e-12)
    assert c["kappa_a"] == pytest.approx(float(s["kappa"]))
    assert (c["kappa_1"], c["kappa_2"], c["kappa_3"]) == pytest.approx(
        (float(d["kappa_signal"]), float(d["kappa_idler"]), float(d["kappa_pump"])))
    assert c["kappa_3"] == pytest.approx(3 * c["kappa_1"]) and c["kappa_2"] == pytest.approx(2 * c["kappa_1"])


def test_uniform_double_matched_l4_law():
    L, P = 2e-3, 1e-4
    b, g = np.array([0.0, L]), np.array([1.0])
    r = e.cascade_undepleted(b, g, 0.0, 0.0, KA, K3, P)
    assert r["P_th"][0] == pytest.approx((K3 * KA) ** 2 * P**3 * L**4 / 4, rel=1e-12)
    f = e.cascade_coupled(b, g, 0.0, 0.0, KA, K1, K2, K3, P)
    assert f["P3"][0] == pytest.approx(r["P_th"][0], rel=2e-3)        # depletion of order η P L² ~ 3e-4


def test_nested_integral_matches_brute_force_quadrature():
    rng = np.random.default_rng(5)
    b, g = shg.domains_from_lengths(rng.uniform(0.4, 1.6, 40) * PA / 2)
    g = g * rng.uniform(0.3, 1.0, g.size)                     # continuous values too
    qa, qb = np.array([DKA, 0.8 * DKA, 1e-3]), np.array([DKB, 1.1 * DKB, 0.5 * DKB])
    J, Ga = e.cascade_integral(b, g, qa, qb, gain_a=-30.0, gain_b=20.0)
    z = np.linspace(0, b[-1], 400001)
    gz = g[np.minimum(np.searchsorted(b, z, side="right") - 1, g.size - 1)]
    for m in range(3):
        fa = gz * np.exp(1j * qa[m] * z - 30.0 * z)
        inner = np.concatenate([[0], np.cumsum(0.5 * (fa[1:] + fa[:-1]) * np.diff(z))])
        fo = gz * np.exp(1j * qb[m] * z + 20.0 * z) * inner
        Jq = np.sum(0.5 * (fo[1:] + fo[:-1]) * np.diff(z))
        assert J[m] == pytest.approx(Jq, rel=2e-4)
        assert Ga[m] == pytest.approx(shg.poling_spectrum(b, g, qa[m], gain=-30.0)[0], rel=1e-9)


def test_two_section_closed_form_and_split_optimum():
    L, P = 3e-3, 1e-3
    vals = []
    for split in (0.3, 0.5, 0.7):
        b, g = e.two_section_domains(L, PA, PB, split)
        r = e.cascade_undepleted(b, g, DKA, DKB, KA, K3, P)
        closed = (2 / np.pi) ** 4 * K3**2 * KA**2 * P**3 * (split * L) ** 2 * ((1 - split) * L) ** 2
        assert r["P_th"][0] == pytest.approx(closed, rel=0.02)
        vals.append(r["P_th"][0])
    assert vals[1] > vals[0] and vals[1] > vals[2]


def test_perturbative_vs_rk4_and_conservation():
    b, g = e.two_section_domains(1.5e-3, PA, PB, 0.5)
    dka, dkb = DKA * np.array([0.999, 1.0, 1.001]), DKB * np.array([0.999, 1.0, 1.001])
    und = e.cascade_undepleted(b, g, dka, dkb, KA, K3, 2e-3)
    full = e.cascade_coupled(b, g, dka, dkb, KA, K1, K2, K3, 2e-3)
    assert full["P3"] == pytest.approx(und["P_th"], rel=0.01)
    assert full["P2"] == pytest.approx(und["P_sh"], rel=0.01)
    strong = e.cascade_coupled(b, g, DKA, DKB, KA, K1, K2, K3, 2.0)
    assert abs(strong["power_residual"][0]) < 1e-7
    assert strong["P3"][0] < e.cascade_undepleted(b, g, DKA, DKB, KA, K3, 2.0)["P_th"][0]


def test_loss_in_undepleted_matches_rk4():
    b, g = e.two_section_domains(1.5e-3, PA, PB, 0.5)
    al = (20.0, 60.0, 150.0)
    und = e.cascade_undepleted(b, g, DKA, DKB, KA, K3, 1e-3, al)
    full = e.cascade_coupled(b, g, DKA, DKB, KA, K1, K2, K3, 1e-3, al)
    assert full["P3"][0] == pytest.approx(und["P_th"][0], rel=3e-3)


def test_commensurate_orders_finds_exact_ratio():
    rows = e.commensurate_orders(1.0e6, 3.0e6, max_order=5)
    best = rows[0]
    assert (best["m_a"], best["m_b"]) == (1, 3) and abs(best["residual"]) < 1e-6
    assert best["strength"] == pytest.approx(max(abs(2 * np.sin(np.pi * d) / np.pi * 2 * np.sin(3 * np.pi * d) / (3 * np.pi))
                                                 for d in np.linspace(0.05, 0.95, 91)))


def test_per_domain_tables_and_summary():
    b, g = e.two_section_domains(1e-3, PA, PB, 0.5)
    ta = np.repeat(np.array([[DKA]]), g.size, axis=1)
    tb = np.repeat(np.array([[DKB]]), g.size, axis=1)
    assert e.cascade_undepleted(b, g, ta, tb, KA, K3, 1e-3)["P_th"] == pytest.approx(
        e.cascade_undepleted(b, g, DKA, DKB, KA, K3, 1e-3)["P_th"], rel=1e-12)
    r = e.two_section_thg(LAM, N1, N2, N3, 22e-12, 22e-12, 0.9e-12, 1.2e-12, 2e-3, 0.005)
    assert r["period_a"] == pytest.approx(PA) and r["period_b"] == pytest.approx(PB)
    assert r["p_th"] == pytest.approx(r["p_th_undepleted"], rel=0.02)
    assert r["p_th_undepleted"] == pytest.approx(r["p_th_closed"], rel=0.02)
    hi = e.two_section_thg(LAM, N1, N2, N3, 22e-12, 22e-12, 0.9e-12, 1.2e-12, 2e-3, 1.0)
    assert any("depletion" in a for a in hi.assumptions)
