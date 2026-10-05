import numpy as np
import pytest

from engines.membrane_mode import engine as mm

LAM = 1.55e-6


def test_thin_te_membrane_puts_almost_all_power_in_the_gas():
    assert mm.solve_mode(LAM, 5e-9, material="si").Gamma > 0.9


def test_tm_couples_more_than_te():
    te = mm.solve_mode(LAM, 220e-9, material="si", polarization="TE")
    tm = mm.solve_mode(LAM, 220e-9, material="si", polarization="TM")
    assert tm.Gamma > te.Gamma
    assert te.Gamma == pytest.approx(0.062, abs=2e-3)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_gamma_matches_brute_force_quadrature(pol):
    m = mm.solve_mode(LAM, 300e-9, material="si", polarization=pol)
    x, F2 = mm.field_profile(m, 14.0, 320001)
    F = np.sqrt(F2)
    outside = np.abs(x) > 150e-9
    k = 2 * np.pi / LAM
    beta = k * m.neff
    trap = getattr(np, "trapezoid", None) or np.trapz
    if pol == "TE":
        g = trap(F2 * outside, x) / trap(F2, x) / m.neff
    else:
        n2 = np.where(outside, 1.0, m.n_core**2)
        E2 = (beta**2 * F2 + np.gradient(F, x) ** 2) / n2**2
        g = trap(E2 * outside, x) / (k * beta * trap(F2 / n2, x))
    assert g == pytest.approx(m.Gamma, rel=3e-3)


def test_te_group_index_energy_identity():
    # non-dispersive TE slab: n_g n_eff = sum_i n_i^2 (share of ∫E² in layer i)
    m = mm.solve_mode(LAM, 400e-9, n_core=2.0, polarization="TE")
    expect = (2.0**2 * (1 - m.f_clad_E) + 1.0 * m.f_clad_E) / m.neff
    assert m.n_group == pytest.approx(expect, rel=1e-5)


def test_material_dispersion_raises_group_index():
    a = mm.solve_mode(LAM, 300e-9, material="si3n4")
    b = mm.solve_mode(LAM, 300e-9, n_core=a.n_core)
    assert a.neff == pytest.approx(b.neff, rel=1e-12)
    assert a.n_group > b.n_group


def test_penetration_depth_and_cutoff():
    m = mm.solve_mode(LAM, 250e-9, material="si", polarization="TM")
    assert m.z_p == pytest.approx(1 / (2 * m.gamma_ev))
    assert m.gamma_ev == pytest.approx(2 * np.pi / LAM * np.sqrt(m.neff**2 - 1))
    odd = mm.solve_mode(LAM, 100e-9, material="si", order=1)
    assert not odd.guided and np.isnan(odd.Gamma)


def test_roughness_scaling_is_relative():
    ref = mm.solve_mode(LAM, 220e-9, material="si")
    thin = mm.solve_mode(LAM, 150e-9, material="si")
    assert mm.roughness_scaled_alpha(ref, ref, 2.3) == pytest.approx(2.3)
    assert mm.roughness_scaled_alpha(thin, ref, 2.3, index_contrast=False) == pytest.approx(2.3 * thin.E_edge2 / ref.E_edge2)


def test_result_front_ends():
    r = mm.evanescent_mode(LAM, 250e-9, material="si", polarization="TM")
    assert r["guided"] and 0 < r["Gamma"] < 2 and r.units["z_p"] == "m"
    v = mm.evanescent_volume(300e-9, 1e-4)
    assert v["V_ev"] == pytest.approx(6e-11)
    assert v["molecules_per_ppb"] == pytest.approx(2.5e25 * 6e-11 * 1e-9)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="material"):
        mm.evanescent_mode(LAM, 250e-9, material="unobtainium")
    with pytest.raises(ValueError, match="thickness"):
        mm.evanescent_mode(LAM, 0.0)
