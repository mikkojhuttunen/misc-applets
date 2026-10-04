import numpy as np
import pytest

from engines.ase_noise import engine as an
from engines.materials import engine as mat
from engines.multimode_amplifier import engine as mm
from engines.step_index_fiber import engine as sif

LAM = 1.064e-6
YB = dict(dopant_density=6e25, sigma_es=2.5e-25, sigma_as=5e-27, sigma_ep=2.5e-24, sigma_ap=2.6e-24,
          lifetime=1e-3, pump_wavelength=0.976e-6)


def test_lp01_matches_step_index_fiber_engine():
    # independent route: the LP01 solver in step_index_fiber (J0/J1 form, Sellmeier cladding)
    a, dn = 4.1e-6, 0.005
    ncl = float(mat.index("sio2", 1.55e-6))
    na = np.sqrt((ncl + dn) ** 2 - ncl**2)
    s = mm.step_index_modes(a, na, 1.55e-6, n_clad=ncl)
    assert s.labels == [(0, 1)]
    assert s.beta[0] * 1.55e-6 / (2 * np.pi) == pytest.approx(sif.lp01(1.55e-6, a, dn)["neff"], rel=1e-10)


def test_lp11_cutoff_at_2405():
    lam, a = 1.0e-6, 3.0e-6
    na_cut = 2.404825557695773 * lam / (2 * np.pi * a)
    assert mm.step_index_modes(a, 0.99 * na_cut, lam).n_modes == 1
    s = mm.step_index_modes(a, 1.01 * na_cut, lam)
    assert s.labels == [(0, 1), (1, 1)] and s.n_modes == 3


def test_mode_counts_approach_v2_rules():
    a, na = 50e-6, 0.2
    V = 2 * np.pi * a * na / LAM
    assert mm.step_index_modes(a, na, LAM).n_modes == pytest.approx(V**2 / 4, rel=0.05)
    G = int(V / 2)
    assert mm.grin_modes(a, na, LAM).n_modes == G * (G + 1) // 2
    p = mm.planar_modes(300e-6, 6e-6, 0.12, LAM)
    k = 2 * np.pi / LAM * 0.12
    assert p.n_modes == sum(1 for m in range(1, 200) for n in range(1, 20)
                            if (m * np.pi / 300e-6) ** 2 + (n * np.pi / 6e-6) ** 2 <= k * k)


def test_intensities_normalised_and_power_fraction_in_core():
    s = mm.step_index_modes(25e-6, 0.1, LAM)
    assert np.allclose(mm.overlaps(s, np.ones_like(s.weights)), 1.0)
    gam = mm.overlaps(s, mm.disk_profile(s, 25e-6))
    assert gam[0] > 0.99 and np.all(gam < 1) and gam[-1] < gam[0]
    # LP01 core power fraction, textbook form Γ = 1 - (U/V)² (1 - K0(W)²/K1(W)²)
    from scipy.optimize import brentq
    from scipy.special import j0, j1, k0, k1
    V = 2 * np.pi * 25e-6 * 0.1 / LAM
    f = lambda U: U * j1(U) / j0(U) - np.sqrt(V * V - U * U) * k1(np.sqrt(V * V - U * U)) / k0(np.sqrt(V * V - U * U))
    U = brentq(f, 1e-6, 2.404)
    W = np.sqrt(V * V - U * U)
    gamma = 1 - (U / V) ** 2 * (1 - k0(W) ** 2 / k1(W) ** 2)
    assert gam[0] == pytest.approx(gamma, rel=1e-4)


def test_grin_lg00_overlap_hand_value():
    g = mm.grin_modes(25e-6, 0.1, LAM, extent=3.0)
    w0 = np.sqrt(2 * 25e-6 / (2 * np.pi / LAM * 0.1))
    rho = 10e-6
    assert mm.overlaps(g, mm.disk_profile(g, rho))[0] == pytest.approx(1 - np.exp(-2 * rho**2 / w0**2), rel=1e-4)


def test_planar_overlap_hand_value():
    p = mm.planar_modes(100e-6, 5e-6, 0.12, LAM)
    gam = mm.overlaps(p, mm.box_profile(p, 50e-6, 5e-6))
    i = p.labels.index((1, 1))
    assert gam[i] == pytest.approx(0.5 + 1 / np.pi, rel=1e-3)   # ∫_{1/4}^{3/4} 2 sin²(πx) dx


def test_uncoupled_channel_is_analytic():
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    ge, ga, al = mm.small_signal_rates(s, mm.disk_profile(s, 15e-6), 3.0, 1.3)
    ch = mm.linear_channel(s, ge, ga, al, 2.0, sections=7)
    fam = s.family_of_mode
    G = np.exp((ge - ga)[fam] * 2.0)
    assert np.allclose(np.real(np.diag(ch.T)) ** 2, G, rtol=1e-12)
    assert np.allclose(np.real(np.diag(ch.D)), 1.3 * (G - 1), rtol=1e-10)
    mn = mm.modal_noise(ch)
    assert np.allclose(mn["noise_figure"], 1 / G + 2 * 1.3 * (G - 1) / G, rtol=1e-10)


def test_distributed_loss_noise_hand_value():
    # constant γe, γa, α: occupation γe (G - 1) / g with g = γe - γa - α (exact Lindblad solution)
    s = mm.step_index_modes(3e-6, 0.05, LAM)
    ch = mm.linear_channel(s, [2.0], [0.5], [0.3], 3.0, sections=5)
    g = 2.0 - 0.5 - 0.3
    assert np.real(ch.D[0, 0]) == pytest.approx(2.0 * np.expm1(3.0 * g) / g, rel=1e-12)


def test_coupling_unitary_is_unitary_and_respects_degeneracy():
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    U = mm.coupling_unitary(s, 0.7, 0.0, rng=3)
    assert np.allclose(U @ U.conj().T, np.eye(s.n_modes), atol=1e-12)
    fam = s.family_of_mode
    off = np.abs(U[fam[:, None] != fam[None, :]])
    assert off.max() < 1e-12
    assert np.allclose(mm.coupling_unitary(s, 0.7, 1e3, rng=3), mm.coupling_unitary(s, 0.7, 1e3, rng=3))


def test_coupling_with_equal_gain_has_no_mdg_and_conserves_ase():
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    n = s.n_families
    ch = mm.linear_channel(s, np.full(n, 2.0), np.full(n, 0.5), np.zeros(n), 2.0, sections=10,
                           coupling_strength=1.0, beta_correlation=1e9)
    m = mm.mode_dependent_gain(ch)
    assert m["mdg"] == pytest.approx(1.0, abs=1e-9)
    ref = mm.linear_channel(s, np.full(n, 2.0), np.full(n, 0.5), np.zeros(n), 2.0)
    assert np.trace(ch.D).real == pytest.approx(np.trace(ref.D).real, rel=1e-10)


def test_strong_coupling_reduces_mdg_growth():
    # Ho & Kahn: uncoupled log-gain spread grows ∝ L, strongly coupled ∝ sqrt(L)
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    ge, ga, al = mm.small_signal_rates(s, mm.disk_profile(s, 7e-6), 1.0, 1.2)
    unc = [mm.mode_dependent_gain(mm.linear_channel(s, ge, ga, al, L))["log_gain_std"] for L in (1.0, 4.0)]
    assert unc[1] / unc[0] == pytest.approx(4.0, rel=1e-9)
    cpl = []
    for L, k in ((1.0, 16), (4.0, 64)):
        cpl.append(np.mean([mm.mode_dependent_gain(mm.linear_channel(s, ge, ga, al, L, sections=k, coupling_strength=3.0,
                                                                    beta_correlation=1e9, seed=sd))["log_gain_std"]
                            for sd in range(6)]))
    assert cpl[0] < unc[0] and 1.5 < cpl[1] / cpl[0] < 2.6


def test_saturated_photon_budget_and_small_signal_limit():
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    prof = mm.disk_profile(s, 15e-6)
    A = np.pi * (100e-6) ** 2
    r = mm.saturated_amplifier(s, 2.0, 20.0, 0.5, prof, cladding_area=A, ase_bandwidth=0.0, steps=400, **YB)
    hs, hp = an.photon_energy(LAM), an.photon_energy(YB["pump_wavelength"])
    lost = (r["pump_power"][0] - r["pump_power"][-1]) / hp - (r["signal_power"][-1] - r["signal_power"][0]) / hs
    decay = np.sum(r["n2_area"]) * (2.0 / 400) / YB["lifetime"]
    assert lost == pytest.approx(decay, rel=1e-4)
    # tiny signal, no pump depletion effect on gain shape: output gain equals the linear channel gain
    weak = mm.saturated_amplifier(s, 2.0, 20.0, 1e-9, prof, cladding_area=A, ase_bandwidth=0.0, steps=100, **YB)
    G = weak["signal_power"][-1] / weak["signal_power"][0]
    assert G == pytest.approx(np.abs(weak["channel"].T[0, 0]) ** 2, rel=1e-9)


def test_spatial_hole_burning_raises_higher_order_gain():
    s = mm.step_index_modes(15e-6, 0.1, LAM)
    prof = mm.disk_profile(s, 15e-6)
    A = np.pi * (100e-6) ** 2
    lo = mm.saturated_amplifier(s, 2.0, 20.0, 1e-6, prof, cladding_area=A, steps=100, **YB)
    hi = mm.saturated_amplifier(s, 2.0, 20.0, 5.0, prof, cladding_area=A, steps=100, **YB)
    g_lo, g_hi = (mm.modal_noise(x["channel"])["gain"] for x in (lo, hi))
    assert g_lo[0] >= g_lo.max() * 0.999          # unsaturated: LP01 has the best overlap
    assert g_hi.max() > g_hi[0]                  # saturated by LP01: a higher-order mode wins


def test_coupled_saturated_matches_uncoupled_when_strength_zero_like():
    s = mm.step_index_modes(10e-6, 0.1, LAM)
    prof = mm.disk_profile(s, 10e-6)
    A = np.pi * (80e-6) ** 2
    a = mm.saturated_amplifier(s, 1.0, 10.0, 0.1, prof, cladding_area=A, steps=50, **YB)
    b = mm.saturated_amplifier(s, 1.0, 10.0, 0.1, prof, cladding_area=A, steps=50, coupling_strength=1e-12, **YB)
    assert b["signal_power"][-1] == pytest.approx(a["signal_power"][-1], rel=1e-8)
    assert b["ase_power"][-1] == pytest.approx(a["ase_power"][-1], rel=1e-8)


def test_validation():
    with pytest.raises(ValueError, match="waveguide"):
        mm.multimode_small_signal("hollow", 25e-6, 0.1, LAM, 1.0, 5.0, 1.2, 1.0)
    with pytest.raises(ValueError, match="n_sp"):
        mm.small_signal_rates(mm.grin_modes(25e-6, 0.1, LAM), 1.0, 5.0, 0.5)
    with pytest.raises(ValueError, match="no guided mode"):
        mm.planar_modes(1e-6, 1e-6, 0.05, LAM)
