import numpy as np
import pytest

from engines.erbium_amplifier import engine as er

LP, LS = 980e-9, 1532e-9


def test_mccumber_equal_cross_sections_at_zero_line():
    # McCumber: σ_e = σ_a where hν equals the zero-line energy ε (λ0 = 1532 nm)
    r = er.cross_sections(er.LAMBDA_ZERO)
    assert r["sigma_e"] == pytest.approx(r["sigma_a"], rel=1e-12)


def test_mccumber_ratio_at_1600_nm_by_hand():
    # hc/kT at 295 K = 1.438777e-2 / 295 = 4.8772e-5 m; (1/1.532 µm - 1/1.6 µm) = 27741.5 1/m
    # σ_e/σ_a = exp(4.8772e-5 · 27741.5) = exp(1.3530) = 3.869
    r = er.cross_sections(1.6e-6)
    assert r["sigma_e"] / r["sigma_a"] == pytest.approx(3.869, rel=2e-3)


def test_peak_absorption_equals_the_set_peak():
    lam = np.linspace(1.45e-6, 1.65e-6, 4001)
    assert er.sigma_absorption(lam, 5.7e-25).max() == pytest.approx(5.7e-25, rel=1e-4)
    assert abs(lam[np.argmax(er.sigma_absorption(lam))] - 1533e-9) < 1e-9


def test_concentration_effects_by_hand():
    r = er.concentration_effects(2e26, 4e-24, "proportional", "proportional", k_q=0.05)
    assert r["c_up_eff"] == pytest.approx(8e-24)
    assert r["f_q"] == pytest.approx(0.10)
    assert er.concentration_effects(2e26, 4e-24, "fixed", "none")["f_q"] == 0.0
    assert er.concentration_effects(1e28, 4e-24, quenching="proportional", k_q=0.05)["f_q"] == 0.95


def test_upper_population_without_upconversion_is_the_two_level_formula():
    # C_up = 0: N2 = R_up N / (R_up + R_down + 1/τ)
    ru, rd, n, tau = 2e3, 500.0, 1e26, 7.5e-3
    want = ru * n / (ru + rd + 1 / tau)
    assert er.upper_population(ru, rd, n, tau, 0.0) == pytest.approx(want, rel=1e-12)


def test_upper_population_solves_the_quadratic():
    ru, rd, n, tau, cup = 2e3, 500.0, 1e26, 7.5e-3, 4e-24
    n2 = er.upper_population(ru, rd, n, tau, cup)
    assert ru * (n - n2) - rd * n2 - n2 / tau - cup * n2 * n2 == pytest.approx(0.0, abs=1e-9 * ru * n)


def test_in_band_pumping_limits_inversion_to_sigma_ratio():
    # strong 1480 nm pump only: N2/N -> σ_a,p / (σ_a,p + σ_e,p)
    sa, se = er.pump_sigmas(1480e-9)
    ru, rd = sa * 1e38, se * 1e38
    assert er.upper_population(ru, rd, 1.0, 7.5e-3, 0.0) == pytest.approx(sa / (sa + se), rel=1e-6)


def test_unpumped_weak_signal_follows_beer_lambert():
    # no pump, weak signal: P_s(L)/P_s(0) = exp(-(σ_a N Γ + α) L)
    n, gam, L, alpha = 1.5e26, 0.3, 0.03, 5.0
    sa = float(er.sigma_absorption(LS))
    r = er.propagate(0.0, 1e-12, L, [0.4], [gam], [1e-12], n, 7.5e-3, 0.0, 0.0, 1.7e-25, 0.0, sa, float(er.sigma_emission(LS)), LP, LS, alpha)
    assert r["gain_ln"] == pytest.approx(-(sa * n * gam + alpha) * L, rel=1e-5)


def test_quenched_ions_cap_the_fully_inverted_gain():
    # very strong 980 nm pump, no upconversion: active ions fully inverted, quenched ions absorb
    # g = Γ (σ_e N (1 - f_q) - σ_a N f_q)
    n, gam, fq, L = 1e26, 0.5, 0.2, 1e-3
    sa, se = float(er.sigma_absorption(LS)), float(er.sigma_emission(LS))
    r = er.propagate(1e3, 1e-9, L, [0.5], [gam], [1e-12], n, 7.5e-3, 0.0, fq, 1.7e-25, 0.0, sa, se, LP, LS)
    assert r["gain_ln"] == pytest.approx(gam * (se * n * (1 - fq) - sa * n * fq) * L, rel=1e-4)


def test_probe_gain_reproduces_the_signal_gain():
    # independent route: the signal's own gain equals the probe formula at λ_s
    sa, se = float(er.sigma_absorption(LS)), float(er.sigma_emission(LS))
    r = er.propagate(0.05, 1e-6, 0.03, [0.2, 0.15], [0.15, 0.12], [4e-13, 4e-13], 1.5e26, 7.5e-3, 6e-24, 0.075,
                     1.7e-25, 0.0, sa, se, LP, LS, alpha=5.76)
    assert er.probe_gain_ln(LS, r["I1"], r["I2"], 0.03, alpha=5.76) == pytest.approx(r["gain_ln"], rel=1e-3)


def test_uniform_amplifier_gains_with_pump_and_validates_input():
    kw = dict(pump_wavelength=LP, signal_wavelength=LS, signal_power=1e-6, length=0.03, n_er=1.5e26,
              gamma_pump=0.4, gamma_signal=0.3, doped_area=8e-13, alpha=5.76)
    assert er.uniform_amplifier(pump_power=0.05, **kw)["gain_ln"] > 0 > er.uniform_amplifier(pump_power=0.0, **kw)["gain_ln"]
    with pytest.raises(ValueError, match="gamma_signal"):
        er.uniform_amplifier(pump_power=0.05, **{**kw, "gamma_signal": 1.5})
