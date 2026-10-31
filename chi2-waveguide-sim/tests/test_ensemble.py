"""Plan section 7, tests 7 (random QPM statistics) and 8 (wall jitter)."""
import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import ensemble as E
from chi2sim import poling as P

K0 = 4 * np.pi * (2.1924 - 2.0272) / 1.55e-6
L0 = np.pi / K0                     # domain length of the ordered first-order grating


def test_exponential_lengths_closed_form_by_hand():
    # φ = 1/(1 - iqμ): ⟨|G|²⟩ = 4 N μ² / (4 + q²μ²), N = L/μ
    mu, L = 2e-6, 1e-3
    d = P.LengthDistribution("exponential", mu)
    q = np.array([1e4, 3e5, 1e6])
    assert E.renewal_mean(d, q, L) == pytest.approx(4 * (L / mu) * mu**2 / (4 + (q * mu) ** 2), rel=1e-12)


def test_random_qpm_mean_matches_closed_form_monte_carlo():
    d = P.LengthDistribution("gaussian", L0, 0.25 * L0)
    L = 600 * L0
    q = K0 * np.linspace(0.7, 1.3, 61)
    mc = E.monte_carlo(lambda r: P.random_domains(L, d, r), 600, q, seed=1)
    th = E.renewal_mean(d, q, L)
    ratio = mc["mean"] / th
    assert ratio.mean() == pytest.approx(1.0, abs=0.02)        # MC error of the band average ≈ 1/√(600·~10)
    assert np.all(np.abs(ratio - 1) < 0.2)


def test_random_qpm_scales_linearly_with_domain_number_and_is_speckle():
    d = P.LengthDistribution("gaussian", L0, 0.4 * L0)
    q = K0 * np.array([0.6, 0.8, 1.0, 1.25])
    m1 = E.monte_carlo(lambda r: P.random_domains(200 * L0, d, r), 2500, q, seed=2)
    m4 = E.monte_carlo(lambda r: P.random_domains(800 * L0, d, r), 2500, q, seed=3)
    # each mean has a 1/√2500 = 2 % MC error; the closed form is exactly linear in N
    assert (m4["mean"] / m1["mean"]) == pytest.approx(np.full(4, 4.0), rel=0.09)
    assert m4["mean"] == pytest.approx(E.renewal_mean(d, q, 800 * L0), rel=0.08)
    # fully developed speckle: |G|² exponentially distributed, σ/mean → 1
    assert m4["contrast"] == pytest.approx(np.ones(4), abs=0.1)


def test_wall_jitter_exact_mean_and_peak_reduction():
    base = P.ordered(400 * 2 * L0, 2 * L0)
    sigma = 0.08 * L0
    q = K0 * np.linspace(0.995, 1.005, 11)
    exact = E.jitter_mean(base, sigma, q)
    mc = E.monte_carlo(lambda r: P.jittered(base, sigma, r), 1500, q, seed=4)
    assert mc["mean"] == pytest.approx(exact, rel=0.03)
    # peak: coherent part (2/π L)² e^{-K²σ²} plus incoherent background (1 - e^{-K²σ²}) N 4/K²
    chi2 = np.exp(-(K0 * sigma) ** 2)
    n_walls = base.walls.size
    hand = (2 / np.pi * base.length) ** 2 * chi2 + (1 - chi2) * n_walls * 4 / K0**2
    assert exact[5] == pytest.approx(hand, rel=0.01)
    assert exact[5] / (2 / np.pi * base.length) ** 2 == pytest.approx(chi2, rel=0.02)
