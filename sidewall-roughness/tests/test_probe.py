import numpy as np
import pytest

from swr.analysis.probe import (alpha_from_kernel, alpha_from_power, kernel_from_probe,
                                periods_in_band, radiation_band, to_db_per_cm)
from swr.roughness import psd_1d


def test_flat_kernel_gives_sigma2_times_h():
    """If H were flat over all K, alpha = H * sigma^2 for one wall."""
    sigma, lc, h0 = 0.003, 0.07, 5.0
    k = np.linspace(0, 400 / lc, 200001)
    alpha = alpha_from_kernel(k, np.full_like(k, h0), lambda q: psd_1d(q, sigma, lc))
    assert alpha == pytest.approx(h0 * sigma**2, rel=1e-4)
    assert alpha_from_kernel(k, np.full_like(k, h0), lambda q: psd_1d(q, sigma, lc), walls=2) == pytest.approx(
        2 * h0 * sigma**2, rel=1e-4)


def test_probe_roundtrip_single_line():
    """A sinusoid of amplitude A is the PSD limit of two lines: kernel -> alpha reproduces alpha_probe."""
    amp, k_line, h_true = 0.002, 10.0, 3.0
    alpha_probe = 0.5 * amp**2 * h_true
    assert kernel_from_probe(alpha_probe, amp) == pytest.approx(h_true)
    width = 1e-3
    k = np.linspace(k_line - 5 * width, k_line + 5 * width, 2001)
    line = lambda q: 0.25 * amp**2 * np.exp(-0.5 * ((q - k_line) / width) ** 2) / (np.sqrt(2 * np.pi) * width)
    assert alpha_from_kernel(k, np.full_like(k, h_true), line) == pytest.approx(alpha_probe, rel=1e-6)


def test_radiation_band_and_periods():
    lam, n_eff = 1.55, 1.9
    kmin, kmax = radiation_band(n_eff, 1.0, lam)
    k0 = 2 * np.pi / lam
    assert kmin == pytest.approx(0.9 * k0) and kmax == pytest.approx(2.9 * k0)
    per = periods_in_band(n_eff, 1.0, lam, 7)
    k = 2 * np.pi / per
    assert np.all((k > kmin) & (k < kmax))


def test_alpha_from_power_small_loss_limit():
    assert alpha_from_power(1e-6, 10.0) == pytest.approx(1e-7, rel=1e-5)
    assert to_db_per_cm(1e-4) == pytest.approx(4.3429, rel=1e-4)
