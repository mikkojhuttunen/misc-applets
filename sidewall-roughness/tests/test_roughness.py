import numpy as np
import pytest
from scipy.integrate import quad

from swr.roughness import (RoughnessSpec, acf_1d, fit_correlation_length, generate,
                           measure_acf, psd_1d, sinusoid)

SIGMA = 0.003
LC = 0.07
NX = 4096
DX = LC / 10
REALIZATIONS = 200


@pytest.mark.parametrize("acf", ["gaussian", "exponential"])
def test_psd_integrates_to_sigma2_and_is_fourier_pair(acf):
    total, _ = quad(lambda k: psd_1d(k, SIGMA, LC, acf), -np.inf, np.inf)
    assert total == pytest.approx(SIGMA**2, rel=1e-6)
    for u in (0.5 * LC, LC, 2 * LC):
        half, _ = quad(lambda k: psd_1d(k, SIGMA, LC, acf), 0, np.inf, weight="cos", wvar=u)
        r = 2 * half
        assert r == pytest.approx(acf_1d(u, SIGMA, LC, acf), rel=1e-4, abs=1e-14)


@pytest.mark.parametrize("acf", ["gaussian", "exponential"])
def test_ensemble_sigma_and_acf(acf):
    spec = RoughnessSpec(sigma=SIGMA, lc=LC, acf=acf)
    fields = np.vstack([generate(spec, NX, DX, seed=s) for s in range(REALIZATIONS)])
    assert np.sqrt(np.mean(fields**2)) == pytest.approx(SIGMA, rel=0.02)
    acf_est = measure_acf(fields)
    lags = np.arange(40) * DX
    expected = acf_1d(lags, SIGMA, LC, acf)
    assert np.max(np.abs(acf_est[:40] - expected)) < 0.03 * SIGMA**2
    assert fit_correlation_length(acf_est, DX) == pytest.approx(LC, rel=0.03)


@pytest.mark.parametrize("acf", ["gaussian", "exponential"])
def test_periodogram_matches_psd(acf):
    spec = RoughnessSpec(sigma=SIGMA, lc=LC, acf=acf)
    fields = np.vstack([generate(spec, NX, DX, seed=1000 + s) for s in range(REALIZATIONS)])
    periodogram = np.mean(np.abs(np.fft.fft(fields, axis=1)) ** 2, axis=0) * DX / (2 * np.pi * NX)
    k = 2 * np.pi * np.fft.fftfreq(NX, d=DX)
    band = (np.abs(k) > 0) & (np.abs(k) < 2.0 / LC)
    ratio = periodogram[band] / psd_1d(k[band], SIGMA, LC, acf)
    assert np.mean(ratio) == pytest.approx(1.0, rel=0.03)


def test_columnar_field_is_uniform_in_z():
    spec = RoughnessSpec(sigma=SIGMA, lc=LC, lz=None)
    h = generate(spec, 512, DX, nz=7, dz=0.05, seed=3)
    assert h.shape == (7, 512)
    assert np.all(h == h[0])


def test_z_correlated_field_has_both_correlation_lengths():
    lz = 0.04
    dz = lz / 8
    spec = RoughnessSpec(sigma=SIGMA, lc=LC, lz=lz)
    fields = [generate(spec, 1024, DX, nz=256, dz=dz, seed=s) for s in range(20)]
    acf_x = np.mean([measure_acf(f, axis=1) for f in fields], axis=0)
    acf_z = np.mean([measure_acf(f, axis=0) for f in fields], axis=0)
    assert np.sqrt(np.mean(np.square(fields))) == pytest.approx(SIGMA, rel=0.03)
    assert fit_correlation_length(acf_x, DX) == pytest.approx(LC, rel=0.05)
    assert fit_correlation_length(acf_z, dz) == pytest.approx(lz, rel=0.05)


def test_seed_reproducible_and_distinct():
    spec = RoughnessSpec(sigma=SIGMA, lc=LC)
    assert np.array_equal(generate(spec, 256, DX, seed=5), generate(spec, 256, DX, seed=5))
    assert not np.array_equal(generate(spec, 256, DX, seed=5), generate(spec, 256, DX, seed=6))


def test_sinusoid_sigma():
    period = 0.4
    x = np.arange(4000) * period / 400
    h = sinusoid(x, 0.005, period)
    assert np.mean(h**2) == pytest.approx(0.005**2 / 2, rel=1e-9)


def test_bad_spec_rejected():
    with pytest.raises(ValueError):
        RoughnessSpec(sigma=SIGMA, lc=LC, acf="lorentzian")
    with pytest.raises(ValueError):
        RoughnessSpec(sigma=SIGMA, lc=-1.0)
