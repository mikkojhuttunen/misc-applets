import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.special import jv  # noqa: E402

from engines.fringe_averaging import engine as fa  # noqa: E402


@pytest.mark.parametrize("x", [0.05, 1.0, 2.404825557695807, 7.3, 24.6, 120.0, 1000.0, -3.2])
def test_bessel_recurrence_matches_scipy(x):
    J = fa.bessel_j_all(x, int(abs(x)) + 60)
    m = np.arange(J.size)
    assert np.max(np.abs(J - jv(m, x))) < 1e-12


@pytest.mark.parametrize("wave", ["sine", "triangle"])
@pytest.mark.parametrize("x", [0.3, 5.0, 60.0])
def test_harmonics_are_complete_and_have_the_right_mean(wave, x):
    m, c = fa.dither_harmonics(x, wave)
    assert fa.truncated_power(m, c) < 2e-9                                   # Parseval
    c0 = c[m == 0][0]
    assert abs(c0) == pytest.approx(abs(jv(0, x)) if wave == "sine" else abs(np.sin(x) / x), abs=1e-12)
    t = np.linspace(0, 1, 7, endpoint=False) + 0.013                        # synthesis reproduces e^{i x w(t)}
    w = np.sin(2 * np.pi * t) if wave == "sine" else 1 - 4 * np.abs(np.mod(t + 0.25, 1) - 0.5)
    synth = (c[None, :] * np.exp(2j * np.pi * m[None, :] * t[:, None])).sum(1)
    assert np.allclose(synth, np.exp(1j * x * w), atol=2e-3 if wave == "triangle" else 1e-10)   # 1/m tail at the corners


def _brute_rms(x1, f1, w1, x2, f2, w2, psi, drift, T, n_t0=64, n_s=40000):
    rng = np.random.default_rng(0)
    vals = []
    for t0 in rng.random(n_t0) * 50.0:
        t = t0 + (np.arange(n_s) + 0.5) / n_s * T
        vals.append(np.mean(np.exp(1j * fa.time_trace(t, x1, f1, w1, x2, f2, w2, psi, drift))))
    return float(np.sqrt(np.mean(np.abs(np.array(vals)) ** 2)))


@pytest.mark.parametrize("case", [
    (2.0, 7, "sine", 0.0, 1, "sine", 0.0, 0.0, 0.37),
    (3.0, 5, "sine", 1.5, 10, "triangle", 0.7, 0.0, 0.5),       # commensurate: 10 = 2 × 5, phase matters
    (1.2, 3, "triangle", 2.5, 7, "sine", 0.0, 0.8, 1.3),         # with drift
])
def test_rms_residual_matches_brute_force_time_average(case):
    x1, f1, w1, x2, f2, w2, psi, drift, T = case
    f, A = fa.fringe_components(x1, f1, w1, x2, f2, w2, psi)
    V = fa.residual_visibility(f, A, T, drift)
    assert V == pytest.approx(_brute_rms(x1, f1, w1, x2, f2, w2, psi, drift, T), rel=0.08)


def test_long_averages_leave_only_the_static_floor():
    f, A = fa.fringe_components(3.0, 1000, "sine", 1.0, 1300, "sine")
    floor = fa.static_floor(f, A)
    assert floor == pytest.approx(abs(jv(0, 3.0) * jv(0, 1.0)), rel=1e-12)
    assert fa.residual_visibility(f, A, 1e3) == pytest.approx(floor, rel=1e-3)
    assert fa.residual_visibility(f, A, 1e-7) == pytest.approx(1.0, rel=1e-6)    # no averaging: full fringe
    assert fa.static_floor(f, A, drift_rad_s=0.1) == 0.0
    assert fa.residual_visibility(f, A, 1e4, drift_rad_s=0.1) < 0.01 * floor + 1e-12


def test_rc_and_boxcar_gains():
    assert fa.filter_gain(0.0, 1.0, "boxcar") == 1.0
    assert fa.filter_gain(2 * np.pi, 1.0, "boxcar") == pytest.approx(0.0, abs=1e-15)
    assert fa.filter_gain(3.0, 2.0, "rc") == pytest.approx(1 / np.sqrt(37))


def test_physical_depths():
    d = fa.depths(1.55e-6, 1.0, 1.0, 0.1, 2.45e5, 1e-5, 1e9, 1e6, 1.5e-4, 1e-3, 1e6)
    assert d["x_angle"] == pytest.approx(2.45)
    assert d["x_freq"] == pytest.approx(2 * np.pi * 0.1 / fa.C0 * 1e9)
    assert d["coherence"] == pytest.approx(np.exp(-np.pi * 1e6 * 0.1 / fa.C0))
    assert d["drift"] == pytest.approx(2 * np.pi / 1.55e-6 * 0.1 * 1.5e-4 * 1e-3 + 2 * np.pi * 0.1 / fa.C0 * 1e6)


def test_regular_cell_estimate_matches_ray_phase():
    from engines.billiard_cell import engine as bc
    from engines.ray_phase import engine as rp
    th = np.radians(37.5)
    opd, a = fa.regular_cell_estimate(5e-3, 24, th, 10, 1.55e-6)
    aa, _ = rp.phase_derivatives(bc.SegmentedCell(5e-3, 24), th, 10, 1.55e-6)
    L = rp.path_lengths(bc.SegmentedCell(5e-3, 24), [th], 10)[0]
    assert a == pytest.approx(abs(aa[-1]), rel=1e-3) and opd == pytest.approx(L[-1], rel=1e-9)


def test_result_front_end():
    r = fa.fringe_residual(freq_dev=5e8, avg_time=1.0, temp_rate=1e-3)
    assert 0 <= r["V_rms"] <= r["coherence"] + 1e-12 and r["fringe_amplitude"] <= r["fringe_undithered"] + 1e-15
    with pytest.raises(ValueError, match="filter"):
        fa.fringe_residual(filter_kind="gaussian")


def test_envelope_and_synchronous_averaging():
    z = np.linspace(50, 200, 200001)                                         # lobe-averaged tail: both ≈ 1/(2 z²)
    assert np.mean(fa.filter_gain(2 * z, 1.0) ** 2) == pytest.approx(np.mean(fa.filter_gain(2 * z, 1.0, envelope=True) ** 2), rel=0.01)
    f, A = fa.fringe_components(2.0, 1000, "sine", 1.0, 1300, "triangle")
    Tc = fa.common_period(1000, 1300)
    assert Tc == pytest.approx(0.01)
    assert fa.residual_visibility(f, A, 7 * Tc) == pytest.approx(fa.static_floor(f, A), rel=1e-9, abs=1e-12)
    assert fa.residual_visibility(f, A, 7.5 * Tc, envelope=True) > fa.static_floor(f, A)


def test_intermodulation_products_at_zero_hz_raise_the_floor():
    xa, xf = 60.66, 31.24
    assert fa.coincidence_orders(1000, 1300) == (13, 10)
    c00, full = fa.floor_split(xa, 1000, "sine", xf, 1300, "triangle")
    assert c00 == pytest.approx(abs(jv(0, xa) * np.sin(xf) / xf), rel=1e-9)
    assert full > 50 * c00                                                    # 13 f_θ = 10 f_ν: products of order (13, 10) at 0 Hz
    c00b, fullb = fa.floor_split(xa, 1000, "sine", xf, 1297, "triangle")      # gcd 1: coinciding orders ~1000, no harmonics there
    assert fullb == pytest.approx(c00b, rel=1e-6)
    assert fa.coincidence_orders(1000, 1297) == (1297, 1000)
