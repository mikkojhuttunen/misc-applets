import numpy as np
import pytest

from engines.billiard_cell import engine as bc
from engines.ray_phase import engine as rp


def test_first_chord_matches_polygon_geometry():
    # from the middle of facet 0 of a regular N-gon the first chord ends on the line of another facet;
    # with the circle limit (large N) L1 -> 2 r cos θ
    cell = bc.SegmentedCell(5e-3, 360)
    th = np.linspace(-1.2, 1.2, 9)
    L1 = rp.path_lengths(cell, th, 1)[:, 0]
    assert np.allclose(L1, 2 * cell.apothem * np.cos(th), rtol=2e-4)


def test_path_lengths_agree_with_trace_path():
    cell = bc.SegmentedCell(5e-3, 24, curvature=20.0)
    x, y, dx, dy = rp.launch(cell, [0.5])
    xs, ys = bc.trace_path(cell, x[0], y[0], dx[0], dy[0], n_hits=30)
    L = rp.path_lengths(cell, [0.5], 30)[0]
    assert np.allclose(L, np.cumsum(np.hypot(np.diff(xs), np.diff(ys))), rtol=1e-12)


@pytest.mark.parametrize("waveform", ["sine", "triangle"])
def test_dither_of_linear_phase(waveform):
    from scipy.special import j0
    A = np.linspace(0.1, 6, 12)
    for a in A:
        d = rp.dither_offsets(1.0, 2001, waveform)
        V = rp.visibility(a * d)
        expect = abs(j0(a)) if waveform == "sine" else abs(np.sin(a) / a)
        assert V == pytest.approx(expect, abs=2e-5)


def test_first_mirror_phase_derivative():
    # exact for a flat facet: L = (a - n·P) / (n·d), dL/dθ = -L (n·d') / (n·d) with d' = (-d_y, d_x)
    cell = bc.SegmentedCell(5e-3, 24)
    th = 0.5
    x, y, dx, dy = rp.launch(cell, [th])
    _, _, _, s = cell.hit(x, y, dx, dy)
    k = int(s[0] // (2 * cell.h))
    n, p = cell.n_k[k], cell.p_k[k]
    nd = n[0] * dx[0] + n[1] * dy[0]
    L = ((p[0] - x[0]) * n[0] + (p[1] - y[0]) * n[1]) / nd
    dL = -L * (n[0] * -dy[0] + n[1] * dx[0]) / nd
    r = rp.first_mirror_phase(5e-3, 24, th, 1.55e-6, 1.0)
    assert r["L1"] == pytest.approx(L, rel=1e-12)
    assert r["dL_dtheta"] == pytest.approx(dL, rel=1e-6)
    assert r["dtheta_2pi"] == pytest.approx(1.55e-6 / abs(dL), rel=1e-6)


def test_visibility_drops_with_amplitude_and_passes():
    cell = bc.SegmentedCell(5e-3, 24)
    small = rp.dither_scan(cell, np.radians(37.5), 1e-7, 24, 1.55e-6)
    large = rp.dither_scan(cell, np.radians(37.5), 2e-4, 24, 1.55e-6)
    assert small["V"][0] > 0.99 and large["V"][-1] < small["V"][-1]
    assert np.all(np.abs(small["dphi"][200]) < 1e-6)        # middle sample of the sine has δ ≈ 0


def test_result_front_ends():
    r = rp.dither_visibility(amplitude=1e-6, n_pass=10)
    assert 0 <= r["V_last"] <= 1 and r["resolved"]
    with pytest.raises(ValueError, match="waveform"):
        rp.dither_visibility(waveform="square")


def test_model_visibility_matches_bessel_and_sinc():
    from scipy.special import j0
    x = np.array([0.0, 0.3, 1.0, 2.404825557695807, 7.0, 55.0, 400.0])
    assert np.allclose(rp.model_visibility(x, "sine"), np.abs(j0(x)), atol=1e-12)
    assert np.allclose(rp.model_visibility(x[1:], "triangle"), np.abs(np.sin(x[1:]) / x[1:]), atol=1e-15)
    for wf in ("sine", "triangle"):
        assert rp.model_visibility(rp.X_V05[wf], wf)[0] == pytest.approx(0.5, abs=1e-9)
        assert rp.model_visibility(rp.X_V09[wf], wf)[0] == pytest.approx(0.9, abs=1e-9)
        assert rp.model_visibility(rp.X_V0[wf], wf)[0] == pytest.approx(0.0, abs=1e-9)


def test_analysis_follows_the_linear_model_while_paths_are_shared():
    cell = bc.SegmentedCell(5e-3, 24)
    for wf in ("sine", "triangle"):
        r = rp.dither_analysis(cell, np.radians(37.5), 1e-5, 30, 1.55e-6, waveform=wf)
        ok = (r["same_path"] == 1) & r["resolved"]
        assert ok.sum() >= 25
        assert np.allclose(r["V"][ok], r["V_model"][ok], atol=1e-3)
        assert np.allclose(r["A_05"] * np.abs(r["a"]), rp.X_V05[wf])


def test_phase_slope_grows_linearly_with_pass_in_a_regular_cell():
    a, b = rp.phase_derivatives(bc.SegmentedCell(5e-3, 24), np.radians(37.5), 40, 1.55e-6)
    p = np.arange(1, 41)
    ratio = np.abs(a) / (p * abs(a[0]))
    assert np.all(np.isfinite(a)) and np.allclose(ratio, 1.0, rtol=0.05)
    L = rp.path_lengths(bc.SegmentedCell(5e-3, 24), np.radians(37.5) + np.array([-1e-6, 1e-6]), 40)
    assert np.allclose(a, 2 * np.pi / 1.55e-6 * (L[1] - L[0]) / 2e-6, rtol=1e-3)


def test_scaling_with_wavelength_index_and_amplitude():
    cell = bc.SegmentedCell(5e-3, 24)
    base = rp.dither_analysis(cell, 0.5, 2e-6, 10, 1.55e-6, 1.0)
    other = rp.dither_analysis(cell, 0.5, 1e-6, 10, 1.55e-6 / 2, 2.0 / 2)   # x ∝ n A / λ: same x
    assert np.allclose(base["x"], other["x"], rtol=1e-4) and np.allclose(base["V"], other["V"], atol=1e-4)


def test_adaptive_sampling_and_noise_floor():
    cell = bc.SegmentedCell(5e-3, 24)
    r = rp.dither_analysis(cell, np.radians(37.5), 1e-3, 20, 1.55e-6, n_max=20001)
    assert r["n_used"] > 401 and r["resolved"].all()
    assert r["noise_floor"] == pytest.approx(np.sqrt(np.pi) / (2 * np.sqrt(r["n_used"])))
    capped = rp.dither_analysis(cell, np.radians(37.5), 1e-3, 20, 1.55e-6, n_max=401)
    assert capped["n_used"] == 401 and not capped["resolved"].all()


def test_path_switching_is_detected():
    tilts = np.zeros(24)
    tilts[5] = 5e-4
    cell = bc.SegmentedCell(5e-3, 24, tilts=tilts)
    r = rp.dither_analysis(cell, np.radians(37.5), 1e-4, 60, 1.55e-6)
    assert r["same_path"][0] == 1.0 and r["same_path"][-1] < 0.5
