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
