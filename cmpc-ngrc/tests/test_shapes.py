import numpy as np
import pytest

from ngrc import harmonics as H
from ngrc.index_map import GridIndex, rasterize_shape
from ngrc.shapes import Perturbation, Shape, random_shape


def _shape():
    return Shape(1e-4, -5e-5, 60e-6, 2e-3, 4e-6, a=[0, 0.05, -0.03, 0.02], b=[0, -0.04, 0.01, 0.0, 0.015])


def test_gradient_hessian_match_finite_differences():
    s = _shape()
    rng = np.random.default_rng(0)
    pts = np.c_[s.x0 + rng.uniform(-80e-6, 80e-6, 40), s.y0 + rng.uniform(-80e-6, 80e-6, 40)]
    h = 1e-9
    for x, y in pts:
        f, gx, gy, hxx, hxy, hyy = s.eval(x, y)
        fx = (s.eval(x + h, y, False)[0] - s.eval(x - h, y, False)[0]) / (2 * h)
        fy = (s.eval(x, y + h, False)[0] - s.eval(x, y - h, False)[0]) / (2 * h)
        gxx = (s.eval(x + h, y)[1] - s.eval(x - h, y)[1]) / (2 * h)
        gxy = (s.eval(x, y + h)[1] - s.eval(x, y - h)[1]) / (2 * h)
        gyy = (s.eval(x, y + h)[2] - s.eval(x, y - h)[2]) / (2 * h)
        scale1 = s.dn / s.edge
        scale2 = s.dn / s.edge**2
        assert abs(fx - gx) < 1e-5 * scale1 and abs(fy - gy) < 1e-5 * scale1
        assert abs(gxx - hxx) < 1e-4 * scale2 and abs(gxy - hxy) < 1e-4 * scale2 and abs(gyy - hyy) < 1e-4 * scale2


def test_bound_contains_the_shape():
    s = _shape()
    th = np.linspace(0, 2 * np.pi, 720)
    x = s.x0 + s.bound * np.cos(th)
    y = s.y0 + s.bound * np.sin(th)
    assert np.max(np.abs(s.eval(x, y, False)[0])) < 1e-6 * s.dn


def test_rotation_coefficients():
    s = Shape(0, 0, 50e-6, 1e-3, 2e-6, a=[0, 0.05, 0.02], b=[0, 0.01, -0.03], rotation=0.4)
    a, b = s.coefficients()
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    r_direct = s.radius(th)
    r_coef = s.R * (1 + sum(a[k] * np.cos((k + 1) * th) + b[k] * np.sin((k + 1) * th) for k in range(3)))
    assert np.allclose(r_direct, r_coef, rtol=1e-12)


def test_chd_recovers_analytic_coefficients_from_the_map():
    s = _shape()
    R0, a, b, _ = H.map_chd(s, m_max=6, n_theta=512)
    Rt, at, bt = H.shape_chd(s, 6)
    assert abs(R0 - Rt) < 0.01 * Rt
    assert np.max(np.abs(a[1:] - at[1:])) < 0.004
    assert np.max(np.abs(b[1:] - bt[1:])) < 0.004


def test_chd_from_a_binary_image():
    s = _shape()
    img = rasterize_shape(s, 1e-6, binary=True)
    g = GridIndex(img, 1e-6, s.x0, s.y0, smooth=1.5e-6)
    R0, a, b, _ = H.map_chd(g, m_max=6, n_theta=512)
    _, at, bt = H.shape_chd(s, 6)
    assert np.max(np.abs(a[1:] - at[1:])) < 0.006
    assert np.max(np.abs(b[1:] - bt[1:])) < 0.006


def test_grid_index_matches_analytic_shape():
    s = Shape(0, 0, 40e-6, 1e-3, 5e-6, a=[0, 0.05])
    g = GridIndex(rasterize_shape(s, 0.5e-6), 0.5e-6, 0, 0)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-60e-6, 60e-6, (2, 200))
    va = s.eval(x, y)
    vg = g.eval(x, y)
    assert np.max(np.abs(va[0] - vg[0])) < 1e-3 * s.dn
    assert np.max(np.abs(va[1] - vg[1])) < 2e-2 * s.dn / s.edge
    assert np.max(np.abs(va[3] - vg[3])) < 0.1 * s.dn / s.edge**2


def test_perturbation_sums_and_wall_check():
    rng = np.random.default_rng(3)
    p = Perturbation([random_shape(rng, 1e-3) for _ in range(3)])
    p.check_inside(1e-3)
    x, y = 1e-4, 2e-4
    assert np.isclose(p.value(x, y), sum(c.eval(x, y, False)[0] for c in p.components))
    with pytest.raises(ValueError):
        Perturbation([Shape(0.95e-3, 0, 50e-6, 1e-3)]).check_inside(1e-3)
