import numpy as np
import pytest

from engines.billiard_cell import engine as bc


def test_circle_mean_chord_and_conserved_angle():
    circ = bc.Stadium(5e-3, 0.0)
    assert circ.mean_chord == pytest.approx(np.pi * 5e-3 / 2)
    t = bc.trace_rays(circ, 100e-6, 400, 200, seed=3)
    longlived = np.nonzero((t.exit_idx < 0) | (t.exit_idx >= 6))[0][:50]
    assert np.abs(t.sinchi[longlived, :6] - t.sinchi[longlived, :1]).max() < 1e-5


def test_stadium_monte_carlo_vs_mean_field():
    st = bc.Stadium(5e-3, 5e-3)
    tab = bc.trace_rays(st, 150e-6, 3000, 2500, seed=2)
    mc = bc.evaluate(tab, lambda s: np.full_like(s, 0.999))
    mf = bc.mean_field_estimate(st, 150e-6, 0.999)
    assert mc.L_mean == pytest.approx(mf["L_mean"], rel=0.25)
    assert mc.T_det == pytest.approx(mf["T_det"], rel=0.25)


def test_mean_chord_of_rays_matches_cauchy():
    st = bc.Stadium(4e-3, 3e-3)
    S, _ = bc.poincare(st, 200, 400, seed=1)
    tab = bc.trace_rays(st, 1e-9, 200, 400, theta0=np.pi / 2 - 1e-3, seed=4)   # tiny ports: nobody leaves
    assert tab.trapped_fraction == 1.0
    assert tab.chord.mean() == pytest.approx(st.mean_chord, rel=0.05)
    assert np.all(np.isfinite(S))


def test_evaluate_reweighting():
    tab = bc.trace_rays(bc.Stadium(5e-3, 5e-3), 200e-6, 1000, 2000, seed=7)
    ideal = bc.evaluate(tab, lambda s: np.ones_like(s))
    assert ideal.T_det == pytest.approx(np.mean(tab.exit_port == 1))
    a = 3.0
    lossy = bc.evaluate(tab, lambda s: np.ones_like(s), alpha_bg=a, Gamma=0.4)
    assert np.allclose(lossy.W, np.exp(-a * lossy.L))
    assert lossy.L_mean < ideal.L_mean
    assert lossy.L_eff_gas == pytest.approx(0.4 * lossy.L_mean)
    assert lossy.S1 == pytest.approx(0.4 * lossy.T_det * lossy.L_mean)


def test_segmented_cell_geometry_and_phase_space():
    sc = bc.SegmentedCell(5e-3, 24)
    xo, yo = sc.outline(2000)
    ang = np.mod(np.arctan2(yo, xo) + np.pi / 24, 2 * np.pi / 24) - np.pi / 24
    assert np.max(np.abs(np.hypot(xo, yo) - 5e-3 * np.cos(np.pi / 24) / np.cos(ang))) < 1e-12
    assert sc.area == pytest.approx(0.5 * 24 * 25e-6 * np.sin(2 * np.pi / 24))
    _, SC = bc.poincare(bc.SegmentedCell(5e-3, 24, 0.01, 1 / 0.05, seed=1), 20, 200, seed=2)
    assert np.all(np.isfinite(SC)) and np.max(np.abs(SC)) <= 1 + 1e-9


def test_star_polygon_orbit_closes_in_regular_cell():
    # {24/7}: launch at chi = pi/2 - 7 pi/24 from a facet centre; after 24 hits the ray is back on facet 0
    sc = bc.SegmentedCell(5e-3, 24)
    x, y, nx, ny = sc.point_at(np.array([sc.h]))
    chi = np.pi / 2 - 7 * np.pi / 24
    dx, dy = -nx * np.cos(chi) + ny * np.sin(chi), -nx * np.sin(chi) - ny * np.cos(chi)
    xs, ys = bc.trace_path(sc, x[0], y[0], dx[0], dy[0], n_hits=24)
    assert np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < 1e-9


def test_occupancy_map_normalised():
    xe, ye, dens, cover = bc.occupancy_map(bc.Stadium(3e-3, 3e-3), 100e-6, lambda s: np.full_like(s, 0.99),
                                           n_rays=100, n_bounce=300, nb=40)
    assert np.nanmean(dens[np.isfinite(dens)]) == pytest.approx(1.0, rel=1e-9) and 0.5 < cover <= 1


def test_result_front_ends():
    mf = bc.mean_field(shape="polygon", radius=5e-3, n_facets=24, port_width=30e-6, R_mean=0.999)
    assert mf["n_bounce"] > 1 and mf.units["L_mean"] == "m"
    rs = bc.ray_statistics(n_rays=300, n_bounce=800)
    assert 0 < rs["T_det"] < 1 and rs["L_mean"] > 0
    assert bc.dB_per_cm_to_alpha(1.0) == pytest.approx(100 * np.log(10) / 10)
    with pytest.raises(ValueError, match="shape"):
        bc.mean_field(shape="triangle")
