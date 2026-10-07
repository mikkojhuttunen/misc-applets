import sys
from pathlib import Path

import numpy as np
import pytest

from gmpc import planar as P
from gmpc import trace2d as T

ENGINES = Path(__file__).resolve().parents[2] / "math-engines"


def _billiard():
    if not (ENGINES / "engines").is_dir():
        pytest.skip("math-engines not found")
    sys.path.insert(0, str(ENGINES))
    from engines.billiard_cell import engine as bc
    return bc


def _launch_bc(cell, s, th):
    x, y, nx, ny = cell.point_at(np.array([s]))
    return x[0], y[0], -nx[0] * np.cos(th) + ny[0] * np.sin(th), -nx[0] * np.sin(th) - ny[0] * np.cos(th)


def test_geometry_of_builders():
    c = P.circle_cell(5e-3)
    assert c.area == pytest.approx(np.pi * 25e-6, rel=1e-12) and c.perimeter == pytest.approx(2 * np.pi * 5e-3, rel=1e-12)
    st = P.stadium_cell(5e-3, 4e-3)
    assert st.area == pytest.approx(2 * 5e-3 * 4e-3 + np.pi * 25e-6, rel=1e-12)
    assert st.perimeter == pytest.approx(2 * 4e-3 + 2 * np.pi * 5e-3, rel=1e-12)
    fs = P.stadium_cell(5e-3, 4e-3, cap_facets=12, straight_segments=3)
    assert fs.n_elements == 2 * 12 + 2 * 3 and fs.area < st.area
    poly = P.polygon_cell(5e-3, 24)
    assert poly.area == pytest.approx(0.5 * 24 * 25e-6 * np.sin(2 * np.pi / 24), rel=1e-12)
    x, y, nx, ny = st.point_at(np.linspace(0, st.perimeter, 400, endpoint=False))
    assert np.allclose(np.hypot(nx, ny), 1.0)
    assert np.all((x - np.clip(x, -2e-3, 2e-3)) ** 2 + y**2 <= 25e-6 * (1 + 1e-9))   # every wall point on the stadium


def test_circle_conserves_the_angle_of_incidence():
    _, _, _, SC, _ = T.trace_path(P.circle_cell(5e-3), 1e-3, 0.5, 80)
    assert np.ptp(SC) < 1e-12 and SC[0] == pytest.approx(np.sin(0.5), rel=1e-12)


def test_polygon_matches_billiard_cell_segmented_cell():
    bc = _billiard()
    tilts = np.zeros(24)
    tilts[5] = 5e-4
    mine = P.perturb(P.polygon_cell(5e-3, 24), tilts=tilts, offsets=np.linspace(-1e-6, 1e-6, 24))
    ref = bc.SegmentedCell(5e-3, 24, tilts=tilts, offsets=np.linspace(-1e-6, 1e-6, 24))
    xa, ya, *_ = T.trace_path(mine, mine.s_in_default, 0.4, 60)
    xb, yb = bc.trace_path(ref, *_launch_bc(ref, ref.h, 0.4), n_hits=60)
    assert np.max(np.hypot(xa - xb, ya - yb)) < 1e-12


def test_smooth_stadium_matches_billiard_cell_stadium():
    bc = _billiard()
    mine, ref = P.stadium_cell(5e-3, 5e-3), bc.Stadium(5e-3, 5e-3)
    s0 = 0.3 * mine.perimeter
    xa, ya, *_ = T.trace_path(mine, s0, 0.3, 12)                  # chaotic: short comparison
    xb, yb = bc.trace_path(ref, *_launch_bc(ref, s0, 0.3), n_hits=12)
    assert np.max(np.hypot(xa - xb, ya - yb)) < 1e-10


def test_hits_lie_on_the_reported_wall_coordinate():
    cell = P.perturb(P.stadium_cell(5e-3, 3e-3, cap_facets=10, straight_segments=2), tilt_rms=1e-3, curvature_rms=20, seed=3)
    x, y, *_ = T.launch(cell, cell.s_in_default, 0.3)
    dx, dy = T.launch(cell, cell.s_in_default, 0.3)[2:]
    for _ in range(80):
        t, nx, ny, s, k = cell.hit(x, y, dx, dy)
        x, y = x + t * dx, y + t * dy
        loc = s[0] - cell.s0[k[0]]
        if 1e-9 < loc < cell.L[k[0]] - 1e-9:                       # away from the clipped corner overlaps
            qx, qy, qnx, qny = cell.point_at(s)
            assert np.hypot(qx - x, qy - y)[0] < 1e-12 and np.hypot(qnx - nx, qny - ny)[0] < 1e-9
        dx, dy = T.reflect(dx, dy, nx, ny)


def test_monte_carlo_vs_mean_field_in_a_chaotic_stadium():
    st = P.stadium_cell(5e-3, 5e-3)
    tab = T.trace_rays(st, 150e-6, 3000, 2500, seed=2)
    mc = T.evaluate(tab, lambda s: np.full_like(s, 0.999))
    mf = T.mean_field_estimate(st, 150e-6, 0.999)
    assert mc.L_mean == pytest.approx(mf["L_mean"], rel=0.25) and tab.leaked_fraction == 0


def test_facets_and_perturbations_make_the_stadium_less_regular():
    smooth_circle = P.circle_cell(5e-3)
    sep, fit = T.twin_divergence(smooth_circle, 1e-3, 0.5, 200)
    assert fit is None or fit["lam"] < 0.05                           # regular: linear growth
    sep, fit = T.twin_divergence(P.stadium_cell(5e-3, 5e-3), 1e-3, 0.5, 200)
    assert fit is not None and fit["lam"] > 0.2                       # Bunimovich stadium: chaotic
    faceted = P.stadium_cell(5e-3, 5e-3, cap_facets=16)
    sep, fit = T.twin_divergence(faceted, 1e-3, 0.5, 300)
    assert fit is None or fit["lam"] < 0.1                            # polygonal walls: not exponentially chaotic
    curved = P.perturb(faceted, curvature_rms=40.0, seed=1, select="cap")
    sep, fit = T.twin_divergence(curved, 1e-3, 0.5, 200)
    assert fit is not None and fit["lam"] > 0.1                       # curved facets restore chaos


def test_perturb_validation_and_selection():
    st = P.stadium_cell(5e-3, 5e-3, cap_facets=8)
    pert = P.perturb(st, tilt_rms=1e-3, seed=2, select="cap")
    caps = [i for i, lab in enumerate(st.labels) if "cap" in lab]
    straight = [i for i, lab in enumerate(st.labels) if "straight" in lab]
    assert np.allclose(pert.A[straight], st.A[straight]) and not np.allclose(pert.A[caps], st.A[caps])
    with pytest.raises(ValueError, match="tilts"):
        P.perturb(st, tilts=np.zeros(3))
