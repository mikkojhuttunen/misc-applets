import numpy as np
import pytest

from gmpc import herriott as H


def test_sphere_newton_intersection_matches_closed_form():
    m = H.Mirror3D(np.zeros(3), np.eye(3), 0.5, 0.5, aperture=0.05)
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.uniform(-0.01, 0.01, 50), rng.uniform(-0.01, 0.01, 50), np.full(50, 0.3)])
    d = np.column_stack([rng.uniform(-0.05, 0.05, 50), rng.uniform(-0.05, 0.05, 50), -np.ones(50)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, P, N, _ = m.intersect(p, d)
    C = np.array([0, 0, 0.5])
    assert np.allclose(np.linalg.norm(P - C, axis=1), 0.5, atol=1e-14)
    assert np.allclose(N, (C - P) / 0.5, atol=1e-12)


def test_ideal_cell_reenters_after_N_hits_with_third_order_aberration():
    errs = []
    for A in (0.012, 0.006, 0.003):
        c = H.herriott_cell(0.5, 30, 7, A)
        tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 200)
        r = H.reentrance(tr)
        assert r["exit"] == "hole" and r["n_hits"] == 30
        assert r["path"] == pytest.approx(30 * c["d"], rel=2e-3)
        errs.append(r["exit_offset"])
    assert errs[0] / errs[1] == pytest.approx(8, rel=0.05) and errs[1] / errs[2] == pytest.approx(8, rel=0.05)


def test_spots_follow_paraxial_theory_for_small_patterns():
    c = H.herriott_cell(1.0, 40, 9, 0.002)
    tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 100)
    n = np.arange(1, 40)
    x, y = H.paraxial_spots(n, c["d"], 0.002, 0.002, 1.0, 1.0)
    assert np.max(np.hypot(tr.hits[0, :39, 0] - x, tr.hits[0, :39, 1] - y)) < 2e-6     # (A/R)² scale
    assert np.allclose(tr.mirror[0, :39], np.where(n % 2, 1, 0))


def test_astigmatic_mirrors_give_a_lissajous_pattern():
    Rx, Ry, d = H.astigmatic_reentrant(1.0, 50, 11, 13)
    assert np.arccos(1 - d / Rx) * 50 == pytest.approx(2 * np.pi * 11) and np.arccos(1 - d / Ry) * 50 == pytest.approx(2 * np.pi * 13)
    c = H.astigmatic_cell(Rx, Ry, d, 0.003, 0.002, 4e-4)
    tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 120)
    n = np.arange(1, 49)
    x, y = H.paraxial_spots(n, d, 0.003, 0.002, Rx, Ry)
    assert np.max(np.hypot(tr.hits[0, :48, 0] - x, tr.hits[0, :48, 1] - y)) < 1e-5
    assert H.reentrance(tr)["exit"] == "hole" and H.reentrance(tr)["n_hits"] == 50


def test_tilt_keeps_reentrance_but_radius_errors_spoil_it():
    # tilting a spherical mirror only moves its centre of curvature: the cell axis tilts, θ is unchanged, and every
    # ray still re-enters after N hits; changing R (or d) changes θ and breaks re-entrance
    R, A = 0.5, 0.008
    c = H.herriott_cell(R, 30, 7, A)
    tr0 = H.trace3d(c["mirrors"], c["p0"], c["d0"], 200)
    alpha = 2e-4
    tilted = [c["mirrors"][0], H.perturb_mirror(c["mirrors"][1], tilt_x=alpha)]
    tr = H.trace3d(tilted, c["p0"], c["d0"], 200)
    r0, r = H.reentrance(tr0), H.reentrance(tr)
    assert r["exit"] == "hole" and r["n_hits"] == 30 and r["exit_offset"] == pytest.approx(r0["exit_offset"], rel=0.05)
    # pattern centre on mirror 2 follows the new axis through both centres of curvature: y = R α (R - d)/(2R - d)
    sel = tr0.mirror[0, :29] == 1
    shift = np.mean(tr.hits[0, :29][sel, 1]) - np.mean(tr0.hits[0, :29][sel, 1])
    d = c["d"]
    assert abs(shift) == pytest.approx(R * alpha * (R - d) / (2 * R - d), rel=0.05)
    bent = [H.perturb_mirror(m, dRx=2e-3, dRy=2e-3) for m in c["mirrors"]]   # +0.4 % radius
    rb = H.reentrance(H.trace3d(bent, c["p0"], c["d0"], 200))
    assert rb["exit"] != "hole" or rb["n_hits"] != 30 or rb["exit_offset"] > 20 * r0["exit_offset"]


def test_trace_is_reversible():
    c = H.herriott_cell(0.5, 30, 7, 0.01)
    m = [H.perturb_mirror(c["mirrors"][0], poly={(3, 0): 2e-3, (1, 2): -1e-3}), c["mirrors"][1]]
    m[0].holes = []                                                      # closed cell for this test
    p0, _ = H.start_on_mirror(m[0], (0.01, 0.0), [0, 0, 1])
    tr = H.trace3d(m, p0, c["d0"], 25)
    k = 24
    p_end, d_end = tr.hits[0, k], None
    # direction of the last chord, reversed
    d_end = -(tr.hits[0, k] - tr.hits[0, k - 1]) / np.linalg.norm(tr.hits[0, k] - tr.hits[0, k - 1])
    back = H.trace3d(m, p_end, d_end, 25)                  # retraces hits 23, 22, ..., 0, then the start point
    assert np.linalg.norm(back.hits[0, 23] - tr.hits[0, 0]) < 1e-9
    assert np.linalg.norm(back.hits[0, 24] - p0) < 1e-9


def test_effective_path_and_spot_metrics():
    c = H.herriott_cell(0.5, 30, 7, 0.015)
    tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], 200)
    L, Leff, I = H.herriott_effective_path(tr, 0.99)
    assert L == pytest.approx(tr.path()) and I == pytest.approx(0.99**29) and Leff < L
    sm = H.spot_metrics(tr, c["mirrors"], w=c["w_mode"])
    assert sm[0]["reflections"] + sm[1]["reflections"] == 29
    assert sm[0]["hole_clearance"] > 0 and sm[0]["min_spacing_in_w"] > 4


def test_stable_herriott_does_not_diverge_exponentially():
    c = H.herriott_cell(0.5, 30, 7, 0.01)
    m = [c["mirrors"][0], c["mirrors"][1]]
    m[0] = H.perturb_mirror(m[0])
    m[0].holes = []
    sep, fit = H.twin_divergence_3d(m, c["p0"], c["d0"], 300)
    assert np.max(sep) < 1e-5 and (fit is None or fit["lam"] < 0.02)
