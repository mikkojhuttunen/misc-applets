import numpy as np
import pytest

from engines.billiard_cell import engine as bc
from engines.bragg_grating.engine import TrenchDBR
from engines.cell_mirror import engine as cmir
from engines.planar_cell import engine as P
from engines.ray_phase import engine as rp


def _launch_bc(cell, s, th):
    x, y, nx, ny = cell.point_at(np.array([s]))
    return x[0], y[0], -nx[0] * np.cos(th) + ny[0] * np.sin(th), -nx[0] * np.sin(th) - ny[0] * np.cos(th)


# ---------------------------------------------------------------- geometry and tracing
def test_geometry_of_builders():
    c = P.circle_cell(5e-3)
    assert c.area == pytest.approx(np.pi * 25e-6, rel=1e-12) and c.perimeter == pytest.approx(2 * np.pi * 5e-3, rel=1e-12)
    st = P.stadium_cell(5e-3, 4e-3)
    assert st.area == pytest.approx(2 * 5e-3 * 4e-3 + np.pi * 25e-6, rel=1e-12)
    assert st.perimeter == pytest.approx(2 * 4e-3 + 2 * np.pi * 5e-3, rel=1e-12)
    assert st.closed and st.s_in_default == pytest.approx(2e-3)
    fs = P.stadium_cell(5e-3, 4e-3, cap_facets=12, straight_segments=3)
    assert fs.n_elements == 2 * 12 + 2 * 3 and fs.area < st.area
    poly = P.polygon_cell(5e-3, 24)
    assert poly.area == pytest.approx(0.5 * 24 * 25e-6 * np.sin(2 * np.pi / 24), rel=1e-12)
    x, y, nx, ny = st.point_at(np.linspace(0, st.perimeter, 400, endpoint=False))
    assert np.allclose(np.hypot(nx, ny), 1.0)
    assert np.all((x - np.clip(x, -2e-3, 2e-3)) ** 2 + y**2 <= 25e-6 * (1 + 1e-9))   # every wall point on the stadium


def test_inside_matches_the_stadium_and_the_open_herriott_cell():
    st = P.stadium_cell(5e-3, 4e-3)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-8e-3, 8e-3, 4000), rng.uniform(-6e-3, 6e-3, 4000)
    ref = (x - np.clip(x, -2e-3, 2e-3)) ** 2 + y**2 < 25e-6
    near = np.abs(np.sqrt((x - np.clip(x, -2e-3, 2e-3)) ** 2 + y**2) - 5e-3) < 1e-4   # sampled arcs: skip the rim
    assert np.all(st.inside(x, y)[~near] == ref[~near])
    hc = P.herriott_planar_cell(10e-3, 20, 3, 1e-3)
    assert not hc.closed and hc.inside(np.array([0.0]), np.array([0.0]))[0] and not hc.inside(np.array([0.0]), np.array([2e-3]))[0]


def test_circle_conserves_the_angle_of_incidence():
    _, _, _, SC, _ = P.trace_path(P.circle_cell(5e-3), 1e-3, 0.5, 80)
    assert np.ptp(SC) < 1e-12 and SC[0] == pytest.approx(np.sin(0.5), rel=1e-12)


def test_polygon_matches_billiard_cell_segmented_cell():
    tilts = np.zeros(24)
    tilts[5] = 5e-4
    mine = P.perturb(P.polygon_cell(5e-3, 24), tilts=tilts, offsets=np.linspace(-1e-6, 1e-6, 24))
    ref = bc.SegmentedCell(5e-3, 24, tilts=tilts, offsets=np.linspace(-1e-6, 1e-6, 24))
    xa, ya, *_ = P.trace_path(mine, mine.s_in_default, 0.4, 60)
    xb, yb = bc.trace_path(ref, *_launch_bc(ref, ref.h, 0.4), n_hits=60)
    assert np.max(np.hypot(xa - xb, ya - yb)) < 1e-12


def test_smooth_stadium_matches_billiard_cell_stadium():
    mine, ref = P.stadium_cell(5e-3, 5e-3), bc.Stadium(5e-3, 5e-3)
    s0 = 0.3 * mine.perimeter
    xa, ya, *_ = P.trace_path(mine, s0, 0.3, 12)                  # chaotic: short comparison
    xb, yb = bc.trace_path(ref, *_launch_bc(ref, s0, 0.3), n_hits=12)
    assert np.max(np.hypot(xa - xb, ya - yb)) < 1e-10


def test_hits_lie_on_the_reported_wall_coordinate():
    cell = P.perturb(P.stadium_cell(5e-3, 3e-3, cap_facets=10, straight_segments=2), tilt_rms=1e-3, curvature_rms=20, seed=3)
    x, y, dx, dy = P.launch(cell, cell.s_in_default, 0.3)
    for _ in range(80):
        t, nx, ny, s, k = cell.hit_k(x, y, dx, dy)
        x, y = x + t * dx, y + t * dy
        loc = s[0] - cell.s0[k[0]]
        if 1e-9 < loc < cell.L[k[0]] - 1e-9:                       # away from the clipped corner overlaps
            qx, qy, qnx, qny = cell.point_at(s)
            assert np.hypot(qx - x, qy - y)[0] < 1e-12 and np.hypot(qnx - nx, qny - ny)[0] < 1e-9
            assert cell.segment_of(s)[0] == k[0]
        dx, dy = P.reflect(dx, dy, nx, ny)


def test_monte_carlo_vs_mean_field_in_a_chaotic_stadium():
    st = P.stadium_cell(5e-3, 5e-3)
    tab = P.trace_rays(st, 150e-6, 3000, 2500, seed=2)
    mc = P.evaluate(tab, lambda s: np.full_like(s, 0.999))
    mf = P.mean_field_estimate(st, 150e-6, 0.999)
    assert mc.L_mean == pytest.approx(mf["L_mean"], rel=0.25) and tab.leaked_fraction == 0


def test_facets_and_perturbations_make_the_stadium_less_regular():
    sep, fit = P.twin_divergence(P.circle_cell(5e-3), 1e-3, 0.5, 200)
    assert fit is None or fit["lam"] < 0.05                           # regular: linear growth
    sep, fit = P.twin_divergence(P.stadium_cell(5e-3, 5e-3), 1e-3, 0.5, 200)
    assert fit is not None and fit["lam"] > 0.2                       # Bunimovich stadium: chaotic
    faceted = P.stadium_cell(5e-3, 5e-3, cap_facets=16)
    sep, fit = P.twin_divergence(faceted, 1e-3, 0.5, 300)
    assert fit is None or fit["lam"] < 0.1                            # polygonal walls: not exponentially chaotic
    curved = P.perturb(faceted, curvature_rms=40.0, seed=1, select="cap")
    sep, fit = P.twin_divergence(curved, 1e-3, 0.5, 200)
    assert fit is not None and fit["lam"] > 0.1                       # curved facets restore chaos


def test_perturb_validation_and_selection():
    st = P.stadium_cell(5e-3, 5e-3, cap_facets=8)
    pert = P.perturb(st, tilt_rms=1e-3, seed=2, select="cap")
    caps = [i for i, lab in enumerate(st.labels) if "cap" in lab]
    straight = [i for i, lab in enumerate(st.labels) if "straight" in lab]
    assert np.allclose(pert.A[straight], st.A[straight]) and not np.allclose(pert.A[caps], st.A[caps])
    assert pert.s_in_default == st.s_in_default
    with pytest.raises(ValueError, match="tilts"):
        P.perturb(st, tilts=np.zeros(3))


# ---------------------------------------------------------------- integrated (planar) Herriott cell
def test_planar_herriott_reenters_through_its_window_with_third_order_error():
    offs = []
    for A in (2e-3, 1e-3, 0.5e-3):
        c = P.herriott_planar_cell(20e-3, 24, 5, A, port_w=A / 10)
        tr = P.herriott_planar_trace(c)
        assert tr["exit"] == "port" and tr["n_hits"] == 24 and tr["reflections"] == 23
        assert tr["path"] == pytest.approx(24 * c.meta["d"], rel=1e-3)
        assert np.all(tr["element"][:23] == np.arange(1, 24) % 2 * 1)      # M2, M1, M2, ... (hit 1 on M2)
        offs.append(abs(tr["exit_angle"]))
    assert offs[0] / offs[1] == pytest.approx(8, rel=0.05) and offs[1] / offs[2] == pytest.approx(8, rel=0.05)   # ∝ A³


def test_planar_herriott_spots_follow_paraxial_theory():
    R, N, M, A = 50e-3, 30, 7, 0.4e-3
    for phase in (0.0, np.pi / N):
        c = P.herriott_planar_cell(R, N, M, A, port_w=A / 40, phase=phase)
        tr = P.herriott_planar_trace(c)
        n = np.arange(1, N)
        assert np.max(np.abs(tr["ys"][1:N] - A * np.cos(n * c.meta["theta"] + phase))) < 2e-3 * A   # (A/R)² scale
        assert tr["exit"] == "port" and tr["n_hits"] == N
    y1, y2, sp0, _ = P.herriott_planar_spots(P.herriott_planar_cell(R, N, M, A).meta)
    assert sp0 < 1e-12 * A                                            # phase 0: the pattern runs out and back
    _, _, sp, clear = P.herriott_planar_spots(P.herriott_planar_cell(R, N, M, A, port_w=A / 40, phase=np.pi / N).meta)
    assert sp > 0.01 * A and clear > 0
    _, _, _, clear = P.herriott_planar_spots(P.herriott_planar_cell(R, N, M, A, port_w=A / 10, phase=np.pi / N).meta)
    assert clear < 0                                                  # a slot this wide swallows the hit-4 spot


def test_planar_herriott_radius_error_spoils_reentrance_but_tilt_keeps_the_period():
    c = P.herriott_planar_cell(20e-3, 24, 5, 1e-3)
    r0 = P.herriott_planar_trace(c)
    bent = P.herriott_planar_cell(20e-3, 24, 5, 1e-3, R2=20.4e-3, d=c.meta["d"])
    rb = P.herriott_planar_trace(bent, n_max=200)
    assert rb["exit"] != "port" or rb["n_hits"] != 24 or abs(rb["exit_offset"]) > 20 * abs(r0["exit_offset"])
    tilted = P.perturb(c, tilts=[0.0, 2e-4])
    rt = P.herriott_planar_trace(tilted)
    assert rt["exit"] == "port" and rt["n_hits"] == 24


def test_planar_herriott_rays_past_the_mirror_ends_leak():
    c = P.herriott_planar_cell(20e-3, 24, 5, 1e-3)
    tab = P.trace_rays(c, 0.0, 50, 100, theta0=0.4, seed=3)            # wide fan from a point
    assert tab.leaked_fraction > 0.3 and np.all(tab.exit_port[tab.exit_port >= 0] == 1)
    assert P.mode_radius(20e-3, c.meta["d"], 1.55e-6, 2.0) == pytest.approx(P.mode_radius(20e-3, c.meta["d"], 0.775e-6))


# ---------------------------------------------------------------- ray_phase and cell_mirror on planar cells
def test_ray_phase_on_a_planar_polygon_matches_the_segmented_cell():
    tilts = np.zeros(24)
    tilts[5] = 5e-4
    seg = bc.SegmentedCell(5e-3, 24, tilts=tilts)
    pla = P.perturb(P.polygon_cell(5e-3, 24), tilts=tilts)
    th = np.radians(37.5) + np.linspace(-1e-4, 1e-4, 41)
    La, ia = rp.path_lengths(seg, th, 30, return_ids=True)
    Lb, ib = rp.path_lengths(pla, th, 30, return_ids=True)
    assert np.nanmax(np.abs(La - Lb)) < 1e-12
    assert np.all((ia[:, 1:] == ia[:, :1]) == (ib[:, 1:] == ib[:, :1]))      # same path switching
    a = rp.dither_analysis(seg, np.radians(37.5), 1e-4, 30, 1.55e-6, n_min=101, n_max=801)
    b = rp.dither_analysis(pla, np.radians(37.5), 1e-4, 30, 1.55e-6, n_min=101, n_max=801)
    assert np.allclose(a["V"], b["V"], atol=1e-9) and np.allclose(a["same_path"], b["same_path"])


def test_cell_mirror_on_a_planar_polygon_matches_the_segmented_cell():
    d = TrenchDBR(n_tooth=2.479, N=4, slab_pol="TM")
    a = cmir.cell_mirror_stats(bc.SegmentedCell(5e-3, 24, curvature=0.0), np.radians(22.5), 0.02, 5, 80, d)
    b = cmir.cell_mirror_stats(P.polygon_cell(5e-3, 24), np.radians(22.5), 0.02, 5, 80, d)
    for k in ("R_mean", "R_eff", "L_eff", "chi_95"):
        assert a[k] == pytest.approx(b[k], rel=1e-10)


def test_herriott_phase_slope_stays_bounded_while_the_polygon_slope_grows():
    hc = P.herriott_planar_cell(20e-3, 24, 5, 1e-3)
    a_h, _ = rp.phase_derivatives(hc, hc.meta["theta_launch"], 23, 1.55e-6)
    a_p, _ = rp.phase_derivatives(P.polygon_cell(5e-3, 24), np.radians(37.5), 23, 1.55e-6)
    assert np.max(np.abs(a_h)) < 3 * np.max(np.abs(a_h[:4]))           # focusing: bounded, oscillating
    assert abs(a_p[-1]) > 8 * abs(a_p[1])                             # polygon: grows ∝ p


def test_dbr_sees_small_angles_in_a_herriott_cell():
    hc = P.herriott_planar_cell(20e-3, 24, 5, 1e-3)
    d = TrenchDBR(n_tooth=2.479, N=4, slab_pol="TE")
    h = cmir.cell_mirror_stats(hc, 0.0, 0.0, 1, 23, d)
    s = cmir.cell_mirror_stats(P.stadium_cell(5e-3, 5e-3, cap_facets=12), 0.5, 0.02, 5, 23, d)
    assert h["chi_max"] < 10 and h["R_eff"] > s["R_eff"]               # no Brewster dip, no TIR edge


def test_front_ends():
    r = P.ray_statistics(n_rays=300, n_bounce=800)
    assert 0 < r["T_det"] < 1 and r["leaked_fraction"] == 0
    assert P.chaos(shape="stadium")["chaotic"] and not P.chaos(shape="faceted_stadium")["chaotic"]
    h = P.planar_herriott()
    assert h["exits_through_port"] and h["reflections"] == 19 and h["path"] == pytest.approx(h["path_design"], rel=1e-3)
    hb = P.planar_herriott(dR2=0.3e-3)                                # the edge window sits at a turning point of the
    assert abs(hb["exit_angle"] - h["exit_angle"]) > 0.02            # pattern: a period error tilts the exit ray first
    with pytest.raises(ValueError, match="even"):
        P.planar_herriott(N=21)
