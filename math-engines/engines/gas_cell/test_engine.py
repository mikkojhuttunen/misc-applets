import math
from statistics import NormalDist

import numpy as np
import pytest

from engines.gas_cell import engine as G
from engines.planar_cell import engine as P
from engines.trace_gas import engine as TG


def test_inverse_normal_and_line_table():
    for p in (1e-4, 0.01, 0.2, 0.5, 0.77, 0.999):
        assert G.inv_normal(p) == pytest.approx(NormalDist().inv_cdf(p), rel=2e-9, abs=1e-12)
    for sp, (a, lam) in G.SPECIES.items():
        a2, nu = TG.peak_alpha_per_ppm(sp)
        assert a == pytest.approx(a2, rel=1e-12) and lam == pytest.approx(1e-2 / nu, rel=1e-12)


def test_baseline_design_and_input_beam_are_the_resonator_mode():
    lam_m = 1.6537e-6 / 2.0
    q0, c0 = G.input_q(lam_m=lam_m)
    assert c0.meta["d"] == pytest.approx(15e-3) and c0.meta["aperture"] == pytest.approx(0.75e-3)
    w, Rw = G.q_width(q0, lam_m)
    assert w == pytest.approx(G.mirror_radius_w(15e-3, c0.meta["R"], lam_m), rel=0.02)     # eigenmode on the mirror
    cp = G.centre_path(c0, c0.s_in_default, c0.meta["theta_launch"], c0.s_out_default, c0.meta["port_w"])
    gb = G.gaussian_path(c0, cp, q0, lam_m, [(c0.s_in_default, c0.meta["port_w"])])
    assert cp["exits"] and cp["n_hits"] == 14
    assert gb["w_max_ratio"] < 1.02 and gb["m_max"] < 1                              # stable, no breathing


def test_herriott_keeps_the_fringe_the_cmpc_washes_it_out():
    r = G.compare(n_side=12).values
    assert r["h_exit_pass"] == 14 and r["h_V"] > 0.99 and r["h_same_path"] == 1.0
    assert r["c_V"] < 0.1 and r["c_T_G"] < 1e-6 and r["c_w_max_ratio"] > 5          # CMPC: washout, no Gaussian beam
    assert r["h_T"] > 2 * r["c_T"] and 0.5 < r["path_gain"] < 2
    assert r["washout_lost"] > 0.8 and r["h_nec_fringe"] > 10 * r["c_nec_fringe"]
    assert r["h_A_ppm"] == pytest.approx(100 * G.SPECIES["CH4"][0] * 0.2 * r["h_L_mean"])
    assert r["c_area"] == pytest.approx(r["h_area"], rel=1e-9)                      # same footprint


def test_deformation_destabilises_the_orbit_and_retuning_recovers_path():
    lam_m = 1.6537e-6 / 2.0
    q0, c0 = G.input_q(lam_m=lam_m)
    pw = c0.meta["port_w"]
    m = []
    for e in (0.0, 0.05, 0.15):
        b = G.baseline_cell(port_w=pw, eps4=e, lam_m=lam_m)
        cp = G.centre_path(b, b.s_in_default, b.meta["theta_launch"], b.s_out_default, pw)
        m.append(G.gaussian_path(b, cp, q0, lam_m, [(b.s_in_default, pw)])["m_max"])
    assert m[0] < 0.95 < m[1] < m[2] and m[2] > 1                                     # round trips turn unstable
    b = G.baseline_cell(port_w=pw, eps4=0.05, lam_m=lam_m)
    fixed = G.cell_figures(b, b.meta["theta_launch"], q0, lam_m, n_side=8)
    th, s1 = G.retune(b, b.meta["theta_launch"], q0, lam_m)
    assert fixed["exit_pass"] < 14 and s1 > 1.5 * fixed["S1"]                        # re-aiming restores the path


# ---------------------------------------------------------------- segmented mirrors and index inclusions
def test_mulberry32_matches_the_js_stream():
    u = P.mulberry32(1)
    assert [u() for _ in range(3)] == [0.6270739405881613, 0.002735721180215478, 0.5274470399599522]


def test_segments_that_follow_the_mirror_trace_like_the_smooth_cell():
    lam_m = 1.6537e-6 / 2.0
    q0, c0 = G.input_q(lam_m=lam_m)
    pw = c0.meta["port_w"]
    smooth, seg = G.baseline_cell(port_w=pw, lam_m=lam_m), G.baseline_cell(port_w=pw, lam_m=lam_m, n_seg=8)
    a = G.centre_path(smooth, smooth.s_in_default, smooth.meta["theta_launch"], smooth.s_out_default, pw)
    b = G.centre_path(seg, seg.s_in_default, seg.meta["theta_launch"], seg.s_out_default, pw)
    assert b["n_hits"] == a["n_hits"] == 14 and b["chords"].sum() == pytest.approx(a["chords"].sum(), rel=1e-12)
    flat = G.baseline_cell(port_w=pw, lam_m=lam_m, n_seg=8, seg_curv=0.0)
    assert np.all(flat.kappa == 0) and flat.n_elements == 16


def _single_inclusion(dn, b_off=0.0):
    c = P.circle_cell(2e-3)
    mc = G.MediumCell(c, dict(cx=[0.0], cy=[b_off], r=[0.4e-3], dn=[dn]), 2.0)
    x, y, dx, dy = np.array([-2e-3]), np.array([0.0]), np.array([1.0]), np.array([0.0])     # along the x axis
    return mc, G.march(mc, x, y, dx, dy, 1, record=True)


def test_inclusion_optical_path_and_snell_deviation():
    mc, r = _single_inclusion(0.01)                                  # through the centre: straight, longer optical path
    assert r["WX"][0, 0] == pytest.approx(2e-3, abs=1e-12) and r["WG"][0, 0] == pytest.approx(4e-3, rel=1e-12)
    assert r["WL"][0, 0] == pytest.approx(4e-3 + 2 * 0.4e-3 * 0.01 / 2.0, rel=1e-12)
    b = 0.25e-3                                                       # off-centre: ball-lens deviation 2(θ1 − θ2)
    mc, r = _single_inclusion(0.05, -b)
    pts = np.array(r["pts"][0])
    v = pts[-1] - pts[-2]
    th1 = math.asin(b / 0.4e-3)
    th2 = math.asin(2.0 * math.sin(th1) / 2.05)
    assert abs(math.atan2(v[1], v[0])) == pytest.approx(2 * (th1 - th2), rel=1e-9)
    mc, r = _single_inclusion(0.0, -b)                               # Δn = 0: no inclusion at all
    assert r["WL"][0, 0] == pytest.approx(r["WG"][0, 0]) and r["WY"][0, 0] == pytest.approx(0.0, abs=1e-15)


def test_total_internal_reflection_keeps_a_grazing_ray_outside_a_low_index_region():
    mc, r = _single_inclusion(-0.2, -0.399e-3)                        # grazing on a lower-index disk
    pts = np.array(r["pts"][0])
    assert len(pts) == 3 and pts[2][1] > 0                            # bounced off the disk (no crossing), then the wall


def test_placed_inclusions_do_not_overlap_and_stay_inside():
    lam_m = 1.6537e-6 / 2.0
    b = G.baseline_cell(lam_m=lam_m)
    inc = G.place_inclusions(b, 30, 50e-6, 0.01, "both", 4)
    cx, cy = np.array(inc["cx"]), np.array(inc["cy"])
    d = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None]) + np.eye(cx.size)
    assert cx.size == 30 and d.min() >= 2.5 * 50e-6 - 1e-15
    assert np.all(np.abs(cy) <= b.meta["aperture"] - 75e-6) and set(np.sign(inc["dn"])) == {-1.0, 1.0}


def test_chaos_makers_wash_out_the_fringe_and_closed_walls_keep_the_light():
    base = dict(n_side=8, retune_launch=0, chaos_grid=0, n_dither_max=8000)
    ideal = G.compare(**base).values
    seg = G.compare(n_seg=8, seg_curv=2.0, **base).values
    inc_open = G.compare(n_inc=20, inc_dn=0.02, **base).values
    inc_closed = G.compare(walls=1, n_inc=20, inc_dn=0.005, **base).values
    assert ideal["h_M2"] < 6 and ideal["h_V"] > 0.99
    assert seg["h_V"] < 0.7 and seg["h_T"] > 0.6                     # curved segments: some washout, light kept
    assert inc_open["h_V"] < 0.1 and inc_open["h_T"] < 0.5            # inclusions in an open cell: light lost sideways
    assert inc_closed["h_V"] < 0.05 and inc_closed["h_T"] > 0.7 and inc_closed["h_nec"] < inc_closed["c_nec"]
    assert inc_closed["h_M2"] > 5 * ideal["h_M2"]                     # the price: the beam is scrambled


def _rippled(L=10e-3, **kw):
    lam_m = 1.6537e-6 / 2.0
    q0, c0 = G.input_q(L=L, lam_m=lam_m)
    return G.baseline_cell(L=L, port_w=c0.meta["port_w"], lam_m=lam_m, **kw), q0, lam_m


def test_tangent_map_is_stable_for_the_ideal_cell_and_tracks_twin_rays_in_a_rippled_one():
    c, q0, lam_m = _rippled()
    th = c.meta["theta_launch"]
    t = G.tangent_chaos(c, *P.launch(c, np.array([c.s_in_default]), np.array([th])), n_hits=300)
    assert t["n_hits"][0] == 300 and t["Y"][0] < 0.1 and t["lam"][0] < 0.01      # elliptic: bounded, MEGNO -> 0
    c, q0, lam_m = _rippled(walls=True, ripple_a=0.6e-6)
    assert c.meta["K_ripple"] == pytest.approx(8 * math.pi**2 * 0.6e-6 * 10e-3 / 0.5e-3**2)
    th = c.meta["theta_launch"] + np.array([-2e-3, 0.0, 2e-3])
    t = G.tangent_chaos(c, *P.launch(c, np.full(3, c.s_in_default), th), n_hits=300)
    tw = [G.twin_divergence(G.as_medium(c), c.s_in_default, x, 300)[1]["lam"] for x in th]
    assert np.all(t["Y"] > 4) and np.median(t["lam"]) == pytest.approx(np.median(tw), rel=0.5)


def test_rippled_mirrors_launch_into_the_sea_or_onto_an_island():
    """K = 8π² a L / Λ² ≈ 0.95 for a = 0.3 µm, Λ = 0.5 mm, L = 10 mm: a mixed phase space. The same ripple shifted by a
    quarter period moves the beam from a regular island into the chaotic sea."""
    c, q0, lam_m = _rippled()
    ideal = G.chaos_map(c, c.meta["theta_launch"], q0, lam_m, 300, 8)
    assert ideal["verdict"] == 0 and ideal["sea"] == 0 and ideal["f_launch"] == 0
    c, q0, lam_m = _rippled(ripple_a=0.3e-6, ripple_phase=0.0)
    assert c.meta["K_ripple"] == pytest.approx(0.947, abs=1e-3)
    island = G.chaos_map(c, c.meta["theta_launch"], q0, lam_m, 300, 12)
    c, q0, lam_m = _rippled(ripple_a=0.3e-6, ripple_phase=1.57)
    sea = G.chaos_map(c, c.meta["theta_launch"], q0, lam_m, 300, 12)
    assert island["verdict"] == 1 and island["f_launch"] < 0.3 and island["sea"] > 0.02
    assert sea["verdict"] == 2 and sea["f_launch"] > 0.6 and sea["Y_launch"] > 4
    assert G.VERDICTS[sea["verdict"]] == "launches into the chaotic sea"
    assert sea["grid"]["Y"].shape == (12, 12)


def test_detector_speckle_averaging_divides_the_fringe_by_sqrt_m2():
    kw = dict(walls=1, n_inc=10, inc_dn=0.01, n_side=8, retune_launch=0, chaos_grid=0, n_dither_max=6000)
    off = G.compare(**kw).values
    on = G.compare(speckle_avg=1, **kw).values
    assert off["h_n_grains"] == 1 and off["h_V_det"] == off["h_V"]
    assert on["h_n_grains"] == pytest.approx(on["h_M2"]) and on["h_V_det"] == pytest.approx(on["h_V"] / math.sqrt(on["h_M2"]))
    assert on["h_nec_fringe"] == pytest.approx(off["h_nec_fringe"] / math.sqrt(on["h_M2"]))
    assert on["h_nec_fringe_hi"] >= on["h_nec_fringe"] and on["h_V_floor"] > 0
    ideal = G.compare(n_side=8, retune_launch=0, chaos_grid=0).values
    assert ideal["h_V_floor"] < 1e-6 and ideal["h_frag"] == 0                   # a regular cell is resolved exactly
