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
