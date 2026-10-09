import math

import numpy as np
import pytest

from engines.bragg_grating.engine import TrenchDBR
from engines.onchip_herriott import engine as O
from engines.planar_cell import engine as P


def _cell(**kw):
    return P.herriott_planar_cell(10e-3, 20, 3, 1e-3, 100e-6, **kw)


def test_mode_matches_the_resonator_mode_radius_and_gaussian_optics():
    c = _cell()
    m = O.mode_profile(10e-3, c.meta["d"], 1.55e-6, 2.479)
    assert m["w_mirror"] == pytest.approx(P.mode_radius(10e-3, c.meta["d"], 1.55e-6, 2.479), rel=1e-12)
    assert m["zR"] == pytest.approx(math.pi * m["w0"] ** 2 / m["lam_m"], rel=1e-12)
    assert O.beam_width(c.meta["d"] / 2, m) == pytest.approx(m["w_mirror"], rel=1e-12)


def test_width_integral_along_the_axis_is_exact():
    m = O.mode_profile(10e-3, 4e-3, 1.55e-6, 2.0)
    F = lambda x: 0.5 * m["w0"] * (x * math.sqrt(1 + (x / m["zR"]) ** 2) + m["zR"] * math.asinh(x / m["zR"]))
    assert O.chord_width_integral(-2e-3, 0.0, 2e-3, 0.0, m) == pytest.approx(F(2e-3) - F(-2e-3), rel=1e-6)


def test_lossless_budget_and_constant_mirror_geometric_series():
    c = _cell()
    b1 = O.budget(c, R_of_chi=None, clip=False)
    eta = b1["eta_window"]
    assert b1["exits"] and b1["reflections"] == 19 and b1["T_out"] == pytest.approx(eta**2)
    assert b1["L_eff"] == pytest.approx(eta * b1["L_geom"]) and b1["V_eff"] == pytest.approx(eta * b1["V_geom"])
    R = 0.98
    b = O.budget(c, R_of_chi=lambda chi: R, clip=False)
    assert b["I_end"] == pytest.approx(eta * R**19)
    assert b["L_eff"] == pytest.approx(eta * sum(R**j * l for j, l in enumerate(b["chords"])), rel=1e-12)
    assert b["A_eff"] == pytest.approx(b["V_eff"] / b["L_eff"]) and b["w_eff_mean"] > b["mode"]["w0"]


def test_closed_cell_reaches_the_cavity_limit():
    c = _cell()
    R = 0.99
    b = O.budget(c, R_of_chi=lambda chi: R, exit_mode="closed", clip=False)
    ell = b["L_geom"] / b["n_hits"]
    assert not b["exits"] and b["L_eff"] == pytest.approx(ell / (1 - R), rel=0.01)


def test_angles_are_set_by_the_geometry_not_the_reflectance():
    s = O.sweep_R(_cell(), [0.9, 0.99, 1.0])
    assert np.ptp(s["chi_mean"]) < 0.02 * s["chi_mean"].mean() and np.all(np.diff(s["L_eff"]) > 0)
    assert np.allclose(s["A_eff"], s["A_eff"][0], rtol=1e-6)        # same widths on every pass


def test_dbr_beam_average_and_polarisation():
    dbr = TrenchDBR(n_tooth=2.479, N=4, m_tooth=1, bounce_loss=0.0, slab_pol="TM")
    point = O.mirror_function("dbr", dbr=dbr, theta0=0.0)
    tiny = O.mirror_function("dbr", dbr=dbr, theta0=1e-7, beam_average=True)
    wide = O.mirror_function("dbr", dbr=dbr, theta0=0.05, beam_average=True)
    assert tiny(0.1) == pytest.approx(point(0.1), rel=1e-9)
    a, wt = O.angle_weights(0.05)
    assert wide(0.0) == pytest.approx(sum(w * point(abs(x)) for x, w in zip(a, wt)), rel=1e-12) and wide(0.0) != point(0.0)
    te = O.path_budget(polarization="TE")
    tm = O.path_budget(polarization="TM")
    assert te["R_mean"] < tm["R_mean"] and te["L_eff"] < tm["L_eff"]


def test_clipping_at_the_window_and_mirror_ends():
    wide = O.budget(_cell(), clip=True)
    assert wide["clip_loss"] < 1e-6 and wide["eta_window"] > 0.99
    narrow = O.budget(P.herriott_planar_cell(10e-3, 20, 3, 1e-3, 20e-6, phase=np.pi / 20), clip=True)
    assert narrow["eta_window"] < 0.6 and narrow["clip_loss"] > 0.01
    assert O.gauss_fraction(-1, 1, 0.0, 1.0) == pytest.approx(math.erf(math.sqrt(2)))


def test_footprint_and_front_end():
    f20 = O.footprint_fraction(_cell())
    f40 = O.footprint_fraction(P.herriott_planar_cell(10e-3, 40, 7, 1e-3, 100e-6))
    assert 0 < f20 < 1 and f40 > f20
    r = O.path_budget(mirror="constant", R_mirror=0.995)
    assert r["exits"] and r["reflections"] == 19 and r["chi_mean_deg"] == pytest.approx(7.5, abs=0.3)
