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


# ---------------------------------------------------------------- loss, beam propagation, stack, erbium
from engines.onchip_herriott import erbium as ER  # noqa: E402
from engines.onchip_herriott import stack as ST  # noqa: E402
from engines.slab_waveguide import engine as SW  # noqa: E402


def test_waveguide_loss_weights_path_and_throughput():
    c = _cell()
    b0 = O.budget(c, clip=False)
    a = O.DB_PER_CM * 1.0                                               # 1 dB/cm
    b = O.budget(c, clip=False, alpha=a)
    assert b["T_out"] == pytest.approx(b0["T_out"] * math.exp(-a * b0["L_geom"]), rel=1e-12)
    eta = b0["eta_window"]
    expect, I = 0.0, eta
    for l in b0["chords"]:
        expect += I * (1 - math.exp(-a * l)) / a
        I *= math.exp(-a * l)
    assert b["L_eff"] == pytest.approx(expect, rel=1e-12)
    s = O.sweep_size(10e-3, 20, 3, 1e-3, 0.0, 100e-6, [1, 4], [0.0, 1.0])
    assert s["L_eff"][0, 1] / s["L_eff"][0, 0] == pytest.approx(4, rel=0.02)    # lossless: scales with size
    assert s["L_eff"][1, 1] / s["L_eff"][1, 0] < 2                              # 1 dB/cm: saturates


def test_matched_beam_is_the_round_trip_eigenmode_and_mismatch_breathes():
    c = _cell()
    lam_m = 1.55e-6 / 2.479
    h = O.trace_hits(c)
    bp = O.beam_on_path(c, h, lam_m, {})
    st = bp["stability"]
    assert st["coupling"] == pytest.approx(1.0, abs=1e-12) and st["stable"]
    assert st["gouy_roundtrip"] == pytest.approx(2 * c.meta["theta"], rel=0.03)  # paraxial 2θ per round trip
    assert bp["W"][1, 0] == pytest.approx(bp["W"][0, -1], rel=1e-12)            # continuous through the reflection
    w2 = bp["W"][1, -1]
    assert w2 == pytest.approx(bp["W"][0, 0], rel=1e-9)                         # back to the launch width after 2 passes
    mis = O.beam_on_path(c, h, lam_m, dict(w_ratio=1.5))
    wm = mis["W"][:-1, -1]
    assert wm.max() / wm.min() > 1.5 and mis["stability"]["coupling"] < 0.95
    assert O.coupling(complex(0.3, 2.0), complex(0.3, 2.0), 1e-6) == pytest.approx(1.0)


def test_unstable_spacing_is_flagged():
    c = P.herriott_planar_cell(10e-3, 20, 3, 1e-3, 100e-6, d=20.5e-3)        # d > 2R
    M = O.path_matrix(c, np.array([20.5e-3, 20.5e-3]), np.array([1, 0]), np.zeros(2), 0, 2)      # on-axis round trip
    assert abs(0.5 * (M[0] + M[3])) > 1 and O.eigen_q(M) is None


def test_stack_mode_matches_slab_waveguide_and_overlaps_sum_to_one():
    for args in (("al2o3", 0.4e-6, 0.1e-6, "sio2", "air", "TE"), ("tfln", 0.3e-6, 0.4e-6, "sio2", "air", "TM"),
                 ("si3n4", 0.2e-6, 0.3e-6, "sio2", "sio2", "TE"), ("si", 0.3e-6, 0.0, "air", "air", "TM")):
        m = ST.stack_mode(*args[:5], wavelength=1.55e-6, polarization=args[5])
        ref = SW.multilayer_neff(1.55e-6, m["indices"], m["thicknesses"], args[5]).values["neff"][0]
        assert m["neff"] == pytest.approx(ref, rel=1e-9) and sum(m["gamma"]) == pytest.approx(1.0, rel=1e-12)
    thin = ST.stack_mode("al2o3", 0.4e-6, 0.05e-6, "sio2", "air", 1.55e-6, "TE")
    thick = ST.stack_mode("al2o3", 0.4e-6, 0.4e-6, "sio2", "air", 1.55e-6, "TE")
    assert 0 < thin["gamma_er"] < thick["gamma_er"] < 1
    assert ST.n_material("sio2", 1.55e-6) == pytest.approx(1.444, abs=1e-3)


def test_erbium_spectroscopy_and_rate_equation_limits():
    assert ER.sigma_a(1.533e-6) == pytest.approx(5.7e-25, rel=0.01)
    assert ER.sigma_e(1.532e-6) == pytest.approx(ER.sigma_a(1.532e-6), rel=1e-9)    # McCumber crossing at λ0
    xs = ER.cross_sections(1.532e-6, 980e-9)
    N = 1e26
    assert ER.populations(0.0, 0.0, xs, 1.532e-6, 980e-9, N, 0.0, 7.5e-3, 0.0) == (N, 0.0)
    n1, n2 = ER.populations(1e12, 0.0, xs, 1.532e-6, 980e-9, N, 0.0, 7.5e-3, 0.0)
    assert n2 / N > 0.99                                                    # strong 980 pump inverts fully
    n1u, n2u = ER.populations(1e9, 0.0, xs, 1.532e-6, 980e-9, N, 0.0, 7.5e-3, 1e-23)
    n1c, n2c = ER.populations(1e9, 0.0, xs, 1.532e-6, 980e-9, N, 0.0, 7.5e-3, 0.0)
    assert n2u < n2c                                                        # upconversion depletes N2


def test_amplifier_passive_limit_and_pumped_gain():
    c = _cell()
    ms = ST.stack_mode("al2o3", 0.4e-6, 0.4e-6, "sio2", "air", 1.532e-6, "TE")
    mp = ST.stack_mode("al2o3", 0.4e-6, 0.4e-6, "sio2", "air", 980e-9, "TE")
    one = lambda chi, t=0.0: 1.0
    sig = dict(wavelength=1.532e-6, n_eff=ms["neff"], gamma_er=ms["gamma_er"], alpha=0.0, R_of_chi=one, beam={})
    pmp = dict(wavelength=980e-9, n_eff=mp["neff"], gamma_er=mp["gamma_er"], alpha=0.0, R_of_chi=one, beam={})
    er = dict(N=1e26, t=0.4e-6, tau=7.5e-3, Cup=0.0)
    dark = O.amplifier(c, sig, pmp, er, 1e-9, 0.0, clip=False)
    a = ms["gamma_er"] * ER.sigma_a(1.532e-6) * 1e26
    assert dark["internal_gain_db"] == pytest.approx(-10 * math.log10(math.e) * a * dark["path"], rel=1e-6)
    lit = O.amplifier(c, sig, pmp, er, 1e-9, 0.2, clip=False)
    assert lit["internal_gain_db"] > 0 and 0 < lit["pump_left"] < 1 and lit["inversion"].max() > 0.5
    r = O.er_amplifier()
    assert r["R_dbr_pump"] < 0.2 < r["R_dbr_signal"] and O.er_amplifier(pump_mirror="dbr")["gain_db"] < r["gain_db"]


def test_new_front_ends():
    s = O.slab_stack()
    assert s["gamma_substrate"] + s["gamma_film"] + s["gamma_er"] + s["gamma_cladding"] == pytest.approx(1.0)
    b = O.beam_stability(w_ratio=1.3)
    assert b["stable"] and b["breathing"] > 1.3 and b["coupling"] < 1
    p = O.path_budget(loss_db_cm=1.0)
    assert p["T_out"] < O.path_budget()["T_out"] and p["m_roundtrip"] == pytest.approx(b["m_roundtrip"])
