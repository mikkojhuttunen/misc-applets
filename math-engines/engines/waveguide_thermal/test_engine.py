import math

import numpy as np
import pytest

from engines.thermo_optic import engine as to
from engines.waveguide_thermal import engine as wt


def test_one_dimensional_stack_exact():
    # heat generated uniformly in the top layer of a laterally uniform stack, adiabatic top:
    # mean temperature of the heated layer = q''(Σ t_i/k_i + t_top/(3 k_top))
    xe = np.linspace(0, 1e-6, 5)
    ye = wt.graded_axis([0, 3e-6, 5e-6, 5.5e-6], 5e-6, 5.5e-6, 0.02e-6)
    yc = 0.5 * (ye[1:] + ye[:-1])
    k = np.where(yc < 3e-6, 148.0, np.where(yc < 5e-6, 1.38, 4.6))[:, None] * np.ones((1, 4))
    q = np.where(yc > 5e-6, 1.0 / 0.5e-6, 0.0)[:, None] * np.ones((1, 4))
    T, q_sink = wt.solve_heat(xe, ye, k, q, h_top=0.0)
    sel = yc > 5e-6
    dy = np.diff(ye)
    Tm = np.sum(T[sel, 0] * dy[sel]) / np.sum(dy[sel])
    assert Tm == pytest.approx(3e-6 / 148 + 2e-6 / 1.38 + 0.5e-6 / (3 * 4.6), rel=1e-4)
    assert q_sink == pytest.approx(1e-6, rel=1e-9)


def test_buried_square_against_image_solution():
    # uniformly heated square (side a) at height d above an isothermal plane in a homogeneous medium:
    # R' ≈ [acosh(d / r_e) + 1/4] / (2π k), r_e = 0.5902 a (conformal radius of a square, Carslaw & Jaeger image
    # solution) and 1/4 the mean-over-source excess of a uniformly heated disc
    a = 0.4e-6
    r = wt.ridge_heating(core_material="sio2", slab_material="sio2", box_material="sio2", substrate_material="sio2",
                         clad_material="sio2", core_width=a, core_height=a, substrate_thickness=1e-6, box_thickness=9e-6,
                         clad_thickness=300e-6, domain_half_width=300e-6)
    d = 10e-6 + a / 2
    k = to.entry("sio2")["k"]
    assert r["R_th"] == pytest.approx((math.acosh(d / (0.5902 * a)) + 0.25) / (2 * math.pi * k), rel=0.03)


def test_heat_balance_and_grid_convergence():
    r1 = wt.ridge_heating()
    r2 = wt.ridge_heating(resolution=2.0)
    assert r1["heat_balance"] == pytest.approx(1.0, abs=1e-9)
    assert r1["dT_core"] == pytest.approx(r2["dT_core"], rel=0.02)
    assert r2["n_cells"] > 2 * r1["n_cells"]


def test_linear_in_heat():
    a = wt.ridge_heating(heat_per_length=1.0)["dT_core"]
    b = wt.ridge_heating(heat_per_length=3.0)["dT_core"]
    assert b == pytest.approx(3 * a, rel=1e-9)


def test_soi_wire_resistance_range():
    # 500 x 220 nm Si wire, 2 µm BOX, oxide clad: ~0.4 K per mW/mm, nearly all of it across the BOX
    r = wt.ridge_heating()
    assert 0.3 < r["R_th"] < 0.5
    assert r["dT_substrate_top"] < 0.1 * r["dT_core"]
    assert 1e-4 < r["tau_E"] < 1e-2


def test_ordering_box_thickness_and_air_clad():
    base = wt.ridge_heating()["R_th"]
    assert wt.ridge_heating(box_thickness=3e-6)["R_th"] > base
    assert wt.ridge_heating(clad_material="air")["R_th"] > base
    assert wt.ridge_heating(substrate_material="sapphire", box_thickness=0.2e-6, box_material="sio2")["R_th"] < base


@pytest.mark.parametrize("geom", [
    dict(core_width=0.5e-6, core_height=0.22e-6),
    dict(core_width=0.5e-6, core_height=0.22e-6, clad_material="air"),
    dict(core_width=1.6e-6, core_height=0.8e-6, core_material="si3n4", box_thickness=4e-6, clad_thickness=3e-6),
    dict(core_width=1.2e-6, core_height=0.3e-6, core_material="ln_e", slab_material="ln_e", slab_thickness=0.3e-6,
         box_thickness=4.7e-6, clad_material="air", clad_thickness=0.0),
    dict(core_width=3e-6, core_height=0.5e-6, core_material="si3n4", box_thickness=8e-6, clad_thickness=8e-6),
    dict(core_width=2e-6, core_height=0.4e-6, core_material="al2o3_film", slab_material="ln_e", slab_thickness=0.3e-6,
         box_thickness=4.7e-6, clad_material="air", clad_thickness=0.0),
])
def test_closed_form_estimate_tracks_solver(geom):
    g = dict(core_width=0.5e-6, core_height=0.22e-6, box_thickness=2e-6, slab_thickness=0.0, clad_material="sio2", slab_material="si")
    g.update(geom)
    fd = wt.ridge_heating(**geom)["R_th"]
    est = to.strip_thermal_resistance(width=g["core_width"], height=g["core_height"], box_thickness=g["box_thickness"],
                                      slab_thickness=g["slab_thickness"], slab_material=g["slab_material"],
                                      clad_material=g["clad_material"])["R_th"]
    assert 0.7 < est / fd < 1.3


def test_thermal_shift_combines():
    r = wt.thermal_shift(heat_per_length=2.0, dneff_dT=1.8e-4, n_group=4.2, length=1e-3)
    assert r["dneff"] == pytest.approx(1.8e-4 * r["dT_core"])
    assert r["delta_lambda"] == pytest.approx(1.55e-6 * r["dneff"] / 4.2)
    assert r["phase"] == pytest.approx(2 * math.pi * r["dneff"] * 1e-3 / 1.55e-6)


def test_return_field_shapes():
    r = wt.ridge_heating(return_field=True)
    assert r["dT"].shape == (len(r["y_edges"]) - 1, len(r["x_edges"]) - 1)


def test_dynamics_step_and_frequency_response_agree():
    # mean response time: ∫(1 - s(t)) dt = w·A⁻¹ C A⁻¹ Q / w·A⁻¹ Q (exact for the discretised model)
    from scipy.sparse.linalg import spsolve
    r = wt.ridge_dynamics(steps_per_doubling=16)
    g = wt.build_ridge()
    A, _ = wt.conduction_matrix(g["xe"], g["ye"], g["k"], 10.0)
    A = A.tocsc()
    dA, core = g["dA"], g["core"]
    C = (g["rc"] * dA).ravel()
    Q = (wt._heat_map(g, 1.0) * dA).ravel()
    w = (np.where(core, dA, 0.0) / np.sum(dA[core])).ravel()
    T0 = spsolve(A, Q)
    t_mean = float(w @ spsolve(A, C * T0)) / float(w @ T0)
    t, s = np.concatenate([[0.0], r["t"]]), np.concatenate([[0.0], r["step"]])
    assert np.all(np.diff(s) >= -1e-12) and s[-1] > 0.99
    assert np.sum(0.5 * ((1 - s[1:]) + (1 - s[:-1])) * np.diff(t)) == pytest.approx(t_mean, rel=0.03)
    assert r["t_10"] < r["t_50"] < r["t_90"]
    assert r["H_1kHz"] > r["H_1MHz"]


def test_soi_thermal_bandwidth_order():
    # SOI wire on 2 µm BOX: fast BOX response (µs), tens of kHz -3 dB
    r = wt.ridge_dynamics()
    assert 1e-6 < r["t_90"] < 1e-4
    assert 1e4 < r["f_3dB"] < 3e5


def test_wide_strip_mode_approaches_slab_te():
    from engines.slab_waveguide.engine import neff_three_layer
    m = wt.ridge_mode(core_material="si3n4", core_width=12e-6, core_height=0.4e-6, clad_material="sio2",
                      cells_per_wavelength=24)
    n_c, n_s = to.index_at("si3n4", 1.55e-6), to.index_at("sio2", 1.55e-6)
    slab = neff_three_layer(1.55e-6, n_s, n_c, n_s, 0.4e-6, "TE", 0)
    assert m["neff"] == pytest.approx(slab, abs=2e-3)       # lateral confinement lowers it by ~1e-3


def test_mode_gamma_sum_rule():
    # scaling all indices by s ≡ scaling k by s, so Σ Γ_r n_r = n_eff - λ ∂n_eff/∂λ at fixed indices
    geo = dict(core_material="si3n4", core_width=1.2e-6, core_height=0.6e-6, clad_material="sio2", cells_per_wavelength=14)
    idx = {"core": 2.0, "box": 1.444, "clad": 1.444, "slab": 1.444}
    m = wt.ridge_mode(1.55e-6, indices=idx, **geo)
    E2, n = m["E"] ** 2 * m["V"], m["n"]
    s = float(np.sum(n * n * E2) / (m["neff"] * np.sum(E2)))
    dl = 1e-8
    mp = wt.ridge_mode(1.55e-6 + dl, indices=idx, h=m["h"], **geo)["neff"]
    mm = wt.ridge_mode(1.55e-6 - dl, indices=idx, h=m["h"], **geo)["neff"]
    assert s == pytest.approx(m["neff"] - 1.55e-6 * (mp - mm) / (2 * dl), rel=2e-3)


def test_mode_weighted_soi():
    r = wt.mode_weighted_heating(cells_per_wavelength=16)
    assert r["dT_mode"] == pytest.approx(r["dT_core"], rel=0.03)
    assert 1.7e-4 < r["dneff_dT"] < 2.0e-4
    assert 60e-12 < r["dlambda_dT"] < 90e-12


def test_mode_weighted_tfln_rib_sees_cooler_film():
    g = dict(core_material="ln_e", slab_material="ln_e", core_width=1.2e-6, core_height=0.3e-6, slab_thickness=0.3e-6,
             box_thickness=4.7e-6, clad_material="air", clad_thickness=0.0)
    r = wt.mode_weighted_heating(**g)
    assert r["dT_mode"] < r["dT_core"]
    assert r["Gamma_slab"] > 0.3
    hot = wt.mode_weighted_heating(heat_in_mode=True, **g)
    assert hot["dT_mode"] > 0


def test_self_consistent_matches_reduced_runaway_model():
    Rm = wt.mode_weighted_heating(heat_in_mode=True, cells_per_wavelength=16)["R_th_mode"]
    red = to.thermal_runaway(R_th=Rm, power=0.1, loss_abs_db_per_cm=1.0, T_scale=20.0)
    P = 0.5 * red["P_threshold"]
    sc = wt.self_consistent_heating(power=P, loss_abs_db_per_cm=1.0, T_scale=20.0, cells_per_wavelength=16)
    assert sc["converged"] and not sc["runaway"]
    assert sc["dT_absorption"] == pytest.approx(to.thermal_runaway(R_th=Rm, power=P, loss_abs_db_per_cm=1.0, T_scale=20.0)["dT"], rel=0.01)
    hot = wt.self_consistent_heating(power=1.3 * red["P_threshold"], loss_abs_db_per_cm=1.0, T_scale=20.0, cells_per_wavelength=16)
    assert hot["runaway"]


def test_thermal_lens_first_order_and_confinement():
    sc = wt.self_consistent_heating(power=3.0, loss_abs_db_per_cm=1.0, cells_per_wavelength=16)
    assert sc["converged"]
    assert sc["dneff"] == pytest.approx(sc["dneff_first_order"], rel=1e-3)
    assert sc["a_eff"] < sc["a_eff0"]                       # positive dn/dT in the hot core focuses the mode


def test_uniform_conductivity_exponent_is_kirchhoff():
    # same exponent everywhere and an isothermal sink: θ = ∫k/k0 dT solves the linear problem exactly
    kw = dict(power=8.0, loss_abs_db_per_cm=1.0, cells_per_wavelength=12, h_top=0.0)
    lin = wt.self_consistent_heating(**kw)
    m = 0.6
    nl = wt.self_consistent_heating(k_exponent_core=m, k_exponent_box=m, k_exponent_substrate=m, k_exponent_clad=m,
                                    k_exponent_slab=m, **kw)
    assert nl["dT_core"] == pytest.approx(float(to.kirchhoff_inverse(lin["dT_core"], m)), rel=0.01)
    assert nl["dT_core"] > lin["dT_core"]


def test_axial_transfer_against_bessel_image_solution():
    # buried square (side a) at height d over an isothermal plane, homogeneous medium, heat ∝ cos(κz):
    # R(κ) = [K0(κ r_e) - K0(2 d κ) + 1/4] / (2π k), r_e = 0.5902 a
    from scipy.special import k0
    a = 0.4e-6
    geo = dict(core_material="sio2", slab_material="sio2", box_material="sio2", substrate_material="sio2", clad_material="sio2",
               core_width=a, core_height=a, substrate_thickness=1e-6, box_thickness=9e-6, clad_thickness=300e-6,
               domain_half_width=300e-6)
    d, re, k = 10e-6 + a / 2, 0.5902 * a, to.entry("sio2")["k"]
    for kd in (0.3, 1.0, 3.0):
        kap = kd / d
        assert wt.axial_transfer(kap, **geo) == pytest.approx((k0(kap * re) - k0(2 * d * kap) + 0.25) / (2 * math.pi * k), rel=0.03)


def test_temperature_along_z_limits():
    z = np.linspace(0, 0.01, 1001)
    r = wt.temperature_along_z(z, np.ones_like(z))
    assert r["dT"][500] == pytest.approx(r["R0"], rel=1e-3)          # uniform heating: the 2D result
    assert 0.3e-6 < r["axial_length"] < 10e-6                        # SOI: heat spreads along z over ~µm
    q = np.exp(-((z - 0.005) / 2e-6) ** 2)                           # a 2 µm hot spot is smoothed
    s = wt.temperature_along_z(z, q, ends="infinite")
    assert s["dT_max"] < 0.8 * s["dT_local_max"]
    zz = np.linspace(0, 0.01, 2001)                                  # energy: ∫ΔT/R0 over a long uniform segment ≈ ∫q
    w = wt.temperature_along_z(zz, (zz > 0.004) & (zz < 0.006), ends="infinite")
    assert np.sum(w["dT"]) == pytest.approx(np.sum(w["dT_local"]), rel=0.02)


def test_heater_pi_power_soi_and_sin():
    # published SOI heaters (oxide-clad, no undercut): P_π ≈ 20-30 mW; thick SiN: ~100-300 mW
    soi = wt.heater_tuning(cells_per_wavelength=14)
    assert 0.012 < soi["P_pi"] < 0.035
    sin = wt.heater_tuning(core_material="si3n4", core_width=1.6e-6, core_height=0.8e-6, box_thickness=4e-6,
                           clad_thickness=4e-6, cells_per_wavelength=14)
    assert 0.06 < sin["P_pi"] < 0.3
    far = wt.heater_tuning(heater_gap=2e-6, cells_per_wavelength=14)
    assert far["P_pi"] > soi["P_pi"]


def test_heater_crosstalk_falls_with_pitch():
    a = wt.heater_tuning(pitch=5e-6, cells_per_wavelength=14)["crosstalk"]
    b = wt.heater_tuning(pitch=20e-6, cells_per_wavelength=14)["crosstalk"]
    assert 0 < b < a < 1


def test_athermal_sin_under_polymer():
    r = wt.athermal_design(parameter="core_height", lower=0.1e-6, upper=0.6e-6, tol=5e-9, core_material="si3n4",
                           core_width=1.2e-6, clad_material="su8", cells_per_wavelength=14)
    assert r["found"]
    assert r["dlambda_dT_lower"] < 0 < r["dlambda_dT_upper"]
    assert abs(r["dlambda_dT"]) < 0.5e-12
    none = wt.athermal_design(lower=0.3e-6, upper=0.6e-6, tol=5e-8, cells_per_wavelength=12)   # Si wire, oxide clad
    assert not none["found"]
