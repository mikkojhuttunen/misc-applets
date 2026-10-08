import numpy as np
import pytest

from engines.algaas_rib import engine as E
from engines.qpm_shg import engine as q
from engines.slab_waveguide.engine import neff_three_layer

LAM = 1.55e-6


def test_gaas_index_matches_skauli():
    # Skauli et al. 2003 GaAs at 1.55 µm: 3.3737; the single-oscillator model is within 5e-3
    assert float(E.algaas_n(0.0, LAM)) == pytest.approx(3.3737, abs=5e-3)


def test_index_falls_with_al_and_is_nan_above_gap():
    assert E.algaas_n(0.5, LAM) < E.algaas_n(0.2, LAM)
    assert np.isnan(E.algaas_n(0.0, 0.7e-6))
    r = E.algaas_index(0.2, 0.775e-6)        # 1.6 eV against a 1.69 eV gap
    assert any("WARNING" in a for a in r.assumptions)


def test_slab_modes_match_three_layer_engine():
    for pol in ("TE", "TM"):
        mine = E.slab_modes(LAM, 1.444, [3.2], [0.5e-6], 1.0, pol)
        ref = [neff_three_layer(LAM, 1.444, 3.2, 1.0, 0.5e-6, pol, m) for m in range(len(mine))]
        assert mine == pytest.approx(ref, abs=1e-6)


def test_multilayer_with_identical_layers_equals_one_layer():
    a = E.slab_modes(LAM, 1.444, [3.2, 3.2], [0.2e-6, 0.3e-6], 1.0, "TE")
    b = E.slab_modes(LAM, 1.444, [3.2], [0.5e-6], 1.0, "TE")
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_fd_reproduces_slab_when_there_is_no_rib(pol):
    # h_rib = 0: the 2D solver must give the analytic slab index (lateral box of 14 µm adds ≈ -5e-4)
    st = E.RibStack(h_core=0.4e-6, h_rib=0.0, width=10e-6)
    n = E.layer_indices(st, LAM)
    ref = E.slab_modes(LAM, n["bottom"], [n["core"]], [st.h_core], n["top"], pol)[0]
    ms = E.rib_modes(st, LAM, pol, 3, 20e-9, True)
    assert ms.neff[0] == pytest.approx(ref, abs=1.2e-3)


def test_richardson_improves_convergence():
    st = E.RibStack(0.25e-6, 0.25e-6, 1.0e-6)
    ref = E.rib_modes(st, LAM, "TE", 2, 10e-9, True).neff[0]
    plain = E.rib_modes(st, LAM, "TE", 2, 30e-9, False).neff[0]
    extra = E.rib_modes(st, LAM, "TE", 2, 30e-9, True).neff[0]
    assert abs(extra - ref) < abs(plain - ref) / 3


def test_mode_labels_and_ordering():
    ms = E.rib_modes(E.RibStack(0.25e-6, 0.25e-6, 1.0e-6), LAM, "TE", 4, 30e-9)
    assert ms.labels[:3] == [(0, 0), (1, 0), (2, 0)]
    assert np.all(np.diff(ms.neff) < 0)
    for f in ms.fields:
        assert np.sum(f**2 * ms.cell_area) == pytest.approx(1.0, rel=1e-9)


def test_eim_is_close_to_fd_and_exact_without_rib():
    st = E.RibStack(0.25e-6, 0.25e-6, 1.0e-6)
    fd = E.rib_modes(st, LAM, "TE", 2, 20e-9, True).neff[0]
    assert E.eim_neff(st, LAM, "TE") == pytest.approx(fd, abs=3e-2)
    flat = E.RibStack(0.4e-6, 0.0, 1.0e-6)
    n = E.layer_indices(flat, LAM)
    assert E.eim_neff(flat, LAM, "TM") == pytest.approx(E.slab_modes(LAM, n["bottom"], [n["core"]], [0.4e-6], n["top"], "TM")[0])


def test_sign_map_has_core_plus_rib_minus():
    ms = E.rib_modes(E.RibStack(0.2e-6, 0.4e-6, 1.0e-6), LAM, "TE", 2, 30e-9)
    area = ms.cell_area
    plus = np.sum(area[ms.sign > 0.5])
    minus = np.sum(area[ms.sign < -0.5])
    assert minus == pytest.approx(0.4e-6 * 1.0e-6, rel=1e-9)
    assert plus == pytest.approx(0.2e-6 * (ms.xe[-1] - ms.xe[0]), rel=1e-9)


def test_overlap_vanishes_for_even_sh_without_sign_flip_in_symmetric_structure():
    # two mirror-symmetric lateral halves: an odd-in-x SH mode (lateral label 1) has zero overlap with an even pump²
    st = E.RibStack(0.2e-6, 0.5e-6, 1.4e-6)
    p = E.rib_modes(st, LAM, "TE", 2, 30e-9)
    s = E.rib_modes(st, LAM / 2, "TM", 40, 30e-9)
    i = s.find((1, 1))
    assert i is not None
    assert E.overlap(p, p.find((0, 0)), s, i)["gamma"] < 1e-3 * 1e6


def test_layer_poling_beats_uniform_sign_for_odd_sh_mode():
    r = E.shg_design(E.RibStack(0.2e-6, 0.5e-6, 1.4e-6), LAM, sh_label=(0, 2))
    assert r["gamma"] > 5 * r["gamma_uniform"]
    assert r["gamma"] <= r["gamma_max"] + 1e-9


def test_delta_k_consistent_with_qpm_engine_and_default_is_phase_matched():
    r = E.shg_design(E.RibStack(0.2e-6, 0.5e-6, 1.4e-6), LAM, sh_label=(0, 2))
    ref = q.phase_matching(LAM, r["n_pump"], r["n_sh"])
    assert r["delta_k"] == pytest.approx(ref["delta_k"], rel=1e-12)
    assert abs(r["delta_n"]) < 5e-3          # the documented default geometry is close to modal phase matching


def test_tuner_finds_zero_of_delta_n():
    st = E.RibStack(0.2e-6, 0.5e-6, 1.4e-6)
    roots, _ = E.tune_parameter(st, LAM, "width", 1.2e-6, 1.6e-6, sh_label=(0, 2), npts=5)
    assert len(roots) == 1
    w, slope = roots[0]
    assert abs(E.delta_n(E.RibStack(0.2e-6, 0.5e-6, w), LAM, sh_label=(0, 2))) < 1e-5
    assert slope < 0


def test_efficiency_scales_with_d_eff_squared():
    st = E.RibStack(0.2e-6, 0.5e-6, 1.4e-6)
    a = E.shg_design(st, LAM, d_eff=50e-12)["eta_norm"]
    b = E.shg_design(st, LAM, d_eff=100e-12)["eta_norm"]
    assert b == pytest.approx(4 * a, rel=1e-9)


def test_bad_inputs():
    with pytest.raises(ValueError, match="h_core"):
        E.RibStack(0.0, 0.3e-6, 1e-6).validate()
    with pytest.raises(ValueError, match="edge"):
        E.layer_indices(E.RibStack(0.2e-6, 0.3e-6, 1e-6, x_core=0.0), 0.7e-6)


def test_thin_tm01_design_is_phase_matched_with_near_ideal_sign_overlap():
    # 150 nm core / 200 nm rib: TM(0,1) matches the TE pump near w = 680 nm (15 nm grid); the SH node sits close
    # to the core/rib interface, so the layer-poled overlap is within 10 % of the best case
    st = E.RibStack(0.15e-6, 0.2e-6, 0.68e-6)
    r = E.shg_design(st, LAM, sh_label=(0, 1), step=20e-9)
    assert abs(r["delta_n"]) < 3e-3
    assert r["gamma"] > 0.9 * r["gamma_max"]
    assert r["gamma"] > 8 * r["gamma_uniform"]
    assert r["gamma"] > 2 * E.shg_design(E.RibStack(0.2e-6, 0.5e-6, 1.4e-6), LAM, sh_label=(0, 2))["gamma"]


def test_tm00_never_phase_matches_the_te_pump():
    # the SH fundamental is always more strongly guided than the pump: Δn stays > 0.2 over a wide range
    for hc, hr, w in ((0.15, 0.2, 0.7), (0.3, 0.4, 1.0), (0.5, 0.6, 2.0)):
        assert E.delta_n(E.RibStack(hc * 1e-6, hr * 1e-6, w * 1e-6), LAM, sh_label=(0, 0)) > 0.2


def test_rib_offset_is_constant_or_linear_in_inverse_wavelength():
    base = E.RibStack(0.15e-6, 0.2e-6, 0.7e-6)
    flat = E.RibStack(0.15e-6, 0.2e-6, 0.7e-6, dn_rib=-0.1)
    disp = E.RibStack(0.15e-6, 0.2e-6, 0.7e-6, dn_rib=-0.1, dn_rib_sh=-0.14)
    for lam in (1.55e-6, 0.775e-6, 1.0e-6):
        assert E.layer_indices(flat, lam)["rib"] - E.layer_indices(base, lam)["rib"] == pytest.approx(-0.1, abs=1e-12)
        assert E.layer_indices(flat, lam)["core"] == E.layer_indices(base, lam)["core"]
    assert E.rib_offset(disp, 1.55e-6) == pytest.approx(-0.10)
    assert E.rib_offset(disp, 0.775e-6) == pytest.approx(-0.14)
    assert E.rib_offset(disp, 1 / (0.5 / 1.55e-6 + 0.5 / 0.775e-6)) == pytest.approx(-0.12)


def test_rib_index_offset_equals_alloy_change_of_same_index():
    # an engineered offset is just an index change of the rib: same n -> same modes
    a = E.RibStack(0.2e-6, 0.3e-6, 1.0e-6, x_rib=0.3, dn_rib=-0.05)
    n_target = E.layer_indices(a, LAM)["rib"]
    ms = E.rib_modes(a, LAM, "TE", 2, 40e-9, False)
    flat = E.RibStack(0.2e-6, 0.3e-6, 1.0e-6, x_rib=0.3, dn_algaas=0.0)
    lowered = E.RibStack(0.2e-6, 0.3e-6, 1.0e-6, x_rib=0.3, dn_rib=0.0)
    assert n_target == pytest.approx(E.layer_indices(flat, LAM)["rib"] - 0.05, abs=1e-12)
    assert ms.neff[0] < E.rib_modes(lowered, LAM, "TE", 2, 40e-9, False).neff[0]


def test_tuner_can_solve_for_the_rib_offset():
    st = E.RibStack(0.2e-6, 0.5e-6, 1.2e-6)   # Δn(0,2) = +0.012 at dn_rib = 0, zero near +0.10
    roots, _ = E.tune_parameter(st, LAM, "dn_rib", 0.0, 0.2, sh_label=(0, 2), npts=9)
    assert roots
    v = roots[0][0]
    from dataclasses import replace
    assert abs(E.delta_n(replace(st, dn_rib=v), LAM, sh_label=(0, 2))) < 1e-5


def test_gap_models_and_margin_rule():
    assert float(E.bandgap(0.3)) == pytest.approx(1.4240 + 1.247 * 0.3)
    assert float(E.bandgap(0.3, "afromowitz")) == pytest.approx(1.424 + 1.266 * 0.3 + 0.266 * 0.09)
    # project note: x = 0.3 gives ~0.2 eV above the 775 nm photon; Eg = 1.60 eV at x ~ 0.14
    assert float(E.absorption_report(0.3, 775e-9)["margin"]) == pytest.approx(0.198, abs=2e-3)
    assert E.min_al_fraction(775e-9, 0.0) == pytest.approx(0.141, abs=1e-3)


def test_urbach_loss_falls_exponentially_with_margin():
    a1 = float(E.urbach_alpha(1.70, 775e-9))
    a2 = float(E.urbach_alpha(1.80, 775e-9))
    assert a1 / a2 == pytest.approx(np.exp(0.10 / E.URBACH_EV), rel=1e-9)


def test_quantum_well_limits_and_trend():
    # infinite-well limit: e1 -> ħ²π²/(2 m L²) for a deep, narrow-in-energy well
    q = E.MQW(5e-9, 10e-9, 0.0, 0.45)
    r = E.mqw_edge(q)
    inf = 0.0380998 * np.pi**2 / (0.067 * 5.0**2)
    assert 0.25 * inf < r["e1"] < inf
    # narrower well: higher edge; deeper barrier: higher edge
    assert E.mqw_edge(E.MQW(2e-9, 10e-9, 0.0, 0.4))["edge"] > E.mqw_edge(E.MQW(5e-9, 10e-9, 0.0, 0.4))["edge"]
    assert E.mqw_edge(E.MQW(3e-9, 10e-9, 0.0, 0.5))["edge"] > E.mqw_edge(E.MQW(3e-9, 10e-9, 0.0, 0.3))["edge"]
    # well wide enough to look like bulk: edge -> bulk gap minus the exciton term
    assert E.mqw_edge(E.MQW(40e-9, 10e-9, 0.0, 0.3))["edge"] == pytest.approx(1.424 - 0.008 + 0.0, abs=0.012)


def test_max_well_width_hits_the_requested_edge():
    w = E.mqw_max_well(0.1, 0.45, 1.75)
    assert E.mqw_edge(E.MQW(w, 10e-9, 0.1, 0.45))["edge"] == pytest.approx(1.75, abs=1e-5)


def test_mqw_form_birefringence_is_positive_and_vanishes_for_equal_layers():
    q = E.MQW(5e-9, 5e-9, 0.2, 0.45)
    n = E.mqw_indices(q, 1.55e-6)
    assert n["n_par"] > n["n_perp"]
    same = E.mqw_indices(E.MQW(5e-9, 5e-9, 0.3, 0.3), 1.55e-6)
    assert same["n_par"] == pytest.approx(same["n_perp"], abs=1e-9)
    assert same["n_par"] == pytest.approx(float(E.algaas_n(0.3, 1.55e-6)), abs=1e-3)


def test_mqw_core_uses_polarisation_dependent_index_and_flags_edge():
    from dataclasses import replace
    q = E.MQW(5e-9, 5e-9, 0.25, 0.45)
    st = E.RibStack(0.15e-6, 0.2e-6, 0.7e-6, core_mqw=q, rib_mqw=q)
    te, tm = E.layer_indices(st, 1.55e-6, "TE")["core"], E.layer_indices(st, 1.55e-6, "TM")["core"]
    assert te > tm
    ab = E.sh_absorption(st, 775e-9)
    assert ab["core"]["edge"] == pytest.approx(float(E.mqw_edge(q)["edge"]))
    bad = replace(st, core_mqw=E.MQW(8e-9, 5e-9, 0.0, 0.45))
    assert E.sh_absorption(bad, 775e-9)["margin"] < 0
