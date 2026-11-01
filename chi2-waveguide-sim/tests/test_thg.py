"""Milestone 5: cascaded THG designs on the FD mode model."""
import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import poling as P
from chi2sim import thg as T
from chi2sim import wafer as Wf
from chi2sim import waveguide as W
from engines.cwe_cascade import engine as casc

LAM0, L = 1.55e-6, 3e-3


@pytest.fixture(scope="module")
def model():
    return W.THGModel(W.Ridge(), np.linspace(1.5e-6, 1.6e-6, 5))


def test_model_consistency_with_shg_model(model):
    shg = W.SHGModel(W.Ridge(), np.linspace(1.5e-6, 1.6e-6, 4), h=15e-9)
    assert float(model.dk_a(LAM0)) == pytest.approx(float(shg(LAM0)), rel=1e-5)
    assert float(model.n("aa", LAM0)) == pytest.approx(float(shg.a_eff(LAM0)), rel=1e-4)
    pa, pb = model.periods(LAM0)
    assert 3.6e-6 < pa < 4.4e-6 and 2.5e-6 < pb < 4.0e-6
    c = model.coupling(LAM0)
    assert float(c["kappa_3"]) == pytest.approx(3 * float(c["kappa_1"]))


def test_two_section_matches_closed_form(model):
    p = T.two_section(model, LAM0, L, fab=None)
    eta = float(T.thg_spectrum(model, p, [LAM0])[0][0])
    assert eta == pytest.approx(T.reference_efficiencies(model, LAM0, L)["two_section_closed"], rel=0.02)
    assert eta < T.reference_efficiencies(model, LAM0, L)["matched_uniform"]


def test_superlattice_has_both_gratings_and_blend_beats_two_section(model):
    p = T.superlattice(model, LAM0, L, 0.5, 0.0, fab=None)
    ka, kb = abs(float(model.dk_a(LAM0))), abs(float(model.dk_b(LAM0)))
    from chi2sim import fast
    G = np.abs(fast.spectrum(p, [ka, kb, 0.5 * (ka + kb)])[0])
    assert G[0] > 0.3 * L and G[1] > 0.3 * L and G[2] < 0.05 * L
    two = float(T.thg_spectrum(model, T.two_section(model, LAM0, L), [LAM0])[0][0])
    pb, params, eta = T.optimise_blend(model, LAM0, L, starts=(0.8, 0.9), ends=(0.1, 0.2), phases=(0.0, np.pi))
    assert eta > two
    assert pb.lengths[1:-1].min() >= P.FAB_PRESETS["reproducible"]["l_min"] - 1e-15


def test_spectrum_matches_rk4_at_low_power(model):
    p = T.two_section(model, LAM0, 1.5e-3)
    lam = np.array([1.549e-6, 1.55e-6])
    eta, eta2 = T.thg_spectrum(model, p, lam, power=1e-3)
    c = model.coupling(LAM0)
    r = casc.cascade_coupled(p.boundaries, p.g, abs(float(model.dk_a(LAM0))), abs(float(model.dk_b(LAM0))),
                             float(c["kappa_a"]), float(c["kappa_1"]), float(c["kappa_2"]), float(c["kappa_3"]), 1e-3)
    assert float(r["P3"][0]) / 1e-9 == pytest.approx(eta[1], rel=0.01)


def test_higher_order_3w_mode_overlap_penalty():
    g = W.Ridge(etch=0.6e-6, width=1.2e-6)
    m00 = W.THGModel(g, np.linspace(1.5e-6, 1.6e-6, 4))
    m10 = W.THGModel(g, np.linspace(1.5e-6, 1.6e-6, 4), order3=(1, 0))
    assert m10.periods(LAM0)[1] > m00.periods(LAM0)[1]          # lower n3: longer SFG period
    assert float(m10.n("ab", LAM0)) > 50 * float(m00.n("ab", LAM0))
    assert m10.notes


def test_thickness_detunes_the_two_steps_differently(model):
    fam = Wf.ThicknessFamily(lambda g: W.THGModel(g, np.linspace(1.5e-6, 1.6e-6, 4)), W.Ridge(), film_points=3)
    s = T.double_matching_shift(fam, LAM0)
    assert abs(s["dlam_a_dt"] - s["dlam_b_dt"]) > 1.0              # nm/nm: re-tuning λ cannot fix both
    p = T.two_section(fam.nominal, LAM0, L)
    r = T.thg_devices(fam, p, np.linspace(1.51e-6, 1.59e-6, 161), LAM0, n_devices=12, seed=1)
    assert r["eta"].shape == (12, 161)
    assert np.median(r["peak_eta"]) < r["nominal"].max()
