"""Waveguide layer: FD solver vs exact slab, EIM, convergence; overlap; crystal data; TFLN/TFLT SHG model."""
import math

import numpy as np
import pytest

import chi2sim  # noqa: F401
from chi2sim import materials as M
from chi2sim import synthesis as S
from chi2sim import waveguide as W
from engines.qpm_shg import engine as qpm

LAM = 1.55e-6


@pytest.fixture(scope="module")
def tfln():
    return W.SHGModel(W.Ridge(), np.linspace(1.45e-6, 1.65e-6, 6))


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_fd_unetched_film_matches_exact_anisotropic_slab(pol):
    # etch = 0: laterally uniform; remove the Dirichlet box's lowest lateral mode (π/W)² analytically
    g = W.Ridge(etch=0.0, width=4e-6)
    exact = W._slab_neff(W.indices(g, LAM), g.film, LAM, pol)
    k0 = 2 * math.pi / LAM
    errs = []
    for h in (20e-9, 10e-9):
        gr = W.build_grid(g, LAM, h, lateral_margin=8.0)
        m = W.solve(g, LAM, pol, gr, n_modes=1, n_min=1.5)[0]
        wbox = gr.xe[-1] - gr.xe[0]
        errs.append(abs(math.sqrt(m.neff**2 + (math.pi / wbox / k0) ** 2) - exact))
    assert errs[0] < 6e-4
    assert errs[1] < errs[0] / 2.5                      # second order in h


def test_fd_converges_and_agrees_with_eim():
    g = W.Ridge()
    ne = []
    for h in (40e-9, 20e-9, 10e-9):
        gr = W.build_grid(g, LAM, h)
        ne.append(W.fundamental(g, LAM, "TE", gr).neff)
    assert abs(ne[2] - ne[1]) < abs(ne[1] - ne[0]) / 2
    assert abs(ne[2] - ne[1]) < 5e-4
    for pol in ("TE", "TM"):
        assert W.fundamental(g, LAM, pol).neff == pytest.approx(W.eim(g, LAM, pol), abs=0.01)


def test_fundamental_mode_is_symmetric_and_guided():
    g = W.Ridge()
    gr = W.build_grid(g, LAM)
    m = W.fundamental(g, LAM, "TE", gr)
    assert np.allclose(m.field, m.field[:, ::-1], atol=1e-6)
    assert m.order == (0, 0) and m.film_fraction > 0.8
    ix = W.indices(g, LAM)
    assert W._slab_neff(ix, g.slab, LAM, "TE") < m.neff < ix["film"][0]


def test_cut_selects_the_index_the_mode_sees():
    # x-cut TE sees n_e (lateral = crystal z); z-cut TE sees n_o; LN is negative uniaxial
    x = W.fundamental(W.Ridge(cut="x"), LAM, "TE").neff
    z = W.fundamental(W.Ridge(cut="z"), LAM, "TE").neff
    assert z > x + 0.03
    assert M.film_tensor("ln", "x", LAM)[0] == pytest.approx(M.principal("ln", LAM)[1])
    assert M.film_tensor("ln", "z", LAM)[1] == pytest.approx(M.principal("ln", LAM)[1])


def test_a_eff_gaussian_hand_value():
    # E_p = E_SH = G = exp(-x²/a² - y²/b²), film everywhere: A_eff = (πab/2)³/(πab/3)² = 9πab/8
    g = W.Ridge(etch=0.0)
    gr = W.build_grid(g, LAM, 10e-9)
    gr.f_film[:] = 1.0
    a, b = 0.8e-6, 0.4e-6
    X, Y = np.meshgrid(gr.xc, gr.yc - 0.3e-6)
    G = np.exp(-(X / a) ** 2 - (Y / b) ** 2)
    assert W.a_eff_shg(G, G, gr) == pytest.approx(9 * math.pi * a * b / 8, rel=2e-3)


def test_miller_scaling_and_coefficients():
    assert M.miller_d("ln", "x", "TE", 1.064e-6) == pytest.approx(25.2e-12)
    d1550 = M.miller_d("ln", "x", "TE", 1.55e-6)
    assert 0.8 * 25.2e-12 < d1550 < 25.2e-12            # normal dispersion: smaller further from resonance
    assert M.miller_d("lt", "x", "TE", 1.064e-6) / M.miller_d("ln", "x", "TE", 1.064e-6) == pytest.approx(13.8 / 25.2)
    assert M.miller_d("ln", "z", "TM", 1.55e-6) == pytest.approx(d1550)
    assert M.miller_d("ln", "x", "TM", 1.55e-6) == 0.0      # E along crystal x: d11 = 0 in class 3m
    assert M.coefficient("ln", "z", "TE")[0] == "d22"


def test_tfln_shg_model_against_literature_scale(tfln):
    # Wang et al., Optica 5, 1438 (2018): x-cut TFLN ridge, first-order period 4 µm, 2600 %/(W cm²) measured
    s = tfln.summary(LAM, length=4e-3)
    assert 3.6e-6 < s["period"] < 4.4e-6
    assert 1e3 < s["eta_qpm"] * 1e-2 < 1e4                  # %/(W cm²)
    assert 2.1 < s["ng_pump"] < 2.5 and s["ng_sh"] > s["ng_pump"]
    # bandwidth from the tabulated group indices agrees with qpm_shg's GVM formula
    ref = qpm.phase_matching(LAM, s["n_pump"], s["n_sh"], s["ng_pump"], s["ng_sh"], 4e-3)
    assert s["fwhm"] == pytest.approx(ref["fwhm"], rel=0.02)
    assert ref["period"] == pytest.approx(s["period"], rel=1e-6)


def test_spline_reproduces_direct_solve(tfln):
    lam = 1.5123e-6
    mp = W.fundamental(tfln.geom, lam, "TE", tfln.grid)
    ms = W.fundamental(tfln.geom, lam / 2, "TE", tfln.grid)
    assert tfln.neff_pump(lam) == pytest.approx(mp.neff, abs=2e-6)
    assert tfln.neff_sh(lam) == pytest.approx(ms.neff, abs=2e-6)
    with pytest.raises(ValueError, match="outside"):
        tfln(1.7e-6)


def test_tflt_is_weaker_than_tfln(tfln):
    lt = W.SHGModel(W.Ridge(material="lt"), np.linspace(1.5e-6, 1.6e-6, 4))
    a, b = tfln.summary(LAM), lt.summary(LAM)
    assert b["eta_norm"] / a["eta_norm"] == pytest.approx((b["d"] / a["d"]) ** 2 * a["a_eff"] / b["a_eff"]
                                                          * (a["n_pump"] ** 2 * a["n_sh"]) / (b["n_pump"] ** 2 * b["n_sh"]), rel=1e-9)
    assert 0.2 < b["eta_norm"] / a["eta_norm"] < 0.45
    assert 3.6e-6 < b["period"] < 4.6e-6


def test_design_with_modal_dispersion(tfln):
    d = S.design(tfln, LAM, 40e-9, 4e-3, mode="chirp")
    assert d.report["fwhm"] == pytest.approx(40e-9, rel=0.02)
    eta = d.efficiency(float(tfln.coupling(LAM)["kappa"]))
    assert eta.max() > 0


def test_cache_roundtrip(tfln, tmp_path):
    tfln.save(tmp_path / "t.npz")
    m = W.SHGModel.load(tmp_path / "t.npz")
    assert m.geom == tfln.geom
    lam = np.linspace(1.46e-6, 1.64e-6, 7)
    assert np.array_equal(m(lam), tfln(lam)) and np.array_equal(m.a_eff(lam), tfln.a_eff(lam))
    c1 = W.SHGModel.cached(W.Ridge(width=1.2e-6), np.linspace(1.5e-6, 1.6e-6, 4), cache_dir=tmp_path)
    c2 = W.SHGModel.cached(W.Ridge(width=1.2e-6), np.linspace(1.5e-6, 1.6e-6, 4), cache_dir=tmp_path)
    assert c2.grid is None and np.array_equal(c1.n_p, c2.n_p)
    assert len(list(tmp_path.glob("shg_*.npz"))) == 1


def test_width_scan_matches_model(tfln):
    ws = W.width_scan(W.Ridge(), [1.3e-6, 1.4e-6, 1.5e-6], LAM)
    assert ws["period"][1] == pytest.approx(float(tfln.period(LAM)), rel=1e-6)
    assert np.all(np.isfinite(ws["ddk_dw"]))
