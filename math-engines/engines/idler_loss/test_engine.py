import numpy as np
import pytest

from engines.idler_loss import engine as il
from engines.opa_chi2 import engine as o2
from engines.opa_chi3 import engine as o3


@pytest.mark.parametrize("dk", [0.0, 80.0, 250.0, 1000.0])
def test_lossless_matches_closed_form(dk):
    s, c = il.propagate_linear(120.0, dk, 0.025)
    G = o2._gain_closed_form(120.0, dk, 0.025)[0]
    assert abs(s) ** 2 == pytest.approx(G, rel=1e-12)
    assert abs(c) ** 2 == pytest.approx(G - 1, rel=1e-12)


def test_dumps_without_attenuation_change_nothing():
    a = il.linear_gain(120.0, 60.0, 0.03)
    b = il.linear_gain(120.0, 60.0, 0.03, dumps=7, dump_loss_db=0.0)
    assert b["signal_gain"] == pytest.approx(a["signal_gain"], rel=1e-12)


@pytest.mark.parametrize("n", [1, 4, 9])
def test_ideal_dumps_restart_each_segment(n):
    # idler fully removed at every dump: each segment starts with c = 0 -> s gains cosh(Γ L/(N+1))
    G, L = 150.0, 0.03
    r = il.linear_gain(G, 0.0, L, dumps=n, dump_loss_db=400.0)
    assert r["signal_gain"] == pytest.approx(np.cosh(G * L / (n + 1)) ** (2 * (n + 1)), rel=1e-12)
    assert r["idler_attenuation_db"] == pytest.approx(400.0 * n)


def test_signal_loss_alone():
    r = il.linear_gain(0.0, 0.0, 2.0, alpha_signal=0.3, dumps=3, dump_loss_signal_db=1.0)
    assert r["signal_gain"] == pytest.approx(np.exp(-0.6) * 10 ** (-0.3), rel=1e-12)


def test_adiabatic_limit():
    G, a, L = 50.0, 2.0e4, 0.2      # α_i ≫ Γ: signal grows at Γ² α / ((α/2)² + Δk²)
    for dk in (0.0, 3000.0, 1.0e4):
        r = il.linear_gain(G, dk, L, alpha_idler=a)
        assert r["signal_gain_db"] == pytest.approx(10 * np.log10(np.exp(r["adiabatic_gain_coefficient"] * L)), rel=2e-3)
    # loss makes the gain less sensitive to Δk
    lossy = [il.linear_gain(G, dk, L, alpha_idler=a)["signal_gain_db"] for dk in (0.0, 3000.0)]
    clean = [il.linear_gain(G, dk, L)["signal_gain_db"] for dk in (0.0, 3000.0)]
    assert lossy[1] / lossy[0] > clean[1] / clean[0]


@pytest.mark.parametrize("kw", [dict(alpha_idler=300.0), dict(dumps=4, dump_loss_db=12.0),
                                dict(alpha_idler=80.0, dumps=2, dump_loss_db=6.0, alpha_signal=5.0, dump_loss_signal_db=0.5)])
def test_chi2_rk4_matches_linear_solution(kw):
    G, dk, L, r0 = 140.0, 70.0, 0.025, 1e-12
    out = o2.propagate_normalised(G, dk, L, r0, alpha_signal=kw.get("alpha_signal", 0.0), alpha_idler=kw.get("alpha_idler", 0.0),
                                  dumps=kw.get("dumps", 0), dump_loss_db=kw.get("dump_loss_db", 0.0),
                                  dump_loss_signal_db=kw.get("dump_loss_signal_db", 0.0), max_steps=20000)
    s, c = il.propagate_linear(G, dk, L, **kw)
    assert out["fs"] / r0 == pytest.approx(abs(s) ** 2, rel=1e-6)
    assert out["fi"] / r0 == pytest.approx(abs(c) ** 2, rel=1e-6)


@pytest.mark.parametrize("kw", [dict(alpha_idler=0.05), dict(dumps=4, dump_loss_db=10.0)])
def test_chi3_rk4_matches_linear_solution(kw):
    lp, ls, g, P, L, db = 1550e-9, 1556e-9, 0.01, 2.0, 400.0, -0.03
    li = float(o3.idler_wavelength(lp, ls))
    r0 = 1e-12
    out = o3.propagate_normalised(g * P, db, L, r0, lp / ls, lp / li, max_steps=50000, **kw)
    _, r, _ = o3._rates(g, P, lp, ls)
    s, c = il.propagate_linear(r, db + 2 * g * P, L, **kw)
    assert out["fs"] / r0 == pytest.approx(abs(s) ** 2, rel=1e-5)
    assert out["fi"] / r0 == pytest.approx(abs(c) ** 2, rel=1e-5)


def test_loss_weight_profiles():
    c, W, e = 1570e-9, 10e-9, 0.2e-9
    lam = np.array([1500e-9, c - W / 2, c, c + W / 2, 1640e-9])
    pas = il.loss_weight("pass", lam, c, W, e)
    assert pas[2] == pytest.approx(1.0, abs=1e-12) and pas[0] == pytest.approx(0.0, abs=1e-12)
    assert pas[1] == pytest.approx(0.5 * np.tanh(W / e), rel=1e-12)       # half loss at the edge
    assert il.loss_weight("stop", lam, c, W, e) == pytest.approx(1 - pas)
    assert il.loss_weight("flat", lam).tolist() == [1.0] * 5
    pts = [(1550e-9, 0.0), (1560e-9, 0.8), (1580e-9, 0.2), (1600e-9, 1.0)]
    cu = il.loss_weight("custom", np.linspace(1540e-9, 1610e-9, 701), points=pts)
    for x, y in pts:
        assert il.loss_weight("custom", x, points=pts) == pytest.approx(y, abs=1e-12)
    assert cu.min() >= 0 and cu.max() <= 1


def test_loss_on_all_waves_also_attenuates_signal():
    ls = np.array([1556e-9])
    li = o3.idler_wavelength(1550e-9, ls)
    kw = dict(alpha_idler=0.2, loss_profile="pass", band_center=1550e-9, band_width=40e-9, band_edge=0.5e-9)
    args = (0.02, 0.0, 300.0, ls, li, 1550e-9, kw["alpha_idler"], 0, 0.0, kw["loss_profile"], kw["band_center"], kw["band_width"], kw["band_edge"])
    a = il.amplifier_with_loss(*args, loss_acts_on="idler")
    b = il.amplifier_with_loss(*args, loss_acts_on="all")
    assert b["gain"][0] < a["gain"][0] and b["weight_pump"] == pytest.approx(1.0) and a["weight_pump"] == 0.0


def test_validation():
    with pytest.raises(ValueError, match="dumps"):
        il.propagate_linear(1.0, 0.0, 1.0, dumps=-1)
    with pytest.raises(ValueError, match="profile"):
        il.loss_weight("gauss", 1e-6)
    with pytest.raises(ValueError, match="band_width"):
        il.loss_weight("pass", 1e-6, 1e-6, 0.0, 1e-9)
    with pytest.raises(ValueError, match="points"):
        il.loss_weight("custom", 1e-6)
    with pytest.raises(ValueError, match="loss_acts_on"):
        il.amplifier_with_loss(1.0, 0.0, 1.0, 1.6e-6, 1.5e-6, 0.78e-6, loss_acts_on="pump")
