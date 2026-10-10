import numpy as np
import pytest

from engines.bragg_grating import engine as bg
from engines.grating_coupler import engine as gc
from engines.slab_waveguide.engine import neff_three_layer

LAM = 1.55e-6
TFLN = dict(n_sub=1.444, n_core=2.138, n_clad=1.0, thickness=0.6e-6)


def grating(**kw):
    a = {**TFLN, "etch_depth": 0.1e-6, "period": 0.85e-6, **kw}
    return gc.SurfaceGrating(**a)


@pytest.mark.parametrize("f,m", [(0.5, 1), (0.3, 1), (0.3, 2), (0.3, 3), (0.7, 2)])
def test_rect_profile_kappa_matches_rectangular_formula(f, m):
    g = grating(fill=f)
    tab = g.neff_table(LAM)
    want = bg.coupling_coefficient(tab["vals"][1] - tab["vals"][0], LAM, f, m)["kappa"]
    assert g.coupling(LAM, m) == pytest.approx(want, rel=1e-4)        # cell-averaged samples: O((πm/1024)²)


def test_table_endpoints_are_slab_modes():
    g = grating(profile="trap", sidewall_angle=np.radians(60))
    tab = g.neff_table(LAM)
    assert tab["vals"][-1] == pytest.approx(neff_three_layer(LAM, 1.444, 2.138, 1.0, 0.6e-6), abs=1e-13)
    assert tab["vals"][0] == pytest.approx(neff_three_layer(LAM, 1.444, 2.138, 1.0, 0.5e-6), abs=1e-13)
    assert np.all(np.diff(tab["vals"]) > 0)


def test_cut_off_slices_take_the_cladding_floor():
    g = grating(etch_depth=0.6e-6)
    tab = g.neff_table(LAM)
    assert tab["cut"] and tab["vals"][0] == 1.444


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_slab_mode_norm_and_continuity(pol):
    m = gc.slab_mode(LAM, 1.444, 2.138, 1.0, 0.6e-6, pol)
    x = np.linspace(-4e-6, 4.6e-6, 400001)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    assert trapz(m["field"](x) ** 2, x) == pytest.approx(m["norm"], rel=1e-6)
    e = 1e-15
    for x0 in (0.0, 0.6e-6):
        assert m["field"](x0 - e) == pytest.approx(float(m["field"](x0 + e)), abs=1e-6)


def test_rect_fill_is_not_quantised():
    for f in (0.3, 0.37, 0.61):
        assert grating(fill=f).g.mean() == pytest.approx(f, abs=1e-14)


def test_profiles():
    rect, is_rect = gc.profile_function("rect", 0.4, 1e-6)
    assert is_rect and rect(0.5) == 1 and rect(0.05) == 0
    assert gc.profile_function("trap", 0.4, 1e-6, 0.1e-6, np.pi / 2)[1]
    assert gc.profile_function("smooth", 0.4, 1e-6, edge_sigma=0.0)[1]
    u = (np.arange(4096) + 0.5) / 4096
    for kind in ("trap", "smooth", "sine", "tri"):
        g, _ = gc.profile_function(kind, 0.4, 1e-6, 0.1e-6, np.radians(70), 20e-9)
        v = np.array([g(x) for x in u])
        assert v.min() >= 0 and v.max() <= 1
        assert np.mean(v > 0.5) == pytest.approx(0.4, abs=2e-3), kind      # symmetric profiles: g > 1/2 over f
    saw, _ = gc.profile_function("saw", 0.3, 1e-6)
    assert saw(0.3 - 1e-9) == pytest.approx(1, abs=1e-6) and saw(0.0) == 0


def test_plane_wave_uniform_medium_and_energy():
    g = gc.SurfaceGrating(1.5, 1.5, 1.5, 0.5e-6, 0.0, 1e-6)
    k = 2 * np.pi / LAM
    pw = g.plane_wave(LAM, 0.3 * k, 0.2e-6)
    assert abs(pw["r"]) < 1e-12 and abs(pw["F"]) == pytest.approx(1, abs=1e-12) and abs(pw["tau"]) == pytest.approx(1, abs=1e-12)
    g = grating(n_handle=3.476 + 0j, box_thickness=1.7e-6)
    for side in ("top", "bottom"):
        for s in (0.0, 0.2, 0.6, 0.95):
            pw = g.plane_wave(LAM, s * k * (1.0 if side == "top" else 3.476), 0.55e-6, side)
            assert abs(pw["r"]) ** 2 + abs(pw["tau"]) ** 2 == pytest.approx(1, abs=1e-12)


def test_first_order_grating_does_not_radiate():
    g = grating()
    N0 = np.mean(g.neff_profile(LAM))
    g = grating(period=LAM / (2 * N0))
    r = g.radiation(LAM, periods=500)
    assert r["alpha_total"] == 0 and r["orders"] == []
    assert r["bragg"]["q"] == 1 and r["bragg"]["kappa"] == pytest.approx(g.coupling(LAM, 1), rel=1e-12)


def test_second_order_emission_angle_and_bragg():
    g = grating()
    N0 = float(np.mean(g.neff_profile(LAM)))
    Lam = 0.95 * LAM / N0
    g = grating(period=Lam)
    r = g.radiation(LAM, periods=300)
    up = [o for o in r["orders"] if o["medium"] == "cladding"]
    assert len(up) == 1 and up[0]["q"] == 1
    assert np.sin(up[0]["theta"]) == pytest.approx(N0 - LAM / Lam, rel=1e-6)
    assert r["bragg"]["q"] == 2 and r["bragg"]["lambda_B"] == pytest.approx(N0 * Lam, rel=1e-12)


def test_shallow_grating_radiation_scales_as_h_squared():
    a = [grating(etch_depth=h).radiation(LAM)["alpha_total"] / h**2 for h in (2e-9, 4e-9, 8e-9)]
    assert a[1] == pytest.approx(a[0], rel=0.01) and a[2] == pytest.approx(a[0], rel=0.03)


def test_gold_handle_closes_the_downward_channel():
    r = gc.grating_radiation(LAM, **TFLN, etch_depth=0.1e-6, period=0.85e-6, handle="au")
    assert r["alpha_down"] == 0 and r["up_fraction"] == 1
    r = gc.grating_radiation(LAM, **TFLN, etch_depth=0.1e-6, period=0.85e-6)
    assert r["alpha_up"] + r["alpha_down"] == pytest.approx(r["alpha_total"]) and 0 < r["up_fraction"] < 1


def test_box_interference_changes_directionality():
    ups = [gc.grating_radiation(LAM, **TFLN, etch_depth=0.1e-6, period=0.85e-6, handle="si", box_thickness=b)["up_fraction"]
           for b in np.linspace(1.5e-6, 2.1e-6, 7)]
    assert max(ups) - min(ups) > 0.1


def test_far_field_peak_and_width():
    g = grating()
    r = g.radiation(LAM)
    N = 200
    th = np.radians(np.linspace(0, 12, 24001))
    I = g.far_field(r, th, 1.0, N, alpha_power=0.0)
    main = [o for o in r["orders"] if o["medium"] == "cladding"][0]
    assert th[np.argmax(I)] == pytest.approx(main["theta"], abs=2e-5)
    s = np.sin(th[I > I.max() / 2])
    assert s.max() - s.min() == pytest.approx(0.8859 * LAM / (N * g.period), rel=0.02)   # uniform array, FWHM in sin θ


def test_profile_coupling_result():
    r = gc.profile_coupling(LAM, **TFLN, etch_depth=0.1e-6, period=0.36e-6, fill_factor=0.5)
    assert r["kappa"] == pytest.approx(r["kappa_rect"], rel=2e-5)
    r = gc.profile_coupling(LAM, **TFLN, etch_depth=0.1e-6, period=0.36e-6, profile="sine")
    assert r["kappa"] < r["kappa_rect"]                    # sinusoid: fundamental π/4 of the square wave's
    assert r["kappa"] == pytest.approx(np.pi / 4 * r["kappa_rect"], rel=0.05)


def test_tm_plane_wave_conserves_energy_and_tm_mode_quantities():
    g = grating(n_handle=3.476 + 0j, box_thickness=1.7e-6, polarization="TM")
    k = 2 * np.pi / LAM
    for side in ("top", "bottom"):
        for sn in (0.0, 0.3, 0.8):
            pw = g.plane_wave(LAM, sn * k * (1.0 if side == "top" else 3.476), 0.55e-6, side, "TM")
            assert abs(pw["r"]) ** 2 + abs(pw["tau"]) ** 2 == pytest.approx(1, abs=1e-12)
    m = gc.slab_mode(LAM, 1.444, 2.138, 1.0, 0.6e-6, "TM")
    x = np.linspace(-4e-6, 4.6e-6, 400001)
    eps = np.where(x < 0, 1.444**2, np.where(x <= 0.6e-6, 2.138**2, 1.0))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    assert trapz(m["field"](x) ** 2 / eps, x) == pytest.approx(m["norm_eps"], rel=1e-6)
    d = np.gradient(m["field"](x), x) / eps                              # (1/ε) dH/dx is continuous and equals dfield
    i = np.searchsorted(x, 0.3e-6)
    assert m["dfield"](x[i]) == pytest.approx(d[i], rel=1e-4)
    assert m["dfield"](0.6e-6 - 1e-15) == pytest.approx(float(m["dfield"](0.6e-6 + 1e-15)), rel=1e-8)


def test_tm_tooth_model_thin_limit():
    # h → 0: ρ = (N² ε2 - (N² - ε2) ε1) / (N² ε2 + (N² - ε2) ε1), independent of the screening constant
    g = grating(etch_depth=1e-14, polarization="TM")
    N = gc.slab_mode(LAM, 1.444, 2.138, 1.0, 0.6e-6, "TM")["neff"]
    e1, e2 = 2.138**2, 1.0
    want = (N * N * e2 - (N * N - e2) * e1) / (N * N * e2 + (N * N - e2) * e1)
    assert g.tm_weights(LAM)["rho"] == pytest.approx(want, rel=1e-6)
    assert grating(polarization="TE").tm_weights(LAM)["rho"] == 1.0
