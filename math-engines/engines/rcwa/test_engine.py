import numpy as np
import pytest

from engines.anisotropic_slab.engine import slab_modes
from engines.bragg_grating.engine import stack_R_oblique
from engines.crigf.engine import CRIGF
from engines.grating_coupler.engine import SurfaceGrating
from engines.rcwa import engine as rc
from engines.slab_waveguide.engine import neff_three_layer

LAM = 1.55e-6
STACK = (1.444, 2.138, 1.0, 0.6e-6)


def coupler(h, pol="TE", detune=0.95, theta=0.0, **kw):
    """Second-order-ish surface grating with period detune × λ / (N0 - sin θ)."""
    g = SurfaceGrating(*STACK, h, 0.8e-6, polarization=pol, **kw)
    N0 = np.mean(g.neff_profile(LAM))
    return SurfaceGrating(*STACK, h, detune * LAM / (N0 - np.sin(theta)), polarization=pol, **kw)


# ---------------------------------------------------------------- plane waves
@pytest.mark.parametrize("pol,tmm", [("TE", "s"), ("TM", "p")])
def test_uniform_stack_matches_transfer_matrix(pol, tmm):
    st = rc.Stack(1e-6, 1.5, 1.0, [(0.3e-6, 2.0), (0.2e-6, 1.7)], pol)
    d = rc.RCWA(st, LAM, 3).diffraction(np.radians(20))
    want = stack_R_oblique(LAM, np.sin(np.radians(20)), 1.0, [(1.7, 0.2e-6), (2.0, 0.3e-6)], 1.5, tmm)
    assert d["R_total"] == pytest.approx(want, rel=1e-12)
    assert d["R_total"] + d["T_total"] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_lossless_grating_conserves_energy_and_converges(pol):
    st = rc.Stack(1.2e-6, 1.45, 1.0, [(0.5e-6, [(0.4, 1.0), (0.6, 2.0)])], pol)
    out = [rc.RCWA(st, LAM, M).diffraction(np.radians(10)) for M in (10, 20, 40)]
    for d in out:
        assert d["R_total"] + d["T_total"] == pytest.approx(1, abs=1e-10)
        assert (d["R"] >= 0).all() and (d["T"] >= 0).all()
    assert out[1]["R_total"] == pytest.approx(out[2]["R_total"], abs=2e-5)


def test_absorbing_layer_takes_power():
    st = rc.Stack(1.0e-6, 1.45, 1.0, [(0.1e-6, [(0.5, 1.0), (0.5, 0.2 + 3j)])], "TE")
    d = rc.RCWA(st, LAM, 15).diffraction(0.0)
    assert 0 < d["R_total"] + d["T_total"] < 1


# ---------------------------------------------------------------- guided modes
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_slab_mode(pol):
    st = rc.Stack(0.8e-6, 1.444, 1.0, [(0.6e-6, 2.138)], pol)
    want = neff_three_layer(LAM, 1.444, 2.138, 1.0, 0.6e-6, pol)
    assert rc.RCWA(st, LAM, 0).bloch_mode(want * 1.01, 1).real == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_subwavelength_lamellar_layer_is_a_uniaxial_effective_medium(pol):
    # Λ << λ: lamellar layer → ε_z = ε_y = arithmetic mean, ε_x (normal to the walls) = harmonic mean; checks Li's rule
    ns, nf, nc, t0, h, f = 1.444, 2.138, 1.0, 0.4e-6, 0.2e-6, 0.5
    ea, eh = f * nf**2 + (1 - f) * nc**2, 1 / (f / nf**2 + (1 - f) / nc**2)
    want = slab_modes(LAM, [ns, nf, (np.sqrt(ea), np.sqrt(ea), np.sqrt(eh)), nc], [t0, h], pol)["neff"][0]
    errs = []
    for Lam in (0.1e-6, 0.05e-6, 0.025e-6):
        st = rc.Stack(Lam, ns, nc, [(t0, nf), (h, [((1 - f) / 2, nc), (f, nf), ((1 - f) / 2, nc)])], pol)
        errs.append(abs(rc.RCWA(st, LAM, 5).bloch_mode(want, 1).real - want))
    assert errs[2] < 3e-5 and errs[1] / errs[2] > 3                 # O((Λ/λ)²)


def test_leaky_mode_converged_in_orders():
    g = coupler(50e-9)
    st, sp = rc.surface_grating_stack(g, LAM)
    a = [rc.RCWA(st, LAM, M).leaky_mode(g.radiation(LAM)["N0"], sp)["alpha"] for M in (10, 20)]
    assert a[0] == pytest.approx(a[1], rel=1e-5)


@pytest.mark.parametrize("h,tol", [(5e-9, 0.01), (20e-9, 0.02), (50e-9, 0.04), (100e-9, 0.05)])
def test_thin_sheet_radiation_TE_against_rigorous(h, tol):
    g = coupler(h)
    est = g.radiation(LAM)
    st, sp = rc.surface_grating_stack(g, LAM)
    m = rc.RCWA(st, LAM, 10).leaky_mode(est["N0"], sp, est["alpha_total"])
    up = sum(o["alpha"] for o in est["orders"] if o["medium"] == "cladding") / est["alpha_total"]
    assert m["alpha"] == pytest.approx(est["alpha_total"], rel=tol)
    assert m["up_fraction"] == pytest.approx(up, abs=0.005)
    assert m["neff"].real == pytest.approx(est["N0"], abs=0.006)


def test_thin_sheet_radiation_with_box_and_handles():
    # BOX interference makes α swing ~10× with the BOX thickness; the thin sheet follows it within ~6 % on Si and
    # ~20 % on gold (stronger, resonant mirror)
    for nh, box, tol in ((3.476 + 0j, 1.7e-6, 0.07), (3.476 + 0j, 2.0e-6, 0.07), (0.52 + 10.7j, 1.9e-6, 0.2),
                         (0.52 + 10.7j, 2.1e-6, 0.2)):
        g = coupler(20e-9, n_handle=nh, box_thickness=box)
        est = g.radiation(LAM)
        st, sp = rc.surface_grating_stack(g, LAM)
        m = rc.RCWA(st, LAM, 10).leaky_mode(est["N0"], sp, est["alpha_total"])
        up = sum(o["alpha"] for o in est["orders"] if o["medium"] == "cladding") / est["alpha_total"]
        assert m["alpha"] == pytest.approx(est["alpha_total"], rel=tol)
        if nh.imag < 0.05:
            assert m["up_fraction"] == pytest.approx(up, abs=0.005)
        else:
            assert m["up_fraction"] > 0.95                         # gold: downward light is absorbed, not radiated


def test_thin_sheet_TM_is_not_reliable():
    # known limit of grating_coupler (task A7): the TM radiation uses TE expressions and comes out ~10× too low
    g = coupler(20e-9, pol="TM")
    est = g.radiation(LAM)
    st, sp = rc.surface_grating_stack(g, LAM)
    m = rc.RCWA(st, LAM, 10).leaky_mode(est["N0"], sp, 9 * est["alpha_total"])
    assert m["alpha"] > 5 * est["alpha_total"]


def test_staircase_profiles_converge():
    g = coupler(50e-9)
    g = SurfaceGrating(*STACK, 50e-9, g.period, 0.5, "trap", np.radians(70))
    est = g.radiation(LAM)
    a = []
    for n in (8, 16):
        st, sp = rc.surface_grating_stack(g, LAM, staircase=n)
        a.append(rc.RCWA(st, LAM, 10).leaky_mode(est["N0"], sp, est["alpha_total"])["alpha"])
    assert a[0] == pytest.approx(a[1], rel=0.01) and a[1] == pytest.approx(est["alpha_total"], rel=0.06)


# ---------------------------------------------------------------- first-order DBR coupling
@pytest.mark.parametrize("h,tol", [(5e-9, 1e-3), (20e-9, 2e-3), (50e-9, 6e-3)])
def test_dbr_kappa_TE_matches_effective_index(h, tol):
    g = SurfaceGrating(*STACK, h, 0.36e-6)
    g = SurfaceGrating(*STACK, h, LAM / (2 * np.mean(g.neff_profile(LAM))))
    r = rc.bragg_band(g, LAM)
    assert r["fit_residual"] < 1e-3
    assert r["kappa"] == pytest.approx(r["kappa_eim"], rel=tol)
    assert r["lambda_B"] == pytest.approx(LAM, rel=1e-3)


def test_dbr_kappa_TM_is_below_effective_index():
    # the EIM treats the etched layer as isotropic; for TM the walls matter (E_x crosses them) and κ is ~2× lower
    g = SurfaceGrating(*STACK, 20e-9, 0.36e-6, polarization="TM")
    g = SurfaceGrating(*STACK, 20e-9, LAM / (2 * np.mean(g.neff_profile(LAM))), polarization="TM")
    r = rc.bragg_band(g, LAM)
    assert r["fit_residual"] < 1e-3 and 0.3 < r["kappa"] / r["kappa_eim"] < 0.6


# ---------------------------------------------------------------- coupled-mode coupler vs rigorous (plane-wave limit)
def _resonance(R, a, b, n=60):
    ls = np.linspace(a, b, n)
    v = np.array([R(l) for l in ls])
    i = int(np.argmax(v))
    lo, hi = ls[max(i - 1, 0)], ls[min(i + 1, n - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        if R(c) > R(d):
            hi = d
        else:
            lo = c
    lp = (lo + hi) / 2
    top, base = R(lp), min(v[0], v[-1])
    half = (top + base) / 2

    def edge(s):
        x0, x1 = lp, lp + s * (b - a)
        for _ in range(40):
            m = (x0 + x1) / 2
            x0, x1 = (m, x1) if R(m) > half else (x0, m)
        return (x0 + x1) / 2

    return lp, top, edge(1) - edge(-1)


@pytest.mark.parametrize("theta", [0.0, np.radians(2)])
def test_coupled_mode_coupler_against_rigorous_resonance(theta):
    h = 20e-9
    g = coupler(h, detune=1.0, theta=theta)
    c = CRIGF(SurfaceGrating(*STACK, 0.0, 0.4e-6), 0, g.period, 1, h, theta=theta)
    st, _ = rc.surface_grating_stack(g, LAM)
    lc, Rc, wc = _resonance(lambda l: c.infinite_grating(l)["R"], LAM - 1.5e-9, LAM + 0.6e-9)
    lr, Rr, wr = _resonance(lambda l: rc.RCWA(st, l, 10).diffraction(theta)["R"][10], LAM - 1.5e-9, LAM + 0.6e-9)
    assert Rc == pytest.approx(1, abs=2e-3) and Rr == pytest.approx(1, abs=2e-3)
    assert wc == pytest.approx(wr, rel=0.05)                       # linewidth = radiative coupling
    assert abs(lc - lr) < 0.2e-9                                    # position: EIM n_eff error
    assert c.infinite_grating(lc)["sum"] == pytest.approx(1, abs=1e-6)


def test_result_front_ends():
    r = rc.grating_leaky_mode(LAM, *STACK, 50e-9, 0.83e-6)
    assert r["alpha"] == pytest.approx(r["alpha_thin_sheet"], rel=0.05) and 0 < r["up_fraction"] < 1
    k = rc.dbr_coupling(LAM, *STACK, 50e-9, 0.396e-6)
    assert k["kappa"] == pytest.approx(k["kappa_eim"], rel=0.01)


# ---------------------------------------------------------------- band edges of second-order gratings (Γ point)
def second_order(h, f=0.5, profile="rect", pol="TE"):
    g = SurfaceGrating(*STACK, h, 0.8e-6, f, profile, polarization=pol)
    Lam = LAM / np.mean(g.neff_profile(LAM))
    return SurfaceGrating(*STACK, h, Lam, f, profile, polarization=pol), \
        CRIGF(SurfaceGrating(*STACK, 0.0, 0.4e-6, polarization=pol), 0, Lam, 1, h, f, gc_profile=profile)


def test_cmt_band_edge_symmetric_tooth():
    _, c = second_order(20e-9)
    m = c.band_edge_modes(LAM)
    assert abs(m["dark"]["Q"]) > 1e12                                 # N2 = 0 at f = 0.5: dark mode does not radiate
    dD = m["n_group"] * 2 * np.pi / LAM**2
    assert m["bright"]["wavelength"].imag == pytest.approx(m["alpha_rad"] / dD, rel=1e-9)


@pytest.mark.parametrize("h,f,q_tol,gap_tol", [(20e-9, 0.5, 0.03, None), (50e-9, 0.5, 0.05, None), (50e-9, 0.3, 0.12, 0.15)])
def test_band_edge_modes_against_rigorous(h, f, q_tol, gap_tol):
    g, c = second_order(h, f)
    cm = c.band_edge_modes(LAM)
    st, sp = rc.surface_grating_stack(g, LAM)
    b = rc.resonance_mode(st, 0.0, cm["bright"]["wavelength"], sp)
    d = rc.resonance_mode(st, 0.0, cm["dark"]["wavelength"], sp)
    assert b["Q"] == pytest.approx(cm["bright"]["Q"], rel=q_tol)
    assert abs(d["Q"]) > 1e8                                           # symmetry-protected bound state
    assert abs(b["wavelength"].real - cm["bright"]["wavelength"].real) < 1e-9
    if gap_tol:
        gap_r = b["wavelength"].real - d["wavelength"].real
        gap_c = cm["bright"]["wavelength"].real - cm["dark"]["wavelength"].real
        assert gap_r == pytest.approx(gap_c, rel=gap_tol)


def test_asymmetric_tooth_lets_both_band_edge_modes_radiate():
    g, c = second_order(50e-9, 0.3, "saw")
    cm = c.band_edge_modes(LAM)
    st, sp = rc.surface_grating_stack(g, LAM)
    r = [rc.resonance_mode(st, 0.0, cm[k]["wavelength"], sp) for k in ("bright", "dark")]
    assert abs(r[0]["wavelength"] - r[1]["wavelength"]) > 1e-9
    for ri, k in zip(r, ("bright", "dark")):
        assert 1e3 < ri["Q"] < 1e5
        assert abs(ri["wavelength"].real - cm[k]["wavelength"].real) < 0.4e-9
    assert r[0]["Q"] == pytest.approx(cm["bright"]["Q"], rel=0.15) and r[1]["Q"] == pytest.approx(cm["dark"]["Q"], rel=0.3)


def test_dark_mode_is_a_bound_state_in_the_continuum():
    g, c = second_order(50e-9, 0.3)
    st, sp = rc.surface_grating_stack(g, LAM)
    K = 2 * np.pi / g.period
    kxs = K * np.array([0.0, 1e-4, 2e-4, 4e-4])
    dark = rc.track_band(st, kxs, c.band_edge_modes(LAM)["dark"]["wavelength"], sp)
    qk2 = [d["Q"] * (kx / K) ** 2 for d, kx in zip(dark[1:], kxs[1:])]
    assert abs(dark[0]["Q"]) > 1e8 and np.ptp(qk2) < 0.06 * np.mean(qk2)   # Q ∝ 1/k_x² off Γ
    bright = rc.track_band(st, kxs, c.band_edge_modes(LAM)["bright"]["wavelength"], sp)
    assert bright[-1]["Q"] == pytest.approx(bright[0]["Q"], rel=0.03)


def test_band_edge_mode_is_a_pole_of_the_reflection():
    g, c = second_order(20e-9)
    st, sp = rc.surface_grating_stack(g, LAM)
    lam = rc.resonance_mode(st, 0.0, c.band_edge_modes(LAM)["bright"]["wavelength"], sp)["wavelength"]
    r = lambda l: abs(rc.RCWA(st, l, 10).diffraction(0.0)["r"][10])
    eps = 1e-6 * lam.imag
    assert r(lam + eps) > 1e4 * r(lam.real) and r(lam + eps) == pytest.approx(10 * r(lam + 10 * eps), rel=0.01)  # 1/(λ - λp)


def test_band_edge_front_end():
    r = rc.band_edge(LAM, *STACK, 50e-9, second_order(50e-9, 0.3)[0].period, 0.3)
    assert r["Q_dark"] > 1e8 and 1e3 < r["Q_bright"] < 1e4 and r["lambda_dark"] < r["lambda_bright"]
