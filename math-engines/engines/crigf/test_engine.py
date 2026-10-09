import numpy as np
import pytest

from engines.crigf import engine as cr
from engines.grating_coupler.engine import SurfaceGrating

LAM = 1.55e-6
STACK = (1.444, 2.138, 1.0, 0.6e-6)
SP_RES = 0.099e-6          # phase spacer that puts a cavity resonance of build() at 1550.1 nm


def build(theta=0.0, ND=60, NG=30, hG=0.05e-6, w0=12e-6, hand=None, box=0.0, aP=0.0, eta=1.0, steps=None, LamG=None, Ls=0.0,
          sp=0.3e-6):
    d = SurfaceGrating(*STACK, 0.1e-6, 0.4e-6, n_handle=hand, box_thickness=box)
    d = SurfaceGrating(*STACK, 0.1e-6, LAM / (2 * np.mean(d.neff_profile(LAM))), n_handle=hand, box_thickness=box)
    N0G = np.mean(SurfaceGrating(*STACK, hG, 0.8e-6).neff_profile(LAM))
    LamG = LamG or LAM / (N0G - np.sin(theta))
    return cr.CRIGF(d, ND, LamG, NG, hG, 0.5, sp, Ls, theta, w0, aP, 0.0, 1.0, eta, steps)


def test_layer_matrix_and_r_t_match_bragg_stack():
    from engines.bragg_grating.engine import stack_reflectance
    k = 2 * np.pi / LAM
    M = np.linalg.matrix_power(cr.layer_matrix(2.0, k, 0.19e-6) @ cr.layer_matrix(1.5, k, 0.26e-6), 7)
    r, t, R, T = cr.r_t(M, 1.5, 1.5)
    want = stack_reflectance(LAM, 2.0, 1.5, 0.19e-6, 0.26e-6, 7)
    assert R == pytest.approx(want["R"], rel=1e-12) and T == pytest.approx(want["T"], rel=1e-12)


def test_beam_normalised_to_unit_power():
    c = build()
    co = c.response(LAM)["coef"]
    x = np.linspace(-6 * c.w0, 6 * c.w0, 20001)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    P = c.dbr.n_clad / (2 * cr.ETA0) * trapz(co["E0"] ** 2 * np.exp(-2 * (x / c.w0) ** 2), x)
    assert P == pytest.approx(1.0, rel=1e-9)


def test_no_etch_gives_the_bare_slab():
    r = build(hG=0.0).response(LAM)
    assert r["R"] == r["Rd"] and r["escL"] == 0 and r["escR"] == 0
    assert r["R"] + r["T"] == pytest.approx(1.0, abs=1e-12)


def test_coupler_radiation_matches_grating_coupler_engine():
    for hand, box in ((None, 0.0), (3.476 + 0j, 2e-6), (0.52 + 10.7j, 1.8e-6)):
        for th in (0.0, np.radians(5)):
            c = build(theta=th, hand=hand, box=box)
            g = SurfaceGrating(*STACK, c.gc_etch_depth, c.gc_period, n_handle=hand, box_thickness=box)
            a1 = sum(o["alpha"] for o in g.radiation(LAM)["orders"] if o["q"] == 1)
            assert c.response(LAM)["alpha_rad"] == pytest.approx(a1, rel=1e-8)      # μ0 ε0 c² = 1 - 1.1e-10 (SI 2019)


def test_mirror_symmetry():
    r = build().response(1.5502e-6)
    assert r["escL"] == pytest.approx(r["escR"], rel=1e-10)
    c1 = build(theta=np.radians(4))
    c2 = build(theta=-np.radians(4), LamG=c1.gc_period)
    for lam in (1.549e-6, 1.5502e-6):
        a, b = c1.response(lam), c2.response(lam)
        assert a["escL"] == pytest.approx(b["escR"], rel=1e-8) and a["escR"] == pytest.approx(b["escL"], rel=1e-8)
        assert a["R"] == pytest.approx(b["R"], rel=1e-10) and a["T"] == pytest.approx(b["T"], rel=1e-10)
        assert abs(a["escL"] - a["escR"]) > 0.1 * a["escL"]                   # oblique beam: asymmetric feed


def test_passive_and_lossless_power_budget():
    lams = np.linspace(1.545e-6, 1.555e-6, 61)
    for c, bound in ((build(ND=0, w0=5e-6), 1e-3), (build(ND=0, w0=12e-6), 2e-4), (build(), 1e-3),
                     (build(sp=SP_RES), 0.05), (build(hand=3.476 + 0j, box=2e-6, sp=SP_RES), 0.05)):
        o = np.array([c.response(l)["other"] for l in lams])
        # coupler alone: beam and radiated field nearly match; on a cavity resonance the radiated profile follows the
        # cavity decay, not the Gaussian, so a few percent go into other modes
        assert o.min() > -1e-9 and o.max() < bound
    # oblique beam: the backward guided wave radiates at -θ, outside the beam mode, so only passivity holds
    c = build(theta=np.radians(3), hand=3.476 + 0j, box=2e-6)
    o = np.array([c.response(l)["other"] for l in lams])
    assert o.min() > -1e-9 and o.max() < 1


def test_losses_enter_the_budget():
    r0 = build(sp=SP_RES).response(LAM)
    r = build(sp=SP_RES, aP=50.0, eta=0.97).response(LAM)
    assert r["lat_loss"] > 1e-3 and r["other"] > -1e-9
    assert r["U_max"] < 0.9 * r0["U_max"] and r["escL"] < r0["escL"]
    assert r["R"] + r["T"] + r["escL"] + r["escR"] + r["lat_loss"] + r["other"] == pytest.approx(1.0, abs=1e-12)


def test_rk4_against_matrix_exponential_in_the_rotating_frame():
    expm = pytest.importorskip("scipy.linalg").expm
    c = build(theta=np.radians(2), NG=40, hG=0.08e-6)
    r = c.response(1.5501e-6)
    co, L = r["coef"], c.gc_length
    rad, G1, N2, k, D, aP = co["rad"], co["G1"], co["N2"], co["k"], co["D"], co["alpha_prop"]
    g2 = abs(G1) ** 2
    # R = ρ e^{iDζ}, S = σ e^{-iDζ}: constant coefficients
    A = np.array([[-rad * g2 - aP / 2 - 1j * D, -rad * G1 * G1 + 1j * k * N2],
                  [rad * np.conj(G1) ** 2 - 1j * k * np.conj(N2), rad * g2 + aP / 2 + 1j * D]])
    rot = np.diag([np.exp(1j * D * L), np.exp(-1j * D * L)])
    hom = rot @ expm(A * L)
    assert np.allclose(r["Y_end"][0], hom[:, 0], rtol=0, atol=1e-6 * np.abs(hom).max())
    assert np.allclose(r["Y_end"][1], hom[:, 1], rtol=0, atol=1e-6 * np.abs(hom).max())

    def b(z):
        E = co["F_top"] * np.exp(1j * (co["kz_in"] * z + co["phase0"])) * co["overlap_y"] * co["E0"] \
            * np.exp(-(((z - co["zc"]) * co["cos_theta"]) / co["w0"]) ** 2)
        return np.array([1j * co["aC"] * G1 * E, -1j * co["aC"] * np.conj(G1) * E])

    x, w = np.polynomial.legendre.leggauss(400)
    zs, ws = (x + 1) * L / 2, w * L / 2
    part = sum(wi * expm(A * (L - zi)) @ b(zi) for zi, wi in zip(zs, ws))
    assert np.allclose(r["Y_end"][2], rot @ part, rtol=0, atol=1e-6 * np.abs(part).max())


def test_rk4_steps_converged():
    a, b = build().response(1.5502e-6), build(steps=400).response(1.5502e-6)
    for key in ("R", "T", "escL", "U_max"):
        assert a[key] == pytest.approx(b[key], rel=1e-5, abs=1e-9)


def test_resonance_is_lorentzian():
    c = build(sp=SP_RES)
    res = c.find_resonance(LAM, 4e-9)
    lam, U0, w = res["wavelength"], res["U_max"], res["fwhm"]
    assert abs(lam - LAM) < 1e-9 and 0.5e-9 < w < 3e-9 and U0 > 1
    for x in (-1.0, -0.5, 0.5, 1.0):                                            # U0 / (1 + (2Δ/w)²)
        assert c.response(lam + x * w)["U_max"] == pytest.approx(U0 / (1 + 4 * x * x), rel=0.03)
    r = c.response(lam)
    assert r["escL"] == pytest.approx(r["escR"], rel=1e-9)


def test_strong_dbrs_return_the_light():
    r = build(ND=800).response(1.5502e-6)
    assert r["escL"] + r["escR"] < 1e-8
    assert abs(r["rR"]) ** 2 > 1 - 1e-8


def test_result_front_end():
    d = SurfaceGrating(*STACK, 0.1e-6, 0.4e-6)
    LamD = LAM / (2 * np.mean(d.neff_profile(LAM)))
    out = cr.crigf_response(LAM, *STACK, 0.1e-6, LamD, 60, 0.05e-6, 0.79637e-6, 30, spacer=0.3e-6, w0=12e-6)
    s = out["R"] + out["T"] + out["escL"] + out["escR"] + out["other"]
    assert s == pytest.approx(1.0, abs=1e-12) and 0 <= out["R_dbr"] <= 1
