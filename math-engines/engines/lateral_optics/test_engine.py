import numpy as np
import pytest

from engines.gaussian_beam import engine as gb
from engines.grating_coupler.engine import slab_mode
from engines.lateral_optics import engine as lo
from engines.slab_waveguide.engine import neff_three_layer

LAM = 1.55e-6
STACK = (1.444, 2.138, 1.0, 0.6e-6)


def test_gauss_overlap_analytic():
    k, w1, w2 = 2 * np.pi / 0.8e-6, 3e-6, 5e-6
    q = lambda w: 1j * np.pi * w * w / 0.8e-6
    assert lo.gauss_overlap(q(w1), q(w1), k) == pytest.approx(1, abs=1e-14)
    assert lo.gauss_overlap(q(w1), q(w2), k) == pytest.approx(2 * w1 * w2 / (w1**2 + w2**2), rel=1e-12)
    R = 200e-6                                                      # same width, curvature mismatch
    q2 = 1 / (1 / R + 1 / q(w1))
    assert lo.gauss_overlap(q(w1), q2, k) == pytest.approx(1 / np.sqrt(1 + (k * w1**2 / (4 * R)) ** 2), rel=1e-12)


@pytest.mark.parametrize("d,Rc", [(30e-6, 100e-6), (50e-6, 60e-6), (10e-6, 1e-3)])
def test_resonator_eigenmode_bounces_without_loss(d, Rc):
    lam = 0.8e-6
    w0 = np.sqrt(lam / np.pi * np.sqrt(d * (Rc - d)))                # z_R² = d (R_c - d): wavefront matches the mirror
    assert lo.bounce(lam, w0, d, Rc)["eta"] == pytest.approx(1, abs=1e-12)
    b = lo.best_bounce(lam, d, Rc)
    assert b["w"] == pytest.approx(w0, rel=1e-4) and b["eta"] == pytest.approx(1, abs=1e-8)


def test_bounce_against_abcd_and_beam_radius():
    lam, w, d, Rc = 0.8e-6, 2.5e-6, 40e-6, 90e-6
    k = 2 * np.pi / lam
    q0 = 1j * np.pi * w * w / lam
    M = np.array(gb.free_space(d)) @ np.array(gb.thin_lens(Rc / 2)) @ np.array(gb.free_space(d))
    qb = gb.abcd_propagate(q0, M)
    b = lo.bounce(lam, w, d, Rc)
    assert b["overlap"] == pytest.approx(lo.gauss_overlap(q0, qb, k), rel=1e-12)
    assert b["w_mirror"] == pytest.approx(gb.beam_at(d, w, lam)["w"], rel=1e-12)
    a = 3e-6
    assert lo.bounce(lam, w, d, Rc, a)["clip"] == pytest.approx(__import__("math").erf(np.sqrt(2) * a / b["w_mirror"]), rel=1e-14)


def test_ridge_limits_and_confinement():
    wide = lo.ridge_reference(LAM, *STACK, 40e-6, 0.3e-6, "TE")
    assert wide["N_t"] == pytest.approx(wide["v_t"], abs=2e-4) and wide["gamma"] > 0.999
    assert lo.side_index(LAM, 1.444, 2.138, 1.0, 0.0) == 1.444
    assert lo.ridge_neff(LAM, *STACK, 2e-6, 0.3e-6, 0.6e-6) == pytest.approx(lo.ridge_reference(LAM, *STACK, 2e-6, 0.3e-6)["N_t"], abs=1e-14)
    for pol in ("TE", "TM"):
        r = lo.ridge_reference(LAM, *STACK, 1.2e-6, 0.4e-6, pol)
        m = slab_mode(LAM, r["n_side"], r["v_t"], r["n_side"], 1.2e-6, lo._flip(pol), 0)
        y = np.linspace(-6e-6, 7.2e-6, 400001)
        f2 = m["field"](y) ** 2
        inside = (y >= 0) & (y <= 1.2e-6)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        assert r["gamma"] == pytest.approx(trapz(f2 * inside, y) / trapz(f2, y), rel=1e-4)     # Γ = ∫_ridge φ² / ∫ φ²
    assert lo.ridge_reference(LAM, *STACK, 2e-6, 0.6e-6) is None    # sides as thick as the ridge: no lateral step


def test_ridge_beam_overlap_converges():
    r = lo.ridge_reference(LAM, *STACK, 2e-6, 0.3e-6)
    a = lo.ridge_beam_overlap(LAM, r["n_side"], r["v_t"], 2e-6, 2e-6)
    b = lo.ridge_beam_overlap(LAM, r["n_side"], r["v_t"], 2e-6, 2e-6, samples=40000)
    assert a["Oy2"] == pytest.approx(b["Oy2"], rel=1e-4) and a["w_eff"] == pytest.approx(b["w_eff"], rel=1e-4)
    assert 0 < b["Oy2"] < 1


def test_front_ends():
    r = lo.ridge_effective_index(LAM, *STACK, 2e-6, 0.3e-6, 0.1e-6)
    assert r["n_side"] < r["n_ridge"] < r["n_slab"] and r["n_etched"] < r["n_ridge"]
    c = lo.curved_dbr_bounce(LAM, 1.9, 50e-6, 120e-6, 20e-6)
    assert 0.9 < c["eta"] <= 1 and c["g"] == pytest.approx(1 - 100 / 120)
    assert neff_three_layer(LAM, 1.444, 2.138, 1.0, 0.6e-6) == pytest.approx(r["n_slab"])
