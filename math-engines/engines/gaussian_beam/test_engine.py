"""Golden and analytic tests. Each expected value states its source."""
import numpy as np
import pytest

from engines.gaussian_beam import engine as gb

HENE = 632.8e-9


def test_rayleigh_range_hene_1mm():
    # z_R = pi w0^2 / lambda = pi (1e-3)^2 / 632.8e-9 = 4.9646 m (hand calculation)
    assert gb.beam_parameters(w0=1e-3, wavelength=HENE)["z_R"] == pytest.approx(4.9646, rel=1e-4)


def test_beam_at_rayleigh_range():
    # at z = z_R: w = sqrt(2) w0, R = 2 z_R, Gouy phase = pi/4 (Kogelnik & Li)
    zr = gb.rayleigh_range(1e-3, HENE)
    r = gb.beam_at(zr, 1e-3, HENE)
    assert r["w"] == pytest.approx(np.sqrt(2) * 1e-3)
    assert r["R"] == pytest.approx(2 * zr)
    assert r["gouy_phase"] == pytest.approx(np.pi / 4)


def test_waist_has_flat_wavefront():
    assert np.isinf(gb.beam_at(0.0, 1e-3, HENE)["R"])


def test_far_field_divergence():
    # far field: w(z) -> theta z with theta = lambda / (pi w0)
    r = gb.beam_at(1e4, 1e-3, HENE)
    theta = gb.beam_parameters(1e-3, HENE)["divergence"]
    assert r["w"] / 1e4 == pytest.approx(theta, rel=1e-6)


def test_index_and_m2_scaling():
    # z_R scales as n / M2
    base = gb.rayleigh_range(1e-3, HENE)
    assert gb.rayleigh_range(1e-3, HENE, n=1.5, M2=2.0) == pytest.approx(base * 1.5 / 2.0)


def test_vectorised_sweep():
    z = np.linspace(-10, 10, 101)
    w = gb.beam_at(z, 1e-3, HENE)["w"]
    assert w.shape == z.shape and np.argmin(w) == 50


def test_abcd_free_space_matches_beam_at():
    # independent route: ABCD free-space propagation must reproduce w(z), R(z)
    q = gb.abcd_propagate(gb.q_parameter(0.0, 1e-3, HENE), gb.free_space(3.0))
    b, r = gb.beam_from_q(q, HENE), gb.beam_at(3.0, 1e-3, HENE)
    assert b["w"] == pytest.approx(r["w"]) and b["R"] == pytest.approx(r["R"])
    assert b["distance_to_waist"] == pytest.approx(-3.0)


def test_thin_lens_against_self_formula():
    # Self (1983): 1/(s' ) form: s' = f + (s - f) f^2 / ((s - f)^2 + z_R^2), m = f / sqrt((s-f)^2 + z_R^2)
    w0, s, f = 1e-3, 0.5, 0.1
    zr = gb.rayleigh_range(w0, HENE)
    s_out = f + (s - f) * f ** 2 / ((s - f) ** 2 + zr ** 2)
    m = f / np.sqrt((s - f) ** 2 + zr ** 2)
    r = gb.thin_lens_focus(w0=w0, s=s, f=f, wavelength=HENE)
    assert r["waist_distance"] == pytest.approx(s_out, rel=1e-9)
    assert r["magnification"] == pytest.approx(m, rel=1e-9)


def test_tutorial_composition_example():
    # waist -> 2 m -> f = 0.5 m lens -> 0.3 m, compared with a single combined ABCD matrix
    q = gb.q_parameter(0.0, 1e-3, 633e-9)
    for M in (gb.free_space(2.0), gb.thin_lens(0.5), gb.free_space(0.3)):
        q = gb.abcd_propagate(q, M)
    Mtot = gb.free_space(0.3) @ gb.thin_lens(0.5) @ gb.free_space(2.0)
    q2 = gb.abcd_propagate(gb.q_parameter(0.0, 1e-3, 633e-9), Mtot)
    assert gb.beam_from_q(q, 633e-9)["w"] == pytest.approx(gb.beam_from_q(q2, 633e-9)["w"])


@pytest.mark.parametrize("bad", [0.0, -1e-3, np.nan])
def test_rejects_nonphysical_waist(bad):
    with pytest.raises(ValueError):
        gb.beam_at(0.0, bad, HENE)
