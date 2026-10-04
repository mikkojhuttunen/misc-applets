import numpy as np
import pytest

from engines.gaussian_beam import engine as gb


def test_rayleigh_range_hene():
    # z_R = pi w0^2 / lambda = pi (1e-3)^2 / 633e-9 = 4.963 m (hand calculation)
    assert gb.beam_parameters(w0=1e-3, wavelength=633e-9)["z_R"] == pytest.approx(4.9630, rel=1e-4)


def test_radius_at_rayleigh_range_is_sqrt2():
    # analytic: w(z_R) = sqrt(2) w0 and R(z_R) = 2 z_R, Gouy phase pi/4
    zr = gb.beam_parameters(w0=1e-3, wavelength=633e-9)["z_R"]
    r = gb.beam_at(z=zr, w0=1e-3, wavelength=633e-9)
    assert r["w"] == pytest.approx(np.sqrt(2) * 1e-3)
    assert r["R"] == pytest.approx(2 * zr)
    assert r["gouy_phase"] == pytest.approx(np.pi / 4)


def test_curvature_infinite_at_waist():
    assert np.isinf(gb.beam_at(z=0.0, w0=1e-3, wavelength=633e-9)["R"])


def test_abcd_free_space_matches_beam_at():
    # q after free space d must give the same w as the closed form at z = d
    q = gb.abcd_propagate(gb.q_parameter(0.0, 1e-3, 633e-9), gb.free_space(2.0))
    assert gb.beam_from_q(q, 633e-9)["w"] == pytest.approx(gb.beam_at(z=2.0, w0=1e-3, wavelength=633e-9)["w"])


def test_lens_at_waist_focuses_at_f_for_large_zr():
    # analytic limit: input waist at the lens with z_R >> f focuses near f, w0' ~ lambda f / (pi w0)
    r = gb.thin_lens_focus(w0=5e-3, s=1e-9, f=0.1, wavelength=1064e-9)
    assert r["z_out"] == pytest.approx(0.1, rel=1e-3)
    assert r["w0_out"] == pytest.approx(1064e-9 * 0.1 / (np.pi * 5e-3), rel=1e-3)


def test_vectorised_positions():
    r = gb.beam_at(z=np.array([0.0, 1.0, 2.0]), w0=1e-3, wavelength=633e-9)
    assert r["w"].shape == (3,)


def test_rejects_non_physical_inputs():
    with pytest.raises(ValueError, match="w0"):
        gb.beam_parameters(w0=-1e-3, wavelength=633e-9)
