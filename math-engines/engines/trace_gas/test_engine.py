import json

import numpy as np
import pytest

pytest.importorskip("scipy")
from engines.trace_gas import engine as tg  # noqa: E402

AMU = 1.66053906660e-27


def test_voigt_area_and_limits():
    x = np.linspace(-200, 200, 400001)
    trap = getattr(np, "trapezoid", None) or np.trapz
    assert trap(tg.voigt(x, 0.05, 0.02), x) == pytest.approx(1.0, rel=2e-4)       # Lorentz wings beyond ±200 miss ~1.6e-4
    assert tg.voigt(0.0, 0.05, 1e-7) == pytest.approx(1 / (np.pi * 0.05), rel=1e-4)
    assert tg.voigt(0.0, 1e-9, 0.02) == pytest.approx(np.sqrt(np.log(2) / np.pi) / 0.02, rel=1e-6)


def test_doppler_width_from_first_principles():
    nu0, T, m = 6046.95, 296.0, 16.04
    expect = nu0 * np.sqrt(2 * np.log(2) * tg.KB * T / (m * AMU * 299792458.0**2))
    assert tg.doppler_hwhm(nu0, T, m) == pytest.approx(expect, rel=1e-4)


def test_number_density_and_reference_strength():
    assert tg.number_density_cm3() == pytest.approx(2.4794e19, rel=1e-4)
    assert tg.line_strength(1e-21, 6000.0, 300.0, tg.T_REF, 1.5) == pytest.approx(1e-21, rel=1e-12)
    assert tg.line_strength(1e-21, 6000.0, 1000.0, 350.0, 1.5) > tg.line_strength(1e-21, 6000.0, 1000.0, 296.0, 1.5)


def test_absorption_scales_with_mole_fraction_and_matches_single_line():
    lines, _ = tg.load_lines(path="/nonexistent")
    a1, nu0 = tg.peak_alpha_per_ppm("NH3", lines=lines)
    a2, _ = tg.alpha_mixture(np.array([nu0]), {"NH3": 2e-6}, lines)
    assert a2[0] > 2 * a1                     # neighbours add a little
    r = tg.line_peak("NH3", 1e-6, path_length=0.5)
    assert r["alpha_peak"] == pytest.approx(100 * a1) and r["absorbance"] == pytest.approx(50 * a1)
    assert r["wavelength"] == pytest.approx(1e-2 / nu0)


def test_through_cell_limits():
    L = np.array([0.1, 0.5, 2.0])
    W = np.array([1.0, 2.0, 1.0])
    weak = tg.through_cell(np.array([1e-7]), L, W, 0.3)[0]
    assert 1 - weak == pytest.approx(0.3 * 1e-7 * 100 * (W @ L) / W.sum(), rel=1e-5)
    one = tg.through_cell(np.array([0.01, 0.1]), np.array([1.0]), np.array([1.0]), 0.5)
    assert np.allclose(one, np.exp(-0.5 * np.array([1.0, 10.0])))


def test_shot_noise_hand_value():
    r = tg.shot_noise(1e-3, 0.05, 1.0, 1.0)
    assert r["sigma_A"] == pytest.approx(np.sqrt(2 * 1.602176634e-19 / 5e-5)) and r["P_det"] == pytest.approx(5e-5)


def test_load_lines_from_hitran_json(tmp_path, monkeypatch):
    p = tmp_path / "lines.json"
    p.write_text(json.dumps({"CH4": {"nu": [6000.0], "sw": [1e-21], "gamma_air": [0.06], "n_air": [0.7], "elower": [100.0]}}))
    lines, src = tg.load_lines(p)
    assert "HITRAN" in src and lines["CH4"][0][0] == 6000.0
    monkeypatch.setenv("TRACE_GAS_LINES", str(p))
    assert tg.load_lines()[1].startswith("HITRAN")
