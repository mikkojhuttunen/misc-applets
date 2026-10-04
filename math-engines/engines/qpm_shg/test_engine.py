import numpy as np
import pytest

from engines.qpm_shg import engine as q


def test_period_hand_value():
    # Λ = λ / (2 (n_SH - n_p)) = 1.55e-6 / (2 * 0.1652) = 4.6913 µm
    r = q.phase_matching(1.55e-6, 2.0272, 2.1924)
    assert r["period"] == pytest.approx(1.55e-6 / (2 * 0.1652), rel=1e-12)
    assert r["coherence_length"] == pytest.approx(r["period"] / 2)


def test_sinc2_half_point_constant():
    assert np.sinc(q.SINC2_HALF / np.pi) ** 2 == pytest.approx(0.5, abs=1e-6)


def test_fwhm_matches_sinc2_curve():
    # independent route: the sinc^2 half-width from the response equals the closed form
    L = 4e-3
    x = np.linspace(0, 10 / L, 200001)
    resp = q.sinc2_response(x, L)
    dk_half = x[np.argmax(resp < 0.5)]
    r = q.phase_matching(1.55e-6, 2.0272, 2.1924, ng_pump=2.33, ng_sh=2.40, length=L)
    ddk = 4 * np.pi / 1.55e-6**2 * (2.40 - 2.33)
    assert r["fwhm"] == pytest.approx(2 * dk_half / ddk, rel=1e-3)


def test_rejects_zero_wavelength():
    with pytest.raises(ValueError, match="wavelength"):
        q.phase_matching(0.0, 2.0, 2.1)
