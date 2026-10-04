import numpy as np
import pytest

from engines.materials import engine as mat


def test_fused_silica_d_line():
    # Malitson fused silica: n_d (587.6 nm) = 1.4585 (catalogue value)
    assert mat.refractive_index("sio2", 587.6e-9)["n"] == pytest.approx(1.4585, abs=2e-4)


def test_fused_silica_telecom():
    # Malitson at 1550 nm: n = 1.4440, group index about 1.4626
    r = mat.refractive_index("sio2", 1550e-9)
    assert r["n"] == pytest.approx(1.4440, abs=2e-4)
    assert r["n_group"] == pytest.approx(1.4626, abs=1e-3)


def test_normal_dispersion_in_the_visible():
    for m in mat.MATERIALS:
        if m == "si":
            continue
        n = mat.index(m, np.array([0.6e-6, 0.8e-6, 1.0e-6]))
        assert np.all(np.diff(n) < 0), m


def test_lithium_niobate_birefringence_sign():
    # LiNbO3 is negative uniaxial: n_e < n_o
    assert mat.index("ln_e", 1.55e-6) < mat.index("ln_o", 1.55e-6)


def test_mgo_lithium_niobate_telecom_values():
    # Zelmon et al. 1997, 5 % MgO:LiNbO3 at 1550 nm: n_e = 2.130, n_o = 2.208
    assert mat.index("ln_e", 1.55e-6) == pytest.approx(2.130, abs=2e-3)
    assert mat.index("ln_o", 1.55e-6) == pytest.approx(2.208, abs=2e-3)


def test_unknown_material_is_rejected():
    with pytest.raises(ValueError, match="material"):
        mat.refractive_index("unobtainium", 1.55e-6)


def test_amorphous_alumina_telecom_and_dispersion():
    # Cauchy fit to sputtered Al2O3: n(1550 nm) = 1.646 + 0.00962/1.55^2 = 1.6500 (hand check)
    assert mat.index("al2o3", 1.55e-6) == pytest.approx(1.6500, abs=1e-4)
    assert mat.index("er_al2o3", 1.55e-6) == mat.index("al2o3", 1.55e-6)


def test_lithium_tantalate_weak_positive_birefringence():
    # LiTaO3: n_e - n_o ≈ +0.004 across the near infrared
    for lam in (0.8e-6, 1.06e-6, 1.55e-6):
        d = mat.index("lt_e", lam) - mat.index("lt_o", lam)
        assert 0.003 < d < 0.005
