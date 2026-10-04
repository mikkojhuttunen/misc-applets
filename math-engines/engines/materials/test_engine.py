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
        rng = mat.VALID_RANGE.get(m)
        if m == "si" or (rng is not None and (rng[0] > 0.6e-6 or rng[1] < 1.0e-6)):
            continue  # Si is absorbing here; fits with a narrower published range are not tested outside it
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


# --- A2: new materials, checked against catalogue / textbook values -------------------------------

@pytest.mark.parametrize("material, wl_um, n_ref, tol", [
    ("ktp_x", 1.064, 1.7377, 1e-3), ("ktp_y", 1.064, 1.7453, 1e-3), ("ktp_z", 1.064, 1.8296, 1e-3),
    ("bbo_o", 0.532, 1.6742, 1e-3), ("bbo_e", 0.532, 1.5555, 1.5e-3),
    ("sapphire_o", 0.5893, 1.7681, 1e-3), ("sapphire_e", 0.5893, 1.7600, 1e-3),
    ("tio2_o", 0.5893, 2.613, 2e-3), ("tio2_e", 0.5893, 2.909, 2e-3),
    ("gaas", 1.55, 3.37, 1e-2), ("aln_o", 1.55, 2.12, 1e-2),
])
def test_new_materials_match_reference_indices(material, wl_um, n_ref, tol):
    assert mat.index(material, wl_um * 1e-6) == pytest.approx(n_ref, abs=tol)


def test_anisotropic_crystal_signs():
    lam = 1.55e-6
    assert mat.index("bbo_e", lam) < mat.index("bbo_o", lam)          # BBO negative uniaxial
    assert mat.index("aln_e", lam) > mat.index("aln_o", lam)          # AlN positive uniaxial
    assert mat.index("tio2_e", 1.0e-6) > mat.index("tio2_o", 1.0e-6)
    assert mat.index("ktp_x", lam) < mat.index("ktp_y", lam) < mat.index("ktp_z", lam)  # biaxial ordering n_x < n_y < n_z


def test_lithium_tantalate_refit_residual_against_bond_table():
    # four-digit points from the Bond (1965) table, independent of the fit that produced the coefficients
    table_e = {0.6: 2.1878, 1.0: 2.1432, 2.0: 2.1115, 3.0: 2.0799}
    table_o = {0.6: 2.1837, 1.0: 2.1383, 2.0: 2.1060, 3.0: 2.0733}
    for lam, n in table_e.items():
        assert mat.index("lt_e_bond", lam * 1e-6) == pytest.approx(n, abs=1.5e-3)
    assert mat.index("lt_e_bond", 1.55e-6) == pytest.approx(2.1227, abs=1.5e-3)


def test_silica_zero_dispersion_wavelength():
    # bulk fused silica has β2 = 0 near 1.27 µm and anomalous dispersion above; β2(1.55 µm) ≈ -28 fs²/mm
    lo, hi = 1.20e-6, 1.35e-6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if mat.refractive_index("sio2", mid)["gvd"] < 0 else (mid, hi)
    assert 0.5 * (lo + hi) == pytest.approx(1.273e-6, abs=5e-9)
    assert mat.refractive_index("sio2", 1.55e-6)["gvd"] == pytest.approx(-27.9e-27, rel=0.02)


def test_group_index_exceeds_phase_index_for_normal_dispersion():
    for m in ("ktp_z", "bbo_o", "sapphire_o", "lt_e_bond"):
        r = mat.refractive_index(m, 1.0e-6)
        assert r["n_group"] > r["n"]


def test_gayer_equals_room_temperature_fit_at_t0():
    # independent datasets (Gayer 2008 at T0 = 24.5 °C vs Zelmon 1997 MgO:LN) agree to a few 1e-3
    for lam in (1.0e-6, 1.55e-6, 2.0e-6):
        t = mat.thermal_index("ln_e_gayer", lam, 297.65)["n"]
        assert t == pytest.approx(mat.index("ln_e", lam), abs=3e-3)


def test_gayer_hand_value_at_t0():
    # f = 0 at T0: n^2 = a1 + a2/(λ²-a3²) + a4/(λ²-a5²) - a6 λ², evaluated by hand at 1550 nm
    l2 = 1.55**2
    n2 = 5.756 + 0.0983 / (l2 - 0.202**2) + 189.32 / (l2 - 12.52**2) - 1.32e-2 * l2
    assert mat.thermal_index("ln_e_gayer", 1.55e-6, 297.65)["n"] == pytest.approx(n2**0.5, rel=1e-12)


def test_gayer_qpm_period_tuning_with_temperature():
    from engines.phase_matching import engine as pm
    def period(T):
        lam = 1.55e-6
        n1 = mat.thermal_index("ln_e_gayer", lam, T)["n"]
        n3 = mat.thermal_index("ln_e_gayer", lam / 2, T)["n"]
        return pm.three_wave_mismatch(lam, lam, n1, n1, n3)["period_required"]
    assert period(298.15) == pytest.approx(19.4e-6, abs=0.3e-6)       # usual 1550 nm PPLN design period
    assert period(298.15) > period(323.15) > period(373.15)             # heating shortens the required period


def test_thermal_index_rejects_materials_without_a_thermal_model():
    with pytest.raises(ValueError, match="material"):
        mat.thermal_index("sio2", 1.55e-6, 300.0)


def test_out_of_range_wavelength_is_flagged_not_rejected():
    r = mat.refractive_index("tio2_o", 1.55e-6)   # Devore fit is published to 1.53 µm
    assert any("WARNING" in a for a in r.assumptions)
    assert not any("WARNING" in a for a in mat.refractive_index("tio2_o", 1.0e-6).assumptions)
