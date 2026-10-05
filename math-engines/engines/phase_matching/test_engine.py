import numpy as np
import pytest

from engines.phase_matching import engine as pm
from engines.qpm_shg import engine as q


def test_degenerate_case_matches_qpm_shg():
    # independent route: the older SHG engine uses 4π (n_SH - n_p) / λ directly
    r = pm.three_wave_mismatch(1.55e-6, 1.55e-6, 2.0272, 2.0272, 2.1924)
    s = q.phase_matching(1.55e-6, 2.0272, 2.1924)
    assert r["wavelength_3"] == pytest.approx(0.775e-6, rel=1e-14)
    assert r["delta_k_bare"] == pytest.approx(s["delta_k"], rel=1e-12)
    assert r["period_required"] == pytest.approx(s["period"], rel=1e-12)


def test_period_cancels_mismatch_for_every_order():
    for m in (1, 2, 3, 5):
        r = pm.three_wave_mismatch(1.55e-6, 1.55e-6, 2.0, 2.0, 2.2, order=m)
        r2 = pm.three_wave_mismatch(1.55e-6, 1.55e-6, 2.0, 2.0, 2.2, period=r["period_required"], order=m)
        assert r["period_required"] == pytest.approx(m * 2 * np.pi / r["delta_k_bare"])
        assert abs(r2["delta_k"]) < 1e-6 * abs(r["delta_k_bare"])


def test_nondegenerate_hand_value():
    # λ1 = 1.0 µm, λ2 = 2.0 µm -> λ3 = 2/3 µm; Δk = 2π (3 n3/2 - n1 - n2/2) / µm, all n equal -> 0
    r = pm.three_wave_mismatch(1.0e-6, 2.0e-6, 2.0, 2.0, 2.0)
    assert r["wavelength_3"] == pytest.approx(2.0e-6 / 3, rel=1e-14)
    assert r["delta_k_bare"] == pytest.approx(0.0, abs=1e-6)


def test_energy_conservation_permutation_symmetry():
    a = pm.three_wave_mismatch(1.0e-6, 1.5e-6, 2.1, 2.05, 2.2)
    b = pm.three_wave_mismatch(1.5e-6, 1.0e-6, 2.05, 2.1, 2.2)
    assert a["delta_k_bare"] == pytest.approx(b["delta_k_bare"], rel=1e-14)


def test_grating_strength_values():
    assert pm.grating_strength(1, 0.5) == pytest.approx(2 / np.pi)
    assert pm.grating_strength(2, 0.5) == pytest.approx(0.0, abs=1e-15)   # even orders vanish at 50 %
    assert pm.grating_strength(3, 0.5) == pytest.approx(2 / (3 * np.pi))
    assert pm.grating_strength(2, 0.25) == pytest.approx(1 / np.pi)       # |2 sin(π/2)/(2π)|


def test_efficiency_peaks_at_matched_period_and_obeys_sinc2():
    r = pm.three_wave_mismatch(1.55e-6, 1.55e-6, 2.0272, 2.0272, 2.1924)
    L = 3e-3
    peak = pm.qpm_efficiency(1.55e-6, 1.55e-6, 2.0272, 2.0272, 2.1924, r["period_required"], L)
    assert peak["efficiency_rel"] == pytest.approx((2 / np.pi) ** 2, rel=1e-9)
    off = pm.qpm_efficiency(1.55e-6, 1.55e-6, 2.0272, 2.0272, 2.1924, r["period_required"] * 1.001, L)
    assert off["efficiency_rel"] < peak["efficiency_rel"]
    # first null: Δk L / 2 = π
    dk_null = 2 * np.pi / L
    assert pm.acceptance_response(dk_null, L) == pytest.approx(0.0, abs=1e-20)


def test_validation():
    with pytest.raises(ValueError, match="wavelength_1"):
        pm.three_wave_mismatch(0.0, 1e-6, 2, 2, 2)
    with pytest.raises(ValueError, match="order"):
        pm.three_wave_mismatch(1e-6, 1e-6, 2, 2, 2, order=0)
    with pytest.raises(ValueError, match="duty_cycle"):
        pm.grating_strength(1, 1.0)
