import numpy as np
import pytest

from engines.billiard_cell import engine as bc
from engines.bragg_grating import engine as bg
from engines.cell_mirror import engine as cm

LAM = 1.55e-6


def test_constant_mirror_reduces_to_reflection_weighted_path():
    cell = bc.SegmentedCell(5e-3, 24, tilts=np.r_[np.zeros(5), [5e-4], np.zeros(18)])
    S, C = cm.hit_angles(cell, [0.6, 0.65], 80)
    st = cm.weighted_stats(S, C, lambda s: np.full_like(s, 0.97))
    assert st["R_mean"] == pytest.approx(0.97) and st["R_eff"] == pytest.approx(0.97)
    ref = [bc.reflection_weighted_path(C[k], 0.97) for k in range(2)]
    assert st["L_eff"] == pytest.approx(np.mean([r[1] for r in ref]), rel=1e-12)
    assert st["I_end"] == pytest.approx(0.97**80, rel=1e-12)


def test_intensity_is_the_product_of_the_reflectances_met():
    dbr = bg.TrenchDBR(n_tooth=2.479, N=3, m_tooth=1, slab_pol="TE", bounce_loss=0)
    cell = bc.SegmentedCell(5e-3, 24, curvature=20.0)
    S, C = cm.hit_angles(cell, [0.4], 60)
    st = cm.weighted_stats(S, C, lambda s: dbr.R(LAM, s))
    exact = np.prod(dbr.R(LAM, S[0]))
    assert st["I_end"] == pytest.approx(exact, rel=1e-12)
    L = sum(np.prod(dbr.R(LAM, S[0][:j])) * C[0][j] for j in range(60))
    assert st["L_eff"] == pytest.approx(L, rel=1e-12)
    assert st["hist"].sum() == pytest.approx(1.0)


def test_star_orbit_sees_a_single_angle():
    cell = bc.SegmentedCell(5e-3, 24)
    chi = 90 - 180 * 9 / 24                                         # {24/9}: 22.5°
    dbr = bg.TrenchDBR(n_tooth=2.479, N=4, m_tooth=1, slab_pol="TM", bounce_loss=0)
    st = cm.cell_mirror_stats(cell, np.radians(chi), 0.0, 1, 48, dbr)
    assert st["chi_50"] == pytest.approx(chi, abs=1e-6) and st["chi_max"] == pytest.approx(chi, abs=1e-6)
    assert st["R_mean"] == pytest.approx(dbr.R(LAM, np.sin(np.radians(chi))), rel=1e-6)


def test_designing_for_the_operating_angle_pays():
    cell = bc.SegmentedCell(5e-3, 24)
    th = np.radians(22.5)
    normal = bg.TrenchDBR(n_tooth=2.479, N=4, m_tooth=1, slab_pol="TM", bounce_loss=0)
    tuned = bg.TrenchDBR(n_tooth=2.479, N=4, m_tooth=1, slab_pol="TM", bounce_loss=0, sin_design=np.sin(th))
    a = cm.cell_mirror_stats(cell, th, 0.0, 1, 48, normal)
    b = cm.cell_mirror_stats(cell, th, 0.0, 1, 48, tuned)
    assert b["R_mean"] > a["R_mean"] and b["L_eff"] > a["L_eff"]


def test_weighted_quantile_and_uniform_average():
    assert cm.weighted_quantile([3, 1, 2], [1, 1, 2], 0.5) == 2
    assert cm.weighted_quantile([3, 1, 2], [1, 1, 2], 0.2) == 1
    assert cm.uniform_average(lambda s: 1 - 0.5 * s) == pytest.approx(0.75, rel=1e-9)


def test_result_front_end():
    r = cm.cell_mirror_reflectance(n_hits=60, tilt_rms=1e-3)
    assert 0 < r["R_eff"] <= r["R_design"] + 1e-12 or r["R_eff"] <= 1
    assert r["chi_50"] <= r["chi_95"] <= r["chi_max"] and r.units["L_eff"] == "m"
    with pytest.raises(ValueError, match="sin_design"):
        cm.cell_mirror_reflectance(sin_design=0.5)


def test_r_eff_is_the_decay_of_the_total_power():
    # TE (p-pol) mirror in a perturbed cell: some rays die in the Brewster dip; R_eff follows the surviving power
    dbr = bg.TrenchDBR(n_tooth=2.8138, N=4, m_tooth=1, slab_pol="TE", bounce_loss=0)
    cell = bc.SegmentedCell(5e-3, 24, tilt_rms=3e-3, seed=1)
    S, C = cm.hit_angles(cell, cm.launch_angles(np.radians(22.5), np.radians(1), 7), 150)
    st = cm.weighted_stats(S, C, lambda s: dbr.R(LAM, s))
    assert st["R_eff"] ** 150 == pytest.approx(st["I_end"], rel=1e-9)
    geo = np.exp(np.nanmean(np.log(dbr.R(LAM, S[np.isfinite(S)]))))
    assert st["R_eff"] > geo
