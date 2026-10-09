import math

import numpy as np
import pytest

from engines.amplifier_thermal import engine as at


def _trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def test_energy_conservation_along_the_amplifier():
    # power lost from the guided beams = heat + spontaneous emission radiated + scattered background loss
    kw = dict(pump_power=0.3, signal_power=1e-4, length=0.03, n_z=400)
    r = at.er_amplifier_heating(**kw)
    p = at._params(**at._DEFAULTS)
    z, Pp, Ps = r["z"], r["P_pump"], r["P_signal"]
    n2 = np.array([at._populations(a, b, p)[1] for a, b in zip(Pp, Ps)])
    rad = p["eta"] * n2 / p["tau"] * p["hvs"] * p["Ad"]
    scat = (1 - p["fabs"]) * p["al"] * (Pp + Ps)
    lost = (Pp[0] - Pp[-1]) + (Ps[0] - Ps[-1])
    assert _trap(r["q"] + rad + scat, z) == pytest.approx(lost, rel=1e-4)


def test_heat_decomposition_per_cross_section():
    # q'_Er = A_d [R_p,net (hν_p - hν_s) + (1 - η) N₂/τ hν_s + C N₂² hν_s] + quenched-ion absorption
    p = at._params(**{**at._DEFAULTS, "quenched_fraction": 0.1, "loss_db_per_cm": 0.0})
    Pp, Ps = 0.15, 2e-3
    q, n1, n2 = at._heat(Pp, Ps, p)
    Ip, Is = p["Gp"] * Pp / p["Ad"], p["Gs"] * Ps / p["Ad"]
    n1a = n1 - p["Nq"]
    Rp = (p["sap"] * n1a - p["sep"] * n2) * Ip / p["hvp"]
    active = p["Ad"] * (Rp * (p["hvp"] - p["hvs"]) + (1 - p["eta"]) * n2 / p["tau"] * p["hvs"] + p["Cup"] * n2**2 * p["hvs"])
    quenched = p["Nq"] * (p["sap"] * Ip + p["sas"] * Is) * p["Ad"]
    assert q == pytest.approx(active + quenched, rel=1e-9)


def test_low_pump_limit_is_the_quantum_defect():
    # weak pump, no upconversion, all decay radiative, no background: heat / absorbed pump = 1 - λ_p/λ_s
    r = at.er_amplifier_heating(pump_power=1e-6, signal_power=0.0, c_up=0.0, eta_rad=1.0, loss_db_per_cm=0.0)
    assert r["heat_fraction"] == pytest.approx(1 - 0.98 / 1.532, rel=2e-3)


def test_fully_quenched_film_turns_all_absorbed_pump_into_heat():
    r = at.er_amplifier_heating(pump_power=0.05, signal_power=0.0, quenched_fraction=1.0, loss_db_per_cm=0.0)
    assert r["heat_fraction"] == pytest.approx(1.0, rel=1e-3)
    # unbleached absorber: P_out = P_in exp(-Γ σ N L)
    assert r["pump_out"] == pytest.approx(0.05 * math.exp(-0.5 * 1.7e-25 * 1.5e26 * 0.03), rel=1e-6)


def test_in_band_pumping_heats_less():
    a = at.er_amplifier_heating(pump_power=0.2)
    b = at.er_amplifier_heating(pump_power=0.2, pump_wavelength=1.48e-6, sigma_a_pump=2.5e-25, sigma_e_pump=0.8e-25)
    assert b["heat_fraction"] < a["heat_fraction"]


def test_hot_spot_at_the_input_and_linear_in_R():
    r = at.er_amplifier_heating(pump_power=0.5)
    assert r["z_hot"] == 0.0
    r2 = at.er_amplifier_heating(pump_power=0.5, R_th=1.0)
    assert r2["dT_max"] == pytest.approx(2 * r["dT_max"])


def test_pump_limit_hits_the_budget():
    lim = at.er_pump_limit(dT_max=5.0, dneff_max=1.0)
    assert lim["limited"]
    assert lim["dT_max"] == pytest.approx(5.0, rel=1e-6)
    lim_n = at.er_pump_limit(dT_max=50.0, dneff_max=5e-5)
    assert abs(lim_n["dneff_max"]) == pytest.approx(5e-5, rel=1e-6) and lim_n["limited_by_dn"]


def test_pump_limit_not_reached():
    lim = at.er_pump_limit(dT_max=1e3, dneff_max=1.0, P_search=0.1)
    assert not lim["limited"] and lim["P_limit"] == 0.1
