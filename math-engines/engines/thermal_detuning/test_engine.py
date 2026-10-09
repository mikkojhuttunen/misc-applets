import math

import numpy as np
import pytest

from engines.thermal_detuning import engine as td


def test_ppln_shg_1550_period():
    # 5 % MgO:PPLN SHG of 1550 nm near room temperature: Λ ≈ 19.4-19.6 µm (commercial gratings 19.5-20.9 µm)
    assert td.qpm_thermal()["period"] == pytest.approx(19.4e-6, abs=0.3e-6)


def test_ppln_shg_1064_period():
    assert td.qpm_thermal(pump_wavelength=0.532e-6, signal_wavelength=1.064e-6)["period"] == pytest.approx(6.95e-6, abs=0.15e-6)


def test_temperature_acceptance_scale():
    # a 40 mm PPLN waveguide for 1560 nm SHG measured 1.98 K FWHM (arXiv:2607.13215) -> about 8 K for 10 mm
    r = td.qpm_thermal(length=0.01)
    assert 7.0 < r["dT_FWHM"] < 12.0
    assert td.qpm_thermal(length=0.02)["dT_FWHM"] == pytest.approx(r["dT_FWHM"] / 2, rel=1e-9)


def test_uniform_constant_mismatch_is_sinc2():
    z = np.linspace(0, 0.01, 4001)
    dk = 400.0
    x = dk * 0.01 / 2
    assert td.phase_matching_factor(z, dk) == pytest.approx((math.sin(x) / x) ** 2, rel=1e-6)


def test_half_maximum_at_half_fwhm_uniform_heating():
    r0 = td.qpm_thermal(length=0.01)
    r = td.qpm_thermal(length=0.01, dT_in=r0["dT_FWHM"] / 2, decay_length=1e6)
    assert r["eta_heated"] == pytest.approx(0.5, abs=2e-3)
    assert r["eta_retuned"] == pytest.approx(1.0, abs=1e-6)


def test_nonuniform_heating_is_only_partly_recoverable():
    r = td.qpm_thermal(length=0.02, dT_in=10.0, decay_length=0.004)
    assert r["eta_heated"] < r["eta_retuned"] < 1.0


def _kappas(r, wavelength=1.55e-6, Qi=1e6, Qc=1e6, f=0.5, R=0.5, L=2 * np.pi * 100e-6, ng=2.3, dndT=3e-5):
    w = 2 * np.pi * td.C0 / wavelength
    ki, ke = w / Qi, w / Qc
    return ki + ke, ke, w * dndT / ng * (R / L) * f * ki


def test_bistability_threshold_from_the_cubic():
    r = td.ring_thermal_bistability()
    k, ke, g = _kappas(r)
    def max_roots(P):
        return max(len(td.ring_energy_roots(d, P, k, ke, g)) for d in np.linspace(-3 * k, 0, 3001))
    assert max_roots(1.05 * r["P_threshold"]) == 3
    assert max_roots(0.95 * r["P_threshold"]) == 1


def test_on_resonance_heating_and_buildup():
    a = td.ring_thermal_bistability(power=0.01)
    b = td.ring_thermal_bistability(power=0.02)
    assert b["dT_resonance"] == pytest.approx(2 * a["dT_resonance"])
    # critically coupled: all input power is dissipated, half of it (absorbing_fraction) as heat
    assert a["P_abs_resonance"] == pytest.approx(0.005, rel=1e-9)
    # circulating power / input = 4 κ_e FSR/κ² ... = F/π for critical coupling, F = FSR/linewidth
    fsr = 1.55e-6**2 / (2.3 * 2 * np.pi * 100e-6)
    assert a["buildup"] == pytest.approx(fsr / a["linewidth"] / np.pi, rel=1e-9)
