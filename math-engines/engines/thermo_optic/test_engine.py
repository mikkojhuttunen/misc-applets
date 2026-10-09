import math

import pytest

from engines.slab_waveguide.engine import neff_three_layer
from engines.thermo_optic import engine as to


def test_lut_rows_are_complete_and_consistent():
    for m in to.MATERIALS:
        e = to.entry(m)
        lo, hi = e["dn_dT_range"]
        assert lo <= e["dn_dT"] <= hi, m
        assert 0 < e["k_range"][0] <= e["k"] <= e["k_range"][1], m
        assert e["rho"] > 0 and e["cp"] > 0 and e["Eg"] > 0, m
        assert e["confidence"] in to.CONFIDENCE, m
        assert e["source"], m
        assert 1.0 <= to.index_at(m, 1.55e-6) < 3.6, m


def test_lut_sellmeier_matches_tabulated_1550_index():
    # where a Sellmeier fit exists, the tabulated 1550 nm index is the same number to 0.01
    for m in to.MATERIALS:
        e = to.entry(m)
        if e["sellmeier"]:
            assert to.index_at(m, 1.55e-6) == pytest.approx(e["n"], abs=0.01), m


@pytest.mark.parametrize("m, dndT", [("si", 1.8e-4), ("sio2", 1.0e-5), ("si3n4", 2.45e-5), ("gaas", 2.35e-4), ("ln_e", 3.4e-5)])
def test_reference_thermo_optic_coefficients(m, dndT):
    assert to.material_properties(m)["dn_dT"] == pytest.approx(dndT, rel=0.1)


def test_negative_dn_dT_materials():
    for m in ("tio2_film", "su8", "air"):
        assert to.entry(m)["dn_dT"] < 0


def test_air_dn_dT_from_gas_law():
    n = to.entry("air")["n"]
    assert to.entry("air")["dn_dT"] == pytest.approx(-(n - 1) / 293.0, rel=0.05)


def test_algaas_endpoints_and_adachi_conductivity():
    g, a = to.algaas_properties(0.0), to.algaas_properties(1.0)
    assert g["n"] == pytest.approx(to.entry("gaas")["n"], abs=1e-9)
    assert g["k"] == pytest.approx(44.05, rel=1e-3)                # 100/2.27
    assert a["k"] == pytest.approx(100 / 1.1, rel=1e-6)            # AlAs end of the Adachi fit
    assert to.algaas_properties(0.5)["k"] < to.algaas_properties(0.2)["k"] < g["k"]  # alloy scattering minimum
    # TPA at 1550 nm (2hν = 1.600 eV) for GaAs, none above x ≈ 0.142
    assert g["beta_tpa"] > 0 and to.algaas_properties(0.18)["beta_tpa"] == 0


def test_diffusivity_si():
    r = to.material_properties("si")
    assert r["diffusivity"] == pytest.approx(148 / (2329 * 705), rel=1e-12)


def test_index_change_and_pi_length():
    r = to.index_change("si", 10.0)
    assert r["delta_n"] == pytest.approx(1.84e-3)
    assert r["length_for_pi"] == pytest.approx(1.55e-6 / (2 * 1.84e-3))
    assert r["delta_n_min"] <= r["delta_n"] <= r["delta_n_max"]
    neg = to.index_change("su8", 10.0)
    assert neg["delta_n_min"] <= neg["delta_n"] <= neg["delta_n_max"] < 0


def test_soi_slab_resonance_drift():
    # 220 nm SOI slab: nearly all of dn_eff/dT from the Si core; dλ/dT ≈ 77 pm/K
    # (strip-waveguide rings measure 70-85 pm/K)
    r = to.slab_thermal_shift()
    assert r["dneff_dT"] == pytest.approx(1.83e-4, rel=0.02)
    assert r["contrib_core"] / r["dneff_dT"] > 0.99
    assert 70e-12 < r["dlambda_dT"] < 85e-12


def test_sin_slab_resonance_drift():
    # thick LPCVD Si3N4 rings drift about 20-25 pm/K
    r = to.slab_thermal_shift(core="si3n4", thickness=0.8e-6)
    assert 17e-12 < r["dlambda_dT"] < 27e-12


def test_athermal_cladding_lowers_drift():
    # a negative-dn/dT cladding pulls dn_eff/dT down
    base = to.slab_thermal_shift(core="si3n4", cladding="sio2", thickness=0.3e-6)
    ath = to.slab_thermal_shift(core="si3n4", cladding="su8", thickness=0.3e-6)
    assert ath["contrib_clad"] < 0 and ath["dneff_dT"] < base["dneff_dT"]


def test_gamma_sum_rule_for_slab():
    # scaling all indices by s is scaling k by s: n_eff(s n, λ) = s n_eff(n, λ/s), so
    # Σ Γ_i n_i = n_eff - λ ∂n_eff/∂λ at fixed indices (the waveguide part of the group index)
    r = to.slab_thermal_shift(core="si3n4", substrate="sio2", cladding="air", thickness=0.4e-6)
    n = [to.index_at(m, 1.55e-6) for m in ("sio2", "si3n4", "air")]
    s = r["Gamma_sub"] * n[0] + r["Gamma_core"] * n[1] + r["Gamma_clad"] * n[2]
    lam, h = 1.55e-6, 1e-9
    d = (neff_three_layer(lam + h, *n, 0.4e-6) - neff_three_layer(lam - h, *n, 0.4e-6)) / (2 * h)
    assert s == pytest.approx(r["neff"] - lam * d, rel=1e-5)


def test_resonance_shift_formula():
    r = to.resonance_shift(1.55e-6, 1.8e-4, 4.2, n_eff=2.4, alpha_L=2.6e-6, delta_T=2.0)
    d = 1.55e-6 * (1.8e-4 + 2.4 * 2.6e-6) / 4.2
    assert r["dlambda_dT"] == pytest.approx(d)
    assert r["delta_lambda"] == pytest.approx(2 * d)
    assert r["dnu_dT"] == pytest.approx(-to.C0 / 1.55e-6**2 * d)


def test_db_conversion():
    assert to.db_per_cm_to_per_m(1.0) == pytest.approx(23.0259, rel=1e-5)


def test_absorbed_heat_by_hand():
    # Si wire, P = 100 mW, A_eff = 0.1 µm², β = 0.8 cm/GW, τ = 1 ns, σ = 1.45e-21 m², 0.1 dB/cm absorbing
    P, A, b, tau, s, lam = 0.1, 0.1e-12, 8e-12, 1e-9, 1.45e-21, 1.55e-6
    r = to.absorbed_heat(P, lam, A, 0.1, b, tau, s)
    hnu = 6.62607015e-34 * 299792458.0 / lam
    I = P / A
    N = tau * b * I**2 / (2 * hnu)
    assert r["q_linear"] == pytest.approx(0.1 * 100 * math.log(10) / 10 * P)
    assert r["q_tpa"] == pytest.approx(b * I**2 * A)
    assert r["carrier_density"] == pytest.approx(N)
    assert N == pytest.approx(3.12e22, rel=0.01)              # 3e16 cm^-3
    assert r["q_fca"] == pytest.approx(s * N * P)
    assert r["q_total"] == pytest.approx(r["q_linear"] + r["q_tpa"] + r["q_fca"])


def test_amplifier_heat_fraction_er():
    r980 = to.amplifier_heat_fraction(0.98e-6, 1.532e-6, 1.0)
    r1480 = to.amplifier_heat_fraction(1.48e-6, 1.532e-6, 1.0)
    assert r980["quantum_defect"] == pytest.approx(0.3603, abs=1e-4)
    assert r980["quantum_defect"] / r1480["quantum_defect"] > 10
    assert to.amplifier_heat_fraction(0.98e-6, 1.532e-6, 0.0)["eta_heat"] == 1.0


def test_strip_resistance_limits():
    soi = to.strip_thermal_resistance()
    assert 0.25 < soi["R_th"] < 0.5                    # SOI wire on 2 µm BOX: ~0.3-0.4 K per mW/mm
    thick = to.strip_thermal_resistance(box_thickness=4e-6)
    assert thick["R_th"] > soi["R_th"]
    assert to.strip_thermal_resistance(clad_material="air")["R_th"] > soi["R_th"]


def test_pump_budget_inverts():
    kw = dict(R_th=0.5, dneff_dT=1.8e-4, wavelength=1.55e-6, a_eff=0.1e-12, loss_abs_db_per_cm=0.2,
              beta_tpa=8e-12, carrier_lifetime=1e-9, sigma_fca=1.45e-21, dT_max=5.0, dneff_max=1e-3)
    r = to.pump_budget(power=0.05, **kw)
    at = to.pump_budget(power=r["P_max_dT"], **kw)
    assert at["dT"] == pytest.approx(5.0, rel=1e-9)
    at_n = to.pump_budget(power=r["P_max_dn"], **kw)
    assert at_n["dneff"] == pytest.approx(1e-3, rel=1e-9)
    assert r["P_max"] == min(r["P_max_dT"], r["P_max_dn"])


def test_pump_budget_linear_closed_form_and_lossless():
    r = to.pump_budget(R_th=0.5, dneff_dT=3.4e-5, power=0.2, a_eff=1e-12, loss_abs_db_per_cm=0.0,
                       pump_absorption_db_per_cm=1.0, eta_heat=0.36, dT_max=10.0, dneff_max=1.0)
    a = 0.36 * 100 * math.log(10) / 10
    assert r["dT"] == pytest.approx(0.5 * a * 0.2)
    assert r["P_max_dT"] == pytest.approx(10.0 / (0.5 * a), rel=1e-9)
    assert math.isinf(to.pump_budget(power=1.0, pump_absorption_db_per_cm=0.0, loss_abs_db_per_cm=0.0)["P_max"])


def test_unknown_material_rejected():
    with pytest.raises(ValueError, match="material"):
        to.material_properties("unobtainium")


def test_lut_exports_are_current():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    import make_thermo_lut
    for path, text in ((make_thermo_lut.CSV_OUT, make_thermo_lut.render_csv()), (make_thermo_lut.MD_OUT, make_thermo_lut.render_md())):
        assert path.exists() and path.read_text(encoding="utf-8") == text, \
            f"{path.name} is stale: run python tools/make_thermo_lut.py"


def test_si_free_carrier_index_soref():
    # 1e18 cm^-3 electrons and holes: Δn = -(8.8e-4 + 8.5e-18 * 1e18^0.8) = -(8.8e-4 + 1.35e-3)
    assert to.si_free_carrier_index(1e24) == pytest.approx(-(8.8e-4 + 8.5e-18 * 1e18**0.8), rel=1e-9)
    r = to.absorbed_heat(0.1, carrier_lifetime=1e-9, beta_tpa=8e-12, sigma_fca=1.45e-21)
    assert r["dn_fc_si"] < 0
