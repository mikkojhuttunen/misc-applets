"""How hard can a waveguide amplifier be pumped before heating detunes it? Worked examples chaining the engines.

    cd math-engines && python examples/pump_heating.py

1. Er:Al2O3 strip on TFLN: cross-section R' and mode-weighted dn_eff/dT, heating along the amplifier at 980 and
   1480 nm, pump power for a 5 K budget.
2. Si and AlGaAs-on-insulator χ(3) wires: TPA and free-carrier heating against the pump power.
3. TFLN χ(2) amplifier (775 nm pump): residual pump absorption heats the input; what that does to phase matching.
4. Ring resonator in SiN: thermal bistability threshold.
Numbers rest on the LUT (see engines/thermo_optic/LUT.md for the confidence of each value).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.amplifier_thermal import engine as at  # noqa: E402
from engines.thermal_detuning import engine as td  # noqa: E402
from engines.thermo_optic import engine as to  # noqa: E402
from engines.waveguide_thermal import engine as wt  # noqa: E402


def er_on_tfln():
    print("1. Er:Al2O3 strip (2 x 0.4 um) on 300 nm TFLN, 4.7 um BOX, air clad")
    geo = dict(core_material="al2o3_film", core_width=2e-6, core_height=0.4e-6, slab_material="ln_e", slab_thickness=0.3e-6,
               box_thickness=4.7e-6, clad_material="air", clad_thickness=0.0)
    m = wt.mode_weighted_heating(wavelength=1.532e-6, heat_in_mode=True, **geo)
    dyn = wt.ridge_dynamics(**geo)
    n_strip = to.index_at("al2o3_film", 1.532e-6)
    mp = wt.mode_weighted_heating(wavelength=0.98e-6, **geo)
    g_s = m["Gamma_core"] * m["neff"] / n_strip            # power share in the strip, ∫strip E² / ∫E²
    g_p = mp["Gamma_core"] * mp["neff"] / n_strip
    print(f"   R'_mode = {m['R_th_mode']:.3f} K m/W, dn_eff/dT = {m['dneff_dT']:.2e} 1/K, overlap with the strip "
          f"signal/pump = {g_s:.2f}/{g_p:.2f}, thermal f_3dB = {dyn['f_3dB'] / 1e3:.0f} kHz")
    cases = (("980 nm", dict(pump_wavelength=0.98e-6, sigma_a_pump=1.7e-25)),
             ("1480 nm", dict(pump_wavelength=1.48e-6, sigma_a_pump=2.5e-25, sigma_e_pump=0.8e-25)),
             ("980 nm, no background loss", dict(pump_wavelength=0.98e-6, sigma_a_pump=1.7e-25, loss_db_per_cm=0.0)),
             ("980 nm, 20 % quenched, no bg", dict(pump_wavelength=0.98e-6, sigma_a_pump=1.7e-25, loss_db_per_cm=0.0, quenched_fraction=0.2)))
    for name, extra in cases:
        kw = dict(R_th=m["R_th_mode"], dneff_dT=m["dneff_dT"], gamma_signal=g_s, gamma_pump=g_p, **extra)
        r = at.er_amplifier_heating(pump_power=0.5, **kw)
        lim = at.er_pump_limit(dT_max=5.0, dneff_max=1e-3, **kw)
        print(f"   {name:30s} 500 mW: gain {r['gain_dB']:.1f} dB, heat {r['heat_total'] * 1e3:.1f} mW "
              f"({r['heat_fraction']:.2f} of the pump lost), dT_max {r['dT_max']:.2f} K; 5 K at {lim['P_limit']:.2f} W" + ("" if lim["limited"] else " (not reached)"))


def chi3_wires():
    print("2. chi(3) wires at 1550 nm, 0.1 dB/cm absorbing loss")
    soi = wt.mode_weighted_heating()
    ag = wt.mode_weighted_heating(core_material="algaas", core_width=0.6e-6, core_height=0.4e-6, box_thickness=3e-6)
    for name, m, beta, tau in (("Si 500x220 (tau 1 ns)", soi, 8e-12, 1e-9), ("Si 500x220 (p-i-n, 10 ps)", soi, 8e-12, 1e-11),
                               ("Al0.2GaAs 600x400", ag, 0.0, 0.0)):
        for P in (0.05, 0.2, 0.5):
            b = to.pump_budget(R_th=m["R_th_mode"], dneff_dT=m["dneff_dT"], power=P, a_eff=0.1e-12 if "Si" in name else 0.2e-12,
                               loss_abs_db_per_cm=0.1, beta_tpa=beta, carrier_lifetime=tau, sigma_fca=1.45e-21,
                               pump_absorption_db_per_cm=0.0, eta_heat=0.0, dT_max=10, dneff_max=1e-4)
            print(f"   {name:28s} P = {P * 1e3:4.0f} mW: q' = {b['q']:7.3f} mW/mm, dT = {b['dT']:6.3f} K, "
                  f"dn_eff = {b['dneff']:.1e}; P_max(dn 1e-4) = {b['P_max_dn'] * 1e3:.0f} mW")


def tfln_opa():
    print("3. TFLN chi(2) amplifier, 775 nm pump, 1550 nm signal, 10 mm poled, 0.3 dB/cm pump absorption")
    geo = dict(core_material="ln_e", slab_material="ln_e", core_width=1.2e-6, core_height=0.3e-6, slab_thickness=0.3e-6,
               box_thickness=4.7e-6, clad_material="air", clad_thickness=0.0)
    R = wt.mode_weighted_heating(wavelength=0.775e-6, **geo)["R_th_mode"]
    alpha = float(to.db_per_cm_to_per_m(0.3))
    for P in (0.1, 1.0, 3.0):
        dT_in = R * alpha * P
        q = td.qpm_thermal(length=0.01, dT_in=dT_in, decay_length=1 / alpha, alpha_L=2.6e-6)
        print(f"   pump {P:.1f} W: dT_in = {dT_in:.3f} K (acceptance FWHM {q['dT_FWHM']:.1f} K), efficiency factor "
              f"{q['eta_heated']:.4f}, after retuning {q['eta_retuned']:.4f}")


def sin_ring():
    print("4. Si3N4 ring, R = 100 um, Q_i = Q_c = 2e6, half of the intrinsic loss absorbed")
    m = wt.mode_weighted_heating(core_material="si3n4", core_width=1.6e-6, core_height=0.8e-6, box_thickness=4e-6, clad_thickness=3e-6)
    r = td.ring_thermal_bistability(ring_length=2 * 3.14159265 * 100e-6, n_group=m["n_group"], Q_intrinsic=2e6, Q_coupling=2e6,
                                    R_th=m["R_th_mode"], dneff_dT=m["dneff_dT"], power=0.05)
    print(f"   bistability above {r['P_threshold'] * 1e3:.2f} mW; at 50 mW on resonance dT = {r['dT_resonance']:.1f} K, "
          f"shift {r['delta_lambda_max'] * 1e12:.0f} pm = {r['shift_linewidths']:.0f} linewidths")


if __name__ == "__main__":
    er_on_tfln()
    chi3_wires()
    tfln_opa()
    sin_ring()
