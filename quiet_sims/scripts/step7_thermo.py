"""Step 7: thermo-optic launch-angle dither on the membrane chip.

Steering: K input beams (fan-in) spread across the output of the port, aperture W = 30 um, each with a thermo-optic
phase shifter, form a phased array; a linear phase ramp phi(x) = k0 n_eff x dtheta across the aperture steers the launch
by dtheta inside the slab. For +-2 mrad: the edge elements need +-k0 n_eff (W/2) dtheta, the total ramp across W is
2 k0 n_eff (W/2) dtheta. (The aperture holds 2 n_eff W / lambda modes: 77 for Si, 64 for SiNx at 1.55 um.)

Heater on a free-standing strip: heat leaves sideways through the membrane (thickness t, conductivity kappa) to heat-sunk
anchors at distance l on both sides: G = 2 kappa t L_h / l, first thermal mode tau = 4 l^2 / (pi^2 D), D = kappa/(rho c_p),
f_3dB = 1/(2 pi tau). The phase needs dT = dphi lambda / (2 pi L_h dn_eff/dT), dn_eff/dT = (dn_eff/dn) dn/dT (slab mode),
so P = G dT = 2 kappa t dphi lambda / (2 pi l dn_eff/dT): independent of the heater length; P ~ 1/l, tau ~ l^2.
A heater only heats: the dither rides on a bias P_dc = P_ac (or push-pull pairs), so the mean power is the swing.
Air below the membrane adds G_air = kappa_air w_h L_h / g (g: gap to the substrate; negligible for a through-etched
release, listed for g = 2 um).

Parameters (LIT unless marked): SiNx (LPCVD) dn/dT = 2.45e-5 /K at 1.55 um (Arbabi & Goddard, Opt. Lett. 38, 3878
(2013)); the brief's 2.5e-5 agrees; the same value assumed at 3.3 um [ASSUMPTION]. Si dn/dT = 1.8e-4 /K (Komma et al.,
APL 101, 041905 (2012): 1.87e-4 at 300 K). Thin-film conductivity: SiNx 2.5 W/mK [ASSUMPTION, LPCVD films 2-3.5],
Si 250 nm 60 W/mK [ASSUMPTION, phonon boundary scattering, bulk 148]; rho c_p: SiNx 3100 kg/m3 x 700 J/kgK, Si 2330 x 700.
-> results/step7_thermo.json   python scripts/step7_thermo.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import slabmodes as SM  # noqa: E402
from qs.metrics import RESULTS, dump  # noqa: E402

W = 30e-6
DTH = 2e-3
LH = 200e-6
CASES = {  # name: (core n, thickness, lambda, dn/dT, kappa, rho, c_p)
    "SiNx 600 nm, 1.55 um": (2.0, 600e-9, 1.55e-6, 2.45e-5, 2.5, 3100, 700),
    "SiNx 1.2 um, 3.3 um": (1.97, 1.2e-6, 3.3e-6, 2.45e-5, 2.5, 3100, 700),
    "Si 250 nm, 1.55 um": (3.48, 250e-9, 1.55e-6, 1.8e-4, 60.0, 2330, 700),
}
ELLS = (5e-6, 10e-6, 20e-6, 50e-6, 100e-6)


def main():
    out = {"aperture_um": W * 1e6, "dither_mrad": DTH * 1e3, "heater_length_um": LH * 1e6, "cases": {}}
    for name, (n, t, lam, dndT, kap, rho, cp) in CASES.items():
        m = SM.mode(n, t, lam, "TM")
        ne, dneT = m["n_eff"], m["dneff_dn"] * dndT
        k0 = 2 * math.pi / lam
        phi_edge = k0 * ne * (W / 2) * DTH
        dT = phi_edge * lam / (2 * math.pi * LH * dneT)
        D = kap / (rho * cp)
        rows = []
        for l in ELLS:
            G = 2 * kap * t * LH / l
            tau = 4 * l * l / (math.pi**2 * D)
            G_air = 0.026 * 10e-6 * LH / 2e-6
            rows.append({"l_um": l * 1e6, "G_W_per_K": G, "P_edge_mW": G * dT * 1e3, "P_edge_with_2um_air_gap_mW": (G + G_air) * dT * 1e3,
                         "tau_us": tau * 1e6, "f3dB_kHz": 1 / (2 * math.pi * tau) / 1e3})
        modes = 2 * ne * W / lam
        out["cases"][name] = {"n_eff": ne, "dneff_dT_per_K": dneT, "modes_in_port": modes, "phase_edge_rad": phi_edge,
                              "phase_ramp_total_rad": 2 * phi_edge, "dT_edge_K": dT, "by_anchor_distance": rows,
                              "K_elements_power_note": "a linear ramp over K elements needs on average half the edge power per element "
                                                       "(sum ~ K/2 x P_edge), plus the same again as bias"}
        print(f"{name}: n_eff {ne:.3f}, dn_eff/dT {dneT:.2e}/K, {modes:.0f} modes in the port; edge phase +-{phi_edge:.3f} rad, "
              f"dT +-{dT:.2f} K for a {LH*1e6:.0f} um heater")
        for r in rows:
            print(f"   anchors at {r['l_um']:5.0f} um: P {r['P_edge_mW']:.3f} mW per edge heater (air gap 2 um: {r['P_edge_with_2um_air_gap_mW']:.3f}), "
                  f"tau {r['tau_us']:.1f} us, f3dB {r['f3dB_kHz']:.2f} kHz")
    out["feasibility_1s"] = ("the 1 s averaging of fig. 3b assumes an angle dither at 1 kHz (f_theta) and needs >= ~100 dither periods per "
                             "measurement: a thermal f3dB >= 1 kHz is enough, met by SiNx with anchors <= ~20 um from the heater and by Si "
                             "for any anchor distance up to ~100 um")
    dump(RESULTS / "step7_thermo.json", out)


if __name__ == "__main__":
    main()
