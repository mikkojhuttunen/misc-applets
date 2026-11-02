"""Step 4a: HITRAN absorption (HAPI, offline line files) of 1 ppm CH4 at 3.31 and 1.65 um, C2H6 at 3.34 um and NH3 at
1.53 um in a breath background (1 atm, 35 C, H2O at saturation 5.5 %, CO2 4.5 %): the strongest line near each target
(peak alpha, line-resolved strip with the H2O / CO2 background), the lines of the Fig. 3 strips for comparison, and the
CH4 3.3 / 1.65 um ratio that replaces the crude 150-230x line-strength scaling. -> results/step4_gas.json
python scripts/step4_gas.py"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import gas  # noqa: E402
from qs.metrics import RESULTS, dump, fig3_data  # noqa: E402

FIG3 = {"CH4_3.31": "CH4 3.3", "CH4_1.65": "CH4 1.6", "C2H6_3.34": "C2H6", "NH3_1.53": "NH3"}


def main():
    warnings.filterwarnings("ignore")
    f3 = {s["key"]: s for s in fig3_data()["strips"]}
    out = {"conditions": {"T_K": gas.T_BREATH, "p_atm": gas.P_ATM, "x_H2O": gas.X_H2O, "x_CO2": gas.X_CO2, "line_files": "HITRAN/hitran_data (HAPI 1.3, offline)"},
           "lines": {}, "fig3_lines": {}}
    for k in gas.TARGETS:
        out["lines"][k] = gas.target_line(k)
        s = f3[FIG3[k]]
        r = gas.target_line(k, nu_center=s["nu0"], half_width=0.3)
        r["fig3_alpha_peak_per_ppm_cm"] = float(max(s["analyte_per_ppm"]))
        out["fig3_lines"][k] = {kk: v for kk, v in r.items() if kk != "strip"}
        L = out["lines"][k]
        print(f"{k:10s} strongest: {L['nu0_cm']:.3f} /cm ({L['lambda0_um']:.4f} um) alpha {L['alpha_peak_per_ppm_cm']:.3e} /cm/ppm, "
              f"H2O {L['bg_at_peak_per_cm']['H2O']:.2e} CO2 {L['bg_at_peak_per_cm']['CO2']:.2e} /cm | Fig.3 line {r['nu0_cm']:.3f}: "
              f"{r['alpha_peak_per_ppm_cm']:.3e} (Fig.3 data {r['fig3_alpha_peak_per_ppm_cm']:.3e})")
    a = {k: v["alpha_peak_per_ppm_cm"] for k, v in out["lines"].items()}
    b = {k: v["alpha_peak_per_ppm_cm"] for k, v in out["fig3_lines"].items()}
    out["ratio_CH4_3p31_over_1p65"] = {"strongest_lines": a["CH4_3.31"] / a["CH4_1.65"], "fig3_lines": b["CH4_3.31"] / b["CH4_1.65"],
                                       "crude_scaling_replaced": [150, 230]}
    print("CH4 3.31 / 1.65 um peak ratio:", out["ratio_CH4_3p31_over_1p65"])
    dump(RESULTS / "step4_gas.json", out)


if __name__ == "__main__":
    main()
