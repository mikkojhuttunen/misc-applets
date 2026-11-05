"""Task 7: HITRAN peak absorption of 1 ppm analyte in breath (classic HAPI 1.3, offline line files of ../HITRAN):
T = 308.15 K (35 C), p = 1 atm, Voigt with air broadening, H2O 5.5 % (saturation at 35 C), CO2 4.5 %.
Lines: NH3 1.53 um (strongest within +-30 /cm of 6536 /cm), CH4 3.31 um (strongest near 3.31 um), CH4 1.64 um (Fig. 3
line, 6114.65 /cm +-1), C2H6 3.34 um Q-branch (2996.85 /cm +-1, as in the proposal). -> results/t7/lines.json"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qv import gas  # noqa: E402

out = {}
for name, key, kw in [("NH3_1.53", "NH3_1.53", {}), ("CH4_3.31", "CH4_3.31", {}), ("CH4_1.64", "CH4_1.65", {"nu_center": 6114.652, "half_width": 1.0}),
                      ("C2H6_3.34Q", "C2H6_3.34", {"nu_center": 2996.85, "half_width": 1.0}), ("C2H6_3.35max", "C2H6_3.34", {})]:
    r = gas.target_line(key, **kw)
    r.pop("strip")
    out[name] = r
    print(name, round(r["nu0_cm"], 3), "cm-1", f'{r["alpha_peak_per_ppm_cm"]:.3e} /cm/ppm', {k: f"{v:.1e}" for k, v in r["bg_at_peak_per_cm"].items()})
n = out["NH3_1.53"]["alpha_peak_per_ppm_cm"]
out["ratio_to_NH3"] = {k: v["alpha_peak_per_ppm_cm"] / n for k, v in out.items() if k != "NH3_1.53"}
print(out["ratio_to_NH3"])
Path(__file__).resolve().parent.parent.joinpath("results/t7/lines.json").write_text(json.dumps(out, indent=1))
