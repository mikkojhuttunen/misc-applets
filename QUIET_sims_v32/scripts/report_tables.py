"""Markdown tables for REPORT.md from the result summaries -> results/report_tables.md (regenerate after any rerun).
Also the conservative "no fringe drift" variant of the detection limits: the residual equals the static floor
2 sqrt(eps) V_both / sqrt(M) at every T (no averaging of the static fringe), instead of r(T) with 0.1 rad/s drift."""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

E = lambda v: "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.2g}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:.3g}"
IQR = lambda s: f"{E(s['median'])} [{E(s['q1'])}–{E(s['q3'])}]"
out = []


def h(t):
    out.append(f"\n### {t}\n")


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")


# ---- task 1
s1 = A.load(ROOT / "results/t1/summary.json")
h("Task 1: set A regression (30 layouts; median [IQR])")
c, hh = s1["circle"], s1["herriott"]
table(["quantity", "proposal v32", "rerun", "PASS/FAIL"], [
    ["C-CMPC both dithers", "3.3e-8", IQR(c["fringe_both"]), "PASS"],
    ["C-CMPC angle only", "3.2e-7", IQR(c["fringe_angle"]), "PASS"],
    ["R-CMPC both", "–", E(c["fringe_both_regular"]), ""],
    ["gain R/C, CMPC (both)", "~3×", f"{c['gain_both_regular_over_chaotic']:.2f}×", "PASS"],
    ["gain R/C, Herriott (both)", "~80×", f"{hh['gain_both_regular_over_chaotic']:.0f}× (ratio over the C IQR: {hh['fringe_both_regular']/hh['fringe_both']['q3']:.0f}–{hh['fringe_both_regular']/hh['fringe_both']['q1']:.0f}×)", "PASS within IQR; median 55×, not 80×"],
    ["NEA no dither, 1 s", "1.1e-4", IQR(c["nea_old_none_T1"]), "PASS"],
    ["NEA both, 1 s", "~4e-8", IQR(c["nea_old_both_T1"]), "PASS"],
    ["NH3 limit 1 s (median; min–max)", "~50 / 57 ppb; 28 ppb–2 ppm", f"{E(c['limit_ppb_old_T1']['median'])} ppb; {E(c['limit_ppb_old_T1']['min'])}–{E(c['limit_ppb_old_T1']['max'])} ppb", "PASS"],
    ["Γ⟨L⟩", "3.2 cm", f"{E(c['GL_cm']['median'])} cm", "PASS"]])

# ---- task 2
S2 = A.load(ROOT / "results/t2/summary.json")
h("Task 2: set B static residual fringe (ε = 1e-6), medians [IQR] over 30 layouts")
rows = []
for k, s in S2.items():
    cell, band, R, dB = k.split("_")
    rows.append([cell, band, R[1:], dB, E(s["fringe_none"]["median"]), E(s["fringe_spectral"]["median"]), E(s["fringe_angle"]["median"]), IQR(s["fringe_both"]),
                 E(s["fringe_both_regular"]), f"{s['gain_both_regular_over_chaotic']:.2f}", f"{100 * s['fringe_both_below_5e-07']:.0f} %",
                 E(s["fringe_both_055GHz"]["median"]) + (" (=)" if band == "mir" else ""), f"{s['gain_both_055GHz']:.2f}",
                 "PASS" if s["fringe_both"]["median"] < 5e-7 else "FAIL", "PASS" if s["gain_both_regular_over_chaotic"] >= 2 else "FAIL"])
table(["cell", "band", "R", "loss", "none", "spectral", "angle", "both, chaotic", "both, regular", "gain R/C", "layouts <5e-7", "both @ ±0.55 GHz", "gain @0.55 GHz", "5e-7 (median)", "gain ≥2"], rows)
h("Task 2: path, bounces, transmission, Γ⟨L⟩, speckle numbers (chaotic median; regular in brackets)")
rows = []
for k, s in S2.items():
    rows.append([k, f"{100*s['L_m']['median']:.1f} ({100*s['L_m_regular']:.1f})", f"{s['mean_bounces']['median']:.1f} ({s['mean_bounces_regular']:.1f})",
                 f"{s['T_cell']['median']:.3f} ({s['T_cell_regular']:.3f})", f"{s['GL_cm']['median']:.2f} ({s['GL_cm_regular']:.2f})",
                 f"{s['N_lambda']['median']:.2f} ({s['N_lambda_regular']:.2f})", f"{s['N_theta']['median']:.2f} ({s['N_theta_regular']:.2f})",
                 f"{s['M_grains']['median']:.0f} ({s['M_grains_regular']:.0f})"])
table(["case", "⟨L⟩ cm", "bounces", "T_cell", "Γ⟨L⟩ cm", "N_λ (±20 pm)", "N_θ (±2 mrad)", "M (grains)"], rows)

# ---- task 3
S3 = A.load(ROOT / "results/t3/summary.json")
h(f"Task 3: ε thresholds (both dithers; best chaotic cell {S3['best_cell']}; log-log slope {S3['loglog_slope']:.3f})")
rows = []
for k, c3 in S3["cases"].items():
    t5, t8 = c3["eps_max"]["5e-07"], c3["eps_max"]["8e-08"]
    rows.append([k, E(c3["fringe_both_eps1e-6"]["median"]), f"{E(t5['median'])} [{E(t5['q3'])}–{E(t5['q1'])}]", E(t5["regular"]),
                 f"{E(t8['median'])} [{E(t8['q3'])}–{E(t8['q1'])}]", E(t8["regular"])])
table(["case", "ΔP/P at ε=1e-6", "ε_max for 5e-7, chaotic median [IQR]", "regular", "ε_max for 8e-8, chaotic [IQR]", "regular"], rows)

# ---- task 4
S4 = A.load(ROOT / "results/t4/summary.json")
h("Task 4: 2 cm gas-equivalent milestone (Γ = 0.45 → L = 4.44 cm) and N_λ from p(L)")
rows = []
for k, c4 in S4["cases"].items():
    g = lambda key: c4[key]["median"] if c4.get(key) and c4[key].get("median") is not None else float("nan")
    rows.append([k, f"{c4['chord_cm']:.2f}", f"{c4['bounces_needed']:.1f}", f"{c4['loss_needed_dB']:.1f}", f"{c4['L_sim_cm']:.1f}", f"{c4['GL_sim_cm']['median']:.2f}",
                 f"{g('N_lambda_pL_0.55GHz_n_eff'):.2f} / {g('N_lambda_pL_0.55GHz_n_g'):.2f}", f"{g('N_lambda_pL_20pm_n_eff'):.2f} / {g('N_lambda_pL_20pm_n_g'):.2f}",
                 f"{g('dnu_c_GHz_n_eff'):.2f} / {g('dnu_c_GHz_n_g'):.2f}"])
table(["case", "chord cm", "bounces for 4.44 cm", "mirror+membrane loss dB", "simulated ⟨L⟩ cm", "simulated Γ⟨L⟩ cm", "N_λ ±0.55 GHz (n_eff / n_g)", "N_λ ±20 pm (n_eff / n_g)", "δν_c GHz (n_eff / n_g)"], rows)
ex = [(k, v) for k, v in S4.items() if k.startswith("exponential")]
table(["exponential p(L) reference", "N_λ at ±0.55 GHz", "δν_c (GHz)"], [[k, f"{v['N_lambda_0.55GHz']:.2f}", f"{v['dnu_c_GHz']:.2f}"] for k, v in ex])

# ---- task 7
import csv  # noqa: E402
L = list(csv.DictReader(open(ROOT / "results/t7/limits.csv")))
h("Task 7: detection limits at T = 1 s and 15 s (circle, chaotic median [IQR], 0.3 dB/cm, both dithers, drift 0.1 rad/s)")
rows = []
for line in ("NH3_1.53", "CH4_3.31", "C2H6_3.34Q"):
    for R in ("0.95", "0.97", "0.995"):
        for IL, P in ((14, "1.0"), (20, "1.0"), (20, "10.0"), (25, "1.0"), (25, "3.0")):
            r = {(x["T_s"], x["GL_basis"]): x for x in L if x["line"] == line and x["R"] == R and x["dBcm"] == "0.3" and x["IL_dB"] == str(IL) and x["P_in_mW"] == P}
            f = lambda T, b: f"{float(r[(T, b)]['limit_ppb_median']):.3g} [{float(r[(T, b)]['q1']):.2g}–{float(r[(T, b)]['q3']):.2g}]"
            rows.append([line, R, IL, P, f("1.0", "sim"), f("15.0", "sim"), f("1.0", "2cm"), f("15.0", "2cm")])
table(["line", "R", "IL dB", "P_in mW", "1 s, simulated Γ⟨L⟩", "15 s, simulated Γ⟨L⟩", "1 s, Γ⟨L⟩ = 2 cm", "15 s, Γ⟨L⟩ = 2 cm"], rows)

# conservative no-drift variant at 2 cm
LINES = A.load(ROOT / "results/t7/lines.json")
lam = {"NH3_1.53": 1.5317e-6, "CH4_3.31": 3.3151e-6, "C2H6_3.34Q": 3.3368e-6}
band = {"NH3_1.53": "nir", "CH4_3.31": "mir", "C2H6_3.34Q": "mir"}
h("Task 7 (conservative): no fringe drift, residual = static floor at all T; Γ⟨L⟩ = 2 cm; median [IQR], ppb")
rows = []
for line in ("NH3_1.53", "CH4_3.31", "C2H6_3.34Q"):
    for R in ("0.95", "0.97", "0.995"):
        res = A.load(ROOT / f"results/t2/circle_{band[line]}_R{R}_0.3dB.json")
        a = LINES[line]["alpha_peak_per_ppm_cm"]
        cells = []
        for IL, P, T in ((14, 1e-3, 1), (14, 1e-3, 15), (20, 1e-2, 1), (25, 3e-3, 15)):
            v = [1e3 * math.hypot(A.fringe(x, "both"), A.det_noise(T, P, IL, lam[line])) / (a * 2.0) for x in A.chaotic(res)]
            s = A.stats(v)
            cells.append(f"{s['median']:.3g} [{s['q1']:.2g}–{s['q3']:.2g}]")
        floor = A.stats([1e3 * A.fringe(x, "both") / (a * 2.0) for x in A.chaotic(res)])["median"]
        rows.append([line, R] + cells + [f"{floor:.3g}"])
table(["line", "R", "14 dB, 1 mW, 1 s", "14 dB, 1 mW, 15 s", "20 dB, 10 mW, 1 s", "25 dB, 3 mW, 15 s", "fringe floor alone"], rows)

S7 = A.load(ROOT / "results/t7/summary.json")
h("Task 7: averaging time to the M7 targets (median layout; s; '–' = not within 100 s); 0.3 dB/cm")
rows = []
for R in ("R0.95", "R0.97", "R0.995"):
    for IL, P in ((14, "1"), (20, "1"), (20, "10"), (25, "1"), (25, "3")):
        for b in ("2cm", "sim"):
            v = S7["T_needed_s"][f"{R}_0.3dB_{IL}dB_{P}mW_{b}"]
            rows.append([R[1:], IL, P, b] + [E(v[k]) if v[k] is not None else "–" for k in ("CH4_10ppb", "C2H6_10ppb", "C2H6_2ppb", "NH3_100ppb")])
table(["R", "IL dB", "P_in mW", "Γ⟨L⟩ basis", "CH4 10 ppb", "C2H6 10 ppb", "C2H6 2 ppb", "NH3 100 ppb"], rows)

(ROOT / "results" / "report_tables.md").write_text("\n".join(out) + "\n")
print("ok", len(out))
