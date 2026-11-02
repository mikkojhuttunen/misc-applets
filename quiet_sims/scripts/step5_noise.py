"""Step 5: noise budget with detector noise and insertion loss, for the mid-infrared SiNx cells of step 4.

Cases: idler power at the cell input P_in = 0.1, 1, 10 mW; cell insertion loss IL = 20 dB (M4 limit, Si cell) and 35 dB
(M4 limit, SiNx cell); detector responsivity 1 A/W, NEP = 1e-12, 1e-11, 1e-10 W/sqrt(Hz). NEA(tau) = sqrt((q/P_det +
(NEP/P_det)^2)/tau + r(tau)^2), P_det = P_in 10^(-IL/10) (qs/noise.py); r(tau) the median fringe residual of the cell's
21 layouts and Gamma <L> its median (the cell, by default the circle and the Herriott case with the lowest CH4 limit at
0.3 dB/cm, can be chosen with --case). Detection limits for CH4 (3.31 um) and C2H6 (3.34 um) with the HITRAN peaks of
step 4, at 1 s and 10 s, and the averaging time for 10 ppb and 1 ppb. -> results/step5_noise.json
python scripts/step5_noise.py [--case circle_neff1.59_0.3dB] [--case ...]"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402
from qs import noise as Nz  # noqa: E402

P_IN = (1e-4, 1e-3, 1e-2)
IL = (20.0, 35.0)
NEP = (1e-12, 1e-11, 1e-10)
TAUS = (1.0, 10.0)


def budget(case, s4, alpha, Tg):
    res = M.load(M.RESULTS / "step4" / f"{case}.json")
    st = s4["cases"][case]
    ch = res["chaotic"][:21]
    r = np.median(np.array([c["env"]["both"] for c in ch]), axis=0)
    gL = st["gammaL_cm"]
    rows = []
    for P, il, nep in itertools.product(P_IN, IL, NEP):
        w = Nz.white(P, il, nep)
        curve = Nz.nea(Tg, w["white"], r)
        row = {"P_in_mW": P * 1e3, "IL_dB": il, "NEP_W_rtHz": nep, "P_det_W": w["P_det"], "shot_per_rtHz": w["shot"], "detector_per_rtHz": w["detector"],
               "white_per_rtHz": w["white"], "limits": {}}
        for g, a in alpha.items():
            A = a * gL
            lim = {f"ppb_{int(t)}s": 1e3 * float(np.exp(np.interp(math.log10(t), np.log10(Tg), np.log(curve)))) / A for t in TAUS}
            lim["tau_10ppb_s"] = Nz.time_to_reach(10e-3 * A, Tg, w["white"], r)
            lim["tau_1ppb_s"] = Nz.time_to_reach(1e-3 * A, Tg, w["white"], r)
            lim["fringe_floor_ppb_100s"] = 1e3 * float(r[-1]) / A
            lim["M7_margin_10ppb_1s"] = 10.0 / lim["ppb_1s"]
            row["limits"][g] = lim
        rows.append(row)
    return {"case": case, "gammaL_cm": gL, "T_cell": st["T_median"], "fringe_residual_1s": float(np.exp(np.interp(0.0, np.log10(Tg), np.log(r)))),
            "nea_tau_grid_s": list(Tg), "fringe_residual_curve": r.tolist(), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append")
    a = ap.parse_args()
    s4 = M.load(M.RESULTS / "step4_summary.json")
    alpha = s4["alpha_peak_per_ppm_cm"]
    Tg = np.array(M.fig3_data()["T"])
    cases = a.case
    if not cases:
        cases = []
        for cell in ("circle", "herriott"):
            cs = {k: v for k, v in s4["cases"].items() if v["cell"] == cell and abs(v["dBcm"] - 0.3) < 1e-9}
            cases.append(min(cs, key=lambda k: cs[k]["limits"]["CH4"]["ppb_1s"]))
    out = {"alpha_peak_per_ppm_cm": alpha, "P_in_W": P_IN, "IL_dB": IL, "NEP_W_rtHz": NEP, "responsivity_A_W": 1.0, "budgets": {}}
    for c in cases:
        b = budget(c, s4, alpha, Tg)
        out["budgets"][c] = b
        print(f"\n{c}: Gamma<L> {b['gammaL_cm']:.2f} cm, fringe residual at 1 s {b['fringe_residual_1s']:.2g}")
        print(" P_in  IL  NEP   | CH4 1 s  10 s  | C2H6 1 s 10 s | M7 margin CH4/C2H6 | 10 ppb at 1 s / 10 s | 1 ppb at 1 s / 10 s")
        for r in b["rows"]:
            ch4, c2 = r["limits"]["CH4"], r["limits"]["C2H6"]
            ok = lambda v, t: "yes" if v <= t else "no"
            print(f" {r['P_in_mW']:4.1f} {r['IL_dB']:3.0f} {r['NEP_W_rtHz']:.0e} | {ch4['ppb_1s']:7.2f} {ch4['ppb_10s']:6.2f} | {c2['ppb_1s']:7.2f} {c2['ppb_10s']:6.2f} | "
                  f"{ch4['M7_margin_10ppb_1s']:6.2f} / {c2['M7_margin_10ppb_1s']:6.2f} | CH4 {ok(ch4['ppb_1s'], 10)}/{ok(ch4['ppb_10s'], 10)} C2H6 {ok(c2['ppb_1s'], 10)}/{ok(c2['ppb_10s'], 10)}"
                  f" | CH4 {ok(ch4['ppb_1s'], 1)}/{ok(ch4['ppb_10s'], 1)} C2H6 {ok(c2['ppb_1s'], 1)}/{ok(c2['ppb_10s'], 1)}")
    M.dump(M.RESULTS / "step5_noise.json", out)


if __name__ == "__main__":
    main()
