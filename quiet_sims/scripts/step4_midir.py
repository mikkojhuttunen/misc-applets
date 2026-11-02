"""Step 4b: detection limits of the mid-infrared SiNx cells (results/step4, scripts/run_step4.sh) with the HITRAN peak
absorption of results/step4_gas.json: CH4 at 3.31 um and C2H6 at 3.34 um, at 1 s, and the averaging time to reach
10 ppb and 1 ppb (median NEA curve over the 21 layouts: shot sqrt(q/(P T)) at 1 mW and the fringe residual r(tau);
inf when the fringe floor stays above the target up to 100 s). -> results/step4_summary.json
python scripts/step4_midir.py"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402
from qs import noise as Nz  # noqa: E402

FILM = {"1.46": 1000, "1.59": 1200, "1.71": 1500}


def main():
    gas = M.load(M.RESULTS / "step4_gas.json")
    alpha = {"CH4": gas["lines"]["CH4_3.31"]["alpha_peak_per_ppm_cm"], "C2H6": gas["lines"]["C2H6_3.34"]["alpha_peak_per_ppm_cm"]}
    Tg = np.array(M.fig3_data()["T"])
    out = {"alpha_peak_per_ppm_cm": alpha, "P_in_W": 1e-3, "cases": {}}
    for f in sorted((M.RESULTS / "step4").glob("*.json")):
        cell, ne, db = re.match(r"(\w+?)_neff([\d.]+)_([\d.]+)dB", f.stem).groups()
        res = M.load(f)
        st = M.ensemble_stats(res, alpha_ppm=alpha["CH4"])
        T, curve, white, env, Tc = M.nea_curve(res, Tg, n_layouts=21)
        L = st["L_median_m"] * 100 * st["gamma"]
        lim = {}
        for g, a in alpha.items():
            A = a * L                                   # absorbance per ppm
            lim[g] = {"ppb_1s": 1e3 * float(np.exp(np.interp(0.0, np.log10(T), np.log(curve)))) / A,
                      "tau_10ppb_s": Nz.time_to_reach(10e-3 * A, T, white, env), "tau_1ppb_s": Nz.time_to_reach(1e-3 * A, T, white, env),
                      "floor_ppb_100s": 1e3 * float(curve[-1]) / A, "A_per_ppm": A}
        st.update(cell=cell, n_eff=float(ne), film_nm=FILM[ne], dBcm=float(db), limits=lim, white_per_rthz=white)
        out["cases"][f.stem] = st
        fmt = lambda t: "never" if math.isinf(t) else f"{t:.3g} s"
        print(f"{cell:8s} {FILM[ne]} nm loss {db} dB/cm: G<L> {L:5.2f} cm T {st['T_median']:.3f} Rb {st['R_bounce_median']:.4f}  "
              f"CH4 {lim['CH4']['ppb_1s']:.1f} ppb@1s (10 ppb {fmt(lim['CH4']['tau_10ppb_s'])}, 1 ppb {fmt(lim['CH4']['tau_1ppb_s'])})  "
              f"C2H6 {lim['C2H6']['ppb_1s']:.1f} ppb@1s (10 ppb {fmt(lim['C2H6']['tau_10ppb_s'])}, 1 ppb {fmt(lim['C2H6']['tau_1ppb_s'])})")
    M.dump(M.RESULTS / "step4_summary.json", out)


if __name__ == "__main__":
    main()
