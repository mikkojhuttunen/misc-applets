"""Step 3: launch-angle study of the circular CMPC (results/step3, scripts/run_step3.sh): for each star orbit m (chi =
90 - 180 m/24 deg from the wall normal) and n_eff 1.66 / 1.36, the share of bounces above the critical angle (total
internal reflection), the bounce-averaged R of an N = 20 mirror optimised for that launch, Gamma <L>, transmission,
the residual fringe (static, both dithers; and after 1 s) and the NH3 limit at 1 s (21 layouts), chaotic and regular.
-> results/step3_summary.json.  python scripts/step3_launch.py"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402


def main():
    a = M.peak_alpha("NH3")
    Tg = M.fig3_data()["T"]
    out = {"alpha_NH3_per_cm_ppm": a, "cases": {}, "best": {}}
    for f in sorted((M.RESULTS / "step3").glob("circle_m*_neff*.json")):
        m, ne = re.match(r"circle_m(\d+)_neff([\d.]+)", f.stem).groups()
        res = M.load(f)
        st = M.ensemble_stats(res, alpha_ppm=a)
        tab = M.load(M.RESULTS / "step3" / f"Rtable_m{m}_neff{ne}.json")
        ch = [c for c in res["chaotic"] if "error" not in c]
        r1 = float(M.np.median([M.env_at(c["env"]["both"], Tg, 1.0) for c in ch]))
        st.update(m=int(m), chi_deg=90 - 180 * int(m) / 24, n_eff=float(ne), design=tab["design"], share_above_critical_hist=tab["design"]["share_above_critical"],
                  fringe_residual_1s_median=r1, inc_ring=res["settings"].get("inc_ring"),
                  passes_to_exit_regular=next(k for k in range(1, 25) if ((24 - int(m)) * k) % 24 == int(m)))
        out["cases"][f.stem] = st
    for ne in sorted({c["n_eff"] for c in out["cases"].values()}):
        cs = {k: v for k, v in out["cases"].items() if v["n_eff"] == ne}
        k = min(cs, key=lambda k: cs[k]["detection_limit"]["ppm"])
        out["best"][f"{ne:.2f}"] = {"case": k, "m": cs[k]["m"], "chi_deg": cs[k]["chi_deg"], "NH3_ppb": 1e3 * cs[k]["detection_limit"]["ppm"]}
    for k, v in sorted(out["cases"].items(), key=lambda kv: (kv[1]["n_eff"], -kv[1]["m"])):
        print(f"n_eff {v['n_eff']:.2f} m {v['m']:2d} chi {v['chi_deg']:5.1f}: above crit {100*v['above_critical_median']:5.1f} %  Rb {v['R_bounce_median']:.5f}  "
              f"G<L> {v['gammaL_cm']:.2f} cm  T {v['T_median']:.3f}  fringe both {v['modes']['both']['chaotic_median']:.2g} (reg {v['modes']['both']['regular']:.2g})  "
              f"r(1s) {v['fringe_residual_1s_median']:.2g}  NH3 {1e3*v['detection_limit']['ppm']:.0f} ppb "
              f"[{1e3*v['detection_limit']['ci68_ppm'][0]:.0f}-{1e3*v['detection_limit']['ci68_ppm'][1]:.0f}]  passes to exit {v['passes_to_exit_regular']}")
    print("best:", out["best"])
    M.dump(M.RESULTS / "step3_summary.json", out)


if __name__ == "__main__":
    main()
