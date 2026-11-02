"""Step 2: ensembles of the SiNx membrane cells with the R(sin chi) tables of step 1 and 0.1 dB/cm membrane loss.
Reads results/step2/<cell>_neff<n>[_rough5nm].json (scripts/run_step2.sh) and writes results/step2_summary.json:
static fringe for no dither / angle / spectral / both (chaotic median over 30 layouts, regular), share of layouts below
5e-7, Gamma <L> with the Gamma of each thickness, transmission, bounce-averaged R, NH3 limit at 1 s (21 layouts,
breath-broadened NH3 peak of the Fig. 3 HITRAN strips; ppb scales as 1/(Gamma <L>)). The baseline (Si, R 0.99) is
included for reference. python scripts/step2_ensembles.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402

FILM = {"1.36": 400, "1.53": 500, "1.66": 600, "1.74": 700}


def main():
    a_nh3 = M.peak_alpha("NH3")
    out = {"alpha_NH3_per_cm_ppm": a_nh3, "threshold": 5e-7, "cases": {}}
    base_f2 = M.fringe_stats(M.load(M.RESULTS / "baseline" / "f2_circle_0.99.json"))
    base_nh3 = M.detection_limit(M.load(M.RESULTS / "baseline" / "ens_tgt.json"), M.fig3_data()["T"], a_nh3, 0.2, T=1.0)
    out["baseline_Si_R0.99"] = {"fringe": base_f2, "nh3": base_nh3}
    for f in sorted((M.RESULTS / "step2").glob("*.json")):
        res = M.load(f)
        st = M.ensemble_stats(res, alpha_ppm=a_nh3)
        ne = f"{res['n_eff']:.2f}"
        st.update(cell=res["cell"], n_eff=res["n_eff"], thickness_nm=FILM.get(ne), dBcm=res["dBcm"], rtable=res["rtable"],
                  design_R_avg=res["design"]["R_avg"] if res.get("design") else None)
        out["cases"][f.stem] = st
        m = st["modes"]
        print(f"{f.stem:28s} G<L> {st['gammaL_cm']:5.2f} cm  T {st['T_median']:.3f}  Rb {st['R_bounce_median']:.5f}  fringe none/ang/spec/both "
              f"{m['none']['chaotic_median']:.2g}/{m['angle']['chaotic_median']:.2g}/{m['spectral']['chaotic_median']:.2g}/{m['both']['chaotic_median']:.2g} "
              f"(reg both {m['both']['regular']:.2g})  <5e-7 {100*m['both']['share_below']:.0f} %  NH3 {1e3*st['detection_limit']['ppm']:.0f} ppb")
    M.dump(M.RESULTS / "step2_summary.json", out)


if __name__ == "__main__":
    main()
