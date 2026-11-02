"""Step 0: the baseline numbers from the stored ensembles and from a rerun with the repo engine.
python scripts/step0_baseline.py [--rerun-dir results/step0] -> results/step0_baseline.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import metrics as M  # noqa: E402
from design_calcs import slab  # noqa: E402

TARGET = {"fringe_both_chaotic": 4.2e-8, "fringe_both_regular": 1.7e-7, "gammaL_cm": 2.5, "nh3_ppb": 110.0,
          "slab_si_neff": 2.0, "slab_si_gamma": 0.64}


def numbers(f2, ens):
    T = M.fig3_data()["T"]
    fs = M.fringe_stats(f2)
    nh3 = M.detection_limit(ens, T, M.peak_alpha("NH3"), 0.2, T=1.0)
    return {"fringe_both_chaotic": fs["modes"]["both"]["chaotic_median"], "fringe_both_regular": fs["modes"]["both"]["regular"],
            "gammaL_cm": fs["gammaL_cm"], "nh3_ppb": nh3["ppm"] * 1e3, "nh3_detail": nh3, "fringe": fs,
            "n_layouts_fringe": fs["n_layouts"], "n_layouts_nh3": len(ens["chaotic"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-dir", default=str(M.RESULTS / "step0"))
    ap.add_argument("--out", default=str(M.RESULTS / "step0_baseline.json"))
    a = ap.parse_args()
    b = M.RESULTS / "baseline"
    out = {"target": TARGET, "stored": numbers(M.load(b / "f2_circle_0.99.json"), M.load(b / "ens_tgt.json"))}
    rd = Path(a.rerun_dir)
    if (rd / "f2_circle_0.99.json").exists() and (rd / "ens_tgt.json").exists():
        out["rerun"] = numbers(M.load(rd / "f2_circle_0.99.json"), M.load(rd / "ens_tgt.json"))
    ne, g = slab.gam(3.48, 250e-9, 1.55e-6, "TM")
    out["slab"] = {"si_250nm_tm0_neff": ne, "si_250nm_tm0_gamma": float(g)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for k in ("stored", "rerun"):
        if k in out:
            n = out[k]
            print(f"{k:7s} fringe both chaotic {n['fringe_both_chaotic']:.3g} regular {n['fringe_both_regular']:.3g}  Gamma<L> {n['gammaL_cm']:.2f} cm  NH3 {n['nh3_ppb']:.0f} ppb")
    print(f"slab Si 250 nm TM0 n_eff {ne:.3f} Gamma {g:.3f}")


if __name__ == "__main__":
    main()
