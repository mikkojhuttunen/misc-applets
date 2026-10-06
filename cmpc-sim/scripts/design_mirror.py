"""Design hole-free three-index TE mirrors (air gap | thinned membrane | tooth) for Ge 300 nm TE0 at 5.263 um.

    python scripts/design_mirror.py [--out data/mirror_designs.json]

Scans the thinned-region thickness (Ge 110/130/160 nm) and the number of periods (12, 16) with differential evolution
(minimum feature 250 nm, seed 2) and a 20 nm layer-error tolerance Monte-Carlo. ~1-2 min. Deterministic."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import argparse, json, time
import numpy as np
import cmpc_sim as cs, mirror_design as md

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/mirror_designs.json"); a = ap.parse_args()
LAM = 5.2629e-6
nA = cs.membrane_mode("Ge", 300e-9, LAM, "TE", with_group_index=False).neff
res = {}
for tC in (110, 130, 160):
    nC = cs.membrane_mode("Ge", tC * 1e-9, LAM, "TE", with_group_index=False).neff
    for N in (12, 16):
        t0 = time.time(); d = md.design(LAM, nA, nC, N=N, min_feat=250e-9, seed=2, maxiter=70)
        tol = md.tolerance(LAM, d, sigma=20e-9, n=100)
        print(f"C=Ge {tC} nm (n={nC:.2f}) N={N}: gap {d['g']*1e9:.0f} thin {d['c']*1e9:.0f} tooth {d['a']*1e9:.0f} nm; <1-R>={d['mean_loss']:.4f} "
              f"max={d['max_loss']:.3f}; tol median {np.median(tol):.4f}, 90% < {np.percentile(tol, 90):.4f} [{time.time()-t0:.0f}s]")
        res[f"{tC}_{N}"] = d
json.dump(res, open(a.out, "w"), default=float)
