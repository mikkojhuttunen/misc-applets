"""Optimise a trench DBR (N periods, as step 1) for an angle histogram and write its R(|sin chi|) table.
python scripts/make_rtable.py --hist results/step3/anglehist_m5_neff1.66.json --n_eff 1.66 --lambda_um 1.55 --N 20
  --out results/step3/Rtable_m5_neff1.66.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import mirror as Mi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hist", required=True)
ap.add_argument("--n_eff", type=float, required=True)
ap.add_argument("--lambda_um", type=float, default=1.55)
ap.add_argument("--N", type=int, default=20)
ap.add_argument("--out", required=True)
a = ap.parse_args()
h = json.loads(Path(a.hist).read_text())
s, p = np.array(h["s"]), np.array(h["p"])
b = Mi.optimise(a.n_eff, s, p, a.N)
S = np.unique(np.r_[np.linspace(0, 1, 1001), 1 / a.n_eff])
R, _, _ = Mi.design_R(a.n_eff, S, b["N"], b["s1"], b["m_gap"], b["m_tooth"], b["chirp"], b["s2"])
b["d_gap_nm"] = [x * a.lambda_um * 1e3 for x in b["d_gap"]]
b["d_tooth_nm"] = [x * a.lambda_um * 1e3 for x in b["d_tooth"]]
b["share_above_critical"] = float(p[s > 1 / a.n_eff].sum())
Path(a.out).write_text(json.dumps({"n_eff": a.n_eff, "lambda_um": a.lambda_um, "hist": a.hist, "design": b, "s": S.tolist(), "R": R.tolist()}))
print(f"{a.out}: N {b['N']} R_avg {b['R_avg']:.5f} m {b['m_gap']}/{b['m_tooth']} above critical {100 * b['share_above_critical']:.1f} %")
