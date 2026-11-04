"""Analyse returned probe runs: power budgets, alpha and H(K) per frequency.

python scripts/analyse_probe_results.py <folder with *_data.hdf5> [--out kernel.csv]

Pairs each corrugated run with the smooth reference of its group (same platform, mode,
wavelength), subtracts it, and writes one row per run and monitor frequency: radiated
fractions (free space, side, deficit), mode conversion, reflection, alpha in dB/cm and
H(K) from the box flux and from the deficit.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from swr.analysis.budget import kernel_point, probe_budget, subtract_reference
from swr.analysis.probe import to_db_per_cm

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import tidy3d as td  # noqa: E402

folder = Path(args.folder)
budgets = {}
for path in sorted(folder.glob("*_data.hdf5")):
    data = td.SimulationData.from_file(str(path))
    b = probe_budget(data)
    budgets[b["attrs"]["name"]] = b

rows = []
for name, b in budgets.items():
    a = b["attrs"]
    if a["amplitude_um"] == 0:
        continue
    group = name.rsplit("_L", 1)[0]
    ref = budgets.get(f"{group}_smooth")
    if ref is None:
        print(f"{name}: no smooth reference {group}_smooth, skipped")
        continue
    d = subtract_reference(b, ref)
    box, deficit = kernel_point(d, "box"), kernel_point(d, "deficit")
    for i, lam in enumerate(d["lam_um"]):
        rows.append({"name": name, "platform": a["platform"], "mode": a["mode"], "lam_um": float(lam),
                     "period_um": a["period_um"], "K_per_um": box["K_per_um"], "amplitude_nm": a["amplitude_um"] * 1e3,
                     "rad_free": float(d["rad_free"][i]), "rad_side": float(d["rad_side"][i]),
                     "deficit": float(d["deficit"][i]), "other_fwd": float(d["other_fwd"][i]),
                     "reflection": float(d["same_bwd"][i] + d["other_bwd"][i]),
                     "alpha_box_db_cm": float(to_db_per_cm(box["alpha_per_um"][i])),
                     "alpha_deficit_db_cm": float(to_db_per_cm(deficit["alpha_per_um"][i])),
                     "H_box": float(box["H"][i]), "H_deficit": float(deficit["H"][i])})

out = Path(args.out) if args.out else folder / "kernel.csv"
if rows:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gap = np.array([r["alpha_deficit_db_cm"] - r["alpha_box_db_cm"] for r in rows])
    print(f"{len(rows)} rows -> {out}; deficit - box alpha: median {np.median(gap):.3g} dB/cm "
          "(radiation leaving through the box ends)")
else:
    print("no corrugated runs with a smooth reference found")
