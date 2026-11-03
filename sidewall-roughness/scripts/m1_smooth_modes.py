"""Milestone 1: smooth-guide mode tables (local mode solver, no cloud).

python scripts/m1_smooth_modes.py [--platforms A_rib_LN B_strip_LN] [--wavelengths 0.633 1.55]
                                  [--config config/milestone1.yaml]
Writes results/m1/modes.json and prints one table per platform and wavelength.
"""
import argparse
import json
from pathlib import Path

from swr import config
from swr.modes import format_report, mode_report

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--platforms", nargs="+", default=["A_rib_LN", "B_strip_LN", "E_slab_LN"])
ap.add_argument("--wavelengths", nargs="+", type=float, default=[0.633, 1.55])
ap.add_argument("--config", nargs="*", default=[str(ROOT / "config" / "milestone1.yaml")])
ap.add_argument("--substrate", default=None)
args = ap.parse_args()

cfg = config.load(*args.config)
out = []
for plat in args.platforms:
    for lam in args.wavelengths:
        rep = mode_report(cfg, plat, lam, substrate=args.substrate)
        print(format_report(rep), flush=True)
        out.append({"platform": plat, "lam_um": lam, "domain_um": rep.domain_um,
                    "n_cladding_max": rep.n_cladding_max, "outside_slab_neff": rep.outside_slab_neff,
                    "info": {k: v for k, v in rep.info.items() if k != "film_source"},
                    "modes": [m.row() for m in rep.modes]})
dest = ROOT / "results" / "m1"
dest.mkdir(parents=True, exist_ok=True)
(dest / "modes.json").write_text(json.dumps(out, indent=1, default=float))
print(f"wrote {dest / 'modes.json'}")
