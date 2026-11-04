"""Write Tidy3D simulation setup files for people to run (no cloud access here).

python scripts/export_tidy3d_setups.py --out setups/m1 [--platforms A_rib_LN] [--modes TE0]
       [--wavelengths 1.55] [--periods 3] [--production] [--fast] [--config config/milestone1.yaml]

For every (platform, mode, wavelength): one smooth reference and --periods corrugated runs
spread over the radiation K band. Writes
  <out>/<name>.hdf5      the td.Simulation (load with td.Simulation.from_file)
  <out>/manifest.csv     one row per run: physics parameters, cells, time steps, file name
  <out>/config_used.yaml the merged config, and the git commit, so every run is traceable
  <out>/RUN_ME.md        how to price, run and return the results
Nothing is uploaded or started.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import yaml

from swr import config
from swr.analysis.probe import periods_in_band
from swr.geometry import build_guide
from swr.sim_builders import domain_from_report, domain_from_slab, grid_cost_proxy, probe_simulation

ROOT = Path(__file__).resolve().parents[1]

RUN_ME = """# Running these setups in Tidy3D

Each `.hdf5` file is one complete `td.Simulation`. For every file:

```python
import tidy3d as td
from tidy3d import web

sim = td.Simulation.from_file("NAME.hdf5")
task_id = web.upload(sim, task_name="NAME", folder_name="sidewall_roughness")
print(web.estimate_cost(task_id))          # check against the budget before starting
web.start(task_id)
web.monitor(task_id)
web.download(task_id, path="NAME_data.hdf5")
```

The GUI works too: upload the `.hdf5` as a simulation, check the estimate, run.

Order: run `manifest.csv` top to bottom (cheapest first; the smooth reference of each
group first). Return the `NAME_data.hdf5` files, keeping the names, together with the
real cost of each run. Analysis: `python scripts/analyse_probe_results.py <folder>`.
"""


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--config", nargs="*", default=[])
ap.add_argument("--platforms", nargs="+", default=None)
ap.add_argument("--modes", nargs="+", default=None)
ap.add_argument("--wavelengths", nargs="+", type=float, default=None)
ap.add_argument("--periods", type=int, default=None, help="corrugated runs per group (default: sweep.periods_per_band)")
ap.add_argument("--production", action="store_true", help="5 nm sidewall mesh instead of the coarse one")
ap.add_argument("--fast", action="store_true", help="domain from the 1D slab estimate, no 2D mode solve")
args = ap.parse_args()

cfg = config.load(*args.config)
sw = cfg.sweep
platforms = args.platforms or list(sw.platforms)
labels = args.modes or list(sw.modes)
lams = args.wavelengths or [l for b in sw.bands for l in cfg.wavelengths_um[b]]
n_per = args.periods or int(sw.periods_per_band)
dl = cfg.fdtd.sidewall_dl_um if args.production else cfg.fdtd.sidewall_dl_coarse_um
amp = cfg.probe.amplitude_nm * 1e-3
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

rows = []
for plat in platforms:
    for lam in lams:
        report = None
        if not args.fast:
            from swr.modes import mode_report
            report = mode_report(cfg, plat, lam)
        for label in labels:
            try:
                if report is not None:
                    m = report.find(label)
                    if sw.skip_flagged_modes and (m.near_cutoff or m.lateral_leak_risk):
                        print(f"skip {plat} {label} {lam:g}: flagged (near cutoff or lateral leakage)")
                        continue
                    dom, n_rad = domain_from_report(report, label, cfg), report.n_cladding_max
                else:
                    guide = build_guide(cfg, plat, lam)
                    dom, n_rad = domain_from_slab(guide, label, cfg), guide.n_cladding_max
            except KeyError as err:
                print(f"skip {plat} {label} {lam:g}: {err}")
                continue
            periods = periods_in_band(dom.n_target, n_rad, lam, n_per, float(sw.period_edge_fraction))
            group = f"{plat}_{label}_{lam:g}"
            runs = [(float(np.median(periods)), 0.0)] + [(float(p), amp) for p in periods]
            for period, a in runs:
                name = f"{group}_smooth" if a == 0 else f"{group}_L{period:.3f}"
                sim, _ = probe_simulation(cfg, plat, lam, period, a, dom, sidewall_dl=dl, name=name)
                sim.to_file(str(out / f"{name}.hdf5"))
                rows.append({"name": name, "group": group, "platform": plat, "mode": label, "lam_um": lam,
                             "period_um": period, "K_per_um": 2 * np.pi / period, "amplitude_nm": a * 1e3,
                             "n_eff": dom.n_target, "domain_from": dom.source, "sidewall_dl_um": dl,
                             **grid_cost_proxy(sim), "file": f"{name}.hdf5"})
            print(f"{group}: {len(runs)} setups", flush=True)

group_cost = {}
for r in rows:
    group_cost[r["group"]] = group_cost.get(r["group"], 0) + r["cell_steps"]
rows.sort(key=lambda r: (group_cost[r["group"]], r["amplitude_nm"] > 0, r["cell_steps"]))
with open(out / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / "config_used.yaml").write_text(
    f"# git commit {git_commit()}\n" + yaml.safe_dump(cfg.to_dict(), sort_keys=False))
(out / "RUN_ME.md").write_text(RUN_ME)
print(f"{len(rows)} setups, {sum(r['cell_steps'] for r in rows):.3e} cell-steps -> {out}")
