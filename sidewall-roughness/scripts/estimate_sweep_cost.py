"""Price the full probe sweep before anything is submitted.

python scripts/estimate_sweep_cost.py [--config ...] [--production] [--modes] [--cloud] [--limit N]

Default: local only. Builds every simulation of the sweep (config `sweep`), with domains from
the 1D slab estimate (fast) or, with --modes, from the 2D mode solver (slow, also applies the
skip-flagged-mode rule), and prints cells x time steps per run, sorted cheapest first.
--cloud  additionally uploads each run, prints web.estimate_cost() and deletes the draft.
         Needs the Tidy3D API key. Nothing is ever started.
--production uses fdtd.sidewall_dl_um instead of the coarse milestone mesh.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from swr import config
from swr.analysis.probe import periods_in_band
from swr.geometry import build_guide
from swr.sim_builders import domain_from_report, domain_from_slab, grid_cost_proxy, probe_simulation

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--config", nargs="*", default=[])
ap.add_argument("--production", action="store_true")
ap.add_argument("--modes", action="store_true")
ap.add_argument("--cloud", action="store_true")
ap.add_argument("--limit", type=int, default=None, help="price only the N cheapest runs on the cloud")
args = ap.parse_args()

cfg = config.load(*args.config)
sw = cfg.sweep
dl = cfg.fdtd.sidewall_dl_um if args.production else cfg.fdtd.sidewall_dl_coarse_um
amp = cfg.probe.amplitude_nm * 1e-3
rows, sims = [], []
for plat in sw.platforms:
    for band in sw.bands:
        for lam in cfg.wavelengths_um[band]:
            report = None
            if args.modes:
                from swr.modes import mode_report
                report = mode_report(cfg, plat, lam)
            for label in sw.modes:
                try:
                    if report is not None:
                        m = report.find(label)
                        if sw.skip_flagged_modes and (m.near_cutoff or m.lateral_leak_risk):
                            print(f"skip {plat} {label} {lam}: flagged")
                            continue
                        dom = domain_from_report(report, label, cfg)
                        n_rad = report.n_cladding_max
                    else:
                        guide = build_guide(cfg, plat, lam)
                        dom = domain_from_slab(guide, label, cfg)
                        n_rad = guide.n_cladding_max
                except KeyError as err:
                    print(f"skip {plat} {label} {lam}: {err}")
                    continue
                periods = periods_in_band(dom.n_target, n_rad, lam, int(sw.periods_per_band),
                                          float(sw.period_edge_fraction))
                amps = [amp] * len(periods) + ([0.0] if sw.smooth_reference else [])
                pers = list(periods) + ([float(np.median(periods))] if sw.smooth_reference else [])
                for period, a in zip(pers, amps):
                    name = f"{plat}_{label}_{lam:g}_L{period:.3f}" + ("_smooth" if a == 0 else "")
                    sim, _ = probe_simulation(cfg, plat, lam, float(period), a, dom, sidewall_dl=dl, name=name)
                    proxy = grid_cost_proxy(sim)
                    rows.append({"name": name, "platform": plat, "mode": label, "lam_um": lam,
                                 "period_um": float(period), "amplitude_um": a, **proxy})
                    sims.append(sim)
                print(f"{plat} {label} {lam:g} um: {len(pers)} runs, domain from {dom.source}", flush=True)

order = np.argsort([r["cell_steps"] for r in rows])
rows = [rows[i] for i in order]
sims = [sims[i] for i in order]
total = sum(r["cell_steps"] for r in rows)
print(f"\n{len(rows)} runs, total {total:.3e} cell-steps (mesh {'production' if args.production else 'coarse'})")
print("cheapest:", rows[0]["name"], f"{rows[0]['cell_steps']:.2e}", " most expensive:", rows[-1]["name"],
      f"{rows[-1]['cell_steps']:.2e}")

if args.cloud:
    from swr.runners import cloud
    n = len(rows) if args.limit is None else args.limit
    spent = 0.0
    for r, sim in zip(rows[:n], sims[:n]):
        _, cost = cloud.estimate(sim, r["name"], folder="sidewall_roughness_estimates")
        r["flexcredit_estimate"] = cost
        spent += cost
    print(f"estimated FlexCredit for the {n} cheapest runs: {spent:.3f}")
    if n < len(rows):
        per_cell_step = spent / sum(r["cell_steps"] for r in rows[:n])
        print(f"extrapolated to all {len(rows)} runs by cell-steps: {per_cell_step * total:.2f} (rough)")

dest = ROOT / "results"
dest.mkdir(exist_ok=True)
with open(dest / "sweep_cost.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {dest / 'sweep_cost.csv'}")
