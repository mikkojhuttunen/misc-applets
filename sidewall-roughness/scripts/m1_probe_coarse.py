"""Milestone 1: one coarse corrugation (spectral-probe) simulation, built and checked locally.

python scripts/m1_probe_coarse.py [--platform A_rib_LN] [--lam 1.55] [--mode TE0] [--period P]
                                  [--fast] [--estimate] [--run]
--fast       domain from the 1D slab estimate instead of the 2D mode solve
--estimate   upload, print web.estimate_cost(), delete the draft (needs the API key; no run)
--run        estimate, print, then ask for the task name before starting (cloud credits!)
Without flags nothing leaves this machine: geometry plots, grid size and time steps only.
"""
import argparse
from pathlib import Path

import numpy as np

from swr import config
from swr.analysis.probe import periods_in_band, radiation_band
from swr.modes import format_report, mode_report
from swr.geometry import build_guide
from swr.sim_builders import domain_from_report, domain_from_slab, grid_cost_proxy, probe_simulation

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--platform", default="A_rib_LN")
ap.add_argument("--lam", type=float, default=1.55)
ap.add_argument("--mode", default="TE0")
ap.add_argument("--period", type=float, default=None)
ap.add_argument("--config", nargs="*", default=[str(ROOT / "config" / "milestone1.yaml")])
ap.add_argument("--fast", action="store_true")
ap.add_argument("--estimate", action="store_true")
ap.add_argument("--run", action="store_true")
args = ap.parse_args()

cfg = config.load(*args.config)
if args.fast:
    guide = build_guide(cfg, args.platform, args.lam)
    dom = domain_from_slab(guide, args.mode, cfg)
    n_rad = guide.n_cladding_max
else:
    rep = mode_report(cfg, args.platform, args.lam)
    print(format_report(rep))
    mode = rep.find(args.mode)
    if mode.lateral_leak_risk or mode.near_cutoff:
        print(f"WARNING: {args.mode} is flagged (lateral leak / near cutoff); probe results there are leakage-dominated")
    dom = domain_from_report(rep, args.mode, cfg)
    n_rad = rep.n_cladding_max

kmin, kmax = radiation_band(dom.n_target, n_rad, args.lam)
period = args.period or float(np.median(periods_in_band(dom.n_target, n_rad, args.lam, 5)))
print(f"radiation band K = {kmin:.3f} .. {kmax:.3f} 1/um (Lambda {2*np.pi/kmax:.3f} .. {2*np.pi/kmin:.3f} um);"
      f" probe period {period:.4f} um")

amp = cfg.probe.amplitude_nm * 1e-3
sim, guide = probe_simulation(cfg, args.platform, args.lam, period, amp, dom,
                              sidewall_dl=cfg.fdtd.sidewall_dl_coarse_um, name="m1_probe")
sim.validate_pre_upload()
proxy = grid_cost_proxy(sim)
print(f"domain {np.round(sim.size, 3).tolist()} um, {proxy['cells']:.3e} cells, {proxy['time_steps']} steps,"
      f" cell-steps {proxy['cell_steps']:.3e}; domain from {dom.source}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = ROOT / "results" / "m1"
    dest.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    sim.plot(z=guide.top_z - 0.5 * (guide.top_z - guide.sidewalls[0].z1) if guide.sidewalls else 0.5 * guide.top_z, ax=ax[0])
    sim.plot(x=0.25 * period, ax=ax[1])
    fig.tight_layout()
    fig.savefig(dest / f"probe_{args.platform}_{args.mode}_{args.lam:g}um.png", dpi=110)
    print(f"wrote {dest}/probe_{args.platform}_{args.mode}_{args.lam:g}um.png")
except ImportError:
    pass

task = f"m1_{args.platform}_{args.mode}_{args.lam:g}_L{period:.3f}"
if args.estimate or args.run:
    from swr.runners import cloud
    if args.run:
        cloud.run(sim, task, folder="sidewall_roughness_m1", path=str(ROOT / "sim_data" / f"{task}.hdf5"))
    else:
        cloud.estimate(sim, task, folder="sidewall_roughness_m1")
