"""FDTD simulations: spectral-probe corrugation runs and smooth references.

Layout along x (corrugation from x = 0 to x = L_c = n_periods * Lambda):

    PML | source ... input monitor | corrugated section | output monitor ... | PML
                          [---- radiation flux planes (y, z faces only) ----]

Monitors give, per frequency:
* "in":  ModeMonitor before the corrugation: backward (reflected) amplitudes in every mode,
* "out": ModeMonitor after it: forward power in the excited mode and in the other modes,
* "rad_y-", "rad_y+", "rad_z-", "rad_z+": flux planes forming a box around the corrugated
  section without x faces. Flux through z faces is free-space radiation; flux through y
  faces also carries light scattered into slab modes of a rib's remaining film (reported
  separately). Each plane gives flux along +axis; outward flux is -flux on the "-" faces.
The radiated power follows twice: from the box faces, and from the deficit
1 - forward(all modes) - backward(all modes). The difference is radiation leaving through
the box ends (README, spectral probe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Cfg
from .geometry import WallProfile, build_guide
from .materials import C0_UM_PER_S, freq
from .roughness import sinusoid


@dataclass
class Domain:
    vertical_clearance: float
    lateral_clearance: float
    n_target: float          # n_eff of the intended mode
    pol: str                 # "TE" or "TM": ModeSpec filter_pol puts this polarization first
    mode_index: int          # k of TEk / TMk = index in the filter_pol-sorted mode list
    num_modes: int           # guided modes to resolve in the monitors
    source: str              # "mode_report" or "slab_estimate"


def domain_from_report(report, label: str, cfg: Cfg) -> Domain:
    mode = report.find(label)
    pol, k = label[:2], int(label[2:])
    c = report.info["clearance_um"]
    scale = cfg.fdtd.pml_clearance_decay_lengths / cfg.modes.clearance_decay_lengths
    return Domain(c["vertical"] * max(1.0, scale), c["lateral"] * max(1.0, scale), mode.n_eff, pol, k,
                  len(report.guided) + int(cfg.probe.extra_monitor_modes), "mode_report")


def domain_from_slab(guide, label: str, cfg: Cfg) -> Domain:
    """Cheap estimate (no 2D mode solve) for cost pricing: tails from the core slab mode."""
    from .modes import _clearances, core_slab_neff, outside_slab_neff
    pol, k = label[:2], int(label[2:])
    core = core_slab_neff(guide, cfg)
    outside = outside_slab_neff(guide, cfg)
    vert, lat = _clearances([(pol, core[pol])], guide, outside, cfg)
    scale = cfg.fdtd.pml_clearance_decay_lengths / cfg.modes.clearance_decay_lengths
    return Domain(vert * max(1.0, scale), lat * max(1.0, scale), core[pol] or 1.0, pol, k,
                  int(cfg.modes.num_modes), "slab_estimate")


def corrugation_profile(cfg: Cfg, period: float, amplitude: float, x_lo: float, x_hi: float) -> WallProfile:
    n_per = int(cfg.probe.n_periods)
    ppp = int(cfg.probe.points_per_period)
    length = n_per * period
    xc = np.linspace(0.0, length, n_per * ppp + 1)
    h = sinusoid(xc, amplitude, period)
    walls = cfg.probe.walls
    if walls == "one":
        hl, hr = h, np.zeros_like(h)
    elif walls == "both_in_phase":
        hl, hr = h, h
    elif walls == "both_antiphase":
        hl, hr = h, -h
    else:
        raise ValueError(f"probe.walls {walls!r}")
    x = np.concatenate([[x_lo], xc, [x_hi]])
    pad = np.zeros(1)
    return WallProfile(x, np.concatenate([pad, hl, pad]), np.concatenate([pad, hr, pad]))


def probe_simulation(cfg: Cfg, platform: str, lam_um: float, period: float, amplitude_um: float,
                     domain: Domain, sidewall_dl: float, substrate: str | None = None, name: str = ""):
    """Corrugated straight guide; amplitude 0 gives the smooth reference of the same layout."""
    import tidy3d as td

    p = cfg.probe
    l_c = int(p.n_periods) * period
    l_in, l_out = float(p.input_length_um), float(p.output_length_um)
    x_min, x_max = -l_in, l_c + l_out
    far = 10.0 * (x_max - x_min)
    profile = corrugation_profile(cfg, period, amplitude_um, -far, far) if amplitude_um > 0 else None
    guide = build_guide(cfg, platform, lam_um, profile=profile, x_extent=(-far, far), substrate=substrate)

    half_w = guide.core_half_width + domain.lateral_clearance
    z_lo, z_hi = -domain.vertical_clearance, guide.top_z + domain.vertical_clearance
    center = (0.5 * (x_min + x_max), 0.0, 0.5 * (z_lo + z_hi))
    size = (x_max - x_min, 2 * half_w, z_hi - z_lo)

    f0 = float(freq(lam_um))
    fwidth = float(cfg.fdtd.source_fwidth_fraction) * f0
    nf = int(cfg.fdtd.monitor_freqs_per_band)
    freqs = np.linspace(f0 - fwidth / 2, f0 + fwidth / 2, nf).tolist() if nf > 1 else [f0]
    mode_spec = td.ModeSpec(num_modes=int(domain.num_modes), target_neff=float(domain.n_target),
                            filter_pol=domain.pol.lower())
    plane_size = (0.0, size[1], size[2])

    source = td.ModeSource(center=(x_min + float(p.source_offset_um), 0.0, center[2]), size=plane_size,
                           source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth), direction="+",
                           mode_spec=mode_spec, mode_index=int(domain.mode_index),
                           num_freqs=int(cfg.fdtd.num_freqs_source), name="src")
    x_in = 0.5 * (x_min + float(p.source_offset_um) - float(p.flux_box_margin_x_um))
    x_out = l_c + 0.5 * l_out + 0.5 * float(p.flux_box_margin_x_um)
    m_in = td.ModeMonitor(center=(x_in, 0.0, center[2]), size=plane_size, freqs=freqs, mode_spec=mode_spec, name="in")
    m_out = td.ModeMonitor(center=(x_out, 0.0, center[2]), size=plane_size, freqs=freqs, mode_spec=mode_spec, name="out")

    gap = float(cfg.fdtd.flux_box_gap_um)
    mx = float(p.flux_box_margin_x_um)
    box_lo = (-mx, -(guide.core_half_width + gap), -gap)
    box_hi = (l_c + mx, guide.core_half_width + gap, guide.top_z + gap)
    rad = []
    for axis, label in ((1, "y"), (2, "z")):
        for side, sign in ((box_lo, "-"), (box_hi, "+")):
            c = [0.5 * (a + b) for a, b in zip(box_lo, box_hi)]
            sz = [b - a for a, b in zip(box_lo, box_hi)]
            c[axis], sz[axis] = side[axis], 0.0
            rad.append(td.FluxMonitor(center=tuple(c), size=tuple(sz), freqs=freqs, name=f"rad_{label}{sign}"))

    overrides = []
    pad = float(cfg.fdtd.sidewall_override_pad_um)
    dx = min(period / int(p.points_per_period), sidewall_dl * 4)
    for w in guide.sidewalls:
        ylo, yhi = min(w.y0, w.y1) - pad, max(w.y0, w.y1) + pad
        zlo, zhi = min(w.z0, w.z1) - pad, max(w.z0, w.z1) + pad
        overrides.append(td.MeshOverrideStructure(
            geometry=td.Box.from_bounds((x_min, ylo, zlo), (x_max, yhi, zhi)),
            dl=(dx, sidewall_dl, sidewall_dl), name="sidewall_mesh"))
    grid = td.GridSpec.auto(min_steps_per_wvl=int(cfg.fdtd.min_steps_per_wvl), wavelength=lam_um,
                            override_structures=overrides)

    n_max = float(np.sqrt(max(max(l.eps) for l in guide.core_stack)))
    t_transit = size[0] * n_max * float(cfg.fdtd.group_index_over_n_max) / C0_UM_PER_S
    run_time = float(cfg.fdtd.run_time_transits) * t_transit + 10.0 / fwidth

    pml = td.Boundary.pml(num_layers=int(cfg.fdtd.pml_layers))
    sim = td.Simulation(
        center=center, size=size, grid_spec=grid, structures=guide.structures, medium=guide.background,
        sources=[source], monitors=[m_in, m_out, *rad], run_time=run_time,
        boundary_spec=td.BoundarySpec(x=pml, y=pml, z=pml), shutoff=float(cfg.fdtd.shutoff),
        subpixel=bool(cfg.fdtd.subpixel),
        attrs={"platform": platform, "lam_um": lam_um, "period_um": period, "amplitude_um": amplitude_um,
               "corrugated_length_um": l_c, "walls": cfg.probe.walls, "mode": f"{domain.pol}{domain.mode_index}",
               "n_target": domain.n_target, "domain_source": domain.source, "name": name})
    return sim, guide


def grid_cost_proxy(sim) -> dict:
    """Local size measures (no cloud call): cells, time steps and their product."""
    cells = int(sim.num_cells)
    steps = int(sim.num_time_steps)
    return {"cells": cells, "time_steps": steps, "cell_steps": cells * steps}
