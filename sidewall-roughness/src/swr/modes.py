"""Mode bookkeeping with the local Tidy3D mode solver.

For one platform and wavelength: list the modes with n_eff, k_eff (leakage, since every
medium is lossless), TE fraction, a TE/TM label, the sidewall power and |E|^2 fractions,
and flags for near-cutoff and lateral leakage. The cross-section domain is grown until
the boundaries sit `clearance_decay_lengths` field decay lengths away from the core for
every guided mode (capped at max_domain_um; modes still touching the PML show as lossy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import Cfg
from .geometry import Guide, build_guide
from .materials import freq
from .slab import decay_length, slab_modes

DB_PER_NEPER = 10.0 / np.log(10.0)


@dataclass
class ModeInfo:
    index: int
    label: str
    n_eff: float
    k_eff: float
    leak_db_per_cm: float
    te_fraction: float
    sidewall_power_fraction: float
    sidewall_e2_fraction: float
    sidewall_e2_per_length_um: float   # (integral of |E|^2 over the band / band width) / integral |E|^2
    guided: bool
    near_cutoff: bool
    lateral_leak_risk: bool

    def row(self):
        return asdict(self)


@dataclass
class ModeReport:
    platform: str
    lam_um: float
    domain_um: tuple[float, float]
    n_cladding_max: float
    outside_slab_neff: dict
    modes: list[ModeInfo]
    info: dict

    @property
    def guided(self):
        return [m for m in self.modes if m.guided]

    def find(self, label):
        for m in self.modes:
            if m.label == label:
                return m
        raise KeyError(f"{self.platform} at {self.lam_um} um has no guided mode {label}")


def outside_slab_neff(guide: Guide, cfg: Cfg) -> dict:
    """Fundamental slab-mode indices of the region beside the guide (empty if fully etched)."""
    if not guide.outside_stack:
        return {"TE": None, "TM": None}
    out = {}
    for pol in ("TE", "TM"):
        n = slab_modes(guide.outside_stack, guide.eps_below, guide.eps_above, guide.lam_um, pol,
                       cfg.modes.slab_scan_points)
        out[pol] = n[0] if n else None
    return out


def core_slab_neff(guide: Guide, cfg: Cfg) -> dict:
    out = {}
    for pol in ("TE", "TM"):
        n = slab_modes(guide.core_stack, guide.eps_below, guide.eps_above, guide.lam_um, pol,
                       cfg.modes.slab_scan_points)
        out[pol] = n[0] if n else None
    return out


def _clearances(modes: list, guide: Guide, outside: dict, cfg: Cfg):
    """Clearances for a list of (pol, n_eff); laterally leaky modes set no lateral clearance."""
    nd = cfg.modes.clearance_decay_lengths
    vert, lat = cfg.modes.vertical_clearance_min_um, cfg.modes.lateral_clearance_min_um
    for pol, n in modes:
        if n is None:
            continue
        for eps in (guide.eps_below, guide.eps_above):
            vert = max(vert, nd * decay_length(eps, n, guide.lam_um, pol))
        n_side = outside.get(pol) or np.sqrt(guide.eps_above[1 if pol == "TE" else 2])
        if n > n_side:
            lat = max(lat, nd * guide.lam_um / (2 * np.pi * np.sqrt(n**2 - n_side**2)))
    cap = cfg.modes.max_domain_um / 2
    return float(min(vert, cap)), float(min(lat, cap))


def interface_centered_coords(lo, hi, interfaces, dl):
    """Grid boundaries from lo to hi, spacing <= dl, with every interface at a cell CENTRE.

    Without subpixel averaging (the local mode solver), an interface lying on a grid line puts
    the tangential-E sample points exactly on it, and the film acts one cell thicker; with the
    interface at a cell centre the samples sit symmetrically on both sides (README, numerics).
    """
    pts = sorted({float(v) for v in interfaces if lo + dl < v < hi - dl})
    if not pts:
        n = max(1, int(np.ceil((hi - lo) / dl)))
        return np.linspace(lo, hi, n + 1)
    gaps = np.diff([lo] + pts + [hi])
    half = [0.5 * min(dl, 0.5 * gaps[i], 0.5 * gaps[i + 1]) for i in range(len(pts))]
    edges = [lo]
    for p, h in zip(pts, half):
        edges += [p - h, p + h]
    edges.append(hi)
    coords = [edges[0]]
    for i in range(0, len(edges) - 1, 2):
        a, b = edges[i], edges[i + 1]
        n = max(1, int(np.ceil((b - a) / dl - 1e-9)))
        coords.extend(np.linspace(a, b, n + 1)[1:].tolist())
        if i + 2 < len(edges):
            coords.append(edges[i + 2])
    return np.array(coords)


def _solve(guide: Guide, cfg: Cfg, vert: float, lat: float):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    dl = cfg.modes.grid_dl_um
    half_w = guide.core_half_width + lat if guide.kind != "slab" else lat
    z_lo, z_hi = -vert, guide.top_z + vert
    z_if = [0.0, guide.top_z] + [w.z1 for w in guide.sidewalls] + np.cumsum(
        [l.thickness for l in guide.core_stack]).tolist() + np.cumsum(
        [l.thickness for l in guide.outside_stack]).tolist()
    y_if = [v for w in guide.sidewalls for v in (w.y0, w.y1)]
    ys = interface_centered_coords(-half_w, half_w, y_if, dl)
    zs = interface_centered_coords(z_lo, z_hi, z_if, dl)
    size_y, size_z = 2 * half_w, z_hi - z_lo
    center = (0.0, 0.0, 0.5 * (z_lo + z_hi))
    grid = td.GridSpec(grid_x=td.UniformGrid(dl=dl), grid_y=td.CustomGridBoundaries(coords=ys.tolist()),
                       grid_z=td.CustomGridBoundaries(coords=zs.tolist()))
    sim = td.Simulation(
        center=center, size=(1.0, size_y, size_z), grid_spec=grid,
        structures=guide.structures, medium=guide.background, run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    plane = td.Box(center=center, size=(0, size_y, size_z))
    core = core_slab_neff(guide, cfg)
    n_guess = max(v for v in core.values() if v is not None)
    spec = td.ModeSpec(num_modes=int(cfg.modes.num_modes), target_neff=n_guess,
                       num_pml=tuple(int(v) for v in cfg.modes.num_pml))
    ms = ModeSolver(simulation=sim, plane=plane, mode_spec=spec, freqs=[float(freq(guide.lam_um))])
    return ms.solve(), (size_y, size_z)


def _pml_fraction(data, mode_index, num_pml):
    sel = dict(f=data.Ey.f.values[0], mode_index=mode_index)
    e2 = sum(np.abs(np.squeeze(getattr(data, c).sel(**sel).values)) ** 2 for c in ("Ex", "Ey", "Ez"))
    inner = e2[num_pml[0]:e2.shape[0] - num_pml[0], num_pml[1]:e2.shape[1] - num_pml[1]].sum()
    return float(1.0 - inner / e2.sum())


def _field_metrics(data, mode_index, guide: Guide, band: float):
    sel = dict(f=data.Ey.f.values[0], mode_index=mode_index)
    comps = {c: np.squeeze(getattr(data, c).sel(**sel).values) for c in ("Ex", "Ey", "Ez", "Hy", "Hz")}
    y = data.Ey.y.values
    z = data.Ey.z.values
    Y, Z = np.meshgrid(y, z, indexing="ij")
    e2 = sum(np.abs(comps[c]) ** 2 for c in ("Ex", "Ey", "Ez"))
    sx = np.real(comps["Ey"] * np.conj(comps["Hz"]) - comps["Ez"] * np.conj(comps["Hy"]))
    if not guide.sidewalls:
        return 0.0, 0.0, 0.0
    dist = np.min([w.distance(Y, Z) for w in guide.sidewalls], axis=0)
    mask = dist <= band
    p_frac = float(sx[mask].sum() / sx.sum())
    e_frac = float(e2[mask].sum() / e2.sum())
    return p_frac, e_frac, e_frac / (2 * band)


def _slab_report(guide: Guide, cfg: Cfg) -> ModeReport:
    modes = []
    n_clad = guide.n_cladding_max
    for pol in ("TE", "TM"):
        ns = slab_modes(guide.core_stack, guide.eps_below, guide.eps_above, guide.lam_um, pol,
                        cfg.modes.slab_scan_points)
        for i, n in enumerate(ns):
            modes.append(ModeInfo(index=i, label=f"{pol}{i}", n_eff=float(n), k_eff=0.0, leak_db_per_cm=0.0,
                                  te_fraction=1.0 if pol == "TE" else 0.0, sidewall_power_fraction=0.0,
                                  sidewall_e2_fraction=0.0, sidewall_e2_per_length_um=0.0,
                                  guided=bool(n > n_clad + cfg.modes.guided_margin),
                                  near_cutoff=bool(n - n_clad < cfg.modes.near_cutoff_dneff),
                                  lateral_leak_risk=False))
    modes.sort(key=lambda m: -m.n_eff)
    info = dict(guide.info, solver="exact 1D slab (no sidewalls)")
    return ModeReport(guide.platform, guide.lam_um, (np.inf, np.nan), n_clad, {"TE": None, "TM": None}, modes, info)


def mode_report(cfg: Cfg, platform: str, lam_um: float, substrate: str | None = None) -> ModeReport:
    guide = build_guide(cfg, platform, lam_um, substrate=substrate)
    if guide.kind == "slab":
        return _slab_report(guide, cfg)
    outside = outside_slab_neff(guide, cfg)
    core = core_slab_neff(guide, cfg)
    vert, lat = _clearances(list(core.items()), guide, outside, cfg)
    num_pml = tuple(int(v) for v in cfg.modes.num_pml)
    for _ in range(int(cfg.modes.max_resize_iterations)):
        data, size = _solve(guide, cfg, vert, lat)
        n_all = np.atleast_1d(np.squeeze(data.n_eff.values))
        te_all = np.atleast_1d(np.squeeze(data.pol_fraction.te.values))
        real = np.array([_pml_fraction(data, i, num_pml) <= cfg.modes.max_pml_energy_fraction
                         for i in range(n_all.size)])
        guided = [("TE" if te >= 0.5 else "TM", n) for n, te, ok in zip(n_all, te_all, real)
                  if ok and n > guide.n_cladding_max + cfg.modes.guided_margin]
        v2, l2 = _clearances(guided, guide, outside, cfg)
        if v2 <= vert * 1.05 and l2 <= lat * 1.05:
            break
        vert, lat = max(vert, v2), max(lat, l2)

    k_all = np.atleast_1d(np.squeeze(data.k_eff.values))
    k0 = 2 * np.pi / lam_um
    n_clad = guide.n_cladding_max
    counters = {"TE": 0, "TM": 0}
    modes = []
    order = np.argsort(-n_all)
    dropped = 0
    for idx in order:
        if not real[idx]:
            dropped += 1
            continue
        n, kk, te = float(n_all[idx]), float(k_all[idx]), float(te_all[idx])
        pol = "TE" if te >= 0.5 else "TM"
        guided = n > n_clad + cfg.modes.guided_margin
        label = f"{pol}{counters[pol]}" if guided else f"unguided{idx}"
        if guided:
            counters[pol] += 1
        other = outside["TM" if pol == "TE" else "TE"]
        same = outside[pol]
        lateral = (other is not None and n < other) or (same is not None and n < same)
        p_frac, e_frac, e_len = _field_metrics(data, int(idx), guide, cfg.modes.sidewall_band_um)
        modes.append(ModeInfo(
            index=int(idx), label=label, n_eff=n, k_eff=kk,
            leak_db_per_cm=DB_PER_NEPER * 2 * k0 * kk * 1e4, te_fraction=te,
            sidewall_power_fraction=p_frac, sidewall_e2_fraction=e_frac, sidewall_e2_per_length_um=e_len,
            guided=bool(guided), near_cutoff=bool(guided and n - n_clad < cfg.modes.near_cutoff_dneff),
            lateral_leak_risk=bool(guided and lateral)))
    info = dict(guide.info)
    lowest = min(float(n) for n, ok in zip(n_all, real) if ok) if real.any() else np.inf
    info.update({"mode_list_complete": bool(lowest <= n_clad + cfg.modes.guided_margin),
                 "spurious_pml_modes_dropped": dropped, "core_slab_neff": core, "clearance_um": {"vertical": vert, "lateral": lat}})
    return ModeReport(platform, lam_um, size, n_clad, outside, modes, info)


def format_report(rep: ModeReport) -> str:
    lines = [f"{rep.platform}  lambda = {rep.lam_um:.3f} um  domain {rep.domain_um[0]:.2f} x {rep.domain_um[1]:.2f} um"
             f"  n_clad,max = {rep.n_cladding_max:.4f}  outside-slab n_eff = "
             + ", ".join(f"{k} {v:.4f}" if v else f"{k} -" for k, v in rep.outside_slab_neff.items()),
             f"  {rep.info}",
             "  label       n_eff     k_eff     leak dB/cm  TE frac  P_sw frac  |E|2_sw frac  flags"]
    for m in rep.modes:
        flags = ",".join(f for f, on in (("near-cutoff", m.near_cutoff), ("lateral-leak", m.lateral_leak_risk)) if on)
        lines.append(f"  {m.label:<10} {m.n_eff:8.5f}  {m.k_eff:9.2e}  {m.leak_db_per_cm:9.3g}  "
                     f"{m.te_fraction:7.3f}  {m.sidewall_power_fraction:9.4f}  {m.sidewall_e2_fraction:11.4f}  {flags}")
    return "\n".join(lines)
