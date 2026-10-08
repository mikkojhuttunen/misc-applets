"""Publication-style n_eff(λ) dispersion figure with transverse-field insets, plus the data behind it.

    from dispersion_figure import export
    export("mydesign", model_state...)        # or: layerpoled_shg_applet.py --export mydesign

Writes  <prefix>.png/.pdf/.svg   the figure
        <prefix>_dispersion.csv  wavelength, n_eff and group index of every plotted mode
        <prefix>_fields.npz      x, y (µm), field maps, χ² sign map and index map of every inset
        <prefix>_meta.json       geometry, wavelengths, Δn, Δk, Γ, solver step

Curves: the TE(0,0) pump mode (red, solid, marker at λ_p) and the TM(0,m) second-harmonic mode (blue, dotted,
marker at λ_p/2), both over the whole wavelength range; dashed lines at the two marker indices and a bracket
for Δn. Extra modes can be added with specs such as "TM:0,0" or "TE:1,0@1000" (pol:lateral,vertical[@marker nm]).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "math-engines"))
from engines.algaas_rib import engine as E  # noqa: E402

RED, BLUE = "#d7191c", "#2030d8"
EXTRA_COLORS = ["#1b7f4c", "#e08a00", "#7b3fa0", "#555555"]


@dataclass
class Curve:
    pol: str
    label: tuple
    color: str
    ls: str
    marker_nm: float | None = None      # wavelength of the marker (and of the inset), None = none
    inset: bool = True
    name: str = ""

    @property
    def key(self):
        """comma-free identifier used in CSV headers and npz keys, e.g. TE00"""
        return f"{self.pol}{self.label[0]}{self.label[1]}"

    def __post_init__(self):
        self.name = self.name or f"{self.pol}({self.label[0]},{self.label[1]})"


def parse_extra(spec: str, i: int) -> Curve:
    """'TM:0,0'  or  'TE:1,0@1000' (marker and inset at 1000 nm)."""
    at = None
    if "@" in spec:
        spec, a = spec.split("@")
        at = float(a)
    pol, lab = spec.split(":")
    mx, my = (int(v) for v in lab.split(","))
    return Curve(pol.upper(), (mx, my), EXTRA_COLORS[i % len(EXTRA_COLORS)], "--", at, at is not None)


def default_curves(lam_pump_nm, sh_vertical):
    return [Curve("TE", (0, 0), RED, "-", lam_pump_nm, True),
            Curve("TM", (0, sh_vertical), BLUE, ":", lam_pump_nm / 2, True)]


def _gradient(lams_nm, n):
    """n_g = n - λ dn/dλ on the finite part of a sampled curve."""
    ng = np.full_like(n, np.nan)
    ok = np.isfinite(n)
    if ok.sum() >= 3:
        ng[ok] = E.group_index(lams_nm[ok], n[ok])
    return ng


def compute(st: E.RibStack, curves, lams_nm, step=30e-9, nmodes_max=70, progress=None):
    """n_eff(λ) of each curve plus the field of each marked curve at its marker wavelength."""
    lams = np.asarray(lams_nm, float)
    neff = {c.name: np.full(len(lams), np.nan) for c in curves}
    pols = sorted({c.pol for c in curves})
    for k, lam in enumerate(lams):
        if progress:
            progress(f"{k + 1}/{len(lams)}")
        nm = int(30 + (nmodes_max - 30) * (1600 - lam) / 850)
        for pol in pols:
            try:
                ms = E.rib_modes(st, lam * 1e-9, pol, nm, step, True)
            except ValueError:
                continue
            for c in curves:
                if c.pol == pol:
                    i = ms.find(c.label)
                    if i is not None:
                        neff[c.name][k] = ms.neff[i]
    fields = {}
    for c in curves:
        if c.marker_nm is None:
            continue
        ms = E.rib_modes(st, c.marker_nm * 1e-9, c.pol, nmodes_max, step, True)
        i = ms.find(c.label)
        if i is None:
            raise ValueError(f"{c.name} not found at {c.marker_nm:.0f} nm")
        fields[c.name] = dict(ms=ms, i=i, n=float(ms.neff[i]), lam_nm=c.marker_nm)
    return dict(lams=lams, neff=neff, ng={k: _gradient(lams, v) for k, v in neff.items()}, fields=fields)


def _draw_inset(fig, ax, c: Curve, d, st, rect, fs, bar_nm):
    ms, i = d["ms"], d["i"]
    ins = ax.inset_axes(rect)
    f = ms.fields[i]
    X, Y = ms.xe * 1e6, ms.ye * 1e6
    v = np.abs(f).max()
    ins.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-v, vmax=v, shading="flat", rasterized=True)
    hw = st.width * 1e6 / 2 + 0.55
    h_core, h_tot = st.h_core * 1e6, (st.h_core + st.h_rib) * 1e6
    ins.set_xlim(-hw, hw)
    ins.set_ylim(-0.35, h_tot + 0.30)
    ins.set_aspect("equal", adjustable="box")
    k = dict(color="k", lw=0.9, ls=(0, (3, 2)))
    ins.plot([-hw, hw], [0, 0], **k)
    ins.plot([-hw, hw], [h_core, h_core], **k)
    if st.h_rib > 0:
        w2 = st.width * 1e6 / 2
        t = np.tan(np.radians(st.sidewall_deg)) * st.h_rib * 1e6
        ins.plot([-w2, -w2 + t, w2 - t, w2], [h_core, h_tot, h_tot, h_core], **k)
    L = bar_nm * 1e-3
    ins.add_patch(__import__("matplotlib").patches.Rectangle((hw - 0.12 - L, -0.30), L, 0.07, color="k", lw=0))
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(c.color)
        s.set_linewidth(2.2)
    return ins


def make_figure(st: E.RibStack, curves, data, lam_pump_nm, lam_range=(750, 1600), figsize=(5.4, 3.4), fontsize=11,
                font=None, ylim=None, inset_rects=None, inset_size=0.30, annotate_dn=True, bar_nm=None):
    import matplotlib
    import matplotlib.pyplot as plt
    if font:
        matplotlib.rcParams["font.family"] = font
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=figsize)
    lams = data["lams"]
    for c in curves:
        if not np.isfinite(data["neff"][c.name]).any():
            continue                      # mode not guided anywhere in the range: no curve, no legend entry
        ax.plot(lams, data["neff"][c.name], c.ls, color=c.color, lw=2.0, label=c.name,
                dash_capstyle="round", solid_capstyle="round")
    allv = np.concatenate([v[np.isfinite(v)] for v in data["neff"].values()])
    lo, hi = allv.min(), allv.max()
    pad = 0.06 * (hi - lo + 1e-9)
    ax.set_xlim(*lam_range)
    ins_marked = [c for c in curves if c.inset and c.marker_nm is not None]
    ar = (st.width * 1e6 + 1.1) / ((st.h_core + st.h_rib) * 1e6 + 0.65)
    w_ax = inset_size
    h_ax = w_ax / ar * figsize[0] / figsize[1]
    if ylim is None:
        ftop = (0.04 + h_ax) if ins_marked else 0.0
        top = hi + pad
        ylim = ((lo - 0.03 * (hi - lo) - ftop * top) / (1 - ftop), top)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Wavelength (nm)", fontsize=fontsize + 1)
    ax.set_ylabel(r"Effective index $n_{\mathrm{eff}}$", fontsize=fontsize + 1)
    ax.tick_params(labelsize=fontsize, width=1.6, length=4)
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    # markers, dashed levels, bracket
    marked = [c for c in curves if c.marker_nm is not None]
    for c in marked:
        d = data["fields"][c.name]
        ax.plot([d["lam_nm"]], [d["n"]], "o", color=c.color, ms=7, zorder=5, mec="white", mew=0.8)
    if len(marked) >= 2:
        a, b = marked[0], marked[1]          # pump, SH
        da, db = data["fields"][a.name], data["fields"][b.name]
        xr = lam_range[1]
        for d, col in ((db, "0.15"), (da, "0.6")):
            ax.plot([d["lam_nm"], xr], [d["n"], d["n"]], ls=(0, (4, 3)), color=col, lw=1.3, zorder=1)
        xb = xr - 0.035 * (lam_range[1] - lam_range[0])
        ylo, yhi = sorted((da["n"], db["n"]))
        cap = 0.012 * (lam_range[1] - lam_range[0])
        if yhi - ylo > 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0]):
            ax.plot([xb, xb], [ylo, yhi], color="k", lw=1.4, zorder=2)
            for y in (ylo, yhi):
                ax.plot([xb - cap, xb + cap], [y, y], color="k", lw=1.4, zorder=2)
        if annotate_dn:
            ax.text(0.5 * (da["lam_nm"] + db["lam_nm"]), yhi + 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    rf"$\Delta n={db['n'] - da['n']:+.4f}$", ha="center", va="bottom", fontsize=fontsize - 1)
    # insets
    if inset_rects is None:
        inset_rects = {}
    bar_used = None
    for c in ins_marked:
        fx = (data["fields"][c.name]["lam_nm"] - lam_range[0]) / (lam_range[1] - lam_range[0])
        x0, y0 = inset_rects.get(c.name, (min(max(fx - w_ax / 2, 0.02), 0.98 - w_ax), 0.04))
        rect = [x0, y0, w_ax, h_ax]
        span = max(st.width * 1e6 + 1.1, 1.0)
        bn = bar_nm or (500 if span > 2.0 else 200)
        bar_used = bn
        ins = _draw_inset(fig, ax, c, data["fields"][c.name], st, rect, fontsize, bn)
        # dotted connector from marker to inset top edge
        d = data["fields"][c.name]
        fig.canvas.draw()
        tr = ax.transData.inverted()
        xm = d["lam_nm"]
        top = ax.transAxes.transform((0, y0 + h_ax))
        ytop = tr.transform(top)[1]
        if ytop < d["n"]:
            ax.plot([xm, xm], [d["n"], ytop], ls=":", color=c.color, lw=1.4, zorder=1)
    fig._scale_bar_nm = bar_used
    if len(curves) > 2:
        ax.legend(fontsize=fontsize - 2, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    return fig


def save_data(prefix, st, curves, data, lam_pump_nm, extra_meta=None):
    prefix = str(prefix)
    lams = data["lams"]
    cols = ["wavelength_nm"] + [f"neff_{c.key}" for c in curves] + [f"ng_{c.key}" for c in curves]
    arr = np.column_stack([lams] + [data["neff"][c.name] for c in curves] + [data["ng"][c.name] for c in curves])
    np.savetxt(prefix + "_dispersion.csv", arr, delimiter=",", header=",".join(cols), comments="", fmt="%.6f")
    npz = {}
    for name, d in data["fields"].items():
        ms, i = d["ms"], d["i"]
        key = name.replace("(", "").replace(")", "").replace(",", "")
        npz[f"{key}_x_um"] = 0.5 * (ms.xe[:-1] + ms.xe[1:]) * 1e6
        npz[f"{key}_y_um"] = 0.5 * (ms.ye[:-1] + ms.ye[1:]) * 1e6
        npz[f"{key}_field"] = ms.fields[i]
        npz[f"{key}_chi2_sign"] = ms.sign
        npz[f"{key}_eps"] = ms.eps
        npz[f"{key}_wavelength_nm"] = np.array(d["lam_nm"])
        npz[f"{key}_neff"] = np.array(d["n"])
    np.savez_compressed(prefix + "_fields.npz", **npz)
    meta = dict(geometry_nm=dict(h_core=st.h_core * 1e9, h_rib=st.h_rib * 1e9, width=st.width * 1e9,
                                 sidewall_deg=st.sidewall_deg),
                al_fraction=dict(core=st.x_core, rib=st.x_rib),
                index_offsets=dict(dn_algaas=st.dn_algaas, dn_rib=st.dn_rib, dn_rib_sh=st.dn_rib_sh),
                claddings=dict(bottom=st.bottom, top=st.top), pump_nm=lam_pump_nm,
                markers={k: dict(wavelength_nm=v["lam_nm"], neff=v["n"]) for k, v in data["fields"].items()},
                note="n_eff by semi-vectorial FD with Richardson extrapolation; field maps are Ex (TE) or Ey (TM).")
    meta.update(extra_meta or {})
    Path(prefix + "_meta.json").write_text(json.dumps(meta, indent=1, default=float))


def export(prefix, st, lam_pump_nm=1550.0, sh_vertical=1, extra=(), lam_range=(750, 1600), n_points=36,
           formats=("png", "pdf", "svg"), step=30e-9, dpi=300, progress=None, **fig_kwargs):
    """Compute, draw and write everything; returns the matplotlib figure."""
    curves = default_curves(lam_pump_nm, sh_vertical) + [parse_extra(s, i) for i, s in enumerate(extra)]
    lams = np.linspace(lam_range[0], lam_range[1], n_points)
    lams = np.unique(np.concatenate([lams, [lam_pump_nm, lam_pump_nm / 2]]))
    data = compute(st, curves, lams, step, progress=progress)
    fig = make_figure(st, curves, data, lam_pump_nm, lam_range, **fig_kwargs)
    for f in formats:
        fig.savefig(f"{prefix}.{f}", dpi=dpi)
    fa, fb = data["fields"][curves[0].name], data["fields"][curves[1].name]
    dn = fb["n"] - fa["n"]
    ov = E.overlap(fa["ms"], fa["i"], fb["ms"], fb["i"]) if fa["ms"].fields[0].shape == fb["ms"].fields[0].shape else {}
    save_data(prefix, st, curves, data, lam_pump_nm,
              dict(delta_n=dn, delta_k_per_m=4 * np.pi * dn / (lam_pump_nm * 1e-9), gamma_per_um=ov.get("gamma", 0) * 1e-6 if ov else None,
                   sh_absorption=E.sh_absorption(st, lam_pump_nm * 0.5e-9),
                   mqw=dict(core=repr(st.core_mqw), rib=repr(st.rib_mqw)),
                   solver_step_nm=step * 1e9, scale_bar_nm=getattr(fig, "_scale_bar_nm", None)))
    return fig
