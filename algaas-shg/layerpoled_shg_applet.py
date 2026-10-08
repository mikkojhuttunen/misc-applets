#!/usr/bin/env python3
"""Layer-poled AlGaAs rib waveguide: dispersion, Δk and mode overlap for modally phase-matched SHG.

Flat core (χ2 > 0) + rib (χ2 < 0) of AlGaAs on SiO2. TE pump fundamental at 1500-1600 nm, TM second
harmonic of higher vertical order at 750-800 nm. Built on engines/algaas_rib (semi-vectorial FD solver).

    python layerpoled_shg_applet.py                       # interactive window
    python layerpoled_shg_applet.py --png out.png         # render the default design and exit
    python layerpoled_shg_applet.py --h-core 200 --h-rib 500 --width 1450 --sh-order 2   # the thicker TM(0,2) design

Sliders update the operating point (modes at the chosen pump wavelength: n_eff, Δk, Γ, fields) live.
"Dispersion" recomputes n_eff(λ) over 750-1600 nm and Δk(λ_p) (about 15 s); the curves are greyed when
the sliders have moved since. "Tune ..." solves Δn = 0 for one geometry parameter. "Sweep" opens a
second window with Δn, Γ and A_eff against width, rib height or core height.
Export: the "Export figure + data" button (or --export PREFIX) writes a publication-style n_eff(λ) figure over
750-1600 nm with transverse-field insets (png/pdf/svg) and the data behind it (csv, npz, json); see dispersion_figure.py.
Needs numpy, scipy, matplotlib. Run from anywhere: the script adds ../math-engines to sys.path.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "math-engines"))
from engines.algaas_rib import engine as E            # noqa: E402
from engines.qpm_shg.engine import SINC2_HALF           # noqa: E402

PUMP_BAND = (1500e-9, 1600e-9)
FULL_BAND = (750e-9, 1600e-9)
NM = 1e-9


class Model:
    """Everything the GUI shows, computed without any plotting."""

    def __init__(self, st: E.RibStack, lam_pump=1550e-9, sh_vertical=1, d_eff=100e-12, length=1e-3,
                 pump_band=PUMP_BAND, step=30e-9):
        self.st, self.lam_pump, self.sh_vertical = st, lam_pump, sh_vertical
        self.d_eff, self.length, self.pump_band, self.step = d_eff, length, pump_band, step
        self.disp = None
        self.disp_key = None

    # ---- operating point
    def operating_point(self):
        st, lam = self.st, self.lam_pump
        pump = E.rib_modes(st, lam, "TE", 12, self.step, True)
        sh = E.rib_modes(st, lam / 2, "TM", 60, self.step, True)
        ip, ish = pump.find((0, 0)), sh.find((0, self.sh_vertical))
        self.pump, self.sh, self.ip, self.ish = pump, sh, ip, ish
        if ip is None or ish is None:
            self.point = None
            return None
        ov = E.overlap(pump, ip, sh, ish)
        n_p, n_s = float(pump.neff[ip]), float(sh.neff[ish])
        dk = 4 * np.pi * (n_s - n_p) / lam
        omega = 2 * np.pi * 299792458.0 / lam
        eta_norm = 2 * omega**2 * self.d_eff**2 * ov["gamma"] ** 2 / (8.8541878128e-12 * 299792458.0**3 * n_p**2 * n_s)
        others = [(abs(n - n_s), lb, n - n_s) for i, (lb, n) in enumerate(zip(sh.labels, sh.neff)) if i != ish]
        self.neighbour = min(others)[1:] if others else None
        self.absorb = E.sh_absorption(self.st, lam / 2)
        self.point = dict(n_p=n_p, n_s=n_s, dn=n_s - n_p, dk=dk, lc=np.pi / abs(dk) if dk else np.inf,
                          eta_norm=eta_norm, **ov)
        return self.point

    # ---- dispersion
    def key(self):
        return (self.st, self.pump_band, self.step)

    def grid(self):
        lo, hi = self.pump_band
        fine_p = np.linspace(lo, hi, 7)
        coarse = np.arange(FULL_BAND[0], FULL_BAND[1] + 1e-12, 50e-9)
        g = np.concatenate([coarse, fine_p, fine_p / 2])
        return np.unique(np.round(g / 1e-10) * 1e-10)

    def compute_dispersion(self, progress=None):
        lams = self.grid()
        te, tm = {}, {}
        for pol, store in (("TE", te), ("TM", tm)):
            out = {}
            for k, lam in enumerate(lams):
                if progress:
                    progress(f"{pol} {k + 1}/{len(lams)}")
                nm = int(24 + 40 * (FULL_BAND[1] - lam) / (FULL_BAND[1] - FULL_BAND[0]))
                d = E.dispersion(self.st, [lam], pol, 4 if pol == "TM" else 3, 1, self.step, True, nm)
                for lb, v in d.items():
                    out.setdefault(lb, np.full(len(lams), np.nan))[k] = v[0]
            store.update(out)
        self.disp = dict(lams=lams, te=te, tm=tm)
        self.disp_key = self.key()
        # Δk(λ_p) on the pump band
        lo, hi = self.pump_band
        lp = np.linspace(lo, hi, 7)
        ip = [int(np.argmin(abs(lams - l))) for l in lp]
        ih = [int(np.argmin(abs(lams - l / 2))) for l in lp]
        n_p = te.get((0, 0), np.full(len(lams), np.nan))[ip]
        n_s = tm.get((0, self.sh_vertical), np.full(len(lams), np.nan))[ih]
        self.pm = dict(lam_p=lp, n_p=n_p, n_s=n_s, dk=4 * np.pi * (n_s - n_p) / lp)
        return self.disp

    def stale(self):
        return self.disp is None or self.disp_key != self.key()

    def pm_summary(self):
        """Phase-matching wavelength, group indices and the sinc² acceptance bandwidth for the stored curves."""
        if self.disp is None:
            return None
        lp, dk, n_p, n_s = (self.pm[k] for k in ("lam_p", "dk", "n_p", "n_s"))
        s = {}
        ok = np.isfinite(dk)
        if ok.sum() >= 3:
            ng_p = E.group_index(lp[ok], n_p[ok])
            ng_s = E.group_index(lp[ok] / 2, n_s[ok])
            j = int(np.argmin(abs(lp[ok] - self.lam_pump)))
            s["ng_p"], s["ng_s"] = ng_p[j], ng_s[j]
            ddk = -4 * np.pi / lp[ok][j] ** 2 * (ng_s[j] - ng_p[j])
            s["bw"] = 4 * SINC2_HALF / (self.length * abs(ddk)) if ddk else np.inf
            sc = np.where(np.sign(dk[ok][:-1]) * np.sign(dk[ok][1:]) < 0)[0]
            if len(sc):
                a = sc[0]
                x0, x1, y0, y1 = lp[ok][a], lp[ok][a + 1], dk[ok][a], dk[ok][a + 1]
                s["lam_pm"] = x0 - y0 * (x1 - x0) / (y1 - y0)
        return s


# ------------------------------------------------------------------------------------------ plotting
def _outline(ax, st, hw):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-hw, 0), 2 * hw, st.h_core * 1e6, fill=False, ec="tab:blue", lw=1.0))
    ax.add_patch(Rectangle((-st.width / 2e-6 * 1.0, st.h_core * 1e6), st.width * 1e6, st.h_rib * 1e6, fill=False, ec="tab:red", lw=1.0))


def draw_fields(axs, m: Model):
    for ax in axs:
        ax.clear()
    p = getattr(m, "point", None)
    st = m.st
    if p is None:
        for ax in axs:
            ax.text(0.5, 0.5, "mode not found\n(not guided or\nraise nmodes)", ha="center", va="center", transform=ax.transAxes)
        return
    sh = m.sh
    X, Y = sh.xe * 1e6, sh.ye * 1e6
    hw = st.width * 1e6 / 2 + 1.0
    y0, y1 = -0.7, (st.h_core + st.h_rib) * 1e6 + 0.7
    fp, fs = m.pump.fields[m.ip], sh.fields[m.ish]
    integrand = sh.sign * fp**2 * fs
    for ax, f, ttl in ((axs[0], fp, f"pump TE(0,0) Ex  n={p['n_p']:.4f}"),
                       (axs[1], fs, f"SH TM(0,{m.sh_vertical}) Ey  n={p['n_s']:.4f}"),
                       (axs[2], integrand, "integrand  s·Ex²·Ey")):
        v = np.abs(f).max()
        ax.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-v, vmax=v, shading="flat")
        ax.plot([-hw, hw], [st.h_core * 1e6] * 2, color="k", lw=0.6, ls=":")
        for xs in (-st.width / 2e-6, st.width / 2e-6):
            ax.plot([xs, xs], [st.h_core * 1e6, (st.h_core + st.h_rib) * 1e6], "k", lw=0.8)
        ax.plot([-st.width / 2e-6, st.width / 2e-6], [(st.h_core + st.h_rib) * 1e6] * 2, "k", lw=0.8)
        ax.plot([-hw, hw], [0, 0], "k", lw=0.8)
        ax.plot([-hw, hw], [st.h_core * 1e6] * 2, "k", lw=0.8)
        ax.text(-hw + 0.05, st.h_core * 1e6 / 2, "χ² +", color="k", fontsize=7, va="center")
        if st.h_rib > 0:
            ax.text(0, (st.h_core + st.h_rib / 2) * 1e6, "χ² −", color="k", fontsize=7, ha="center", va="center")
        ax.set_xlim(-hw, hw)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("x (µm)", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("y (µm)", fontsize=7)


def draw_dispersion(ax, axpm, ax2, m: Model, stale=False):
    for a in (ax, axpm, ax2):
        a.clear()
    st = m.st
    if m.disp is None:
        ax.text(0.5, 0.5, "press 'Dispersion' to compute", ha="center", va="center", transform=ax.transAxes)
        return
    lams = m.disp["lams"] * 1e9
    alpha = 0.35 if stale else 1.0
    cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}
    for pol, ls, store in (("TE", "-", m.disp["te"]), ("TM", "--", m.disp["tm"])):
        for (mx, my), v in sorted(store.items()):
            if mx != 0:
                continue
            lw = 2.2 if ((pol == "TE" and (mx, my) == (0, 0)) or (pol == "TM" and (mx, my) == (0, m.sh_vertical))) else 1.0
            ax.plot(lams, v, ls, color=cmap[my], lw=lw, alpha=alpha, label=f"{pol}({mx},{my})")
    n_core = [float(E.algaas_n(st.x_core, l * 1e-9)) + st.dn_algaas for l in lams]
    n_sio2 = [E._clad_n("sio2", l * 1e-9) for l in lams]
    ax.plot(lams, n_core, "k:", lw=0.8)
    ax.plot(lams, n_sio2, "k:", lw=0.8)
    ax.axvspan(750, 800, color="tab:blue", alpha=0.08)
    ax.axvspan(m.pump_band[0] * 1e9, m.pump_band[1] * 1e9, color="tab:red", alpha=0.08)
    ax.set_xlim(750, 1600)
    ax.set_ylim(1.4, n_core[0] + 0.05)
    ax.set_xlabel("wavelength (nm)", fontsize=8)
    ax.set_ylabel("n_eff", fontsize=8)
    ax.set_title("dispersion" + ("  (stale: geometry changed)" if stale else "") + "   shaded: SH band / pump band", fontsize=8)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.tick_params(labelsize=7)

    lp = m.pm["lam_p"] * 1e9
    axpm.plot(lp, m.pm["n_p"], "o-", color="tab:blue", alpha=alpha, label="TE(0,0) pump  n(λ_p)")
    axpm.plot(lp, m.pm["n_s"], "s--", color="tab:green", alpha=alpha, label=f"TM(0,{m.sh_vertical}) SH  n(λ_p/2)")
    axpm.set_xlabel("pump wavelength λ_p (nm)", fontsize=8)
    axpm.set_ylabel("n_eff", fontsize=8)
    axpm.legend(fontsize=6, loc="best")
    axpm.tick_params(labelsize=7)
    ax2.plot(lp, m.pm["dk"] * 1e-2, "k-", lw=1.5, alpha=alpha)
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Δk (1/cm)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_zorder(axpm.get_zorder() + 1)
    ax2.patch.set_visible(False)
    s = m.pm_summary() or {}
    t = []
    if "lam_pm" in s:
        t.append(f"λ_PM = {s['lam_pm'] * 1e9:.1f} nm")
    if "bw" in s:
        t.append(f"BW(FWHM, L={m.length * 1e3:.1f} mm) = {s['bw'] * 1e9:.2f} nm")
    if "ng_p" in s:
        t.append(f"n_g,p={s['ng_p']:.3f}  n_g,SH={s['ng_s']:.3f}")
    axpm.set_title("phase matching   " + "  ".join(t), fontsize=7)


def format_text(m: Model):
    p = getattr(m, "point", None)
    st = m.st
    mq = lambda q: f"MQW {q.well * 1e9:.1f}/{q.barrier * 1e9:.1f} nm x={q.x_well:.2f}/{q.x_barrier:.2f}"
    head = (f"core {st.h_core * 1e9:.0f} nm ({mq(st.core_mqw) if st.core_mqw else f'x={st.x_core:.2f}'}, χ²>0)\n"
            f"rib  {st.h_rib * 1e9:.0f} nm ({mq(st.rib_mqw) if st.rib_mqw else f'x={st.x_rib:.2f}'}, Δn={st.dn_rib:+.3f}{'' if st.dn_rib_sh in (None, st.dn_rib) else f'/{st.dn_rib_sh:+.3f} SH'}, χ²<0), w={st.width * 1e9:.0f} nm, wall {st.sidewall_deg:.0f}°\n"
            f"pump {m.lam_pump * 1e9:.1f} nm TE(0,0) → SH {m.lam_pump * 5e8:.1f} nm TM(0,{m.sh_vertical})\n\n")
    if p is None:
        return head + "mode not found / not guided"
    lc = p["lc"] * 1e6
    return head + (f"n_p        {p['n_p']:.5f}\nn_SH       {p['n_s']:.5f}\nΔn         {p['dn']:+.5f}\n"
                   f"Δk         {p['dk'] * 1e-2:+.1f} 1/cm\nL_coh      {lc:.1f} µm\n\n"
                   f"Γ (poled)  {p['gamma'] * 1e-6:.3f} 1/µm\nΓ (uniform χ²) {p['gamma_uniform'] * 1e-6:.3f} 1/µm\n"
                   f"Γ max (s follows SH) {p['gamma_max'] * 1e-6:.3f} 1/µm\n"
                   f"A_eff      {p['a_eff'] * 1e12:.2f} µm²\n"
                   f"η_norm     {p['eta_norm'] * 1e-4:.1f} 1/(W cm²)\n"
                   f"            (d_eff = {m.d_eff * 1e12:.0f} pm/V)\n"
                   f"η at L={m.length * 1e3:.1f} mm, Δk=0: {p['eta_norm'] * m.length ** 2 * 100:.2f} %/W\n"
                   + (f"SH gap margin {m.absorb['margin']:.3f} eV ({m.absorb['worst']}), Urbach loss {m.absorb['loss_db_per_cm']:.1e} dB/cm"
                      + ("\n  WARNING: margin < 0.10 eV, SH will be absorbed" if m.absorb['margin'] < E.MIN_MARGIN_EV else "")
                      + ("\n  x > 0.45: indirect gap, formulas invalid" if max(m.st.x_core, m.st.x_rib) > 0.45 and m.st.core_mqw is None else "") + "\n"
                      + (f"nearest other TM mode {m.neighbour[0]}: Δn = {m.neighbour[1]:+.4f}"
                      + ("\n  crowded: expect mode mixing" if abs(m.neighbour[1]) < 3e-3 else "") if m.neighbour else "")))


def sweep_figure(m: Model, name, lo, hi, n=15):
    import matplotlib.pyplot as plt
    vals = np.linspace(lo, hi, n)
    r = E.sweep(m.st, m.lam_pump, name, vals, (0, 0), (0, m.sh_vertical), m.d_eff, m.step, True, 40)
    scale = 1.0 if name in ("dn_rib", "x_core", "x_rib") else 1e9
    unit = "" if scale == 1.0 else " (nm)"
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)
    ax[0].plot(vals * scale, r["delta_n"], "o-")
    ax[0].axhline(0, color="gray", lw=0.7)
    ax[0].set_ylabel("n_SH − n_p")
    ax[1].plot(vals * scale, r["gamma"] * 1e-6, "o-", label="Γ (poled)")
    ax[1].plot(vals * scale, r["gamma_uniform"] * 1e-6, "s--", label="Γ (uniform χ²)")
    ax[1].plot(vals * scale, r["gamma_max"] * 1e-6, "^:", label="Γ max")
    ax[1].set_ylabel("Γ (1/µm)")
    ax[1].legend(fontsize=7)
    ax[2].plot(vals * scale, r["eta_norm"] * 1e-4, "o-")
    ax[2].set_ylabel("η_norm (1/(W cm²), Δk=0)")
    ax[2].set_xlabel(name + unit)
    zero = np.where(np.sign(r["delta_n"][:-1]) * np.sign(r["delta_n"][1:]) < 0)[0]
    for z in zero:
        x0 = vals[z] - r["delta_n"][z] * (vals[z + 1] - vals[z]) / (r["delta_n"][z + 1] - r["delta_n"][z])
        for a in ax:
            a.axvline(x0 * scale, color="tab:red", lw=0.8, ls=":")
        ax[0].set_title(f"Δn = 0 at {name} ≈ {x0 * scale:.4g}{unit} (grid-interpolated)", fontsize=9)
    fig.tight_layout()
    return fig, r


# ------------------------------------------------------------------------------------------ GUI
def build_gui(m: Model, interactive=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    fig = plt.figure(figsize=(15.5, 9))
    fig.suptitle("Layer-poled AlGaAs rib: TE pump → higher-order TM second harmonic (modal phase matching)", fontsize=11)
    ax_d = fig.add_axes([0.30, 0.58, 0.34, 0.33])
    ax_pm = fig.add_axes([0.71, 0.58, 0.26, 0.33])
    ax_dk = ax_pm.twinx()
    ax_f = [fig.add_axes([0.30 + 0.135 * i, 0.10, 0.12, 0.38]) for i in range(3)]
    ax_t = fig.add_axes([0.72, 0.06, 0.26, 0.44])
    ax_t.axis("off")
    st = m.st

    specs = [("core h (nm)", 100, 700, st.h_core / NM, 5, "h_core"),
             ("rib h (nm)", 0, 1000, st.h_rib / NM, 5, "h_rib"),
             ("width (nm)", 400, 3000, st.width / NM, 10, "width"),
             ("x core", 0.15, 0.60, st.x_core, 0.01, "x_core"),
             ("x rib", 0.15, 0.60, st.x_rib, 0.01, "x_rib"),
             ("Δn rib (bulk)", -0.15, 0.15, st.dn_rib, 0.005, "dn_rib"),
             ("Δn rib extra at SH", -0.05, 0.05, 0.0 if st.dn_rib_sh is None else st.dn_rib_sh - st.dn_rib, 0.005, "dn_disp"),
             ("wall (°)", 0, 20, st.sidewall_deg, 1, "sidewall_deg"),
             ("pump λ (nm)", 1500, 1600, m.lam_pump / NM, 1, "lam"),
             ("d_eff (pm/V)", 10, 200, m.d_eff * 1e12, 5, "d_eff"),
             ("L (mm)", 0.1, 10, m.length * 1e3, 0.1, "L"),
             ("FD step (nm)", 10, 50, m.step / NM, 5, "step")]
    sliders = {}
    for i, (lab, lo, hi, v0, stp, key) in enumerate(specs):
        ax = fig.add_axes([0.07, 0.91 - 0.040 * i, 0.17, 0.024])
        sliders[key] = Slider(ax, lab, lo, hi, valinit=v0, valstep=stp)
        sliders[key].label.set_fontsize(8)
        sliders[key].valtext.set_fontsize(8)
    ax_r = fig.add_axes([0.04, 0.29, 0.09, 0.13])
    ax_r.set_title("SH vertical order", fontsize=8)
    radio = RadioButtons(ax_r, ("0", "1", "2", "3", "4"), active=m.sh_vertical)
    for t in radio.labels:
        t.set_fontsize(8)
    ax_r2 = fig.add_axes([0.14, 0.29, 0.09, 0.13])
    ax_r2.set_title("sweep / tune", fontsize=8)
    radio2 = RadioButtons(ax_r2, ("width", "h_rib", "h_core", "dn_rib"), active=0)
    for t in radio2.labels:
        t.set_fontsize(8)
    status = fig.text(0.04, 0.02, "", fontsize=8)
    btn = {}
    for i, (lab, key) in enumerate((("Dispersion (FD)", "disp"), ("Tune → Δn = 0", "tune"), ("Sweep window", "sweep"),
                                    ("Export figure + data", "export"), ("Save window PNG", "png"))):
        axb = fig.add_axes([0.05, 0.22 - 0.042 * i, 0.17, 0.034])
        btn[key] = Button(axb, lab)

    def read_state():
        m.st = replace(m.st, h_core=sliders["h_core"].val * NM, h_rib=sliders["h_rib"].val * NM,
                       width=sliders["width"].val * NM, x_core=sliders["x_core"].val, x_rib=sliders["x_rib"].val,
                       dn_rib=sliders["dn_rib"].val,
                       dn_rib_sh=sliders["dn_rib"].val + sliders["dn_disp"].val, sidewall_deg=sliders["sidewall_deg"].val)
        m.lam_pump = sliders["lam"].val * NM
        m.d_eff = sliders["d_eff"].val * 1e-12
        m.length = sliders["L"].val * 1e-3
        m.step = sliders["step"].val * NM
        m.sh_vertical = int(radio.value_selected)

    def refresh(_=None):
        try:
            read_state()
            m.operating_point()
            draw_fields(ax_f, m)
            ax_t.clear()
            ax_t.axis("off")
            ax_t.text(0, 1, format_text(m), va="top", family="monospace", fontsize=8)
            if m.disp is not None:
                draw_dispersion(ax_d, ax_pm, ax_dk, m, stale=m.stale())
            status.set_text("")
        except ValueError as e:
            status.set_text(f"invalid: {e}")
        fig.canvas.draw_idle()

    def do_disp(_=None):
        read_state()
        status.set_text("computing dispersion ...")
        fig.canvas.draw()
        m.compute_dispersion(progress=lambda s: (status.set_text(f"dispersion {s}"), fig.canvas.flush_events()))
        refresh()

    def do_tune(_=None):
        read_state()
        name = radio2.value_selected
        cur = getattr(m.st, name)
        lo, hi = (-0.15, 0.15) if name == "dn_rib" else (0.6 * cur, 1.5 * cur)
        status.set_text(f"tuning {name} ...")
        fig.canvas.draw()
        roots, _ = E.tune_parameter(m.st, m.lam_pump, name, lo, hi, (0, 0), (0, m.sh_vertical), 9, m.step, True, 40)
        if not roots:
            status.set_text(f"no Δn = 0 for {name} in [{lo / (1 if name == 'dn_rib' else NM):.3g}, {hi / (1 if name == 'dn_rib' else NM):.3g}] for SH order {m.sh_vertical}")
            fig.canvas.draw_idle()
            return
        best = min(roots, key=lambda r: abs(r[0] - cur))[0]
        if name == "dn_rib":
            sliders[name].set_val(round(best / 0.005) * 0.005)
            status.set_text(f"Δn = 0 at dn_rib = {best:+.4f} (slider rounded to 0.005)")
        else:
            sliders[name].set_val(round(best / NM))
            status.set_text(f"Δn = 0 at {name} = {best / NM:.1f} nm (slider rounded)")

    def do_sweep(_=None):
        read_state()
        name = radio2.value_selected
        cur = getattr(m.st, name)
        status.set_text(f"sweeping {name} ...")
        fig.canvas.draw()
        sweep_figure(m, name, *((-0.15, 0.15) if name == "dn_rib" else (0.7 * cur, 1.3 * cur)))
        plt.show(block=False)
        status.set_text("")

    def do_export(_=None):
        read_state()
        from dispersion_figure import export
        status.set_text("exporting dispersion figure ...")
        fig.canvas.draw()
        f = export("layerpoled_shg_dispersion", m.st, m.lam_pump / NM, m.sh_vertical, step=m.step,
                   progress=lambda t: (status.set_text(f"export {t}"), fig.canvas.flush_events()))
        status.set_text("wrote layerpoled_shg_dispersion.{png,pdf,svg}, _dispersion.csv, _fields.npz, _meta.json")
        plt.close(f)

    def do_png(_=None):
        fig.savefig("layerpoled_shg.png", dpi=150)
        status.set_text("saved layerpoled_shg.png")

    timer = fig.canvas.new_timer(interval=200)
    timer.single_shot = True
    timer.add_callback(refresh)
    def debounce(_=None):
        timer.stop()
        timer.start()
    for s in sliders.values():
        s.on_changed(debounce)
    radio.on_clicked(debounce)
    btn["disp"].on_clicked(do_disp)
    btn["tune"].on_clicked(do_tune)
    btn["sweep"].on_clicked(do_sweep)
    btn["export"].on_clicked(do_export)
    btn["png"].on_clicked(do_png)
    fig._keepalive = (sliders, radio, radio2, btn, timer)
    return fig, refresh, do_disp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h-core", type=float, default=150, help="core height, nm")
    ap.add_argument("--h-rib", type=float, default=200, help="rib height, nm")
    ap.add_argument("--width", type=float, default=620, help="rib width at its base, nm")
    ap.add_argument("--x-core", type=float, default=0.35, help="Al fraction of the core; 0.35 keeps >= 0.2 eV gap margin over 750-800 nm")
    ap.add_argument("--x-rib", type=float, default=0.35)
    ap.add_argument("--core-mqw", default=None, metavar="LW,LB,XW,XB", help="core as an MQW stack: well nm, barrier nm, x well, x barrier (x-core ignored)")
    ap.add_argument("--rib-mqw", default=None, metavar="LW,LB,XW,XB", help="rib as an MQW stack; defaults to the core stack when --core-mqw is given")
    ap.add_argument("--dn-rib", type=float, default=0.0, help="extra bulk index of the rib relative to the core (both wavelengths)")
    ap.add_argument("--dn-rib-sh", type=float, default=None, help="rib index offset at the SH (775 nm); default = --dn-rib")
    ap.add_argument("--wall", type=float, default=0.0, help="side-wall angle from vertical, degrees")
    ap.add_argument("--lam", type=float, default=1550, help="pump wavelength, nm")
    ap.add_argument("--sh-order", type=int, default=1, help="vertical order of the TM second-harmonic mode")
    ap.add_argument("--d-eff", type=float, default=100, help="pm/V")
    ap.add_argument("--length", type=float, default=1.0, help="mm")
    ap.add_argument("--dn-algaas", type=float, default=0.0, help="additive AlGaAs index calibration")
    ap.add_argument("--step", type=float, default=30, help="FD grid step, nm (Richardson-extrapolated)")
    ap.add_argument("--export", metavar="PREFIX", help="write the dispersion figure (png/pdf/svg) and data (csv/npz/json) and exit")
    ap.add_argument("--formats", default="png,pdf,svg", help="figure formats for --export")
    ap.add_argument("--range", default="750,1600", help="wavelength range for --export, nm")
    ap.add_argument("--n-points", type=int, default=36, help="wavelength samples for --export")
    ap.add_argument("--extra", action="append", default=[], metavar="POL:MX,MY[@NM]",
                    help="extra mode curve for --export, e.g. TM:0,0 or TE:1,0@1000 (marker + inset at 1000 nm); repeatable")
    ap.add_argument("--figsize", default="5.4,3.4", help="inches, for --export")
    ap.add_argument("--fontsize", type=float, default=11)
    ap.add_argument("--font", default=None, help="matplotlib font family for --export")
    ap.add_argument("--ylim", default=None, help="y limits for --export, e.g. 1.5,3.2")
    ap.add_argument("--inset-size", type=float, default=0.30, help="inset width as a fraction of the axes")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--png", help="render the figure to this file (with dispersion) and exit")
    ap.add_argument("--sweep", nargs=3, metavar=("PARAM", "LO_NM", "HI_NM"), help="with --png: also write <png>_sweep.png")
    a = ap.parse_args()
    def mqw(arg):
        if not arg:
            return None
        lw, lb, xw, xb = (float(v) for v in arg.split(","))
        return E.MQW(lw * NM, lb * NM, xw, xb)
    cm = mqw(a.core_mqw)
    rm = mqw(a.rib_mqw) or cm
    st = E.RibStack(a.h_core * NM, a.h_rib * NM, a.width * NM, a.x_core, a.x_rib, sidewall_deg=a.wall, dn_algaas=a.dn_algaas,
                    dn_rib=a.dn_rib, dn_rib_sh=a.dn_rib_sh, core_mqw=cm, rib_mqw=rm)
    st.validate()
    m = Model(st, a.lam * NM, a.sh_order, a.d_eff * 1e-12, a.length * 1e-3, step=a.step * NM)
    import matplotlib
    if a.export:
        matplotlib.use("Agg")
        from dispersion_figure import export
        lo, hi = (float(v) for v in a.range.split(","))
        fs = tuple(float(v) for v in a.figsize.split(","))
        fig = export(a.export, st, a.lam, a.sh_order, a.extra, (lo, hi), a.n_points, tuple(a.formats.split(",")),
                     a.step * NM, a.dpi, progress=lambda t: print("\rsolving", t, end="", flush=True), figsize=fs,
                     fontsize=a.fontsize, font=a.font, inset_size=a.inset_size,
                     ylim=tuple(float(v) for v in a.ylim.split(",")) if a.ylim else None)
        print(f"\nwrote {a.export}.{{{a.formats}}}, _dispersion.csv, _fields.npz, _meta.json; "
              f"inset scale bar = {fig._scale_bar_nm} nm")
        return
    if a.png:
        matplotlib.use("Agg")
    fig, refresh, do_disp = build_gui(m)
    if a.png:
        refresh()
        m.compute_dispersion()
        refresh()
        fig.savefig(a.png, dpi=130)
        if a.sweep:
            f2, r = sweep_figure(m, a.sweep[0], float(a.sweep[1]) * NM, float(a.sweep[2]) * NM)
            f2.savefig(a.png.replace(".png", "_sweep.png"), dpi=130)
        return
    refresh()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
