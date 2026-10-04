"""Shared JSON test vectors from the Python reference engines.

JavaScript ports (dbr-structures/dbr-engine.js, er-waveguide-amplifier/er-engine.js,
the physics inside parametric-amplifier/index.html) are checked against these by tools/check_js_ports.mjs.
Run from math-engines/:  python tools/make_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines.bragg_grating import engine as bg  # noqa: E402
from engines.channel_waveguide import engine as cw  # noqa: E402
from engines.erbium_amplifier import engine as er  # noqa: E402
from engines.materials import engine as mat  # noqa: E402
from engines.slab_waveguide import engine as sw  # noqa: E402
from engines.step_index_fiber import engine as sif  # noqa: E402

OUT = ROOT / "test_vectors" / "vectors.json"


def build() -> dict:
    lams = [0.532e-6, 0.775e-6, 1.064e-6, 1.55e-6, 2.0e-6]
    v = {"units": "SI (m, 1/m); indices dimensionless", "materials": [], "slab": [], "cmt": [], "stack": [], "lp01": []}
    for m in ("sio2", "si3n4", "ln_e", "ln_o", "lt_e"):
        for l in lams:
            v["materials"].append({"material": m, "wavelength": l, "n": float(mat.index(m, l))})
    for (l, ns, nf, nc, d, pol, o) in [
        (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TE", 0), (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TM", 0),
        (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TE", 1), (0.775e-6, 1.454, 2.17, 1.0, 0.4e-6, "TE", 0),
        (1.55e-6, 1.444, 1.996, 1.444, 0.4e-6, "TE", 0), (1.55e-6, 1.444, 3.476, 1.444, 0.22e-6, "TE", 0),
    ]:
        v["slab"].append({"wavelength": l, "n_sub": ns, "n_core": nf, "n_clad": nc, "thickness": d, "polarization": pol, "order": o,
                          "neff": float(sw.neff_three_layer(l, ns, nf, nc, d, pol, o))})
    for (kap, L, dl) in [(2e4, 3e-4, 0.0), (2e4, 3e-4, 1.5e4), (2e4, 3e-4, 5e4), (5e3, 1e-3, 3e3)]:
        lamB, neff = 1.55e-6, 2.0
        lam = 1 / (dl / (2 * np.pi * neff) + 1 / lamB)
        v["cmt"].append({"kappa": kap, "length": L, "half_detuning": dl, "R": float(bg.cmt_reflectance(lam, lamB, neff, kap, L))})
    for (l, nH, nL, dH, dL, N, nin, nout) in [
        (1.55e-6, 2.0, 1.5, 1.55e-6 / 8, 1.55e-6 / 6, 5, 1.5, 1.5),
        (1.551e-6, 2.0272, 2.0095, 0.1918e-6, 0.1918e-6, 800, 2.0272, 2.0272),
        (1.548e-6, 2.0272, 2.0095, 0.1918e-6, 0.1918e-6, 800, 2.0272, 2.0272),
    ]:
        r = bg.stack_reflectance(l, nH, nL, dH, dL, N, n_in=nin, n_out=nout)
        v["stack"].append({"wavelength": l, "n_high": nH, "n_low": nL, "d_high": dH, "d_low": dL, "periods": N, "n_in": nin, "n_out": nout, "R": float(r["R"]), "T": float(r["T"])})
    for (l, a, dn) in [(1.55e-6, 4.1e-6, 0.005), (0.532e-6, 2.0e-6, 0.005), (1.064e-6, 3.0e-6, 0.008)]:
        r = sif.lp01(l, a, dn)
        v["lp01"].append({"wavelength": l, "core_radius": a, "delta_n": dn, "neff": float(r["neff"]), "V": float(r["V"]), "mode_radius": float(r["mode_radius"])})
    build_er(v)
    return v


def _fd_case():
    """Strip-loaded TFLN guide on an explicit non-uniform grid shared with the JS port."""
    xe = np.concatenate([np.linspace(-2.5e-6, -1e-6, 7), np.linspace(-1e-6, 1e-6, 21)[1:], np.linspace(1e-6, 2.5e-6, 7)[1:]])
    ye = np.concatenate([np.linspace(-1.2e-6, 0, 7), np.linspace(0, 0.3e-6, 7)[1:], np.linspace(0.3e-6, 0.7e-6, 9)[1:], np.linspace(0.7e-6, 1.7e-6, 6)[1:]])
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    reg = cw.region_at("strip", xc[None, :], yc[:, None], 2e-6, 0.3e-6, 0.4e-6)
    return xe, ye, reg


def build_er(v: dict) -> None:
    lams = [0.98e-6, 1.48e-6, 1.532e-6, 1.6e-6]
    v["materials_er"] = [{"material": m, "wavelength": l, "n": float(mat.index(m, l))}
                         for m in ("sio2", "si3n4", "ln_e", "ln_o", "lt_e", "lt_o", "al2o3", "er_al2o3") for l in lams]
    v["er_cross_sections"] = []
    for (l, pk, T) in [(1.45e-6, 5.7e-25, 295.0), (1.48e-6, 5.7e-25, 295.0), (1.5e-6, 5.7e-25, 295.0), (1.532e-6, 5.7e-25, 295.0),
                       (1.533e-6, 5.7e-25, 295.0), (1.56e-6, 5.7e-25, 295.0), (1.6e-6, 5.7e-25, 295.0), (1.64e-6, 4.0e-25, 77.0)]:
        r = er.cross_sections(l, pk, T)
        v["er_cross_sections"].append({"wavelength": l, "sigma_a_peak": pk, "temperature": T, "sigma_a": float(r["sigma_a"]), "sigma_e": float(r["sigma_e"])})
    v["er_pump_sigmas"] = []
    for lp in (0.98e-6, 0.975e-6, 1.48e-6):
        a, e = er.pump_sigmas(lp, 5.7e-25, 1.7e-25, 295.0)
        v["er_pump_sigmas"].append({"pump_wavelength": lp, "sigma_a_peak": 5.7e-25, "sigma_a_980": 1.7e-25, "temperature": 295.0, "sigma_a": a, "sigma_e": e})
    v["er_concentration"] = []
    for (n, cup, up, q, kq, fq) in [(1.5e26, 4e-24, "proportional", "proportional", 0.05, 0.1), (3e26, 4e-24, "fixed", "fixed", 0.05, 0.25),
                                    (2e26, 2e-24, "proportional", "none", 0.05, 0.1), (5e27, 4e-24, "fixed", "proportional", 0.05, 0.1)]:
        r = er.concentration_effects(n, cup, up, q, kq, fq)
        v["er_concentration"].append({"n_er": n, "c_up": cup, "upconversion": up, "quenching": q, "k_q": kq, "f_q": fq,
                                      "c_up_eff": float(r["c_up_eff"]), "f_q_eff": float(r["f_q"])})
    v["er_upper_population"] = []
    for (ru, rd, n, tau, cup) in [(2e3, 500.0, 1e26, 7.5e-3, 0.0), (2e3, 500.0, 1e26, 7.5e-3, 4e-24), (5e4, 1.5e4, 3e26, 5e-3, 1.2e-23), (0.0, 10.0, 1e26, 7.5e-3, 4e-24)]:
        v["er_upper_population"].append({"rate_up": ru, "rate_down": rd, "n_active": n, "tau": tau, "c_up": cup, "n2": float(er.upper_population(ru, rd, n, tau, cup))})
    v["er_propagate"] = []
    sas, ses = float(er.sigma_absorption(1.532e-6)), float(er.sigma_emission(1.532e-6))
    cases = [
        dict(pump_power=0.05, signal_power=1e-6, length=0.03, weight_pump=[0.2, 0.12, 0.05], weight_signal=[0.15, 0.1, 0.06], cell_area=[3e-13, 3e-13, 2e-13],
             n_er=1.5e26, tau=7.5e-3, c_up_eff=6e-24, f_q=0.075, sigma_ap=1.7e-25, sigma_ep=0.0, sigma_as=sas, sigma_es=ses, pump_wavelength=0.98e-6,
             signal_wavelength=1.532e-6, alpha=5.76, steps=60),
        dict(pump_power=0.2, signal_power=1e-3, length=0.1, weight_pump=[0.3, 0.2], weight_signal=[0.25, 0.2], cell_area=[4e-13, 4e-13],
             n_er=3e26, tau=5e-3, c_up_eff=1.2e-23, f_q=0.15, sigma_ap=float(er.sigma_absorption(1.48e-6)), sigma_ep=float(er.sigma_emission(1.48e-6)),
             sigma_as=float(er.sigma_absorption(1.55e-6)), sigma_es=float(er.sigma_emission(1.55e-6)), pump_wavelength=1.48e-6, signal_wavelength=1.55e-6,
             alpha=2.3, steps=40),
        dict(pump_power=0.0, signal_power=1e-6, length=0.02, weight_pump=[0.4], weight_signal=[0.3], cell_area=[8e-13], n_er=1e26, tau=7.5e-3,
             c_up_eff=4e-24, f_q=0.0, sigma_ap=1.7e-25, sigma_ep=0.0, sigma_as=sas, sigma_es=ses, pump_wavelength=0.98e-6, signal_wavelength=1.532e-6,
             alpha=0.0, steps=20),
    ]
    for c in cases:
        r = er.propagate(**c)
        v["er_propagate"].append({**c, "gain_ln": r["gain_ln"], "pump_out": float(r["pump_power"][-1]), "I1": r["I1"], "I2": r["I2"],
                                  "inversion_in": float(r["inversion"][0]), "inversion_out": float(r["inversion"][-1]),
                                  "probe_1550": float(er.probe_gain_ln(1.55e-6, r["I1"], r["I2"], c["length"], c["alpha"]))})
    v["eim"] = []
    for (geo, w, t, h, ts, d, nsup, nstrip, nfilm, nsub, lam) in [
        ("strip", 2e-6, 0.3e-6, 0.4e-6, 0.0, 0.0, 1.444, 1.6501, 2.1381, 1.444, 1.532e-6),
        ("ridge", 1.2e-6, 0.4e-6, 0.0, 0.0, 0.0, 1.444, 1.0, 1.996, 1.444, 1.55e-6),
        ("rib", 1.2e-6, 0.6e-6, 0.0, 0.3e-6, 0.0, 1.0, 1.0, 2.13, 1.444, 1.55e-6),
        ("buried", 2e-6, 2e-6, 0.0, 0.0, 3e-6, 1.0, 1.0, 1.47, 1.444, 1.55e-6),
    ]:
        for pol in ("TE", "TM"):
            r = cw.eim_modes(geo, w, t, nsup, nfilm, nsub, lam, pol, strip_height=h, n_strip=nstrip, slab_thickness=ts, depth=d)
            floor = max(nsub, nsup)
            ne = [float(x) for x in r["neff"] if x > floor + 2e-3][:3]
            v["eim"].append({"geometry": geo, "width": w, "film_thickness": t, "strip_height": h, "slab_thickness": ts, "depth": d,
                             "n_sup": nsup, "n_strip": nstrip, "n_film": nfilm, "n_sub": nsub, "wavelength": lam, "polarization": pol, "neff": ne})
    xe, ye, reg = _fd_case()
    ix = {"sub": 1.444, "film": 2.1381, "strip": 1.6501, "sup": 1.444}
    nmap = np.choose(reg, [ix["sub"], ix["film"], ix["strip"], ix["sup"]])
    v["fd"] = []
    for pol in ("TE", "TM"):
        r = cw.fd_modes(xe, ye, nmap, 1.532e-6, pol, 2, n_floor=1.444)
        v["fd"].append({"x_edges": xe.tolist(), "y_edges": ye.tolist(), "regions": reg.ravel().tolist(), "indices": ix, "wavelength": 1.532e-6,
                        "polarization": pol, "core_y": [0.0, 0.7e-6], "neff": [float(f"{x:.12g}") for x in r["neff"]]})


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
