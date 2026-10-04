"""Shared JSON test vectors from the Python reference engines.

JavaScript ports (dbr-structures/dbr-engine.js, the physics inside
parametric-amplifier.html) are checked against these by tools/check_js_ports.mjs.
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
from engines.idler_loss import engine as il  # noqa: E402
from engines.materials import engine as mat  # noqa: E402
from engines.opa_chi2 import engine as opa2  # noqa: E402
from engines.opa_chi3 import engine as opa3  # noqa: E402
from engines.slab_waveguide import engine as sw  # noqa: E402
from engines.step_index_fiber import engine as sif  # noqa: E402

OUT = ROOT / "test_vectors" / "vectors.json"


def build() -> dict:
    lams = [0.532e-6, 0.775e-6, 1.064e-6, 1.55e-6, 2.0e-6]
    v = {"units": "SI (m, 1/m); indices dimensionless", "materials": [], "slab": [], "cmt": [], "stack": [], "lp01": [], "chi2_overlap": [], "chi2_propagation": [], "chi3_area": [], "chi3_propagation": [],
         "idler_linear": [], "idler_weight": [], "chi2_dumps": [], "chi3_dumps": []}
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
    for (wp, ws, wi) in [(2e-6, 2e-6, 2e-6), (1.5e-6, 2.5e-6, 3.5e-6), (4.1e-6, 5.2e-6, 6.0e-6)]:
        v["chi2_overlap"].append({"w_pump": wp, "w_signal": ws, "w_idler": wi, "overlap": float(opa2.mode_overlap(wp, ws, wi))})
    for (kap, dk, L, r0, ap, as_, ai) in [
        (130.0, 0.0, 0.02, 1e-3, 0.0, 0.0, 0.0), (130.0, 150.0, 0.02, 1e-6, 0.0, 0.0, 0.0),
        (200.0, 0.0, 0.03, 1e-2, 0.0, 0.0, 0.0), (130.0, 60.0, 0.02, 1e-4, 1.0, 2.0, 50.0),
    ]:
        r = opa2.propagate_normalised(kap, dk, L, r0, ap, as_, ai)
        v["chi2_propagation"].append({"kappa": kap, "delta_k": dk, "length": L, "flux_ratio": r0, "alpha_pump": ap,
                                      "alpha_signal": as_, "alpha_idler": ai, "steps": r["steps"],
                                      "fp": r["fp"], "fs": r["fs"], "fi": r["fi"]})
    for (wp, ws, wi) in [(2.6e-6, 2.6e-6, 2.6e-6), (2.0e-6, 2.6e-6, 3.1e-6)]:
        v["chi3_area"].append({"w_pump": wp, "w_signal": ws, "w_idler": wi, "area_eff": float(opa3.effective_area(wp, ws, wi))})
    for (gP, db, L, ls, Ps, ap, as_, ai) in [
        (0.01, -0.02, 500.0, 1556e-9, 1e-4, 0.0, 0.0, 0.0), (0.01, 0.0, 400.0, 1560e-9, 1e-6, 0.0, 0.0, 0.0),
        (0.02, -0.03, 300.0, 1570e-9, 1e-2, 0.0, 0.0, 0.0), (0.01, -0.02, 500.0, 1556e-9, 1e-4, 1e-4, 2e-4, 5e-4),
    ]:
        lp = 1550e-9
        li = float(opa3.idler_wavelength(lp, ls))
        r0 = Ps * ls / lp      # pump normalised to 1 W
        r = opa3.propagate_normalised(gP, db, L, r0, lp / ls, lp / li, ap, as_, ai)
        v["chi3_propagation"].append({"gamma_power": gP, "delta_beta": db, "length": L, "flux_ratio": r0,
                                      "wavelength_pump": lp, "wavelength_signal": ls, "wavelength_idler": li,
                                      "alpha_pump": ap, "alpha_signal": as_, "alpha_idler": ai, "steps": r["steps"],
                                      "fp": r["fp"], "fs": r["fs"], "fi": r["fi"]})
    for (G, dk, L, ai, N, db, as_, dbs) in [
        (120.0, 50.0, 0.03, 0.0, 4, 30.0, 0.0, 0.0), (120.0, 0.0, 0.03, 300.0, 0, 0.0, 0.0, 0.0),
        (50.0, 3000.0, 0.2, 2.0e4, 0, 0.0, 0.0, 0.0), (140.0, 70.0, 0.025, 80.0, 2, 6.0, 5.0, 0.5),
        (0.02, 0.01, 400.0, 0.05, 9, 15.0, 0.0, 0.0),
    ]:
        s_, c_ = il.propagate_linear(G, dk, L, ai, N, db, as_, dbs)
        v["idler_linear"].append({"gamma": G, "delta_k": dk, "length": L, "alpha_idler": ai, "dumps": N, "dump_loss_db": db,
                                  "alpha_signal": as_, "dump_loss_signal_db": dbs, "fs": float(abs(s_) ** 2), "fi": float(abs(c_) ** 2)})
    pts = [[1550e-9, 0.0], [1560e-9, 0.8], [1580e-9, 0.2], [1600e-9, 1.0]]
    for prof in ("pass", "stop", "custom"):
        for lam in (1540e-9, 1565e-9, 1569e-9, 1570e-9, 1575.2e-9, 1590e-9, 1610e-9):
            w = il.loss_weight(prof, lam, 1570e-9, 10e-9, 0.5e-9, pts if prof == "custom" else None)
            v["idler_weight"].append({"profile": prof, "wavelength": lam, "band_center": 1570e-9, "band_width": 10e-9,
                                      "band_edge": 0.5e-9, "points": pts if prof == "custom" else None, "weight": float(w)})
    for (kap, dk, L, r0, N, dbi, dbs, dbp, ai) in [(130.0, 40.0, 0.02, 1e-3, 4, 30.0, 0.0, 0.0, 0.0),
                                                   (200.0, 0.0, 0.03, 1e-2, 9, 10.0, 0.5, 0.2, 20.0)]:
        r = opa2.propagate_normalised(kap, dk, L, r0, alpha_idler=ai, dumps=N, dump_loss_db=dbi,
                                      dump_loss_signal_db=dbs, dump_loss_pump_db=dbp)
        v["chi2_dumps"].append({"kappa": kap, "delta_k": dk, "length": L, "flux_ratio": r0, "alpha_idler": ai, "dumps": N,
                                "dump_loss_db": dbi, "dump_loss_signal_db": dbs, "dump_loss_pump_db": dbp,
                                "steps": r["steps"], "fp": r["fp"], "fs": r["fs"], "fi": r["fi"]})
    for (gP, db, L, ls, Ps, N, dbi, ai) in [(0.01, -0.02, 500.0, 1556e-9, 1e-4, 4, 30.0, 0.0), (0.02, -0.03, 300.0, 1570e-9, 1e-2, 7, 6.0, 0.01)]:
        lp = 1550e-9
        li = float(opa3.idler_wavelength(lp, ls))
        r0 = Ps * ls / lp
        r = opa3.propagate_normalised(gP, db, L, r0, lp / ls, lp / li, alpha_idler=ai, dumps=N, dump_loss_db=dbi)
        v["chi3_dumps"].append({"gamma_power": gP, "delta_beta": db, "length": L, "flux_ratio": r0, "wavelength_pump": lp,
                                "wavelength_signal": ls, "wavelength_idler": li, "alpha_idler": ai, "dumps": N, "dump_loss_db": dbi,
                                "steps": r["steps"], "fp": r["fp"], "fs": r["fs"], "fi": r["fi"]})
    return v


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
