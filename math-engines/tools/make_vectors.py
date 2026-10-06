"""Shared JSON test vectors from the Python reference engines.

JavaScript ports (dbr-structures/dbr-engine.js, the physics inside
parametric-amplifier.html and cmpc-ray-tracer.html) are checked against these by tools/check_js_ports.mjs.
Run from math-engines/:  python tools/make_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines.billiard_cell import engine as bc  # noqa: E402
from engines.bragg_grating import engine as bg  # noqa: E402
from engines.ray_phase import engine as rp  # noqa: E402
from engines.cell_mirror import engine as cmir  # noqa: E402
from engines.fringe_averaging import engine as fav  # noqa: E402
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
    v["segmented_cell"] = [_cell_vector(*c) for c in _CELL_CASES]
    v["reflection_path"] = []
    for chords, R in (([1e-3, 2e-3, 3e-3], 0.95), (list(np.linspace(5e-3, 9e-3, 40)), 0.8), ([7.8e-3] * 200, 0.999)):
        L, Leff, I = bc.reflection_weighted_path(chords, R)
        v["reflection_path"].append({"chords": chords, "R": R, "L": L, "L_eff": Leff, "I_end": I})
    v["ray_phase"] = []
    for (r, N, curv, thc, A, npass, wf) in ((5e-3, 24, 0.0, np.radians(37.5), 2e-5, 12, "sine"),
                                            (5e-3, 24, 20.0, np.radians(30.0), 5e-6, 10, "triangle"),
                                            (4e-3, 16, 0.0, 0.3, 1e-4, 6, "sine")):
        cell = bc.SegmentedCell(r, N, curvature=curv)
        sc = rp.dither_scan(cell, thc, A, npass, 1.55e-6, 1.0, wf, n_samples=101)
        v["ray_phase"].append({"radius": r, "n_facets": N, "curvature": curv, "theta_c": float(thc), "amplitude": A,
                               "n_pass": npass, "waveform": wf, "wavelength": 1.55e-6, "n_index": 1.0, "n_samples": 101,
                               "offsets": [float(x) for x in sc["offsets"]], "L0": [float(x) for x in sc["L0"]],
                               "V": [float(x) for x in sc["V"]]})
    v["ray_phase_analysis"] = []
    for (N, tilt5, thc, A, npass, wf, nmin, nmax) in ((24, 0.0, np.radians(37.5), 1e-5, 20, "sine", 201, 201),
                                                     (24, 5e-4, np.radians(37.5), 1e-4, 40, "triangle", 101, 801),
                                                     (16, 0.0, 0.3, 3e-6, 12, "sine", 51, 1601)):
        tilts = [0.0] * N
        tilts[5] = tilt5
        cell = bc.SegmentedCell(5e-3, N, tilts=tilts)
        r = rp.dither_analysis(cell, thc, A, npass, 1.55e-6, 1.0, wf, n_min=nmin, n_max=nmax)
        f = lambda arr: [None if not np.isfinite(x) else float(x) for x in arr]
        v["ray_phase_analysis"].append({"radius": 5e-3, "n_facets": N, "tilts": tilts, "theta_c": float(thc), "amplitude": A,
                                        "n_pass": npass, "waveform": wf, "wavelength": 1.55e-6, "n_index": 1.0,
                                        "n_min": nmin, "n_max": nmax, "n_used": r["n_used"], "V": f(r["V"]), "V_model": f(r["V_model"]),
                                        "a": f(r["a"]), "b": f(r["b"]), "same_path": f(r["same_path"]),
                                        "resolved": [bool(x) for x in r["resolved"]]})
    v["oblique_stack"] = []
    for (lam, s, nin, layers, nout, pol) in ((1.55e-6, 0.25, 2.8, [], 1.0, "s"), (1.55e-6, 0.25, 2.8, [], 1.0, "p"),
                                             (1.55e-6, 0.6, 2.8, [(1.0, 2e-7)], 2.8, "s"),
                                             (1.5e-6, 0.3, 2.479, [(1.0, 4.2e-7), (2.479, 1.6e-7)] * 5, 1.0, "p"),
                                             (1.6e-6, 0.1, 2.0, [(1.0, 3.9e-7), (2.0, 1.9e-7)] * 8, 1.0, "s")):
        v["oblique_stack"].append({"wavelength": lam, "sin_in": s, "n_in": nin, "layers": [list(l) for l in layers], "n_out": nout,
                                   "polarization": pol, "R": float(bg.stack_R_oblique(lam, s, nin, layers, nout, pol))})
    v["trench_dbr"] = []
    for (nt, N, mg, mt, pol, sd, loss) in ((2.479, 6, 1, 1, "TM", 0.0, 0.0), (2.814, 4, 1, 3, "TE", 0.2, 3e-4), (2.011, 10, 3, 1, "TM", 0.35, 0.0)):
        d = bg.TrenchDBR(n_tooth=nt, N=N, m_gap=mg, m_tooth=mt, slab_pol=pol, sin_design=sd, bounce_loss=loss)
        sg = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.6, 0.9]
        v["trench_dbr"].append({"n_tooth": nt, "periods": N, "m_gap": mg, "m_tooth": mt, "slab_pol": pol, "sin_design": sd,
                                "bounce_loss": loss, "wavelength": 1.55e-6, "d_gap": float(d.d_gap), "d_tooth": float(d.d_tooth),
                                "sin": sg, "R": [float(x) for x in d.R(1.55e-6, np.array(sg))]})
    v["cell_mirror"] = []
    for (N, tilt5, curv, thc, fan, nb, nh, nt, pol, per, sd) in ((24, 5e-4, 0.0, np.radians(22.5), np.radians(1.1), 5, 80, 2.479, "TM", 4, 0.0),
                                                              (24, 0.0, 20.0, np.radians(22.5), np.radians(2), 4, 25, 2.814, "TE", 3, 0.0),     # chaotic: keep it short
                                                              (16, 1e-3, 0.0, 0.5, 0.0, 1, 50, 2.011, "TM", 8, 0.3)):
        tilts = [0.0] * N
        tilts[5] = tilt5
        cell = bc.SegmentedCell(5e-3, N, tilts=tilts, curvatures=[curv] * N)
        d = bg.TrenchDBR(n_tooth=nt, N=per, m_tooth=1, slab_pol=pol, sin_design=sd, bounce_loss=0.0)
        st = cmir.cell_mirror_stats(cell, thc, fan, nb, nh, d)
        v["cell_mirror"].append({"n_facets": N, "tilts": tilts, "curvature": curv, "theta_c": float(thc), "fan": float(fan),
                                 "n_beams": nb, "n_hits": nh, "n_tooth": nt, "slab_pol": pol, "periods": per, "sin_design": sd,
                                 **{k: float(st[k]) for k in ("R_mean", "R_eff", "I_end", "L_eff", "L_geom", "chi_50", "chi_95",
                                                              "chi_max", "R_design", "R_uniform")},
                                 "hist": [float(x) for x in st["hist"]]})
    v["bessel"] = [{"x": x, "J": [float(j) for j in fav.bessel_j_all(x, 40)]} for x in (0.0, 0.7, 5.5, 33.0, -12.0)]
    v["fringe"] = []
    for (x1, f1, w1, x2, f2, w2, psi, drift, coh, kind) in ((2.4, 1000, "sine", 0.0, 1300, "triangle", 0.0, 0.0, 1.0, "boxcar"),
                                                           (5.0, 1000, "sine", 3.0, 2000, "triangle", 0.6, 0.0, 0.9, "boxcar"),
                                                           (12.0, 700, "triangle", 1.5, 1300, "sine", 0.0, 0.3, 1.0, "rc"),
                                                           (40.0, 50, "sine", 8.0, 37, "sine", 1.1, 0.0, 1.0, "boxcar")):
        f, A = fav.fringe_components(x1, f1, w1, x2, f2, w2, psi)
        Ts = [1e-5, 3e-4, 2e-3, 0.0123, 0.5, 3.0]
        v["fringe"].append({"x_angle": x1, "f_angle": f1, "wave_angle": w1, "x_freq": x2, "f_freq": f2, "wave_freq": w2,
                            "phase": psi, "drift": drift, "coherence": coh, "filter": kind, "T": Ts,
                            "V": [float(x) for x in fav.residual_visibility(f, A, np.array(Ts), drift, coh, kind)],
                            "floor": fav.static_floor(f, A, drift, coh), "n_groups": int(f.size)})
    return v


_CELL_CASES = [
    (5e-3, 24, None, None, None, np.radians(37.5), 24),
    (5e-3, 12, [0, 0, 0, 2e-3, 0, 0, -1e-3, 0, 0, 0, 5e-4, 0], [0, 40, -40, 15, 0, -25, 60, 0, -10, 30, 5, -60],
     list(np.linspace(-1e-6, 1e-6, 12)), 0.4, 20),
    (4e-3, 24, [0] * 5 + [5e-4] + [0] * 18, [20.0] * 24, None, np.radians(30.0), 20),
]


def _cell_vector(r, N, tilts, curvs, offsets, theta, n_hits):
    """Launch from the middle of facet 0 at theta from the inward normal and record every hit (x, y, s, sin χ)."""
    cell = bc.SegmentedCell(r, N, tilts=tilts, curvatures=curvs, offsets=offsets)
    x, y, nx, ny = cell.point_at(np.array([cell.h]))
    ux, uy = -nx, -ny
    dx, dy = ux * np.cos(theta) - uy * np.sin(theta), ux * np.sin(theta) + uy * np.cos(theta)
    start = [float(x[0]), float(y[0]), float(dx[0]), float(dy[0])]
    hits = []
    for _ in range(n_hits):
        t, hnx, hny, hs = cell.hit(x, y, dx, dy)
        x, y = x + t * dx, y + t * dy
        hits.append([float(x[0]), float(y[0]), float(hs[0]), float((dx * hny - dy * hnx)[0])])
        dx, dy = bc._reflect(dx, dy, hnx, hny)
    return {"radius": r, "n_facets": N, "tilts": tilts, "curvatures": curvs, "offsets": offsets, "theta": float(theta),
            "start": start, "hits": hits, "mean_chord": float(cell.mean_chord)}


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
