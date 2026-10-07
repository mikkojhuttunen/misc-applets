"""Reference outputs of gmpc (planar chain cells, Herriott traces) for the JS port in cmpc-ray-tracer.html.

Run from general-mpc/:  python tools/make_js_vectors.py   (writes tests/js_vectors.json; tests check it is current;
math-engines/tools/check_js_ports.mjs checks the JS port against it)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmpc import herriott as H, planar as P, trace2d as T  # noqa: E402

OUT = ROOT / "tests" / "js_vectors.json"


def _f(a):
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)]


def build() -> dict:
    v = {"planar": [], "herriott": []}
    cases = [
        ("stadium", dict(r=5e-3, a=5e-3, cap_facets=None, straight_segments=1), None, None, None, 0.3, 0.45, 14),
        ("stadium", dict(r=5e-3, a=4e-3, cap_facets=12, straight_segments=3), "tilt", None, None, 0.13, 0.3, 30),
        ("stadium", dict(r=4e-3, a=6e-3, cap_facets=8, straight_segments=2), None, None, "curv", 0.2, 0.62, 16),
        ("circle", dict(r=5e-3), None, None, None, 0.0, 0.5, 30),
        ("polygon", dict(r=5e-3, n_facets=24), "tilt", "off", None, None, 0.4, 40),
    ]
    for kind, kw, tilt, off, curv, s_frac, theta, n in cases:
        cell = {"stadium": P.stadium_cell, "circle": P.circle_cell, "polygon": P.polygon_cell}[kind](**kw)
        E = cell.n_elements
        tilts = (np.sin(np.arange(E) * 1.7) * 5e-4).tolist() if tilt else None
        offsets = (np.cos(np.arange(E) * 0.9) * 1e-6).tolist() if off else None
        curvs = (np.sin(np.arange(E) * 2.3) * 30.0).tolist() if curv else None
        if tilts or offsets or curvs:
            cell = P.perturb(cell, tilts=tilts, offsets=offsets, curvatures=curvs)
        s0 = cell.s_in_default if s_frac is None else s_frac * cell.perimeter
        xs, ys, S, SC, K = T.trace_path(cell, s0, theta, n)
        v["planar"].append({"kind": kind, "params": kw, "tilts": tilts, "offsets": offsets, "curvatures": curvs,
                            "s0": float(s0), "theta": theta, "n_hits": n, "x": _f(xs), "y": _f(ys), "s": _f(S), "sinchi": _f(SC),
                            "element": [int(k) for k in K], "perimeter": cell.perimeter, "area": cell.area,
                            "n_elements": E, "s_in_default": cell.s_in_default})
    for (R, N, M, A, B, pert) in ((0.5, 30, 7, 0.012, None, None), (1.0, 40, 9, 0.004, 0.003, None),
                                  (0.5, 30, 7, 0.01, None, dict(tilt_x=2e-4, tilt_y=-1e-4, dRx=1e-3, dRy=-5e-4, shift=(5e-5, 0.0, 1e-4)))):
        c = H.herriott_cell(R, N, M, A, B)
        ms = list(c["mirrors"])
        if pert:
            ms[1] = H.perturb_mirror(ms[1], **pert)
        tr = H.trace3d(ms, c["p0"], c["d0"], 3 * N)
        k = int(tr.n_hits[0])
        rr = H.reentrance(tr)
        v["herriott"].append({"R": R, "N": N, "M": M, "A": A, "B": B, "pert": pert, "d": c["d"], "hole_radius": c["hole_radius"],
                              "w_mode": c["w_mode"], "p0": _f(c["p0"]), "d0": _f(c["d0"]), "n_hits": k, "exit": str(tr.exit[0]),
                              "hits": _f(tr.hits[0, :k]), "mirror": [int(x) for x in tr.mirror[0, :k]], "cos_inc": _f(tr.cos_inc[0, :k]),
                              "path": rr["path"], "exit_offset": rr.get("exit_offset"), "reentry_angle": rr.get("reentry_angle")})
    return v


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
