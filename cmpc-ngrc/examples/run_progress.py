"""First experiments of the cmpc-ngrc plan; writes results/progress.json (read by the explorer applet) and figures.

  E1  sensitivity per harmonic order m vs launch angle (4 symmetric ports, one input); ray coverage maps
  E2  port layouts: symmetric / asymmetric, 1, 2 or 4 inputs, 8 ports: sensitivity and ridge-readout R²
      (shape at a fixed position)
  E3  readouts on one layout: ridge (linear), NGRC (quadratic), linear SVR, RBF SVR, SVC for the dominant m;
      baselines: the same readouts on the raw Δn image, and the exact CHD
  E4  validity of the phase-screen model against full curved rays vs Δn, and speckle decorrelation vs Δn

python examples/run_progress.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "math-engines"))

from ngrc import analysis as A                                     # noqa: E402
from ngrc import readout as RO                                     # noqa: E402
from ngrc.dataset import Ensemble, build                           # noqa: E402
from ngrc.features import ngrc_features                            # noqa: E402
from ngrc.geometry import CircularCell, Port, ports_at, symmetric_ports  # noqa: E402
from ngrc.perturbative import PhaseScreenModel                     # noqa: E402
from ngrc.rays import Source                                       # noqa: E402
from ngrc.shapes import Perturbation, Shape, random_shape          # noqa: E402

M_MAX = 6
PORT_W = 20e-6
SRC = dict(n_pos=3, n_ang=41)


def cell_with(ports):
    return CircularCell(radius=1e-3, n_eff=1.80, wavelength=1.55e-6, ports=ports, reflectance=0.99)


def layouts(launch=0.35, fan=0.4):
    kw = dict(width=PORT_W, launch=launch, fan=fan)
    asym = [0, 67, 151, 238]
    return {
        "sym4 · 1 in": symmetric_ports(4, inputs=(0,), **kw),
        "asym4 · 1 in": ports_at(asym, inputs=(0,), **kw),
        "sym4 · 2 in (opposite)": symmetric_ports(4, inputs=(0, 2), **kw),
        "asym4 · 2 in": ports_at(asym, inputs=(0, 1), **kw),
        "sym4 · 4 in": symmetric_ports(4, inputs=(0, 1, 2, 3), **kw),
        "sym8 · 1 in": symmetric_ports(8, inputs=(0,), **kw),
    }


def model_for(cell):
    return PhaseScreenModel(cell, [Source(i, **SRC) for i in cell.inputs])


def base_shapes(n, seed=7, place=0.6e-3):
    rng = np.random.default_rng(seed)
    return [random_shape(rng, 1e-3, M_MAX, (60e-6, 100e-6), (1e-3, 1e-3), 3e-6, 0.06, 0.5, place_radius=place)
            for _ in range(n)]


def image_features(perts, n=40, half=0.75e-3):
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    return np.array([p.value(X, Y).ravel() for p in perts])


def targets(D):
    names = D["names"]
    ia = [i for i, s in enumerate(names) if s[0] in "ab"]
    ip = [i for i, s in enumerate(names) if s[0] == "p"]
    return D["Y"][:, ia], D["Y"][:, ip], [names[i] for i in ia], [names[i] for i in ip], D["Y"][:, -1].astype(int)


def per_m(r2_ab, m_max=M_MAX):
    k = m_max - 1
    return ((np.asarray(r2_ab[:k]) + np.asarray(r2_ab[k:2 * k])) / 2).tolist()


def main(quick=False):
    t0 = time.time()
    out = dict(meta=dict(date=time.strftime("%Y-%m-%d"), quick=quick, m=list(range(2, M_MAX + 1)),
                         cell=dict(radius=1e-3, n_eff=1.8, wavelength=1.55e-6, R=0.99, port_width=PORT_W),
                         source=SRC))
    nb = 2 if quick else 5
    N2 = 120 if quick else 400
    N3 = 200 if quick else 800

    # tests
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT)
    last = [l for l in r.stdout.strip().splitlines() if "passed" in l or "failed" in l]
    out["tests"] = dict(summary=last[-1] if last else r.stdout[-200:], ok=r.returncode == 0)
    print("tests:", out["tests"]["summary"], flush=True)

    # E1: launch-angle sweep
    bases = base_shapes(nb)
    launches = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if not quick else [0.0, 0.5, 1.0]
    e1 = dict(launch=launches, S=[], sv=[], occupancy={})
    for la in launches:
        cell = cell_with(symmetric_ports(4, inputs=(0,), width=PORT_W, launch=la, fan=0.4))
        m = model_for(cell)
        S, sv = A.sensitivity(m, bases, M_MAX)
        e1["S"].append(S.tolist())
        e1["sv"].append((sv / sv[0] if sv[0] > 0 else sv).tolist())
        if la in (0.0, 0.4, 0.8, 1.0) or quick:
            e1["occupancy"][f"{la:.1f}"] = np.round(A.occupancy(m, 0, n=64), 3).tolist()
        print(f"E1 launch {la:.1f}: S_m = {np.round(S, 3)}", flush=True)
    e1["bases"] = [b.to_dict() for b in bases]
    out["E1"] = e1

    # E2: layouts
    ens2 = Ensemble(n=N2, m_max=M_MAX, R_range=(60e-6, 110e-6), sigma=0.08, decay=0.5, centre=(0.25e-3, 0.1e-3), seed=11)
    e2 = dict(layouts=[], N=N2)
    for name, ports in layouts().items():
        cell = cell_with(ports)
        m = model_for(cell)
        S, sv = A.sensitivity(m, bases[:2], M_MAX)
        sv = sv if sv[0] == 0 else sv / sv[0]
        D = build(cell, ens2, model=m)
        Yab, Yp, nab, np_, _ = targets(D)
        rr = RO.evaluate_ridge(D["X"], Yab)
        rp = RO.evaluate_ridge(D["X"], Yp)
        e2["layouts"].append(dict(name=name, ports=[dict(angle=float(p.angle), role=p.role) for p in ports],
                                  n_features=int(D["X"].shape[1]), S=S.tolist(), sv=sv.tolist(),
                                  r2_ab=per_m(rr["r2"]), r2_p=rp["r2"].tolist()))
        print(f"E2 {name}: features {D['X'].shape[1]}, S {np.round(S, 3)}, R²(a,b) {np.round(per_m(rr['r2']), 2)}, "
              f"R²(p) {np.round(rp['r2'], 2)}", flush=True)
    out["E2"] = e2
    save(out, t0)

    # E3: readouts on the 2-input asymmetric layout, fixed and random positions
    e3 = dict(N=N3, cases=[])
    for case, ens in (("fixed position", Ensemble(n=N3, m_max=M_MAX, R_range=(60e-6, 110e-6), sigma=0.08, decay=0.5,
                                                  centre=(0.25e-3, 0.1e-3), seed=21)),
                      ("random position", Ensemble(n=N3, m_max=M_MAX, R_range=(60e-6, 110e-6), sigma=0.08, decay=0.5,
                                                   place_radius=0.6e-3, seed=22))):
        cell = cell_with(layouts()["asym4 · 2 in"])
        m = model_for(cell)
        D = build(cell, ens, model=m)
        Yab, Yp, nab, np_, mdom = targets(D)
        X = D["X"]
        Xq = ngrc_features(X, constant=False, n_quad=3000)
        Ximg = image_features(D["perts"])
        res = {}
        res["ridge (linear)"] = per_m(RO.evaluate_ridge(X, Yab)["r2"])
        res["NGRC ridge (lin + quad)"] = per_m(RO.evaluate_ridge(Xq, Yab)["r2"])
        res["linear SVR"] = per_m(RO.evaluate_svr(X, Yab, "linear")["r2"])
        res["RBF SVR"] = per_m(RO.evaluate_svr(X, Yab, "rbf")["r2"])
        res["image → ridge (baseline)"] = per_m(RO.evaluate_ridge(Ximg, Yab)["r2"])
        res["image → RBF SVR (baseline)"] = per_m(RO.evaluate_svr(Ximg, Yab, "rbf")["r2"])
        pw = {}
        pw["ridge (linear)"] = RO.evaluate_ridge(X, Yp)["r2"].tolist()
        pw["NGRC ridge (lin + quad)"] = RO.evaluate_ridge(Xq, Yp)["r2"].tolist()
        pw["RBF SVR"] = RO.evaluate_svr(X, Yp, "rbf")["r2"].tolist()
        cls = dict(linear=RO.evaluate_svc(X, mdom, "linear"), rbf=RO.evaluate_svc(X, mdom, "rbf"),
                   image_rbf=RO.evaluate_svc(Ximg, mdom, "rbf"))
        rr = RO.evaluate_ridge(X, Yab)
        k2 = 0                                     # a2 scatter for the applet
        sc = dict(true=Yab[rr["test"], k2].round(5).tolist(), pred=rr["pred"][:, k2].round(5).tolist(), name=nab[k2])
        e3["cases"].append(dict(name=case, n_features=int(X.shape[1]), r2_ab=res, r2_p=pw,
                                svc={k: dict(accuracy=v["accuracy"], chance=v["chance"]) for k, v in cls.items()},
                                scatter=sc))
        print(f"E3 {case}:", {k: np.round(v, 2).tolist() for k, v in res.items()}, flush=True)
        print("   p_m:", {k: np.round(v, 2).tolist() for k, v in pw.items()}, " SVC:",
              {k: round(v["accuracy"], 2) for k, v in cls.items()}, flush=True)
    out["E3"] = e3
    save(out, t0)

    # E4: validity of the phase-screen model and decorrelation vs Δn
    cell = cell_with(symmetric_ports(4, inputs=(0,), width=PORT_W, launch=0.35, fan=0.4))
    src = Source(0, n_pos=3, n_ang=31)
    pert = Perturbation([Shape(2.5e-4, 1e-4, 80e-6, 1.0, 3e-6, a=[0, 0.06, 0.03], b=[0, 0.0, 0.0, 0.02])])
    scales = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3] if quick else [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    ms = PhaseScreenModel(cell, [src])
    val = A.validity_scan(cell, src, pert, scales, model=ms)
    dec = A.decorrelation(ms, pert, np.logspace(-5, -2, 13))
    out["E4"] = dict(dn=scales, corr_phase_vs_curved=val.tolist(), dn_dec=np.logspace(-5, -2, 13).tolist(),
                     corr_vs_unperturbed=dec.tolist())
    print("E4 phase-screen vs curved:", dict(zip(scales, np.round(val, 3))), flush=True)
    print("E4 decorrelation:", np.round(dec, 3), flush=True)

    save(out, t0)
    print(f"done in {out['meta']['runtime_s']} s -> results/progress.json")


def save(out, t0):
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "progress.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
