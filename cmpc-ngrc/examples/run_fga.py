"""Experiments with the frozen-Gaussian (Herman–Kluk) model; datasets through the JS engine (node, all cores).

Writes results/progress.json with the keys the explorer applet charts (E1–E12); the earlier results from the
Gaussian-beam-summation model (wide beamlets, hard ports) are kept in results/progress_gbs.json.

  E1  sensitivity per order vs launch angle            E7  learnability: curved rays vs phase screen
  E2  port layouts                                     E8  launch-angle / wavelength multiplexing
  E3  readouts (ridge, NGRC, SVR, SVC; image baseline) E9  FGA vs BPM, one pass through a dot
  E4  phase screen vs curved rays, decorrelation       E10 stadium vs circle (Python engine)
  E5  position tolerance and jitter                    E11 detector noise and temperature drift
  E6  learning curves                                  E12 readout quality vs Δn

python examples/run_fga.py [--only E1,E2] [--quick]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT.parent / "math-engines"))

from ngrc import readout as RO                                       # noqa: E402
from ngrc.dataset import Ensemble, labels, sample_shapes             # noqa: E402
from ngrc.features import ngrc_features, stack_intensities          # noqa: E402
from ngrc.field import field_correlation                             # noqa: E402
from ngrc.geometry import CircularCell, ports_at, symmetric_ports    # noqa: E402
from ngrc.jsengine import node_fields                                # noqa: E402
from ngrc.shapes import Perturbation, Shape, random_shape            # noqa: E402
from run_progress import image_features, per_m                       # noqa: E402

M_MAX = 6
PORT_W = 20e-6
R_WALL = 0.97
FIXED = (0.25e-3, 0.1e-3)
FGA = dict(frozen=20e-6, n_ang=6000, amp_min=0.02, max_bounces=300)
FGA_CURVED = dict(frozen=20e-6, n_ang=1500, amp_min=0.02, max_bounces=300)
CACHE = ROOT / "results" / "cache_fga"
ASYM = [0, 67, 151, 238]


def cell_with(ports, **kw):
    return CircularCell(radius=1e-3, n_eff=1.80, wavelength=1.55e-6, ports=ports, reflectance=R_WALL, **kw)


def layouts(launch=0.35, fan=0.4):
    kw = dict(width=PORT_W, launch=launch, fan=fan)
    return {
        "sym4 · 1 in": symmetric_ports(4, inputs=(0,), **kw),
        "asym4 · 1 in": ports_at(ASYM, inputs=(0,), **kw),
        "sym4 · 2 in (opposite)": symmetric_ports(4, inputs=(0, 2), **kw),
        "asym4 · 2 in": ports_at(ASYM, inputs=(0, 1), **kw),
        "sym4 · 4 in": symmetric_ports(4, inputs=(0, 1, 2, 3), **kw),
        "sym8 · 1 in": symmetric_ports(8, inputs=(0,), **kw),
    }


def ens(n, seed, centre=None, place=0.6e-3, dn=1e-3, jitter=0.0):
    return Ensemble(n=n, m_max=M_MAX, R_range=(60e-6, 110e-6), dn_range=(dn, dn), sigma=0.08, decay=0.5,
                    centre=centre, place_radius=None if centre else place, jitter=jitter, seed=seed)


def fields(cell, perts, mode="phase", cfg=FGA, dn_global=None):
    return node_fields(cell, perts, mode=mode, dn_global=dn_global, **cfg)


def dataset(cell, e, mode="phase", cfg=FGA, tag=None):
    """Cached features (relative intensities), labels, positions and image features."""
    key = json.dumps([tag or cell.to_dict(), e.key(), mode, cfg], sort_keys=True, default=str)
    f = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    perts = sample_shapes(cell, e)
    F = fields(cell, [None] + perts, mode, cfg)
    ref = F[0]
    X = np.array([stack_intensities(x, reference=ref) for x in F[1:]])
    Y, names = labels(perts, M_MAX)
    ia = [i for i, s in enumerate(names) if s[0] in "ab"]
    ip = [i for i, s in enumerate(names) if s[0] == "p"]
    I0 = np.concatenate([np.abs(ref[k]) ** 2 for k in sorted(ref)])
    out = dict(X=X, Yab=Y[:, ia], Yp=Y[:, ip], mdom=Y[:, -1], I0=I0,
               pos=np.array([[p.components[0].x0, p.components[0].y0] for p in perts]),
               img=image_features(perts), seconds=np.array(time.time() - t0))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"  dataset {mode} n={e.n}: {time.time() - t0:.0f} s", flush=True)
    return out


def split(n, n_test, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[n_test:], idx[:n_test]


def fit_ridge(X, Y, tr):
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-3 * X[tr].std(0).mean())
    a, _ = RO.ridge_cv((X[tr] - mu) / sd, Y[tr])
    W, b = RO.ridge_fit((X[tr] - mu) / sd, Y[tr], a)
    return lambda Z: ((Z - mu) / sd) @ W + b


def base_shapes(n, seed=7, place=0.6e-3):
    rng = np.random.default_rng(seed)
    return [random_shape(rng, 1e-3, M_MAX, (60e-6, 100e-6), (1e-3, 1e-3), 3e-6, 0.06, 0.5, place_radius=place)
            for _ in range(n)]


def sensitivity(cell, bases, h=0.01):
    perts, cols = [None], []
    for base in bases:
        for m in range(2, M_MAX + 1):
            for coef in ("a", "b"):
                for sgn in (1, -1):
                    s = copy.deepcopy(base)
                    arr = getattr(s, coef).copy()
                    arr[m - 1] += sgn * h
                    setattr(s, coef, arr)
                    perts.append(Perturbation([s]))
    F = fields(cell, perts)
    keys = sorted(F[0])
    I = lambda f: np.concatenate([np.abs(f[k]) ** 2 for k in keys])
    I0 = I(F[0]).mean()
    S, k = [], 1
    for _ in bases:
        J = []
        for m in range(2, M_MAX + 1):
            for coef in ("a", "b"):
                J.append((I(F[k]) - I(F[k + 1])) / (2 * h) / I0)
                k += 2
        J = np.array(J).T
        S.append(np.sqrt((J[:, 0::2] ** 2 + J[:, 1::2] ** 2).mean(axis=0) / 2))
    return np.mean(S, axis=0)


def occupancy(cell, n=64):
    """Geometric ray density (gbs tracer, unperturbed chords) for the applet's coverage maps."""
    from ngrc.perturbative import PhaseScreenModel
    from ngrc.analysis import occupancy as occ
    from ngrc.rays import Source
    g = copy.deepcopy(cell)
    g.model = "gbs"
    return occ(PhaseScreenModel(g, [Source(cell.inputs[0], n_pos=3, n_ang=61)]), 0, n=n)


# ------------------------------------------------------------------ experiments
def e1(q):
    bases = base_shapes(2 if q else 4)
    out = dict(launch=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], S=[], occupancy={})
    for la in out["launch"]:
        cell = cell_with(symmetric_ports(4, inputs=(0,), width=PORT_W, launch=la, fan=0.4))
        S = sensitivity(cell, bases)
        out["S"].append(S.tolist())
        if la in (0.0, 0.4, 0.8, 1.0):
            out["occupancy"][f"{la:.1f}"] = np.round(occupancy(cell), 3).tolist()
        print(f"E1 launch {la:.1f}: S {np.round(S, 3)}", flush=True)
    return out


def e2(q):
    N = 200 if q else 800
    out = dict(N=N, layouts=[])
    bases = base_shapes(2)
    for name, ports in layouts().items():
        cell = cell_with(ports)
        D = dataset(cell, ens(N, 11, centre=FIXED))
        tr, te = split(N, N // 4)
        r = RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)
        rp = RO.evaluate_krr(D["X"], D["Yp"], "poly2", train_idx=tr, test_idx=te)
        S = sensitivity(cell, bases)
        out["layouts"].append(dict(name=name, n_features=int(D["X"].shape[1]), S=S.tolist(), r2_ab=per_m(r["r2"]),
                                   r2_p=rp["r2"].tolist()))
        print(f"E2 {name}: R² {np.round(per_m(r['r2']), 2)}, p {np.round(rp['r2'], 2)}, S {np.round(S, 2)}", flush=True)
    return out


def e3(q):
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    out = dict(N=N, cases=[])
    for case, e in (("fixed position", ens(N, 41, centre=FIXED)), ("random position", ens(N, 31))):
        D = dataset(cell, e)
        X, Yab, Yp, Ximg, mdom = D["X"], D["Yab"], D["Yp"], D["img"], D["mdom"].astype(int)
        tr, te = split(N, N // 4)
        kw = dict(train_idx=tr, test_idx=te)
        res = {"ridge (linear)": per_m(RO.evaluate_krr(X, Yab, "linear", **kw)["r2"]),
               "NGRC ridge (lin + quad)": per_m(RO.evaluate_krr(X, Yab, "poly2", **kw)["r2"]),
               "linear SVR": per_m(RO.evaluate_svr(X, Yab, "linear", seed=0)["r2"]),
               "RBF SVR": per_m(RO.evaluate_svr(X, Yab, "rbf", seed=0)["r2"]),
               "image → ridge (baseline)": per_m(RO.evaluate_krr(Ximg, Yab, "linear", **kw)["r2"]),
               "image → RBF SVR (baseline)": per_m(RO.evaluate_svr(Ximg, Yab, "rbf", seed=0)["r2"])}
        pw = {"ridge (linear)": RO.evaluate_krr(X, Yp, "linear", **kw)["r2"].tolist(),
              "NGRC ridge (lin + quad)": RO.evaluate_krr(X, Yp, "poly2", **kw)["r2"].tolist(),
              "RBF SVR": RO.evaluate_svr(X, Yp, "rbf", seed=0)["r2"].tolist()}
        cls = dict(linear=RO.evaluate_svc(X, mdom, "linear"), rbf=RO.evaluate_svc(X, mdom, "rbf"),
                   image_rbf=RO.evaluate_svc(Ximg, mdom, "rbf"))
        rr = RO.evaluate_krr(X, Yab, "linear", **kw)
        out["cases"].append(dict(name=case, n_features=int(X.shape[1]), r2_ab=res, r2_p=pw,
                                 svc={k: dict(accuracy=v["accuracy"], chance=v["chance"]) for k, v in cls.items()},
                                 scatter=dict(true=Yab[te, 0].round(5).tolist(), pred=rr["pred"][:, 0].round(5).tolist(),
                                              name="a2")))
        print(f"E3 {case}:", {k: np.round(v, 2).tolist() for k, v in res.items()}, flush=True)
        print("   p:", {k: np.round(v, 2).tolist() for k, v in pw.items()}, "SVC", {k: round(v["accuracy"], 2) for k, v in cls.items()}, flush=True)
    return out


def e4(q):
    cell = cell_with(layouts()["asym4 · 2 in"])
    base = Shape(FIXED[0], FIXED[1], 80e-6, 1.0, 3e-6, a=[0, 0.06, 0.03], b=[0, 0, 0, 0.02])
    dns = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    perts = []
    for dn in dns:
        s = copy.deepcopy(base)
        s.dn = dn
        perts.append(Perturbation([s]))
    Fp = fields(cell, [None] + perts, "phase", FGA_CURVED)
    Fc = fields(cell, [None] + perts, "curved", FGA_CURVED)
    keys = sorted(Fp[0])
    cv = [float(np.mean([field_correlation(Fp[i + 1][k], Fc[i + 1][k]) for k in keys])) for i in range(len(dns))]
    dec = [float(np.mean([field_correlation(Fp[0][k], Fp[i + 1][k]) for k in keys])) for i in range(len(dns))]
    decc = [float(np.mean([field_correlation(Fc[0][k], Fc[i + 1][k]) for k in keys])) for i in range(len(dns))]
    print("E4 phase vs curved:", np.round(cv, 4), "decorrelation phase", np.round(dec, 3), "curved", np.round(decc, 3), flush=True)
    return dict(dn=dns, corr_phase_vs_curved=cv, dn_dec=dns, corr_vs_unperturbed=dec, corr_vs_unperturbed_curved=decc)


def e5(q):
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    e = ens(N, 41, centre=FIXED)
    D = dataset(cell, e)
    tr, te = split(N, N // 4)
    pred = fit_ridge(D["X"], D["Yab"], tr)
    perts = sample_shapes(cell, e)
    te_s = te[:200]
    rng = np.random.default_rng(5)
    shifts = [0, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6, 10e-6, 20e-6]
    batch, ref_idx = [None], []
    for d in shifts:
        for i in te_s:
            p = copy.deepcopy(perts[i])
            ph = rng.uniform(0, 2 * np.pi)
            for c in p.components:
                c.x0 += d * np.cos(ph)
                c.y0 += d * np.sin(ph)
            batch.append(p)
    F = fields(cell, batch)
    ref = F[0]
    out = dict(N=N, shift=dict(delta_um=[], r2=[]))
    for k, d in enumerate(shifts):
        Xd = np.array([stack_intensities(F[1 + k * len(te_s) + j], reference=ref) for j in range(len(te_s))])
        r = RO.r2(D["Yab"][te_s], pred(Xd))
        out["shift"]["delta_um"].append(d * 1e6)
        out["shift"]["r2"].append(per_m(r))
        print(f"E5 shift {d * 1e6:.1f} µm: {np.round(per_m(r), 2)}", flush=True)
    Nj = max(200, N // 2)
    jit = dict(N=Nj, sigma_um=[], r2=[])
    for sg in (0, 2e-6, 5e-6, 10e-6, 20e-6):
        Dj = dataset(cell, ens(Nj, 71, centre=FIXED, jitter=sg))
        trj, tej = split(Nj, Nj // 4)
        r = RO.evaluate_krr(Dj["X"], Dj["Yab"], "linear", train_idx=trj, test_idx=tej)
        jit["sigma_um"].append(sg * 1e6)
        jit["r2"].append(per_m(r["r2"]))
        print(f"E5 jitter {sg * 1e6:.0f} µm: {np.round(per_m(r['r2']), 2)}", flush=True)
    out["jitter"] = jit
    Dr = dataset(cell, ens(N, 31))
    trr, ter = split(N, N // 4)
    out["anywhere"] = dict(linear=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "linear", train_idx=trr, test_idx=ter)["r2"]),
                           rbf=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "rbf", train_idx=trr, test_idx=ter)["r2"]),
                           linear_pos=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "linear", pos=Dr["pos"], train_idx=trr,
                                                            test_idx=ter)["r2"]))
    print("E5 anywhere:", {k: np.round(v, 2).tolist() for k, v in out["anywhere"].items()}, flush=True)
    return out


def e6(q):
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    out = dict(cases=[])
    for name, e, kind, pos in (("fixed position · ridge", ens(N, 41, centre=FIXED), "linear", False),
                               ("fixed position · NGRC poly2", ens(N, 41, centre=FIXED), "poly2", False),
                               ("random position · linear ⊗ position", ens(N, 31), "linear", True)):
        D = dataset(cell, e)
        tr_all, te = split(N, N // 4)
        ns = [n for n in (100, 200, 400, 800, 1200) if n <= len(tr_all)]
        curves = [per_m(RO.evaluate_krr(D["X"], D["Yab"], kind, pos=D["pos"] if pos else None, train_idx=tr_all[:n],
                                        test_idx=te)["r2"]) for n in ns]
        out["cases"].append(dict(name=name, n_train=ns, r2=curves))
        print(f"E6 {name}: " + ", ".join(f"{n}: {np.mean(c):.2f}" for n, c in zip(ns, curves)), flush=True)
    return out


def e7_e12(q):
    """Readout quality vs Δn with the phase screen (6000 angles) and, at the same 1500-angle ray set, phase screen
    vs curved rays; with and without 1 % detector noise."""
    cell = cell_with(layouts()["asym4 · 2 in"])
    N = 160 if q else 400
    Nc = 120 if q else 240
    dns = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    rng = np.random.default_rng(9)
    e12 = dict(N=N, dn=dns, sigma=0.01)

    def evals(D, n):
        tr, te = split(n, n // 4)
        clean = per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        Xn = D["X"] + rng.normal(0, 1, D["X"].shape) * 0.01 * D["I0"].mean() / D["I0"]
        noisy = per_m(RO.evaluate_krr(Xn, D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        poly = per_m(RO.evaluate_krr(D["X"], D["Yab"], "poly2", train_idx=tr, test_idx=te)["r2"])
        return clean, noisy, poly

    rows = dict(clean=[], noisy=[])
    for dn in dns:
        c, n_, _ = evals(dataset(cell, ens(N, 91, centre=FIXED, dn=dn)), N)
        rows["clean"].append(c)
        rows["noisy"].append(n_)
        print(f"E12 phase Δn={dn:g}: clean {np.round(c, 2)}, noisy {np.round(n_, 2)}", flush=True)
    e12["phase"] = rows
    e7 = dict(N=Nc, cases=[])
    crow = dict(clean=[], noisy=[])
    for dn in (1e-4, 1e-3):
        row = dict(name=f"Δn = {dn:g}, fixed position")
        for mode in ("phase", "curved"):
            D = dataset(cell, ens(Nc, 93, centre=FIXED, dn=dn), mode, FGA_CURVED)
            c, n_, p_ = evals(D, Nc)
            row["node-curved" if mode == "curved" else "phase"] = dict(linear=c, poly2=p_, noisy=n_, seconds=float(D["seconds"]))
            if mode == "curved":
                crow["clean"].append(c)
                crow["noisy"].append(n_)
        e7["cases"].append(row)
        print(f"E7 Δn={dn:g}: phase {np.round(row['phase']['linear'], 2)} / curved {np.round(row['node-curved']['linear'], 2)}", flush=True)
    e12["curved_dn"] = [1e-4, 1e-3]
    e12["node-curved"] = crow
    return e7, e12


def e8(q):
    N = 300 if q else 800
    cell = cell_with(ports_at(ASYM, inputs=(0,), width=PORT_W, launch=0.35, fan=0.4))
    lam = cell.wavelength
    e = ens(N, 61, centre=FIXED)
    sets = {"1 input, 1 launch": [dict()],
            "3 launch angles (±6°)": [dict(launch_offset=d) for d in (-0.1, 0.0, 0.1)],
            "3 wavelengths (±0.2 nm)": [dict(wavelength=lam + d) for d in (-0.2e-9, 0.0, 0.2e-9)],
            "3 × 3 launch × wavelength": [dict(launch_offset=a, wavelength=lam + d) for a in (-0.1, 0.0, 0.1)
                                          for d in (-0.2e-9, 0.0, 0.2e-9)]}
    out = dict(N=N, sets=[])
    F0 = fields(cell, [None])[0]
    dec = []
    for dl in (0.05e-9, 0.1e-9, 0.2e-9, 0.5e-9):
        Fl = fields(cell.variant(wavelength=lam + dl), [None])[0]
        dec.append([dl * 1e9, float(np.mean([field_correlation(F0[k], Fl[k]) for k in F0]))])
    out["lambda_decorrelation"] = dec
    for name, vs in sets.items():
        X = np.hstack([dataset(cell.variant(**v), e)["X"] for v in vs])
        Y = dataset(cell.variant(**vs[0]), e)["Yab"]
        tr, te = split(N, N // 4)
        r = RO.evaluate_krr(X, Y, "linear", train_idx=tr, test_idx=te)
        out["sets"].append(dict(name=name, n_features=int(X.shape[1]), r2=per_m(r["r2"])))
        print(f"E8 {name}: {X.shape[1]} features, R² {np.round(per_m(r['r2']), 2)}", flush=True)
    D2 = dataset(cell_with(ports_at(ASYM, inputs=(0, 1), width=PORT_W, launch=0.35, fan=0.4)), e)
    tr, te = split(N, N // 4)
    r = RO.evaluate_krr(D2["X"], D2["Yab"], "linear", train_idx=tr, test_idx=te)
    out["sets"].append(dict(name="2 inputs (reference)", n_features=int(D2["X"].shape[1]), r2=per_m(r["r2"])))
    print("E8 λ decorrelation", dec, flush=True)
    return out


def e9(q):
    """FGA (curved and straight) vs split-step BPM for one pass across a 1 mm cell through a dot."""
    from ngrc.field import beamlet_matrix
    from ngrc.geometry import Port
    from ngrc.rays import Source, trace
    from ngrc.wave_ref import bpm
    Rc = 1e-3
    c = CircularCell(radius=Rc, ports=[Port(0.0, width=600e-6, role="in", launch=0.0, fan=0.06),
                                       Port(np.pi, width=600e-6, role="out")], reflectance=0.0)
    lam, n0 = c.wavelength, c.n_eff
    w_in = lam / (np.pi * n0 * np.tan(0.03))
    yobs = np.linspace(-150e-6, 150e-6, 301)
    pts = np.c_[np.full_like(yobs, -Rc), yobs]
    y = (np.arange(8192) - 4096) * (2e-3 / 8192)
    rows, E0 = [], None
    for dn in (0.0, 1e-3, 3e-3, 1e-2):
        p = Perturbation([Shape(0.4e-3, 8e-6, 50e-6, dn, 3e-6, a=[0, 0.05, 0.03])]) if dn else None
        Eb = bpm(lambda z, yy: (p.value(np.full_like(yy, Rc - z), yy) if p else 0 * yy), n0, lam, y, 2 * Rc,
                 np.exp(-(y / w_in) ** 2), dz=2e-6)
        Eb = np.interp(yobs, y, Eb.real) + 1j * np.interp(yobs, y, Eb.imag)
        out = {}
        for mode in ("curved", "straight"):
            ex = trace(c, Source(0, n_pos=None, n_ang=2001, frozen=6e-6), p, mode, max_bounces=1)
            out[mode] = beamlet_matrix(c, ex, 1, points=pts)[0].sum(1)
        if E0 is None:
            E0 = (Eb, out)
            continue
        sb = Eb - E0[0]
        r = dict(z_mm=1.4, dn=dn, scat_fraction=float(np.linalg.norm(sb) / np.linalg.norm(Eb)))
        for m in ("curved", "straight"):
            sm = out[m] - E0[1][m]
            r[f"scat_{m}"] = field_correlation(sb, sm)
            a = np.vdot(sm, sb) / np.vdot(sm, sm)                  # best complex scale (normalisations differ)
            r[f"err_{m}"] = float(np.linalg.norm(a * sm - sb) / np.linalg.norm(sb))
        rows.append(r)
        print(f"E9 Δn={dn:g}: |ΔE|/|E| {r['scat_fraction']:.3f}, curved {r['scat_curved']:.4f} ({r['err_curved']:.3f}), "
              f"phase screen {r['scat_straight']:.4f} ({r['err_straight']:.3f})", flush=True)
    return dict(rows=rows, beam_w_um=float(w_in * 1e6), dot=dict(R_um=50, edge_um=3, z_um=600), model="fga")


def e10(q):
    """Stadium vs circle with the Python FGA phase screen (the JS engine has the circle only)."""
    from ngrc.dataset import build
    from ngrc.geometry import WallCell, stadium, wall_ports
    from ngrc.perturbative import PhaseScreenModel
    from ngrc.rays import Source
    N = 200 if q else 600
    src = lambda i: Source(i, n_pos=None, n_ang=4000 if not q else 2000, frozen=20e-6)
    st = stadium(0.6e-3, 1.2e-3)
    cells = {"circle": cell_with(layouts()["asym4 · 2 in"]),
             "stadium": WallCell(wall=st, n_eff=1.8, wavelength=1.55e-6, reflectance=R_WALL,
                                 ports=wall_ports(st, [0.05, 0.27, 0.52, 0.78], inputs=(0, 1), width=PORT_W, launch=0.35,
                                                  fan=0.4))}
    rows = []
    for name, cell in cells.items():
        f = CACHE / f"e10_{name}_{N}.npz"
        if f.exists():
            z = np.load(f)
            X, Yab, Yp = z["X"], z["Yab"], z["Yp"]
        else:
            t0 = time.time()
            m = PhaseScreenModel(cell, [src(i) for i in cell.inputs], max_bounces=300, amp_min=0.02)
            D = build(cell, ens(N, 81, centre=(0.15e-3, 0.1e-3)), model=m)
            names = D["names"]
            Yab = D["Y"][:, [i for i, s in enumerate(names) if s[0] in "ab"]]
            Yp = D["Y"][:, [i for i, s in enumerate(names) if s[0] == "p"]]
            X = D["X"]
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, X=X, Yab=Yab, Yp=Yp)
            print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
        tr, te = split(N, N // 4)
        r = RO.evaluate_krr(X, Yab, "linear", train_idx=tr, test_idx=te)
        rp = RO.evaluate_krr(X, Yp, "poly2", train_idx=tr, test_idx=te)
        rows.append(dict(name=name, r2=per_m(r["r2"]), r2_p=rp["r2"].tolist(), mean_bounces=0.0))
        print(f"E10 {name}: R² {np.round(per_m(r['r2']), 2)}, p {np.round(rp['r2'], 2)}", flush=True)
    return dict(N=N, readout=rows, sensitivity=None)


def e11(q):
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    e = ens(N, 41, centre=FIXED)
    D = dataset(cell, e)
    X, Y, I0 = D["X"], D["Yab"], D["I0"]
    tr, te = split(N, N // 4)
    pred = fit_ridge(X, Y, tr)
    rng = np.random.default_rng(3)
    noise = dict(sigma=[], clean_trained=[], noise_trained=[])
    for sg in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
        Xn = X + rng.normal(0, 1, X.shape) * sg * I0.mean() / I0
        noise["sigma"].append(sg)
        noise["clean_trained"].append(per_m(RO.r2(Y[te], pred(Xn[te]))))
        noise["noise_trained"].append(per_m(RO.evaluate_krr(Xn, Y, "linear", train_idx=tr, test_idx=te)["r2"]))
        print(f"E11 noise {sg}: {np.round(noise['clean_trained'][-1], 2)} / {np.round(noise['noise_trained'][-1], 2)}", flush=True)
    perts = sample_shapes(cell, e)
    te_s = te[:150]
    dns = [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
    batch = [None] + [None] * len(dns) + [perts[i] for _ in dns for i in te_s]
    dn_list = [0.0] + dns + [d for d in dns for _ in te_s]
    F = fields(cell, batch, dn_global=dn_list)
    ref = F[0]
    keys = sorted(ref)
    drift = dict(dn=dns, r2=[], corr_ref=[])
    for k, d in enumerate(dns):
        Xd = np.array([stack_intensities(F[1 + len(dns) + k * len(te_s) + j], reference=ref) for j in range(len(te_s))])
        drift["r2"].append(per_m(RO.r2(Y[te_s], pred(Xd))))
        drift["corr_ref"].append(float(np.mean([field_correlation(ref[q], F[1 + k][q]) for q in keys])))
        print(f"E11 drift {d:g}: {np.round(drift['r2'][-1], 2)}, speckle corr {drift['corr_ref'][-1]:.3f}", flush=True)
    return dict(N=N, noise=noise, drift=drift, mean_path_mm=None)


EXPS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E8": e8, "E9": e9, "E10": e10, "E11": e11}


def main(only, quick):
    pj = ROOT / "results" / "progress.json"
    t0 = time.time()
    order = ["E9", "E4", "E1", "E3", "E6", "E5", "E11", "E2", "E7", "E8", "E10"]
    for name in order:
        if only and name not in only and not (name == "E7" and "E12" in only):
            continue
        if name == "E7":
            r7, r12 = e7_e12(quick)
            res = {"E7": r7, "E12": r12}
        else:
            res = {name: EXPS[name](quick)}
        cur = json.loads(pj.read_text()) if pj.exists() else {}
        cur.update(res)
        meta = cur.setdefault("meta", {})
        meta.update(model="fga", date=time.strftime("%Y-%m-%d"), quick=quick, m=list(range(2, M_MAX + 1)),
                    cell=dict(radius=1e-3, n_eff=1.8, wavelength=1.55e-6, R=R_WALL, port_width=PORT_W),
                    source=dict(n_pos="auto", n_ang=FGA["n_ang"], frozen_um=20), runtime_s=round(time.time() - t0, 1))
        pj.write_text(json.dumps(cur, indent=1))
        print(f"[{name} saved, {time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main([s.strip() for s in a.only.split(",") if s.strip()] or None, a.quick)
