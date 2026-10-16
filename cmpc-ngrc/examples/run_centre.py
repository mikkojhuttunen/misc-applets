"""Single dot at the cell centre (decision D9): the experiments of run_fga.py redone for a dot at (0, 0).

A centred dot is only reached by rays with R_c |sin χ| < dot radius, so the launch angle must be near 0 (E1 here
sweeps launch and fan). Writes the same keys as run_fga.py into results/progress.json; the earlier off-centre
results (dot at (250, 100) µm) are archived in results/progress_offcentre.json, and the experiments that are
about position, curved rays, BPM or other cells (E4, E5, E7, E7b, E9, E10) are carried over from there.

python examples/run_centre.py [--only E1,E3] [--quick] [--launch 0.05] [--fan 0.4]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import run_fga as F                                                   # noqa: E402
from run_fga import (ASYM, FGA, M_MAX, PORT_W, R_WALL, RO, dataset, ens, field_correlation, fields,  # noqa: E402
                     fit_ridge, occupancy, per_m, sample_shapes, sensitivity, split, stack_intensities)
from run_progress import image_features                         # noqa: E402
from ngrc.geometry import ports_at, symmetric_ports                  # noqa: E402
from ngrc.shapes import Perturbation, Shape, random_shape                   # noqa: E402

CENTRE = (0.0, 0.0)
LAUNCH, FAN = 0.05, 0.2
DN = 3e-5                    # every low-angle chord crosses a centred dot: Δn = 1e-3 accumulates tens of radians (E12)
CARRY = ("E4", "E5", "E7", "E7b", "E9", "E10")


def cell_with(ports, **kw):
    return F.cell_with(ports, **kw)


def layouts(launch=None, fan=None):
    kw = dict(width=PORT_W, launch=LAUNCH if launch is None else launch, fan=FAN if fan is None else fan)
    return {
        "sym4 · 1 in": symmetric_ports(4, inputs=(0,), **kw),
        "asym4 · 1 in": ports_at(ASYM, inputs=(0,), **kw),
        "sym4 · 2 in (opposite)": symmetric_ports(4, inputs=(0, 2), **kw),
        "sym4 · 2 in (adjacent)": symmetric_ports(4, inputs=(0, 1), **kw),
        "asym4 · 2 in": ports_at(ASYM, inputs=(0, 1), **kw),
        "sym4 · 4 in": symmetric_ports(4, inputs=(0, 1, 2, 3), **kw),
        "sym8 · 1 in": symmetric_ports(8, inputs=(0,), **kw),
        "asym8 · 2 in": ports_at([0, 41, 97, 133, 170, 218, 262, 311], inputs=(0, 3), **kw),
        "asym4 · 2 in + 2 exits opposite": ports_at([0, 67, 151, 238, 180 + 2 * np.degrees(kw["launch"]),
                                                     247 + 2 * np.degrees(kw["launch"])], inputs=(0, 1), **kw),
    }


def ensc(n, seed, dn=None, jitter=0.0):
    dn = DN if dn is None else dn
    return ens(n, seed, centre=CENTRE, dn=dn, jitter=jitter)


def base_shapes(n, seed=7):
    rng = np.random.default_rng(seed)
    return [random_shape(rng, 1e-3, M_MAX, (60e-6, 100e-6), (DN, DN), 3e-6, 0.06, 0.5, centre=CENTRE)
            for _ in range(n)]


def e1(q):
    bases = base_shapes(2 if q else 4)
    launch = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    fans = [0.2, 0.4]
    out = dict(launch=launch, fans=fans, S=[], S_fan={}, occupancy={}, layout="asym4 · 1 in", centre=CENTRE)
    for fan in fans:
        rows = []
        for la in launch:
            cell = cell_with(ports_at(ASYM, inputs=(0,), width=PORT_W, launch=la, fan=fan))
            S = sensitivity(cell, bases)
            rows.append(S.tolist())
            if fan == 0.4 and la in (0.0, 0.05, 0.1, 0.3):
                out["occupancy"][f"{la:.2f}"] = np.round(occupancy(cell), 3).tolist()
            print(f"E1 fan {fan:.1f} launch {la:.2f}: S {np.round(S, 3)}", flush=True)
        out["S_fan"][f"{fan:.1f}"] = rows
    out["S"] = out["S_fan"]["0.4"]
    return out


def e2(q):
    N = 200 if q else 800
    out = dict(N=N, layouts=[], launch=LAUNCH, fan=FAN)
    bases = base_shapes(2)
    for name, ports in layouts().items():
        cell = cell_with(ports)
        D = dataset(cell, ensc(N, 11))
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
    D = dataset(cell, ensc(N, 41))
    X, Yab, Yp, Ximg, mdom = D["X"], D["Yab"], D["Yp"], D["img"], D["mdom"].astype(int)
    Xzoom = image_features(sample_shapes(cell, ensc(N, 41)), n=40, half=150e-6)
    tr, te = split(N, N // 4)
    kw = dict(train_idx=tr, test_idx=te)
    res = {"ridge (linear)": per_m(RO.evaluate_krr(X, Yab, "linear", **kw)["r2"]),
           "NGRC ridge (lin + quad)": per_m(RO.evaluate_krr(X, Yab, "poly2", **kw)["r2"]),
           "linear SVR": per_m(RO.evaluate_svr(X, Yab, "linear", seed=0)["r2"]),
           "RBF SVR": per_m(RO.evaluate_svr(X, Yab, "rbf", seed=0)["r2"]),
           "image → ridge (baseline)": per_m(RO.evaluate_krr(Ximg, Yab, "linear", **kw)["r2"]),
           "image → RBF SVR (baseline)": per_m(RO.evaluate_svr(Ximg, Yab, "rbf", seed=0)["r2"]),
           "zoomed image → ridge (ideal camera)": per_m(RO.evaluate_krr(Xzoom, Yab, "linear", **kw)["r2"])}
    pw = {"ridge (linear)": RO.evaluate_krr(X, Yp, "linear", **kw)["r2"].tolist(),
          "NGRC ridge (lin + quad)": RO.evaluate_krr(X, Yp, "poly2", **kw)["r2"].tolist(),
          "RBF SVR": RO.evaluate_svr(X, Yp, "rbf", seed=0)["r2"].tolist()}
    cls = dict(linear=RO.evaluate_svc(X, mdom, "linear"), rbf=RO.evaluate_svc(X, mdom, "rbf"),
               image_rbf=RO.evaluate_svc(Ximg, mdom, "rbf"))
    rr = RO.evaluate_krr(X, Yab, "linear", **kw)
    case = dict(name="centred dot", n_features=int(X.shape[1]), r2_ab=res, r2_p=pw,
                svc={k: dict(accuracy=v["accuracy"], chance=v["chance"]) for k, v in cls.items()},
                scatter=dict(true=Yab[te, 0].round(5).tolist(), pred=rr["pred"][:, 0].round(5).tolist(), name="a2"))
    print("E3 centre:", {k: np.round(v, 2).tolist() for k, v in res.items()}, flush=True)
    print("   p:", {k: np.round(v, 2).tolist() for k, v in pw.items()}, "SVC", {k: round(v["accuracy"], 2) for k, v in cls.items()}, flush=True)
    return dict(N=N, cases=[case])


def e6(q):
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    D = dataset(cell, ensc(N, 41))
    tr_all, te = split(N, N // 4)
    ns = [n for n in (100, 200, 400, 800, 1200) if n <= len(tr_all)]
    out = dict(cases=[])
    for name, kind in (("centred dot · ridge", "linear"), ("centred dot · NGRC poly2", "poly2")):
        curves = [per_m(RO.evaluate_krr(D["X"], D["Yab"], kind, train_idx=tr_all[:n], test_idx=te)["r2"]) for n in ns]
        out["cases"].append(dict(name=name, n_train=ns, r2=curves))
        print(f"E6 {name}: " + ", ".join(f"{n}: {np.mean(c):.2f}" for n, c in zip(ns, curves)), flush=True)
    return out


def e12(q):
    cell = cell_with(layouts()["asym4 · 2 in"])
    N = 160 if q else 400
    dns = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    rng = np.random.default_rng(9)
    rows = dict(clean=[], noisy=[])
    for dn in dns:
        D = dataset(cell, ensc(N, 91, dn=dn))
        tr, te = split(N, N // 4)
        c = per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        Xn = D["X"] + rng.normal(0, 1, D["X"].shape) * 0.01 * D["I0"].mean() / D["I0"]
        n_ = per_m(RO.evaluate_krr(Xn, D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        rows["clean"].append(c)
        rows["noisy"].append(n_)
        print(f"E12 Δn={dn:g}: clean {np.round(c, 2)}, noisy {np.round(n_, 2)}", flush=True)
    return dict(N=N, dn=dns, sigma=0.01, phase=rows, curved_dn=[], **{"node-curved": dict(clean=[], noisy=[])})


def e8(q):
    N = 300 if q else 800
    cell = cell_with(ports_at(ASYM, inputs=(0,), width=PORT_W, launch=LAUNCH, fan=FAN))
    lam = cell.wavelength
    e = ensc(N, 61)
    sets = {"1 input, 1 launch": [dict()],
            "3 launch angles (±2°)": [dict(launch_offset=d) for d in (-0.035, 0.0, 0.035)],
            "3 wavelengths (±0.2 nm)": [dict(wavelength=lam + d) for d in (-0.2e-9, 0.0, 0.2e-9)],
            "3 × 3 launch × wavelength": [dict(launch_offset=a, wavelength=lam + d) for a in (-0.035, 0.0, 0.035)
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
    D2 = dataset(cell_with(ports_at(ASYM, inputs=(0, 1), width=PORT_W, launch=LAUNCH, fan=FAN)), e)
    tr, te = split(N, N // 4)
    r = RO.evaluate_krr(D2["X"], D2["Yab"], "linear", train_idx=tr, test_idx=te)
    out["sets"].append(dict(name="2 inputs (reference)", n_features=int(D2["X"].shape[1]), r2=per_m(r["r2"])))
    print("E8 λ decorrelation", dec, flush=True)
    return out


def e11(q):
    """As run_fga.e11, for the centred dot."""
    N = 400 if q else 1600
    cell = cell_with(layouts()["asym4 · 2 in"])
    e = ensc(N, 41)
    D = dataset(cell, e)
    I0 = D["I0"]
    X = D["X"] * I0 / I0.mean()
    Y = D["Yab"]
    tr, te = split(N, N // 4)
    rng = np.random.default_rng(3)
    noise = dict(sigma=[], noise_trained=[])
    for sg in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
        Xn = X + rng.normal(0, 1, X.shape) * sg
        noise["sigma"].append(sg)
        noise["noise_trained"].append(per_m(RO.evaluate_krr(Xn, Y, "linear", train_idx=tr, test_idx=te)["r2"]))
        print(f"E11 noise {sg}: {np.round(noise['noise_trained'][-1], 2)}", flush=True)
    s1 = 0.01
    pred = fit_ridge(X + rng.normal(0, 1, X.shape) * s1, Y, tr)
    perts = sample_shapes(cell, e)
    te_s = te[:150]
    dns = [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
    batch = [None] + [None] * len(dns) + [perts[i] for _ in dns for i in te_s]
    dn_list = [0.0] + dns + [d for d in dns for _ in te_s]
    Fd_all = fields(cell, batch, dn_global=dn_list)
    ref = Fd_all[0]
    keys = sorted(ref)
    drift = dict(dn=dns, sigma=s1, r2_abs=[], r2_diff=[], corr_ref=[])
    for k, d in enumerate(dns):
        Fd = [Fd_all[1 + len(dns) + k * len(te_s) + j] for j in range(len(te_s))]
        Xa = np.array([stack_intensities(f, reference=ref, mode="delta") for f in Fd]) / I0.mean()
        Xd = np.array([stack_intensities(f, reference=Fd_all[1 + k], mode="delta") for f in Fd]) / I0.mean()
        nz = rng.normal(0, 1, Xa.shape) * s1
        drift["r2_abs"].append(per_m(RO.r2(Y[te_s], pred(Xa + nz))))
        drift["r2_diff"].append(per_m(RO.r2(Y[te_s], pred(Xd + nz))))
        drift["corr_ref"].append(float(np.mean([field_correlation(ref[c], Fd_all[1 + k][c]) for c in keys])))
        print(f"E11 drift {d:g}: absolute {np.round(drift['r2_abs'][-1], 2)}, differential {np.round(drift['r2_diff'][-1], 2)}, "
              f"speckle corr {drift['corr_ref'][-1]:.3f}", flush=True)
    return dict(N=N, noise=noise, drift=drift)


def e13(q):
    """Odd orders of a centred dot: after a bounce a near-diametral chord is traversed again almost point-reflected
    through the centre, which flips the sign of odd harmonics, so their phases cancel chord by chord up to a residual
    that grows with the incidence angle χ (rotation 2mχ per bounce). Sweep launch and fan at small Δn."""
    N = 160 if q else 400
    dn = 3e-5
    launch = [0.03, 0.06, 0.09, 0.12]
    fans = [0.1, 0.2]
    out = dict(N=N, dn=dn, launch=launch, fans=fans, r2={})
    for fan in fans:
        rows = []
        for la in launch:
            cell = cell_with(layouts(launch=la, fan=fan)["asym4 · 2 in"])
            D = dataset(cell, ensc(N, 91, dn=dn))
            tr, te = split(N, N // 4)
            r = per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
            rows.append(r)
            print(f"E13 fan {fan:.2f} launch {la:.2f}: {np.round(r, 2)}", flush=True)
        out["r2"][f"{fan:.1f}"] = rows
    return out


def e14(q):
    """Odd vs even orders against the wall reflectance (path length): fewer bounces leave fewer cancelling chord
    pairs for the odd orders and less coherent accumulation for the even ones."""
    N = 160 if q else 400
    dn = 3e-5
    Rs = [0.5, 0.8, 0.9, 0.97]
    rng = np.random.default_rng(14)
    out = dict(N=N, dn=dn, R=Rs, clean=[], noisy=[], launch=LAUNCH, fan=FAN)
    for Rw in Rs:
        cell = cell_with(layouts()["asym4 · 2 in"])
        cell.reflectance = Rw
        D = dataset(cell, ensc(N, 91, dn=dn))
        tr, te = split(N, N // 4)
        c = per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        Xn = D["X"] + rng.normal(0, 1, D["X"].shape) * 0.01 * D["I0"].mean() / D["I0"]
        n_ = per_m(RO.evaluate_krr(Xn, D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
        out["clean"].append(c)
        out["noisy"].append(n_)
        print(f"E14 R={Rw}: clean {np.round(c, 2)}, noisy {np.round(n_, 2)}", flush=True)
    return out


def e15(q):
    """Centred dot in the stadium vs the circle (Python FGA phase screen; the JS engine has the circle only). The
    stadium has no conserved angular momentum, so successive chords through the centre are not point-reflected
    copies of each other and the odd orders need not cancel."""
    from ngrc.dataset import build
    from ngrc.geometry import WallCell, stadium, wall_ports
    from ngrc.perturbative import PhaseScreenModel
    from ngrc.rays import Source
    N = 150 if q else 300
    src = lambda i: Source(i, n_pos=None, n_ang=2000 if not q else 1000, frozen=20e-6)
    st = stadium(0.6e-3, 1.2e-3)
    cells = {"circle (launch 3°)": cell_with(layouts()["asym4 · 2 in"]),
             "stadium (launch 20°)": WallCell(wall=st, n_eff=1.8, wavelength=1.55e-6, reflectance=R_WALL,
                                              ports=wall_ports(st, [0.05, 0.27, 0.52, 0.78], inputs=(0, 1),
                                                               width=PORT_W, launch=0.35, fan=0.4))}
    rows = []
    for name, cell in cells.items():
        f = F.CACHE / f"e15_{name.split()[0]}_{N}_{DN:g}.npz"
        if f.exists():
            z = np.load(f)
            X, Yab, Yp = z["X"], z["Yab"], z["Yp"]
        else:
            t0 = time.time()
            m = PhaseScreenModel(cell, [src(i) for i in cell.inputs], max_bounces=300, amp_min=0.02)
            print(f"  {name}: model built in {time.time() - t0:.0f} s", flush=True)
            D = build(cell, ensc(N, 81), model=m, progress=50)
            names = D["names"]
            Yab = D["Y"][:, [i for i, s in enumerate(names) if s[0] in "ab"]]
            Yp = D["Y"][:, [i for i, s in enumerate(names) if s[0] == "p"]]
            X = D["X"]
            F.CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, X=X, Yab=Yab, Yp=Yp)
            print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
        tr, te = split(N, N // 4)
        r = RO.evaluate_krr(X, Yab, "linear", train_idx=tr, test_idx=te)
        rp = RO.evaluate_krr(X, Yp, "poly2", train_idx=tr, test_idx=te)
        rows.append(dict(name=name, r2=per_m(r["r2"]), r2_p=rp["r2"].tolist()))
        print(f"E15 {name}: R² {np.round(per_m(r['r2']), 2)}, p {np.round(rp['r2'], 2)}", flush=True)
    return dict(N=N, dn=DN, readout=rows)


def wall_dataset(name, cell, N, q):
    """Python FGA phase-screen dataset for any cell (cached by name)."""
    from ngrc.dataset import build
    from ngrc.perturbative import PhaseScreenModel
    from ngrc.rays import Source
    f = F.CACHE / f"wall_{name}_{N}_{DN:g}_{LAUNCH:g}_{FAN:g}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    m = PhaseScreenModel(cell, [Source(i, n_pos=None, n_ang=1000 if q else 2000, frozen=20e-6) for i in cell.inputs],
                         max_bounces=300, amp_min=0.02)
    D = build(cell, ensc(N, 81), model=m, progress=100)
    names = D["names"]
    out = dict(X=D["X"], Yab=D["Y"][:, [i for i, s in enumerate(names) if s[0] in "ab"]],
               Yp=D["Y"][:, [i for i, s in enumerate(names) if s[0] == "p"]],
               I0=np.concatenate([np.abs(D["reference"][k]) ** 2 for k in sorted(D["reference"])]))
    F.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    return out


def e16(q):
    """Break the circle's point symmetry with weak deformations that keep rays near the centre: a flat cut (D-cell,
    chaotic), a short stadium (weakly chaotic, still point-symmetric), a 24-facet polygon (segmented circle). Centred
    dot, ports at the asym4 angles with two inputs, launch 3°, Python FGA phase screen."""
    from ngrc.geometry import WallCell, d_cell, polygon, ports_by_angle, stadium
    N = 150 if q else 300
    walls = {"circle": None,
             "D-cell, cut at 0.95 R": d_cell(1e-3, 0.95),
             "D-cell, cut at 0.80 R": d_cell(1e-3, 0.80),
             "short stadium (0.1 mm straights)": stadium(0.95e-3, 0.1e-3),
             "24-facet polygon": polygon(1e-3, 24)}
    rows = []
    for name, w in walls.items():
        if w is None:
            cell = cell_with(layouts()["asym4 · 2 in"])
        else:
            cell = WallCell(wall=w, n_eff=1.8, wavelength=1.55e-6, reflectance=R_WALL,
                            ports=ports_by_angle(w, ASYM, inputs=(0, 1), width=PORT_W, launch=LAUNCH, fan=FAN))
        D = wall_dataset("".join(c if c.isalnum() else "_" for c in name), cell, N, q)
        X, Yab, Yp = D["X"], D["Yab"], D["Yp"]
        tr, te = split(N, N // 4)
        r = per_m(RO.evaluate_krr(X, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
        rp = RO.evaluate_krr(X, Yp, "poly2", train_idx=tr, test_idx=te)["r2"].tolist()
        rows.append(dict(name=name, r2=r, r2_p=rp))
        print(f"E16 {name}: R² {np.round(r, 2)}, p {np.round(rp, 2)}", flush=True)
    return dict(N=N, dn=DN, launch=LAUNCH, fan=FAN, readout=rows)


def wall_cell(w, angles=ASYM, inputs=(0, 1)):
    from ngrc.geometry import WallCell, ports_by_angle
    return WallCell(wall=w, n_eff=1.8, wavelength=1.55e-6, reflectance=R_WALL,
                    ports=ports_by_angle(w, angles, inputs=inputs, width=PORT_W, launch=LAUNCH, fan=FAN))


def sweep(cells, N, seed, rng):
    """Linear readout of a_m, b_m per cell (JS engine, phase screen, 6000 directions): clean, with 1 % detector noise
    (trained with the noise), and p_m with the NGRC readout."""
    out = dict(clean=[], noisy=[], r2_p=[])
    for name, cell in cells:
        D = dataset(cell, ensc(N, seed))
        X, Yab = D["X"], D["Yab"]
        tr, te = split(N, N // 4)
        c = per_m(RO.evaluate_krr(X, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
        Xn = X + rng.normal(0, 1, X.shape) * 0.01 * D["I0"].mean() / np.maximum(D["I0"], 1e-30)
        n_ = per_m(RO.evaluate_krr(Xn, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
        rp = RO.evaluate_krr(X, D["Yp"], "poly2", train_idx=tr, test_idx=te)["r2"].tolist()
        out["clean"].append(c)
        out["noisy"].append(n_)
        out["r2_p"].append(rp)
        print(f"  {name}: clean {np.round(c, 2)}, noisy {np.round(n_, 2)}, p {np.round(rp, 2)}", flush=True)
    return out


def e17(q):
    """Stadium straight length a (cap radius 0.95 mm; a = 0 is a circle): small a moves the cap centres off the cell
    centre, so the wall normals stop pointing at it and each bounce changes a ray's offset from the centre."""
    from ngrc.geometry import stadium
    N = 200 if q else 800
    a_um = [0, 10, 20, 50, 100, 300]
    out = dict(N=N, dn=DN, a_um=a_um)
    out.update(sweep([(f"stadium a = {a} µm", wall_cell(stadium(0.95e-3, a * 1e-6))) for a in a_um], N, 11,
                     np.random.default_rng(17)))
    return out


def e18(q):
    """Segmented circle: n flat facets (circumradius 1 mm); facet normals point at the centre only at the facet
    midpoints, so a ray's offset from the centre changes at every bounce."""
    from ngrc.geometry import polygon
    N = 200 if q else 800
    nf = [12, 24, 48, 96, 192]
    out = dict(N=N, dn=DN, n_facets=nf)
    out.update(sweep([(f"{n} facets", wall_cell(polygon(1e-3, n))) for n in nf], N, 11, np.random.default_rng(18)))
    return out


def e19(q):
    """The 24-facet polygon as the working cell: readouts (as E3), learning curve and detector noise (as E11)."""
    from ngrc.geometry import polygon
    N = 400 if q else 1600
    cell = wall_cell(polygon(1e-3, 24))
    D = dataset(cell, ensc(N, 41))
    X, Yab, Yp, I0 = D["X"], D["Yab"], D["Yp"], D["I0"]
    tr, te = split(N, N // 4)
    kw = dict(train_idx=tr, test_idx=te)
    res = {"ridge (linear)": per_m(RO.evaluate_krr(X, Yab, "linear", **kw)["r2"]),
           "NGRC ridge (lin + quad)": per_m(RO.evaluate_krr(X, Yab, "poly2", **kw)["r2"]),
           "linear SVR": per_m(RO.evaluate_svr(X, Yab, "linear", seed=0)["r2"]),
           "RBF SVR": per_m(RO.evaluate_svr(X, Yab, "rbf", seed=0)["r2"])}
    pw = {"NGRC ridge (lin + quad)": RO.evaluate_krr(X, Yp, "poly2", **kw)["r2"].tolist(),
          "RBF SVR": RO.evaluate_svr(X, Yp, "rbf", seed=0)["r2"].tolist()}
    print("E19 readouts:", {k: np.round(v, 2).tolist() for k, v in res.items()}, "p:", {k: np.round(v, 2).tolist() for k, v in pw.items()}, flush=True)
    ns = [n for n in (100, 200, 400, 800, 1200) if n <= len(tr)]
    lc = [per_m(RO.evaluate_krr(X, Yab, "linear", train_idx=tr[:n], test_idx=te)["r2"]) for n in ns]
    print("E19 learning:", [round(float(np.mean(c)), 2) for c in lc], flush=True)
    Xr = X * I0 / I0.mean()
    rng = np.random.default_rng(19)
    noise = dict(sigma=[], r2=[])
    for sg in (0.0, 0.003, 0.01, 0.03, 0.1):
        r = per_m(RO.evaluate_krr(Xr + rng.normal(0, 1, X.shape) * sg, Yab, "linear", **kw)["r2"])
        noise["sigma"].append(sg)
        noise["r2"].append(r)
        print(f"E19 noise {sg}: {np.round(r, 2)}", flush=True)
    return dict(N=N, dn=DN, n_features=int(X.shape[1]), r2_ab=res, r2_p=pw, learning=dict(n_train=ns, r2=lc), noise=noise)


def e20(q):
    """24-facet cell: Δn against detector noise. A larger Δn gives a larger intensity change per unit shape change
    until the dot phase wraps; with noise the best Δn moves up."""
    from ngrc.geometry import polygon
    N = 200 if q else 800
    cell = wall_cell(polygon(1e-3, 24))
    dns = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    rng = np.random.default_rng(20)
    out = dict(N=N, dn=dns, sigma=[0.0, 0.003, 0.01, 0.03], r2={})
    for dn in dns:
        D = dataset(cell, ensc(N, 11, dn=dn))
        X, Yab, I0 = D["X"], D["Yab"], D["I0"]
        Xr = X * I0 / I0.mean()
        tr, te = split(N, N // 4)
        rows = [per_m(RO.evaluate_krr(Xr + rng.normal(0, 1, X.shape) * sg, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
                for sg in out["sigma"]]
        out["r2"][f"{dn:g}"] = rows
        print(f"E20 Δn={dn:g}: " + "; ".join(f"σ={sg}: {np.round(r, 2)}" for sg, r in zip(out["sigma"], rows)), flush=True)
    return out


def e21(q):
    """Curved rays vs phase screen in the 24-facet cell at the noise-robust Δn. For a dot and the same dot with
    a_2 + 0.02 (or a_3 + 0.02), correlate the change of the field ΔE and of the intensity ΔI between the two engines
    as the number of launch directions grows: sampling noise falls with more directions, a physical difference
    (ray bending) does not. Also the field correlation of the perturbed speckle itself."""
    from ngrc.geometry import polygon
    cell = wall_cell(polygon(1e-3, 24))
    out = dict(cases=[])
    nas = (750, 1500) if q else (1500, 3000, 6000, 12000)
    for dn in (3e-4, 1e-3):
        for m in (2, 3):
            s1 = Shape(0.0, 0.0, 80e-6, dn, 3e-6, a=[0, 0.06, 0.03, 0, 0, 0], b=[0, 0, 0, 0.02, 0, 0])
            s2 = copy.deepcopy(s1)
            s2.a = s2.a.copy()
            s2.a[m - 1] += 0.02
            batch = [None, Perturbation([s1]), Perturbation([s2])]
            row = dict(dn=dn, m=m, n_ang=[], corr_dE=[], corr_dI=[], corr_E=[])
            for na in nas:
                cfg = dict(FGA, n_ang=na)
                Fp, Fc = fields(cell, batch, "phase", cfg), fields(cell, batch, "curved", cfg)
                keys = sorted(Fp[0])
                cat = lambda F, f: np.concatenate([f(F, k) for k in keys])
                dEp, dEc = cat(Fp, lambda F, k: F[2][k] - F[1][k]), cat(Fc, lambda F, k: F[2][k] - F[1][k])
                dIp = cat(Fp, lambda F, k: np.abs(F[2][k]) ** 2 - np.abs(F[1][k]) ** 2)
                dIc = cat(Fc, lambda F, k: np.abs(F[2][k]) ** 2 - np.abs(F[1][k]) ** 2)
                row["n_ang"].append(na)
                row["corr_dE"].append(field_correlation(dEp, dEc))
                row["corr_dI"].append(float(np.corrcoef(dIp, dIc)[0, 1]))
                row["corr_E"].append(float(np.mean([field_correlation(Fp[1][k], Fc[1][k]) for k in keys])))
                print(f"E21 Δn={dn:g} a{m}: {na} directions: corr ΔE {row['corr_dE'][-1]:.3f}, ΔI {row['corr_dI'][-1]:.3f}, "
                      f"E {row['corr_E'][-1]:.4f}", flush=True)
            out["cases"].append(row)
    return out


def e22(q):
    """Fabrication errors of the 24-facet cell (random facet tilts and offsets).
    (a) Trained on the chip: a readout trained on each fabricated cell (as one would on a real chip).
    (b) Trained on the design (a digital twin): the nominal cell's readout (E19 data) applied to fabricated cells,
        with features relative to the fabricated empty cell's own speckle."""
    from ngrc.geometry import polygon
    N = 200 if q else 800
    nominal = wall_cell(polygon(1e-3, 24))
    out = dict(N=N, dn=DN, own=[], twin=[])
    for tilt, off in ((1e-3, 0.0), (0.0, 1e-6), (1e-3, 1e-6)):
        for seed in (1, 2):
            cell = wall_cell(polygon(1e-3, 24, tilt_rms=tilt, offset_rms=off, seed=seed))
            D = dataset(cell, ensc(N, 11))
            tr, te = split(N, N // 4)
            r = per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"])
            out["own"].append(dict(tilt_urad=tilt * 1e6, offset_nm=off * 1e9, seed=seed, r2=r))
            print(f"E22 own: tilt {tilt * 1e6:.0f} µrad, offset {off * 1e9:.0f} nm, seed {seed}: {np.round(r, 2)}", flush=True)
    Nn = 400 if q else 1600
    e = ensc(Nn, 41)
    Dn = dataset(nominal, e)
    tr, te = split(Nn, Nn // 4)
    pred = fit_ridge(Dn["X"], Dn["Yab"], tr)
    perts = sample_shapes(nominal, e)
    te_s = te[:200]
    for off in (0.0, 1e-9, 3e-9, 10e-9, 30e-9, 100e-9):
        cell = wall_cell(polygon(1e-3, 24, offset_rms=off, seed=3)) if off > 0 else nominal
        F = fields(cell, [None] + [perts[i] for i in te_s])
        Xt = np.array([stack_intensities(f, reference=F[0]) for f in F[1:]])
        r = per_m(RO.r2(Dn["Yab"][te_s], pred(Xt)))
        out["twin"].append(dict(offset_nm=off * 1e9, r2=r))
        print(f"E22 twin: offset {off * 1e9:.0f} nm: {np.round(r, 2)}", flush=True)
    return out


def e23(q):
    """Ray bending without sampling noise: the same frozen-Gaussian rays traced with curved rays and with the phase
    screen (straight chords + ∫Δn ds), records paired ray by ray in order up to the first change of the port/bounce
    sequence. For a paired record the phase it contributes at its own Gaussian centre differs by
    k0 [L_c − L_s + n0 t_c·δ + ½ K_c q²] (δ = r_s − r_c, q its transverse part, K_c = Im P / Im Q the point-source
    wavefront curvature): the first-order part of the endpoint shift cancels (eikonal), so this is the second-order
    bending error. It is compared with the phase change from the shape signal (a_2 + 0.02) on the same records."""
    from ngrc.geometry import polygon
    from ngrc.rays import Source, trace
    n_ang = 51 if q else 101
    out = dict(n_ang=n_ang, cells=[])
    for cname, cell in (("smooth circle", cell_with(layouts()["asym4 · 2 in"])), ("24-facet cell", wall_cell(polygon(1e-3, 24)))):
        k0, n0 = cell.k0, cell.n_eff
        src = Source(0, n_pos=None, n_ang=n_ang, frozen=20e-6)
        row = dict(name=cname, dn=[], paired=[], diverge_bounce=[], bend_rms=[], bend_median=[], tail=[], core_rms=[], signal_rms=[], shift_um=[])
        for dn in (1e-4, 3e-4, 1e-3):
            s1 = Shape(0, 0, 80e-6, dn, 3e-6, a=[0, .06, .03, 0, 0, 0], b=[0, 0, 0, .02, 0, 0])
            s2 = copy.deepcopy(s1)
            s2.a = s2.a.copy()
            s2.a[1] += 0.02
            es = trace(cell, src, Perturbation([s1]), "straight", max_bounces=300, amp_min=0.02)
            es2 = trace(cell, src, Perturbation([s2]), "straight", max_bounces=300, amp_min=0.02)
            ec = trace(cell, src, Perturbation([s1]), "curved", max_bounces=300, amp_min=0.02)
            def recs(ex):
                d = {}
                for j in np.argsort(ex.ray, kind="stable"):
                    d.setdefault(int(ex.ray[j]), []).append(j)
                return d
            rs, rc, r2_ = recs(es), recs(ec), recs(es2)
            wtot = wsame = 0.0
            ph, sig, w, dx, div = [], [], [], [], []
            for ray, js in rs.items():
                jc, j2 = rc.get(ray, []), r2_.get(ray, [])
                for i, a in enumerate(js):
                    wa = es.amp[a] * abs(es.W[a])
                    wtot += wa
                    if i >= len(jc) or es.port[a] != ec.port[jc[i]] or es.nb[a] != ec.nb[jc[i]]:
                        div.append(es.nb[a])
                        wtot += sum(es.amp[b] * abs(es.W[b]) for b in js[i + 1:])
                        break
                    b = jc[i]
                    wsame += wa
                    d = np.array([es.x[a] - ec.x[b], es.y[a] - ec.y[b]])
                    qp = -ec.ty[b] * d[0] + ec.tx[b] * d[1]
                    K = ec.P[b].imag / ec.Q[b].imag if ec.Q[b].imag != 0 else 0.0
                    ph.append(k0 * (ec.L[b] - es.L[a] + n0 * (ec.tx[b] * d[0] + ec.ty[b] * d[1]) + 0.5 * K * qp * qp))
                    sig.append(k0 * (es2.L[j2[i]] - es.L[a]) if i < len(j2) and es2.nb[j2[i]] == es.nb[a] else np.nan)
                    w.append(wa)
                    dx.append(np.hypot(*d))
            ph = np.angle(np.exp(1j * np.array(ph)))
            sig, w, dx = np.array(sig), np.array(w), np.array(dx)
            core = np.abs(ph) <= 0.3
            rms = lambda v, m=None: float(np.sqrt(np.nansum((w * v ** 2)[m] if m is not None else w * v ** 2) /
                                                  np.sum(w[m] if m is not None else w[np.isfinite(v)])))
            for k, v in (("dn", dn), ("paired", wsame / wtot), ("diverge_bounce", float(np.median(div)) if div else -1.0),
                         ("bend_rms", rms(ph)), ("bend_median", float(np.median(np.abs(ph)))),
                         ("tail", float(np.sum(w[~core]) / np.sum(w))), ("core_rms", rms(ph, core)),
                         ("signal_rms", rms(sig)), ("shift_um", rms(dx) * 1e6)):
                row[k].append(v)
            print(f"E23 {cname} Δn={dn:g}: paired {row['paired'][-1]:.2f} (diverge at bounce {row['diverge_bounce'][-1]:.0f}), "
                  f"bend rms {row['bend_rms'][-1]:.1e} median {row['bend_median'][-1]:.1e} tail {row['tail'][-1]:.3f} "
                  f"core {row['core_rms'][-1]:.1e}; signal {row['signal_rms'][-1]:.1e}; shift {row['shift_um'][-1]:.1f} µm", flush=True)
        out["cells"].append(row)
    return out


WAVE_R = (2.0e-6, 3.5e-6)        # dot radii in the 25 µm wave cell (8–14 % of the cell, as 60–110 µm in 1 mm)


def wave_dataset(name, cell, N, dn, seed=11):
    """Full-wave (FDFD) dataset of centred dots, cached and resumable every 25 samples."""
    import dataclasses
    from ngrc.dataset import labels
    f = F.CACHE / f"wave_{name}_{N}_{dn:g}_{seed}.npz"
    part = F.CACHE / f"wave_{name}_{N}_{dn:g}_{seed}.part.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    e = dataclasses.replace(ensc(N, seed, dn=dn), R_range=WAVE_R, edge=0.25e-6)
    perts = sample_shapes(cell, e)
    t0 = time.time()
    cell.setup()
    ref = cell.fields(None)
    rows, its = [], []
    if part.exists():
        z = np.load(part)
        rows, its = list(z["X"]), list(z["its"])
    for j in range(len(rows), N):
        Fj = cell.fields(perts[j])
        rows.append(stack_intensities(Fj, reference=ref))
        its.append(max(v[0] for v in cell.last_info.values()))
        if (j + 1) % 25 == 0 or j + 1 == N:
            F.CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(part, X=np.array(rows), its=np.array(its))
            print(f"  {name} Δn={dn:g}: {j + 1}/{N}, {time.time() - t0:.0f} s, GMRES ≤ {max(its)} iterations", flush=True)
    Y, names = labels(perts, M_MAX)
    out = dict(X=np.array(rows), Yab=Y[:, [i for i, s_ in enumerate(names) if s_[0] in "ab"]],
               Yp=Y[:, [i for i, s_ in enumerate(names) if s_[0] == "p"]], its=np.array(its),
               I0=np.concatenate([np.abs(ref[k]) ** 2 for k in sorted(ref)]))
    np.savez_compressed(f, **out)
    part.unlink(missing_ok=True)
    return out


def e24(q):
    """Full-wave reservoir: 2D scalar Helmholtz (FDFD) in a 25 µm cell, centred dot, ports at the asym4 angles
    (1.5 µm openings), two inputs. Δn × cell size sets both the accumulated dot phase and the bending
    displacement, so Δn_w = 1.2e-3 / 1.2e-2 here emulates Δn = 3e-5 / 3e-4 in the 1 mm cell. Includes diffraction
    at corners and port edges, ray bending and multiple scattering. Linear readout, clean and with 1 % noise."""
    from ngrc.fdfd import WaveCell
    N = 120 if q else 400
    rng = np.random.default_rng(24)
    out = dict(N=N, radius_um=25, dot_R_um=[r * 1e6 for r in WAVE_R], cases=[])
    for name, fac, dn in (("24-facet", 24, 1.2e-3), ("circle", 0, 1.2e-3), ("24-facet", 24, 1.2e-2)):
        cell = WaveCell(radius=25e-6, facets=fac)
        D = wave_dataset(f"{name}", cell, N, dn)
        X, Yab = D["X"], D["Yab"]
        tr, te = split(N, N // 4)
        c = per_m(RO.evaluate_krr(X, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
        Xn = X + rng.normal(0, 1, X.shape) * 0.01 * D["I0"].mean() / np.maximum(D["I0"], 1e-30)
        n_ = per_m(RO.evaluate_krr(Xn, Yab, "linear", train_idx=tr, test_idx=te)["r2"])
        rp = RO.evaluate_krr(X, D["Yp"], "poly2", train_idx=tr, test_idx=te)["r2"].tolist()
        out["cases"].append(dict(name=f"{name}, Δn_w = {dn:g}", facets=fac, dn=dn, clean=c, noisy=n_, r2_p=rp,
                                 gmres_max=int(D["its"].max())))
        print(f"E24 {name} Δn_w={dn:g}: clean {np.round(c, 2)}, noisy {np.round(n_, 2)}, p {np.round(rp, 2)}", flush=True)
    return out


def e25(q):
    """Noise robustness of the 24-facet cell through extra views: 1, 3 or 5 wavelengths 0.2 nm apart (independent
    speckles, each with its own detector noise), at Δn = 3e-4 and 1e-3. Linear readout trained with the noise."""
    from ngrc.geometry import polygon
    N = 200 if q else 800
    cell = wall_cell(polygon(1e-3, 24))
    lam = cell.wavelength
    rng = np.random.default_rng(25)
    sets = {"1 wavelength": [0.0], "3 wavelengths": [-0.2e-9, 0.0, 0.2e-9], "5 wavelengths": [-0.4e-9, -0.2e-9, 0.0, 0.2e-9, 0.4e-9]}
    out = dict(N=N, sigma=[0.003, 0.01, 0.03], cases=[])
    for dn in (3e-4, 1e-3):
        e = ensc(N, 11, dn=dn)
        Ds = {d: dataset(cell.variant(wavelength=lam + d), e) for d in sets["5 wavelengths"]}
        Y = Ds[0.0]["Yab"]
        tr, te = split(N, N // 4)
        for name, ds in sets.items():
            rows = []
            for sg in out["sigma"]:
                X = np.hstack([Ds[d]["X"] * Ds[d]["I0"] / Ds[d]["I0"].mean() + rng.normal(0, 1, Ds[d]["X"].shape) * sg for d in ds])
                rows.append(per_m(RO.evaluate_krr(X, Y, "linear", train_idx=tr, test_idx=te)["r2"]))
            out["cases"].append(dict(name=f"Δn = {dn:g}, {name}", dn=dn, views=len(ds), r2=rows))
            print(f"E25 Δn={dn:g} {name}: " + "; ".join(f"σ={sg}: {np.round(r, 2)}" for sg, r in zip(out["sigma"], rows)), flush=True)
    return out


EXPS = {"E1": e1, "E3": e3, "E6": e6, "E12": e12, "E2": e2, "E8": e8, "E11": e11, "E13": e13, "E14": e14, "E15": e15, "E16": e16, "E17": e17, "E18": e18, "E19": e19, "E20": e20, "E21": e21, "E22": e22, "E23": e23, "E24": e24, "E25": e25}


def main(only, quick):
    pj = ROOT / "results" / "progress.json"
    arch = json.loads((ROOT / "results" / "progress_offcentre.json").read_text())
    t0 = time.time()
    for name, fn in EXPS.items():
        if only and name not in only:
            continue
        res = fn(quick)
        cur = json.loads(pj.read_text()) if pj.exists() else {}
        if cur.get("meta", {}).get("dot") != "centre":
            cur = {k: arch[k] for k in CARRY if k in arch}
        cur[name] = res
        meta = cur.setdefault("meta", {})
        meta.update(model="fga", dot="centre", centre_um=[0, 0], launch=LAUNCH, fan=FAN, dn=DN,
                    carried_offcentre=list(CARRY), offcentre_um=[250, 100], date=time.strftime("%Y-%m-%d"),
                    quick=quick, m=list(range(2, M_MAX + 1)),
                    cell=dict(radius=1e-3, n_eff=1.8, wavelength=1.55e-6, R=R_WALL, port_width=PORT_W),
                    source=dict(n_pos="auto", n_ang=FGA["n_ang"], frozen_um=20))
        meta["runtime_s"] = round(meta.get("runtime_s", 0) + time.time() - t0, 1) if name != "E1" else round(time.time() - t0, 1)
        pj.write_text(json.dumps(cur, indent=1))
        print(f"[{name} saved, {time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--launch", type=float, default=LAUNCH)
    ap.add_argument("--fan", type=float, default=FAN)
    a = ap.parse_args()
    LAUNCH, FAN = a.launch, a.fan
    main([s.strip() for s in a.only.split(",") if s.strip()] or None, a.quick)
