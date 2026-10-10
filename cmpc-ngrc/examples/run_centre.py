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
from ngrc.shapes import Perturbation, random_shape                   # noqa: E402

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


EXPS = {"E1": e1, "E3": e3, "E6": e6, "E12": e12, "E2": e2, "E8": e8, "E11": e11, "E13": e13, "E14": e14, "E15": e15}


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
