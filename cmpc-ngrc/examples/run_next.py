"""Second round of experiments; adds E5–E9 to results/progress.json (shown in the explorer applet).

  E5  position tolerance (T8.4): a readout trained at one position tested on displaced dots, training with
      placement jitter, and dots anywhere (also with a position-dependent readout, product kernel ⊗ position)
  E6  learning curves (T8.2): R² vs number of training samples, fixed and random positions
  E7  engine (T3.4): full curved rays (JS engine through node) vs the phase-screen model, same dots
  E8  multiplexing (T6.1): launch-angle and wavelength steps as extra virtual nodes
  E9  wave reference (T5.2): one pass of a Gaussian beam through a dot, beamlet sums vs split-step BPM
  E10 other cells (T8.6): chaotic stadium vs circle
  E11 robustness (T8.5): detector noise and uniform index (temperature) drift
  E12 nonlinearity regime (T8.3): readout vs Δn, curved rays vs phase screen, with and without noise

Datasets are cached in results/cache/ (git-ignored).   python examples/run_next.py [--only E5,E6] [--quick]
"""
from __future__ import annotations

import argparse
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
from ngrc.dataset import Ensemble, build, build_multiplexed          # noqa: E402
from ngrc.field import field_correlation                             # noqa: E402
from ngrc.geometry import ports_at                                   # noqa: E402
from ngrc.jsengine import available as node_available                # noqa: E402
from ngrc.shapes import Perturbation, Shape                          # noqa: E402
from run_progress import M_MAX, PORT_W, SRC, cell_with, image_features, layouts, model_for, per_m  # noqa: E402

CACHE = ROOT / "results" / "cache"
FIXED = (0.25e-3, 0.1e-3)


def ens(n, seed, centre=None, place=0.6e-3):
    return Ensemble(n=n, m_max=M_MAX, R_range=(60e-6, 110e-6), sigma=0.08, decay=0.5, centre=centre,
                    place_radius=None if centre else place, seed=seed)


def dataset(cell, e, engine="phase", variants=None, cell_tag=None):
    """Cached (X, Yab, Yp, pos, names, perts-free) for a cell, ensemble and engine."""
    tag = json.dumps([cell_tag or cell.to_dict(), e.key(), engine, variants], sort_keys=True, default=str)
    f = CACHE / f"{hashlib.sha1(tag.encode()).hexdigest()[:16]}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    kw = dict(n_pos=SRC["n_pos"], n_ang=SRC["n_ang"]) if engine.startswith("node") else {}
    if variants:
        D = build_multiplexed(cell, e, variants, engine=engine, model=None, **kw)
    else:
        D = build(cell, e, engine=engine, model=model_for(cell) if engine == "phase" else None, **kw)
    names = D["names"]
    ia = [i for i, s in enumerate(names) if s[0] in "ab"]
    ip = [i for i, s in enumerate(names) if s[0] == "p"]
    pos = np.array([[p.components[0].x0, p.components[0].y0] for p in D["perts"]])
    img = image_features(D["perts"])
    ref = D.get("reference")
    I0 = np.concatenate([np.abs(ref[k]) ** 2 for k in sorted(ref)]) if ref is not None and not variants else np.zeros(0)
    out = dict(X=D["X"], Yab=D["Y"][:, ia], Yp=D["Y"][:, ip], mdom=D["Y"][:, -1], pos=pos, img=img, I0=I0,
               seconds=np.array(time.time() - t0))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"  built {engine} dataset n={e.n} in {time.time() - t0:.0f} s", flush=True)
    return out


def fixed_split(n, n_test, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[n_test:], idx[:n_test]


def e5(N):
    """Position tolerance: (a) readout trained at one position, tested on dots displaced by δ; (b) trained and
    tested with random placement jitter σ; (c) dots anywhere in the central 600 µm (KRR, also ⊗ true position)."""
    import copy
    from ngrc.dataset import sample_shapes
    from ngrc.features import stack_intensities
    cell = cell_with(layouts()["asym4 · 2 in"])
    m = model_for(cell)
    e = ens(N, 41, centre=FIXED)
    D = dataset(cell, e)
    X, Y = D["X"], D["Yab"]
    tr, te = fixed_split(len(X), N // 4)
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-3 * X[tr].std(0).mean())
    a, _ = RO.ridge_cv((X[tr] - mu) / sd, Y[tr])
    W, b = RO.ridge_fit((X[tr] - mu) / sd, Y[tr], a)
    perts = sample_shapes(cell, e)
    ref = m.fields(None)
    rng = np.random.default_rng(5)
    shift = dict(delta_um=[], r2=[])
    te_s = te[:200]
    for d in (0, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6, 10e-6, 20e-6):
        Xd = []
        for i in te_s:
            p = copy.deepcopy(perts[i])
            ph = rng.uniform(0, 2 * np.pi)
            for c in p.components:
                c.x0 += d * np.cos(ph)
                c.y0 += d * np.sin(ph)
            Xd.append(stack_intensities(m.fields(p), reference=ref))
        r = RO.r2(Y[te_s], ((np.array(Xd) - mu) / sd) @ W + b)
        shift["delta_um"].append(d * 1e6)
        shift["r2"].append(per_m(r))
        print(f"E5 shift {d * 1e6:.1f} µm: {np.round(per_m(r), 2)}", flush=True)
    jit = dict(sigma_um=[], r2=[])
    Nj = max(200, N // 2)
    for sg in (0, 2e-6, 5e-6, 10e-6, 20e-6):
        ej = Ensemble(**{**e.__dict__, "n": Nj, "jitter": sg, "seed": 71})
        Dj = dataset(cell, ej)
        trj, tej = fixed_split(Nj, Nj // 4)
        r = RO.evaluate_krr(Dj["X"], Dj["Yab"], "linear", train_idx=trj, test_idx=tej)
        jit["sigma_um"].append(sg * 1e6)
        jit["r2"].append(per_m(r["r2"]))
        print(f"E5 jitter {sg * 1e6:.0f} µm (N={Nj}): {np.round(per_m(r['r2']), 2)}", flush=True)
    Dr = dataset(cell, ens(N, 31))
    trr, ter = fixed_split(N, N // 4)
    anywhere = dict(linear=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "linear", train_idx=trr, test_idx=ter)["r2"]),
                    rbf=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "rbf", train_idx=trr, test_idx=ter)["r2"]),
                    linear_pos=per_m(RO.evaluate_krr(Dr["X"], Dr["Yab"], "linear", pos=Dr["pos"], train_idx=trr,
                                                     test_idx=ter)["r2"]))
    print("E5 anywhere:", {k: np.round(v, 2).tolist() for k, v in anywhere.items()}, flush=True)
    return dict(N=N, shift=shift, jitter=dict(N=Nj, **jit), anywhere=anywhere)


def e6(N):
    cell = cell_with(layouts()["asym4 · 2 in"])
    out = dict(cases=[])
    for case, e, kind, use_pos in (("fixed position · ridge", ens(N, 41, centre=FIXED), "linear", False),
                                   ("fixed position · NGRC poly2", ens(N, 41, centre=FIXED), "poly2", False),
                                   ("random position · linear ⊗ position", ens(N, 31), "linear", True)):
        D = dataset(cell, e)
        X, Y = D["X"], D["Yab"]
        tr_all, te = fixed_split(len(X), N // 4)
        ns = [n for n in (100, 200, 400, 800, 1600, 3200) if n <= len(tr_all)]
        curves = []
        for n in ns:
            r = RO.evaluate_krr(X, Y, kind, pos=D["pos"] if use_pos else None, train_idx=tr_all[:n], test_idx=te)
            curves.append(per_m(r["r2"]))
        out["cases"].append(dict(name=case, n_train=ns, r2=curves))
        print(f"E6 {case}: " + ", ".join(f"{n}: {np.mean(c):.2f}" for n, c in zip(ns, curves)), flush=True)
    return out


def e7(N):
    if not node_available():
        return None
    cell = cell_with(layouts()["asym4 · 2 in"])
    out = dict(N=N, cases=[])
    for case, e, use_pos in (("fixed position", ens(N, 51, centre=FIXED), False),
                             ("random position (⊗ true position)", ens(N, 52), True)):
        row = dict(name=case)
        for engine in ("phase", "node-curved"):
            D = dataset(cell, e, engine)
            tr, te = fixed_split(len(D["X"]), N // 4)
            pos = D["pos"] if use_pos else None
            row[engine] = dict(linear=per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", pos=pos, train_idx=tr, test_idx=te)["r2"]),
                               poly2=per_m(RO.evaluate_krr(D["X"], D["Yab"], "poly2", pos=pos, train_idx=tr, test_idx=te)["r2"]),
                               seconds=float(D["seconds"]))
        out["cases"].append(row)
        print(f"E7 {case}: phase {np.round(row['phase']['linear'], 2)} / curved {np.round(row['node-curved']['linear'], 2)}"
              f"  (poly2: {np.round(row['phase']['poly2'], 2)} / {np.round(row['node-curved']['poly2'], 2)})", flush=True)
    return out


def e8(N):
    asym = [0, 67, 151, 238]
    cell = cell_with(ports_at(asym, inputs=(0,), width=PORT_W, launch=0.35, fan=0.4))
    lam = cell.wavelength
    sets = {
        "1 input, 1 launch": None,
        "3 launch angles (±6°)": [dict(launch_offset=d) for d in (-0.1, 0.0, 0.1)],
        "3 wavelengths (±0.2 nm)": [dict(wavelength=lam + d) for d in (-0.2e-9, 0.0, 0.2e-9)],
        "3 × 3 launch × wavelength": [dict(launch_offset=a, wavelength=lam + d) for a in (-0.1, 0.0, 0.1)
                                      for d in (-0.2e-9, 0.0, 0.2e-9)],
        "2 inputs (reference)": "two",
    }
    e = ens(N, 61, centre=FIXED)
    out = dict(N=N, sets=[])
    m = model_for(cell)
    ref0 = m.fields(None)
    for dl in (0.05e-9, 0.1e-9, 0.2e-9, 0.5e-9):
        r = model_for(cell.variant(wavelength=lam + dl)).fields(None)
        out.setdefault("lambda_decorrelation", []).append(
            [dl * 1e9, float(np.mean([field_correlation(ref0[k], r[k]) for k in ref0]))])
    for name, v in sets.items():
        c = cell_with(ports_at(asym, inputs=(0, 1), width=PORT_W, launch=0.35, fan=0.4)) if v == "two" else cell
        D = dataset(c, e, "phase", None if v in (None, "two") else v)
        tr, te = fixed_split(len(D["X"]), N // 4)
        r = RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)
        out["sets"].append(dict(name=name, n_features=int(D["X"].shape[1]), r2=per_m(r["r2"])))
        print(f"E8 {name}: features {D['X'].shape[1]}, R² {np.round(per_m(r['r2']), 2)}", flush=True)
    print("   λ decorrelation:", out["lambda_decorrelation"], flush=True)
    return out


def e9():
    from ngrc.wave_ref import compare
    rows = []
    for z in (6e-3, 20e-3, 60e-3):
        for dn in (3e-4, 1e-3, 3e-3):
            p = Perturbation([Shape(1.0e-3, 10e-6, 50e-6, dn, 3e-6, a=[0, 0.05, 0.03])])
            r = compare(p, z_end=z, dz=4e-6, window=3e-3, ny=8192)
            rows.append(dict(z_mm=z * 1e3, dn=dn, curved=r["corr_curved"], straight=r["corr_straight"],
                             scat_curved=r["scat_curved"], scat_straight=r["scat_straight"],
                             err_curved=r["err_curved"], err_straight=r["err_straight"], scat_fraction=r["scat_fraction"]))
            print(f"E9 z={z * 1e3:.0f} mm Δn={dn:g}: |ΔE|/|E| {r['scat_fraction']:.3f}; scattered-field corr curved "
                  f"{r['scat_curved']:.4f} (err {r['err_curved']:.3f}), phase screen {r['scat_straight']:.4f} "
                  f"(err {r['err_straight']:.3f})", flush=True)
    return dict(rows=rows, beam_w_um=60, dot=dict(R_um=50, edge_um=3, z_um=1000))


def stadium_cell(launch=0.35, inputs=(0, 1)):
    from ngrc.geometry import WallCell, stadium, wall_ports
    st = stadium(0.6e-3, 1.2e-3)
    cell = WallCell(wall=st, n_eff=1.8, wavelength=1.55e-6, reflectance=0.99,
                    ports=wall_ports(st, [0.05, 0.27, 0.52, 0.78], inputs=inputs, width=PORT_W, launch=launch, fan=0.4))
    return cell, f"stadium r=0.6 a=1.2 ports .05/.27/.52/.78 in={inputs} launch={launch}"


def e10(N):
    """Stadium (chaotic) vs circle (integrable): sensitivity vs launch angle, fixed-position readout, position
    tolerance."""
    from ngrc import analysis as A
    from run_progress import base_shapes
    out = dict(N=N)
    bases = base_shapes(3, seed=7, place=0.4e-3)
    sens = dict(launch=[0.0, 0.4, 0.8, 1.0], circle=[], stadium=[])
    for la in sens["launch"]:
        c = cell_with(ports_at([0, 67, 151, 238], inputs=(0,), width=PORT_W, launch=la, fan=0.4))
        sc, _ = stadium_cell(la, inputs=(0,))
        for name, cc in (("circle", c), ("stadium", sc)):
            S, _ = A.sensitivity(model_for(cc), bases, M_MAX)
            sens[name].append(S.tolist())
        print(f"E10 launch {la:.1f}: circle S {np.round(sens['circle'][-1], 2)}, stadium S {np.round(sens['stadium'][-1], 2)}", flush=True)
    out["sensitivity"] = sens
    rows = []
    for name in ("circle", "stadium"):
        if name == "circle":
            cell, tag = cell_with(layouts()["asym4 · 2 in"]), None
        else:
            cell, tag = stadium_cell()
        D = dataset(cell, ens(N, 81, centre=(0.15e-3, 0.1e-3)), cell_tag=tag)
        tr, te = fixed_split(N, N // 4)
        r = RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)
        rp = RO.evaluate_krr(D["X"], D["Yp"], "poly2", train_idx=tr, test_idx=te)
        oc = model_for(cell)
        rows.append(dict(name=name, r2=per_m(r["r2"]), r2_p=rp["r2"].tolist(),
                         mean_bounces=float(np.mean([t["exits"].nb.mean() for t in oc.tables]))))
        print(f"E10 {name}: R² {np.round(per_m(r['r2']), 2)}, p_m {np.round(rp['r2'], 2)}, "
              f"mean bounces {rows[-1]['mean_bounces']:.0f}", flush=True)
    out["readout"] = rows
    return out


def e11(N):
    """Robustness (T8.5): readout trained on clean speckle, tested with detector noise or a uniform index drift
    (temperature); and retrained with the same noise."""
    from ngrc.dataset import sample_shapes
    from ngrc.features import stack_intensities
    cell = cell_with(layouts()["asym4 · 2 in"])
    m = model_for(cell)
    e = ens(N, 41, centre=FIXED)
    D = dataset(cell, e)
    X, Y = D["X"], D["Yab"]
    tr, te = fixed_split(len(X), N // 4)
    ref = m.fields(None)
    I0 = np.concatenate([np.abs(ref[k]) ** 2 for k in sorted(ref)])
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-3 * X[tr].std(0).mean())
    a, _ = RO.ridge_cv((X[tr] - mu) / sd, Y[tr])
    W, b = RO.ridge_fit((X[tr] - mu) / sd, Y[tr], a)
    rng = np.random.default_rng(3)
    noise = dict(sigma=[], clean_trained=[], noise_trained=[])
    for sg in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
        Xn = X + rng.normal(0, 1, X.shape) * sg * I0.mean() / I0        # additive noise σ·Ī0 on I
        r1 = RO.r2(Y[te], ((Xn[te] - mu) / sd) @ W + b)
        r2_ = RO.evaluate_krr(Xn, Y, "linear", train_idx=tr, test_idx=te)["r2"]
        noise["sigma"].append(sg)
        noise["clean_trained"].append(per_m(r1))
        noise["noise_trained"].append(per_m(r2_))
        print(f"E11 noise σ={sg}: clean-trained {np.round(per_m(r1), 2)}, noise-trained {np.round(per_m(r2_), 2)}", flush=True)
    perts = sample_shapes(cell, e)
    drift = dict(dn=[], r2=[], corr_ref=[])
    te_s = te[:200]
    for dn in (0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6):
        Xd = np.array([stack_intensities(m.fields(perts[i], dn_global=dn), reference=ref) for i in te_s])
        r = RO.r2(Y[te_s], ((Xd - mu) / sd) @ W + b)
        rd = m.fields(None, dn_global=dn)
        drift["dn"].append(dn)
        drift["r2"].append(per_m(r))
        drift["corr_ref"].append(float(np.mean([field_correlation(ref[k], rd[k]) for k in ref])))
        print(f"E11 drift δn={dn:g}: R² {np.round(per_m(r), 2)}, speckle corr {drift['corr_ref'][-1]:.3f}", flush=True)
    paths = np.concatenate([t["path"] for t in m.tables])
    return dict(N=N, noise=noise, drift=drift, mean_path_mm=float(np.mean(paths) * 1e3))


def e12(N):
    """Nonlinearity regime (T8.3): readout quality vs Δn of the dots, full curved rays vs phase screen, without
    and with detector noise (σ = 1 % of the mean intensity, readout trained with the noise)."""
    cell = cell_with(layouts()["asym4 · 2 in"])
    dns = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    out = dict(N=N, dn=dns, sigma=0.01)
    rng = np.random.default_rng(9)
    for engine in ("phase", "node-curved"):
        if engine.startswith("node") and not node_available():
            continue
        rows = dict(clean=[], noisy=[])
        for dn in dns:
            e = Ensemble(n=N, m_max=M_MAX, R_range=(60e-6, 110e-6), dn_range=(dn, dn), sigma=0.08, decay=0.5,
                         centre=FIXED, seed=91)
            D = dataset(cell, e, engine)
            tr, te = fixed_split(N, N // 4)
            rows["clean"].append(per_m(RO.evaluate_krr(D["X"], D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"]))
            I0 = D["I0"]
            Xn = D["X"] + rng.normal(0, 1, D["X"].shape) * 0.01 * I0.mean() / I0
            rows["noisy"].append(per_m(RO.evaluate_krr(Xn, D["Yab"], "linear", train_idx=tr, test_idx=te)["r2"]))
            print(f"E12 {engine} Δn={dn:g}: clean {np.round(rows['clean'][-1], 2)}, noisy {np.round(rows['noisy'][-1], 2)}", flush=True)
        out[engine] = rows
    return out


def main(only=None, quick=False, out_file="progress.json"):
    N = 400 if quick else 1600
    pj = ROOT / "results" / out_file
    out = json.loads(pj.read_text()) if pj.exists() else {}
    t0 = time.time()
    for name, fn in (("E5", lambda: e5(N)), ("E6", lambda: e6(N)), ("E7", lambda: e7(N // 2)),
                     ("E8", lambda: e8(N // 2)), ("E9", e9), ("E10", lambda: e10(N // 2)),
                     ("E11", lambda: e11(N)), ("E12", lambda: e12(N // 2))):
        if only and name not in only:
            continue
        r = fn()
        if r is not None:
            cur = json.loads(pj.read_text()) if pj.exists() else {}      # merge: other runs may have written
            cur[name] = r
            cur.setdefault("meta", {})[f"{name}_date"] = time.strftime("%Y-%m-%d")
            pj.write_text(json.dumps(cur, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> results/progress.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="progress.json", help="results file in results/ (merged, not overwritten)")
    a = ap.parse_args()
    main([s.strip() for s in a.only.split(",") if s.strip()] or None, a.quick, a.out)
