"""Scan: speckle contrast vs chaoticity of the cell (facet tilt, facet curvature), number of mutually incoherent beams K.
Ge 300 nm TE0 at 5.263 um, 1 cm segmented cell, 100 um ports, flat mirror R = 1-3e-4 for the coherence part;
the same traces are re-weighted with the two-material TE DBR and the tri-index (hole-free) DBR.
Writes chaos_scan.pkl. ~3-5 min on one core."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import json, pickle, time
import numpy as np
import cmpc_sim as cs, coherence_model as cm, mirror_design as md

LAM = 5.2629e-6
mA = cs.membrane_mode("Ge", 300e-9, LAM, "TE")
W = 100e-6
TH0 = LAM / (mA.neff * W)
CHI = np.pi / 2 - np.pi * 6 / 24
ALPHA = cs.dB_per_cm_to_alpha(0.03)
BL = 3e-4
NR, NB = 6000, 1000
R_flat = lambda s: np.full_like(s, 1 - BL)
dbr2 = cs.DBR(n_tooth=mA.neff, lam_design=LAM, N=20, m_gap=1, m_tooth=1, bounce_loss=BL, slab_pol="TE")
R2 = lambda s: dbr2.R(LAM, s)
des = json.load(open("data/mirror_designs.json"))["130_16"]
Rtri = md.R_function(LAM, des, BL)
SGL = np.linspace(0, 1, 2049)
LUT = {"two-material": 1 - R2(SGL), "tri-index": 1 - Rtri(SGL)}
deg = np.radians
CELLS = {"regular": {}, "tilt 0.05°": dict(tilt_rms=deg(0.05)), "tilt 0.1°": dict(tilt_rms=deg(0.1)), "tilt 0.3°": dict(tilt_rms=deg(0.3)),
         "tilt 1°": dict(tilt_rms=deg(1.0)), "curved ρ=20 cm": dict(curvature=1 / 0.20), "curved ρ=10 cm": dict(curvature=1 / 0.10),
         "curved ρ=5 cm": dict(curvature=1 / 0.05)}
MULTI = ("regular", "tilt 0.3°", "curved ρ=5 cm")


def trace(cell, k, K, seed=5):
    P = cell.perimeter
    ports = [cell.h + j * P / K for j in range(K)]
    return cs.trace_rays(cell, W, NR, NB, TH0, seed=seed + 10 * K + k, max_path=4.0, keep_start=False, theta_c=CHI,
                         s_in=ports[k], s_out=cell.h + 9 * 2 * cell.h, extra_ports=[q for j, q in enumerate(ports) if j != k])


def path_loss(tab, lut):
    sel = np.nonzero((tab.exit_port == 1) & (tab.exit_idx >= 0))[0]
    e = tab.exit_idx[sel]
    mask = np.arange(tab.n_bounce)[None, :] < e[:, None]
    return float((np.interp(np.abs(tab.sinchi[sel]), SGL, lut) * mask).sum() / max(mask.sum(), 1))


out = {}
t00 = time.time()
for name, kw in CELLS.items():
    cell = cs.SegmentedCell(5e-3, 24, seed=1, **kw)
    rec = dict(lyap=cs.lyapunov(cell), paths={}, mirrors={})
    for K in ((1, 2, 4) if name in MULTI else (1,)):
        rec["paths"][K] = []
        for k in range(K):
            tab = trace(cell, k, K)
            rec["paths"][K].append(cm.prepare(tab, R_flat, ALPHA, mA.Gamma, LAM, mA.neff, mA.n_group, W))
            if K == 1:
                p0 = rec["paths"][1][0]
                rec["mirrors"]["flat"] = dict(T=p0.T_det, L=p0.L_mean, GL=mA.Gamma * p0.L_mean, loss=BL)
                for lab, Rf in (("two-material", R2), ("tri-index", Rtri)):
                    r = cs.evaluate(tab, Rf, ALPHA, mA.Gamma)
                    rec["mirrors"][lab] = dict(T=r.T_det, L=r.L_mean, GL=r.L_eff_gas, loss=path_loss(tab, LUT[lab]))
            del tab
    out[name] = rec
    p = rec["paths"][1][0]
    print(f"[{time.time()-t00:4.0f}s] {name:15s} lyap={rec['lyap']:.3f}  T={p.T_det:.3f} <L>={p.L_mean*100:.0f}cm paths/mode={p.n_paths_eff()[0]:.1f} M={p.M_spat():.0f} C(1MHz)={cm.contrast(p,1e6,True):.3f} | "
          + " ".join(f"{k}: GL={v['GL']*100:.1f}cm loss/bounce={v['loss']:.4f}" for k, v in rec["mirrors"].items() if k != "flat"), flush=True)
    pickle.dump(out, open("results/chaos_scan.pkl", "wb"))
print("DONE", flush=True)
