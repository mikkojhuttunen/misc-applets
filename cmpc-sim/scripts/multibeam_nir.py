"""Single-port multi-beam scan, Si TM0 250 nm membrane, 1.55 um, curved-facet cell (rho = 5 cm), 30 um port.
Long-path case: 0.04 dB/cm (the loss needed for 50 cm gas-equivalent path); short-path case: 0.3 dB/cm.
kinds: ports (K separate input ports) / angles (ONE port, K beams at different angles, spacing 2.2 mode widths)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import pickle, time
import numpy as np
import cmpc_sim as cs, coherence_model as cm

LAM = 1.55e-6
m = cs.membrane_mode("Si", 250e-9, LAM, "TM")
W = 30e-6; TH0 = LAM / (m.neff * W); CHI = np.pi / 2 - np.pi * 7 / 24
BL = 3e-4; Rf = lambda s: np.full_like(s, 1 - BL)
cell = cs.SegmentedCell(5e-3, 24, seed=1, curvature=20.0)
NR, NB = 5000, 1000


def beam(kind, K, k, seed):
    P = cell.perimeter; base = dict(seed=seed, max_path=5.0, keep_start=False, s_out=cell.h + 18 * cell.h)
    if kind == "ports":
        ports = [cell.h + j * P / K for j in range(K)]
        return cs.trace_rays(cell, W, NR, NB, TH0, theta_c=CHI, s_in=ports[k], extra_ports=[q for j, q in enumerate(ports) if j != k], **base)
    return cs.trace_rays(cell, W, NR, NB, TH0, theta_c=CHI + (k - (K - 1) / 2) * 2.2 * TH0, s_in=cell.h, **base)


out = {}; t0 = time.time()
plan = [("long", 0.04, "ports", (1, 2, 4, 8)), ("long", 0.04, "angles", (2, 4, 8, 16)), ("short", 0.3, "angles", (1, 8))]
for tag, adb, kind, Ks in plan:
    alpha = cs.dB_per_cm_to_alpha(adb)
    for K in Ks:
        pl = []
        for k in range(K):
            tab = beam(kind, K, k, 200 + 10 * K + k)
            pl.append(cm.prepare(tab, Rf, alpha, m.Gamma, LAM, m.neff, m.n_group, W)); del tab
        out[(tag, kind, K)] = pl
        print(f"[{time.time()-t0:4.0f}s] {tag} {kind:6s} K={K:2d}: <T>={np.mean([p.T_det for p in pl]):.3f} <L>={np.mean([p.L_mean for p in pl])*100:.1f}cm", flush=True)
    if tag == "long" and kind == "ports": out[("long", "angles", 1)] = out[("long", "ports", 1)]
    pickle.dump(out, open("results/multibeam_nir.pkl", "wb"))
print("DONE", flush=True)
