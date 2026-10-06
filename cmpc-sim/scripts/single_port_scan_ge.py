"""K mutually incoherent beams: (ports) K separate input ports; (angles) ONE port, K beams at different angles;
(positions) ONE port, K narrow (divergent) sub-apertures at the same centre angle. Si TM 250 nm, 1.55 um, curved-facet cell
(rho = 5 cm) and regular cell, 30 um port, 0.1 dB/cm. Writes single_port_scan_ge.pkl."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import pickle, time
import numpy as np
import cmpc_sim as cs, coherence_model as cm

import json, mirror_design as md
LAM = 5.2629e-6
m = cs.membrane_mode("Ge", 300e-9, LAM, "TE")
W = 100e-6; TH0 = LAM / (m.neff * W); CHI = np.pi / 2 - np.pi * 6 / 24
ALPHA = cs.dB_per_cm_to_alpha(0.013); BL = 3e-4
Rf = md.R_function(LAM, json.load(open("data/mirror_designs.json"))["130_16"], BL)
NR, NB = 5000, 1000
CELLS = {"curved": dict(curvature=20.0)}


def beam(cell, kind, K, k, seed):
    P = cell.perimeter; s_out = cell.h + 10.5 * 2 * cell.h
    base = dict(seed=seed, max_path=5.0, keep_start=False, s_out=s_out)
    if kind == "ports":
        ports = [cell.h + j * P / K for j in range(K)]
        return cs.trace_rays(cell, W, NR, NB, TH0, theta_c=CHI, s_in=ports[k], extra_ports=[q for j, q in enumerate(ports) if j != k], **base)
    if kind == "angles":
        return cs.trace_rays(cell, W, NR, NB, TH0, theta_c=CHI + (k - (K - 1) / 2) * 3 * TH0, s_in=cell.h, **base)
    lw = 0.9 * W / K
    return cs.trace_rays(cell, W, NR, NB, LAM / (m.neff * lw), theta_c=CHI, s_in=cell.h, launch_w=lw, launch_off=(k - (K - 1) / 2) * W / K, **base)


out = {}; t0 = time.time()
for cname, kw in CELLS.items():
    cell = cs.SegmentedCell(5e-3, 24, seed=1, **kw)
    for kind in ("ports", "angles", "positions"):
        for K in (1, 2, 4, 8):
            if K == 1 and kind != "ports":
                out[(cname, kind, 1)] = out[(cname, "ports", 1)]; continue
            pl = []
            for k in range(K):
                tab = beam(cell, kind, K, k, 100 + 10 * K + k)
                pl.append(cm.prepare(tab, Rf, ALPHA, m.Gamma, LAM, m.neff, m.n_group, W)); del tab
            out[(cname, kind, K)] = pl
            print(f"[{time.time()-t0:4.0f}s] {cname:7s} {kind:9s} K={K}: <T>={np.mean([p.T_det for p in pl]):.3f} <L>={np.mean([p.L_mean for p in pl])*100:.1f}cm contrast={cm.contrast(cm.merge_incoherent(pl) if K>1 else pl[0],1e6,True):.3f}", flush=True)
    pickle.dump(out, open("results/single_port_scan_ge.pkl", "wb"))
print("DONE", flush=True)
