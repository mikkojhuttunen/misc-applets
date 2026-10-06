"""Suppression budget: curved-facet cell (rho = 5 cm) with the tri-index hole-free TE mirror, Ge 300 nm TE0 at 5.263 um.
K mutually incoherent beams (ports spaced P/8) at port widths 100 um and 400 um. Writes budget_scan.pkl."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import json, pickle, time
import numpy as np
import cmpc_sim as cs, coherence_model as cm, mirror_design as md

LAM = 5.2629e-6
mA = cs.membrane_mode("Ge", 300e-9, LAM, "TE")
CHI = np.pi / 2 - np.pi * 6 / 24
ALPHA = cs.dB_per_cm_to_alpha(0.013)
des = json.load(open("data/mirror_designs.json"))["130_16"]
Rtri = md.R_function(LAM, des, 3e-4)
cell = cs.SegmentedCell(5e-3, 24, curvature=1 / 0.05)
NR, NB = 5000, 1000
out = {}
t0 = time.time()
for W, KSET in ((100e-6, (1, 2, 4, 8)), (400e-6, (1, 4))):
    th0 = LAM / (mA.neff * W)
    for K in KSET:
        P = cell.perimeter
        ports = [cell.h + j * P / K for j in range(K)]
        plist = []
        for k in range(K):
            tab = cs.trace_rays(cell, W, NR, NB, th0, seed=11 + 10 * K + k, max_path=5.0, keep_start=False, theta_c=CHI, s_in=ports[k],
                                s_out=cell.h + 10.5 * 2 * cell.h, extra_ports=[q for j, q in enumerate(ports) if j != k])
            plist.append(cm.prepare(tab, Rtri, ALPHA, mA.Gamma, LAM, mA.neff, mA.n_group, W))
            del tab
        out[(W, K)] = plist
        print(f"[{time.time()-t0:4.0f}s] w={W*1e6:.0f}um K={K}: mean T={np.mean([p.T_det for p in plist]):.3f} <L>={np.mean([p.L_mean for p in plist])*100:.0f}cm M(beam)={np.mean([p.M_spat() for p in plist]):.0f}", flush=True)
        pickle.dump(out, open("results/budget_scan.pkl", "wb"))
print("DONE", flush=True)
