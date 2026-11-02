"""Step 1: angle-dependent trench-DBR reflectivity R(sin chi) of the membrane cells.

For every n_eff (1.55 um: 1.36, 1.53, 1.66, 1.74 = SiNx 400, 500, 600, 700 nm; 3.3 um: 1.46, 1.59, 1.71 = 1.0, 1.2,
1.5 um) and every cell (circle CMPC, Herriott), the trench design (N periods up to 20, odd quarter-wave gap and tooth
orders, design angle, linear chirp, two-band) that maximises the bounce-averaged R over the cell's distribution of
|sin chi| (results/step1/anglehist_<cell>.json, from scripts/angle_hist.js). Total internal reflection above the critical
angle is part of the transfer matrix. Also: roughness (sidewall rms 0, 2, 5, 10 nm), the share of bounces above the
critical angle, and the vertical-diffraction estimate (qs/mirror.py: isolated-gap bound). Writes
results/step1_mirrors.json and the R tables results/step1/Rtable_<cell>_neff<n>.json ({s, R}) used by the ensembles.
python scripts/step1_mirrors.py [--cells circle,herriott] [--quick]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from design_calcs import slab  # noqa: E402
from qs import mirror as Mi  # noqa: E402
from qs.metrics import RESULTS  # noqa: E402

FILMS = [  # (wavelength, core index, thickness, nominal n_eff)
    (1.55e-6, 2.0, 400e-9, 1.36), (1.55e-6, 2.0, 500e-9, 1.53), (1.55e-6, 2.0, 600e-9, 1.66), (1.55e-6, 2.0, 700e-9, 1.74),
    (3.3e-6, 1.97, 1.0e-6, 1.46), (3.3e-6, 1.97, 1.2e-6, 1.59), (3.3e-6, 1.97, 1.5e-6, 1.71)]
NS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
SIGMAS_NM = (0.0, 2.0, 5.0, 10.0)
S_TABLE = np.unique(np.r_[np.linspace(0, 1, 1001), [1 / f[3] for f in FILMS]])


def film_info(lam, n, t, nominal):
    ne, g = slab.gam(n, t, lam, "TM")
    return {"lambda_um": lam * 1e6, "core_index": n, "thickness_nm": t * 1e9, "n_eff_nominal": nominal, "n_eff_slab": float(ne), "gamma": float(g),
            "critical_angle_deg": float(np.degrees(np.arcsin(1 / nominal)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="circle,herriott")
    ap.add_argument("--quick", action="store_true", help="fewer N and no two-band search (tests)")
    ap.add_argument("--out", default=str(RESULTS / "step1_mirrors.json"))
    a = ap.parse_args()
    Ns = (1, 4, 20) if a.quick else NS
    out = {"films": [film_info(*f) for f in FILMS], "N_values": list(Ns), "cells": {}, "notes": [
        "R_avg: bounce average of R over the |sin chi| distribution of the detected light; R_eff: exp(<ln R>) (sets T)",
        "design: odd quarter-wave orders m_gap, m_tooth at sin chi = s1 (second half of the periods at s2 for two-band), "
        "linear chirp of both thicknesses; thicknesses in nm",
        "period_over_lambda < 0.5: first-order mirror, stop-band Bloch mode below the light line (no radiating orders)",
        "diffraction: isolated-gap bound (every air-gap traversal loses 1 - eta of the power, eta from the exact scalar "
        "angular-spectrum propagation of the TM0 profile); pessimistic, see qs/mirror.py"]}
    Path(RESULTS / "step1").mkdir(parents=True, exist_ok=True)
    for cell in a.cells.split(","):
        h = json.loads((RESULTS / "step1" / f"anglehist_{cell}.json").read_text())
        s, p = np.array(h["s"]), np.array(h["p"])
        cres = {"angle_hist": f"results/step1/anglehist_{cell}.json", "films": {}}
        for (lam, ncore, t, ne) in FILMS:
            t0 = time.time()
            key = f"{ne:.2f}"
            above = float(p[s > 1 / ne].sum())
            byN = {}
            for N in Ns:
                b = Mi.optimise(ne, s, p, N, two_band=not a.quick, chirps=(0.0,) if a.quick else (0.0, -0.1, 0.1, -0.2, 0.2))
                b["period_over_lambda"] = Mi.period(ne, b["d_gap"], b["d_tooth"])
                b["d_gap_nm"] = [x * lam * 1e9 for x in b["d_gap"]]
                b["d_tooth_nm"] = [x * lam * 1e9 for x in b["d_tooth"]]
                byN[N] = b
            best = max(byN.values(), key=lambda b: b["R_avg"])
            R_tab, _, _ = Mi.design_R(ne, S_TABLE, best["N"], best["s1"], best["m_gap"], best["m_tooth"], best["chirp"], best["s2"])
            rough = {}
            for sg in SIGMAS_NM:
                Rr = R_tab * Mi.roughness_factor(ne, S_TABLE, sg * 1e-9, lam)
                rough[sg] = Mi.bounce_average(np.interp(s, S_TABLE, Rr), p)[0]
            eta = Mi.gap_coupling(ncore, t, 1.0 * lam, "TM")
            etaL = lambda z, f=eta, L=lam: f(np.asarray(z) * L)          # z in wavelengths
            Rd, _, _ = Mi.design_R(ne, s, best["N"], best["s1"], best["m_gap"], best["m_tooth"], best["chirp"], best["s2"], eta=etaL)
            bd = Mi.optimise(ne, s, p, 8, eta=etaL, two_band=False, chirps=(0.0,), orders=(1,))
            tab_file = RESULTS / "step1" / f"Rtable_{cell}_neff{key}.json"
            tab_file.write_text(json.dumps({"cell": cell, "n_eff": ne, "lambda_um": lam * 1e6, "design": best,
                                            "s": S_TABLE.tolist(), "R": R_tab.tolist()}))
            cres["films"][key] = {
                "lambda_um": lam * 1e6, "thickness_nm": t * 1e9, "share_above_critical": above,
                "by_N": {N: {"R_avg": b["R_avg"], "R_eff": b["R_eff"], "m_gap": b["m_gap"], "m_tooth": b["m_tooth"], "s1": b["s1"], "s2": b["s2"],
                             "chirp": b["chirp"], "period_over_lambda": b["period_over_lambda"],
                             "gap_nm": float(np.mean(b["d_gap_nm"])), "tooth_nm": float(np.mean(b["d_tooth_nm"]))} for N, b in byN.items()},
                "best": {k: best[k] for k in ("N", "R_avg", "R_eff", "m_gap", "m_tooth", "s1", "s2", "chirp", "period_over_lambda", "d_gap_nm", "d_tooth_nm")},
                "R_sub_critical_avg": float(np.sum((p * np.interp(s, S_TABLE, R_tab))[s <= 1 / ne]) / max(np.sum(p[s <= 1 / ne]), 1e-300)),
                "roughness_R_avg": rough,
                "diffraction_bound": {"eta_quarter_wave_gap": float(eta(lam / 4)), "R_avg_best_design": Mi.bounce_average(Rd, p)[0],
                                      "R_avg_reoptimised_N8_first_order": bd["R_avg"]},
                "R_table": str(tab_file.relative_to(RESULTS.parent))}
            print(f"{cell:8s} n_eff {ne:.2f}: above critical {100 * above:4.1f} %  R_avg " +
                  " ".join(f"N{N}:{b['R_avg']:.5f}" for N, b in byN.items()) +
                  f"  | best N{best['N']} m{best['m_gap']}/{best['m_tooth']}  rough 5 nm {rough[5.0]:.5f}  diffraction bound {cres['films'][key]['diffraction_bound']['R_avg_reoptimised_N8_first_order']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        out["cells"][cell] = cres
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() and a.cells != "circle,herriott" else None
    if prev:
        prev["cells"].update(out["cells"])
        out = prev
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
