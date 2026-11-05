"""Task 2 sensitivity: static fringe (both dithers) of the mid-IR circle (1000 nm SiNx, 3315.1 nm) against the
wavelength-dither depth, read from the stored fringe-vs-depth curves of the set-B runs (no new simulation); the nominal
+-20 pm is +-0.55 GHz, +-81 pm is +-2.2 GHz. -> results/t2/mir_dither_sensitivity.json"""
import math
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

out = {}
for R in ("0.95", "0.97", "0.995"):
    for dB in ("0.3", "0.5"):
        d = A.load(ROOT / f"results/t2/circle_mir_R{R}_{dB}dB.json")
        dl = np.array(d["stat_dl"]) * 1e12
        ch = A.chaotic(d)
        for pm in (20, 40, 81, 160):
            j = int(np.argmin(np.abs(np.log(dl) - math.log(pm))))
            v = [x["stat"]["both"][j] for x in ch]
            st = A.stats(v)
            out[f"R{R}_{dB}dB_{pm}pm"] = st | {"dither_pm": float(dl[j]), "dither_GHz": float(dl[j] * 1e-12 * 299792458 / 3.3151e-6 ** 2 / 1e9),
                                              "regular": d["regular"]["stat"]["both"][j], "gain": d["regular"]["stat"]["both"][j] / st["median"],
                                              "share_below_5e-7": float(np.mean(np.array(v) < 5e-7))}
A.dump(ROOT / "results/t2/mir_dither_sensitivity.json", out)
print("ok", len(out))
