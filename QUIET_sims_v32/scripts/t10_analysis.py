"""Task 10 analysis: per design variant the static both-dither fringe (eps 1e-6, no drift), transmission, Gamma<L>, speckle
grains M, and the static NH3 floor fringe / (alpha Gamma<L> s) with s the line peak kept under the wavelength dither
(Task 9b: 0.90 at +-20 pm, 0.80 at +-35 pm); medians and quartiles over 30 layouts, the regular cell for reference;
gain = old-dither floor / variant floor. -> results/t10/summary.json, results/t10/table.md"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

LN = A.load(ROOT / "results/t7/lines.json")
GAS = {"nir": [("NH3", LN["NH3_1.53"]["alpha_peak_per_ppm_cm"], {"20pm": 0.896, "35pm": 0.804, "81pm": 0.518})],
       "mir": [("CH4", LN["CH4_3.31"]["alpha_peak_per_ppm_cm"], {"20pm": 0.993, "35pm": 0.979, "81pm": 0.921}),
               ("C2H6", LN["C2H6_3.34Q"]["alpha_peak_per_ppm_cm"], {"20pm": 0.986, "35pm": 0.959, "81pm": 0.824})]}
LABEL = {"base": "reference: 4 mm, 1 in, 1 out, 16 × 50 µm × 0.01, 75 µm ±60 mrad ±100 µm", "old_dither": "old dither (40 µm, ±2 mrad)",
         "in2": "2 input ports", "in3": "3 input ports", "in4": "4 input ports", "out1r": "1 output, random position", "out2": "2 outputs, random",
         "out4": "4 outputs, random", "out8": "8 outputs, random", "inc8": "8 inclusions", "inc32": "32 inclusions", "inc64": "64 inclusions",
         "r25": "inclusions r 25 µm", "r25n64": "64 × r 25 µm", "r100": "inclusions r 100 µm", "dn005": "|Δn| 0.005", "dn02": "|Δn| 0.02", "dn04": "|Δn| 0.04",
         "d7": "7 mm cell, 16 inclusions", "d7s": "7 mm cell, 49 inclusions", "d10": "10 mm cell, 16 inclusions", "d10s": "10 mm cell, 100 inclusions",
         "d10_in2": "10 mm, 2 input ports", "d10_inc8": "10 mm, 8 inclusions", "d10_r25": "10 mm, r 25 µm", "d14": "14 mm cell", "d20": "20 mm cell",
         "d20_r25": "20 mm, r 25 µm", "d10_old": "10 mm, old dither", "d20_old": "20 mm, old dither"}
LABEL.update({f"of{i}": f"output on facet {i}" for i in range(1, 24)})
LABEL.update({"in2g": "2 inputs, facets 0 + 19", "in3g": "3 inputs, facets 0 + 19 + 4", "d20r25_in2g": "20 mm, r 25 µm, 2 inputs (0 + 19)",
              "d20r25_in3g": "20 mm, r 25 µm, 3 inputs (0 + 19 + 4)", "d20_r25": "20 mm, r 25 µm"})
q = lambda v, p: float(np.quantile(v, p))
out, md = {}, []
for f in sorted((ROOT / "results/t10").glob("*_R*_0.3dB*.json")):
    d = json.load(open(f))
    key = f.stem
    gases = GAS[d["band"]]
    DLS = ("20pm", "35pm", "81pm") if d["band"] == "mir" else ("20pm", "35pm")
    out[key] = {}
    for nm, V in d["variants"].items():
        ch = [x for x in V["chaotic"] if "error" not in x]
        reg = V["regular"]
        row = {"n": len(ch), "T": q([x["T"] for x in ch], .5), "GL_cm": q([x["GL_cm"] for x in ch], .5), "M": q([np.mean(x["M"]) for x in ch], .5)}
        for dl in DLS:
            fr = [x["fringe"][dl]["both"] for x in ch]
            row[dl] = {"fringe": [q(fr, .25), q(fr, .5), q(fr, .75)], "below_5e-7": float(np.mean(np.array(fr) < 5e-7)),
                       "spectral": q([x["fringe"][dl]["spectral"] for x in ch], .5), "input": q([x["fringe"][dl]["input"] for x in ch], .5),
                       "regular_fringe": reg["fringe"][dl]["both"] if "error" not in reg else None, "ppb": {}, "regular_ppb": {}}
            for g, al, sm in gases:
                ppb = [1e3 * x["fringe"][dl]["both"] / (al * x["GL_cm"] * sm[dl]) for x in ch]
                row[dl]["ppb"][g] = [q(ppb, .25), q(ppb, .5), q(ppb, .75)]
                row[dl]["regular_ppb"][g] = 1e3 * reg["fringe"][dl]["both"] / (al * reg["GL_cm"] * sm[dl]) if "error" not in reg else None
        out[key][nm] = row
    g0 = gases[0][0]
    for dl in (("35pm", "81pm") if d["band"] == "mir" else ("35pm",)):
        rk = "old_dither" if "old_dither" in out[key] else ("of11" if "of11" in out[key] else "base")
        ref = out[key][rk][dl]["ppb"][g0][1]
        md.append(f"\n#### {key}: static both-dither fringe and {'/'.join(g for g, _, _ in gases)} floor, ±{dl[:-2]} pm wavelength dither (median [IQR] over 30 layouts)\n")
        md.append(f"| variant | T | Γ⟨L⟩ cm | M | fringe | < 5e-7 | {' | '.join(g + ' static floor ppb' for g, _, _ in gases)} | gain vs {LABEL.get(rk, rk).split(':')[0]} | regular cell {g0} ppb (indicative) |")
        md.append("|---|---|---|---|---|---|" + "---|" * len(gases) + "---|---|")
        for nm, r in out[key].items():
            x = r[dl]
            md.append(f"| {LABEL.get(nm, nm)} | {r['T']:.3f} | {r['GL_cm']:.2f} | {r['M']:.0f} | {x['fringe'][1]:.2g} [{x['fringe'][0]:.2g}–{x['fringe'][2]:.2g}] | {100 * x['below_5e-7']:.0f} % | "
                      + " | ".join(f"{x['ppb'][g][1]:.3g} [{x['ppb'][g][0]:.2g}–{x['ppb'][g][2]:.2g}]" for g, _, _ in gases)
                      + f" | {ref / x['ppb'][g0][1]:.1f}× | {x['regular_ppb'][g0]:.3g} |")
A.dump(ROOT / "results/t10/summary.json", out)
(ROOT / "results/t10/table.md").write_text("\n".join(md) + "\n")
print("ok", len(md))
