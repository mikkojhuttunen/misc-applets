"""Data for the Fig. 6 Playground artifact: the spectra of fig6_spectra_noise.py (per ppm analyte and each background gas
separately) and the Chaos Lab noise levels and exit paths. Writes results/figs_prelim/fig6_playground_data.json and the
self-contained page results/figs_prelim/fig6-playground.html (template: fig6_playground_template.html)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "HITRAN"))
import json
from pathlib import Path
import numpy as np
import hitran_lbl as hl

ROOT = Path(__file__).resolve().parents[1]
GHZ, WING = 29.9792458, 25.0
LINES = [("NH3", "NH3", 6528.77, "NH3"), ("CH4 1.6", "CH4", 6114.652, "CH4"), ("CH4 3.3", "CH4", 3017.80, "CH4"),
         ("C2H6", "C2H6", 2996.852, "C2H6"), ("NO", "NO", 1900.076, "NO")]
BG = {"H2O": 0.06, "CO2": 0.045, "CH4": 1.9e-6}
r = lambda a: [float(f"{v:.4g}") for v in a]
strips = []
for key, sp, nu0, _ in LINES:
    nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 601)
    bg = {}
    for g, xg in BG.items():
        if (g == "CH4" and sp == "CH4") or not hl.find_tables(g, nu.min(), nu.max()):
            continue
        bg[g] = r(hl.alpha(nu, g, xg, wing_cm=WING))
    files = sorted({str(f.relative_to(ROOT.parent)) for g in [sp, *bg] for f in hl.find_tables(g, nu.min(), nu.max())})
    strips.append(dict(key=key, species=sp, nu0=nu0, lam_um=1e4 / nu0, x_ghz=r((nu - nu0) * GHZ), hitran_files=files,
                       analyte_per_ppm=r(hl.alpha(nu, sp, 1e-6, wing_cm=WING)), bg=bg, bg_x=BG))
d = json.load(open(ROOT / "data" / "chaoslab_nh3.json"))
cell = lambda c: dict(gamma=c["gamma"], L_mean_m=c["L_mean_m"], T_trans=c["T_trans"], M2=c["M2"], Ls_m=r(c["Ls_m"]), Ws=r(c["Ws"]),
                      env={k: r(v["env"]) for k, v in c["scen"].items()}, floor={k: v["floor"] for k, v in c["scen"].items()})
out = dict(strips=strips, T=r(d["T"]), P_in=d["settings"]["circle"]["P_in"], n_eff=d["settings"]["circle"]["n_eff"],
           eps=d["settings"]["circle"]["f_parasitic"], current=cell(d["cells"]["circle_current"]), regular=cell(d["cells"]["circle_regular"]),
           chaos=dict(source=d["settings"]["source"], drift_rad_s=d["settings"]["drift_rad_s"],
                      circle={k: d["settings"]["circle"][k] for k in ("c_radius", "n_facets", "c_theta", "R_mirror", "dither", "waveform", "f_theta",
                                                                       "dlam", "lwave", "f_lambda", "f_parasitic", "stray", "n_inc", "inc_radius", "inc_dn", "inc_sign", "inc_seed")}))
p = ROOT / "results" / "figs_prelim" / "fig6_playground_data.json"; p.write_text(json.dumps(out, separators=(",", ":")))
print(p, p.stat().st_size // 1024, "kB")
tpl = Path(__file__).with_name("fig6_playground_template.html").read_text()
h = p.with_name("fig6-playground.html"); h.write_text(tpl.replace("__DATA__", p.read_text().replace("</", "<\\/")))
print(h)
