"""Fig. 2 and Fig. 3 artifacts with the v32 set-B data as default and the earlier data sets selectable for comparison.

python scripts/build_fig23_v32.py            (reads artifacts/fig{2,3}_src_live.html, writes artifacts/fig{2,3}_v32.html)

The sources are the live claude.ai artifacts as last edited (Fig. 2 BiVxtNey99hyrtPLYcvhm8, Fig. 3 9sezcN5zEXTv9Som5LCpBr).
Data sets (selector "Data set" in both figures):
  B03 / B05  v32 set B: 500 nm SiNx, TM0 at 1.5317 um (n_eff 1.5391, Gamma_E 0.4532), constant R 99.5 / 97 / 95 %,
             0.3 or 0.5 dB/cm (results/t2/*_nir_*); default B03.
  S10        QUIET step 10: 600 nm SiNx, R(|sin chi|) tables, ideal / target / minimum (../quiet_sims/results/step10);
             this is what the figures showed before.
  A          set A, the parameters of the published Fig. 2f / Fig. 3 (n_eff 2, Gamma 0.2, R 99.5 %, no loss; results/t1,
             the bit-for-bit rerun of the published ensembles).
Fig. 3 also gets a noise-model selector: shot noise sqrt(q / (P_in T_cell T)) (the earlier figures) or the proposal T1.3
detector model sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T)), P_d = P_in 10^(-IL/10), NEP 14 pW/rtHz (set B default:
IL 14 dB, 1 mW). The analysis table is recomputed in the page from per-layout values for the selected noise model.
The robustness panels (Fig. 2 g-i, Fig. 3 c-d) stay on the step-11 basis (600 nm-class films, R(chi) tables).
"""
import json
import math
import re

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T2 = ROOT / "results" / "t2"
T1 = ROOT / "results" / "t1"
S10 = ROOT.parent / "quiet_sims" / "results" / "step10"
ART = ROOT / "artifacts"
MODES = ["none", "angle", "spectral", "both"]


def q(a, p):
    b = sorted(a)
    i = (len(b) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(b) - 1)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def load(p):
    return json.load(open(p))


# ------------------------------------------------------------------ data-set definitions
B_SLOTS = {"ideal": 0.995, "target": 0.97, "lossy": 0.95}          # Fig. 2 slots: lightest (ideal) .. darkest (lossy)


def fig2_files(ds):
    if ds in ("B03", "B05"):
        dB = "0.3" if ds == "B03" else "0.5"
        return {k: {c: T2 / f"{c}_nir_R{R}_{dB}dB.json" for c in ("circle", "herriott")} for k, R in B_SLOTS.items()}
    if ds == "S10":
        return {k: {c: S10 / f"{c}_{f}.json" for c in ("circle", "herriott")} for k, f in (("ideal", "ideal"), ("target", "target"), ("lossy", "minimum"))}
    return {"lossy": {c: T1 / f"{c}_A.json" for c in ("circle", "herriott")}}


DSM2 = {
    "B03": {"label": "v32 set B: 500 nm SiNx, 1.53 µm, 0.3 dB/cm (default)", "slots": {"ideal": "R 99.5 %", "target": "R 97 %", "lossy": "R 95 %"},
            "live": {"n_eff": 1.5391, "gamma": 0.4532, "R": B_SLOTS, "dB": {"ideal": 0.3, "target": 0.3, "lossy": 0.3}, "sp": "NH3"}, "liveCase": "target", "defaultOn": ["target"],
            "desc": "v32 set B (realistic loss budget): free-standing 500 nm SiNx membrane, TM₀ at 1.5317 µm (n_eff 1.539, Γ_E 0.453 from the slab solver), "
                    "constant mirror reflectivity per bounce (no angle dependence, no TIR bonus: conservative), membrane loss 0.3 dB/cm. Cases: R 99.5 %, 97 %, 95 %; "
                    "default shows R 97 % (paired with Fig. 3); R 95 % is the T1.3 baseline."},
    "B05": {"label": "v32 set B: 500 nm SiNx, 1.53 µm, 0.5 dB/cm", "slots": {"ideal": "R 99.5 %", "target": "R 97 %", "lossy": "R 95 %"},
            "live": {"n_eff": 1.5391, "gamma": 0.4532, "R": B_SLOTS, "dB": {"ideal": 0.5, "target": 0.5, "lossy": 0.5}, "sp": "NH3"}, "liveCase": "target", "defaultOn": ["target"],
            "desc": "v32 set B as above with membrane loss 0.5 dB/cm. Cases: R 99.5 %, 97 %, 95 %."},
    "S10": {"label": "earlier: step 10, 600 nm SiNx, R(χ) tables", "slots": {"ideal": "ideal", "target": "target", "lossy": "minimum"},
            "live": {"rtab": True, "n_eff": 1.66, "gamma": 0.313, "dB": {"ideal": 0.0, "target": 0.1, "lossy": 0.3}, "sp": "CH4"}, "liveCase": "target",
            "desc": "Earlier version (QUIET step 10): free-standing 600 nm SiNx membrane, TM₀ (n_eff 1.66, Γ 0.31), trench-DBR mirrors with the angle-dependent "
                    "reflectivity R(χ) of a 20-period design (total internal reflection above θc = 37°). Ideal: designed R(χ), no membrane loss; target: designed R(χ), "
                    "0.1 dB/cm; minimum: R capped at 95 % below θc, 0.3 dB/cm."},
    "A": {"label": "published (set A: n_eff 2, Γ 0.2, R 99.5 %, no loss)", "slots": {"lossy": "set A"},
          "live": {"n_eff": 2.0, "gamma": 0.2, "R": {"lossy": 0.995}, "dB": {"lossy": 0.0}, "sp": "CH4"}, "liveCase": "lossy",
          "desc": "Parameters of the published Fig. 2f / Fig. 3 (proposal v32 Sec. 3.2, set A): n_eff 2, Γ 0.2, R 99.5 % per bounce, no membrane loss, 1.654 µm. "
                  "Ensembles: the bit-for-bit rerun of the published 30 layouts."},
}
ORDER = ["B03", "B05", "S10", "A"]
CAVEAT = (" Caveat (REPORT Task 5b): the engine averages the angle-dither fringe along single rays; for the 40 µm input beam a "
          "±2 mrad tilt stays inside the beam divergence (7.9 mrad), and wave optics limits the angle-dither factor to ≥ 0.98, so the "
          "“angle” and “both” results are optimistic: with both dithers the wave-bounded fringe is about the wavelength-dither one.")

for _k in ("B03", "B05"):
    DSM2[_k]["desc"] += CAVEAT


def vals(r):
    a = 2 * 1e-3 / (r["grains"] ** 0.5)
    return {"none": a, "spectral": a * r["V_lambda"], "angle": a * r["V_theta"], "both": a * r["V_joint"],
            "fl_angle": a * r["V_floor"], "fl_both": a * r["V_floor_both"], "T": r["T_trans"], "grains": r["grains"],
            "L": r["L_mean_m"], "GL_cm": r["gamma"] * r["L_mean_m"] * 100, "R_bounce": r.get("R_bounce", float("nan"))}


def ens2(ds):
    E = {}
    for key, cells in fig2_files(ds).items():
        E[key] = {}
        for cell, p in cells.items():
            d = load(p)
            ch = [vals(dict(x, gamma=x.get("gamma", d["gamma"]))) for x in d["chaotic"] if "error" not in x]
            st = {k: {"med": q([x[k] for x in ch], .5), "q1": q([x[k] for x in ch], .25), "q3": q([x[k] for x in ch], .75),
                      "min": min(x[k] for x in ch), "max": max(x[k] for x in ch)} for k in ch[0]}
            E[key][cell] = {"regular": vals(dict(d["regular"], gamma=d["regular"].get("gamma", d["gamma"]))), "chaotic": st, "n": len(ch),
                            "seeds": {k: [x[k] for x in ch] for k in ("both", "angle", "spectral")}}
    return E


# ------------------------------------------------------------------ Fig. 3 packs (circle, NH3)
def fig3_files(ds):
    if ds in ("B03", "B05"):
        dB = "0.3" if ds == "B03" else "0.5"
        # slot keys of the drawing code: current = solid, lossy = dashed, target = dash-dot
        return {"current": T2 / f"circle_nir_R0.97_{dB}dB.json", "lossy": T2 / f"circle_nir_R0.95_{dB}dB.json",
                "target": T2 / f"circle_nir_R0.995_{dB}dB.json", "mir": T2 / f"circle_mir_R0.97_{dB}dB.json",
                "mir_l": T2 / f"circle_mir_R0.95_{dB}dB.json", "mir_t": T2 / f"circle_mir_R0.995_{dB}dB.json"}
    if ds == "S10":
        return {"current": S10 / "circle_target.json", "lossy": S10 / "circle_minimum.json", "target": S10 / "circle_ideal.json"}
    return {"current": T1 / "circle_A.json"}


DSM3 = {
    "B03": {"label": DSM2["B03"]["label"], "n_eff": 1.5391, "n_eff_mir": 1.4253, "noise": {"model": "det", "IL": 14, "P": 10}, "T": 15, "win": True,
            "names": {"current": "R 97 %, 0.3 dB/cm (solid)", "lossy": "R 95 %, 0.3 dB/cm (dashed)", "target": "R 99.5 %, 0.3 dB/cm (dash-dot)"},
            "short": {"lossy": "R 95 %", "target": "R 99.5 %"},
            "desc": "v32 set B: chaotic circular cell, 500 nm SiNx membrane, TM₀ at 1.5317 µm (n_eff 1.539, Γ_E 0.453), constant R per bounce, 0.3 dB/cm, NH₃ line, "
                    "10 mW, marked window 15–30 s; medians and min–max bands over 30 inclusion layouts. Panel (b): solid R 97 %, dashed R 95 % (T1.3 baseline), "
                    "dash-dot R 99.5 %. Panel (a): the 3.3 µm strips use the simulated 3.3 µm cell (1000 nm SiNx, n_eff 1.425, Γ_E 0.553, R 97 %, "
                    "±0.55 GHz wavelength dither), the near-IR strips the 1.53 µm cell. Fringe residual against T: the stray-light fringe phase is assumed to drift at 0.1 rad/s (thermal), so the static fringe averages down for T ≳ 10 s; untick “fringe drift” for the conservative case without drift (residual held at the static floor). "},
    "B05": {"label": DSM2["B05"]["label"], "n_eff": 1.5391, "n_eff_mir": 1.4253, "noise": {"model": "det", "IL": 14, "P": 10}, "T": 15, "win": True,
            "names": {"current": "R 97 %, 0.5 dB/cm (solid)", "lossy": "R 95 %, 0.5 dB/cm (dashed)", "target": "R 99.5 %, 0.5 dB/cm (dash-dot)"},
            "short": {"lossy": "R 95 %", "target": "R 99.5 %"},
            "desc": "v32 set B as above with 0.5 dB/cm membrane loss, 10 mW, window 15–30 s. Panel (b): solid R 97 %, dashed R 95 %, dash-dot R 99.5 %. Fringe residual against T: the stray-light fringe phase is assumed to drift at 0.1 rad/s (thermal), so the static fringe averages down for T ≳ 10 s; untick “fringe drift” for the conservative case without drift (residual held at the static floor). "},
    "S10": {"label": DSM2["S10"]["label"], "n_eff": 1.66, "noise": {"model": "shot", "IL": 14, "P": 1},
            "names": {"current": "target (designed R(χ), 0.1 dB/cm; solid)", "lossy": "minimum (R ≤ 95 % below θc, 0.3 dB/cm; dashed)", "target": "ideal (designed R(χ), no loss; dash-dot)"},
            "short": {"lossy": "min. spec", "target": "ideal"},
            "desc": "Earlier version (QUIET step 10): 600 nm SiNx membrane, TM₀ (n_eff 1.66, Γ 0.31), trench-DBR mirrors with angle-dependent R(χ), NH₃ line, 1 mW; "
                    "medians and min–max bands over 30 inclusion layouts. Panel (b): solid, target spec (designed R(χ), 0.1 dB/cm); dashed, minimum spec "
                    "(R ≤ 95 % below θc, 0.3 dB/cm); dash-dot, ideal (designed R(χ), no membrane loss)."},
    "A": {"label": DSM2["A"]["label"], "n_eff": 2.0, "noise": {"model": "shot", "IL": 14, "P": 1},
          "names": {"current": "set A (n_eff 2, Γ 0.2, R 99.5 %, no loss)"}, "short": {},
          "desc": "Published version (set A): n_eff 2, Γ 0.2, R 99.5 % per bounce, no membrane loss, NH₃ line, 1 mW, shot noise; medians and min–max bands over "
                  "the 30 published inclusion layouts (bit-for-bit rerun)."},
}

for _k in ("B03", "B05"):
    DSM3[_k]["desc"] += CAVEAT


def at1(arr, lg):
    i = 0
    while i < len(lg) - 2 and lg[i + 1] < 0:
        i += 1
    f = (0 - lg[i]) / (lg[i + 1] - lg[i])
    return math.exp(math.log(arr[i]) * (1 - f) + math.log(arr[i + 1]) * f)


def pack3(path, lg, rep=None):
    r = load(path)
    c = [x for x in r["chaotic"] if "error" not in x]
    g0 = r["gamma"]
    nT = len(lg)
    assert all(len(x["env"]["both"]) == nT for x in c), path
    env = {m: [q([x["env"][m][i] for x in c], .5) for i in range(nT)] for m in MODES}
    lo = {m: [min(x["env"][m][i] for x in c) for i in range(nT)] for m in MODES}
    hi = {m: [max(x["env"][m][i] for x in c) for i in range(nT)] for m in MODES}
    fr1 = [at1(x["env"]["both"], lg) for x in c]
    atT = lambda arr, T: math.exp(float(np.interp(math.log(T), lg, [math.log(v) for v in arr])))
    if rep is not None:
        Ls, Ws, Lr, Wr = rep["chaotic"]["Ls_m"], rep["chaotic"]["Ws"], rep["regular"]["Ls_m"], rep["regular"]["Ws"]
    else:   # representative layout: the one whose both-dither residual at 1 s is closest to the median
        med = q(fr1, .5)
        k = min(range(len(c)), key=lambda i: abs(math.log(fr1[i] / med)))
        Ls, Ws, Lr, Wr = c[k]["Ls_m"], c[k]["Ws"], r["regular"]["Ls_m"], r["regular"]["Ws"]
    cur = {"gamma": g0, "L_mean_m": q([x["L_mean_m"] for x in c], .5), "T_trans": q([x["T_trans"] for x in c], .5), "M2": q([x["M2"] for x in c], .5),
           "Ls_m": Ls, "Ws": Ws, "env": env, "env_lo": lo, "env_hi": hi, "floor": {m: q([x["floor"][m] for x in c], .5) for m in MODES},
           "R_bounce": q([x["R_bounce"] for x in c], .5),
           "lay": {"fr": {str(T): [atT(x["env"]["both"], T) for x in c] for T in (1, 15, 30, 60)}, "none": {str(T): [atT(x["env"]["none"], T) for x in c] for T in (1, 15, 30, 60)},
                   "fl": [x["floor"]["both"] for x in c], "fln": [x["floor"]["none"] for x in c], "T": [x["T_trans"] for x in c], "L": [x["L_mean_m"] for x in c]}}
    g = r["regular"]
    reg = {"gamma": g0, "L_mean_m": g["L_mean_m"], "T_trans": g["T_trans"], "M2": g["M2"], "Ls_m": Lr, "Ws": Wr, "env": g["env"], "floor": g["floor"]}
    return cur, reg, r


def data3(ds, lg):
    out = {}
    for slot, p in fig3_files(ds).items():
        rep = None
        if ds == "S10":
            rep = load(S10 / ("rep_circle_" + {"current": "target", "lossy": "minimum", "target": "ideal"}[slot] + ".json"))
        cur, reg, r = pack3(p, lg, rep)
        out[slot] = {"cur": cur, "reg": reg}
        if slot == "current":
            st = r["settings"]
            out["circle"] = {k: st.get(k) for k in ("c_radius", "n_facets", "c_theta", "dither", "waveform", "f_theta", "dlam", "lwave", "f_lambda",
                                                    "f_parasitic", "stray", "n_inc", "inc_radius", "inc_dn", "inc_sign")}
            out["circle"].update({"inc_seed": 1, "n_eff": st["n_eff"], "gamma": r["gamma"], "wavelength": st["wavelength"],
                                  "R_mirror": st["R_mirror"] if st.get("R_table") is None else "R(|sin chi|) table",
                                  "membrane_loss_dB_cm": round(st.get("alpha_m", 0) * 4.3429448190325175 * 0.01, 3), "n_seeds": len(r["chaotic"])})
    return out


# ------------------------------------------------------------------ Fig. 2 page
def rep_(s, a, b, cnt=1):
    assert a in s, a[:100]
    return s.replace(a, b, cnt)


def build2(s):
    ensall = {ds: ens2(ds) for ds in ORDER}
    i0 = s.index('<script id="ens" type="application/json">')
    i1 = s.index('</script>', i0)
    s = s[:i0] + '<script id="ens" type="application/json">' + json.dumps(ensall) + s[i1:]
    s = s.replace('<script id="rtab" type="application/json">', '<script id="dsets" type="application/json">' + json.dumps(DSM2, ensure_ascii=False)
                  + '</script>\n<script id="rtab" type="application/json">', 1)
    s = rep_(s, '<title>Fig. 2 raytracing and robustness</title>', '<title>Fig. 2 raytracing and robustness</title>')
    s = rep_(s, '<h1>Multipass Chaos Figure: spec ensembles and robustness (QUIET steps 10–11)</h1>',
             '<h1>Multipass Chaos Figure: v32 realistic parameters, earlier data for comparison</h1>')
    s = re.sub(r'<p>Composite paper figure.*?</p>',
               '<p>Composite paper figure from the Multipass Chaos Lab simulations: ray traces of the regular and chaotic circular (CMPC) and Herriott cells, '
               'their Poincaré sections, and the static residual fringe for each dither. The lab\'s engine runs here in the background. All figure text is Arial '
               '(Liberation Sans in the PDF); set the size below the figure. Choose the data set in the rail; the earlier versions stay selectable for comparison. '
               'Bars (f): medians and quartiles over 30 inclusion layouts. <span id="dsetDesc"></span></p>', s, count=1, flags=re.S)
    opts = "".join(f'<option value="{k}"{" selected" if k == "B03" else ""}>{DSM2[k]["label"]}</option>' for k in ORDER)
    s = rep_(s, '<div class="btns"><button type="button" class="primary" id="run">Run the simulation</button>',
             f'<div class="ctl"><label for="dset">Data set</label><select id="dset">{opts}</select></div>\n'
             '        <div class="btns"><button type="button" class="primary" id="run">Run the simulation</button>')
    s = rep_(s, '<div class="ctl"><label>Cases in (f): ideal · target · minimum</label>', '<div class="ctl"><label id="caseLbl">Cases in (f)</label>')
    s = rep_(s, 'Order: ideal, target, minimum. Whiskers', 'Order as the buttons. Whiskers')
    s = rep_(s, '<p class="hint">Chaotic median with quartiles over 30 inclusion layouts; the regular cell has no inclusions. Below target = share of the 30 layouts under 5e-7. 600 nm SiNx, TM₀, n_eff 1.66; Γ⟨L⟩ with Γ = 0.31.</p>',
             '<p class="hint">Chaotic median with quartiles over 30 inclusion layouts; the regular cell has no inclusions. Below target = share of the 30 layouts under 5e-7. <span id="ensDesc"></span></p>')
    s = rep_(s, '<p class="hint">Circle CMPC, target spec (designed R(χ), 0.1 dB/cm), SiNx films 500–700 nm',
             '<p class="hint">Step-11 basis for every data set (not yet rerun at set B). Circle CMPC, target spec (designed R(χ), 0.1 dB/cm), SiNx films 500–700 nm')
    # data-set state
    s = rep_(s, 'const ENS = JSON.parse($("ens").textContent);',
             'const ENSALL = JSON.parse($("ens").textContent), DSM = JSON.parse($("dsets").textContent), dsK = () => $("dset").value;\n'
             '  let ENS = ENSALL[dsK()];')
    s = rep_(s, 'const CASES = ["ideal", "target", "lossy"].filter(c => ENS[c]);', 'let CASES = ["ideal", "target", "lossy"].filter(c => ENS[c]);')
    s = rep_(s, 'if (!ENS[b.dataset.case]) b.hidden = true; b.setAttribute',
             'b.hidden = !ENS[b.dataset.case]; b.textContent = DSM[dsK()].slots[b.dataset.case] || b.dataset.case; b.title = DSM[dsK()].slots[b.dataset.case] || ""; b.setAttribute')
    s = rep_(s, 'syncCases();\n  const ensSets',
             'function applyDset() {\n'
             '    const M = DSM[dsK()]; ENS = ENSALL[dsK()]; CASES = ["ideal", "target", "lossy"].filter(c => ENS[c]);\n'
             '    for (const c of ["ideal", "target", "lossy"]) caseOn[c] = (M.defaultOn || ["lossy"]).includes(c);\n'
             '    $("cMinOnly").textContent = (M.slots.lossy || "") + " only"; $("cMinOnly").hidden = CASES.length < 2;\n'
             '    $("caseLbl").textContent = "Cases in (f): " + CASES.map(c => M.slots[c]).join(" · ");\n'
             '    const lc = $("liveCase"); for (const o of lc.options) { o.hidden = !M.slots[o.value]; o.disabled = !M.slots[o.value];\n'
             '      o.textContent = M.slots[o.value] ? (M.live.rtab ? ({ ideal: "ideal (designed R(χ), no loss)", target: "target (designed R(χ), 0.1 dB/cm)", lossy: "minimum (R ≤ 95 % below θc, 0.3 dB/cm)" })[o.value]\n'
             '        : `${M.slots[o.value]}, ${M.live.dB[o.value]} dB/cm`) : o.value; }\n'
             '    lc.value = M.liveCase; $("sp").value = M.live.sp;\n'
             '    $("dsetDesc").textContent = M.desc; $("ensDesc").textContent = M.desc;\n'
             '    syncCases();\n'
             '  }\n'
             '  $("dset").addEventListener("change", () => { applyDset(); $("ensPanel").hidden = true; if (typeof syncOut === "function") syncOut(); runStatus(); draw(); });\n'
             '  applyDset();\n'
             '  const ensSets')
    s = rep_(s, 'const nm = {"ideal": "ideal (designed R(χ), no loss)", "target": "target (designed R(χ), 0.1 dB/cm)", "lossy": "minimum (R ≤ 95 % below θc, 0.3 dB/cm)"}, f = v',
             'const nm = DSM[dsK()].slots, f = v')
    # live run parameters of the selected data set
    s = rep_(s, 'const lc = $("liveCase").value, RTB = JSON.parse($("rtab").textContent), tabOf = c => RTB[c][lc === "lossy" ? "min95" : "design"];\n'
                '    const DBC = {"ideal": 0.0, "target": 0.1, "lossy": 0.3};\n'
                '    const common = Object.assign({ wavelength: LAMS[$("sp").value], species: $("sp").value, n_eff: 1.66, R_mirror: 0.99, alpha_m: DBC[lc] / 4.3429448190325175 / 0.01, gamma: 0.313, P_in: 1e-3,',
             'const LM = DSM[$("dset").value].live, lc = LM.dB[$("liveCase").value] === undefined ? Object.keys(LM.dB)[0] : $("liveCase").value;\n'
             '    const RTB = JSON.parse($("rtab").textContent), tabOf = c => RTB[c][lc === "lossy" ? "min95" : "design"], useT = !!LM.rtab;\n'
             '    const rOpt = c => useT ? { R_table: { s: tabOf(c).s, R: tabOf(c).R }, R_mirror: Math.min(tabOf(c).R_avg, 0.99999) } : { R_table: null, R_mirror: LM.R[lc] };\n'
             '    const common = Object.assign({ wavelength: LAMS[$("sp").value], species: $("sp").value, n_eff: LM.n_eff, R_mirror: 0.99, alpha_m: LM.dB[lc] / 4.3429448190325175 / 0.01, gamma: LM.gamma, P_in: 1e-3,')
    s = rep_(s, 'inc_ring: 1, R_table: { s: tabOf("circle").s, R: tabOf("circle").R }, R_mirror: Math.min(tabOf("circle").R_avg, 0.99999) }, inc("C"));',
             'inc_ring: 1 }, rOpt("circle"), inc("C"));')
    s = rep_(s, 'ripple_period: 0.5e-3, R_table: { s: tabOf("herriott").s, R: tabOf("herriott").R }, R_mirror: Math.min(tabOf("herriott").R_avg, 0.99999) }, inc("H"));',
             'ripple_period: 0.5e-3 }, rOpt("herriott"), inc("H"));')
    s = rep_(s, '$("valsNote").textContent = `Live run: ${({ ideal: "ideal", target: "target", lossy: "minimum" })[$("liveCase").value]} spec, 600 nm SiNx TM₀ (n_eff 1.66), R(χ) table, ${(p.alpha_m * 4.3429448190325175 * 0.01).toFixed(1)} dB/cm.',
             '$("valsNote").textContent = `Live run: ${DSM[$("dset").value].label}, case ${DSM[$("dset").value].slots[$("liveCase").value] || ""} (n_eff ${p.n_eff}, Γ ${p.gamma}, ${p.R_table ? "R(χ) table" : "R " + (100 * p.R_mirror).toFixed(1) + " %"}, ${(p.alpha_m * 4.3429448190325175 * 0.01).toFixed(1)} dB/cm).')
    return s


# ------------------------------------------------------------------ Fig. 3 page
def build3(s):
    m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', s, re.S)
    d = json.loads(m.group(2))
    lg = [math.log(t) for t in d["T"]]
    dsets = {}
    for ds in ORDER:
        x = data3(ds, lg)
        dsets[ds] = dict(DSM3[ds], data=x)
    chaos0 = d["chaos"]
    for k in ("current", "regular", "lossy_current", "lossy_regular", "target_current", "target_regular", "stats"):
        d.pop(k, None)
    d["chaos0"] = chaos0
    s = s[:m.start(2)] + json.dumps(d) + s[m.end(2):]
    s = s.replace('<script id="data" type="application/json">', '<script id="dsets" type="application/json">' + json.dumps(dsets, ensure_ascii=False)
                  + '</script>\n<script id="data" type="application/json">', 1)
    s = rep_(s, '<h1>Fig. 3 spectra and noise: spec cases and robustness (QUIET steps 10–11)</h1>',
             '<h1>Fig. 3 spectra and noise: v32 realistic parameters, earlier data for comparison</h1>')
    s = re.sub(r'(<header>\s*<h1>.*?</h1>\s*<p>)(.*?)(</p>)',
               lambda mm: mm.group(1) + 'All figure text Arial (10 pt by default, adjustable below the figure); set the figure size below the figure. Try noise overlays on the '
               'HITRAN spectra of panel (a), and choose the data set, the noise model and the curves of panel (b). Spectra: line-by-line Voigt from the repo\'s HITRAN '
               'files. <span id="dsDesc3"></span> <span id="nzDesc3"></span>' + mm.group(3), s, count=1, flags=re.S)
    opts = "".join(f'<option value="{k}"{" selected" if k == "B03" else ""}>{DSM3[k]["label"]}</option>' for k in ORDER)
    s = rep_(s, '<legend>Panel (b): curves</legend>',
             '<legend>Panel (b): data and noise</legend>\n'
             f'        <div class="ctl"><label for="dset3">Data set</label><select id="dset3">{opts}</select></div>\n'
             '        <div class="ctl"><label for="nzModel">Noise model</label><select id="nzModel"><option value="det">detector, proposal T1.3 (NEP 14 pW/√Hz, insertion loss)</option>'
             '<option value="shot">shot noise of the transmitted light (earlier figures)</option></select></div>\n'
             '        <div class="checks"><label class="sel mono" for="nzIL">insertion loss (dB) <select id="nzIL"><option>14</option><option>20</option><option>25</option></select></label>'
             '<label class="sel mono" for="nzP">P_in (mW) <select id="nzP"><option>1</option><option>3</option><option>10</option></select></label></div>\n'
             '        <p class="hint">Detector model: δ = √(NEP² + 2hνP_d) / (P_d √(2T)), P_d = P_in·10^(−IL/10), independent of the simulated cell transmission. '
             'Shot model: √(q / (P_in T_cell T)). Selecting a data set loads its own noise model (set B: detector, 14 dB, 1 mW; earlier sets: shot, 1 mW).</p>\n'
             '        <legend style="padding:0;margin-top:4px">Curves</legend>')
    s = rep_(s, '<button type="button" id="anaBtn">Analyse all three cases (1 s)</button>', '<button type="button" id="anaBtn">Analyse the cases at 1 s</button>')
    s = rep_(s, '<p class="hint">NH₃ 1531.7 nm line, 1 mW, shot noise and the fringe residual after 1 s',
             '<p class="hint">Step-11 basis for every data set (not yet rerun at set B; shot noise as computed then). NH₃ 1531.7 nm line, 1 mW, shot noise and the fringe residual after 1 s')
    # state: data-set slots become reassignable
    s = rep_(s, 'const D = JSON.parse(document.getElementById("data").textContent);',
             'const D = JSON.parse(document.getElementById("data").textContent), DS3 = JSON.parse(document.getElementById("dsets").textContent);\n'
             '  const H_PL = 6.62607015e-34, NEP = 14e-12, LAM_N = 1.5317e-6;')
    s = rep_(s, '  const cur = D.current, reg = D.regular, GL = cur.gamma * cur.L_mean_m * 100;\n'
                '  const lcur = D.lossy_current, lreg = D.lossy_regular, lGL = lcur.gamma * lcur.L_mean_m * 100;\n'
                '  const lshotAt = T => Math.sqrt(Q_E / (D.P_in * lcur.T_trans * T));\n'
                '  const tcur = D.target_current, treg = D.target_regular, tGL = tcur ? tcur.gamma * tcur.L_mean_m * 100 : 0;\n'
                '  const tshotAt = T => Math.sqrt(Q_E / (D.P_in * tcur.T_trans * T));\n',
             '  let cur, reg, GL, lcur, lreg, lGL, tcur, treg, tGL, DSX;\n'
             '  const NZ = () => ({ m: document.getElementById("nzModel").value, IL: +document.getElementById("nzIL").value, P: +document.getElementById("nzP").value * 1e-3 });\n'
             '  function noiseAt(o, T) {   /* white noise floor (fractional power) after averaging T */\n'
             '    const z = NZ(); if (z.m === "shot") return Math.sqrt(Q_E / (z.P * o.T_trans * T));\n'
             '    const Pd = z.P * 10 ** (-z.IL / 10); return Math.sqrt(NEP ** 2 + 2 * H_PL * C0 / LAM_N * Pd) / (Pd * Math.sqrt(2 * T));\n'
             '  }\n'
             '  const noiseName = () => NZ().m === "shot" ? "shot" : "detector";\n'
             '  const noiseDesc = () => { const z = NZ(); return z.m === "shot" ? `shot noise √(q / (P_in T_cell T)), P_in ${z.P * 1e3} mW` : `detector noise (proposal T1.3): NEP 14 pW/√Hz, insertion loss ${z.IL} dB, P_in ${z.P * 1e3} mW`; };\n'
             '  function setDS(k, keepNoise) {\n'
             '    DSX = DS3[k]; const x = DSX.data;\n'
             '    cur = x.current.cur; reg = x.current.reg; GL = cur.gamma * cur.L_mean_m * 100;\n'
             '    lcur = x.lossy ? x.lossy.cur : null; lreg = x.lossy ? x.lossy.reg : null; lGL = lcur ? lcur.gamma * lcur.L_mean_m * 100 : 0;\n'
             '    tcur = x.target ? x.target.cur : null; treg = x.target ? x.target.reg : null; tGL = tcur ? tcur.gamma * tcur.L_mean_m * 100 : 0;\n'
             '    D.n_eff = DSX.n_eff;\n'
             '    D.chaos = Object.assign({}, D.chaos0, { source: DSX.desc, circle: Object.assign({}, D.chaos0.circle, x.circle), lossy: lcur ? { name: DSX.names.lossy } : null, target: tcur ? { name: DSX.names.target } : null });\n'
             '    if (!keepNoise) { $("nzModel").value = DSX.noise.model; $("nzIL").value = String(DSX.noise.IL); $("nzP").value = String(DSX.noise.P); }\n'
             '  }\n'
             '  const lshotAt = T => noiseAt(lcur, T), tshotAt = T => noiseAt(tcur, T);\n')
    s = rep_(s, '  const shotAt = T => Math.sqrt(Q_E / (D.P_in * cur.T_trans * T));\n', '  const shotAt = T => noiseAt(cur, T);\n')
    # minimum / ideal slots may be absent (set A)
    s = s.replace('on("lossy")) extra(', 'on("lossy") && lcur) extra(').replace('if (on("lossy")) s += bnd(lcur', 'if (on("lossy") && lcur) s += bnd(lcur')
    s = rep_(s, 'if (on("lossy")) s += lbl(lcur, lshotAt, lGL, "min. spec", -5);\n      if (on("target") && tcur) s += lbl(tcur, tshotAt, tGL, "ideal", 13);',
             'if (on("lossy") && lcur) s += lbl(lcur, lshotAt, lGL, DSX.short.lossy, -5);\n      if (on("target") && tcur) s += lbl(tcur, tshotAt, tGL, DSX.short.target, 13);')
    s = rep_(s, '["shot", 3.1, 0.2, "shot", "#595959"]', '["shot", 3.1, 0.2, noiseName(), "#595959"]')
    # analysis: recomputed from per-layout values for the selected noise model
    s = rep_(s, 'const S = D.stats, names = { ideal: "ideal (designed R(χ), no loss)", target: "target (designed R(χ), 0.1 dB/cm)", lossy: "minimum (R ≤ 95 % below θc, 0.3 dB/cm)" }, ks = ["ideal", "target", "lossy"].filter(k => S[k]);',
             'const S = statsNow(), names = DSX.names, ks = ["current", "lossy", "target"].filter(k => S[k]);')
    s = rep_(s, '["fringe noise, both (NEA)", k => f(S[k].fringe_med)], ["shot noise (NEA)", k => f(S[k].shot_med)], ["fringe / shot", k => f(S[k].fringe_med / S[k].shot_med)]',
             '["fringe noise, both (NEA)", k => f(S[k].fringe_med)], [`${noiseName()} noise (NEA)`, k => f(S[k].shot_med)], [`fringe / ${noiseName()}`, k => f(S[k].fringe_med / S[k].shot_med)]')
    s = re.sub(r'\$\("anaNote"\)\.textContent = "Medians over 30 inclusion layouts;.*?";',
               '$("anaNote").textContent = `Medians over ${cur.lay.T.length} inclusion layouts; the regular cell is the same cell and launch without inclusions. ${DSX.label}. '
               'ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_NH₃) at T = 1 s: fringe residual (drift 0.1 rad/s) and ${noiseDesc()}. α_NH₃ from the HITRAN strip of panel (a).`;',
               s, count=1, flags=re.S)
    s = rep_(s, '  const cache = new Map();',
             '  const qq = (a, p) => { const b = [...a].sort((u, v) => u - v), i = (b.length - 1) * p, lo = Math.floor(i), hi = Math.min(lo + 1, b.length - 1); return b[lo] + (b[hi] - b[lo]) * (i - lo); };\n'
             '  function statsNow() {\n'
             '    const out = {};\n'
             '    for (const [k, o, rg] of [["current", cur, reg], ["lossy", lcur, lreg], ["target", tcur, treg]]) {\n'
             '      if (!o) continue; const Ly = o.lay, g = o.gamma, sh = Ly.T.map(t => noiseAt({ T_trans: t }, 1));\n'
             '      const nea = Ly.fr1.map((v, i) => Math.hypot(v, sh[i])), ppb = nea.map((n, i) => n / (g * Ly.L[i] * 100 * aNH3) * 1e3);\n'
             '      const rppb = Math.hypot(interpLog(rg.env.both, 1), noiseAt(rg, 1)) / (g * rg.L_mean_m * 100 * aNH3) * 1e3, pm = qq(ppb, .5);\n'
             '      out[k] = { GL_cm: g * o.L_mean_m * 100, T: o.T_trans, M2: o.M2, ppb_med: pm, ppb_min: Math.min(...ppb), ppb_max: Math.max(...ppb), ppb_q1: qq(ppb, .25), ppb_q3: qq(ppb, .75),\n'
             '        fringe_med: qq(Ly.fr1, .5), shot_med: qq(sh, .5), reg_ppb: rppb, adv: rppb / pm, dither_gain: qq(Ly.none1, .5) / qq(nea, .5), R_bounce: o.R_bounce };\n'
             '    }\n'
             '    return out;\n'
             '  }\n'
             '  const cache = new Map();')
    s = rep_(s, 'shot_noise: { input_power_W: D.P_in, formula: "sqrt(q / (P_in T_cell T))" },',
             'data_set: { key: $("dset3").value, label: DSX.label }, noise_model: { model: NZ().m, insertion_loss_dB: NZ().m === "det" ? NZ().IL : null, input_power_W: NZ().P, NEP_W_rtHz: NZ().m === "det" ? NEP : null, '
             'formula: NZ().m === "shot" ? "sqrt(q / (P_in T_cell T))" : "sqrt(NEP^2 + 2 h nu P_d) / (P_d sqrt(2 T)), P_d = P_in 10^(-IL/10)", text: noiseDesc() },')
    s = rep_(s, 'note: "solid curves and the right ppb axis: target spec (Gamma<L> of that cell); the dashed (minimum) and dash-dot (ideal) sets are labelled with their own Gamma<L>"',
             'note: "solid curves and the right ppb axis: the solid case (Gamma<L> of that cell); the dashed and dash-dot cases are labelled with their own Gamma<L>", case_names: DSX.names')
    # wire the selectors: initialise before the first draw, relabel the curve checkboxes
    s = rep_(s, '  const bc = $("bCurves");',
             '  setDS($("dset3").value);\n'
             '  function relabel() {\n'
             '    const L = { lossy: (DSX.names.lossy || "dashed case") , target: (DSX.names.target || "dash-dot case"), shot: noiseName() + " noise", shotx: noiseName() + " noise of the dashed / dash-dot cases (gray dotted)" };\n'
             '    for (const [k, t] of Object.entries(L)) { const el = $("b-" + k); if (!el) continue; el.parentNode.lastChild.textContent = " " + t; el.parentNode.hidden = (k === "lossy" && !lcur) || (k === "target" && !tcur); }\n'
             '    $("dsDesc3").textContent = DSX.desc; $("nzDesc3").textContent = "Noise: " + noiseDesc() + ".";\n'
             '    $("nzIL").disabled = NZ().m === "shot"; if (!$("anaPanel").hidden) $("anaBtn").onclick();\n'
             '  }\n'
             '  const bc = $("bCurves");')
    s = rep_(s, '  const setSize = (w, h) =>',
             '  relabel();\n'
             '  $("dset3").addEventListener("change", () => { setDS($("dset3").value); cache.clear(); relabel(); draw(); });\n'
             '  for (const id of ["nzModel", "nzIL", "nzP"]) $(id).addEventListener("change", () => { relabel(); draw(); });\n'
             '  const setSize = (w, h) =>')
    s = build3_more(s)
    s = build3_species(s)
    return s


def build3_more(s):
    """15-30 s window, fringe-drift toggle, analysis time, and the 3.3 um cell for the 3.3 um strips of panel (a)."""
    s = rep_(s, '        <legend style="padding:0;margin-top:4px">Curves</legend>',
             '        <div class="checks"><label class="check"><input type="checkbox" id="drift" checked> fringe drift 0.1 rad/s (static fringe averages down)</label>'
             '<label class="check"><input type="checkbox" id="win"> mark 15–30 s window</label></div>\n'
             '        <div class="ctl"><label for="anaT">Analysis time (s)</label><select id="anaT"><option>1</option><option>15</option><option>30</option><option>60</option></select></div>\n'
             '        <legend style="padding:0;margin-top:4px">Curves</legend>')
    s = rep_(s, '<button type="button" id="anaBtn">Analyse the cases at 1 s</button>', '<button type="button" id="anaBtn">Analyse the cases</button>')
    s = rep_(s, 'function noiseAt(o, T) {   /* white noise floor (fractional power) after averaging T */\n'
                '    const z = NZ(); if (z.m === "shot") return Math.sqrt(Q_E / (z.P * o.T_trans * T));\n'
                '    const Pd = z.P * 10 ** (-z.IL / 10); return Math.sqrt(NEP ** 2 + 2 * H_PL * C0 / LAM_N * Pd) / (Pd * Math.sqrt(2 * T));',
             'function noiseAt(o, T, lam) {   /* white noise floor (fractional power) after averaging T */\n'
             '    const z = NZ(); if (z.m === "shot") return Math.sqrt(Q_E / (z.P * o.T_trans * T));\n'
             '    const Pd = z.P * 10 ** (-z.IL / 10); return Math.sqrt(NEP ** 2 + 2 * H_PL * C0 / (lam || LAM_N) * Pd) / (Pd * Math.sqrt(2 * T));')
    s = rep_(s, '    D.n_eff = DSX.n_eff;\n',
             '    D.n_eff = DSX.n_eff; mcur = x.mir ? x.mir.cur : null;\n'
             '    if (!keepNoise) { $("avgT").value = String(Math.log10(DSX.T || 1)); $("win").checked = !!DSX.win; $("anaT").value = String(DSX.T || 1); }\n'
             '    applyDrift();\n')
    s = rep_(s, '  let cur, reg, GL, lcur, lreg, lGL, tcur, treg, tGL, DSX;\n',
             '  let cur, reg, GL, lcur, lreg, lGL, tcur, treg, tGL, DSX, mcur = null;\n'
             '  /* without drift the residual cannot fall below the static fringe (held at the floor) */\n'
             '  function applyDrift() {\n'
             '    const on = document.getElementById("drift").checked;\n'
             '    for (const o of [cur, reg, lcur, lreg, tcur, treg, mcur]) { if (!o) continue; if (!o.env0) { o.env0 = o.env; o.lo0 = o.env_lo; o.hi0 = o.env_hi; }\n'
             '      if (on) { o.env = o.env0; o.env_lo = o.lo0; o.env_hi = o.hi0; continue; }\n'
             '      const f = m => o.env0[m].map(v => Math.max(1, o.floor[m] / v));\n'
             '      o.env = Object.fromEntries(MODES.map(m => [m, o.env0[m].map(v => Math.max(v, o.floor[m]))]));\n'
             '      if (o.lo0) { o.env_lo = Object.fromEntries(MODES.map(m => { const g = f(m); return [m, o.lo0[m].map((v, i) => v * g[i])]; }));\n'
             '        o.env_hi = Object.fromEntries(MODES.map(m => { const g = f(m); return [m, o.hi0[m].map((v, i) => v * g[i])]; })); }\n'
             '    }\n'
             '  }\n')
    s = rep_(s, '  for (const id of ["nzModel", "nzIL", "nzP"]) $(id).addEventListener("change", () => { relabel(); draw(); });',
             '  for (const id of ["nzModel", "nzIL", "nzP", "anaT"]) $(id).addEventListener("change", () => { relabel(); draw(); });\n'
             '  $("drift").addEventListener("change", () => { applyDrift(); relabel(); draw(); });\n'
             '  $("win").addEventListener("change", () => draw());')
    s = rep_(s, "if (!o) continue; const Ly = o.lay, g = o.gamma, sh = Ly.T.map(t => noiseAt({ T_trans: t }, 1));\n"
                "      const nea = Ly.fr1.map((v, i) => Math.hypot(v, sh[i])), ppb = nea.map((n, i) => n / (g * Ly.L[i] * 100 * aNH3) * 1e3);\n"
                "      const rppb = Math.hypot(interpLog(rg.env.both, 1), noiseAt(rg, 1)) / (g * rg.L_mean_m * 100 * aNH3) * 1e3, pm = qq(ppb, .5);",
             "if (!o) continue; const Ly = o.lay, g = o.gamma, TA = +$(\"anaT\").value, dr = $(\"drift\").checked, sh = Ly.T.map(t => noiseAt({ T_trans: t }, TA));\n"
             "      const fr = Ly.fr[TA].map((v, i) => dr ? v : Math.max(v, Ly.fl[i])), nn = Ly.none[TA].map((v, i) => dr ? v : Math.max(v, Ly.fln[i]));\n"
             "      const nea = fr.map((v, i) => Math.hypot(v, sh[i])), ppb = nea.map((n, i) => n / (g * Ly.L[i] * 100 * aNH3) * 1e3);\n"
             "      const rppb = Math.hypot(interpLog(rg.env.both, TA), noiseAt(rg, TA)) / (g * rg.L_mean_m * 100 * aNH3) * 1e3, pm = qq(ppb, .5);")
    s = rep_(s, "fringe_med: qq(Ly.fr1, .5), shot_med: qq(sh, .5), reg_ppb: rppb, adv: rppb / pm, dither_gain: qq(Ly.none1, .5) / qq(nea, .5)",
             "fringe_med: qq(fr, .5), shot_med: qq(sh, .5), reg_ppb: rppb, adv: rppb / pm, dither_gain: qq(nn, .5) / qq(nea, .5)")
    s = rep_(s, '["NH₃ limit at 1 s, median (ppb)"', '[`NH₃ limit at ${$("anaT").value} s, median (ppb)`')
    s = rep_(s, 'at T = 1 s: fringe residual (drift 0.1 rad/s) and ${noiseDesc()}.',
             'at T = ${$("anaT").value} s: fringe residual (${$("drift").checked ? "drift 0.1 rad/s" : "no drift, static floor"}) and ${noiseDesc()}.')
    s = rep_(s, '  function noiseFor(si, mode, seed) {\n    const key = `${si}|${mode}|${seed}`;',
             '  function noiseFor(si, mode, seed, cc) {\n    const key = `${si}|${mode}|${seed}|${cc.key}`;')
    s = rep_(s, 'for (let k = 0; k < cur.Ls_m.length; k++) {\n      const a = Math.sqrt(cur.Ws[k]), ph = 2 * Math.PI * rnd(), q = 2 * Math.PI * 1e9 * D.n_eff * cur.Ls_m[k] / C0;',
             'for (let k = 0; k < cc.o.Ls_m.length; k++) {\n      const a = Math.sqrt(cc.o.Ws[k]), ph = 2 * Math.PI * rnd(), q = 2 * Math.PI * 1e9 * cc.n * cc.o.Ls_m[k] / C0;')
    s = rep_(s, '      const si = D.strips.indexOf(st), top = yt + pos * (hs + gap), bot = top + hs, x = st.x_ghz, N = x.length;',
             '      const si = D.strips.indexOf(st), top = yt + pos * (hs + gap), bot = top + hs, x = st.x_ghz, N = x.length;\n'
             '      const mid = st.lam_um > 3 && mcur, CC = mid ? { o: mcur, n: DSX.n_eff_mir, key: "mir" } : { o: cur, n: D.n_eff, key: "nir" };\n'
             '      const GL = CC.o.gamma * CC.o.L_mean_m * 100, sh = noiseAt(CC.o, T, mid ? st.lam_um * 1e-6 : LAM_N), lev = {}; for (const m of MODES) lev[m] = interpLog(CC.o.env[m], T);')
    s = rep_(s, 'const nz = noiseFor(si, m, seed), L = lev[m];', 'const nz = noiseFor(si, m, seed, CC), L = lev[m];')
    s = rep_(s, '    s += `<g clip-path="url(#clipb)">`;\n    if (on("bands")) {',
             '    const WIN = $("win").checked ? [15, 30] : null;\n'
             '    const rnd2 = v => v >= 100 ? Math.round(v) : +v.toPrecision(2), ppbOf = (o, shf, gl, t) => Math.hypot(interpLog(o.env.both, t), shf(t)) / (gl * aNH3) * 1e3;\n'
             '    const ppbTxt = (o, shf, gl) => WIN ? `${rnd2(ppbOf(o, shf, gl, WIN[1]))}–${rnd2(ppbOf(o, shf, gl, WIN[0]))} ppb` : `${rnd2(ppbOf(o, shf, gl, T))} ppb`;\n'
             '    s += `<g clip-path="url(#clipb)">`;\n'
             '    if (WIN) s += `<rect x="${BX(WIN[0])}" y="${bt}" width="${BX(WIN[1]) - BX(WIN[0])}" height="${bb - bt}" fill="#000" fill-opacity="0.07"/>`;\n'
             '    if (on("bands")) {')
    s = rep_(s, 's += `<circle cx="${BX(T)}" cy="${BY(Math.hypot(interpLog(cur.env.both, T), sh))}" r="2.6" fill="#000"/>`;',
             's += `<circle cx="${BX(T)}" cy="${BY(Math.hypot(interpLog(cur.env.both, T), sh))}" r="2.6" fill="#000"/>`;\n'
             '      if (WIN) for (const t of WIN) if (Math.abs(Math.log(t / T)) > 0.05) s += `<circle cx="${BX(t)}" cy="${BY(Math.hypot(interpLog(cur.env.both, t), shotAt(t)))}" r="2" fill="#000"/>`;')
    s = rep_(s, 'lab = `${ppbv >= 100 ? Math.round(ppbv) : +ppbv.toPrecision(2)} ppb`;', 'lab = ppbTxt(cur, shotAt, GL) + (WIN ? " (15–30 s)" : "");')
    s = rep_(s, 'l = `${p >= 100 ? Math.round(p) : +p.toPrecision(2)} ppb (${nm})`;', 'l = `${ppbTxt(o, shf, gl)} (${nm})`;')
    s = rep_(s, '["shot", 3.1, 0.2, noiseName(), "#595959"]', '["shot", $("win").checked ? 0.02 : 3.1, $("win").checked ? 0.15 : 0.2, noiseName(), "#595959"]')
    s = rep_(s, 'const ax = Math.max(BX(T) - 36.8, bl + 30)', 'const ax = Math.max(BX(T) - (WIN ? 52 : 36.8), bl + 30)')
    s = rep_(s, 'data_set: { key: $("dset3").value, label: DSX.label },',
             'data_set: { key: $("dset3").value, label: DSX.label }, fringe_drift: $("drift").checked ? "0.1 rad/s (static fringe averages down)" : "none (held at the static floor)", window_s: $("win").checked ? [15, 30] : null,')
    return s


def build3_species(s):
    """Panel (b) for NH3 1.53 um, CH4 3.31 um and C2H6 3.34 um: set B uses its simulated 3.3 um cell, the earlier data sets
    put the 1.53 um cell's noise-equivalent absorbance on the 3.3 um line (they have no 3.3 um simulation)."""
    s = rep_(s, '#robSheet { min-width: 560px; }',
             '#robSheet { min-width: 560px; }\n#spBtns button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }')
    s = rep_(s, '        <div class="ctl"><label for="nzModel">Noise model</label>',
             '        <div class="ctl"><label>Gas line of (b)</label><div class="btns" id="spBtns">'
             '<button type="button" data-sp="NH3" aria-pressed="true">NH₃ 1.53 µm</button><button type="button" data-sp="CH4" aria-pressed="false">CH₄ 3.31 µm</button>'
             '<button type="button" data-sp="C2H6" aria-pressed="false">C₂H₆ 3.34 µm</button></div><p class="hint" id="spNote"></p></div>\n'
             '        <div class="ctl"><label for="nzModel">Noise model</label>')
    s = rep_(s, 'const H_PL = 6.62607015e-34, NEP = 14e-12, LAM_N = 1.5317e-6;', 'const H_PL = 6.62607015e-34, NEP = 14e-12; let LAM_N = 1.5317e-6;')
    s = rep_(s, '  const nh3 = D.strips[0], aNH3 = Math.max(...nh3.analyte_per_ppm);', '  const nh3 = D.strips[0]; let aNH3 = Math.max(...nh3.analyte_per_ppm);')
    s = rep_(s, '  let cur, reg, GL, lcur, lreg, lGL, tcur, treg, tGL, DSX, mcur = null;\n',
             '  let cur, reg, GL, lcur, lreg, lGL, tcur, treg, tGL, DSX, mcur = null, cur0 = null, SPK = "NH3", SLOT0 = null;\n'
             '  const SPECS = { NH3: { mol: "NH3", txt: "NH₃", strip: "NH3", lam: 1.5317e-6 }, CH4: { mol: "CH4", txt: "CH₄", strip: "CH4 3.3", lam: 3.3137e-6 },\n'
             '    C2H6: { mol: "C2H6", txt: "C₂H₆", strip: "C2H6", lam: 3.3368e-6 } };\n'
             '  function applySpecies() {\n'
             '    const sp = SPECS[SPK], x = DSX.data, st = D.strips.find(q => q.key === sp.strip), [c0, r0, l0, lr0, t0, tr0] = SLOT0;\n'
             '    aNH3 = Math.max(...st.analyte_per_ppm); LAM_N = sp.lam;\n'
             '    const sim = SPK !== "NH3" && x.mir;\n'
             '    if (sim) { cur = x.mir.cur; reg = x.mir.reg; lcur = x.mir_l ? x.mir_l.cur : null; lreg = x.mir_l ? x.mir_l.reg : null; tcur = x.mir_t ? x.mir_t.cur : null; treg = x.mir_t ? x.mir_t.reg : null; }\n'
             '    else { cur = c0; reg = r0; lcur = l0; lreg = lr0; tcur = t0; treg = tr0; }\n'
             '    GL = cur.gamma * cur.L_mean_m * 100; lGL = lcur ? lcur.gamma * lcur.L_mean_m * 100 : 0; tGL = tcur ? tcur.gamma * tcur.L_mean_m * 100 : 0;\n'
             '    document.querySelectorAll("#spBtns button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.sp === SPK)));\n'
             '    const nt = document.getElementById("spNote");\n'
             '    if (nt) nt.textContent = SPK === "NH3" ? "1.53 µm cell (500 nm SiNx in set B)." : sim ? "Simulated 3.3 µm cell of set B: 1000 nm SiNx, n_eff 1.425, Γ_E 0.553, same R cases, ±0.55 GHz wavelength dither (±20 pm), ±2 mrad angle dither; detector noise at 3.3 µm."\n'
             '      : "This data set has no 3.3 µm simulation: the 1.53 µm cell\'s noise-equivalent absorbance is put on the 3.3 µm line (the scaling the proposal used).";\n'
             '    applyDrift();\n'
             '  }\n')
    s = rep_(s, '    D.n_eff = DSX.n_eff; mcur = x.mir ? x.mir.cur : null;\n',
             '    D.n_eff = DSX.n_eff; mcur = x.mir ? x.mir.cur : null; cur0 = cur; SLOT0 = [cur, reg, lcur, lreg, tcur, treg];\n')
    s = rep_(s, '    applyDrift();\n    D.chaos = ', '    applySpecies();\n    D.chaos = ')
    s = rep_(s, 'for (const o of [cur, reg, lcur, lreg, tcur, treg, mcur]) {', 'for (const o of [cur, reg, lcur, lreg, tcur, treg, mcur, cur0, ...(SLOT0 || []), ...["mir", "mir_l", "mir_t"].flatMap(k => DSX && DSX.data[k] ? [DSX.data[k].cur, DSX.data[k].reg] : [])]) {')
    s = rep_(s, 'CC = mid ? { o: mcur, n: DSX.n_eff_mir, key: "mir" } : { o: cur, n: D.n_eff, key: "nir" };', 'CC = mid ? { o: mcur, n: DSX.n_eff_mir, key: "mir" } : { o: cur0, n: D.n_eff, key: "nir" };')
    s = rep_(s, 'sh = noiseAt(CC.o, T, mid ? st.lam_um * 1e-6 : LAM_N)', 'sh = noiseAt(CC.o, T, mid ? st.lam_um * 1e-6 : 1.5317e-6)')
    s = rep_(s, 's += txt(br + 34 * k, (bt + bb) / 2, `${chem("NH3")} (ppb)`', 's += txt(br + 34 * k, (bt + bb) / 2, `${chem(SPECS[SPK].mol)} (ppb)`')
    s = rep_(s, 's += txt(bl, bt - 6, `(b) ${chem("NH3")} noise floor`);', 's += txt(bl, bt - 6, `(b) ${chem(SPECS[SPK].mol)} noise floor`);')
    s = rep_(s, '[`NH₃ limit at ${$("anaT").value} s, median (ppb)`', '[`${SPECS[SPK].txt} limit at ${$("anaT").value} s, median (ppb)`')
    s = rep_(s, 'ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_NH₃)', 'ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_${SPECS[SPK].txt})')
    s = rep_(s, '["angle", 1.0, 2.5, "angle", "#2ca02c"],', '["angle", SPK === "NH3" ? 1.0 : 0.004, SPK === "NH3" ? 2.5 : 4, "angle", "#2ca02c"],')
    s = rep_(s, '["shot", $("win").checked ? 0.02 : 3.1, $("win").checked ? 0.15 : 0.2,', '["shot", SPK !== "NH3" ? 3e-4 : $("win").checked ? 0.02 : 3.1, SPK !== "NH3" ? 0.12 : $("win").checked ? 0.15 : 0.2,')
    s = rep_(s, 'ppb_axis: { line: "NH3 6528.77 cm^-1", alpha_per_ppm_cm: aNH3 },', 'ppb_axis: { line: SPECS[SPK].strip, alpha_per_ppm_cm: aNH3, cell: SPK !== "NH3" && DSX.data.mir ? "simulated 3.3 um cell" : "1.53 um cell" },')
    s = rep_(s, '  $("win").addEventListener("change", () => draw());',
             '  $("win").addEventListener("change", () => draw());\n'
             '  document.querySelectorAll("#spBtns button").forEach(b => b.addEventListener("click", () => { SPK = b.dataset.sp; applySpecies(); cache.clear(); relabel(); draw(); }));')
    return s


GC_SIZES = ["d5", "d10", "d15", "d20"]


def gc_data():
    art = json.load(open(ROOT / "results" / "t11" / "artifact_data.json"))
    summ = json.load(open(ROOT / "results" / "t11" / "summary.json"))
    out = {"cases": {}, "sizes": GC_SIZES, "design": art["design"]}
    for band, R in (("nir", 0.97), ("nir", 0.995), ("mir", 0.97), ("mir", 0.995)):
        ck = f"{band}_R{R}"
        cs = {"bars": {}, "old": None, "scaling": {}, "table": {}}
        for cell in ["d4ref"] + GC_SIZES:
            k = f"{ck}_{cell}"
            A_ = art["cells"][k]
            cs["bars"][cell] = {m: {"reg": A_["bars"][m]["reg"], "q": A_["bars"][m]["q"]} for m in ("none", "spectral", "input", "both")}
            if cell == "d4ref":
                cs["old"] = {m: A_["bars_old"][m] for m in ("none", "spectral", "input", "both")}
            cs["scaling"][cell] = [{"name": x["name"], "N": round(x["N"], 2), "both": x["both"], "f": round(x["input_factor"], 3)} for x in art["scaling"][k]]
            sm = summ[k]
            cs["table"][cell] = {"T": sm["T"], "GL_cm": sm["GL_cm"], "M": sm["M"], "static": sm["static"]["both"],
                                 "lim": {kk: (v[1] if isinstance(v, list) else v) for kk, v in sm["limits"].items() if "fringe" not in kk}}
        cs["dlam"] = art["cells"][f"{ck}_d10"]["dlam"]
        out["cases"][ck] = cs
    return out


def build2_design(s):
    """Fig. 2 (j)-(k): the grating-coupler design at 5-20 mm (Task 11), with size buttons that also set the live cell."""
    s = rep_(s, '<input id="cR" class="sim" type="range" min="0.5" max="5" step="0.05" value="2">', '<input id="cR" class="sim" type="range" min="0.5" max="10" step="0.05" value="2">')
    panel = '''      <div class="panel" id="gcPanel">
        <h2>Fig. 2 (j)–(k): grating-coupler design, 5–20 mm cells (QUIET Task 11, wave-consistent beamlet model)</h2>
        <div class="btns" id="gcSize"><span class="sel">cell diameter</span><button type="button" data-gc="d5" aria-pressed="false">5 mm</button><button type="button" data-gc="d10" aria-pressed="true">10 mm</button><button type="button" data-gc="d15" aria-pressed="false">15 mm</button><button type="button" data-gc="d20" aria-pressed="false">20 mm</button>
          <label class="sel mono" for="gcCase">case <select id="gcCase"><option value="nir_R0.97" selected>1.53 µm, R 97 %, ±35 pm</option><option value="nir_R0.995">1.53 µm, R 99.5 %, ±35 pm</option><option value="mir_R0.97">3.3 µm, R 97 %, ±81 pm</option><option value="mir_R0.995">3.3 µm, R 99.5 %, ±81 pm</option></select></label>
          <label class="check"><input type="checkbox" id="gcLive" checked> size buttons also set the live cell above</label></div>
        <div class="sheetwrap"><div class="sheet" id="gcSheet"></div></div>
        <div class="btns">
          <label class="sel mono" for="gcW">width (cm) <input id="gcW" type="number" min="8" max="28" step="0.1" value="16.8" style="width:5em"></label>
          <label class="sel mono" for="gcH">height (cm) <input id="gcH" type="number" min="3" max="15" step="0.1" value="6.4" style="width:5em"></label>
          <label class="sel mono" for="gcFs">font (pt) <input id="gcFs" type="number" min="6" max="14" step="0.5" value="10" style="width:4.5em"></label>
          <button type="button" id="gcPdf">Download PDF</button><button type="button" id="gcSvgBtn">Download SVG</button><button type="button" id="gcCsv">Download data (CSV)</button>
          <span class="status" id="gcStatus" role="status"></span>
        </div>
        <p class="hint">Design: free-space beam on a large-area input grating coupler (waist 75 µm, angle dither ±60 mrad, position dither ±100 µm), 24-facet circle CMPC, 8 index inclusions r 25 µm |Δn| 0.01, output port on facet 11 (best or equal-best of 23 at every size, chosen on layouts 1–15, checked on 16–30), set B, 0.3 dB/cm, ε = 10⁻⁶; static fringe (no drift), medians and quartiles over 30 layouts. (j) fringe per dither mode (none, wavelength, input angle + position, both) for the selected size, against the 4 mm cell with the old ±2 mrad, 40 µm launch. (k) both-dither fringe against the number of input states N swept by 17 input-dither designs (angle, position, both), one series per cell size; dashed: N<sup>−½</sup> from the 10 mm single-state value. The input dither follows N<sup>−½</sup> (Task 9b); the larger cells start lower because the wavelength dither averages better on their longer paths. Table: limits with drift averaging (0.1 rad/s) and the detector (10 mW, simulated transmission, +8 dB for two grating couplers, NEP 14 pW/√Hz).</p>
        <div class="tbl"><table class="vals" id="gcTbl"></table></div>
      </div>
'''
    s = rep_(s, '      <div class="panel" id="robPanel">', panel + '      <div class="panel" id="robPanel">')
    s = s.replace('<script id="dsets" type="application/json">', '<script id="gcdata" type="application/json">' + json.dumps(gc_data()) + '</script>\n<script id="dsets" type="application/json">', 1)
    s = rep_(s, '#robSheet { min-width: 560px; }', '#robSheet, #gcSheet { min-width: 560px; }\n#gcSize button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }')
    js = r'''
  /* ---------------------------------------------------------------- grating-coupler design (QUIET Task 11) */
  const GC = JSON.parse(document.getElementById("gcdata").textContent);
  const GC_COL = { d5: "#9ecae1", d10: "#4292c6", d15: "#08519c", d20: "#08306b" }, GC_D = { d5: 5, d10: 10, d15: 15, d20: 20 };
  let gcSel = "d10", gcLast = null;
  const MODES4 = [["none", "none"], ["spectral", "λ"], ["input", "input"], ["both", "both"]];
  function gcPanels() {
    const cs = GC.cases[$("gcCase").value], R = $("colCR").value, C = $("colCC").value, bars = cs.bars[gcSel], x = [1, 2, 3, 4];
    const pj = { title: `(j) ${GC_D[gcSel]} mm cell`, xr: [0.5, 4.5], xticks: MODES4.map(([, l], i) => [i + 1, l]), xlab: [["Dither", 0]],
      ylog: true, yr: [1e-8, 1e-3], ylab: [["Static fringe ΔP/P", 0]], hlines: [{ y: 5e-7, col: "#d62728" }],
      bands: [{ x, lo: MODES4.map(([m]) => bars[m].q[0]), hi: MODES4.map(([m]) => bars[m].q[2]), col: C, a: 0.22 }],
      series: [{ x, y: MODES4.map(([m]) => cs.old[m][1]), col: "#7f7f7f", lw: 1.2, mk: 2.2, dash: "3 2" },
               { x, y: MODES4.map(([m]) => bars[m].reg), col: R, lw: 1.2, mk: 2.2 },
               { x, y: MODES4.map(([m]) => bars[m].q[1]), col: C, lw: 1.4, mk: 2.4 }] };
    const ser = GC.sizes.map(c => { const v = cs.scaling[c].slice().sort((a, b) => a.N - b.N);
      return { x: v.map(p => p.N), y: v.map(p => p.both[1]), col: GC_COL[c], lw: c === gcSel ? 1.0 : 0.5, mk: c === gcSel ? 2.6 : 1.7 }; });
    const one = cs.scaling[gcSel].find(p => p.name === "w40 A2") || cs.scaling[gcSel][0], Ns = [1, 600];
    ser.push({ x: Ns, y: Ns.map(n => one.both[1] * n ** -0.5), col: "#000000", lw: 0.9, dash: "4 3" });
    const pk = { title: "(k) fringe against input states", xlog: true, xr: [0.8, 600], xticks: [[1, "1"], [10, "10"], [100, "100"]], xlab: [["Input states N", 0]],
      ylog: true, yr: [1e-8, 1e-4], ylab: [["ΔP/P, both dithers", 0]], hlines: [{ y: 5e-7, col: "#d62728" }], series: ser };
    return [pj, pk];
  }
  function gcLegend() {
    return [{ label: "4 mm, ±2 mrad", col: "#7f7f7f", lw: 1.4, dash: 1 }, { label: "regular (indicative)", col: $("colCR").value, lw: 1.6 }, { label: "chaotic", col: $("colCC").value, lw: 1.6 },
      ...GC.sizes.map(c => ({ label: `${GC_D[c]} mm`, col: GC_COL[c], lw: 2.2 }))];
  }
  function gcTable() {
    const cs = GC.cases[$("gcCase").value], nir = $("gcCase").value.startsWith("nir"), gases = nir ? ["NH3"] : ["CH4", "C2H6"], f = v => v >= 100 ? Math.round(v) : +v.toPrecision(2);
    let h = `<thead><tr><th>cell</th><th>T</th><th>Γ⟨L⟩ cm</th><th>M</th><th>static fringe, both</th>${gases.map(g => `<th>${g} 1 s</th><th>${g} 15 s</th><th>${g} 30 s</th>`).join("")}${nir ? "" : "<th>CH4 30 s, NEP 1 nW/√Hz</th>"}</tr></thead><tbody>`;
    for (const c of ["d4ref", ...GC.sizes]) { const t = cs.table[c];
      h += `<tr${c === gcSel ? ' style="font-weight:700"' : ""}><td>${c === "d4ref" ? "4 mm, 16 × 50 µm" : GC_D[c] + " mm"}</td><td>${t.T.toFixed(4)}</td><td>${t.GL_cm.toFixed(1)}</td><td>${Math.round(t.M)}</td><td>${t.static[1].toExponential(1)}</td>`
        + gases.map(g => [1, 15, 30].map(T => `<td>${f(t.lim[`${g}_${T}s_NEP14pW`])} ppb</td>`).join("")).join("") + (nir ? "" : `<td>${f(t.lim["CH4_30s_NEP1000pW"])} ppb</td>`) + "</tr>"; }
    return h + "</tbody>";
  }
  const gcSize = () => ({ w: +$("gcW").value, h: +$("gcH").value, fs: +$("gcFs").value });
  function gcDraw() { const { w, h, fs } = gcSize(); gcLast = robSvg(gcPanels(), w, h, fs, gcLegend()); $("gcSheet").innerHTML = gcLast.svg; $("gcTbl").innerHTML = gcTable(); }
  function gcSet(c) {
    gcSel = c;
    document.querySelectorAll("#gcSize button[data-gc]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.gc === c)));
    if ($("gcLive").checked) {      /* the live cell: radius, 8 inclusions r 25 um |dn| 0.01, output facet 11 */
      $("cR").value = String(GC_D[c] / 2); $("nIncC").value = "8"; $("incRC").value = "25"; $("incDnC").value = "-2"; $("cOf").value = "0.458"; $("cNf").value = "24"; $("cTh").value = "7.5";
      syncOut(); runStatus();
    }
    gcDraw();
  }
  document.querySelectorAll("#gcSize button[data-gc]").forEach(b => b.addEventListener("click", () => gcSet(b.dataset.gc)));
  for (const id of ["gcW", "gcH", "gcFs", "gcCase"]) $(id).addEventListener("input", gcDraw);
  $("gcPdf").onclick = async () => {
    const st = $("gcStatus"); st.textContent = "Writing the PDF…";
    try {
      await ROB_CFG.loadPdfLibs();
      const r = gcLast, host = document.createElement("div");
      host.style.cssText = `position:absolute;left:-99999px;top:0;width:${r.W}px;height:${r.H}px;`;
      host.innerHTML = r.svg.replace("<svg ", `<svg width="${r.W}" height="${r.H}" `);
      document.body.appendChild(host);
      const svg = host.firstElementChild, doc = new window.jspdf.jsPDF({ unit: "pt", format: [r.W, r.H], orientation: r.W >= r.H ? "landscape" : "portrait", compress: true });
      ROB_CFG.addFont(doc); svg.setAttribute("font-family", ROB_CFG.pdfFont);
      try { if (typeof doc.svg === "function") await doc.svg(svg, { x: 0, y: 0, width: r.W, height: r.H }); else await window.svg2pdf.svg2pdf(svg, doc, { x: 0, y: 0, width: r.W, height: r.H }); }
      finally { host.remove(); }
      await ROB_CFG.save(`fig2_grating_design_${GC_D[gcSel]}mm_${gcSize().w}x${gcSize().h}cm.pdf`, doc.output("blob"), st);
    } catch (e) { st.textContent = `PDF failed: ${e && e.message || e}`; }
  };
  $("gcSvgBtn").onclick = () => ROB_CFG.save(`fig2_grating_design_${GC_D[gcSel]}mm_${gcSize().w}x${gcSize().h}cm.svg`,
    new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + gcLast.svg.replace("<svg ", `<svg width="${gcSize().w}cm" height="${gcSize().h}cm" `)], { type: "image/svg+xml" }), $("gcStatus"));
  $("gcCsv").onclick = () => {
    const rows = ["case,cell,dither,N,input_factor_rms,both_q1,both_median,both_q3"];
    for (const [ck, cs] of Object.entries(GC.cases)) for (const [c, v] of Object.entries(cs.scaling)) for (const p of v) rows.push([ck, c, p.name, p.N, p.f, ...p.both].join(","));
    rows.push("", "case,cell,mode,regular,chaotic_q1,chaotic_median,chaotic_q3,old_4mm_median");
    for (const [ck, cs] of Object.entries(GC.cases)) for (const [c, b] of Object.entries(cs.bars)) for (const [m] of MODES4) rows.push([ck, c, m, b[m].reg, ...b[m].q, cs.old[m][1]].join(","));
    ROB_CFG.save("fig2_grating_design_data.csv", new Blob([rows.join("\n") + "\n"], { type: "text/csv" }), $("gcStatus"));
  };
  gcDraw();
'''
    s = rep_(s, '\n  robDraw();\n', '\n  robDraw();\n' + js)
    return s


def gc3_sets():
    art = json.load(open(ROOT / "results" / "t11" / "artifact_data.json"))
    out = {}
    for c, D in (("d5", 5), ("d10", 10), ("d15", 15), ("d20", 20)):
        def ren(o):    # Fig. 3 names the input dither "angle" (here angle + position of the grating-coupler beam)
            o = dict(o)
            for k in ("env", "env_lo", "env_hi", "floor"):
                if k in o:
                    o[k] = {("angle" if m == "input" else m): v for m, v in o[k].items()}
            return o
        sl = lambda band, R: {"cur": ren(art["cells"][f"{band}_R{R}_{c}"]["cur"]), "reg": ren(art["cells"][f"{band}_R{R}_{c}"]["reg"])}
        circle = {"c_radius": D * 0.5e-3, "n_facets": 24, "c_theta": 0.1308996938995747, "dither": "input: 75 um waist, +-60 mrad angle, +-100 um position (grating coupler)",
                  "waveform": "sine", "f_theta": 1000, "dlam": 35e-12, "lwave": "triangle", "f_lambda": 1297, "f_parasitic": 1e-6, "stray": 1, "n_inc": 8, "inc_radius": 25e-6,
                  "inc_dn": 0.01, "inc_sign": "both", "inc_seed": 1, "n_eff": 1.5391, "gamma": 0.4532, "wavelength": 1.5317e-6, "R_mirror": 0.97, "membrane_loss_dB_cm": 0.3,
                  "n_seeds": 30, "output_facet": 11}
        out[f"G{D}"] = {
            "label": f"grating-coupler design, {D} mm cell (Task 11)", "n_eff": 1.5391, "n_eff_mir": 1.4253, "noise": {"model": "det", "IL": "sim", "P": 10}, "T": 15, "win": True,
            "smear": {"NH3": 0.804, "CH4": 0.921, "C2H6": 0.824},
            "names": {"current": f"R 97 %, {D} mm (solid)", "target": f"R 99.5 %, {D} mm (dash-dot)"}, "short": {"target": "R 99.5 %"},
            "desc": (f"Grating-coupler design, {D} mm 24-facet circle CMPC (QUIET Task 11, wave-consistent beamlet model): free-space beam on a large-area input "
                     "grating coupler (75 µm waist, ±60 mrad angle and ±100 µm position dither), 8 index inclusions r 25 µm |Δn| 0.01, output on facet 11, "
                     "set B (500 nm SiNx at 1.53 µm, 1000 nm at 3.3 µm), 0.3 dB/cm, ε = 10⁻⁶; “angle” curves = the input angle + position dither; wavelength dither ±35 pm at 1.53 µm (keeps 80 % of the NH₃ peak), "
                     "±81 pm (±2.2 GHz) at 3.3 µm (92 % CH₄, 82 % C₂H₆), line smearing included in the ppb axis. Solid R 97 %, dash-dot R 99.5 %; medians "
                     "and min–max bands over 30 layouts. Curves: undithered fringe below the 1 kHz dither period, the static residual above, rolled off by the "
                     "0.1 rad/s fringe drift (engine envelope; untick “fringe drift” for the static floor). Detector: simulated transmission + 8 dB for the two "
                     "grating couplers, NEP 14 pW/√Hz (a mid-IR detector is likely noisier)."),
            "data": {"current": sl("nir", 0.97), "target": sl("nir", 0.995), "mir": sl("mir", 0.97), "mir_t": sl("mir", 0.995), "circle": circle}}
    return out


def build3_gc(s):
    m = re.search(r'(<script id="dsets" type="application/json">)(.*?)(</script>)', s, re.S)
    ds = json.loads(m.group(2))
    ds.update(gc3_sets())
    s = s[:m.start(2)] + json.dumps(ds, ensure_ascii=False) + s[m.end(2):]
    s = s.replace('<option value="B03" selected>', '<option value="B03">', 1)
    i = s.index('<select id="dset3">') + len('<select id="dset3">')
    s = s[:i] + "".join(f'<option value="G{D}"{" selected" if D == 10 else ""}>grating-coupler design, {D} mm cell (Task 11)</option>' for D in (5, 10, 15, 20)) + s[i:]
    s = rep_(s, '        <div class="ctl"><label>Gas line of (b)</label>',
             '        <div class="btns" id="gcBtns3"><span class="sel">design cell</span>' + "".join(f'<button type="button" data-ds="G{D}" aria-pressed="{str(D == 10).lower()}">{D} mm</button>' for D in (5, 10, 15, 20)) + '</div>\n'
             '        <div class="ctl"><label>Gas line of (b)</label>')
    s = rep_(s, '#spBtns button[aria-pressed="true"] {', '#gcBtns3 button[aria-pressed="true"], #spBtns button[aria-pressed="true"] {')
    # noise: simulated cell transmission + 8 dB couplers
    s = rep_(s, '<select id="nzIL"><option>14</option><option>20</option><option>25</option></select>',
             '<select id="nzIL"><option value="sim">sim. T + 8 dB</option><option>14</option><option>20</option><option>25</option></select>')
    s = rep_(s, 'IL: +document.getElementById("nzIL").value,', 'IL: document.getElementById("nzIL").value,')
    s = rep_(s, '    const Pd = z.P * 10 ** (-z.IL / 10); return Math.sqrt(NEP ** 2 + 2 * H_PL * C0 / (lam || LAM_N) * Pd) / (Pd * Math.sqrt(2 * T));',
             '    const IL = z.IL === "sim" ? -10 * Math.log10(Math.max(o.T_trans, 1e-9)) + 8 : +z.IL;\n'
             '    const Pd = z.P * 10 ** (-IL / 10); return Math.sqrt(NEP ** 2 + 2 * H_PL * C0 / (lam || LAM_N) * Pd) / (Pd * Math.sqrt(2 * T));')
    s = s.replace('insertion loss ${z.IL} dB', 'insertion loss ${z.IL === "sim" ? "simulated cell transmission + 8 dB couplers" : z.IL + " dB"}')
    # line smearing of the design's wavelength dither on the ppb axis
    s = rep_(s, '    aNH3 = Math.max(...st.analyte_per_ppm); LAM_N = sp.lam;',
             '    aNH3 = Math.max(...st.analyte_per_ppm) * ((DSX.smear || {})[sp.mol] || 1); LAM_N = sp.lam;')
    # size buttons follow the data set; ratio readout
    s = rep_(s, '  $("dset3").addEventListener("change", () => { setDS($("dset3").value); cache.clear(); relabel(); draw(); });',
             '  const syncGc3 = () => document.querySelectorAll("#gcBtns3 button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.ds === $("dset3").value)));\n'
             '  $("dset3").addEventListener("change", () => { setDS($("dset3").value); cache.clear(); relabel(); syncGc3(); draw(); });\n'
             '  document.querySelectorAll("#gcBtns3 button").forEach(b => b.addEventListener("click", () => { $("dset3").value = b.dataset.ds; $("dset3").dispatchEvent(new Event("change")); }));')
    s = rep_(s, '        <legend style="padding:0;margin-top:4px">Curves</legend>',
             '        <div class="ctl"><label>C₂H₆ / CH₄ ratio (each line measured for T/2)</label><div class="checks">'
             '<label class="sel mono" for="rCH4">CH₄ (ppm) <input id="rCH4" type="number" min="0.1" max="1000" step="0.1" value="2.5" style="width:5em"></label>'
             '<label class="sel mono" for="rC2">C₂H₆ (ppb) <input id="rC2" type="number" min="0.1" max="100000" step="1" value="125" style="width:6em"></label></div>'
             '<p class="hint mono" id="ratioOut"></p></div>\n'
             '        <legend style="padding:0;margin-top:4px">Curves</legend>')
    s = rep_(s, '  const lshotAt = T => noiseAt(lcur, T), tshotAt = T => noiseAt(tcur, T);',
             '  const lshotAt = T => noiseAt(lcur, T), tshotAt = T => noiseAt(tcur, T);\n'
             '  /* C2H6 / CH4 ratio precision: both 3.3 um lines, each averaged over T/2 (sequential tuning), 1 sigma = the noise-equivalent mixing ratio */\n'
             '  function ratioText(T) {\n'
             '    const x = DSX.data, o = x.mir ? x.mir.cur : cur0, sim = !!x.mir, s = {};\n'
             '    for (const [g, key, lam] of [["CH4", "CH4 3.3", 3.3137e-6], ["C2H6", "C2H6", 3.3368e-6]]) {\n'
             '      const st = D.strips.find(q => q.key === key), al = Math.max(...st.analyte_per_ppm) * ((DSX.smear || {})[g] || 1), Th = T / 2;\n'
             '      s[g] = Math.hypot(interpLog(o.env.both, Th), noiseAt(o, Th, lam)) / (o.gamma * o.L_mean_m * 100 * al) * 1e3;   /* ppb */\n'
             '    }\n'
             '    const c1 = 1e3 * +$("rCH4").value, c2 = +$("rC2").value, r = c2 / c1, sr = r * Math.hypot(s.CH4 / c1, s.C2H6 / c2);\n'
             '    return `C₂H₆/CH₄ = ${(100 * r).toPrecision(3)} % ± ${(100 * sr).toPrecision(2)} % (1σ, T = ${+T.toPrecision(2)} s; σ CH₄ ${+s.CH4.toPrecision(2)} ppb, σ C₂H₆ ${+s.C2H6.toPrecision(2)} ppb${sim ? "" : "; 1.53 µm cell scaled"}). CH₄ absorption under the C₂H₆ line not removed.`;\n'
             '  }')
    s = rep_(s, '  for (const id of ["nzModel", "nzIL", "nzP", "anaT"]) $(id).addEventListener("change", () => { relabel(); draw(); });',
             '  for (const id of ["nzModel", "nzIL", "nzP", "anaT"]) $(id).addEventListener("change", () => { relabel(); draw(); });\n'
             '  for (const id of ["rCH4", "rC2"]) $(id).addEventListener("input", () => draw());')
    s = rep_(s, '    $("avgTOut").textContent = fmtT(T);', '    $("avgTOut").textContent = fmtT(T); try { $("ratioOut").textContent = ratioText(T); } catch (e) { $("ratioOut").textContent = ""; }')
    return s


def main():
    s2 = build2(open(ART / "fig2_src_live.html", encoding="utf-8").read())
    s2 = build2_design(s2)
    (ART / "fig2_v32.html").write_text(s2, encoding="utf-8")
    s3 = build3(open(ART / "fig3_src_live.html", encoding="utf-8").read())
    s3 = build3_gc(s3)
    (ART / "fig3_v32.html").write_text(s3, encoding="utf-8")
    print("ok", len(s2), len(s3))


if __name__ == "__main__":
    main()
