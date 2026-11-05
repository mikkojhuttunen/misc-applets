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
                "target": T2 / f"circle_nir_R0.995_{dB}dB.json", "mir": T2 / f"circle_mir_R0.97_{dB}dB.json"}
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


def main():
    s2 = build2(open(ART / "fig2_src_live.html", encoding="utf-8").read())
    (ART / "fig2_v32.html").write_text(s2, encoding="utf-8")
    s3 = build3(open(ART / "fig3_src_live.html", encoding="utf-8").read())
    (ART / "fig3_v32.html").write_text(s3, encoding="utf-8")
    print("ok", len(s2), len(s3))


if __name__ == "__main__":
    main()
