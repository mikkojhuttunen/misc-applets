"""Fig. 3 artifact (HITRAN spectra + NH3 noise floor) with the step-10 spec cases of the 600 nm SiNx circle.

python scripts/build_fig3_artifact.py SRC.html OUT.html
SRC is the saved Fig. 3 artifact HTML (ideal / minimum / target version of scripts/original/build_fig3.py).
Panel (b): solid curves = target spec (designed R(chi), 0.1 dB/cm; the proposal numbers), dashed = minimum spec
(R <= 95 % below the critical angle, 0.3 dB/cm), dash-dot = ideal (designed R(chi), no loss). Medians and min-max bands
over 30 inclusion layouts (results/step10/circle_<case>.json); the noise traces of panel (a) use the exit paths of
the representative target layout (results/step10/rep_circle_<case>.json from scripts/rep_paths.js).
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
MODES = ['none', 'angle', 'spectral', 'both']
SPEC = {'ideal': ('designed R(χ)', 0.0), 'target': ('designed R(χ)', 0.1), 'minimum': ('R ≤ 95 % below θc', 0.3)}


def q(a, p):
    b = sorted(a)
    i = (len(b) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(b) - 1)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def main(src_path, out_path):
    s = open(src_path, encoding='utf-8').read()
    m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', s, re.S)
    d = json.loads(m.group(2))
    Tg = d['T']
    lg = [math.log(t) for t in Tg]

    def at1(arr):
        i = 0
        while i < len(lg) - 2 and lg[i + 1] < 0:
            i += 1
        f = (0 - lg[i]) / (lg[i + 1] - lg[i])
        return math.exp(math.log(arr[i]) * (1 - f) + math.log(arr[i + 1]) * f)

    shot = lambda T: math.sqrt(1.602176634e-19 / (1e-3 * T))
    aNH3 = max(d['strips'][0]['analyte_per_ppm'])

    def pack(case):
        r = json.load(open(os.path.join(RES, 'step10', f'circle_{case}.json')))
        rp = json.load(open(os.path.join(RES, 'step10', f'rep_circle_{case}.json')))
        c = [x for x in r['chaotic'] if 'error' not in x]
        g0 = r['gamma']
        nT = len(Tg)
        env = {m_: [q([x['env'][m_][i] for x in c], .5) for i in range(nT)] for m_ in MODES}
        lo = {m_: [min(x['env'][m_][i] for x in c) for i in range(nT)] for m_ in MODES}
        hi = {m_: [max(x['env'][m_][i] for x in c) for i in range(nT)] for m_ in MODES}
        Tm, Lm, M2 = (q([x[k] for x in c], .5) for k in ('T_trans', 'L_mean_m', 'M2'))
        cur = {'gamma': g0, 'L_mean_m': Lm, 'T_trans': Tm, 'M2': M2, 'Ls_m': rp['chaotic']['Ls_m'], 'Ws': rp['chaotic']['Ws'],
               'env': env, 'env_lo': lo, 'env_hi': hi, 'floor': {m_: q([x['floor'][m_] for x in c], .5) for m_ in MODES}}
        g = r['regular']
        reg = {'gamma': g0, 'L_mean_m': g['L_mean_m'], 'T_trans': g['T_trans'], 'M2': g['M2'], 'Ls_m': rp['regular']['Ls_m'],
               'Ws': rp['regular']['Ws'], 'env': g['env'], 'floor': g['floor']}
        nea = [math.hypot(at1(x['env']['both']), shot(x['T_trans'])) for x in c]
        ppb = [n / (g0 * x['L_mean_m'] * 100 * aNH3) * 1e3 for n, x in zip(nea, c)]
        fr = [at1(x['env']['both']) for x in c]
        sh = [shot(x['T_trans']) for x in c]
        nn = [at1(x['env']['none']) for x in c]
        rn = math.hypot(at1(g['env']['both']), shot(g['T_trans']))
        rppb = rn / (g0 * g['L_mean_m'] * 100 * aNH3) * 1e3
        st = {'GL_cm': g0 * Lm * 100, 'T': Tm, 'M2': M2, 'ppb_med': q(ppb, .5), 'ppb_min': min(ppb), 'ppb_max': max(ppb),
              'ppb_q1': q(ppb, .25), 'ppb_q3': q(ppb, .75), 'nea_med': q(nea, .5), 'fringe_med': q(fr, .5), 'shot_med': q(sh, .5),
              'reg_ppb': rppb, 'reg_GL_cm': g0 * g['L_mean_m'] * 100, 'reg_T': g['T_trans'], 'adv': rppb / q(ppb, .5),
              'dither_gain': q(nn, .5) / q(nea, .5), 'n': len(c), 'R_bounce': q([x['R_bounce'] for x in c], .5)}
        return cur, reg, st, r

    stats = {}
    packs = {k: pack(k) for k in ('ideal', 'target', 'minimum')}
    d['current'], d['regular'], stats['target'], rt = packs['target']
    d['lossy_current'], d['lossy_regular'], stats['lossy'], _ = packs['minimum']
    d['target_current'], d['target_regular'], stats['ideal'], _ = packs['ideal']
    d['stats'] = stats
    d['n_eff'] = 1.66
    st_ = rt['settings']
    d['chaos']['circle'].update({'c_theta': st_['c_theta'], 'R_mirror': 'R(|sin chi|) table, designed 20-period trench DBR',
                                 'membrane_loss_dB_cm': 0.1, 'n_seeds': 30, 'n_eff': 1.66, 'gamma': rt['gamma']})
    d['chaos']['lossy'] = {'R_mirror': 'designed R(chi) capped at 0.95 below the critical angle', 'membrane_loss_dB_cm': 0.3, 'n_seeds': 30}
    d['chaos']['target'] = {'R_mirror': 'designed R(chi) (no cap)', 'membrane_loss_dB_cm': 0.0, 'n_seeds': 30,
                            'note': 'the "target" key of the drawing code holds the ideal case; the solid curves are the target spec'}
    d['chaos']['source'] = ('QUIET step 10 (DitherLab with membrane loss and R(|sin chi|) tables): 600 nm SiNx circle, TM0, n_eff 1.66, Gamma 0.313, '
                            '30 inclusion layouts (medians, min-max bands). Solid: target spec (designed R(chi), 0.1 dB/cm); dashed: minimum spec '
                            '(R <= 0.95 below the critical angle, 0.3 dB/cm); dash-dot: ideal (designed R(chi), no loss)')
    s = s[:m.start(2)] + json.dumps(d) + s[m.end(2):]

    def rep(a, b):
        nonlocal s
        assert a in s, a[:90]
        s = s.replace(a, b, 1)

    rep('<h1>Fig. 6 Playground: ideal vs minimum vs target cell</h1>', '<h1>Fig. 3 spectra and noise: ideal vs target vs minimum spec</h1>')
    rep("Noise: the Multipass Chaos Lab's chaotic circular cell, NH₃ line, built from its traced exit paths.",
        "Noise: the chaotic circular cell of the QUIET proposal, 600 nm SiNx membrane, TM₀ (n_eff 1.66, Γ 0.31), trench-DBR mirrors with "
        "angle-dependent R(χ), NH₃ line, 1 mW; medians and min–max bands over 30 inclusion layouts. Panel (b): solid, target spec (designed R(χ), "
        "0.1 dB/cm); dashed, minimum spec (R ≤ 95 % below the critical angle, 0.3 dB/cm); dash-dot, ideal (designed R(χ), no membrane loss).")
    rep('["lossy", "minimum spec (R 95 %, 0.3 dB/cm), dashed"], ["target", "target spec (R 99 %, 0.1 dB/cm), dash-dot"], ["bands", "min–max bands (21 layouts)"]',
        '["lossy", "minimum spec (R ≤ 95 % below θc, 0.3 dB/cm), dashed"], ["target", "ideal (designed R(χ), no loss), dash-dot"], ["bands", "min–max bands (30 layouts)"]')
    rep('${k === "target" ? "" : "checked"}', 'checked')
    # spec labels at T: beside the marker (right of it unless T is near the right axis), white halo; minimum above, ideal below
    rep('return `<circle cx="${BX(T)}" cy="${BY(v)}" r="2.6" fill="none" stroke="#000" stroke-width="1"/>` + txt(Math.max(BX(T) - 36.8, bl + 40), Math.min(BY(v) - dy, bb - 4), l, { anchor: "middle" }); };',
        'const rt = BX(T) < br - 95, y = Math.min(Math.max(BY(v) + dy, bt + 10), bb - 4);\n'
        '        return `<circle cx="${BX(T)}" cy="${BY(v)}" r="2.6" fill="none" stroke="#000" stroke-width="1"/>` + txt(rt ? BX(T) + 6 : BX(T) - 6, y, l, { anchor: rt ? "start" : "end" }).replace("<text ", \'<text stroke="#fff" stroke-width="3" stroke-linejoin="round" paint-order="stroke" \'); };')
    rep('if (on("lossy")) s += lbl(lcur, lshotAt, lGL, "min. spec", 8);\n      if (on("target") && tcur) s += lbl(tcur, tshotAt, tGL, "target spec", 20);',
        'if (on("lossy")) s += lbl(lcur, lshotAt, lGL, "min. spec", -5);\n      if (on("target") && tcur) s += lbl(tcur, tshotAt, tGL, "ideal", 13);')
    rep('names = { ideal: "ideal (R 99.5 %, 0 dB/cm)", target: "target (R 99 %, 0.1 dB/cm)", lossy: "minimum (R 95 %, 0.3 dB/cm)" }',
        'names = { ideal: "ideal (designed R(χ), no loss)", target: "target (designed R(χ), 0.1 dB/cm)", lossy: "minimum (R ≤ 95 % below θc, 0.3 dB/cm)" }')
    rep('["fringe / shot", k => f(S[k].fringe_med / S[k].shot_med)],\n      ["3.3 µm, scaled by 150–230× (ppb)", k => `${f(S[k].ppb_med / 230)} to ${f(S[k].ppb_med / 150)}`], ["3.3 µm, worst layout (ppb)", k => `${f(S[k].ppb_max / 230)} to ${f(S[k].ppb_max / 150)}`]];',
        '["fringe / shot", k => f(S[k].fringe_med / S[k].shot_med)], ["⟨R⟩ per bounce (detected light)", k => (+S[k].R_bounce).toFixed(4)]];')
    rep('"Medians over 21 inclusion layouts; the regular cell is the same cell and launch without inclusions. ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_NH₃) at T = 1 s with 1 mW input; 3.3 µm values assume the crude 150–230× line-strength scaling at equal path and noise."',
        '"Medians over 30 inclusion layouts; the regular cell is the same cell and launch without inclusions. 600 nm SiNx, TM₀, Γ = 0.31. ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_NH₃) at T = 1 s with 1 mW input (shot noise and fringe residual, drift 0.1 rad/s). At 3.3 µm the 1.2 µm SiNx circle reaches 1.5 ppb CH₄ at 0.1 dB/cm and 3.1 ppb at 0.3 dB/cm (QUIET step 4, HITRAN line strengths)."')
    rep('note: "right ppb axis uses the ideal cell Gamma<L>; each other set is labelled with its own Gamma<L>"',
        'note: "solid curves and the right ppb axis: target spec (Gamma<L> of that cell); the dashed (minimum) and dash-dot (ideal) sets are labelled with their own Gamma<L>"')
    open(out_path, 'w', encoding='utf-8').write(s)
    print('ok', {k: round(v['ppb_med'], 1) for k, v in stats.items()})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
