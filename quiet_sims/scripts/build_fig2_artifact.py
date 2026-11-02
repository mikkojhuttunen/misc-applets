"""Fig. 2 artifact (ray tracing, static residual fringe) with the step-10 spec cases of the 600 nm SiNx cell.

python scripts/build_fig2_artifact.py SRC.html OUT.html
SRC is the saved Fig. 2 artifact HTML (ideal / target / minimum version of scripts/original/build_fig2.py). The script
  - swaps the page's engine for engine/quiet_engine.js (R(|sin chi|) tables),
  - replaces the ensemble data with results/step10/<cell>_<case>.json (30 layouts + regular),
  - adds the R tables and a case selector for the live run (n_eff 1.66, Gamma 0.313),
  - relabels the three cases.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
CASES = [('ideal', 'ideal'), ('target', 'target'), ('lossy', 'minimum')]   # artifact key, step-10 file
NAMES = {'ideal': 'ideal (designed R(χ), no loss)', 'target': 'target (designed R(χ), 0.1 dB/cm)',
         'lossy': 'minimum (R ≤ 95 % below θc, 0.3 dB/cm)'}
DBCM = {'ideal': 0.0, 'target': 0.1, 'lossy': 0.3}


def q(a, p):
    b = sorted(a)
    i = (len(b) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(b) - 1)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def vals(r):
    a = 2 * 1e-3 / (r['grains'] ** 0.5)     # 2 sqrt(eps) / sqrt(grains), eps = 1e-6
    return {'none': a, 'spectral': a * r['V_lambda'], 'angle': a * r['V_theta'], 'both': a * r['V_joint'],
            'fl_angle': a * r['V_floor'], 'fl_both': a * r['V_floor_both'], 'T': r['T_trans'], 'grains': r['grains'],
            'L': r['L_mean_m'], 'GL_cm': r['gamma'] * r['L_mean_m'] * 100, 'R_bounce': r.get('R_bounce', float('nan'))}


def ensembles():
    E = {}
    for key, fname in CASES:
        E[key] = {}
        for cell in ('circle', 'herriott'):
            d = json.load(open(os.path.join(RES, 'step10', f'{cell}_{fname}.json')))
            ch = [vals(x) for x in d['chaotic'] if 'error' not in x]
            stat = {k: {'med': q([x[k] for x in ch], .5), 'q1': q([x[k] for x in ch], .25), 'q3': q([x[k] for x in ch], .75),
                        'min': min(x[k] for x in ch), 'max': max(x[k] for x in ch)} for k in ch[0]}
            E[key][cell] = {'regular': vals(d['regular']), 'chaotic': stat, 'n': len(ch),
                            'seeds': {k: [x[k] for x in ch] for k in ('both', 'angle', 'spectral')}}
    return E


def rtables():
    out = {}
    for cell in ('circle', 'herriott'):
        out[cell] = {}
        for key, suf in (('design', ''), ('min95', '_min95')):
            t = json.load(open(os.path.join(RES, 'step1', f'Rtable_{cell}_neff1.66{suf}.json')))
            out[cell][key] = {'s': [round(v, 7) for v in t['s']], 'R': [round(v, 9) for v in t['R']], 'R_avg': t['design']['R_avg']}
    return out


def engine_block():
    src = open(os.path.join(ROOT, 'engine', 'quiet_engine.js')).read()
    a = src.index('/* engine:begin')
    b = src.index('/* engine:end */') + len('/* engine:end */')
    return src[a:b]


def main(src_path, out_path):
    s = open(src_path, encoding='utf-8').read()

    def rep(a, b, cnt=1):
        nonlocal s
        n = s.count(a)
        assert n >= 1, a[:90]
        s = s.replace(a, b, cnt)

    # engine with R(|sin chi|) tables
    a = s.index('/* engine:begin')
    b = s.index('/* engine:end */') + len('/* engine:end */')
    s = s[:a] + engine_block() + s[b:]
    # data
    i0 = s.index('<script id="ens" type="application/json">')
    i1 = s.index('</script>', i0)
    s = s[:i0] + '<script id="ens" type="application/json">' + json.dumps(ensembles()) + s[i1:]
    s = s.replace('<script id="engine">', '<script id="rtab" type="application/json">' + json.dumps(rtables()) + '</script>\n<script id="engine">', 1)

    # target line of panel (f) at 5e-7, as the proposal caption and the ensemble table
    rep('<input id="tgt" type="range" min="-7" max="-2" step="0.1" value="-5">', '<input id="tgt" type="range" min="-7" max="-2" step="0.1" value="-6.3">')
    # titles and text
    rep('<title>Fig. 2 MPC raytracing analysis: ideal vs lossy ensembles</title>', '<title>Fig. 2 MPC raytracing analysis</title>')
    rep('<h1>Multipass Chaos Figure: ideal vs lossy cell (30-layout ensembles)</h1>',
        '<h1>Multipass Chaos Figure: ideal vs target vs minimum spec (30-layout ensembles)</h1>')
    rep('All figure text is Arial (Liberation Sans in the PDF); set the size below the figure.</p>',
        'All figure text is Arial (Liberation Sans in the PDF); set the size below the figure. QUIET cell: free-standing 600 nm SiNx membrane, TM₀ '
        '(n_eff 1.66, Γ 0.31), trench-DBR mirrors with the angle-dependent reflectivity R(χ) of a 20-period first-order design (total internal '
        'reflection above the critical angle θc = 37°). Specs: ideal, the designed R(χ) without membrane loss; target, designed R(χ) and 0.1 dB/cm; '
        'minimum, R capped at 95 % below θc and 0.3 dB/cm. Bars (f): medians and quartiles over 30 inclusion layouts.</p>')
    rep('<option value="both" selected>ensemble medians: lossy filled, ideal outlined</option><option value="lossy">ensemble: lossy only (R 95 %, 0.3 dB/cm)</option>'
        '<option value="ideal">ensemble: ideal only (R 99.5 %, no loss)</option><option value="live">live single layout (as original)</option>',
        '<option value="both" selected>ensemble medians: minimum filled, ideal outlined</option><option value="target">ensemble: target only (designed R(χ), 0.1 dB/cm)</option>'
        '<option value="lossy">ensemble: minimum only (R ≤ 95 % below θc, 0.3 dB/cm)</option><option value="ideal">ensemble: ideal only (designed R(χ), no loss)</option>'
        '<option value="live">live single layout (case below)</option>')
    rep('<label class="check"><input type="checkbox" id="ensTarget"> add target spec (R 99 %, 0.1 dB/cm) as dashed ticks</label>',
        '<label class="check"><input type="checkbox" id="ensTarget" checked> add target spec (designed R(χ), 0.1 dB/cm) as dashed ticks</label>\n'
        '        <div class="ctl"><label for="liveCase">Live run: case</label><select id="liveCase" class="sim"><option value="ideal">ideal (designed R(χ), no loss)</option>'
        '<option value="target" selected>target (designed R(χ), 0.1 dB/cm)</option><option value="lossy">minimum (R ≤ 95 % below θc, 0.3 dB/cm)</option></select></div>')
    rep('const nm = { ideal: "ideal (R 99.5 %, 0 dB/cm)", target: "target (R 99 %, 0.1 dB/cm)", lossy: "minimum (R 95 %, 0.3 dB/cm)" }',
        'const nm = ' + json.dumps(NAMES, ensure_ascii=False))
    rep('let l = m === "both" ? ["ideal", "lossy"] : [m]; if ($("ensTarget").checked && ENS.target)',
        'let l = m === "both" ? ["ideal", "lossy"] : [m]; if (m === "both" && $("ensTarget").checked && ENS.target)')
    rep('<p class="hint">Chaotic median with quartiles; the regular cell has no inclusions. Below target = share of the 30 layouts under 5e-7.</p>',
        '<p class="hint">Chaotic median with quartiles over 30 inclusion layouts; the regular cell has no inclusions. Below target = share of the 30 layouts '
        'under 5e-7. 600 nm SiNx, TM₀, n_eff 1.66; Γ⟨L⟩ with Γ = 0.31.</p>')
    # analysis table: add transmission, Gamma<L> and bounce-averaged R per case (both dithers row)
    rep('$("ensTbl").innerHTML = h + "</tbody>"; $("ensPanel").hidden = false;',
        'h += "</tbody>";\n'
        '    let h2 = "<thead><tr><th>cell</th><th>case</th><th>T (chaotic)</th><th>Γ⟨L⟩ cm (chaotic)</th><th>⟨R⟩ per bounce</th><th>T (regular)</th><th>Γ⟨L⟩ cm (regular)</th></tr></thead><tbody>";\n'
        '    for (const [c, cn] of [["circle", "CMPC"], ["herriott", "Herriott"]]) for (const k of ks) { const d = ENS[k][c], g = (x, n = 3) => (+x).toPrecision(n);\n'
        '      h2 += `<tr><td>${cn}</td><td>${nm[k]}</td><td>${g(d.chaotic.T.med, 2)}</td><td>${g(d.chaotic.GL_cm.med)}</td><td>${(+d.chaotic.R_bounce.med).toFixed(4)}</td><td>${g(d.regular.T, 2)}</td><td>${g(d.regular.GL_cm)}</td></tr>`; }\n'
        '    $("ensTbl").innerHTML = h + "</table><table class=\\"vals\\">" + h2 + "</tbody>"; $("ensPanel").hidden = false;')
    # live run: the QUIET cell of the selected case
    rep('const common = Object.assign({ wavelength: LAMS[$("sp").value], species: $("sp").value, n_eff: 2, R_mirror: 0.95, alpha_m: 0.3 / 4.3429448190325175 / 0.01, gamma: 0.2, P_in: 1e-3,',
        'const lc = $("liveCase").value, RTB = JSON.parse($("rtab").textContent), tabOf = c => RTB[c][lc === "lossy" ? "min95" : "design"];\n'
        '    const DBC = ' + json.dumps(DBCM) + ';\n'
        '    const common = Object.assign({ wavelength: LAMS[$("sp").value], species: $("sp").value, n_eff: 1.66, R_mirror: 0.99, alpha_m: DBC[lc] / 4.3429448190325175 / 0.01, gamma: 0.313, P_in: 1e-3,')
    rep('f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1 }, inc("C"));',
        'f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1, R_table: { s: tabOf("circle").s, R: tabOf("circle").R }, R_mirror: Math.min(tabOf("circle").R_avg, 0.99999) }, inc("C"));')
    rep('tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0, ripple_period: 0.5e-3 }, inc("H"));',
        'tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0, ripple_period: 0.5e-3, R_table: { s: tabOf("herriott").s, R: tabOf("herriott").R }, R_mirror: Math.min(tabOf("herriott").R_avg, 0.99999) }, inc("H"));')
    rep('$("valsNote").textContent = `Static residual fringe 2√ε V / √(speckle grains), ε = ${sci(eps(), 2)};',
        '$("valsNote").textContent = `Live run: ${({ ideal: "ideal", target: "target", lossy: "minimum" })[$("liveCase").value]} spec, 600 nm SiNx TM₀ (n_eff 1.66), R(χ) table, ${(p.alpha_m * 4.3429448190325175 * 0.01).toFixed(1)} dB/cm. Static residual fringe 2√ε V / √(speckle grains), ε = ${sci(eps(), 2)};')
    open(out_path, 'w', encoding='utf-8').write(s)
    print('ok', len(s))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
