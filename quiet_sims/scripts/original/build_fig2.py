import json,re,sys,os
SRC='/mnt/user-data/outputs/artifacts/2ac86552-411c-4836-94a2-6eb8b081cde2/index.html'
OUT='/mnt/user-data/outputs/fig2_lossy/index.html'
def q(a,p):
    b=sorted(a);i=(len(b)-1)*p;lo=int(i);hi=min(lo+1,len(b)-1);return b[lo]+(b[hi]-b[lo])*(i-lo)
def vals(r):
    a=2*1e-3/(r['grains']**0.5)   # 2 sqrt(eps)/sqrt(grains), eps=1e-6 baseline
    return {'none':a,'spectral':a*r['V_lambda'],'angle':a*r['V_theta'],'both':a*r['V_joint'],'fl_angle':a*r['V_floor'],'fl_both':a*r['V_floor_both'],'T':r['T'],'grains':r['grains'],'L':r['L_mean']}
E={}
SETS=[('ideal','0.995'),('lossy','0.95')]
if os.path.exists('/home/claude/sim/f2_circle_0.99.json') and os.path.exists('/home/claude/sim/f2_herriott_0.99.json') and all(len(json.load(open(f'/home/claude/sim/f2_{c}_0.99.json'))['chaotic'])>=30 for c in ('circle','herriott')): SETS.insert(1,('target','0.99'))
for sname,R in SETS:
    E[sname]={}
    for cell in ('circle','herriott'):
        d=json.load(open(f'/home/claude/sim/f2_{cell}_{R}.json'))
        ch=[vals(x) for x in d['chaotic'] if 'error' not in x]
        reg=vals(d['regular'])
        stat={}
        for k in ('none','spectral','angle','both','fl_angle','fl_both','T','grains','L'):
            v=[x[k] for x in ch]
            stat[k]={'med':q(v,.5),'q1':q(v,.25),'q3':q(v,.75),'min':min(v),'max':max(v)}
        E[sname][cell]={'regular':reg,'chaotic':stat,'n':len(ch),'seeds':{k:[x[k] for x in ch] for k in ('both','angle','spectral')}}
s=open(SRC).read()
def rep(a,b,cnt=1):
    global s
    assert a in s,a[:80]
    s=s.replace(a,b,cnt)

# engine loss patch (same as the node runs)
rep("function ensemble(mc, starts, portW, sIn, sOut, nBounce, Rm, gamma, lamM) {","let ALPHA_M = 0;\n  function setAlpha(a) { ALPHA_M = a; }\n  function ensemble(mc, starts, portW, sIn, sOut, nBounce, Rm, gamma, lamM) {")
rep("Ws.push(Rm < 1 ? Rm ** (nh - 1) : 1);","Ws.push((Rm < 1 ? Rm ** (nh - 1) : 1) * Math.exp(-ALPHA_M * r.WG[nh - 1]));")
rep("return { SPECIES, invNormal,","return { setAlpha, SPECIES, invNormal,")
rep("amp = Float64Array.from(tr.hits, h => (h ? R ** ((h - 1) / 2) : 0));","amp = Float64Array.from(tr.hits, (h, i) => (h ? R ** ((h - 1) / 2) * Math.exp(-0.5 * (o.alpha_m || 0) * tr.L[i]) : 0));")
rep("rays.amp.push(o.R_mirror ** (nh - 1));","rays.amp.push(o.R_mirror ** (nh - 1) * Math.exp(-(o.alpha_m || 0) * r.WG[nh - 1]));")
rep("const S = setup(opts), { o, lamM, mc, q0 } = S, c = mc.cell, pw = c.meta.portW;","const S = setup(opts), { o, lamM, mc, q0 } = S, c = mc.cell, pw = c.meta.portW;\n    G.setAlpha(o.alpha_m || 0);")

# data block
rep('<script id="engine">','<script id="ens" type="application/json">'+json.dumps(E)+'</script>\n<script id="engine">')
rep('<title>Fig. 2 MPC raytracing analysis</title>','<title>Fig. 2 MPC raytracing analysis: ideal vs lossy ensembles</title>')
rep('<h1>Multipass Chaos Figure</h1>','<h1>Multipass Chaos Figure: ideal vs lossy cell (30-layout ensembles)</h1>')
# control for bar mode: insert before the fp control
rep('<div class="ctl"><div class="row"><label for="fp">Parasitic reflectance ε</label>',
 '<div class="ctl"><label for="ensMode">Bars (f)</label><select id="ensMode"><option value="both" selected>ensemble medians: lossy filled, ideal outlined</option><option value="lossy">ensemble: lossy only (R 95 %, 0.3 dB/cm)</option><option value="ideal">ensemble: ideal only (R 99.5 %, no loss)</option><option value="live">live single layout (as original)</option></select></div>\n        <label class="check"><input type="checkbox" id="ensTarget"> add target spec (R 99 %, 0.1 dB/cm) as dashed ticks</label>\n        <div class="btns"><button type="button" id="ensAna">Analyse ensembles</button></div>\n        <div class="ctl"><div class="row"><label for="fp">Parasitic reflectance ε</label>')
# extract original bars function and build barsEns
m=re.search(r'  function bars\(x, y, w, h\) \{.*?\n  \}\n\n  /\* -+ the figure',s,re.S)
orig=m.group(0)
b=orig.replace('function bars(','function barsEns(',1)
b=b.split('\n\n  /* ---')[0]
def brep(a,c):
    global b
    assert a in b,a[:80]
    b=b.replace(a,c,1)
brep('have = ser.filter(([c, k]) => run.res[c][k] && !run.res[c][k].error);','have = ser;')
brep('const vals = have.flatMap(([c, k]) => CATS.map(cat => barVal(run.res[c][k], cat)))','const vals = have.flatMap(([c, k]) => CATS.flatMap(cat => EVS(c, k, cat)))')
brep('const r = run.res[c][k]; if (!r || r.error) return; const v = barVal(r, cat); if (v > 0) tops.push(','const v = Math.max(...EVS(c, k, cat, true)); if (v > 0) tops.push(')
# replace the drawing block
i0=b.index('    CATS.forEach((cat, i) => ser.forEach(([c, k, , col], j) => {\n      const r = run.res[c][k];')
i1=b.index('    if (tgtOn && tgt >= lo && tgt <= hi) {')
draw='''    CATS.forEach((cat, i) => ser.forEach(([c, k, , col], j) => {
      const bx = xt(i) - gw / 2 + j * bw, sets = ensSets();
      for (const sn of sets) {
        const e = EV(sn, c, k, cat); if (!e || !(e.med > 0)) continue;
        const by = Y(Math.max(e.med, lo)), filled = sets.length === 1 || (sn === sets[sets.length - 1] && sn !== "target");
        if (sn === "target" && !filled) path([[bx + 0.4, by], [bx + bw - 0.4, by]], { stroke: "#000", sw: 1.6, dash: [2, 1.2] });
        else if (filled) rect(bx + 0.4, by, bw - 0.8, y + h - by, { fill: col });
        else rect(bx + 0.4, by, bw - 0.8, y + h - by, { stroke: col, sw: 1.1 });
        if (e.q3 > e.q1) { const xc = bx + bw / 2, yl = Y(Math.max(e.q1, lo)), yh = Y(e.q3);
          path([[xc, yl], [xc, yh]], { stroke: "#000", sw: 0.8 }); path([[xc - 1.6, yl], [xc + 1.6, yl]], { stroke: "#000", sw: 0.8 }); path([[xc - 1.6, yh], [xc + 1.6, yh]], { stroke: "#000", sw: 0.8 }); }
        if (filled) { const le = e.med <= 3 * e.fl;
          if ($("bVals").checked) text(bx + bw / 2 + FSZ * 0.25, by - 2 - (e.q3 > e.med ? Math.min(14, Math.max(0, Y(e.med) - Y(e.q3))) : 0), `${le ? "≤" : ""}${sci(e.med, 2).replace(/e(-?\\d+)/, (m, ee) => `·10^{${ee.replace("-", "−")}}`)}`, { rot: true, fs: FSZ * 0.7 }); }
      }
    }));
'''
b=b[:i0]+draw+b[i1:]
helpers='''
  /* ensemble data: medians and quartiles over 30 inclusion layouts (chaotic), single values (regular); amplitudes at ε = 1e-6, scaled by √ε */
  const ENS = JSON.parse($("ens").textContent);
  $("ensMode").addEventListener("change", () => draw());
  $("ensTarget").addEventListener("change", () => draw());
  if (!ENS.target) $("ensTarget").parentElement.hidden = true;
  const ensSets = () => { const m = $("ensMode").value; let l = m === "both" ? ["ideal", "lossy"] : [m]; if ($("ensTarget").checked && ENS.target) l = ["ideal", "target", "lossy"].filter(s => l.includes(s) || s === "target"); return l; };
  function EV(sn, c, k, cat) {
    const d = ENS[sn][c], f = Math.sqrt(eps() / 1e-6), key = cat[1];
    if (k === "regular") { const v = d.regular[key] * f; return { med: v, q1: v, q3: v, fl: cat[3] ? 0 : 0 }; }
    const st = d.chaotic[key], fk = key === "angle" ? "fl_angle" : key === "both" ? "fl_both" : null;
    return { med: st.med * f, q1: st.q1 * f, q3: st.q3 * f, fl: fk ? d.chaotic[fk].med * f : 0 };
  }
  function EVS(c, k, cat, topOnly) {
    const out = [];
    for (const sn of ensSets()) { const e = EV(sn, c, k, cat); if (e.med > 0) { out.push(topOnly ? Math.max(e.q3, e.med) : e.med); if (!topOnly && e.q1 > 0) out.push(e.q1); } }
    return out.length ? out : [0];
  }
'''
rep('  function bars(x, y, w, h) {', helpers+'  '+b.strip()+'\n  const barsLive = (...a) => barsOrig(...a);\n  function bars(x, y, w, h) {\n    if ($("ensMode").value !== "live") return barsEns(x, y, w, h);\n    return barsOrig(x, y, w, h);\n  }\n  function barsOrig(x, y, w, h) {')
rep('        <p class="hint" id="valsNote"></p>\n      </div>','        <p class="hint" id="valsNote"></p>\n      </div>\n      <div class="panel" id="ensPanel" hidden><h2>Ensemble analysis (30 layouts, static residual fringe, target 5e-7)</h2><div class="tbl"><table class="vals" id="ensTbl"></table></div><p class="hint">Chaotic median with quartiles; the regular cell has no inclusions. Below target = share of the 30 layouts under 5e-7.</p></div>')
ana = r"""
  $("ensAna").onclick = () => {
    const nm = { ideal: "ideal (R 99.5 %, 0 dB/cm)", target: "target (R 99 %, 0.1 dB/cm)", lossy: "minimum (R 95 %, 0.3 dB/cm)" }, f = v => v.toExponential(1), ks = ["ideal", "target", "lossy"].filter(k => ENS[k]), fz = Math.sqrt(eps() / 1e-6), TG = 5e-7 / fz;
    let h = "<thead><tr><th>cell</th><th>case</th><th>dither</th><th>regular</th><th>chaotic median</th><th>quartiles</th><th>regular/chaotic</th><th>layouts below target</th></tr></thead><tbody>";
    for (const [c, cn] of [["circle", "CMPC"], ["herriott", "Herriott"]]) for (const k of ks) for (const m of ["spectral", "angle", "both"]) {
      const d = ENS[k][c], st = d.chaotic[m], below = d.seeds[m].filter(v => v < TG).length;
      h += `<tr><td>${cn}</td><td>${nm[k]}</td><td>${m}</td><td>${f(d.regular[m] * fz)}</td><td>${f(st.med * fz)}</td><td>${f(st.q1 * fz)} to ${f(st.q3 * fz)}</td><td>${(d.regular[m] / st.med).toFixed(1)}</td><td>${below} / ${d.seeds[m].length}</td></tr>`;
    }
    $("ensTbl").innerHTML = h + "</tbody>"; $("ensPanel").hidden = false;
  };
"""
rep('  const barsLive = (...a) => barsOrig(...a);', ana + '  const barsLive = (...a) => barsOrig(...a);')

# live runs: use the lossy cell settings (R 95 %, loss) so the rays and tables match the lossy set
rep('n_eff: 2, R_mirror: 0.995, gamma: 0.2, P_in: 1e-3,\n      dither: +$("dA")','n_eff: 2, R_mirror: 0.95, alpha_m: 0.3 / 4.3429448190325175 / 0.01, gamma: 0.2, P_in: 1e-3,\n      dither: +$("dA")')
# redraw on mode change: add to listener list if exists
open(OUT,'w').write(s)
print('ok',len(s),{k:{c:E[k][c]['n'] for c in E[k]} for k in E})
