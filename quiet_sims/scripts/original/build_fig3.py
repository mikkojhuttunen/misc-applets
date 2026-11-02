import json,re,math,os
SRC='/mnt/user-data/outputs/artifacts/628b037c-494d-4560-828b-d75d8d275190/index.html'
OUT='/mnt/user-data/outputs/fig3_lossy/index.html'
d=json.load(open('/home/claude/sim/fig3data.json'))
Tg=d['T']
def q(a,p):
    b=sorted(a);i=(len(b)-1)*p;lo=int(i);hi=min(lo+1,len(b)-1);return b[lo]+(b[hi]-b[lo])*(i-lo)
def at1(arr):
    x=math.log(1.0);lg=[math.log(t) for t in Tg];i=0
    while i<len(lg)-2 and lg[i+1]<x:i+=1
    f=(x-lg[i])/(lg[i+1]-lg[i]);return math.exp(math.log(arr[i])*(1-f)+math.log(arr[i+1])*f)
shot=lambda T:math.sqrt(1.602176634e-19/(1e-3*T*1.0))
aNH3=max(d['strips'][0]['analyte_per_ppm'])
modes=['none','angle','spectral','both']
def pack(r,n=21):
    c=r['chaotic'][:n]
    env={m:[q([x['env'][m][i] for x in c],.5) for i in range(161)] for m in modes}
    lo={m:[min(x['env'][m][i] for x in c) for i in range(161)] for m in modes}
    hi={m:[max(x['env'][m][i] for x in c) for i in range(161)] for m in modes}
    Tm=q([x['T_trans'] for x in c],.5);Lm=q([x['L_mean_m'] for x in c],.5);M2=q([x['M2'] for x in c],.5)
    rep=min(c,key=lambda x:abs(x['L_mean_m']-Lm))
    cur={'gamma':0.2,'L_mean_m':Lm,'T_trans':Tm,'M2':M2,'Ls_m':rep['Ls_m'],'Ws':rep['Ws'],'env':env,'env_lo':lo,'env_hi':hi,'floor':{m:q([x['floor'][m] for x in c],.5) for m in modes}}
    g=r['regular']
    reg={'gamma':0.2,'L_mean_m':g['L_mean_m'],'T_trans':g['T_trans'],'M2':g['M2'],'Ls_m':g['Ls_m'],'Ws':g['Ws'],'env':g['env'],'floor':g['floor']}
    # statistics at T = 1 s
    nea=[math.hypot(at1(x['env']['both']),shot(x['T_trans'])) for x in c]
    ppb=[n/(0.2*x['L_mean_m']*100*aNH3)*1e3 for n,x in zip(nea,c)]
    fr=[at1(x['env']['both']) for x in c]; sh=[shot(x['T_trans']) for x in c]
    nn=[at1(x['env']['none']) for x in c]
    rn=math.hypot(at1(g['env']['both']),shot(g['T_trans'])); rppb=rn/(0.2*g['L_mean_m']*100*aNH3)*1e3
    st={'GL_cm':0.2*Lm*100,'T':Tm,'M2':M2,'ppb_med':q(ppb,.5),'ppb_min':min(ppb),'ppb_max':max(ppb),'ppb_q1':q(ppb,.25),'ppb_q3':q(ppb,.75),
        'nea_med':q(nea,.5),'fringe_med':q(fr,.5),'shot_med':q(sh,.5),'reg_ppb':rppb,'reg_GL_cm':0.2*g['L_mean_m']*100,'reg_T':g['T_trans'],
        'adv':rppb/q(ppb,.5),'dither_gain':q(nn,.5)/q(nea,.5),'n':len(c)}
    return cur,reg,st
sets={'ideal':'ens_base.json','lossy':'ens_new.json'}
if os.path.exists('/home/claude/sim/ens_tgt.json') and len(json.load(open('/home/claude/sim/ens_tgt.json'))['chaotic'])>=21: sets['target']='ens_tgt.json'
stats={}
for k,f in sets.items():
    cur,reg,st=pack(json.load(open('/home/claude/sim/'+f)))
    stats[k]=st
    if k=='ideal': d['current'],d['regular']=cur,reg
    elif k=='lossy': d['lossy_current'],d['lossy_regular']=cur,reg
    else: d['target_current'],d['target_regular']=cur,reg
d['stats']=stats
d['chaos']['circle'].update({'c_theta':7.5*math.pi/180,'R_mirror':0.995,'membrane_loss_dB_cm':0.0,'n_seeds':21})
d['chaos']['lossy']={'R_mirror':0.95,'membrane_loss_dB_cm':0.3,'n_seeds':21}
d['chaos']['target']={'R_mirror':0.99,'membrane_loss_dB_cm':0.1,'n_seeds':21}
d['chaos']['source']='Multipass Chaos Lab engine (DitherLab) re-run with membrane loss exp(-alpha L): 21 inclusion layouts (medians, min-max bands); ideal R=99.5 %, no loss; minimum R=95 %, 0.3 dB/cm; target R=99 %, 0.1 dB/cm'
s=open(SRC).read()
s=re.sub(r'(<script id="data" type="application/json">)(.*?)(</script>)',lambda m:m.group(1)+json.dumps(d)+m.group(3),s,count=1,flags=re.S)
def rep(a,b):
    global s
    assert a in s,a[:70]
    s=s.replace(a,b,1)
rep('<h1>Fig. 6 Playground</h1>','<h1>Fig. 6 Playground: ideal vs minimum vs target cell</h1>')
rep('const cur = D.current, reg = D.regular, GL = cur.gamma * cur.L_mean_m * 100;',
 'const cur = D.current, reg = D.regular, GL = cur.gamma * cur.L_mean_m * 100;\n  const lcur = D.lossy_current, lreg = D.lossy_regular, lGL = lcur.gamma * lcur.L_mean_m * 100;\n  const lshotAt = T => Math.sqrt(Q_E / (D.P_in * lcur.T_trans * T));\n  const tcur = D.target_current, treg = D.target_regular, tGL = tcur ? tcur.gamma * tcur.L_mean_m * 100 : 0;\n  const tshotAt = T => Math.sqrt(Q_E / (D.P_in * tcur.T_trans * T));')
rep('["total", "total"]];','["total", "total"], ["lossy", "minimum spec (R 95 %, 0.3 dB/cm), dashed"], ["target", "target spec (R 99 %, 0.1 dB/cm), dash-dot"], ["bands", "min–max bands (21 layouts)"]];')
rep('`<label class="check"><input type="checkbox" id="b-${k}" checked> ${n}</label>`','`<label class="check"><input type="checkbox" id="b-${k}" ${k === "target" ? "" : "checked"}> ${n}</label>`')
old='    for (const [k, arr, col, w, da] of BC) if (on(k)) s += '
new='''    if (on("bands")) {
      for (const [m, col] of [["none", "#9467bd"], ["spectral", "#e377c2"], ["angle", "#2ca02c"], ["both", "#1f77b4"]]) {
        const bnd = (o, op) => { const up = Ts.map((t, i) => [BX(t), BY(o.env_hi[m][i])]), dn = Ts.map((t, i) => [BX(t), BY(o.env_lo[m][i])]).reverse(); return `<path d="${P(up)}L${P(dn).slice(1)}Z" fill="${col}" fill-opacity="${op}" stroke="none"/>`; };
        s += bnd(cur, 0.13);
        if (on("lossy")) s += bnd(lcur, 0.07);
        if (on("target") && tcur) s += bnd(tcur, 0.07);
      }
    }
    const extra = (o, rg, shf, da, dar) => {
      const sh = Ts.map(shf), tt = Ts.map((t, i) => Math.hypot(o.env.both[i], sh[i]));
      for (const [arr, col, dd, w] of [[o.env.angle, "#2ca02c", da, 1.4], [o.env.spectral, "#e377c2", da, 1.4], [o.env.both, "#1f77b4", da, 1.4], [rg.env.both, "#1f77b4", dar, 1.4], [sh, "#737373", dar, 1.0], [tt, "#000", da, 2.0]])
        s += `<path d="${P(Ts.map((t, i) => [BX(t), BY(arr[i])]))}" fill="none" stroke="${col}" stroke-width="${w}" stroke-dasharray="${dd}"/>`;
    };
    if (on("lossy")) extra(lcur, lreg, lshotAt, "5 2.5", "1.2 2.2");
    if (on("target") && tcur) extra(tcur, treg, tshotAt, "7 2 1.5 2", "1.2 4.2");
    for (const [k, arr, col, w, da] of BC) if (on(k)) s += '''
rep(old,new)
rep('      s += `<path d="M${BX(T)} ${BY(tv)}L${ax + 14} ${ay - 11}" stroke="#000" stroke-width="0.6"/>` + txt(ax, ay, lab, { anchor: "middle" });',
'''      s += `<path d="M${BX(T)} ${BY(tv)}L${ax + 14} ${ay - 11}" stroke="#000" stroke-width="0.6"/>` + txt(ax, ay, lab, { anchor: "middle" });
      const lbl = (o, shf, gl, nm, dy) => { const v = Math.hypot(interpLog(o.env.both, T), shf(T)), p = v / (gl * aNH3) * 1e3, l = `${p >= 100 ? Math.round(p) : +p.toPrecision(2)} ppb (${nm})`;
        return `<circle cx="${BX(T)}" cy="${BY(v)}" r="2.6" fill="none" stroke="#000" stroke-width="1"/>` + txt(Math.max(BX(T) - 36.8, bl + 40), Math.min(BY(v) - dy, bb - 4), l, { anchor: "middle" }); };
      if (on("lossy")) s += lbl(lcur, lshotAt, lGL, "min. spec", 8);
      if (on("target") && tcur) s += lbl(tcur, tshotAt, tGL, "target spec", 20);''')
rep('mirror_R: c.R_mirror,','mirror_R: c.R_mirror, other_sets: D.chaos, note: "right ppb axis uses the ideal cell Gamma<L>; each other set is labelled with its own Gamma<L>",')
rep('</style>','table.vals { border-collapse: collapse; font: 12px var(--f-data); font-variant-numeric: tabular-nums; width: 100%; }\ntable.vals th, table.vals td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }\ntable.vals th:first-child, table.vals td:first-child { text-align: left; }\n.tbl { overflow-x: auto; }\n</style>')
# analysis button + panel
rep('<label class="check"><input type="checkbox" id="bMark" checked> Mark T and the total at T</label>',
    '<label class="check"><input type="checkbox" id="bMark" checked> Mark T and the total at T</label>\n        <div class="btns"><button type="button" id="anaBtn">Analyse all three cases (1 s)</button></div>')
rep('<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>','<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>\n      <div class="panel" id="anaPanel" hidden><h2>Ensemble analysis at T = 1 s (NH₃ line, 21 layouts)</h2><div class="tbl"><table class="vals" id="anaTbl"></table></div><p class="hint" id="anaNote"></p></div>')
rep('  const cache = new Map();','''  $("anaBtn").onclick = () => {
    const S = D.stats, names = { ideal: "ideal (R 99.5 %, 0 dB/cm)", target: "target (R 99 %, 0.1 dB/cm)", lossy: "minimum (R 95 %, 0.3 dB/cm)" }, ks = ["ideal", "target", "lossy"].filter(k => S[k]);
    const f = (v, n = 2) => Number.isFinite(v) ? (v >= 1e4 || v < 0.01 ? v.toExponential(n - 1) : +v.toPrecision(n)) : "–";
    const rows = [["Γ⟨L⟩ (cm), chaotic", k => f(S[k].GL_cm)], ["transmission", k => f(S[k].T)], ["speckle grains M²", k => f(S[k].M2, 3)],
      ["NH₃ limit at 1 s, median (ppb)", k => f(S[k].ppb_med)], ["layouts, quartiles (ppb)", k => `${f(S[k].ppb_q1)} to ${f(S[k].ppb_q3)}`], ["layouts, min to max (ppb)", k => `${f(S[k].ppb_min)} to ${f(S[k].ppb_max)}`],
      ["regular cell, both dithers (ppb)", k => f(S[k].reg_ppb)], ["chaotic advantage (×)", k => f(S[k].adv)], ["gain none → both (×)", k => f(S[k].dither_gain, 3)],
      ["fringe noise, both (NEA)", k => f(S[k].fringe_med)], ["shot noise (NEA)", k => f(S[k].shot_med)], ["fringe / shot", k => f(S[k].fringe_med / S[k].shot_med)],
      ["3.3 µm, scaled by 150–230× (ppb)", k => `${f(S[k].ppb_med / 230)} to ${f(S[k].ppb_med / 150)}`], ["3.3 µm, worst layout (ppb)", k => `${f(S[k].ppb_max / 230)} to ${f(S[k].ppb_max / 150)}`]];
    $("anaTbl").innerHTML = `<thead><tr><th>quantity</th>${ks.map(k => `<th>${names[k]}</th>`).join("")}</tr></thead><tbody>${rows.map(([n, g]) => `<tr><td>${n}</td>${ks.map(k => `<td>${g(k)}</td>`).join("")}</tr>`).join("")}</tbody>`;
    $("anaNote").textContent = "Medians over 21 inclusion layouts; the regular cell is the same cell and launch without inclusions. ppb = noise-equivalent absorbance / (Γ⟨L⟩ α_NH₃) at T = 1 s with 1 mW input; 3.3 µm values assume the crude 150–230× line-strength scaling at equal path and noise.";
    $("anaPanel").hidden = false;
  };
  const cache = new Map();''')
open(OUT,'w').write(s)
print('ok',list(sets))
