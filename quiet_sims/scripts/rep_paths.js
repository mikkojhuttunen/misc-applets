/* Exit-path lists (Ls, Ws) of the representative layout of an ensemble (the layout whose mean path is closest to the
   median) and of its regular cell, for the noise traces of the Fig. 3 artifact (panel a).
   node scripts/rep_paths.js results/step10/circle_target.json out.json */
const fs = require('fs');
const { loadEngine } = require('./engine_path.js');
const { DitherLab: D } = loadEngine([]);
const [inp, out] = process.argv.slice(2);
const d = JSON.parse(fs.readFileSync(inp));
const tab = d.rtable ? JSON.parse(fs.readFileSync(d.rtable)) : null;
const base = Object.assign({}, d.settings, { R_table: tab ? { s: tab.s, R: tab.R } : null });
const ch = d.chaotic.filter(x => !x.error), Ls = ch.map(x => x.L_mean_m).sort((a, b) => a - b);
const med = Ls.length % 2 ? Ls[Ls.length >> 1] : 0.5 * (Ls[Ls.length / 2 - 1] + Ls[Ls.length / 2]);
const rep = ch.reduce((a, x) => (Math.abs(x.L_mean_m - med) < Math.abs(a.L_mean_m - med) ? x : a));
const run = o => { const r = D.analyse(o, { full: false }); return { Ls_m: Array.from(r.Ls), Ws: Array.from(r.Ws), L_mean_m: r.L_mean, T_trans: r.T }; };
const res = { seed: rep.seed, chaotic: run(Object.assign({}, base, { inc_seed: rep.seed })), regular: run(D.regular(Object.assign({}, base, { inc_seed: 1 }))) };
fs.writeFileSync(out, JSON.stringify(res));
console.log(inp, 'seed', rep.seed, res.chaotic.Ls_m.length, res.chaotic.L_mean_m, rep.L_mean_m);
