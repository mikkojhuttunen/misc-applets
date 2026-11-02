/* Fig. 3 ensembles (fringe residual vs averaging time, NH3 noise curves), as scripts/original/ens.js.
   node scripts/ens.js R dB_per_cm nSeeds out.json [--engine quiet|fig2|lab] [--fig3 results/baseline/fig3data.json] */
const fs = require('fs'), path = require('path');
const { loadEngine, takeFlag, RESULTS } = require('./engine_path.js');
const argv = process.argv.slice(2);
const fig3 = takeFlag(argv, 'fig3', path.join(RESULTS, 'baseline', 'fig3data.json'));
const { DitherLab: D } = loadEngine(argv);
const data = JSON.parse(fs.readFileSync(fig3));
const Tgrid = data.T, eps = 1e-6, drift = 0.1;
const R = +argv[0], dBcm = +argv[1], nSeed = +argv[2], out = argv[3];
const alpha = dBcm / 4.3429448190325175 / 0.01;
const base = Object.assign({}, data.chaos.circle, { cell: 'circle', n_inc: 16, c_theta: 7.5 * Math.PI / 180, R_mirror: R, alpha_m: alpha });
const MODES = ['none', 'angle', 'spectral', 'both'];
function summarise(o, label) {
  const r = D.analyse(o, { full: false, keepSpec: true });
  const sp = r.spec, s2 = 2 * Math.sqrt(eps) / Math.sqrt(r.M2);
  const env = {}, floor = {};
  for (const m of MODES) {
    const spec = m === 'none' ? { F: [0], P: [1], floor: 1 } : sp[m];
    env[m] = Tgrid.map(T => s2 * D.residual(spec, T, drift, true));
    floor[m] = s2 * spec.floor;
  }
  return { label, seed: o.inc_seed, T_trans: r.T, L_mean_m: r.L_mean, M2: r.M2, gamma: o.gamma, Ls_m: r.Ls, Ws: r.Ws, env, floor, n_pass: r.n_pass,
    V: { none: 1, angle: sp.angle.floor, spectral: sp.spectral.floor, both: sp.both.floor }, sampFloor: r.V_floor, sampFloorBoth: r.V_floor_both };
}
const res = { R, dBcm, alpha_m: alpha, settings: base, chaotic: [], regular: null };
res.regular = summarise(D.regular(Object.assign({}, base, { inc_seed: 1 })), 'regular');
console.log('regular done', res.regular.T_trans, res.regular.L_mean_m, res.regular.floor);
for (let s = 1; s <= nSeed; s++) {
  res.chaotic.push(summarise(Object.assign({}, base, { inc_seed: s }), 'seed' + s));
  if (s % 5 === 0) { console.log('seed', s); fs.writeFileSync(out, JSON.stringify(res)); }
}
fs.writeFileSync(out, JSON.stringify(res));
console.log('done');
