/* QUIET ensembles: one cell (circle CMPC or Herriott) with n_eff, membrane loss and a constant R or an R(|sin chi|)
   table, the regular design and nSeeds inclusion layouts. Per layout: static fringe floors for no dither / angle /
   spectral / both (V = 1 without dither) and the fringe residual vs averaging time (as scripts/ens.js, on the Fig. 3
   time grid, drift 0.1 rad/s), transmission T, mean path L, M2 (speckle grains), mean bounces, the bounce-averaged R
   seen by the detected light and the share of its bounces above the critical angle.
   node scripts/ensemble.js --cell circle|herriott --seeds 30 --out results/step2/x.json [--n_eff 1.66] [--R 0.99]
     [--rtable results/step1/Rtable_circle_neff1.66.json] [--dBcm 0.1] [--m 11] [--wavelength 1.6537e-6] [--gamma 0.31]
     [--first 1] [--engine quiet] */
const fs = require('fs'), path = require('path');
const { config, args } = require('./cells.js');
const { loadEngine, RESULTS } = require('./engine_path.js');
const argv = process.argv.slice(2);
const { DitherLab: D, GasEngine: G } = loadEngine(argv);
const a = args(argv);
const cell = a.cell || 'circle', nSeed = +(a.seeds || 30), first = +(a.first || 1), out = a.out;
if (!out) throw new Error('--out is required');
const nEff = +(a.n_eff || 2), dBcm = a.dBcm !== undefined ? +a.dBcm : 0.1, m = a.m !== undefined ? +a.m : null;
const wavelength = a.wavelength ? +a.wavelength : undefined, gamma = a.gamma ? +a.gamma : 0.2;
const tab = a.rtable ? JSON.parse(fs.readFileSync(a.rtable)) : null;
const R_table = tab ? { s: tab.s, R: tab.R } : null;
const fig3 = JSON.parse(fs.readFileSync(path.join(RESULTS, 'baseline', 'fig3data.json')));
const Tgrid = fig3.T, eps = 1e-6, drift = 0.1, MODES = ['none', 'angle', 'spectral', 'both'];

/* bounce statistics of the detected beam rays with the table (or constant R) */
function bounceStats(o, theta) {
  const S = D.setup(o), c = S.mc.cell, pw = c.meta.portW, nB = Math.min(6000, Math.ceil(Math.log(1e-4) / Math.log(Math.min(o.R_mirror, 0.99999))));
  G.setAlpha(o.alpha_m); G.setRTable(o.R_table || null);
  let wR = 0, w = 0, wAbove = 0, wb = 0, wr = 0;
  for (const st of G.beamRays(c, c.sIn, theta, S.q0, S.lamM, 12)) {
    const r = G.marchRay(S.mc, st, nB, c.sIn, c.sOut, pw, 'both');
    if (r.exit !== 1) continue;
    const n = r.nWalls, wt = Math.exp(G.lnBounce(r, n, o.R_mirror) - o.alpha_m * r.WG[n - 1]);
    for (let i = 0; i < n - 1; i++) {
      const s = Math.abs(r.WC[i]), R = o.R_table ? Math.exp(G.lnBounce({ WC: [s] }, 2, 1)) : o.R_mirror;
      wR += wt * R; w += wt; if (s > 1 / o.n_eff) wAbove += wt;
    }
    wb += wt * (n - 1); wr += wt;
  }
  return { R_bounce: w ? wR / w : NaN, above_critical: w ? wAbove / w : NaN, mean_bounces: wr ? wb / wr : NaN };
}

function summarise(o, label) {
  const r = D.analyse(o, { full: false, keepSpec: true });
  const sp = r.spec, s2 = 2 * Math.sqrt(eps) / Math.sqrt(r.M2), env = {}, floor = {};
  for (const md of MODES) {
    const spec = md === 'none' ? { F: [0], P: [1], floor: 1 } : sp[md];
    env[md] = Tgrid.map(T => s2 * D.residual(spec, T, drift, true));
    floor[md] = s2 * spec.floor;
  }
  const bs = bounceStats(o, r.theta);
  return Object.assign({ label, seed: o.inc_seed, T_trans: r.T, L_mean_m: r.L_mean, M2: r.M2, grains: r.grains, gamma: o.gamma, env, floor, n_pass: r.n_pass,
    V_theta: r.V_theta, V_lambda: r.V_lambda, V_joint: r.V_joint, V_floor: r.V_floor, V_floor_both: r.V_floor_both, theta: r.theta }, bs);
}

const base = config(cell, { n_eff: nEff, R: tab ? Math.min(tab.design.R_avg, 0.99999) : +(a.R || 0.99), R_table, dBcm, m, wavelength, gamma });
const res = { cell, n_eff: nEff, dBcm, m, gamma, rtable: a.rtable || null, R_const: tab ? null : base.R_mirror, design: tab ? tab.design : null,
  settings: Object.assign({}, base, { R_table: R_table ? '(table)' : null }), regular: null, chaotic: [] };
const t0 = Date.now();
const save = () => fs.writeFileSync(out, JSON.stringify(res));
fs.mkdirSync(path.dirname(out), { recursive: true });
if (first <= 1) {
  try { res.regular = summarise(D.regular(Object.assign({}, base, { inc_seed: 1 })), 'regular'); } catch (e) { res.regular = { error: String(e.message) }; }
  console.log('regular', res.regular.T_trans, res.regular.L_mean_m, res.regular.floor && res.regular.floor.both, ((Date.now() - t0) / 1000).toFixed(1) + 's');
}
for (let s = first; s < first + nSeed; s++) {
  try { res.chaotic.push(summarise(Object.assign({}, base, { inc_seed: s }), 'seed' + s)); } catch (e) { res.chaotic.push({ seed: s, error: String(e.message) }); }
  if (s % 5 === 0) { console.log('seed', s, ((Date.now() - t0) / 1000).toFixed(0) + 's'); save(); }
}
save(); console.log('done', ((Date.now() - t0) / 1000).toFixed(0) + 's');
