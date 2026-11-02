/* Distribution of |sin chi| (chi from the wall normal) over the bounces of the light that reaches the output port:
   n_side^2 beam rays per layout, every wall hit before the exit, weighted by the ray's detected power (mirror R and
   membrane loss as configured; default R = 1, so the weights are the membrane loss only). Also the share of bounces
   above the critical angle for a list of n_eff.
   node scripts/angle_hist.js --cell circle|herriott [--m 11] [--n_eff 1.66] [--seeds 8] [--regular 0|1] [--R 1]
     [--dBcm 0.1] [--bins 400] [--out results/step1/anglehist_circle_m11.json] */
const fs = require('fs'), path = require('path');
const { config, args } = require('./cells.js');
const { loadEngine, RESULTS } = require('./engine_path.js');
const argv = process.argv.slice(2);
const { DitherLab: D, GasEngine: G } = loadEngine(argv);
const a = args(argv);
const cell = a.cell || 'circle', m = a.m !== undefined ? +a.m : null, nSeeds = +(a.seeds || 8), bins = +(a.bins || 400), R = +(a.R || 1);
const nEff = +(a.n_eff || 1.66), dBcm = a.dBcm !== undefined ? +a.dBcm : 0.1, regular = !!+(a.regular || 0), nSide = +(a.n_side || 12);
const out = a.out || path.join(RESULTS, 'step1', `anglehist_${cell}${m === null ? '' : '_m' + m}${regular ? '_regular' : ''}.json`);
const hist = new Float64Array(bins), crit = [1.36, 1.46, 1.53, 1.59, 1.66, 1.71, 1.74, 2.0];
let wsum = 0, nRays = 0, nDet = 0, hitsW = 0, pathW = 0, wRay = 0;
for (let seed = 1; seed <= (regular ? 1 : nSeeds); seed++) {
  let o = config(cell, { n_eff: nEff, R, dBcm, m });
  o = Object.assign(o, { inc_seed: seed });
  if (regular) o = D.regular(o);
  const S = D.setup(o), c = S.mc.cell, pw = c.meta.portW;
  G.setAlpha(o.alpha_m); G.setRTable(null);
  let theta = S.theta;
  if (cell !== 'circle' && o.retune) theta = G.retune(S.mc, theta, S.q0, S.lamM, { Rm: o.R_mirror, gamma: o.gamma, nEff: o.n_eff, nSide: 6 }).theta;
  const nB = Math.min(6000, Math.ceil(Math.log(1e-4) / Math.log(Math.min(R, 0.99999))));
  for (const st of G.beamRays(c, c.sIn, theta, S.q0, S.lamM, nSide)) {
    nRays++;
    const r = G.marchRay(S.mc, st, nB, c.sIn, c.sOut, pw, 'both');
    if (r.exit !== 1) continue;
    nDet++;
    const n = r.nWalls, w = (R < 1 ? R ** (n - 1) : 1) * Math.exp(-o.alpha_m * r.WG[n - 1]);
    for (let i = 0; i < n - 1; i++) { const s = Math.abs(r.WC[i]); hist[Math.min(bins - 1, Math.floor(s * bins))] += w; wsum += w; }
    hitsW += w * (n - 1); pathW += w * r.WG[n - 1]; wRay += w;
  }
}
const p = Array.from(hist, v => v / wsum), centres = Array.from(hist, (_, i) => (i + 0.5) / bins);
const above = {};
for (const ne of crit) { let s = 0; for (let i = 0; i < bins; i++) if (centres[i] > 1 / ne) s += p[i]; above[ne] = s; }
const res = { cell, m, n_eff: nEff, R, dBcm, regular, seeds: regular ? 1 : nSeeds, n_side: nSide, bins, s: centres, p, rays: nRays, detected: nDet,
  mean_bounces: hitsW / wRay, mean_path_m: pathW / wRay, share_above_critical: above };
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(res));
console.log(`${cell} m=${m} n_eff=${nEff}: ${nDet}/${nRays} rays detected; share of bounces above the critical angle:`,
  Object.entries(above).map(([k, v]) => `${k}: ${(100 * v).toFixed(1)} %`).join(', '));
