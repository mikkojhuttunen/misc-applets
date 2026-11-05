/* Poincare coverage of the launch beam (QUIET v32, tasks 6 and 8): Birkhoff section (s / perimeter, sin chi) of 5 rays
   launched through the input port within +-2 mrad of the launch angle, 2000 wall hits each, ports closed (the
   phase-space region the injected light explores), binned on a 40 x 20 grid: coverage = share of occupied bins,
   H = normalised Shannon entropy of the hit histogram (1: uniform). A regular star orbit visits a few bins only.
   node scripts/chaos_metrics.js --cell circle --n_eff 1.5391 --wavelength 1.5317e-6 --seeds 30 --out x.json
     [--n_inc 16 --inc_dn 0.01 --inc_radius 50e-6] */
const fs = require('fs');
const { config, args } = require('./cells.js');
const { loadEngine } = require('./engine_path.js');
const { DitherLab: D, GasEngine: G, PlanarEngine: Pl } = loadEngine();
const a = args(process.argv.slice(2));
const nRay = 5, nHits = 2000, BS = 40, BP = 20;
const base = config(a.cell || 'circle', { n_eff: +(a.n_eff || 1.5391), wavelength: +(a.wavelength || 1.5317e-6), R: 0.97, dBcm: 0,
  n_inc: a.n_inc !== undefined ? +a.n_inc : 16, inc_dn: a.inc_dn !== undefined ? +a.inc_dn : 0.01, inc_radius: a.inc_radius !== undefined ? +a.inc_radius : 50e-6 });
function metrics(o) {
  const S = D.setup(o), c0 = S.mc.cell, h = new Float64Array(BS * BP);
  for (let r = 0; r < nRay; r++) {
    const th = S.theta + 2e-3 * (2 * r / (nRay - 1) - 1), m = G.marchRay(S.mc, Pl.launch(c0, c0.sIn, th), nHits);
    for (let i = 0; i < m.nWalls; i++) {
      const s = ((m.WS[i] / c0.perimeter % 1) + 1) % 1, c = Math.min(Math.max((m.WC[i] + 1) / 2, 0), 1 - 1e-12);
      h[Math.floor(s * BS) * BP + Math.floor(c * BP)]++;
    }
  }
  let n = 0, H = 0, occ = 0;
  for (const v of h) n += v;
  for (const v of h) if (v > 0) { occ++; const q = v / n; H -= q * Math.log(q); }
  return { coverage: occ / h.length, entropy: H / Math.log(h.length), n_hits: n };
}
const out = { cell: a.cell || 'circle', settings: base, regular: metrics(D.regular(Object.assign({}, base, { inc_seed: 1 }))), chaotic: [] };
for (let s = 1; s <= +(a.seeds || 30); s++) out.chaotic.push(Object.assign({ seed: s }, metrics(Object.assign({}, base, { inc_seed: s }))));
fs.writeFileSync(a.out, JSON.stringify(out));
console.log('done', a.out);
