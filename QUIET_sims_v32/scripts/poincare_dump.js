/* Fig. 2e data (QUIET v32): Birkhoff sections (s / perimeter, sin chi) of R-/C-CMPC and R-/C-Herriott at the set-B near-IR
   index (n_eff 1.5391, 1531.7 nm), inclusion layout seed 1, grid of 16 x 10 launches, 700 wall hits each (the "normal"
   sampling of the Fig. 2 artifact). node scripts/poincare_dump.js results/fig2e_poincare.json */
const fs = require('fs');
const { config } = require('./cells.js');
const { loadEngine } = require('./engine_path.js');
const { DitherLab: D } = loadEngine();
const out = {};
for (const cell of ['circle', 'herriott']) {
  const base = config(cell, { n_eff: 1.5391, wavelength: 1.5317e-6, R: 0.97, dBcm: 0.3, gamma: 0.4532 });
  for (const kind of ['regular', 'chaotic']) {
    const o = kind === 'regular' ? D.regular(Object.assign({}, base, { inc_seed: 1 })) : Object.assign({}, base, { inc_seed: 1 });
    const S = D.setup(o), p = D.poincare(S.mc, { nS: 16, nP: 10, nHits: 700 });
    out[`${cell}_${kind}`] = { S: Array.from(p.S, v => +v.toFixed(5)), C: Array.from(p.C, v => +v.toFixed(5)), pmax: p.pmax };
  }
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
console.log('ok', Object.keys(out).map(k => `${k}:${out[k].S.length}`).join(' '));
