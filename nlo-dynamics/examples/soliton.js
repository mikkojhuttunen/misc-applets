// Higher-order soliton breathing: N=2 soliton compresses and recovers over z0 = pi/2 * LD.
// Run: node examples/soliton.js
const NLO = require('../nlo-engine.js');
const beta2 = -1, gamma = 1, T0 = 1;
const s = NLO.analysis.soliton(beta2, gamma, T0, 2);            // LD, z0, P0 for N = 2
const nlse = new NLO.NLSE({ N: 1024, T: 40, linear: { beta: [beta2] }, gamma: gamma });
const A = NLO.analysis.sech(nlse.grid, T0, Math.sqrt(s.P0));
const map = nlse.evolve(A, { dz: s.z0 / 400, steps: 400, every: 40 });
console.log('z/z0   peak|A|^2   FWHM');
map.z.forEach((z, i) => {
  let pk = 0; for (const v of map.I[i]) pk = Math.max(pk, v);
  const half = pk / 2, c = map.I[i].indexOf(pk); let l = c, r = c;
  while (map.I[i][l] > half && l > 0) l--; while (map.I[i][r] > half && r < 1023) r++;
  console.log((z / s.z0).toFixed(2).padStart(4), pk.toFixed(3).padStart(10), ((r - l) * nlse.grid.dt).toFixed(3).padStart(8));
});
