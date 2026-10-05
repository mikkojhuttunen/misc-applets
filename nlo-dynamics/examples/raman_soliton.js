// Raman soliton self-frequency shift and self-steepening with the generalised NLSE.
// Normalised units: beta2 = -1, gamma = 1, T0 = 1, fundamental soliton P0 = 1.
// Run: node examples/raman_soliton.js
const NLO = require('../nlo-engine.js');
const N = 4096, T = 40, TR = 0.01;
const nlse = new NLO.NLSE({
  N, T, linear: { beta: [-1] },
  nonlinear: NLO.nl.gnlse({ N, T, gamma: 1, tShock: 0.02, raman: { TR } })   // tShock = 1/omega0, TR = Raman time
});
const A = NLO.analysis.sech(nlse.grid, 1, 1);
console.log('z     spectral centroid   Gordon prediction -8 TR z/15');
nlse.propagate(A, { dz: 0.02, steps: 750, every: 150, onSample: (z, re, im) =>
  console.log(z.toFixed(1).padStart(4), NLO.analysis.spectralCentroid(re, im, nlse.grid).toFixed(4).padStart(14), (-8 * TR * z / 15).toFixed(4).padStart(16)) });
