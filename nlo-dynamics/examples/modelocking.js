// Self-starting mode-locking from noise (Haus master equation), KLM-like vs SESAM.
// Run: node examples/modelocking.js
const NLO = require('../nlo-engine.js');
for (const [label, tauA, mode] of [['KLM-like (instant SA)', 0, 'legacy'], ['SESAM, 5-bin recovery, fast-time', 5, 'fastTime']]) {
  const h = new NLO.HausModelocking({ seed: 1, params: { tauA: tauA }, absorberMode: mode });
  const res = h.run({ snapshotEvery: 1000 });
  const m = NLO.analysis.pulseMetrics(h.re, h.im, 1);
  console.log(label.padEnd(34), 'converged:', res.converged, 'after', res.rounds, 'round trips; FWHM', m.fwhm.toFixed(1), 'bins; peak', m.peak.toFixed(1));
}
