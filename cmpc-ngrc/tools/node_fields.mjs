// Batch detector fields with the JS engine (web/ngrc-engine.js) on a pool of worker threads.
//   node tools/node_fields.mjs job.json out.bin
// job.json: { cell, opt: {mode ('curved' | 'straight' | 'none' | 'phase'), nPos, nAng, frozen, waist, ds, prune,
//            ampMin, maxBounces}, samples: [[shape, ...] | {shapes: [...], dn: uniform index change}, ...], workers? }
// out.bin: Float64 little-endian, per sample, per (input, output) key in numeric order, n_pix × (re, im).
// out.bin.json: { keys: [[in, out], ...], n_pix, n_samples, ms }
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { availableParallelism } from 'node:os';
const require = createRequire(import.meta.url);
const N = require('../web/ngrc-engine.js');

function keysOf(cell) {
  const ins = cell.ports.map((p, i) => N.isIn(p) ? i : -1).filter(i => i >= 0);
  const outs = cell.ports.map((p, i) => N.isOut(p) ? i : -1).filter(i => i >= 0);
  return ins.flatMap(i => outs.map(o => [i, o]));
}

function run(job, lo, hi) {
  const { cell, opt } = job, keys = keysOf(cell), np = cell.detector.n_pix;
  const out = new Float64Array((hi - lo) * keys.length * np * 2);
  const ps = opt.mode === 'phase' ? new N.PhaseScreen(cell, opt) : null;
  for (let s = lo; s < hi; s++) {
    const smp = job.samples[s], list = Array.isArray(smp) ? smp : smp.shapes, dn = Array.isArray(smp) ? 0 : (smp.dn || 0);
    const shapes = list.map(N.makeShape);
    if (dn && !ps) throw new Error('a uniform index change needs the phase-screen mode');
    const f = ps ? ps.fields(shapes, dn) : N.curvedFields(cell, shapes, opt);
    let o = (s - lo) * keys.length * np * 2;
    for (const [i, k] of keys) {
      const e = f[i + ',' + k];
      for (let p = 0; p < np; p++) { out[o++] = e.re[p]; out[o++] = e.im[p]; }
    }
  }
  return out;
}

if (isMainThread) {
  const [jobPath, outPath] = process.argv.slice(2);
  const job = JSON.parse(readFileSync(jobPath, 'utf8'));
  const t0 = Date.now(), n = job.samples.length;
  const nw = Math.max(1, Math.min(job.workers || availableParallelism(), n));
  const chunks = [...Array(nw).keys()].map(w => [Math.floor(w * n / nw), Math.floor((w + 1) * n / nw)]);
  const parts = await Promise.all(chunks.map(([lo, hi]) => new Promise((res, rej) => {
    const w = new Worker(new URL(import.meta.url), { workerData: { jobPath, lo, hi } });
    w.once('message', res); w.once('error', rej);
  })));
  const total = parts.reduce((a, p) => a + p.length, 0), all = new Float64Array(total);
  let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
  writeFileSync(outPath, Buffer.from(all.buffer));
  writeFileSync(outPath + '.json', JSON.stringify({ keys: keysOf(job.cell), n_pix: job.cell.detector.n_pix, n_samples: n, ms: Date.now() - t0 }));
} else {
  const job = JSON.parse(readFileSync(workerData.jobPath, 'utf8'));
  parentPort.postMessage(run(job, workerData.lo, workerData.hi));
}
