/**
 * Plain-Node verification for abcd-gaussian-engine.js. No test framework —
 * run with `node beam-propagation/test/abcd-gaussian-engine.test.js`.
 * Exits non-zero (via assert throwing) on any failure.
 */
'use strict';
const assert = require('assert');
const {
  obliqueMirrorFocalLengths, unitCell, stabilityG, cAdd, cMirror,
  widthFromQ, zRFromWaist, solveMatchedQ, propagateUnitCellSequence
} = require('../abcd-gaussian-engine.js');

let checks = 0;

// --- astigmatism: fT=fS at normal incidence, diverge with angle ---
{
  const n = obliqueMirrorFocalLengths(100, 0);
  assert.ok(Math.abs(n.fT - n.fS) < 1e-9, 'fT should equal fS at normal incidence');
  const o = obliqueMirrorFocalLengths(100, (62 * Math.PI) / 180);
  assert.ok(o.fT < o.fS, 'fT should be smaller than fS at oblique incidence');
  checks += 2;
}

// --- matched-mode self-consistency: solveMatchedQ must be an exact fixed
//     point of its own unit cell (within the elementary cMirror/cAdd ops) ---
{
  const L = 18.765, f = 9.428; // representative values from a real configuration
  const cell = unitCell(L, f);
  const qm = solveMatchedQ(cell.A, cell.B, cell.C, cell.D);
  assert.ok(qm, 'expected a matched mode for a stable unit cell');
  const after = cMirror(cAdd(qm, L), f);
  assert.ok(Math.abs(after.re - qm.re) < 1e-6, `matched q.re not self-consistent: ${after.re} vs ${qm.re}`);
  assert.ok(Math.abs(after.im - qm.im) < 1e-6, `matched q.im not self-consistent: ${after.im} vs ${qm.im}`);
  checks += 2;

  // width at the mirror plane should stay exactly constant bounce after bounce
  let q = { re: qm.re, im: qm.im };
  let w0 = null;
  for (let k = 0; k < 6; k++) {
    const qEnd = cAdd(q, L);
    const w = widthFromQ(qEnd, 633e-6);
    if (w0 === null) w0 = w; else assert.ok(Math.abs(w - w0) < 1e-9, `matched-mode width drifted: ${w} vs ${w0}`);
    q = cMirror(qEnd, f);
  }
  checks++;
}

// --- unstable cell: solveMatchedQ returns null, width diverges, no NaNs ---
{
  const L = 18.765, f = 1.178; // small f relative to L -> |g|>=1
  const cell = unitCell(L, f);
  assert.ok(Math.abs(stabilityG(cell.A, cell.D)) >= 1, 'expected this cell to be unstable');
  assert.strictEqual(solveMatchedQ(cell.A, cell.B, cell.C, cell.D), null);
  checks += 2;

  let q = { re: 0, im: zRFromWaist(0.05, 633e-6) };
  let prevW = widthFromQ(cAdd(q, L), 633e-6);
  for (let k = 1; k < 8; k++) {
    q = cMirror(cAdd(q, L), f);
    const w = widthFromQ(cAdd(q, L), 633e-6);
    assert.ok(Number.isFinite(w), `width became non-finite at bounce ${k}`);
    assert.ok(w > prevW, `expected monotonic growth in an unstable cell at bounce ${k}`);
    prevW = w;
  }
  checks++;
}

// --- propagateUnitCellSequence matches a hand-rolled equivalent loop ---
{
  const L = 18.765, f = 9.428, w0 = 0.10, lambdaMm = 633e-6, aperture = 3.921;
  const result = propagateUnitCellSequence(40, L, f, w0, lambdaMm, aperture);

  let q = { re: 0, im: zRFromWaist(w0, lambdaMm) };
  let sTotal = 0, gouyTotal = 0, clipIndex = -1;
  for (let k = 1; k <= 40; k++) {
    assert.ok(Math.abs(result.legs[k - 1].qStart.re - q.re) < 1e-12, `leg ${k} qStart.re mismatch`);
    assert.ok(Math.abs(result.legs[k - 1].qStart.im - q.im) < 1e-12, `leg ${k} qStart.im mismatch`);
    assert.ok(Math.abs(result.legs[k - 1].sStart - sTotal) < 1e-9, `leg ${k} sStart mismatch`);
    const qEnd = cAdd(q, L);
    const w = widthFromQ(qEnd, lambdaMm);
    if (clipIndex < 0 && 2 * w > aperture) clipIndex = k;
    sTotal += L;
    gouyTotal += Math.atan2(qEnd.re, qEnd.im) - Math.atan2(q.re, q.im);
    q = cMirror(qEnd, f);
    if (clipIndex > 0) break;
  }
  assert.strictEqual(result.clipIndex, clipIndex, 'clipIndex mismatch vs hand-rolled loop');
  checks += result.legs.length + 1;
}

console.log(`abcd-gaussian-engine.test.js: ${checks} checks passed.`);
