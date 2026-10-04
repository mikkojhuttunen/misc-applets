/**
 * Plain-Node verification for ray-tracing-engine.js. No test framework —
 * run with `node beam-propagation/test/ray-tracing-engine.test.js`.
 * Exits non-zero (via assert throwing) on any failure.
 */
'use strict';
const assert = require('assert');
const {
  gcd, polygonVertices, rotateAbout, findStableOrbit, materializeOrbitPath
} = require('../ray-tracing-engine.js');

const R = 200, cx = 0, cy = 0;
let checks = 0;

function checkOrbit(N, m, expect) {
  const orbit = findStableOrbit(N, m, R, cx, cy);
  assert.ok(orbit, `N=${N} m=${m}: expected an orbit, got null`);

  // Closure: rotating the entry point by period*Delta must return to itself
  // to near machine precision.
  const back = rotateAbout(orbit.entry, orbit.period * orbit.Delta, cx, cy);
  const closeErr = Math.hypot(back.x - orbit.entry.x, back.y - orbit.entry.y);
  assert.ok(closeErr < 1e-9, `N=${N} m=${m}: closure error ${closeErr} too large`);
  checks++;

  // Edge-stepping: bounce 1 should land on facet index (m mod N).
  const verts = polygonVertices(N, R, cx, cy);
  const p1 = rotateAbout(orbit.entry, orbit.Delta, cx, cy);
  function nearestEdge(pt) {
    let best = -1, bestD = Infinity;
    for (let i = 0; i < N; i++) {
      const a = verts[i], b = verts[(i + 1) % N];
      const ex = b.x - a.x, ey = b.y - a.y;
      const t = Math.max(0, Math.min(1, ((pt.x - a.x) * ex + (pt.y - a.y) * ey) / (ex * ex + ey * ey)));
      const d = Math.hypot(pt.x - (a.x + t * ex), pt.y - (a.y + t * ey));
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }
  assert.strictEqual(nearestEdge(p1), m % N, `N=${N} m=${m}: bounce 1 landed on wrong facet`);
  checks++;

  if (expect) {
    if (expect.period !== undefined) assert.strictEqual(orbit.period, expect.period, `N=${N} m=${m}: period`);
    if (expect.angleDegApprox !== undefined) {
      assert.ok(Math.abs(orbit.angleDeg - expect.angleDegApprox) < 0.05,
        `N=${N} m=${m}: angleDeg ${orbit.angleDeg} vs expected ~${expect.angleDegApprox}`);
    }
    checks++;
  }
  return orbit;
}

// Regular case, gcd(N,m)=1: full period = N.
checkOrbit(32, 5, { period: 32, angleDegApprox: 61.88 });
checkOrbit(7, 2, { period: 7 });
checkOrbit(60, 7, { period: 60 });

// Degenerate diameter case m = N/2: path is a straight back-and-forth,
// normal incidence (angleDeg = 0), period 2.
checkOrbit(32, 16, { period: 2, angleDegApprox: 0 });

// Sub-star case, gcd(N,m) = g > 1: period = N/g, visits every g-th facet.
checkOrbit(12, 4, { period: 3 });

// materializeOrbitPath should wrap around exactly every `period` points.
{
  const orbit = findStableOrbit(23, 11, R, cx, cy);
  const path = materializeOrbitPath(orbit, orbit.period * 2, cx, cy);
  const d = Math.hypot(path[0].x - path[orbit.period].x, path[0].y - path[orbit.period].y);
  assert.ok(d < 1e-9, `materializeOrbitPath did not repeat after one period (d=${d})`);
  const d2 = Math.hypot(path[0].x - path[2 * orbit.period].x, path[0].y - path[2 * orbit.period].y);
  assert.ok(d2 < 1e-9, `materializeOrbitPath did not repeat after two periods (d2=${d2})`);
  checks += 2;
}

// gcd sanity
assert.strictEqual(gcd(12, 4), 4);
assert.strictEqual(gcd(23, 11), 1);
checks += 2;

console.log(`ray-tracing-engine.test.js: ${checks} checks passed.`);
