/**
 * ray-tracing-engine.js
 *
 * Geometric-optics billiard engine for a "circular segmented multipass cell"
 * (CMPC): a regular N-gon of flat mirror facets approximating a circular
 * reflector. Used by the fys501-laser-physics multipass-cell-ray-tracer and
 * cmpc-gaussian-beam applets (in mikkojhuttunen/physics-applets).
 *
 * Two complementary ways to trace a path:
 *
 *   1. simulateBilliard(...)  — general billiard: fire a ray at an arbitrary
 *      entry angle and follow true polygon-edge collisions. For most angles
 *      this does NOT close on itself; it slowly fills the ring.
 *
 *   2. findStableOrbit(...)   — exact closed-form solver: for a chosen integer
 *      "winding step" m (how many facets the beam skips each bounce), solves
 *      for the one entry point on facet 0 whose path is invariant under
 *      rotation by Delta = 2*pi*m/N. That symmetry means the reflection law
 *      holds at every bounce if it holds once, so the whole path is just
 *      rotated copies of a single solved point — exact to machine precision,
 *      and periodic with period N/gcd(N,m).
 *
 * Conventions:
 *   - All coordinates are plain {x, y} objects in a shared 2D plane (units
 *     are the caller's choice — pixels, mm, whatever `R` is given in).
 *   - Angles passed in as "Deg" are degrees; everything computed internally
 *     is radians.
 *   - The polygon is built with polygonVertices(): vertex 0 sits at the top
 *     (angle -pi/2 from center), vertices proceed counter-clockwise by
 *     increasing index. Facet k runs from vertex k to vertex k+1.
 *   - "Angle of incidence" (returned as angleDeg) is measured from the local
 *     facet normal, standard optics convention (0 = normal incidence).
 *
 * Verified (see test/ray-tracing-engine.test.js): findStableOrbit closes to
 * ~1e-13 absolute position error after exactly `period` bounces, across a
 * range of N and m including the degenerate m = N/2 "diameter" case.
 *
 * No dependencies. Works as a CommonJS module (Node) or a plain browser
 * global (RayTracingEngine) via a <script> tag.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = mod;
  } else {
    root.RayTracingEngine = mod;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /** Greatest common divisor (non-negative). */
  function gcd(a, b) {
    a = Math.abs(a); b = Math.abs(b);
    while (b) { const t = b; b = a % b; a = t; }
    return a;
  }

  /**
   * Vertices of a regular N-gon of circumradius R centered at (cx, cy).
   * Vertex 0 is at the top; winds counter-clockwise.
   * @returns {{x:number,y:number}[]} length-N array
   */
  function polygonVertices(N, R, cx, cy) {
    const rot = -Math.PI / 2;
    const v = [];
    for (let i = 0; i < N; i++) {
      const a = rot + (i * 2 * Math.PI) / N;
      v.push({ x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
    }
    return v;
  }

  /**
   * Rotate point p by angle `ang` (radians) about (cx, cy).
   */
  function rotateAbout(p, ang, cx, cy) {
    const s = Math.sin(ang), c = Math.cos(ang);
    const dx = p.x - cx, dy = p.y - cy;
    return { x: cx + dx * c - dy * s, y: cy + dx * s + dy * c };
  }

  /**
   * Unit inward normal and unit tangent of the segment v0->v1, where "inward"
   * means pointing toward (cx, cy).
   * @returns {{nx:number,ny:number,tx:number,ty:number}}
   */
  function edgeNormalTangent(v0, v1, cx, cy) {
    let ex = v1.x - v0.x, ey = v1.y - v0.y;
    const elen = Math.hypot(ex, ey); ex /= elen; ey /= elen;
    let nx = -ey, ny = ex;
    const midx = (v0.x + v1.x) / 2, midy = (v0.y + v1.y) / 2;
    if (nx * (cx - midx) + ny * (cy - midy) < 0) { nx = -nx; ny = -ny; }
    return { nx, ny, tx: ex, ty: ey };
  }

  /**
   * Intersection of ray (px,py)+t*(dx,dy), t>0, with segment (ax,ay)-(bx,by),
   * s in [0,1]. Returns null if parallel or outside range.
   * @returns {{t:number,s:number,x:number,y:number}|null}
   */
  function rayIntersectSegment(px, py, dx, dy, ax, ay, bx, by) {
    const ex = bx - ax, ey = by - ay;
    const det = ex * dy - ey * dx;
    if (Math.abs(det) < 1e-9) return null;
    const t = (ex * (ay - py) - ey * (ax - px)) / det;
    const s = (dx * (ay - py) - dy * (ax - px)) / det;
    if (t > 1e-6 && s > -1e-6 && s < 1 + 1e-6) {
      return { t, s, x: px + dx * t, y: py + dy * t };
    }
    return null;
  }

  /**
   * General polygon billiard: fire from the midpoint of facet 0 at
   * `entryAngleDeg` from the inward normal, follow up to `bounces` true
   * polygon-edge reflections. For generic angles this will NOT close on
   * itself (unlike findStableOrbit) — useful for showing the contrast, or
   * for exploring non-symmetric entry conditions.
   *
   * @param {number} N number of mirror facets
   * @param {number} bounces max reflections to simulate
   * @param {number} entryAngleDeg angle from the inward normal at facet 0, degrees
   * @param {number} R circumradius of the facet polygon
   * @param {number} cx,cy center of the polygon
   * @returns {{verts:Array, path:Array<{x:number,y:number}>, totalLen:number}}
   *   path[0] is the entry point; totalLen is the summed Euclidean path
   *   length in the same units as R.
   */
  function simulateBilliard(N, bounces, entryAngleDeg, R, cx, cy) {
    const verts = polygonVertices(N, R, cx, cy);
    const edges = verts.map((v, i) => [v, verts[(i + 1) % N]]);
    const [a0, b0] = edges[0];
    const mid = { x: (a0.x + b0.x) / 2, y: (a0.y + b0.y) / 2 };
    let ex = b0.x - a0.x, ey = b0.y - a0.y;
    const elen = Math.hypot(ex, ey); ex /= elen; ey /= elen;
    let nx = -ey, ny = ex;
    const toCx = cx - mid.x, toCy = cy - mid.y;
    if (nx * toCx + ny * toCy < 0) { nx = -nx; ny = -ny; }
    const th = (entryAngleDeg * Math.PI) / 180;
    let dx = nx * Math.cos(th) + ex * Math.sin(th);
    let dy = ny * Math.cos(th) + ey * Math.sin(th);
    const dl = Math.hypot(dx, dy); dx /= dl; dy /= dl;

    let p = { x: mid.x, y: mid.y };
    const path = [p];
    let lastEdge = 0;
    let totalLen = 0;
    for (let i = 0; i < bounces; i++) {
      let best = null, bestEdge = -1;
      for (let k = 0; k < edges.length; k++) {
        if (k === lastEdge) continue;
        const [a, b] = edges[k];
        const hit = rayIntersectSegment(p.x, p.y, dx, dy, a.x, a.y, b.x, b.y);
        if (hit && (!best || hit.t < best.t)) { best = hit; bestEdge = k; }
      }
      if (!best) break;
      const [a, b] = edges[bestEdge];
      let eex = b.x - a.x, eey = b.y - a.y;
      const el2 = Math.hypot(eex, eey); eex /= el2; eey /= el2;
      const nnx = -eey, nny = eex;
      const dot = dx * nnx + dy * nny;
      const rdx = dx - 2 * dot * nnx, rdy = dy - 2 * dot * nny;
      totalLen += Math.hypot(best.x - p.x, best.y - p.y);
      p = { x: best.x, y: best.y };
      path.push(p);
      dx = rdx; dy = rdy;
      lastEdge = bestEdge;
    }
    return { verts, path, totalLen };
  }

  /**
   * Exact periodic ("rotationally symmetric") billiard orbit: the entry point
   * on facet 0 whose path is invariant under rotation by Delta = 2*pi*m/N,
   * found by a bisection root-find on the single free parameter (position
   * along facet 0). See file header for why this closes to machine precision.
   *
   * @param {number} N number of mirror facets
   * @param {number} m winding step: bounce advances m facets each time
   *   (1 <= m <= floor(N/2) is the useful range; m and N-m give the same
   *   geometric orbit traversed in the opposite direction)
   * @param {number} R circumradius of the facet polygon
   * @param {number} cx,cy center of the polygon
   * @returns {{verts:Array, entry:{x:number,y:number}, Delta:number,
   *   period:number, groupG:number, angleDeg:number, m:number}|null}
   *   `period` = N/gcd(N,m) is the number of bounces until the path closes;
   *   `groupG` = gcd(N,m) > 1 means the orbit only visits every groupG-th
   *   facet (a smaller "sub-star"); `angleDeg` is the solved angle of
   *   incidence from the facet normal. Returns null if no valid root is found
   *   (shouldn't happen for 1 <= m <= floor(N/2), N >= 3).
   */
  function findStableOrbit(N, m, R, cx, cy) {
    m = ((m % N) + N) % N;
    if (m === 0) return null;
    const verts = polygonVertices(N, R, cx, cy);
    const v0 = verts[0], v1 = verts[1];
    const { nx, ny } = edgeNormalTangent(v0, v1, cx, cy);
    const Delta = (2 * Math.PI * m) / N;

    function P(t) { return { x: v0.x + t * (v1.x - v0.x), y: v0.y + t * (v1.y - v0.y) }; }
    function residual(t) {
      const p0 = P(t);
      const pPrev = rotateAbout(p0, -Delta, cx, cy);
      const pNext = rotateAbout(p0, Delta, cx, cy);
      const vinx = p0.x - pPrev.x, viny = p0.y - pPrev.y;
      const voutx = pNext.x - p0.x, vouty = pNext.y - p0.y;
      const dotN = vinx * nx + viny * ny;
      const rx = vinx - 2 * dotN * nx, ry = viny - 2 * dotN * ny; // vin reflected off facet 0
      return {
        cross: rx * vouty - ry * voutx,
        dot: rx * voutx + ry * vouty,
        p0, voutx, vouty
      };
    }

    const steps = 1500;
    let prevT = 0.001, prevRes = residual(prevT);
    let found = null;
    for (let i = 1; i <= steps && !found; i++) {
      const t = 0.001 + (0.998 * i) / steps;
      const res = residual(t);
      if ((prevRes.cross > 0) !== (res.cross > 0)) {
        let lo = prevT, hi = t, rLo = prevRes;
        for (let b = 0; b < 60; b++) {
          const mid = (lo + hi) / 2, rm = residual(mid);
          if ((rLo.cross > 0) !== (rm.cross > 0)) hi = mid; else { lo = mid; rLo = rm; }
        }
        const tRoot = (lo + hi) / 2;
        const rRoot = residual(tRoot);
        if (rRoot.dot > 1e-7) found = { t: tRoot, ...rRoot };
      }
      prevT = t; prevRes = res;
    }
    if (!found) return null;

    const g = gcd(N, m);
    const period = N / g;
    const dirLen = Math.hypot(found.voutx, found.vouty) || 1;
    const dirx = found.voutx / dirLen, diry = found.vouty / dirLen;
    const cosA = Math.max(-1, Math.min(1, dirx * nx + diry * ny));
    const angleDeg = (Math.acos(cosA) * 180) / Math.PI;

    return { verts, entry: found.p0, Delta, period, groupG: g, angleDeg, m };
  }

  /**
   * Convenience wrapper: given a target angle of incidence, search all valid
   * integer winding steps m (1..floor(N/2)) and return the findStableOrbit()
   * result whose solved angleDeg is closest to targetDeg. Useful when the
   * user wants to dial in "angle of incidence" directly rather than an
   * integer winding step (the two are related but only certain discrete
   * angles give an exactly-closed orbit for a given N).
   *
   * @returns {object|null} same shape as findStableOrbit(), or null if N < 2
   */
  function pickWindingStep(N, targetDeg, R, cx, cy) {
    const maxM = Math.max(1, Math.floor(N / 2));
    let best = null, bestDiff = Infinity;
    for (let m = 1; m <= maxM; m++) {
      const orbit = findStableOrbit(N, m, R, cx, cy);
      if (!orbit) continue;
      const diff = Math.abs(orbit.angleDeg - targetDeg);
      if (diff < bestDiff) { bestDiff = diff; best = orbit; }
    }
    return best;
  }

  /**
   * Materialize a full point sequence from a findStableOrbit() result by
   * rotating the solved entry point k*Delta for k = 0..count. Because
   * count*Delta wraps around exactly every `period` steps, requesting more
   * points than `period` simply retraces the same closed path — which is
   * itself the physically interesting fact (extra passes add no new
   * coverage once the star has closed).
   *
   * @param {object} orbit a findStableOrbit() result
   * @param {number} count number of bounces to generate (path has count+1 points)
   * @param {number} cx,cy center of rotation (must match what findStableOrbit used)
   * @returns {Array<{x:number,y:number}>}
   */
  function materializeOrbitPath(orbit, count, cx, cy) {
    const path = [];
    for (let k = 0; k <= count; k++) path.push(rotateAbout(orbit.entry, k * orbit.Delta, cx, cy));
    return path;
  }

  return {
    gcd,
    polygonVertices,
    rotateAbout,
    edgeNormalTangent,
    rayIntersectSegment,
    simulateBilliard,
    findStableOrbit,
    pickWindingStep,
    materializeOrbitPath
  };
});
