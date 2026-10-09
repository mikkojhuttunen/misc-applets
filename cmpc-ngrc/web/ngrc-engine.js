/* cmpc-ngrc JS engine: port of the Python reference (ngrc/shapes.py, rays.py, field.py, perturbative.py,
   harmonics.py, readout.py ridge). Per-ray scalar loops; the physics and step logic follow the Python code so the
   results agree with tools/vectors.json (checked by tools/check_js.mjs). SI units.
   Exposes NGRC = { makeShape, shapeEval, trace, detectorFields, PhaseScreen, chdShape, chdContour, randomShape,
   rng, ridge... } on globalThis and as module.exports. */
(function (root) {
  'use strict';
  const TAIL_U = 7.0, STEP_U = 4.5;
  const TAU = 2 * Math.PI;

  // ---------- shapes ----------
  function makeShape(o) {
    const M = Math.max((o.a || []).length, (o.b || []).length);
    const a = new Float64Array(M), b = new Float64Array(M);
    (o.a || []).forEach((v, i) => a[i] = v); (o.b || []).forEach((v, i) => b[i] = v);
    const s = { x0: o.x0, y0: o.y0, R: o.R, dn: o.dn, edge: o.edge ?? 3e-6, a, b, rotation: o.rotation || 0, M };
    let sa = 0; for (let i = 0; i < M; i++) sa += Math.abs(a[i]) + Math.abs(b[i]);
    s.bound = s.R * (1 + sa) + TAIL_U * s.edge;
    return s;
  }
  function radiusAll(s, th, out) {
    const t = th - s.rotation;
    let r = 0, r1 = 0, r2 = 0;
    for (let k = 0; k < s.M; k++) {
      const m = k + 1, c = Math.cos(m * t), sn = Math.sin(m * t);
      const ac = s.a[k] * c, bs = s.b[k] * sn;
      r += ac + bs; r1 += m * (s.b[k] * c - s.a[k] * sn); r2 -= m * m * (ac + bs);
    }
    out[0] = s.R * (1 + r); out[1] = s.R * r1; out[2] = s.R * r2;
  }
  const _r = new Float64Array(3);
  function shapeEval(s, x, y, out, hess) {
    const dx = x - s.x0, dy = y - s.y0;
    const rho = Math.max(Math.hypot(dx, dy), 1e-4 * s.R);
    radiusAll(s, Math.atan2(dy, dx), _r);
    const r = _r[0], r1 = _r[1], r2 = _r[2], w = s.edge;
    const u = (r - rho) / w, t = Math.tanh(u);
    const S = 0.5 * (1 + t), S1 = 0.5 * (1 - t * t);
    const rx = dx / rho, ry = dy / rho, rho2 = rho * rho;
    const thx = -dy / rho2, thy = dx / rho2;
    const ux = (r1 * thx - rx) / w, uy = (r1 * thy - ry) / w;
    out[0] += s.dn * S; out[1] += s.dn * S1 * ux; out[2] += s.dn * S1 * uy;
    if (!hess) return;
    const S2 = -2 * t * S1, rho4 = rho2 * rho2;
    const thxx = 2 * dx * dy / rho4, thxy = (dy * dy - dx * dx) / rho4, thyy = -2 * dx * dy / rho4;
    const pxx = (1 - rx * rx) / rho, pxy = -rx * ry / rho, pyy = (1 - ry * ry) / rho;
    const uxx = (r2 * thx * thx + r1 * thxx - pxx) / w;
    const uxy = (r2 * thx * thy + r1 * thxy - pxy) / w;
    const uyy = (r2 * thy * thy + r1 * thyy - pyy) / w;
    out[3] += s.dn * (S2 * ux * ux + S1 * uxx); out[4] += s.dn * (S2 * ux * uy + S1 * uxy);
    out[5] += s.dn * (S2 * uy * uy + S1 * uyy);
  }
  function shapeValue(s, x, y) { const o = [0, 0, 0]; shapeEval(s, x, y, o, false); return o[0]; }
  function pertEval(shapes, x, y, out, hess) {
    out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0;
    for (let i = 0; i < shapes.length; i++) shapeEval(shapes[i], x, y, out, hess);
  }
  function safeStep(shapes, x, y, ds) {
    let h = Infinity;
    for (const s of shapes) {
      const dx = x - s.x0, dy = y - s.y0;
      radiusAll(s, Math.atan2(dy, dx), _r);
      h = Math.min(h, Math.max(0.5 * (Math.abs(_r[0] - Math.hypot(dx, dy)) - STEP_U * s.edge), 0));
    }
    return Math.max(h, ds);
  }

  // ---------- cell ----------
  function portIndex(cell, phi) {
    let idx = -1;
    for (let i = 0; i < cell.ports.length; i++) {
      const p = cell.ports[i];
      const half = Math.asin(Math.min(1, p.width / (2 * cell.radius)));
      let d = phi - p.angle; d = Math.atan2(Math.sin(d), Math.cos(d));
      if (Math.abs(d) <= half) idx = i;
    }
    return idx;
  }
  const isIn = p => p.role === 'in' || p.role === 'inout';
  const isOut = p => p.role === 'out' || p.role === 'inout';
  function linspace(n) { if (n <= 1) return [0]; const o = []; for (let i = 0; i < n; i++) o.push(-0.5 + i / (n - 1)); return o; }

  function launch(cell, port, src) {
    const p = cell.ports[port], a = p.angle;
    const nx = Math.cos(a), ny = Math.sin(a), tnx = -ny, tny = nx;
    const cx = cell.radius * nx, cy = cell.radius * ny;
    const U = linspace(src.nPos).map(v => v * 0.9 * p.width), V = linspace(src.nAng).map(v => v * p.fan);
    const mw = src.modeWidth ?? p.width / 2;
    const base = Math.atan2(-ny, -nx) + p.launch;
    const rays = [];
    let norm = 0;
    for (const u of U) for (const v of V) {
      let amp = Math.exp(-((u / mw) ** 2));
      if (src.angWidth != null) amp *= Math.exp(-((v / src.angWidth) ** 2));
      const ang = base + v;
      rays.push({ x: cx + u * tnx, y: cy + u * tny, tx: Math.cos(ang), ty: Math.sin(ang), amp });
      norm += amp * amp;
    }
    norm = Math.sqrt(norm);
    rays.forEach(r => r.amp /= norm);
    return rays;
  }

  // ---------- tracer ----------
  // state vector s = [x, y, px, py, L, Qr, Qi, Pr, Pi]
  const _e = new Float64Array(6);
  function derivs(shapes, n0, curved, s, d) {
    pertEval(shapes, s[0], s[1], _e, curved);
    const n = n0 + _e[0];
    const pn = Math.hypot(s[2], s[3]), tx = s[2] / pn, ty = s[3] / pn;
    d[0] = tx; d[1] = ty; d[4] = n;
    if (curved) {
      const ex = -ty, ey = tx;
      const nn = _e[1] * ex + _e[2] * ey;
      const nnn = ex * ex * _e[3] + 2 * ex * ey * _e[4] + ey * ey * _e[5];
      const c = nnn - 2 * nn * nn / n;
      d[2] = _e[1]; d[3] = _e[2];
      d[5] = s[7] / n; d[6] = s[8] / n;
      d[7] = c * s[5]; d[8] = c * s[6];
    } else {
      d[2] = 0; d[3] = 0; d[5] = s[7] / n0; d[6] = s[8] / n0; d[7] = 0; d[8] = 0;
    }
  }
  const K1 = new Float64Array(9), K2 = new Float64Array(9), K3 = new Float64Array(9), K4 = new Float64Array(9), T = new Float64Array(9);
  function rk4(shapes, n0, curved, h, s) {
    derivs(shapes, n0, curved, s, K1);
    for (let i = 0; i < 9; i++) T[i] = s[i] + 0.5 * h * K1[i];
    derivs(shapes, n0, curved, T, K2);
    for (let i = 0; i < 9; i++) T[i] = s[i] + 0.5 * h * K2[i];
    derivs(shapes, n0, curved, T, K3);
    for (let i = 0; i < 9; i++) T[i] = s[i] + h * K3[i];
    derivs(shapes, n0, curved, T, K4);
    for (let i = 0; i < 9; i++) s[i] += h / 6 * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
  }

  function defaultDs(shapes) {
    let ds = 2e-6; for (const s of shapes) ds = Math.min(ds, s.edge / 2); return ds;
  }

  /* trace(cell, port, shapes, opt) → { exits: [...], nLaunched, lost, paths?, chords? }
     opt: mode ('curved' | 'straight' | 'none'), nPos, nAng, waist, ds, maxBounces, ampMin, adaptive,
          recordPaths (first n rays) or pathIds (list of ray ids), recordChords */
  function trace(cell, port, shapes, opt = {}) {
    const mode = opt.mode || 'curved';
    shapes = mode === 'none' ? [] : (shapes || []);
    const curved = mode === 'curved';
    const src = { nPos: opt.nPos ?? 5, nAng: opt.nAng ?? 41, waist: opt.waist ?? 5e-6, modeWidth: opt.modeWidth, angWidth: opt.angWidth };
    const Rc = cell.radius, n0 = cell.n_eff, k0 = TAU / cell.wavelength;
    const R = cell.reflectance ?? 0.99, phR = cell.reflection_phase ?? Math.PI;
    const ds = opt.ds ?? defaultDs(shapes), adaptive = opt.adaptive ?? true;
    const maxB = opt.maxBounces ?? 400, ampMin = opt.ampMin ?? 1e-3;
    for (const s of shapes) if (Math.hypot(s.x0, s.y0) + s.bound > Rc) throw new Error('a perturbation reaches the wall');
    const rays = launch(cell, port, src);
    const amp0max = Math.max(...rays.map(r => r.amp));
    const zR = k0 * n0 * src.waist ** 2 / 2;
    const exits = [], lost = { bounces: 0, amplitude: 0 };
    const paths = [], chords = opt.recordChords ? [] : null;
    const pathSet = opt.pathIds ? new Set(opt.pathIds) : null;
    const s = new Float64Array(9);
    for (let id = 0; id < rays.length; id++) {
      const r0 = rays[id];
      let x = r0.x, y = r0.y, tx = r0.tx, ty = r0.ty, amp = r0.amp;
      let L = 0, Qr = 1, Qi = 0, Pr = 0, Pi = n0 / zR, argQ = 0, phase = 0, nb = 0, inside = false;
      let px = 0, py = 0;
      const path = (id < (opt.recordPaths || 0) || (pathSet && pathSet.has(id))) ? [x, y] : null;
      const ch = chords ? [] : null;
      for (let guard = 0; guard < 1e6; guard++) {
        if (!inside) {
          const b = x * tx + y * ty;
          const sWall = -b + Math.sqrt(Math.max(b * b - (x * x + y * y) + Rc * Rc, 0));
          let sReg = Infinity;
          for (const sh of shapes) {
            const dx = x - sh.x0, dy = y - sh.y0, bb = dx * tx + dy * ty;
            const cc = dx * dx + dy * dy - sh.bound * sh.bound, disc = bb * bb - cc;
            if (cc > 0 && bb < 0 && disc > 0) sReg = Math.min(sReg, -bb - Math.sqrt(disc));
          }
          const toReg = sReg < sWall, sg = toReg ? sReg : sWall;
          if (ch) ch.push(x, y, tx, ty, sg);
          x += sg * tx; y += sg * ty; L += n0 * sg;
          const nQr = Qr + Pr * sg / n0, nQi = Qi + Pi * sg / n0;
          argQ += Math.atan2(nQi * Qr - nQr * Qi, nQr * Qr + nQi * Qi);
          Qr = nQr; Qi = nQi;
          if (toReg) { inside = true; px = n0 * tx; py = n0 * ty; if (path) path.push(x, y); continue; }
          const rr = Math.hypot(x, y); x *= Rc / rr; y *= Rc / rr;
          if (path) path.push(x, y);
          const pid = portIndex(cell, Math.atan2(y, x));
          if (pid >= 0) {
            exits.push({ port: pid, ray: id, x, y, tx, ty, L, Qr, Qi, Pr, Pi, argQ, amp, phase, nb });
            break;
          }
          const nx = x / Rc, ny = y / Rc, cosc = tx * nx + ty * ny;
          tx -= 2 * cosc * nx; ty -= 2 * cosc * ny;
          const f = 2 * n0 / (Rc * Math.max(cosc, 1e-9));
          Pr -= f * Qr; Pi -= f * Qi;
          amp *= Math.sqrt(typeof R === 'function' ? R(cosc) : R);
          phase += phR; nb++;
          if (nb >= maxB) { lost.bounces++; break; }
          if (amp < ampMin * amp0max) { lost.amplitude++; break; }
        } else {
          const h = adaptive ? safeStep(shapes, x, y, ds) : ds;
          s[0] = x; s[1] = y; s[2] = px; s[3] = py; s[4] = L; s[5] = Qr; s[6] = Qi; s[7] = Pr; s[8] = Pi;
          rk4(shapes, n0, curved, h, s);
          argQ += Math.atan2(s[6] * Qr - s[5] * Qi, s[5] * Qr + s[6] * Qi);
          x = s[0]; y = s[1]; px = s[2]; py = s[3]; L = s[4]; Qr = s[5]; Qi = s[6]; Pr = s[7]; Pi = s[8];
          if (x * x + y * y >= Rc * Rc) throw new Error('ray reached the wall inside a perturbation');
          let any = false;
          for (const sh of shapes) if ((x - sh.x0) ** 2 + (y - sh.y0) ** 2 < sh.bound * sh.bound) { any = true; break; }
          if (path) path.push(x, y);
          if (!any) { const pn = Math.hypot(px, py); tx = px / pn; ty = py / pn; inside = false; }
        }
      }
      if (path) paths.push({ id, pts: path });
      if (ch) chords.push(ch);
    }
    return { exits, nLaunched: rays.length, lost, paths, chords, cell, src };
  }

  // ---------- detector field ----------
  function detectorPoints(cell, port) {
    const a = cell.ports[port].angle, nx = Math.cos(a), ny = Math.sin(a);
    const d = cell.detector || { distance: 100e-6, width: 400e-6, n_pix: 64 };
    const cx = cell.radius * nx + d.distance * nx, cy = cell.radius * ny + d.distance * ny;
    const pts = [];
    for (let i = 0; i < d.n_pix; i++) {
      const u = (d.n_pix > 1 ? -0.5 + i / (d.n_pix - 1) : 0) * d.width;
      pts.push([cx - u * ny, cy + u * nx]);
    }
    return pts;
  }
  // per-exit beamlet contributions at the pixels: returns {re, im} arrays [nPix][nExit] flattened, and ray ids
  function beamletMatrix(cell, exits, port) {
    const n0 = cell.n_eff, k0 = TAU / cell.wavelength;
    const ex = exits.filter(e => e.port === port);
    const pts = detectorPoints(cell, port), np = pts.length, ne = ex.length;
    const re = new Float64Array(np * ne), im = new Float64Array(np * ne);
    for (let j = 0; j < ne; j++) {
      const e = ex[j];
      for (let i = 0; i < np; i++) {
        const dx = pts[i][0] - e.x, dy = pts[i][1] - e.y;
        const s = dx * e.tx + dy * e.ty, q = -dx * e.ty + dy * e.tx;
        const Qr = e.Qr + e.Pr * s / n0, Qi = e.Qi + e.Pi * s / n0;
        const qq = Qr * Qr + Qi * Qi;
        const Mr = (e.Pr * Qr + e.Pi * Qi) / qq, Mi = (e.Pi * Qr - e.Pr * Qi) / qq;
        const argQs = e.argQ + Math.atan2(Qi * e.Qr - Qr * e.Qi, Qr * e.Qr + Qi * e.Qi);
        const mag = e.amp * Math.pow(qq, -0.25) * Math.exp(-k0 * 0.5 * Mi * q * q);
        const ph = -0.5 * argQs + k0 * (e.L + n0 * s + 0.5 * Mr * q * q) + e.phase;
        re[i * ne + j] = mag * Math.cos(ph); im[i * ne + j] = mag * Math.sin(ph);
      }
    }
    return { re, im, np, ne, rays: ex.map(e => e.ray) };
  }
  function applyMatrix(B, phase) {   // phase: per-ray extra phase (Float64Array over launched rays) or null
    const Er = new Float64Array(B.np), Ei = new Float64Array(B.np);
    const cr = new Float64Array(B.ne), ci = new Float64Array(B.ne);
    for (let j = 0; j < B.ne; j++) { const p = phase ? phase[B.rays[j]] : 0; cr[j] = Math.cos(p); ci[j] = Math.sin(p); }
    for (let i = 0; i < B.np; i++) {
      let a = 0, b = 0; const o = i * B.ne;
      for (let j = 0; j < B.ne; j++) { const r = B.re[o + j], m = B.im[o + j]; a += r * cr[j] - m * ci[j]; b += r * ci[j] + m * cr[j]; }
      Er[i] = a; Ei[i] = b;
    }
    return { re: Er, im: Ei };
  }
  function detectorFields(cell, tr) {
    const out = {};
    cell.ports.forEach((p, i) => { if (isOut(p)) out[i] = applyMatrix(beamletMatrix(cell, tr.exits, i), null); });
    return out;
  }

  // ---------- phase-screen model ----------
  class PhaseScreen {
    constructor(cell, opt = {}) {
      this.cell = cell; this.opt = opt;
      this.inputs = cell.ports.map((p, i) => isIn(p) ? i : -1).filter(i => i >= 0);
      this.outputs = cell.ports.map((p, i) => isOut(p) ? i : -1).filter(i => i >= 0);
      this.tables = this.inputs.map(port => {
        const tr = trace(cell, port, [], Object.assign({}, opt, { mode: 'none', recordChords: true }));
        const B = {}; for (const o of this.outputs) B[o] = beamletMatrix(cell, tr.exits, o);
        return { port, tr, B };
      });
    }
    deltaL(k, shapes) {
      const t = this.tables[k], dL = new Float64Array(t.tr.nLaunched);
      for (const sh of shapes) {
        const st = Math.min(3e-6, sh.edge), R2 = sh.bound * sh.bound;
        t.tr.chords.forEach((ch, ray) => {
          let acc = 0;
          for (let c = 0; c < ch.length; c += 5) {
            const x = ch[c], y = ch[c + 1], tx = ch[c + 2], ty = ch[c + 3], ln = ch[c + 4];
            const dx = x - sh.x0, dy = y - sh.y0, b = dx * tx + dy * ty;
            const disc = b * b - (dx * dx + dy * dy - R2);
            if (disc <= 0) continue;
            const sq = Math.sqrt(disc), s1 = Math.max(-b - sq, 0), s2 = Math.min(-b + sq, ln);
            if (s2 <= s1) continue;
            const nq = Math.max(8, Math.ceil((s2 - s1) / st) + 1), hq = (s2 - s1) / nq;
            let sum = 0;
            for (let q = 0; q < nq; q++) { const s = s1 + (q + 0.5) * hq; sum += shapeValue(sh, x + s * tx, y + s * ty); }
            acc += sum * hq;
          }
          dL[ray] += acc;
        });
      }
      return dL;
    }
    fields(shapes) {   // {"in,out": {re, im}}
      const out = {}, k0 = TAU / this.cell.wavelength;
      this.tables.forEach((t, k) => {
        let ph = null;
        if (shapes && shapes.length) { ph = this.deltaL(k, shapes); for (let i = 0; i < ph.length; i++) ph[i] *= k0; }
        for (const o of this.outputs) out[t.port + ',' + o] = applyMatrix(t.B[o], ph);
      });
      return out;
    }
  }
  function curvedFields(cell, shapes, opt = {}) {
    const out = {};
    cell.ports.forEach((p, i) => {
      if (!isIn(p)) return;
      const tr = trace(cell, i, shapes, opt);
      const f = detectorFields(cell, tr);
      for (const o in f) out[i + ',' + o] = f[o];
    });
    return out;
  }

  // ---------- harmonics ----------
  function chdShape(s, mMax) {
    const ph = k => (k + 1) * s.rotation;
    const a = [], b = [];
    for (let k = 0; k < mMax; k++) {
      const ak = s.a[k] || 0, bk = s.b[k] || 0;
      a.push(ak * Math.cos(ph(k)) - bk * Math.sin(ph(k))); b.push(ak * Math.sin(ph(k)) + bk * Math.cos(ph(k)));
    }
    return { R0: s.R, a, b };
  }
  function chdContour(valueFn, x0, y0, rMax, mMax = 8, nTheta = 256, nR = 400, level = 0.5) {
    // centroid
    let sw = 0, sx = 0, sy = 0; const ng = 129;
    for (let i = 0; i < ng; i++) for (let j = 0; j < ng; j++) {
      const X = x0 - rMax + 2 * rMax * j / (ng - 1), Y = y0 - rMax + 2 * rMax * i / (ng - 1);
      const w = Math.abs(valueFn(X, Y)); sw += w; sx += w * X; sy += w * Y;
    }
    const cx = sw ? sx / sw : x0, cy = sw ? sy / sw : y0;
    const r = new Float64Array(nTheta); let pk = 0;
    const prof = [];
    for (let k = 0; k < nTheta; k++) {
      const th = TAU * k / nTheta, c = Math.cos(th), sn = Math.sin(th), row = new Float64Array(nR);
      for (let j = 0; j < nR; j++) { const rr = rMax * j / (nR - 1); row[j] = Math.abs(valueFn(cx + rr * c, cy + rr * sn)); pk = Math.max(pk, row[j]); }
      prof.push(row);
    }
    for (let k = 0; k < nTheta; k++) {
      const v = prof[k]; let jl = -1;
      for (let j = 0; j < nR; j++) if (v[j] >= level * pk) jl = j;
      if (jl < 0) continue;
      if (jl + 1 >= nR) { r[k] = rMax; continue; }
      const f = v[jl] !== v[jl + 1] ? (v[jl] - level * pk) / (v[jl] - v[jl + 1]) : 0;
      r[k] = rMax * (jl + f) / (nR - 1);
    }
    let R0 = 0; for (let k = 0; k < nTheta; k++) R0 += r[k]; R0 /= nTheta;
    const a = [], b = [];
    for (let m = 1; m <= mMax; m++) {
      let cr = 0, ci = 0;
      for (let k = 0; k < nTheta; k++) { const th = TAU * k / nTheta; cr += r[k] * Math.cos(m * th); ci -= r[k] * Math.sin(m * th); }
      cr /= nTheta; ci /= nTheta; a.push(2 * cr / R0); b.push(-2 * ci / R0);
    }
    return { R0, a, b, cx, cy, r: Array.from(r) };
  }

  // ---------- random numbers / shapes ----------
  function rng(seed) {
    let t = seed >>> 0, spare = null;
    const u = () => { t += 0x6D2B79F5; let r = Math.imul(t ^ t >>> 15, 1 | t); r ^= r + Math.imul(r ^ r >>> 7, 61 | r); return ((r ^ r >>> 14) >>> 0) / 4294967296; };
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let a, b, s; do { a = 2 * u() - 1; b = 2 * u() - 1; s = a * a + b * b; } while (s >= 1 || s === 0);
      const f = Math.sqrt(-2 * Math.log(s) / s); spare = b * f; return a * f;
    };
    return { uniform: (lo = 0, hi = 1) => lo + (hi - lo) * u(), normal };
  }
  function randomShape(g, cellRadius, o = {}) {
    const mMax = o.mMax ?? 6, sigma = o.sigma ?? 0.08, decay = o.decay ?? 0.5;
    const a = [], b = [];
    for (let m = 1; m <= mMax; m++) {
      const sd = m >= 2 ? sigma / Math.pow(Math.max(m - 1, 1), decay) : 0;
      a.push(g.normal() * sd); b.push(g.normal() * sd);
    }
    const R = g.uniform(...(o.Rrange || [60e-6, 110e-6]));
    const dn = g.uniform(...(o.dnRange || [1e-3, 1e-3]));
    const s = makeShape({ x0: 0, y0: 0, R, dn, edge: o.edge ?? 3e-6, a, b });
    if (o.centre) { s.x0 = o.centre[0]; s.y0 = o.centre[1]; }
    else {
      const rmax = o.placeRadius ?? (cellRadius - 40e-6 - s.bound);
      const rr = rmax * Math.sqrt(g.uniform()), ph = g.uniform(0, TAU);
      s.x0 = rr * Math.cos(ph); s.y0 = rr * Math.sin(ph);
    }
    if (Math.hypot(s.x0, s.y0) + s.bound > cellRadius - 10e-6) {   // keep inside
      const f = (cellRadius - 10e-6 - s.bound) / Math.hypot(s.x0, s.y0); s.x0 *= f; s.y0 *= f;
    }
    return s;
  }

  // ---------- features / ridge ----------
  function stackRelative(fields, ref) {
    const keys = Object.keys(ref).sort(), out = [];
    for (const k of keys) {
      const f = fields[k], r = ref[k];
      let mean = 0; for (let i = 0; i < r.re.length; i++) mean += r.re[i] ** 2 + r.im[i] ** 2; mean /= r.re.length;
      for (let i = 0; i < f.re.length; i++) {
        const I = f.re[i] ** 2 + f.im[i] ** 2, I0 = r.re[i] ** 2 + r.im[i] ** 2;
        out.push(I / (I0 + 1e-12 * mean) - 1);
      }
    }
    return out;
  }
  function cholSolve(A, n, B, m) {   // A n×n SPD (row-major, overwritten), B n×m → solution
    for (let j = 0; j < n; j++) {
      let d = A[j * n + j];
      for (let k = 0; k < j; k++) d -= A[j * n + k] ** 2;
      d = Math.sqrt(Math.max(d, 1e-300)); A[j * n + j] = d;
      for (let i = j + 1; i < n; i++) {
        let v = A[i * n + j];
        for (let k = 0; k < j; k++) v -= A[i * n + k] * A[j * n + k];
        A[i * n + j] = v / d;
      }
    }
    const X = Float64Array.from(B);
    for (let c = 0; c < m; c++) {
      for (let i = 0; i < n; i++) { let v = X[i * m + c]; for (let k = 0; k < i; k++) v -= A[i * n + k] * X[k * m + c]; X[i * m + c] = v / A[i * n + i]; }
      for (let i = n - 1; i >= 0; i--) { let v = X[i * m + c]; for (let k = i + 1; k < n; k++) v -= A[k * n + i] * X[k * m + c]; X[i * m + c] = v / A[i * n + i]; }
    }
    return X;
  }
  // ridge in dual form on standardised features; returns predictions for Xte and the alpha used
  function ridgeFitPredict(Xtr, Ytr, Xte, alpha) {
    const n = Xtr.length, d = Xtr[0].length, m = Ytr[0].length;
    const mu = new Float64Array(d), sd = new Float64Array(d);
    for (const r of Xtr) for (let j = 0; j < d; j++) mu[j] += r[j] / n;
    for (const r of Xtr) for (let j = 0; j < d; j++) sd[j] += (r[j] - mu[j]) ** 2 / n;
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    const Z = Xtr.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
    const Zt = Xte.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
    const ym = new Float64Array(m); for (const r of Ytr) for (let c = 0; c < m; c++) ym[c] += r[c] / n;
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let v = 0; const a = Z[i], b = Z[j]; for (let k = 0; k < d; k++) v += a[k] * b[k];
      K[i * n + j] = K[j * n + i] = v;
    }
    // centre the kernel (equivalent to an unpenalised intercept with centred features: features already centred)
    for (let i = 0; i < n; i++) K[i * n + i] += alpha;
    const B = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let c = 0; c < m; c++) B[i * m + c] = Ytr[i][c] - ym[c];
    const Acoef = cholSolve(K, n, B, m);
    return Zt.map(z => {
      const p = Array.from(ym);
      for (let i = 0; i < n; i++) { let v = 0; const zi = Z[i]; for (let k = 0; k < d; k++) v += z[k] * zi[k]; for (let c = 0; c < m; c++) p[c] += v * Acoef[i * m + c]; }
      return p;
    });
  }
  function r2(Y, P) {
    const m = Y[0].length, out = [];
    for (let c = 0; c < m; c++) {
      let mean = 0; for (const r of Y) mean += r[c] / Y.length;
      let ss = 0, se = 0; Y.forEach((r, i) => { ss += (r[c] - mean) ** 2; se += (r[c] - P[i][c]) ** 2; });
      out.push(ss > 0 ? 1 - se / ss : 0);
    }
    return out;
  }
  function ridgeCV(X, Y, alphas, seed = 0) {
    const n = X.length, g = rng(seed + 99), idx = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(g.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nv = Math.max(2, Math.round(0.2 * n)), va = idx.slice(0, nv), tr = idx.slice(nv);
    let best = alphas[0], bs = -Infinity;
    for (const a of alphas) {
      const P = ridgeFitPredict(tr.map(i => X[i]), tr.map(i => Y[i]), va.map(i => X[i]), a);
      const s = r2(va.map(i => Y[i]), P).reduce((p, q) => p + q, 0);
      if (s > bs) { bs = s; best = a; }
    }
    return best;
  }

  const NGRC = { makeShape, shapeEval, shapeValue, pertEval, safeStep, portIndex, launch, trace, detectorPoints,
    beamletMatrix, applyMatrix, detectorFields, PhaseScreen, curvedFields, chdShape, chdContour, rng, randomShape,
    stackRelative, ridgeFitPredict, ridgeCV, r2, isIn, isOut };
  root.NGRC = NGRC;
  if (typeof module !== 'undefined' && module.exports) module.exports = NGRC;
})(typeof globalThis !== 'undefined' ? globalThis : this);
