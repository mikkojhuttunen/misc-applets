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
    s.cr = Math.cos(s.rotation); s.sr = Math.sin(s.rotation);
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
  // r(θ), r'(θ), r''(θ) at the direction of (dx, dy) (length rho): cos mθ, sin mθ by recurrence, no trig calls
  function radiusXY(s, dx, dy, rho, out) {
    const c0 = dx / rho, s0 = dy / rho;
    const c1 = c0 * s.cr + s0 * s.sr, s1 = s0 * s.cr - c0 * s.sr;
    let c = c1, sn = s1, r = 0, r1 = 0, r2 = 0;
    for (let k = 0; k < s.M; k++) {
      const m = k + 1, ac = s.a[k] * c, bs = s.b[k] * sn;
      r += ac + bs; r1 += m * (s.b[k] * c - s.a[k] * sn); r2 -= m * m * (ac + bs);
      const cn = c * c1 - sn * s1; sn = sn * c1 + c * s1; c = cn;
    }
    out[0] = s.R * (1 + r); out[1] = s.R * r1; out[2] = s.R * r2;
  }
  function shapeEval(s, x, y, out, hess) {
    const dx = x - s.x0, dy = y - s.y0;
    const rho = Math.max(Math.hypot(dx, dy), 1e-4 * s.R);
    radiusXY(s, dx, dy, rho, _r);
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
  // Δn only, allocation-free (hot path of the chord integrals)
  let _rr = 0, _rr1 = 0;
  function radius01(s, dx, dy, rho) {   // r and r' at the direction of (dx, dy) → _rr, _rr1
    const c0 = dx / rho, s0 = dy / rho, c1 = c0 * s.cr + s0 * s.sr, s1 = s0 * s.cr - c0 * s.sr;
    let c = c1, sn = s1, r = 0, r1 = 0;
    for (let k = 0; k < s.M; k++) {
      const m = k + 1;
      r += s.a[k] * c + s.b[k] * sn; r1 += m * (s.b[k] * c - s.a[k] * sn);
      const cn = c * c1 - sn * s1; sn = sn * c1 + c * s1; c = cn;
    }
    _rr = s.R * (1 + r); _rr1 = s.R * r1;
  }
  function shapeValue(s, x, y) {
    const dx = x - s.x0, dy = y - s.y0, rho = Math.max(Math.hypot(dx, dy), 1e-4 * s.R);
    radius01(s, dx, dy, rho);
    return s.dn * 0.5 * (1 + Math.tanh((_rr - rho) / s.edge));
  }
  function pertEval(shapes, x, y, out, hess) {
    out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0;
    for (let i = 0; i < shapes.length; i++) shapeEval(shapes[i], x, y, out, hess);
  }
  function safeStep(shapes, x, y, ds) {
    let h = Infinity;
    for (const s of shapes) {
      const dx = x - s.x0, dy = y - s.y0, rho = Math.max(Math.hypot(dx, dy), 1e-4 * s.R);
      radiusXY(s, dx, dy, rho, _r);
      h = Math.min(h, Math.max(0.5 * (Math.abs(_r[0] - Math.hypot(dx, dy)) - STEP_U * s.edge), 0));
    }
    return Math.max(h, ds);
  }

  // ---------- erf (|error| < 1e-15: series below 2.5, Lentz continued fraction for erfc above) ----------
  function erf(x) {
    const ax = Math.abs(x);
    if (ax < 2.5) {
      let term = ax, sum = ax, x2 = ax * ax;
      for (let n = 1; n < 100; n++) { term *= -x2 / n; const t = term / (2 * n + 1); sum += t; if (Math.abs(t) < 1e-17 * Math.abs(sum)) break; }
      const v = 2 / Math.sqrt(Math.PI) * sum; return x < 0 ? -v : v;
    }
    if (ax > 27) return x < 0 ? -1 : 1;
    // erfc(x) = exp(-x²)/sqrt(pi) · 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...)))) via modified Lentz
    let f = ax, C = ax, D = 0;
    for (let n = 1; n < 300; n++) {
      const a = n / 2;
      D = ax + a * D; D = 1 / (D || 1e-300);
      C = ax + a / (C || 1e-300);
      const d = C * D; f *= d; if (Math.abs(d - 1) < 1e-16) break;
    }
    const erfc = Math.exp(-ax * ax) / Math.sqrt(Math.PI) / f;
    return x < 0 ? erfc - 1 : 1 - erfc;
  }

  // ---------- cell ----------
  const isIn = p => p.role === 'in' || p.role === 'inout';
  const isOut = p => p.role === 'out' || p.role === 'inout';
  const modelOf = cell => cell.model ?? 'fga';
  function portIndex(cell, phi) {   // gbs: port containing wall angle phi (hard aperture)
    let idx = -1;
    for (let i = 0; i < cell.ports.length; i++) {
      const p = cell.ports[i];
      const half = Math.asin(Math.min(1, p.width / (2 * cell.radius)));
      let d = phi - p.angle; d = Math.atan2(Math.sin(d), Math.cos(d));
      if (Math.abs(d) <= half) idx = i;
    }
    return idx;
  }
  function portOffset(cell, i, sWall) {   // signed boundary distance from the centre of port i
    const per = TAU * cell.radius;
    let u = sWall - cell.ports[i].angle * cell.radius + per / 2;
    return ((u % per) + per) % per - per / 2;
  }
  function portFrame(cell, i) {
    const a = cell.ports[i].angle, nx = Math.cos(a), ny = Math.sin(a);
    return { cx: cell.radius * nx, cy: cell.radius * ny, nx, ny, tx: -ny, ty: nx };
  }
  function aperturePoints(cell, i) {
    const p = cell.ports[i], n = Math.ceil(p.width / (cell.wavelength / cell.n_eff / 3)) + 1, pts = [];
    for (let k = 0; k < n; k++) { const a = p.angle + ((k + 0.5) / n * p.width - p.width / 2) / cell.radius; pts.push([cell.radius * Math.cos(a), cell.radius * Math.sin(a)]); }
    return { pts, du: p.width / n };
  }
  function linspace(n) { if (n <= 1) return [0]; const o = []; for (let i = 0; i < n; i++) o.push(-0.5 + i / (n - 1)); return o; }

  function launch(cell, port, src) {   // gbs launch grid
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
  function fgaParams(cell, src) {
    const zR = TAU / cell.wavelength * cell.n_eff * src.frozen ** 2 / 2;
    return { beta: cell.n_eff / zR, gamma: 2 / src.frozen ** 2 };
  }
  // frozen-Gaussian launch grid: positions q × directions φ, weights W = <g_{q,p}|E0> Δq Δp k0/2π (see rays.launch_fga)
  function launchFga(cell, port, src) {
    const p = cell.ports[port], n0 = cell.n_eff, k0 = TAU / cell.wavelength, lam = cell.wavelength;
    const { gamma } = fgaParams(cell, src);
    const thd = Math.max(p.fan / 2, 1e-4);
    const wIn = src.modeWidth ?? lam / (Math.PI * n0 * Math.tan(thd));
    const cosL = Math.max(Math.cos(p.launch), 1e-3), wLine = wIn / cosL;
    const qmax = 2.5 * Math.sqrt(wLine ** 2 + src.frozen ** 2);
    const nPos = src.nPos || Math.ceil(2 * qmax / (0.5 * src.frozen)) + 1;
    const q = nPos > 1 ? [...Array(nPos).keys()].map(i => -qmax + 2 * qmax * i / (nPos - 1)) : [0];
    const span = Math.min(2.5 * thd + 3 / (k0 * n0 * src.frozen), 1.4);
    const phi = src.nAng > 1 ? [...Array(src.nAng).keys()].map(i => p.launch + span * (-1 + 2 * i / (src.nAng - 1))) : [p.launch];
    const dq = q.length > 1 ? q[1] - q[0] : Math.sqrt(TAU / gamma), dphi = phi.length > 1 ? phi[1] - phi[0] : 1;
    const f = portFrame(cell, port), base = Math.atan2(-f.ny, -f.nx);
    const p0 = -n0 * Math.sin(p.launch), a = gamma / 2 + 1 / wLine ** 2, norm = (gamma / Math.PI) ** 0.25 * Math.sqrt(Math.PI / a);
    const rays = [];
    for (const qq of q) for (const ph of phi) {
      const pp = -n0 * Math.sin(ph);
      // <g|E0> = norm exp(b²/4a + c), b = γq − i k0 (pp − p0), c = −γq²/2 + i k0 pp q
      const br = gamma * qq, bi = -k0 * (pp - p0);
      const er = (br * br - bi * bi) / (4 * a) - gamma / 2 * qq * qq, ei = 2 * br * bi / (4 * a) + k0 * pp * qq;
      const mag = norm * Math.exp(er) * dq * n0 * Math.cos(ph) * dphi * k0 / TAU;
      const ang = base + ph;
      rays.push({ x: f.cx + qq * f.tx, y: f.cy + qq * f.ty, tx: Math.cos(ang), ty: Math.sin(ang), Wr: mag * Math.cos(ei), Wi: mag * Math.sin(ei), cos0: Math.cos(ph) });
    }
    return rays;
  }
  // z = ½ (A + D − i(γ/k0) B + i(k0/γ) C) with Q = A + iβB, P = C + iβD; returns [re, im]
  function zfun(Qr, Qi, Pr, Pi, beta, gamma, k0) {
    const A = Qr, B = Qi / beta, C = Pr, D = Pi / beta;
    return [0.5 * (A + D), 0.5 * (-(gamma / k0) * B + (k0 / gamma) * C)];
  }
  const zang = (a, b) => Math.atan2(b[1] * a[0] - b[0] * a[1], b[0] * a[0] + b[1] * a[1]);   // arg(b / a)

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
  function defaultDs(shapes) { let ds = 2e-6; for (const s of shapes) ds = Math.min(ds, s.edge / 2); return ds; }

  /* trace(cell, port, shapes, opt) → { exits, nLaunched, lost, paths?, chords?, model, beta, gamma }
     cell.model: 'fga' (default, frozen Gaussians, smooth ports) or 'gbs' (evolving beamlets, hard ports).
     opt: mode ('curved' | 'straight' | 'none'), nPos (fga: 0/null = automatic), nAng, frozen (fga w_f), waist (gbs),
          ds, maxBounces, ampMin, recTol, adaptive, recordPaths (first n rays) or pathIds, recordChords */
  function trace(cell, port, shapes, opt = {}) {
    const mode = opt.mode || 'curved', fga = modelOf(cell) === 'fga';
    shapes = mode === 'none' ? [] : (shapes || []);
    const curved = mode === 'curved';
    const src = { nPos: opt.nPos ?? (fga ? null : 5), nAng: opt.nAng ?? 41, waist: opt.waist ?? 5e-6, frozen: opt.frozen ?? 20e-6, modeWidth: opt.modeWidth, angWidth: opt.angWidth };
    const Rc = cell.radius, n0 = cell.n_eff, k0 = TAU / cell.wavelength;
    const R = cell.reflectance ?? 0.99, phR = cell.reflection_phase ?? Math.PI;
    const ds = opt.ds ?? defaultDs(shapes), adaptive = opt.adaptive ?? true;
    const maxB = opt.maxBounces ?? 400, ampMin = opt.ampMin ?? 1e-3, recTol = opt.recTol ?? 1e-3, zMax = opt.zMax ?? 1e4;
    for (const s of shapes) if (Math.hypot(s.x0, s.y0) + s.bound > Rc) throw new Error('a perturbation reaches the wall');
    let rays, beta = 0, gamma = 0;
    if (fga) {
      ({ beta, gamma } = fgaParams(cell, src));
      rays = launchFga(cell, port, src);
      let wmax = 0; for (const r of rays) wmax = Math.max(wmax, Math.hypot(r.Wr, r.Wi));
      rays = rays.filter(r => Math.hypot(r.Wr, r.Wi) > recTol * wmax);
      rays.forEach(r => { r.amp = 1; });
    } else rays = launch(cell, port, src);
    let amp0max = 0; for (const r of rays) amp0max = Math.max(amp0max, r.amp);
    const zR = k0 * n0 * src.waist ** 2 / 2;
    const exits = [], lost = { bounces: 0, amplitude: 0 };
    const paths = [], chords = opt.recordChords ? [] : null;
    const pathSet = opt.pathIds ? new Set(opt.pathIds) : null;
    const s = new Float64Array(9), np = cell.ports.length;
    const widths = cell.ports.map(p => p.width), D = new Float64Array(np), Tp = new Float64Array(np);
    const ww = src.frozen / Math.SQRT2;
    for (let id = 0; id < rays.length; id++) {
      const r0 = rays[id];
      let x = r0.x, y = r0.y, tx = r0.tx, ty = r0.ty, amp = r0.amp;
      let L = 0, Qr, Qi, Pr, Pi, argQ = 0, argZ = 0, phase = 0, nb = 0, nch = 0, inside = false;
      if (fga) { Qr = r0.cos0; Qi = 0; Pr = 0; Pi = beta / r0.cos0; } else { Qr = 1; Qi = 0; Pr = 0; Pi = n0 / zR; }
      let px = 0, py = 0;
      const zstep = (nQr, nQi, nPr, nPi) => { if (fga) argZ += zang(zfun(Qr, Qi, Pr, Pi, beta, gamma, k0), zfun(nQr, nQi, nPr, nPi, beta, gamma, k0)); };
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
          nch++;
          x += sg * tx; y += sg * ty; L += n0 * sg;
          const nQr = Qr + Pr * sg / n0, nQi = Qi + Pi * sg / n0;
          zstep(nQr, nQi, Pr, Pi);
          argQ += Math.atan2(nQi * Qr - nQr * Qi, nQr * Qr + nQi * Qi);
          Qr = nQr; Qi = nQi;
          if (toReg) { inside = true; px = n0 * tx; py = n0 * ty; if (path) path.push(x, y); continue; }
          if (path) path.push(x, y);
          const rr = Math.hypot(x, y), nx = x / rr, ny = y / rr;
          const sW = ((Math.atan2(y, x) % TAU) + TAU) % TAU * Rc;
          let Ttot = 0;
          for (let pi = 0; pi < np; pi++) {
            D[pi] = portOffset(cell, pi, sW);
            let near;
            if (fga) {
              near = Math.abs(D[pi]) < widths[pi] / 2 + 3 * src.frozen;
              Tp[pi] = near ? 0.5 * (erf((widths[pi] / 2 - D[pi]) / ww) + erf((widths[pi] / 2 + D[pi]) / ww)) : 0;
            } else {
              near = Math.abs(D[pi]) <= widths[pi] / 2;
              Tp[pi] = near ? 1 : 0;
            }
            Ttot += Tp[pi];
            if (near) {
              const e = { port: pi, ray: id, x, y, tx, ty, L, Qr, Qi, Pr, Pi, argQ, amp, phase, nb, nch };
              if (fga) { e.argZ = argZ; e.Wr = r0.Wr; e.Wi = r0.Wi; }
              exits.push(e);
            }
          }
          if (!fga) Ttot = Ttot > 0 ? 1 : 0;
          Ttot = Math.min(Ttot, 1); if (Ttot > 1 - 1e-9) Ttot = 1;
          if (!(Ttot < 1)) break;
          const cosc = tx * nx + ty * ny;
          tx -= 2 * cosc * nx; ty -= 2 * cosc * ny;
          const tn = Math.hypot(tx, ty); tx /= tn; ty /= tn;
          const f = 2 * n0 / (Rc * Math.max(cosc, 1e-9));
          const nPr = Pr - f * Qr, nPi = Pi - f * Qi;
          zstep(Qr, Qi, nPr, nPi);
          Pr = nPr; Pi = nPi;
          amp *= Math.sqrt(1 - Ttot);
          amp *= Math.sqrt(typeof R === 'function' ? R(cosc) : R);
          phase += phR; nb++;
          if (fga) { const z = zfun(Qr, Qi, Pr, Pi, beta, gamma, k0); if (Math.hypot(z[0], z[1]) > zMax) { lost.prefactor = (lost.prefactor || 0) + 1; break; } }
          if (nb >= maxB) { lost.bounces++; break; }
          if (amp < ampMin * amp0max) { lost.amplitude++; break; }
        } else {
          const h = adaptive ? safeStep(shapes, x, y, ds) : ds;
          s[0] = x; s[1] = y; s[2] = px; s[3] = py; s[4] = L; s[5] = Qr; s[6] = Qi; s[7] = Pr; s[8] = Pi;
          rk4(shapes, n0, curved, h, s);
          zstep(s[5], s[6], s[7], s[8]);
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
    return { exits, nLaunched: rays.length, lost, paths, chords, cell, src, model: fga ? 'fga' : 'gbs', beta, gamma };
  }
  function recordWeight(tr, e) {   // fga: |amp W R|, the record's amplitude at the port
    const k0 = TAU / tr.cell.wavelength, z = zfun(e.Qr, e.Qi, e.Pr, e.Pi, tr.beta, tr.gamma, k0);
    return e.amp * Math.hypot(e.Wr, e.Wi) * Math.sqrt(Math.hypot(z[0], z[1]));
  }

  // ---------- detector field ----------
  function detectorSines(cell) {
    const d = cell.detector, n = d.n_pix, ms = d.max_sin ?? 0.9;
    return [...Array(n).keys()].map(i => (n > 1 ? -1 + 2 * i / (n - 1) : 0) * ms);
  }
  function detectorPoints(cell, port) {
    const a = cell.ports[port].angle, nx = Math.cos(a), ny = Math.sin(a);
    const d = cell.detector || { distance: 100e-6, width: 400e-6, n_pix: 64 };
    if (d.farfield ?? true) {
      const R = d.ff_radius ?? 20e-3, cx = cell.radius * nx, cy = cell.radius * ny;
      return detectorSines(cell).map(sn => { const cs = Math.sqrt(1 - sn * sn); return [cx + R * (cs * nx - sn * ny), cy + R * (cs * ny + sn * nx)]; });
    }
    const cx = cell.radius * nx + d.distance * nx, cy = cell.radius * ny + d.distance * ny;
    const pts = [];
    for (let i = 0; i < d.n_pix; i++) {
      const u = (d.n_pix > 1 ? -0.5 + i / (d.n_pix - 1) : 0) * d.width;
      pts.push([cx - u * ny, cy + u * nx]);
    }
    return pts;
  }
  // K[pixel, k] from aperture samples: 2D Rayleigh–Sommerfeld sqrt(k/(2π i R)) e^{ikR} cos θ du
  function propagationMatrix(cell, port, ap) {
    const f = portFrame(cell, port), k = TAU / cell.wavelength * cell.n_eff, n = ap.pts.length;
    if (cell.detector.farfield ?? true) {   // far field: sqrt(k/(2π i)) e^{−ik d̂·r} cos θ du
      const sns = detectorSines(cell), re = new Float64Array(sns.length * n), im = new Float64Array(sns.length * n), amp0 = Math.sqrt(k / TAU) * ap.du;
      sns.forEach((sn, i) => {
        const cs = Math.sqrt(1 - sn * sn), dx = cs * f.nx + sn * f.tx, dy = cs * f.ny + sn * f.ty;
        ap.pts.forEach((q, j) => { const ph = -k * (dx * q[0] + dy * q[1]) - Math.PI / 4; re[i * n + j] = amp0 * cs * Math.cos(ph); im[i * n + j] = amp0 * cs * Math.sin(ph); });
      });
      return { re, im, np: sns.length, n };
    }
    const pts = detectorPoints(cell, port);
    const re = new Float64Array(pts.length * n), im = new Float64Array(pts.length * n);
    pts.forEach((p, i) => ap.pts.forEach((q, j) => {
      const dx = p[0] - q[0], dy = p[1] - q[1], Rr = Math.hypot(dx, dy);
      const cs = Math.max((dx * f.nx + dy * f.ny) / Rr, 0), mag = Math.sqrt(k / (TAU * Rr)) * cs * ap.du, ph = k * Rr - Math.PI / 4;
      re[i * n + j] = mag * Math.cos(ph); im[i * n + j] = mag * Math.sin(ph);
    }));
    return { re, im, np: pts.length, n };
  }
  // fga: B[k, j] frozen-Gaussian field of record j at aperture point k (see field._frozen_at)
  function frozenMatrix(tr, list, port, ap) {
    const cell = tr.cell, n0 = cell.n_eff, k0 = TAU / cell.wavelength, f = portFrame(cell, port), g0 = (tr.gamma / Math.PI) ** 0.25;
    const n = ap.pts.length, ne = list.length, re = new Float32Array(n * ne), im = new Float32Array(n * ne);
    const xp = ap.pts.map(p => (p[0] - f.cx) * f.tx + (p[1] - f.cy) * f.ty);
    for (let j = 0; j < ne; j++) {
      const e = list[j];
      const cosc = Math.max(Math.abs(e.tx * f.nx + e.ty * f.ny), 1e-3);
      const z = zfun(e.Qr / cosc, e.Qi / cosc, e.Pr * cosc, e.Pi * cosc, tr.beta, tr.gamma, k0);
      const zr = zfun(e.Qr, e.Qi, e.Pr, e.Pi, tr.beta, tr.gamma, k0);
      const argz = e.argZ + zang(zr, z), rm = Math.sqrt(Math.hypot(z[0], z[1]));
      // c = amp W R e^{i(k0 L + phase)}
      const cph = 0.5 * argz + k0 * e.L + e.phase, cm = e.amp * rm;
      const cr = cm * (e.Wr * Math.cos(cph) - e.Wi * Math.sin(cph)), ci = cm * (e.Wr * Math.sin(cph) + e.Wi * Math.cos(cph));
      const xt = (e.x - f.cx) * f.tx + (e.y - f.cy) * f.ty;
      for (let k = 0; k < n; k++) {
        const d = xp[k] - xt, sx = (ap.pts[k][0] - e.x) * e.tx + (ap.pts[k][1] - e.y) * e.ty;
        const gm = g0 * Math.exp(-tr.gamma * d * d / 2), gp = k0 * n0 * sx;
        const grr = gm * Math.cos(gp), gii = gm * Math.sin(gp);
        re[k * ne + j] = cr * grr - ci * gii; im[k * ne + j] = cr * gii + ci * grr;
      }
    }
    return { re, im, n, ne };
  }
  // gbs: beamlet matrix straight to the detector pixels
  function beamletMatrix(cell, exits, port) {
    const n0 = cell.n_eff, k0 = TAU / cell.wavelength;
    const idx = [];
    exits.forEach((e, i) => { if (e.port === port) idx.push(i); });
    const ex = idx.map(i => exits[i]);
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
    return { re, im, np, ne, rays: ex.map(e => e.ray), idx };
  }
  function cmatvec(M, rows, cols, vr, vi) {   // complex (rows × cols) · v
    const Er = new Float64Array(rows), Ei = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      let a = 0, b = 0; const o = i * cols;
      for (let j = 0; j < cols; j++) { const r = M.re[o + j], m = M.im[o + j]; a += r * vr[j] - m * vi[j]; b += r * vi[j] + m * vr[j]; }
      Er[i] = a; Ei[i] = b;
    }
    return { re: Er, im: Ei };
  }
  // per-port operator: field = op.apply(per-record phases or null)
  function portOperator(tr, port, keep) {
    const idx = [];
    tr.exits.forEach((e, i) => { if (e.port === port && (!keep || keep[i])) idx.push(i); });
    if (tr.model === 'fga') {
      const ap = aperturePoints(tr.cell, port), K = propagationMatrix(tr.cell, port, ap);
      const B = frozenMatrix(tr, idx.map(i => tr.exits[i]), port, ap);
      return { idx, apply(ph) {
        const cr = new Float64Array(B.ne), ci = new Float64Array(B.ne);
        for (let j = 0; j < B.ne; j++) { const p = ph ? ph[idx[j]] : 0; cr[j] = Math.cos(p); ci[j] = Math.sin(p); }
        const A = cmatvec(B, B.n, B.ne, cr, ci);
        return cmatvec(K, K.np, K.n, A.re, A.im);
      } };
    }
    const G = beamletMatrix(tr.cell, tr.exits, port);
    return { idx, apply(ph) {
      const cr = new Float64Array(G.ne), ci = new Float64Array(G.ne);
      for (let j = 0; j < G.ne; j++) { const p = ph ? ph[G.idx[j]] : 0; cr[j] = Math.cos(p); ci[j] = Math.sin(p); }
      return cmatvec(G, G.np, G.ne, cr, ci);
    } };
  }
  function detectorFields(cell, tr) {
    const out = {};
    cell.ports.forEach((p, i) => { if (isOut(p)) out[i] = portOperator(tr, i, null).apply(null); });
    return out;
  }

  // ---------- chord integrals (Newton edges + Gauss–Legendre edge windows; see Shape.chord_integral) ----------
  function gaussLegendre(n) {   // nodes and weights on [−1, 1]
    const x = [], w = [];
    for (let i = 1; i <= n; i++) {
      let z = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), pp = 0;
      for (let it = 0; it < 100; it++) {
        let p1 = 1, p2 = 0;
        for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
        pp = n * (z * p1 - p2) / (z * z - 1);
        const z1 = z; z = z1 - p1 / pp; if (Math.abs(z - z1) < 1e-16) break;
      }
      x.push(-z); w.push(2 / ((1 - z * z) * pp * pp));
    }
    return { x, w };
  }
  const GL = gaussLegendre(12), GL32 = gaussLegendre(32);
  function chordIntegral(sh, x, y, tx, ty, ln) {   // returns value or NaN (→ numerical fallback)
    const dx = x - sh.x0, dy = y - sh.y0, b = dx * tx + dy * ty, c = dx * dx + dy * dy;
    if (sh.rmax === undefined) { let sa = 0; for (let i = 0; i < sh.M; i++) sa += Math.abs(sh.a[i]) + Math.abs(sh.b[i]); sh.rmax = sh.R * (1 + sa) + TAIL_U * sh.edge; }
    if (b * b - (c - sh.rmax * sh.rmax) <= 0) return 0;
    const disc0 = b * b - (c - sh.R * sh.R);
    if (!(disc0 > (0.1 * sh.R) ** 2)) return NaN;
    const sq = Math.sqrt(disc0);
    let s1 = -b - sq, s2 = -b + sq, e1 = 0, e2 = 0, w1 = 0, w2 = 0;
    for (let root = 0; root < 2; root++) {
      let s_ = root ? s2 : s1;
      for (let it = 0; it <= 6; it++) {
        const px = dx + s_ * tx, py = dy + s_ * ty, rho = Math.hypot(px, py);
        radius01(sh, px, py, rho);
        const fp = _rr1 * (px * ty - py * tx) / (rho * rho) - (px * tx + py * ty) / rho;
        if (it === 6) {
          const e = Math.abs(_rr - rho), w = 7.5 * sh.edge / Math.max(Math.abs(fp), 1e-12);
          if (root) { s2 = s_; e2 = e; w2 = w; } else { s1 = s_; e1 = e; w1 = w; }
          break;
        }
        s_ -= (_rr - rho) / (Math.abs(fp) > 1e-12 ? fp : 1e-12);
      }
    }
    if (!(e1 < 1e-9 * sh.R && e2 < 1e-9 * sh.R && s1 + w1 < s2 - w2 && s1 - w1 > 0 && s2 + w2 < ln)) return NaN;
    let tot = sh.dn * ((s2 - w2) - (s1 + w1));
    for (let root = 0; root < 2; root++) {
      const sc = root ? s2 : s1, wc = root ? w2 : w1;
      let acc = 0;
      for (let q = 0; q < GL.x.length; q++) { const s = sc + wc * GL.x[q]; acc += GL.w[q] * shapeValue(sh, x + s * tx, y + s * ty); }
      tot += acc * wc;
    }
    return tot;
  }

  // Radon table of a dot about its centre: T[i][j] = ∫Δn along direction α_i = π i/nA, offset p_j (see Shape.radon)
  function radonTable(sh) {
    if (sh.rmax === undefined) chordIntegral(sh, sh.x0 + 10 * sh.bound, sh.y0, 1, 0, 1e-9);
    const L = sh.bound, nP = Math.ceil(2 * L / (sh.edge / 3)) + 1;
    const nA = Math.ceil(Math.PI / Math.min(0.012 * 6 / Math.max(1, sh.M), 0.05)), dp = 2 * L / (nP - 1);
    const T = new Float64Array(nA * nP);
    for (let i = 0; i < nA; i++) {
      const al = Math.PI * i / nA, tx = Math.cos(al), ty = Math.sin(al), nx = -ty, ny = tx;
      for (let j = 0; j < nP; j++) {
        const p = -L + j * dp, x = sh.x0 + p * nx - 1.001 * L * tx, y = sh.y0 + p * ny - 1.001 * L * ty;
        let v = chordIntegral(sh, x, y, tx, ty, 2.002 * L);
        if (Number.isNaN(v)) {
          const dx = x - sh.x0, dy = y - sh.y0, b = dx * tx + dy * ty, disc = Math.max(b * b - (dx * dx + dy * dy - L * L), 0);
          const s1 = -b - Math.sqrt(disc), s2 = -b + Math.sqrt(disc), h = (s2 - s1) / 2, mid = (s1 + s2) / 2;
          let sum = 0;
          for (let q = 0; q < 32; q++) { const s = mid + h * GL32.x[q]; sum += GL32.w[q] * shapeValue(sh, x + s * tx, y + s * ty); }
          v = sum * h;
        }
        T[i * nP + j] = v;
      }
    }
    return { T, nA, nP, p0: -L, dp };
  }
  function radonLookup(tab, cx, cy, x, y, tx, ty) {
    let al = Math.atan2(ty, tx), sgn = 1;
    if (al < 0) { al += Math.PI; sgn = -1; }
    if (al >= Math.PI) al -= Math.PI;
    const p = sgn * ((x - cx) * (-ty) + (y - cy) * tx);
    if (!(p > tab.p0 && p < -tab.p0)) return 0;
    const fa = al / Math.PI * tab.nA, ia = Math.floor(fa), wa = fa - ia;
    const fp = (p - tab.p0) / tab.dp, ip = Math.min(Math.max(Math.floor(fp), 0), tab.nP - 2), wp = Math.min(Math.max(fp - ip, 0), 1);
    const T = tab.T, nP = tab.nP;
    const v0 = T[(ia % tab.nA) * nP + ip] * (1 - wp) + T[(ia % tab.nA) * nP + ip + 1] * wp;
    let v1;
    if (ia + 1 >= tab.nA) { const jp = nP - 1 - ip - 1, w = 1 - wp; v1 = T[jp] * (1 - w) + T[jp + 1] * w; }   // α + π: p → −p
    else v1 = T[(ia + 1) * nP + ip] * (1 - wp) + T[(ia + 1) * nP + ip + 1] * wp;
    return (1 - wa) * v0 + wa * v1;
  }

  // ---------- phase-screen model ----------
  class PhaseScreen {
    constructor(cell, opt = {}) {
      this.cell = cell; this.opt = opt;
      this.inputs = cell.ports.map((p, i) => isIn(p) ? i : -1).filter(i => i >= 0);
      this.outputs = cell.ports.map((p, i) => isOut(p) ? i : -1).filter(i => i >= 0);
      const prune = opt.prune ?? 0.02;
      this.tables = this.inputs.map(port => {
        const tr = trace(cell, port, [], Object.assign({}, opt, { mode: 'none', recordChords: true }));
        let keep = null;
        if (tr.model === 'fga' && prune) {
          const w = tr.exits.map(e => recordWeight(tr, e)); let wm = 0; for (const v of w) if (v > wm) wm = v;
          keep = w.map(v => v > prune * wm);
        }
        const ops = {}; for (const o of this.outputs) ops[o] = portOperator(tr, o, keep);
        // flat tables: chords of every ray up to its last kept record (Float32), exit → (ray, chords travelled)
        const nR = tr.chords.length, need = new Int32Array(nR);
        tr.exits.forEach((e, i) => { if (!keep || keep[i]) need[e.ray] = Math.max(need[e.ray], e.nch); });
        const start = new Int32Array(nR + 1);
        for (let r = 0; r < nR; r++) start[r + 1] = start[r] + need[r];
        const C = new Float32Array(start[nR] * 5);
        tr.chords.forEach((ch, r) => { for (let k = 0; k < need[r]; k++) for (let q = 0; q < 5; q++) C[(start[r] + k) * 5 + q] = ch[k * 5 + q]; });
        const nE = tr.exits.length, eRay = new Int32Array(nE), eNch = new Int32Array(nE);
        tr.exits.forEach((e, i) => { eRay[i] = e.ray; eNch[i] = (!keep || keep[i]) ? e.nch : 0; });
        const t = { port, ops, nR, start, C, eRay, eNch, pref: new Float64Array(start[nR] + nR), nExits: nE,
          nLaunched: tr.nLaunched, lost: tr.lost, model: tr.model };
        tr.exits = null; tr.chords = null;
        return t;
      });
    }
    deltaL(k, shapes, dnGlobal = 0) {   // ΔL per exit record: sum over the chords the ray travelled before that exit
      const t = this.tables[k], C = t.C, start = t.start, pref = t.pref;
      pref.fill(0);
      for (const sh of shapes) {
        const R2 = sh.bound * sh.bound, tab = radonTable(sh), x0 = sh.x0, y0 = sh.y0;
        for (let r = 0; r < t.nR; r++) {
          const off = start[r] + r;
          for (let i = start[r], k = 1; i < start[r + 1]; i++, k++) {
            const o = i * 5, x = C[o], y = C[o + 1], tx = C[o + 2], ty = C[o + 3], ln = C[o + 4];
            const dx = x - x0, dy = y - y0, b = dx * tx + dy * ty;
            const disc = b * b - (dx * dx + dy * dy - R2);
            if (disc <= 0) continue;
            const sq = Math.sqrt(disc);
            if (Math.min(-b + sq, ln) <= Math.max(-b - sq, 0)) continue;
            pref[off + k] += radonLookup(tab, x0, y0, x, y, tx, ty);
          }
        }
      }
      for (let r = 0; r < t.nR; r++) {
        const off = start[r] + r;
        if (dnGlobal) for (let i = start[r], k = 1; i < start[r + 1]; i++, k++) pref[off + k] += dnGlobal * C[i * 5 + 4];
        for (let k = 1; k <= start[r + 1] - start[r]; k++) pref[off + k] += pref[off + k - 1];
      }
      const out = new Float64Array(t.nExits);
      for (let e = 0; e < t.nExits; e++) { const r = t.eRay[e]; out[e] = pref[start[r] + r + t.eNch[e]]; }
      return out;
    }
    fields(shapes, dnGlobal = 0) {   // {"in,out": {re, im}}
      const out = {}, k0 = TAU / this.cell.wavelength;
      this.tables.forEach((t, k) => {
        let ph = null;
        if ((shapes && shapes.length) || dnGlobal) { ph = this.deltaL(k, shapes || [], dnGlobal); for (let i = 0; i < ph.length; i++) ph[i] *= k0; }
        for (const o of this.outputs) out[t.port + ',' + o] = t.ops[o].apply(ph);
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

  const NGRC = { radonTable, radonLookup, erf, zfun, launchFga, recordWeight, portOperator, chordIntegral, aperturePoints, makeShape, shapeEval, shapeValue, pertEval, safeStep, portIndex, launch, trace, detectorPoints,
    beamletMatrix, detectorFields, PhaseScreen, curvedFields, chdShape, chdContour, rng, randomShape,
    stackRelative, ridgeFitPredict, ridgeCV, r2, isIn, isOut };
  root.NGRC = NGRC;
  if (typeof module !== 'undefined' && module.exports) module.exports = NGRC;
})(typeof globalThis !== 'undefined' ? globalThis : this);
