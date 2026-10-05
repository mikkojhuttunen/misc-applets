/*
Shared waveguide mode solver for the photonics applets.
WGCORE(): materials, 1D slab solver, effective index method, 2D semi-vectorial
finite-difference solver. WGWORKER(): Web Worker entry used by the mode explorer.
Both are plain functions so a page can rebuild them inside a Blob worker with
Function.prototype.toString().
*/
/* ======================================================================
   WGCORE: materials, 1D slab solver, EIM, 2D semi-vectorial FD solver.
   Self-contained so it can also run inside a Web Worker.
   ====================================================================== */
function WGCORE() {
  'use strict';
  var TWO_PI = 2 * Math.PI;
  function sq(v) { return v * v; }

  /* ---------- materials: n² = A + Σ B λ²/(λ² − C), λ and √C in µm ---------- */
  var MAT = {
    const: { name: 'Constant index' },
    air: { name: 'Air', f: function () { return 1.0; }, r: [0.2, 20] },
    sio2: { name: 'SiO₂, fused silica', s: [1, 0.6961663, sq(0.0684043), 0.4079426, sq(0.1162414), 0.8974794, sq(9.896161)], r: [0.21, 6.7] },
    gesio2: { name: 'Ge:SiO₂ core (+0.75 %)', base: 'sio2', scale: 1.0075, r: [0.21, 6.7] },
    pmma: { name: 'PMMA', s: [1, 1.1819, 0.011313, 0, 0, 0, 0], r: [0.437, 1.052] },
    su8: { name: 'SU-8 (Cauchy)', cauchy: [1.566, 0.00796, 0.00014], r: [0.32, 1.6] },
    si3n4: { name: 'Si₃N₄', s: [1, 3.0249, sq(0.1353406), 40314, sq(1239.842), 0, 0], r: [0.31, 5.5] },
    ln_e: { name: 'LiNbO₃, n_e', s: [1, 2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08], r: [0.4, 5] },
    ln_o: { name: 'LiNbO₃, n_o', s: [1, 2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6], r: [0.4, 5] },
    lt_e: { name: 'LiTaO₃, n_e (approx.)', s: [1, 3.49474, 0.02789, 0, 0, 1.5, 150], r: [0.5, 2.2] },
    lt_o: { name: 'LiTaO₃, n_o (approx.)', s: [1, 3.47785, 0.02797, 0, 0, 1.5, 150], r: [0.5, 2.2] },
    al2o3: { name: 'Al₂O₃, amorphous film', cauchy: [1.646, 0.00962, 0], r: [0.4, 2] },
    eral2o3: { name: 'Er:Al₂O₃ (amorphous)', cauchy: [1.646, 0.00962, 0], r: [0.4, 2] },
    sapph: { name: 'Al₂O₃, sapphire n_o', s: [1, 1.4313493, 0.0726631 * 0.0726631, 0.65054713, 0.1193242 * 0.1193242, 5.3414021, 18.028251 * 18.028251], r: [0.2, 5.5] },
    gaas: { name: 'GaAs (approx.)', s: [3.5, 7.4969, sq(0.4082), 1.9347, sq(37.17), 0, 0], r: [0.89, 4.1] },
    si: { name: 'Si', s: [1, 10.6684293, sq(0.301516485), 0.0030434748, sq(1.13475115), 1.54133408, sq(1104)], r: [1.36, 11] },
    custom: { name: 'Custom Sellmeier' }
  };
  function sell(s, lam) {
    var l2 = lam * lam, n2 = s[0];
    for (var k = 1; k < 7; k += 2) {
      if (!s[k]) continue;
      var den = l2 - s[k + 1];
      if (Math.abs(den) < 1e-9) return NaN;
      n2 += s[k] * l2 / den;
    }
    return n2 > 0 ? Math.sqrt(n2) : NaN;
  }
  function nMat(spec, lam) {
    if (!spec) return NaN;
    if (spec.id === 'const') return spec.n;
    if (spec.id === 'custom') return sell(spec.cust, lam);
    var m = MAT[spec.id];
    if (!m) return NaN;
    if (m.f) return m.f(lam);
    if (m.cauchy) { var c = m.cauchy, l2 = lam * lam; return c[0] + c[1] / l2 + c[2] / (l2 * l2); }
    if (m.base) return m.scale * sell(MAT[m.base].s, lam);
    return sell(m.s, lam);
  }
  function indices(G, lam) {
    return { sup: nMat(G.mats.sup, lam), strip: nMat(G.mats.strip, lam), film: nMat(G.mats.film, lam), sub: nMat(G.mats.sub, lam) };
  }
  function validIx(ix) {
    return [ix.sup, ix.strip, ix.film, ix.sub].every(function (v) { return isFinite(v) && v > 0.5 && v < 8; });
  }

  /* ---------- geometry ---------- */
  function confined(G, ix) {
    if (G.geo === 'strip') return G.h > 0.005 && Math.abs(ix.strip - ix.sup) > 1e-6;
    if (G.geo === 'rib') return G.tout < G.t - 0.005;
    return true;
  }
  function stacks(G, ix) {
    var fl = Math.max(ix.sub, ix.sup), film = { n: ix.film, d: G.t };
    switch (G.geo) {
      case 'strip': return { inS: G.h > 0.005 ? [film, { n: ix.strip, d: G.h }] : [film], outS: [film], y0: 0, fb: fl };
      case 'rib': return { inS: [film], outS: G.tout > 0.005 ? [{ n: ix.film, d: Math.min(G.tout, G.t) }] : [], y0: 0, fb: fl };
      case 'ridge': return { inS: [film], outS: [], y0: 0, fb: fl };
      default: return { inS: G.d > 0.001 ? [film, { n: ix.sub, d: G.d }] : [film], outS: [], y0: -(G.d + G.t), fb: ix.sub };
    }
  }
  /* region codes: 0 substrate, 1 film/core, 2 strip, 3 superstrate */
  function regionAt(G, x, y) {
    var a = Math.abs(x) < G.w / 2;
    switch (G.geo) {
      case 'strip': if (y < 0) return 0; if (y < G.t) return 1; if (a && G.h > 0.005 && y < G.t + G.h) return 2; return 3;
      case 'rib': if (y < 0) return 0; if (y < G.tout) return 1; if (a && y < G.t) return 1; return 3;
      case 'ridge': if (y < 0) return 0; if (a && y < G.t) return 1; return 3;
      default: if (y >= 0) return 3; if (a && y >= -(G.d + G.t) && y < -G.d) return 1; return 0;
    }
  }
  function regIndex(ix, r) { return r === 0 ? ix.sub : r === 1 ? ix.film : r === 2 ? ix.strip : ix.sup; }

  /* ---------- 1D multilayer slab (transfer matrix, exact) ---------- */
  function pw(pol, n) { return pol === 'TE' ? 1 : 1 / (n * n); }
  function step(f, g, n, d, k0, N, p) {
    var q2 = k0 * k0 * (n * n - N * N), k, c, s;
    if (q2 > 1e-12) { k = Math.sqrt(q2); c = Math.cos(k * d); s = Math.sin(k * d); return [f * c + g * s / (p * k), -f * p * k * s + g * c]; }
    if (q2 < -1e-12) { k = Math.sqrt(-q2); c = Math.cosh(k * d); s = Math.sinh(k * d); return [f * c + g * s / (p * k), f * p * k * s + g * c]; }
    return [f + g * d / p, g];
  }
  function charF(nS, layers, nC, k0, pol, N) {
    var f = 1, g = pw(pol, nS) * k0 * Math.sqrt(N * N - nS * nS);
    for (var i = 0; i < layers.length; i++) {
      var r = step(f, g, layers[i].n, layers[i].d, k0, N, pw(pol, layers[i].n)); f = r[0]; g = r[1];
      var sc = Math.max(Math.abs(f), Math.abs(g) / k0);
      if (sc > 1e6) { f /= sc; g /= sc; }
    }
    return g + pw(pol, nC) * k0 * Math.sqrt(N * N - nC * nC) * f;
  }
  function slabModes(nS, layers, nC, k0, pol) {
    var lo = Math.max(nS, nC), hi = -1, out = [];
    layers.forEach(function (l) { if (l.d > 0) hi = Math.max(hi, l.n); });
    if (hi <= lo + 1e-8) return out;
    var M = 600, a = lo + (hi - lo) * 1e-7, b = hi - (hi - lo) * 1e-7;
    var pN = a, pF = charF(nS, layers, nC, k0, pol, a);
    for (var i = 1; i <= M; i++) {
      var N = a + (b - a) * i / M, F = charF(nS, layers, nC, k0, pol, N);
      if (pF === 0 || pF * F < 0) {
        var x0 = pN, x1 = N, f0 = pF;
        for (var it = 0; it < 56; it++) {
          var xm = 0.5 * (x0 + x1), fm = charF(nS, layers, nC, k0, pol, xm);
          if (f0 * fm <= 0) x1 = xm; else { x0 = xm; f0 = fm; }
        }
        out.push(0.5 * (x0 + x1));
      }
      pN = N; pF = F;
    }
    return out.sort(function (u, v) { return v - u; });
  }
  /* tangential field (E for TE, H for TM) of a slab mode at positions xs (x = 0 at the bottom of layers) */
  function profile(nS, layers, nC, k0, pol, N, xs) {
    var gs = k0 * Math.sqrt(N * N - nS * nS), gc = k0 * Math.sqrt(N * N - nC * nC);
    var st = [], f = 1, g = pw(pol, nS) * gs, x0 = 0;
    layers.forEach(function (L) {
      st.push({ x0: x0, f: f, g: g, L: L });
      var r = step(f, g, L.n, L.d, k0, N, pw(pol, L.n)); f = r[0]; g = r[1]; x0 += L.d;
    });
    var top = x0, fTop = f;
    return xs.map(function (x) {
      if (x < 0) return Math.exp(gs * x);
      if (x >= top) return fTop * Math.exp(-gc * (x - top));
      for (var i = 0; i < st.length; i++) if (x < st[i].x0 + st[i].L.d) return step(st[i].f, st[i].g, st[i].L.n, x - st[i].x0, k0, N, pw(pol, st[i].L.n))[0];
      return fTop;
    });
  }
  function nInStack(nS, layers, nC, x) {
    if (x < 0) return nS;
    var acc = 0;
    for (var i = 0; i < layers.length; i++) { acc += layers[i].d; if (x < acc) return layers[i].n; }
    return nC;
  }
  function normalise(v) {
    var m = 0, s = 1;
    for (var i = 0; i < v.length; i++) if (Math.abs(v[i]) > m) { m = Math.abs(v[i]); s = v[i] < 0 ? -1 : 1; }
    m = m || 1;
    return v.map(function (y) { return s * y / m; });
  }

  /* ---------- effective index method ---------- */
  function solveEIM(G, lam, pols) {
    var ix = indices(G, lam);
    if (!validIx(ix)) return { ok: false, ix: ix, modes: [], outN: {} };
    var k0 = TWO_PI / lam, st = stacks(G, ix), fl = Math.max(ix.sub, ix.sup), conf = confined(G, ix), modes = [], outN = { TE: null, TM: null };
    ['TE', 'TM'].forEach(function (pol) {
      if (!pols[pol]) return;
      var vin = slabModes(ix.sub, st.inS, ix.sup, k0, pol);
      if (!conf) { vin.forEach(function (N, m) { modes.push({ key: pol + m, pol: pol, m: m, n: null, N: N, Nin: N, Nout: N, leaky: false }); }); return; }
      var vout = st.outS.length ? slabModes(ix.sub, st.outS, ix.sup, k0, pol) : [];
      outN[pol] = vout.length ? vout[0] : null;
      var latPol = pol === 'TE' ? 'TM' : 'TE';
      vin.forEach(function (Nin, m) {
        var Nout = m < vout.length ? vout[m] : st.fb;
        if (Nin <= Nout + 1e-9) return;
        slabModes(Nout, [{ n: Nin, d: G.w }], Nout, k0, latPol).forEach(function (N, n) {
          if (N > fl + 1e-9) modes.push({ key: pol + m + '_' + n, pol: pol, m: m, n: n, N: N, Nin: Nin, Nout: Nout, leaky: outN[pol] !== null && N < outN[pol] });
        });
      });
    });
    modes.sort(function (a, b) { return b.N - a.N; });
    return { ok: true, ix: ix, modes: modes, conf: conf, floor: fl, outN: outN };
  }
  /* dominant E component of an EIM mode: vertical factor V(y) and lateral factor L(x) */
  function eimVert(G, ix, md, lam, ys) {
    var st = stacks(G, ix), k0 = TWO_PI / lam;
    var yy = ys.map(function (y) { return y - st.y0; });
    var f = profile(ix.sub, st.inS, ix.sup, k0, md.pol, md.Nin, yy);
    if (md.pol === 'TM') f = f.map(function (v, i) { return v / sq(nInStack(ix.sub, st.inS, ix.sup, yy[i])); });
    return normalise(f);
  }
  function eimLat(G, md, lam, xs) {
    if (md.n === null) return xs.map(function () { return 1; });
    var latPol = md.pol === 'TE' ? 'TM' : 'TE', w2 = G.w / 2;
    var f = profile(md.Nout, [{ n: md.Nin, d: G.w }], md.Nout, TWO_PI / lam, latPol, md.N, xs.map(function (x) { return x + w2; }));
    if (latPol === 'TM') f = f.map(function (v, i) { return v / sq(Math.abs(xs[i]) < w2 ? md.Nin : md.Nout); });
    return normalise(f);
  }

  /* ---------- non-uniform 2D grid ---------- */
  function coreBox(G) {
    switch (G.geo) {
      case 'strip': return { y0: 0, y1: G.t + (G.h > 0.005 ? G.h : 0), yI: G.h > 0.005 ? [0, G.t, G.t + G.h] : [0, G.t] };
      case 'rib': return { y0: 0, y1: G.t, yI: (G.tout > 0.005 && G.tout < G.t - 0.005) ? [0, G.tout, G.t] : [0, G.t] };
      case 'ridge': return { y0: 0, y1: G.t, yI: [0, G.t] };
      default: return { y0: -(G.d + G.t), y1: -G.d, yI: G.d > 0.005 ? [-(G.d + G.t), -G.d, 0] : [-G.t, 0] };
    }
  }
  function axisEdges(pts, c0, c1, h, hmax, r) {
    var edges = [c0], inner = pts.filter(function (p) { return p > c0 + 1e-9 && p < c1 - 1e-9; });
    var cuts = [c0].concat(inner, [c1]);
    for (var k = 0; k < cuts.length - 1; k++) {
      var L = cuts[k + 1] - cuts[k], n = Math.max(3, Math.ceil(L / h - 1e-9));
      for (var i = 1; i <= n; i++) edges.push(cuts[k] + L * i / n);
    }
    function grade(seq, from, dir) {
      var hc = h, prev = from, outE = [];
      seq.forEach(function (p) {
        var L = Math.abs(p - prev), s = [], tot = 0;
        while (tot < L - 1e-12) { hc = Math.min(hc * r, hmax); s.push(hc); tot += hc; }
        if (s.length > 1 && tot - L > 0.5 * s[s.length - 1]) tot -= s.pop();
        var sc = L / tot, acc = 0;
        s.forEach(function (v, i) { acc += v * sc; outE.push(i === s.length - 1 ? p : prev + dir * acc); });
        hc = s[s.length - 1] * sc; prev = p;
      });
      return outE;
    }
    var up = pts.filter(function (p) { return p > c1 + 1e-9; }).sort(function (a, b) { return a - b; });
    var dn = pts.filter(function (p) { return p < c0 - 1e-9; }).sort(function (a, b) { return b - a; });
    var e = grade(dn, c0, -1).reverse().concat(edges, grade(up, c1, 1));
    return e;
  }
  var RES = { coarse: [14, 12], normal: [24, 20], fine: [38, 32] };
  function buildGrid(G, ix, lam, res, eim) {
    var R = RES[res] || RES.normal, cb = coreBox(G), w2 = G.w / 2, fl = Math.max(ix.sub, ix.sup);
    function dl(N, nc) { var q = N * N - nc * nc; return q > 1e-8 ? lam / (TWO_PI * Math.sqrt(q)) : 30; }
    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    var m0 = eim && eim.modes.length ? eim.modes[0] : null;
    var Nv = m0 ? m0.Nin : fl + 0.5 * (ix.film - fl);
    var Nl = m0 ? m0.N : fl + 0.3 * (ix.film - fl), Nlo = m0 && m0.n !== null ? m0.Nout : fl;
    var dSub = dl(Nv, ix.sub), dSup = dl(Nv, ix.sup), dLat = dl(Nl, Nlo);
    var yLo = cb.y0 - clamp(6 * dSub, 0.3 * lam, 40), yHi;
    if (G.geo === 'buried') {
      yHi = cb.y1 + clamp(6 * dSub, 0.3 * lam, 40);
      if (yHi > 0) yHi = Math.max(clamp(5 * dSup, 0.2 * lam, 40), 0.05);
    } else yHi = cb.y1 + clamp(6 * dSup, 0.3 * lam, 40);
    var xM = clamp(6 * dLat, 0.4 * lam, 40);
    var hx = G.w / R[0], hy = (cb.y1 - cb.y0) / R[1];
    var yPts = cb.yI.filter(function (y) { return y > yLo && y < yHi; }).concat([yLo, yHi]);
    var xe = axisEdges([-w2 - xM, -w2, w2, w2 + xM], -w2, w2, hx, Math.max(hx, lam / 4), 1.15);
    var ye = axisEdges(yPts, cb.y0, cb.y1, hy, Math.max(hy, lam / 4), 1.15);
    var Nx = xe.length - 1, Ny = ye.length - 1;
    var xc = new Float64Array(Nx), dx = new Float64Array(Nx), yc = new Float64Array(Ny), dy = new Float64Array(Ny);
    for (var i = 0; i < Nx; i++) { xc[i] = 0.5 * (xe[i] + xe[i + 1]); dx[i] = xe[i + 1] - xe[i]; }
    for (var j = 0; j < Ny; j++) { yc[j] = 0.5 * (ye[j] + ye[j + 1]); dy[j] = ye[j + 1] - ye[j]; }
    var reg = new Uint8Array(Nx * Ny);
    for (j = 0; j < Ny; j++) for (i = 0; i < Nx; i++) reg[i + Nx * j] = regionAt(G, xc[i], yc[j]);
    return { Nx: Nx, Ny: Ny, xc: xc, yc: yc, dx: dx, dy: dy, reg: reg, cb: cb };
  }

  /* ---------- dense linear algebra for the small Arnoldi problem ---------- */
  function hqr(a, n) {
    var wr = new Float64Array(n + 1), wi = new Float64Array(n + 1);
    var nn, m, l, k, j, its, i, mmin, z = 0, y, x, w, v, u, t, s, r = 0, q = 0, p = 0, anorm = 0;
    for (i = 1; i <= n; i++) for (j = Math.max(i - 1, 1); j <= n; j++) anorm += Math.abs(a[i][j]);
    nn = n; t = 0;
    while (nn >= 1) {
      its = 0;
      do {
        for (l = nn; l >= 2; l--) {
          s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]);
          if (s === 0) s = anorm;
          if (Math.abs(a[l][l - 1]) + s === s) { a[l][l - 1] = 0; break; }
        }
        x = a[nn][nn];
        if (l === nn) { wr[nn] = x + t; wi[nn--] = 0; }
        else {
          y = a[nn - 1][nn - 1]; w = a[nn][nn - 1] * a[nn - 1][nn];
          if (l === nn - 1) {
            p = 0.5 * (y - x); q = p * p + w; z = Math.sqrt(Math.abs(q)); x += t;
            if (q >= 0) {
              z = p + (p >= 0 ? z : -z);
              wr[nn - 1] = wr[nn] = x + z;
              if (z) wr[nn] = x - w / z;
              wi[nn - 1] = wi[nn] = 0;
            } else { wr[nn - 1] = wr[nn] = x + p; wi[nn - 1] = -(wi[nn] = z); }
            nn -= 2;
          } else {
            if (its === 60) throw new Error('eigenvalue iteration did not converge');
            if (its === 10 || its === 20) {
              t += x;
              for (i = 1; i <= nn; i++) a[i][i] -= x;
              s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]);
              y = x = 0.75 * s; w = -0.4375 * s * s;
            }
            ++its;
            for (m = nn - 2; m >= l; m--) {
              z = a[m][m]; r = x - z; s = y - z;
              p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
              q = a[m + 1][m + 1] - z - r - s;
              r = a[m + 2][m + 1];
              s = Math.abs(p) + Math.abs(q) + Math.abs(r);
              p /= s; q /= s; r /= s;
              if (m === l) break;
              u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r));
              v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]));
              if (u + v === v) break;
            }
            for (i = m + 2; i <= nn; i++) { a[i][i - 2] = 0; if (i !== m + 2) a[i][i - 3] = 0; }
            for (k = m; k <= nn - 1; k++) {
              if (k !== m) {
                p = a[k][k - 1]; q = a[k + 1][k - 1]; r = 0;
                if (k !== nn - 1) r = a[k + 2][k - 1];
                if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) { p /= x; q /= x; r /= x; }
              }
              s = Math.sqrt(p * p + q * q + r * r); if (p < 0) s = -s;
              if (s !== 0) {
                if (k === m) { if (l !== m) a[k][k - 1] = -a[k][k - 1]; }
                else a[k][k - 1] = -s * x;
                p += s; x = p / s; y = q / s; z = r / s; q /= p; r /= p;
                for (j = k; j <= nn; j++) {
                  p = a[k][j] + q * a[k + 1][j];
                  if (k !== nn - 1) { p += r * a[k + 2][j]; a[k + 2][j] -= p * z; }
                  a[k + 1][j] -= p * y; a[k][j] -= p * x;
                }
                mmin = nn < k + 3 ? nn : k + 3;
                for (i = l; i <= mmin; i++) {
                  p = x * a[i][k] + y * a[i][k + 1];
                  if (k !== nn - 1) { p += z * a[i][k + 2]; a[i][k + 2] -= p * r; }
                  a[i][k + 1] -= p * q; a[i][k] -= p;
                }
              }
            }
          }
        }
      } while (l < nn - 1);
    }
    return { wr: wr, wi: wi };
  }
  function denseSolve(M, b) {
    var n = b.length, A = M.map(function (r) { return Float64Array.from(r); }), x = Float64Array.from(b), i, j, k;
    for (k = 0; k < n; k++) {
      var piv = k, best = Math.abs(A[k][k]);
      for (i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > best) { best = Math.abs(A[i][k]); piv = i; }
      if (piv !== k) { var tr = A[k]; A[k] = A[piv]; A[piv] = tr; var tb = x[k]; x[k] = x[piv]; x[piv] = tb; }
      if (A[k][k] === 0) A[k][k] = 1e-300;
      for (i = k + 1; i < n; i++) {
        var f = A[i][k] / A[k][k];
        if (!f) continue;
        for (j = k; j < n; j++) A[i][j] -= f * A[k][j];
        x[i] -= f * x[k];
      }
    }
    for (i = n - 1; i >= 0; i--) { var s = x[i]; for (j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; }
    return x;
  }

  /* ---------- semi-vectorial FD eigen-solver ---------- */
  function fdSolve(grid, ix, lam, pol, nWant) {
    var k0 = TWO_PI / lam, Nx = grid.Nx, Ny = grid.Ny, n = Nx * Ny, dx = grid.dx, dy = grid.dy;
    var N2 = new Float64Array(n), n2max = 0, i, j, g;
    for (g = 0; g < n; g++) { N2[g] = sq(regIndex(ix, grid.reg[g])); if (N2[g] > n2max) n2max = N2[g]; }
    var sigma = k0 * k0 * n2max * 1.0001;
    var xMaj = Nx <= Ny, bw = xMaj ? Nx : Ny, W = 2 * bw + 1;
    var id = xMaj ? function (a, b) { return a + Nx * b; } : function (a, b) { return b + Ny * a; };
    var A = new Float64Array(n * W);
    function add(r, c, v) { A[r * W + c - r + bw] += v; }
    var wx = pol === 'TE', wy = pol === 'TM';
    for (j = 0; j < Ny; j++) for (i = 0; i < Nx; i++) {
      var r = id(i, j), n2c = N2[i + Nx * j], s, ii, jj, hh, cf, nn2, a;
      for (s = -1; s <= 1; s += 2) {
        ii = i + s;
        if (ii < 0 || ii >= Nx) { add(r, r, -1 / (dx[i] * dx[i])); continue; }
        hh = 0.5 * (dx[i] + dx[ii]); cf = 1 / (hh * dx[i]); nn2 = N2[ii + Nx * j];
        if (wx) { a = 2 / (n2c + nn2); add(r, id(ii, j), cf * a * nn2); add(r, r, -cf * a * n2c); }
        else { add(r, id(ii, j), cf); add(r, r, -cf); }
      }
      for (s = -1; s <= 1; s += 2) {
        jj = j + s;
        if (jj < 0 || jj >= Ny) { add(r, r, -1 / (dy[j] * dy[j])); continue; }
        hh = 0.5 * (dy[j] + dy[jj]); cf = 1 / (hh * dy[j]); nn2 = N2[i + Nx * jj];
        if (wy) { a = 2 / (n2c + nn2); add(r, id(i, jj), cf * a * nn2); add(r, r, -cf * a * n2c); }
        else { add(r, id(i, jj), cf); add(r, r, -cf); }
      }
      add(r, r, k0 * k0 * n2c - sigma);
    }
    /* banded LU without pivoting (matrix is diagonally dominant after the shift) */
    var k, rr, cc, cmax, f, offR, offK;
    for (k = 0; k < n; k++) {
      var pk = A[k * W + bw], rmax = Math.min(n - 1, k + bw);
      cmax = rmax; offK = k * W - k + bw;
      for (rr = k + 1; rr <= rmax; rr++) {
        offR = rr * W - rr + bw;
        f = A[offR + k];
        if (f === 0) continue;
        f /= pk; A[offR + k] = f;
        for (cc = k + 1; cc <= cmax; cc++) A[offR + cc] -= f * A[offK + cc];
      }
    }
    function solve(b) {
      var x = Float64Array.from(b), q, c, sum, off;
      for (q = 0; q < n; q++) { off = q * W - q + bw; sum = x[q]; for (c = Math.max(0, q - bw); c < q; c++) sum -= A[off + c] * x[c]; x[q] = sum; }
      for (q = n - 1; q >= 0; q--) { off = q * W - q + bw; sum = x[q]; var ce = Math.min(n - 1, q + bw); for (c = q + 1; c <= ce; c++) sum -= A[off + c] * x[c]; x[q] = sum / A[off + q]; }
      return x;
    }
    /* start vector: seeded noise under a Gaussian envelope centred on the core */
    var seed = 12345, cx = 0, cy = 0.5 * (grid.cb.y0 + grid.cb.y1), sx = 0, sy = 0.5 * (grid.cb.y1 - grid.cb.y0) + 0.3;
    for (i = 0; i < Nx; i++) sx = Math.max(sx, Math.abs(grid.xc[i]));
    sx = Math.max(0.3, sx * 0.35);
    var v0 = new Float64Array(n);
    for (j = 0; j < Ny; j++) for (i = 0; i < Nx; i++) {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      v0[id(i, j)] = (0.3 + seed / 2147483648) * Math.exp(-sq((grid.xc[i] - cx) / sx) - sq((grid.yc[j] - cy) / sy));
    }
    var m = Math.min(n - 2, Math.max(30, Math.min(100, 3 * nWant + 24)));
    /* Arnoldi with full re-orthogonalisation */
    var V = [], H = [], q, nv = 0;
    for (q = 0; q <= m; q++) H.push(new Float64Array(m));
    for (g = 0; g < n; g++) nv += v0[g] * v0[g];
    nv = Math.sqrt(nv); V.push(v0.map(function (v) { return v / nv; }));
    var mEff = m;
    for (k = 0; k < m; k++) {
      var wv = solve(V[k]);
      for (var pass = 0; pass < 2; pass++) for (q = 0; q <= k; q++) {
        var h = 0, Vq = V[q];
        for (g = 0; g < n; g++) h += Vq[g] * wv[g];
        H[q][k] += h;
        for (g = 0; g < n; g++) wv[g] -= h * Vq[g];
      }
      var hn = 0;
      for (g = 0; g < n; g++) hn += wv[g] * wv[g];
      hn = Math.sqrt(hn); H[k + 1][k] = hn;
      if (hn < 1e-13) { mEff = k + 1; break; }
      for (g = 0; g < n; g++) wv[g] /= hn;
      V.push(wv);
    }
    var Hs = [];
    for (q = 0; q < mEff; q++) Hs.push(Float64Array.from(H[q].subarray(0, mEff)));
    var a1 = [new Float64Array(mEff + 1)];
    for (q = 0; q < mEff; q++) { var row = new Float64Array(mEff + 1); for (k = 0; k < mEff; k++) row[k + 1] = Hs[q][k]; a1.push(row); }
    var ev = hqr(a1, mEff), cand = [];
    for (q = 1; q <= mEff; q++) if (Math.abs(ev.wi[q]) <= 1e-9 * Math.abs(ev.wr[q]) && ev.wr[q] < 0) cand.push(ev.wr[q]);
    cand.sort(function (a, b) { return a - b; });
    var fl = Math.max(ix.sub, ix.sup), out = [];
    for (var c = 0; c < cand.length && out.length < nWant; c++) {
      var mu = cand[c], beta2 = sigma + 1 / mu;
      if (beta2 <= 0) continue;
      var Neff = Math.sqrt(beta2) / k0;
      if (Neff <= fl + 1e-6) break;
      var Ms = Hs.map(function (rw, ri) { var z = Float64Array.from(rw); z[ri] -= mu * (1 + 1e-11); return z; });
      var y = new Float64Array(mEff).fill(1);
      for (var it = 0; it < 3; it++) {
        y = denseSolve(Ms, y);
        var yn = 0; for (q = 0; q < mEff; q++) yn += y[q] * y[q];
        yn = Math.sqrt(yn); for (q = 0; q < mEff; q++) y[q] /= yn;
      }
      var resid = mEff < m ? 0 : Math.abs(H[mEff][mEff - 1] * y[mEff - 1]) / Math.abs(mu);
      if (resid > 2e-3) continue;
      var x = new Float64Array(n);
      for (q = 0; q < mEff; q++) { var yq = y[q], Vq2 = V[q]; for (g = 0; g < n; g++) x[g] += yq * Vq2[g]; }
      var field = new Float32Array(n), mx = 0, sg = 1;
      for (j = 0; j < Ny; j++) for (i = 0; i < Nx; i++) { var v = x[id(i, j)]; field[i + Nx * j] = v; if (Math.abs(v) > mx) { mx = Math.abs(v); sg = v < 0 ? -1 : 1; } }
      for (g = 0; g < n; g++) field[g] = sg * field[g] / mx;
      out.push({ pol: pol, N: Neff, field: field, resid: resid });
    }
    return out;
  }
  function classify(grid, field) {
    var Nx = grid.Nx, Ny = grid.Ny, best = 0, bi = 0, bj = 0, i, j;
    for (j = 0; j < Ny; j++) for (i = 0; i < Nx; i++) { var a = Math.abs(field[i + Nx * j]); if (a > best) { best = a; bi = i; bj = j; } }
    function count(get, len) {
      var mx = 0, k, c = 0, last = 0;
      for (k = 0; k < len; k++) mx = Math.max(mx, Math.abs(get(k)));
      for (k = 0; k < len; k++) { var v = get(k); if (Math.abs(v) < 0.08 * mx) continue; var sgn = v > 0 ? 1 : -1; if (last && sgn !== last) c++; last = sgn; }
      return c;
    }
    return {
      n: count(function (k) { return field[k + Nx * bj]; }, Nx),
      m: count(function (k) { return field[bi + Nx * k]; }, Ny),
      px: grid.xc[bi], py: grid.yc[bj]
    };
  }
  /* share of ∫|E|² dA in substrate, film/core, strip and superstrate */
  function fractions(grid, E) {
    var f = [0, 0, 0, 0], tot = 0;
    for (var j = 0; j < grid.Ny; j++) for (var i = 0; i < grid.Nx; i++) {
      var v = E(i, j), e = v * v * grid.dx[i] * grid.dy[j];
      f[grid.reg[i + grid.Nx * j]] += e; tot += e;
    }
    return f.map(function (x) { return tot > 0 ? x / tot : 0; });
  }
  function fdRun(G, lam, pols, res, grid) {
    var eim = solveEIM(G, lam, pols), ix = eim.ix;
    if (!eim.ok) return { grid: grid, modes: [] };
    if (!grid) grid = buildGrid(G, ix, lam, res, eim);
    var out = [], seen = {};
    ['TE', 'TM'].forEach(function (pol) {
      if (!pols[pol]) return;
      var cnt = eim.modes.filter(function (md) { return md.pol === pol; }).length;
      var lim = Math.max(eim.floor, eim.outN[pol] || 0);
      fdSolve(grid, ix, lam, pol, Math.max(4, cnt + 3)).forEach(function (md) {
        if (md.N <= lim + 1e-6) return;
        var c = classify(grid, md.field);
        md.frac = fractions(grid, function (i, j) { return md.field[i + grid.Nx * j]; });
        md.m = c.m; md.n = c.n; md.px = c.px; md.py = c.py; md.gamma = md.frac[1];
        md.key = pol + c.m + '_' + c.n;
        while (seen[md.key]) { md.key += '*'; md.dup = (md.dup || 0) + 1; }
        seen[md.key] = 1;
        out.push(md);
      });
    });
    out.sort(function (a, b) { return b.N - a.N; });
    return { grid: grid, modes: out, eim: eim };
  }

  return {
    MAT: MAT, nMat: nMat, indices: indices, validIx: validIx, confined: confined, stacks: stacks, regionAt: regionAt, coreBox: coreBox,
    fractions: fractions, slabModes: slabModes, solveEIM: solveEIM, eimVert: eimVert, eimLat: eimLat, buildGrid: buildGrid, fdRun: fdRun, hqr: hqr
  };
}

/* worker side: 2D solve at λ (with λ ± δ for n_g), then a coarse λ sweep */
function WGWORKER() {
  onmessage = function (e) {
    var d = e.data, G = d.G, t0 = Date.now();
    try {
      var P = C.fdRun(G, d.lam, d.pols, d.res, null);
      if (!P.grid) { postMessage({ type: 'error', msg: 'Invalid refractive index at this wavelength.' }); return; }
      var dl = d.lam * 0.004;
      var Pm = C.fdRun(G, d.lam - dl, d.pols, d.res, P.grid), Pp = C.fdRun(G, d.lam + dl, d.pols, d.res, P.grid);
      var find = function (L, k) { for (var i = 0; i < L.modes.length; i++) if (L.modes[i].key === k) return L.modes[i]; return null; };
      var tr = [];
      var modes = P.modes.map(function (md) {
        var a = find(Pm, md.key), b = find(Pp, md.key);
        tr.push(md.field.buffer);
        return { key: md.key, pol: md.pol, m: md.m, n: md.n, dup: md.dup || 0, N: md.N, ng: a && b ? md.N - d.lam * (b.N - a.N) / (2 * dl) : null, gamma: md.gamma, frac: md.frac, px: md.px, py: md.py, field: md.field };
      });
      var g = P.grid;
      postMessage({ type: 'point', lam: d.lam, ms: Date.now() - t0, grid: { Nx: g.Nx, Ny: g.Ny, xc: g.xc, yc: g.yc, dx: g.dx, dy: g.dy, reg: g.reg }, modes: modes }, tr);
      var K = 14, gl = d.lamMax, eimMax = C.solveEIM(G, gl, d.pols);
      if (!eimMax.ok) { postMessage({ type: 'done', ms: Date.now() - t0 }); return; }
      var gS = C.buildGrid(G, eimMax.ix, gl, d.res, eimMax.modes.length ? eimMax : C.solveEIM(G, d.lam, d.pols));
      for (var k = 0; k < K; k++) {
        var l = d.lamMin + (d.lamMax - d.lamMin) * k / (K - 1);
        var R = C.fdRun(G, l, d.pols, d.res, gS);
        postMessage({ type: 'sweep', lam: l, modes: R.modes.map(function (md) { return { key: md.key, pol: md.pol, m: md.m, n: md.n, dup: md.dup || 0, N: md.N }; }) });
      }
      postMessage({ type: 'done', ms: Date.now() - t0 });
    } catch (err) {
      postMessage({ type: 'error', msg: String(err && err.message || err) });
    }
  };
}
