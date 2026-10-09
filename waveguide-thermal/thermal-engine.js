/* thermal-engine.js — JavaScript port of the math-engines thermal engines (UI-free, SI units).

   Ports: thermo_optic (LUT, closed-form R', thermal_runaway), waveguide_thermal (build_ridge, finite-volume heat
   solve, scalar finite-volume mode, mode_weighted_heating), amplifier_thermal (er_amplifier_heating, er_pump_limit),
   thermal_detuning (qpm_thermal, ring_thermal_bistability). Reference implementation: math-engines/ (Python).
   Checked against math-engines/test_vectors/vectors.json by math-engines/tools/check_js_ports.mjs.

   THERMALENGINE() is a plain function returning the API, so a page can rebuild it inside a Blob Web Worker with
   Function.prototype.toString() (as WGCORE in waveguide-core). Linear systems: banded Cholesky (the conduction
   matrix and σM − A of the mode problem are symmetric positive definite); the mode by inverse iteration at the
   shift σ = (k n_max)², which converges to the same fundamental mode as scipy's shift-invert eigsh.
*/
function THERMALENGINE() {
  'use strict';
  var H = 6.62607015e-34, C0 = 299792458, DB = Math.log(10) / 10;
  var REGIONS = ['substrate', 'box', 'slab', 'core', 'clad'];

  /* ---------- materials: thermo_optic LUT (room temperature, 1550 nm) ---------- */
  var SELL = { sio2: [[0.6961663, 0.0684043 * 0.0684043], [0.4079426, 0.1162414 * 0.1162414], [0.8974794, 9.896161 * 9.896161]],
    si3n4: [[3.0249, 0.1353406 * 0.1353406], [40314.0, 1239.842 * 1239.842]],
    ln_e: [[2.2454, 0.01242], [1.3005, 0.0513], [6.8972, 331.33]], ln_o: [[2.4272, 0.01478], [1.4617, 0.05612], [9.6536, 371.216]],
    si: [[10.6684293, 0.301516485 * 0.301516485], [1.54133408, 1104.0 * 1104.0]] };
  var POLES = { gaas: [4.372514, [[5.466742, 0.4431307], [0.02429960, 0.8746453], [1.957522, 36.9166]]],
    aln_o: [2.1399, [[1.3786, 0.1715], [3.861, 15.03]]],
    sapphire_o: [0.0, [[1.4313493, 0.0726631], [0.65054713, 0.1193242], [5.3414021, 18.028251]]],
    lt_o_bond: [0.0, [[3.5030447406, 0.162397983924], [5.4986874, 15.811749571]]],
    lt_e_bond: [0.0, [[3.5184224161, 0.162906889986], [3.7507069272, 13.294141074]]] };
  function sellmeier(name, lam) {
    var lu = lam * 1e6, l2 = lu * lu, n2, i;
    if (POLES[name]) { n2 = 1 + POLES[name][0]; for (i = 0; i < POLES[name][1].length; i++) { var p = POLES[name][1][i]; n2 += p[0] * l2 / (l2 - p[1] * p[1]); } return Math.sqrt(n2); }
    n2 = 1; for (i = 0; i < SELL[name].length; i++) n2 += SELL[name][i][0] * l2 / (l2 - SELL[name][i][1]);
    return Math.sqrt(n2);
  }
  /* name, sellmeier, n(1550), dn/dT, k, rho, cp, alpha_L, Eg, beta_tpa */
  var LUT = {
    si: ['Si', 'si', 3.476, 1.84e-4, 148.0, 2329, 705, 2.6e-6, 1.12, 8.0e-12],
    sio2: ['SiO₂', 'sio2', 1.444, 0.95e-5, 1.38, 2200, 740, 0.55e-6, 9.0, 0],
    si3n4: ['Si₃N₄ (LPCVD)', 'si3n4', 1.996, 2.45e-5, 10.0, 3100, 700, 3.3e-6, 5.0, 0],
    sinx: ['SiNx (PECVD)', null, 1.90, 4.0e-5, 1.5, 2500, 750, 2.0e-6, 4.0, 0],
    ln_e: ['LiNbO₃ e (TFLN)', 'ln_e', 2.138, 3.4e-5, 4.6, 4650, 630, 7.5e-6, 3.8, 0],
    ln_o: ['LiNbO₃ o (TFLN)', 'ln_o', 2.211, 0.3e-5, 4.6, 4650, 630, 15.4e-6, 3.8, 0],
    lt_e: ['LiTaO₃ e (TFLT)', 'lt_e_bond', 2.119, 1.5e-5, 4.6, 7450, 424, 4.1e-6, 4.6, 0],
    lt_o: ['LiTaO₃ o (TFLT)', 'lt_o_bond', 2.123, 0.5e-5, 4.6, 7450, 424, 16.1e-6, 4.6, 0],
    gaas: ['GaAs', 'gaas', 3.374, 2.35e-4, 55.0, 5317, 330, 5.73e-6, 1.424, 1.0e-10],
    algaas: ['Al₀.₂Ga₀.₈As', null, 3.374 - 0.454 * 0.2, 2.35e-4 - 0.95e-4 * 0.2, 100 / (2.27 + 28.83 * 0.2 - 30 * 0.04), 5317 - 1557 * 0.2, 330 + 120 * 0.2, (5.73 - 0.53 * 0.2) * 1e-6, 1.424 + 1.247 * 0.2, 0],
    inp: ['InP', null, 3.167, 2.0e-4, 68.0, 4810, 310, 4.6e-6, 1.344, NaN],
    aln: ['AlN', 'aln_o', 2.12, 2.3e-5, 30.0, 3255, 740, 4.2e-6, 6.0, 0],
    al2o3_film: ['Al₂O₃ film', null, 1.65, 1.3e-5, 1.6, 3200, 780, 4.2e-6, 6.5, 0],
    sapphire: ['Sapphire', 'sapphire_o', 1.746, 1.3e-5, 35.0, 3980, 760, 5.0e-6, 8.8, 0],
    tio2_film: ['TiO₂ film', null, 2.30, -1.0e-4, 1.0, 3800, 690, 8.0e-6, 3.3, 0],
    su8: ['SU-8 / polymer', null, 1.57, -1.1e-4, 0.2, 1200, 1200, 52e-6, 4.0, 0],
    air: ['Air', null, 1.00027, -0.92e-6, 0.026, 1.2, 1005, 0.0, 12.0, 0]
  };
  function entry(m) {
    var r = LUT[m]; if (!r) throw new Error('unknown material ' + m);
    return { name: r[0], sellmeier: r[1], n: r[2], dn_dT: r[3], k: r[4], rho: r[5], cp: r[6], alpha_L: r[7], Eg: r[8], beta_tpa: r[9] };
  }
  function indexAt(m, lam) { var e = entry(m); return e.sellmeier ? sellmeier(e.sellmeier, lam) : e.n; }

  /* ---------- numerics ---------- */
  function linspace(a, b, n) { var o = new Float64Array(n), s = (b - a) / (n - 1); for (var i = 0; i < n; i++) o[i] = a + s * i; o[n - 1] = b; return o; }
  function interp1(x, xp, fp) {
    if (x <= xp[0]) return fp[0];
    var n = xp.length; if (x >= xp[n - 1]) return fp[n - 1];
    var lo = 0, hi = n - 1; while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
    return fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / (xp[hi] - xp[lo]);
  }
  function uniqSorted(a) { var b = a.slice().sort(function (p, q) { return p - q; }), o = []; for (var i = 0; i < b.length; i++) if (!o.length || b[i] !== o[o.length - 1]) o.push(b[i]); return o; }
  function gradedAxis(breaks, fineLo, fineHi, hFine, growth, minCells) {
    minCells = minCells || 2;
    var b = uniqSorted(breaks), edges = [b[0]];
    for (var s = 0; s + 1 < b.length; s++) {
      var x = linspace(b[s], b[s + 1], 801), inv = new Float64Array(801), cum = new Float64Array(801), i;
      for (i = 0; i < 801; i++) { var d = Math.max(Math.max(fineLo - x[i], x[i] - fineHi), 0); inv[i] = 1 / (hFine + growth * d); }
      for (i = 1; i < 801; i++) cum[i] = cum[i - 1] + 0.5 * (inv[i] + inv[i - 1]) * (x[i] - x[i - 1]);
      var n = Math.max(minCells, Math.ceil(cum[800])), t = linspace(0, cum[800], n + 1);
      for (i = 1; i <= n; i++) edges.push(interp1(t[i], cum, x));
    }
    edges[edges.length - 1] = b[b.length - 1];
    return Float64Array.from(edges);
  }
  /* banded SPD matrix, bandwidth bw: entry (i, j), j <= i, at i*(bw+1) + (i-j) */
  function Band(n, bw) { this.n = n; this.bw = bw; this.a = new Float64Array(n * (bw + 1)); }
  Band.prototype.add = function (i, j, v) { if (j > i) { var t = i; i = j; j = t; } this.a[i * (this.bw + 1) + (i - j)] += v; };
  Band.prototype.cholesky = function () {
    var n = this.n, bw = this.bw, w = bw + 1, a = this.a;
    for (var i = 0; i < n; i++) {
      var j0 = Math.max(0, i - bw);
      for (var j = j0; j <= i; j++) {
        var s = a[i * w + (i - j)], k0 = Math.max(j0, j - bw);
        for (var k = k0; k < j; k++) s -= a[i * w + (i - k)] * a[j * w + (j - k)];
        if (j === i) { if (!(s > 0)) throw new Error('matrix not positive definite'); a[i * w] = Math.sqrt(s); }
        else a[i * w + (i - j)] = s / a[j * w];
      }
    }
    return this;
  };
  Band.prototype.solve = function (b) {
    var n = this.n, bw = this.bw, w = bw + 1, a = this.a, y = Float64Array.from(b), i, k;
    for (i = 0; i < n; i++) { var s = y[i]; for (k = Math.max(0, i - bw); k < i; k++) s -= a[i * w + (i - k)] * y[k]; y[i] = s / a[i * w]; }
    for (i = n - 1; i >= 0; i--) { var t = y[i]; for (k = i + 1; k <= Math.min(n - 1, i + bw); k++) t -= a[k * w + (k - i)] * y[k]; y[i] = t / a[i * w]; }
    return y;
  };

  /* ---------- waveguide_thermal: geometry and conduction ---------- */
  var GEO_DEFAULTS = { core_material: 'si', core_width: 0.5e-6, core_height: 0.22e-6, slab_material: 'si', slab_thickness: 0,
    box_material: 'sio2', box_thickness: 2e-6, substrate_material: 'si', substrate_thickness: 500e-6, clad_material: 'sio2',
    clad_thickness: 2e-6, domain_half_width: 500e-6, resolution: 1 };
  function geo(o) { var g = {}, k; for (k in GEO_DEFAULTS) g[k] = GEO_DEFAULTS[k]; for (k in o || {}) g[k] = o[k]; return g; }
  function buildRidge(o) {
    var G = geo(o), mats = { substrate: G.substrate_material, box: G.box_material, slab: G.slab_material, core: G.core_material, clad: G.clad_material };
    var kk = {}, rc = {}, r;
    for (r in mats) { kk[r] = G['k_' + r] != null ? G['k_' + r] : entry(mats[r]).k; rc[r] = entry(mats[r]).rho * entry(mats[r]).cp; }
    var H = G.substrate_thickness, tb = G.box_thickness, ts = G.slab_thickness, hc = G.core_height;
    var yBox = H + tb, yFilm = H + tb + ts, yCore = yFilm + hc, yTop = Math.max(H + tb + G.clad_thickness, yCore), w2 = G.core_width / 2;
    var feats = [w2, hc, tb]; if (ts > 0) feats.push(ts); if (yTop > yCore) feats.push(yTop - yCore);
    var res = G.resolution, hf = Math.min.apply(null, feats) / (6 * res);
    hf = Math.max(hf, Math.min(w2, hc) / (12 * res));
    var growth = 0.15 / res;
    var xb = [0, w2, G.domain_half_width].concat((G.x_breaks || []).filter(function (v) { return v > 0 && v < G.domain_half_width; }));
    var yb = [0, H, yBox, yFilm, yCore, yTop].concat((G.y_breaks || []).filter(function (v) { return v > 0 && v < yTop; }));
    var xe = gradedAxis(xb, 0, w2, Math.min(hf, w2 / 4), growth), ye = gradedAxis(yb, yBox, yCore, hf, growth);
    var nx = xe.length - 1, ny = ye.length - 1, N = nx * ny;
    var xc = new Float64Array(nx), yc = new Float64Array(ny), i, j;
    for (i = 0; i < nx; i++) xc[i] = 0.5 * (xe[i] + xe[i + 1]);
    for (j = 0; j < ny; j++) yc[j] = 0.5 * (ye[j] + ye[j + 1]);
    var reg = new Uint8Array(N), k = new Float64Array(N), rcm = new Float64Array(N), dA = new Float64Array(N), core = new Uint8Array(N);
    for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) {
      var X = xc[i], Y = yc[j], c = 4;
      if (Y < yBox) c = 1; if (Y < H) c = 0;
      if (ts > 0 && Y > yBox && Y < yFilm) c = 2;
      if (Y > yFilm && Y < yCore && X < w2) c = 3;
      var id = i + nx * j; reg[id] = c; k[id] = kk[REGIONS[c]]; rcm[id] = rc[REGIONS[c]];
      dA[id] = (ye[j + 1] - ye[j]) * (xe[i + 1] - xe[i]); core[id] = c === 3 ? 1 : 0;
    }
    return { xe: xe, ye: ye, xc: xc, yc: yc, nx: nx, ny: ny, reg: reg, k: k, rc: rcm, dA: dA, core: core, mats: mats, kk: kk,
      levels: { H: H, y_box: yBox, y_film: yFilm, y_core: yCore, y_top: yTop, w2: w2 } };
  }
  /* A T = Q (heat per cell per length); bottom isothermal, top convective h_top, sides adiabatic */
  function conductionMatrix(g, k, hTop) {
    var nx = g.nx, ny = g.ny, B = new Band(nx * ny, nx), i, j, xe = g.xe, ye = g.ye, Gb = new Float64Array(nx);
    for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) {
      var id = i + nx * j, dx = xe[i + 1] - xe[i], dy = ye[j + 1] - ye[j], G;
      if (i + 1 < nx) { var dx2 = xe[i + 2] - xe[i + 1]; G = dy / (dx / (2 * k[id]) + dx2 / (2 * k[id + 1])); B.add(id, id, G); B.add(id + 1, id + 1, G); B.add(id + 1, id, -G); }
      if (j + 1 < ny) { var dy2 = ye[j + 2] - ye[j + 1]; G = dx / (dy / (2 * k[id]) + dy2 / (2 * k[id + nx])); B.add(id, id, G); B.add(id + nx, id + nx, G); B.add(id + nx, id, -G); }
      if (j === 0) { Gb[i] = dx / (dy / (2 * k[id])); B.add(id, id, Gb[i]); }
      if (j === ny - 1 && hTop > 0) B.add(id, id, dx / (dy / (2 * k[id]) + 1 / hTop));
    }
    return { B: B, Gb: Gb };
  }
  function heatMap(g, qp, weight) {
    var q = new Float64Array(g.k.length), s = 0, i;
    for (i = 0; i < q.length; i++) if (g.core[i]) { q[i] = weight ? weight[i] : 1; s += q[i] * g.dA[i]; }
    for (i = 0; i < q.length; i++) q[i] *= (qp / 2) / s;
    return q;
  }
  function solveHeat(g, k, q, hTop) {
    var m = conductionMatrix(g, k, hTop), b = new Float64Array(q.length);
    for (var i = 0; i < q.length; i++) b[i] = q[i] * g.dA[i];
    return m.B.cholesky().solve(b);
  }
  function coreMean(g, F) { var s = 0, a = 0; for (var i = 0; i < F.length; i++) if (g.core[i]) { s += F[i] * g.dA[i]; a += g.dA[i]; } return s / a; }
  function ridgeHeating(o) {
    var qp = o.heat_per_length != null ? o.heat_per_length : 1, hTop = o.h_top != null ? o.h_top : 10;
    var g = buildRidge(o), q = heatMap(g, qp, o.heat_weight), T = solveHeat(g, g.k, q, hTop);
    var Tc = coreMean(g, T), Tmax = -Infinity, E = 0, col = new Float64Array(g.ny), i;
    for (i = 0; i < T.length; i++) { if (T[i] > Tmax) Tmax = T[i]; E += g.rc[i] * T[i] * g.dA[i]; }
    for (i = 0; i < g.ny; i++) col[i] = T[i * g.nx];
    return { dT_core: Tc, dT_max: Tmax, R_th: Tc / qp, tau_E: E / (qp / 2), dT_box_top: interp1(g.levels.y_box, g.yc, col),
      dT_substrate_top: interp1(g.levels.H, g.yc, col), n_cells: T.length, T: T, grid: g };
  }

  /* ---------- scalar finite-volume mode (ridge_mode) ---------- */
  function ridgeMode(o, lam, opt) {
    opt = opt || {};
    var G = geo(o), margin = opt.margin != null ? opt.margin : 1.5e-6, cpw = opt.cells_per_wavelength || 24;
    var mats = { box: G.box_material, slab: G.slab_material, core: G.core_material, clad: G.clad_material }, nOf = {}, r;
    for (r in mats) nOf[r] = opt.indices && opt.indices[r] != null ? opt.indices[r] : indexAt(mats[r], lam);
    var w2 = G.core_width / 2, ts = G.slab_thickness, hc = G.core_height, yFilm = ts, yCore = ts + hc, nMax = 0;
    for (r in nOf) if (r !== 'slab' || ts > 0) nMax = Math.max(nMax, nOf[r]);
    var h = opt.h || Math.min(lam / (nMax * cpw), hc / 6, ts > 0 ? ts / 3 : hc / 6, w2 / 6);
    function axis(br) {
      var e = [br[0]];
      for (var s = 0; s + 1 < br.length; s++) { var lo = br[s], hi = br[s + 1], m = Math.max(1, Math.ceil((hi - lo) / h - 1e-9)); for (var t = 1; t <= m; t++) e.push(lo + (hi - lo) * t / m); }
      return Float64Array.from(e);
    }
    var xe = axis([-(w2 + margin), -w2, w2, w2 + margin]);
    var ye = axis([-Math.min(margin, G.box_thickness)].concat(ts > 0 ? [0, yFilm] : [0]).concat([yCore, yCore + margin]));
    var nx = xe.length - 1, ny = ye.length - 1, N = nx * ny, i, j;
    var x = new Float64Array(nx), y = new Float64Array(ny), dx = new Float64Array(nx), dy = new Float64Array(ny);
    for (i = 0; i < nx; i++) { x[i] = 0.5 * (xe[i] + xe[i + 1]); dx[i] = xe[i + 1] - xe[i]; }
    for (j = 0; j < ny; j++) { y[j] = 0.5 * (ye[j] + ye[j + 1]); dy[j] = ye[j + 1] - ye[j]; }
    var reg = new Uint8Array(N), n = new Float64Array(N), V = new Float64Array(N), k0 = 2 * Math.PI / lam, sigma = Math.pow(k0 * nMax, 2);
    var nByReg = REGIONS.map(function (rr) { return nOf[rr] != null ? nOf[rr] : 1.0; });
    for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) {
      var id = i + nx * j, c = 4;
      if (y[j] < 0) c = 1;
      if (ts > 0 && y[j] > 0 && y[j] < yFilm) c = 2;
      if (y[j] > yFilm && y[j] < yCore && Math.abs(x[i]) < w2) c = 3;
      reg[id] = c; n[id] = nByReg[c] + (opt.delta_n ? opt.delta_n[id] : 0); V[id] = dx[i] * dy[j];
    }
    /* B = σM - A (SPD); A = -Σ couplings + k²n²V on the diagonal */
    var A = new Band(N, nx), Bm = new Band(N, nx), diag = new Float64Array(N), off = [];
    for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) {
      var id2 = i + nx * j, G2;
      diag[id2] += Math.pow(k0 * n[id2], 2) * V[id2];
      if (i + 1 < nx) { G2 = dy[j] / (0.5 * (dx[i] + dx[i + 1])); diag[id2] -= G2; diag[id2 + 1] -= G2; off.push(id2, id2 + 1, G2); }
      if (j + 1 < ny) { G2 = dx[i] / (0.5 * (dy[j] + dy[j + 1])); diag[id2] -= G2; diag[id2 + nx] -= G2; off.push(id2, id2 + nx, G2); }
      if (i === 0) diag[id2] -= dy[j] / (0.5 * dx[0]);
      if (i === nx - 1) diag[id2] -= dy[j] / (0.5 * dx[nx - 1]);
      if (j === 0) diag[id2] -= dx[i] / (0.5 * dy[0]);
      if (j === ny - 1) diag[id2] -= dx[i] / (0.5 * dy[ny - 1]);
    }
    for (i = 0; i < N; i++) Bm.add(i, i, sigma * V[i] - diag[i]);
    for (i = 0; i < off.length; i += 3) Bm.add(off[i + 1], off[i], -off[i + 2]);
    Bm.cholesky();
    function Aapply(v) {
      var o2 = new Float64Array(N), t;
      for (t = 0; t < N; t++) o2[t] = diag[t] * v[t];
      for (t = 0; t < off.length; t += 3) { o2[off[t]] += off[t + 2] * v[off[t + 1]]; o2[off[t + 1]] += off[t + 2] * v[off[t]]; }
      return o2;
    }
    var E = new Float64Array(N), lamPrev = 0, ev = 0, it;
    for (i = 0; i < N; i++) E[i] = reg[i] === 3 ? 1 : 0.1;
    for (it = 0; it < 5000; it++) {
      var rhs = new Float64Array(N); for (i = 0; i < N; i++) rhs[i] = V[i] * E[i];
      E = Bm.solve(rhs);
      var nrm = 0; for (i = 0; i < N; i++) nrm += E[i] * E[i] * V[i];
      nrm = Math.sqrt(nrm); for (i = 0; i < N; i++) E[i] /= nrm;
      var AE = Aapply(E), num = 0; for (i = 0; i < N; i++) num += E[i] * AE[i];
      ev = num;
      if (it > 3 && Math.abs(ev - lamPrev) <= 1e-14 * Math.abs(ev)) break;
      lamPrev = ev;
    }
    return { x: x, y: y, nx: nx, ny: ny, E: E, n: n, reg: reg, V: V, neff: Math.sqrt(ev) / k0, h: h, mats: mats, iterations: it };
  }

  /* bilinear on a rectilinear grid (values F[j*nx + i] at (ys[j], xs[i])); extrapolate or zero outside */
  function bilinear(xs, ys, F, x, y, zeroOutside) {
    var nx = xs.length, ny = ys.length;
    if (zeroOutside && (x < xs[0] || x > xs[nx - 1] || y < ys[0] || y > ys[ny - 1])) return 0;
    function cell(a, v) { var lo = 0, hi = a.length - 1; if (v <= a[0]) return 0; if (v >= a[hi]) return hi - 1; while (hi - lo > 1) { var m = (lo + hi) >> 1; if (a[m] <= v) lo = m; else hi = m; } return lo; }
    var i = cell(xs, x), j = cell(ys, y), tx = (x - xs[i]) / (xs[i + 1] - xs[i]), ty = (y - ys[j]) / (ys[j + 1] - ys[j]);
    return (1 - ty) * ((1 - tx) * F[j * nx + i] + tx * F[j * nx + i + 1]) + ty * ((1 - tx) * F[(j + 1) * nx + i] + tx * F[(j + 1) * nx + i + 1]);
  }
  function gridToMode(g, m, T) {
    var nx = g.nx, xm = new Float64Array(2 * nx), Tm = new Float64Array(2 * nx * g.ny), out = new Float64Array(m.nx * m.ny), i, j;
    for (i = 0; i < nx; i++) { xm[i] = -g.xc[nx - 1 - i]; xm[nx + i] = g.xc[i]; }
    for (j = 0; j < g.ny; j++) for (i = 0; i < nx; i++) { Tm[j * 2 * nx + nx + i] = T[j * nx + i]; Tm[j * 2 * nx + nx - 1 - i] = T[j * nx + i]; }
    var y0 = g.levels.y_box, dxs = m.shift || 0;
    for (j = 0; j < m.ny; j++) for (i = 0; i < m.nx; i++) out[j * m.nx + i] = bilinear(xm, g.yc, Tm, m.x[i] + dxs, m.y[j] + y0, false);
    return out;
  }
  function modeToGrid(g, m, F) {
    var ys = new Float64Array(m.ny), out = new Float64Array(g.nx * g.ny), i, j;
    for (j = 0; j < m.ny; j++) ys[j] = m.y[j] + g.levels.y_box;
    for (j = 0; j < g.ny; j++) for (i = 0; i < g.nx; i++) out[j * g.nx + i] = bilinear(m.x, ys, F, g.xc[i], g.yc[j], true);
    return out;
  }
  function modeWeightedHeating(o, lam, opt) {
    opt = opt || {};
    var qp = opt.heat_per_length != null ? opt.heat_per_length : 1, hTop = opt.h_top != null ? opt.h_top : 10;
    var m = ridgeMode(o, lam, opt), g = buildRidge(o), N = m.E.length, i;
    var E2 = new Float64Array(N), E2V = new Float64Array(N), tot = 0;
    for (i = 0; i < N; i++) { E2[i] = m.E[i] * m.E[i]; E2V[i] = E2[i] * m.V[i]; tot += E2V[i]; }
    var weight = opt.heat_in_mode ? modeToGrid(g, m, E2) : null;
    var T = solveHeat(g, g.k, heatMap(g, qp, weight), hTop), Tm = gridToMode(g, m, T);
    var gam = [0, 0, 0, 0, 0], dn = 0, dndT = REGIONS.map(function (r) { return m.mats[r] ? entry(m.mats[r]).dn_dT : 0; });
    for (i = 0; i < N; i++) { gam[m.reg[i]] += m.n[i] * E2V[i]; dn += m.n[i] * dndT[m.reg[i]] * Tm[i] * E2V[i]; }
    for (i = 0; i < 5; i++) gam[i] /= m.neff * tot;
    dn /= m.neff * tot;
    var dndTeff = 0; for (i = 1; i < 5; i++) dndTeff += gam[i] * dndT[i];
    var dl = lam * 0.01, oo = { margin: opt.margin, h: m.h, cells_per_wavelength: opt.cells_per_wavelength };
    var ng = m.neff - lam * (ridgeMode(o, lam + dl, oo).neff - ridgeMode(o, lam - dl, oo).neff) / (2 * dl);
    return { neff: m.neff, n_group: ng, Gamma_core: gam[3], Gamma_slab: gam[2], Gamma_box: gam[1], Gamma_clad: gam[4],
      dneff_dT: dndTeff, dT_core: coreMean(g, T), dT_mode: dn / dndTeff, dneff: dn, R_th_mode: dn / dndTeff / qp,
      dlambda_dT: lam * dndTeff / ng, delta_lambda: lam * dn / ng, mode: m, grid: g, T: T };
  }

  /* ---------- thermo_optic: closed-form R', pump budget, runaway ---------- */
  function stripThermalResistance(o) {
    var w = o.width, h = o.height, t = o.box_thickness, kb = o.box_k != null ? o.box_k : entry(o.box_material || 'sio2').k;
    var ks = o.substrate_k != null ? o.substrate_k : entry(o.substrate_material || 'si').k, kf = o.slab_k != null ? o.slab_k : entry(o.slab_material || 'si').k;
    var ts = o.slab_thickness || 0, H = o.substrate_thickness || 500e-6, d, r, m, weff;
    if (ts > 0 || o.clad_material === 'air') { weff = w + 2 * Math.sqrt(kf * ts * t / (2 * kb)); d = t + ts; r = weff / 4; m = 1; }
    else { weff = w; d = t + h / 2; r = (w + 2 * h) / 4; m = 2; }
    var Rb = Math.acosh(Math.max(d / r, 1)) / (m * Math.PI * kb), w2 = weff + 2 * t, arg = 8 * H / (Math.PI * w2);
    var Rs = arg > Math.E ? Math.log(arg) / (Math.PI * ks) : H / (ks * w2);
    return { R_th: Rb + Rs, R_box: Rb, R_sub: Rs, w_eff: weff };
  }
  function maxPower(c1, c2, c3, budget) {
    if (budget <= 0) return 0;
    if (c1 === 0 && c2 === 0 && c3 === 0) return Infinity;
    var f = function (P) { return c1 * P + c2 * P * P + c3 * P * P * P; }, lo = 0, hi = 1;
    while (f(hi) < budget) { hi *= 2; if (hi > 1e12) return Infinity; }
    for (var i = 0; i < 200; i++) { var mid = 0.5 * (lo + hi); if (f(mid) < budget) lo = mid; else hi = mid; }
    return 0.5 * (lo + hi);
  }
  function kirchhoff(dT, m, T0) { var x = 1 + dT / T0; return Math.abs(m - 1) < 1e-12 ? T0 * Math.log(x) : T0 / (1 - m) * (Math.pow(x, 1 - m) - 1); }
  function thermalRunaway(o) {
    var R = o.R_th, lam = o.wavelength || 1.55e-6, A = o.a_eff || 0.1e-12, Ta = o.T_scale != null ? o.T_scale : Infinity;
    var a0 = (o.loss_abs_db_per_cm != null ? o.loss_abs_db_per_cm : 1) * 100 * DB, beta = o.beta_tpa || 0, tau = o.carrier_lifetime || 0;
    var sig = o.sigma_fca || 0, m = o.k_exponent || 0, T0 = o.T0 || 300, lim = o.dT_limit || 3000, P = o.power;
    var hnu = H * C0 / lam, c2 = beta / A, c3 = sig * tau * beta / (2 * hnu * A * A);
    var ex = function (v) { return Math.exp(Math.min(v / Ta, 700)); };
    var q = function (v, p) { return a0 * ex(v) * p + c2 * p * p + c3 * p * p * p; };
    var Pof = function (v) { return maxPower(a0 * ex(v), c2, c3, kirchhoff(v, m, T0) / R); };
    var xs = [0], i; for (i = 0; i < 1500; i++) xs.push(1e-6 * Math.pow(lim / 1e-6, i / 1499));
    var Px = xs.map(Pof), ib = 0; for (i = 1; i < Px.length; i++) if (Px[i] > Px[ib]) ib = i;
    var Pth, dTth, kind;
    if (ib > 0 && ib < xs.length - 1) {
      var lo = xs[ib - 1], hi = xs[ib + 1], gr = (Math.sqrt(5) - 1) / 2;
      for (i = 0; i < 80; i++) { var a = hi - gr * (hi - lo), b = lo + gr * (hi - lo); if (Pof(a) > Pof(b)) hi = b; else lo = a; }
      dTth = 0.5 * (lo + hi); Pth = Pof(dTth); kind = 'fold';
    } else if (m > 1) { dTth = Infinity; Pth = Ta === Infinity ? maxPower(a0, c2, c3, T0 / (m - 1) / R) : Math.max.apply(null, Px); kind = 'conduction limit'; }
    else { dTth = Infinity; Pth = Infinity; kind = 'none'; }
    var dT = NaN, run = P > Pth;
    if (!run) {
      var F = function (v) { return kirchhoff(v, m, T0) - R * q(v, P); }, prev = 0;
      for (i = 1; i < xs.length; i++) {
        if (F(xs[i]) >= 0) { var l2 = prev, h2 = xs[i]; for (var t = 0; t < 100; t++) { var md = 0.5 * (l2 + h2); if (F(md) < 0) l2 = md; else h2 = md; } dT = 0.5 * (l2 + h2); break; }
        prev = xs[i];
      }
    }
    return { dT: dT, dT_no_feedback: R * q(0, P), runaway: run, P_threshold: Pth, dT_threshold: dTth, threshold_kind: kind };
  }

  /* ---------- amplifier_thermal ---------- */
  var ER_DEFAULTS = { pump_power: 0.2, signal_power: 1e-6, length: 0.03, R_th: 0.5, dneff_dT: 3e-5, pump_wavelength: 0.98e-6,
    signal_wavelength: 1.532e-6, n_er: 1.5e26, quenched_fraction: 0, tau: 7.5e-3, c_up: 4e-24, sigma_a_pump: 1.7e-25,
    sigma_e_pump: 0, sigma_a_signal: 5.7e-25, sigma_e_signal: 5.7e-25, gamma_pump: 0.5, gamma_signal: 0.4, doped_area: 0.8e-12,
    loss_db_per_cm: 0.25, absorbing_fraction: 0.5, eta_rad: 0.8, n_z: 300 };
  function erAmplifierHeating(o) {
    var a = {}, k; for (k in ER_DEFAULTS) a[k] = ER_DEFAULTS[k]; for (k in o || {}) a[k] = o[k];
    var p = { hvp: H * C0 / a.pump_wavelength, hvs: H * C0 / a.signal_wavelength, N: a.n_er * (1 - a.quenched_fraction), Nq: a.n_er * a.quenched_fraction,
      tau: a.tau, Cup: a.c_up, sap: a.sigma_a_pump, sep: a.sigma_e_pump, sas: a.sigma_a_signal, ses: a.sigma_e_signal, Gp: a.gamma_pump,
      Gs: a.gamma_signal, Ad: a.doped_area, al: a.loss_db_per_cm * 100 * DB, fabs: a.absorbing_fraction, eta: a.eta_rad };
    function pops(Pp, Ps) {
      var Ip = p.Gp * Pp / p.Ad, Is = p.Gs * Ps / p.Ad, Ra = p.sap * Ip / p.hvp + p.sas * Is / p.hvs, Re = p.sep * Ip / p.hvp + p.ses * Is / p.hvs;
      var b = Ra + Re + 1 / p.tau, n2 = Ra > 0 ? 2 * Ra * p.N / (b + Math.sqrt(b * b + 4 * p.Cup * Ra * p.N)) : 0;
      return [p.N - n2 + p.Nq, n2];
    }
    function rhs(Pp, Ps) { var n = pops(Pp, Ps); return [-(p.Gp * (p.sap * n[0] - p.sep * n[1]) + p.al) * Pp, (p.Gs * (p.ses * n[1] - p.sas * n[0]) - p.al) * Ps]; }
    function heat(Pp, Ps) {
      var n = pops(Pp, Ps);
      return [p.Gp * (p.sap * n[0] - p.sep * n[1]) * Pp + p.Gs * (p.sas * n[0] - p.ses * n[1]) * Ps - p.eta * n[1] / p.tau * p.hvs * p.Ad + p.fabs * p.al * (Pp + Ps), n[1]];
    }
    var nz = a.n_z | 0, dz = a.length / nz, z = new Float64Array(nz + 1), Pp = new Float64Array(nz + 1), Ps = new Float64Array(nz + 1), i;
    Pp[0] = a.pump_power; Ps[0] = a.signal_power;
    for (i = 0; i < nz; i++) {
      var x = Pp[i], y = Ps[i], k1 = rhs(x, y), k2 = rhs(x + 0.5 * dz * k1[0], y + 0.5 * dz * k1[1]), k3 = rhs(x + 0.5 * dz * k2[0], y + 0.5 * dz * k2[1]), k4 = rhs(x + dz * k3[0], y + dz * k3[1]);
      Pp[i + 1] = Math.max(0, x + dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]));
      Ps[i + 1] = Math.max(0, y + dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]));
      z[i + 1] = a.length * (i + 1) / nz;
    }
    var q = new Float64Array(nz + 1), inv = new Float64Array(nz + 1), dT = new Float64Array(nz + 1), heatTot = 0, iHot = 0, mean = 0;
    for (i = 0; i <= nz; i++) { var hh = heat(Pp[i], Ps[i]); q[i] = hh[0]; inv[i] = hh[1] / (p.N + p.Nq); dT[i] = a.R_th * q[i]; mean += dT[i]; if (q[i] > q[iHot]) iHot = i; }
    for (i = 0; i < nz; i++) heatTot += 0.5 * (q[i] + q[i + 1]) * (z[i + 1] - z[i]);
    var pabs = a.pump_power - Pp[nz];
    return { gain_dB: a.signal_power > 0 && Ps[nz] > 0 ? 10 * Math.log10(Ps[nz] / a.signal_power) : NaN, pump_out: Pp[nz], pump_absorbed: pabs,
      heat_total: heatTot, heat_fraction: pabs > 0 ? heatTot / pabs : NaN, q_max: q[iHot], z_hot: z[iHot], dT_max: dT[iHot],
      dT_mean: mean / (nz + 1), dneff_max: a.dneff_dT * dT[iHot], inversion_in: inv[0], inversion_out: inv[nz],
      z: z, P_pump: Pp, P_signal: Ps, q: q, dT: dT, inversion: inv };
  }
  function erPumpLimit(o, dTmax, dnmax, Psearch) {
    Psearch = Psearch || 20;
    var oo = {}, k; for (k in o) oo[k] = o[k]; if (oo.n_z == null) oo.n_z = 200;
    function over(P) { oo.pump_power = P; var r = erAmplifierHeating(oo); return r.dT_max > dTmax || Math.abs(r.dneff_max) > dnmax; }
    var P = Psearch, limited = false;
    if (over(Psearch)) { var lo = 0, hi = Psearch; for (var i = 0; i < 40; i++) { var mid = 0.5 * (lo + hi); if (over(mid)) hi = mid; else lo = mid; } P = lo; limited = true; }
    oo.pump_power = P;
    var r = erAmplifierHeating(oo);
    return { P_limit: P, limited: limited, gain_dB: r.gain_dB, dT_max: r.dT_max, dneff_max: r.dneff_max, heat_total: r.heat_total };
  }

  /* ---------- thermal_detuning ---------- */
  function gayerE(lam, Tk) {
    var a = [5.756, 0.0983, 0.202, 189.32, 12.52, 1.32e-2], b = [2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4], T0 = 24.5, tc = Tk - 273.15;
    var f = (tc - T0) * (tc + T0 + 2 * 273.16), lu = lam * 1e6, l2 = lu * lu;
    return Math.sqrt(a[0] + b[0] * f + (a[1] + b[1] * f) / (l2 - Math.pow(a[2] + b[2] * f, 2)) + (a[3] + b[3] * f) / (l2 - a[4] * a[4]) - a[5] * l2);
  }
  function qpmDk(lp, ls, T) { var li = 1 / (1 / lp - 1 / ls); return [2 * Math.PI * (gayerE(lp, T) / lp - gayerE(ls, T) / ls - gayerE(li, T) / li), li]; }
  function phaseMatchingFactor(z, dk) {
    var re = 0, im = 0, phi = 0, pr = 1, pi = 0;
    for (var i = 1; i < z.length; i++) {
      phi += 0.5 * (dk[i] + dk[i - 1]) * (z[i] - z[i - 1]);
      var cr = Math.cos(phi), ci = Math.sin(phi);
      re += 0.5 * (cr + pr) * (z[i] - z[i - 1]); im += 0.5 * (ci + pi) * (z[i] - z[i - 1]); pr = cr; pi = ci;
    }
    var L = z[z.length - 1] - z[0]; return (re * re + im * im) / (L * L);
  }
  function bestRetunedFactor(z, dk) {
    var L = z[z.length - 1] - z[0], mean = 0, mn = Infinity, mx = -Infinity, i;
    for (i = 1; i < z.length; i++) mean += 0.5 * (dk[i] + dk[i - 1]) * (z[i] - z[i - 1]);
    mean /= L; for (i = 0; i < dk.length; i++) { mn = Math.min(mn, dk[i]); mx = Math.max(mx, dk[i]); }
    var span = 4 * Math.PI / L + (mx - mn), offs = linspace(-mean - span, -mean + span, 401);
    var add = function (o) { var d = new Float64Array(dk.length); for (var t = 0; t < dk.length; t++) d[t] = dk[t] + o; return phaseMatchingFactor(z, d); };
    var vals = Array.prototype.map.call(offs, add), ib = 0; for (i = 1; i < vals.length; i++) if (vals[i] > vals[ib]) ib = i;
    var lo = offs[Math.max(ib - 1, 0)], hi = offs[Math.min(ib + 1, offs.length - 1)], gr = (Math.sqrt(5) - 1) / 2;
    for (i = 0; i < 60; i++) { var a = hi - gr * (hi - lo), b = lo + gr * (hi - lo); if (add(a) > add(b)) hi = b; else lo = a; }
    var o = 0.5 * (lo + hi); return [add(o), o];
  }
  function qpmThermal(o) {
    var lp = o.pump_wavelength || 0.775e-6, ls = o.signal_wavelength || 1.55e-6, L = o.length || 0.01, dTin = o.dT_in != null ? o.dT_in : 1;
    var ell = o.decay_length || 1, T = o.temperature || 297.65, aL = o.alpha_L != null ? o.alpha_L : 15.4e-6, nz = o.n_z || 2001;
    var d0 = qpmDk(lp, ls, T), slope = (qpmDk(lp, ls, T + 0.5)[0] - qpmDk(lp, ls, T - 0.5)[0]) + d0[0] * aL;
    var z = linspace(0, L, nz), dk = new Float64Array(nz);
    for (var i = 0; i < nz; i++) dk[i] = slope * dTin * Math.exp(-z[i] / ell);
    var rt = bestRetunedFactor(z, dk);
    return { period: 2 * Math.PI / d0[0], idler_wavelength: d0[1], dDk_dT: slope, dT_FWHM: 4 * 1.391557378 / (L * Math.abs(slope)),
      eta_heated: phaseMatchingFactor(z, dk), eta_retuned: rt[0], retune_dT: slope !== 0 ? -rt[1] / slope : 0,
      phase_mismatch: Math.abs(slope) * dTin * ell * (1 - Math.exp(-L / ell)) };
  }
  function ringThermalBistability(o) {
    var lam = o.wavelength || 1.55e-6, Lr = o.ring_length || 2 * Math.PI * 100e-6, ng = o.n_group || 2.3, Qi = o.Q_intrinsic || 1e6, Qc = o.Q_coupling || 1e6;
    var f = o.absorbing_fraction != null ? o.absorbing_fraction : 0.5, R = o.R_th || 0.5, dndT = o.dneff_dT != null ? o.dneff_dT : 3e-5, P = o.power != null ? o.power : 0.01;
    var w = 2 * Math.PI * C0 / lam, ki = w / Qi, ke = w / Qc, kap = ki + ke, kabs = f * ki, Rr = R / Lr, g = w * Math.abs(dndT) / ng * Rr * kabs;
    var Pth = g > 0 ? Math.pow(kap, 3) / (3 * Math.sqrt(3) * g * ke) : Infinity, U = 4 * ke * P / (kap * kap);
    return { Q_loaded: w / kap, linewidth: lam * kap / w, P_threshold: Pth, U_resonance: U, P_abs_resonance: kabs * U, dT_resonance: Rr * kabs * U,
      shift_linewidths: g * U / kap, delta_lambda_max: lam * g * U / w, buildup: 4 * ke * C0 / (kap * kap * ng * Lr), bistable: P > Pth };
  }

  return { REGIONS: REGIONS, LUT: LUT, entry: entry, indexAt: indexAt, gradedAxis: gradedAxis, Band: Band, buildRidge: buildRidge,
    ridgeHeating: ridgeHeating, ridgeMode: ridgeMode, modeWeightedHeating: modeWeightedHeating, gridToMode: gridToMode,
    stripThermalResistance: stripThermalResistance, maxPower: maxPower, kirchhoff: kirchhoff, thermalRunaway: thermalRunaway,
    erAmplifierHeating: erAmplifierHeating, erPumpLimit: erPumpLimit, qpmThermal: qpmThermal, phaseMatchingFactor: phaseMatchingFactor,
    ringThermalBistability: ringThermalBistability, dbToNeper: function (db) { return db * 100 * DB; } };
}
if (typeof module !== 'undefined' && module.exports) module.exports = THERMALENGINE;
