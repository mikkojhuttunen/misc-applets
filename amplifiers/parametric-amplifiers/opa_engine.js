/*
 * opa_engine.js — physics of the fiber parametric amplifier applet, free of any DOM access.
 *
 * This is a JavaScript port of the Python reference engines in
 * math-engines/engines/{fiber_mode,parametric_amplifier,stimulated_scattering}.
 * The applet needs it because its spectra and optimiser call the solvers tens of
 * thousands of times per update. Shared JSON test vectors generated from Python
 * (math-engines/engines/parametric_amplifier/test_vectors.json) keep the two in
 * step: run  node --test amplifiers/parametric-amplifiers/test/
 *
 * Conventions: SI units throughout (m, s, rad, W). process is "chi2" (w_p = w_s + w_i)
 * or "chi3" (2 w_p = w_s + w_i). Losses are power coefficients in 1/m; lumped dump
 * factors (tAmp, tS, tP) are amplitude transmissions.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.OPA = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var HB = 1.054571817e-34, C0 = 299792458, EPS0 = 8.8541878128e-12, NIDX = 1.45, LP = 532e-9;
  var KB = 1.380649e-23, HPL = 6.62607015e-34, VAC = 5960, GB0 = 5e-11;
  var RT1 = 12.2e-15, RT2 = 32e-15;

  // ---------- process kinematics ----------
  function lamSum(process, lamPump) { return process === "chi3" ? lamPump / 2 : lamPump; }
  function idlerOf(process, lamPump, ls) { return 1 / (1 / lamSum(process, lamPump) - 1 / ls); }
  function SGN(process) { return process === "chi3" ? -1 : 1; }
  // Coupling overlap 1/√A (χ2, three Gaussian modes) or 1/A_eff (χ3, f_p² f_s f_i)
  function overlapFor(process, wp, ws, wi) {
    if (process === "chi3") return (Math.PI / (2 / (wp * wp) + 1 / (ws * ws) + 1 / (wi * wi))) / (Math.pow(Math.PI / 2, 2) * wp * wp * ws * wi);
    var sumInv = 1 / (wp * wp) + 1 / (ws * ws) + 1 / (wi * wi);
    return (Math.PI / sumInv) / Math.sqrt(Math.pow(Math.PI / 2, 3) * wp * wp * ws * ws * wi * wi);
  }
  // Small-signal gain coefficient Γ and related quantities (mirrors parametric_amplifier.coupling)
  function coupling(process, lamPump, ls, wp, ws, wi, Pp, deff, n2) {
    var li = idlerOf(process, lamPump, ls), wP = 2 * Math.PI * C0 / lamPump, wS = 2 * Math.PI * C0 / ls, wI = 2 * Math.PI * C0 / li;
    var theta = overlapFor(process, wp, ws, wi), Fp0 = Pp / (HB * wP), K, kappa, gP = 0, gammaP = 0;
    if (process === "chi3") {
      gammaP = n2 * wP / C0 * theta; gP = gammaP * Pp;
      kappa = gP * Math.sqrt(wS * wI) / wP; K = kappa / Math.sqrt(Fp0);
    } else {
      var gam = 2 * deff * Math.sqrt(HB * wP * wS * wI / (2 * EPS0 * C0 * C0 * C0 * NIDX * NIDX * NIDX));
      K = gam * theta; kappa = K * Math.sqrt(Fp0);
    }
    return { kappa: kappa, K: K, gP: gP, gammaP: gammaP, theta: theta, Aeff: process === "chi3" ? 1 / theta : 1 / (theta * theta),
      Fp0: Fp0, li: li, wP: wP, wS: wS, wI: wI, wSum: 2 * Math.PI * C0 / lamSum(process, lamPump) };
  }

  // ---------- fiber dispersion: fused-silica Sellmeier + weakly guiding step-index LP01 ----------
  function nSilica(lam) {
    var x = lam * 1e6, x2 = x * x;
    return Math.sqrt(1 + 0.6961663 * x2 / (x2 - 0.0684043 * 0.0684043) + 0.4079426 * x2 / (x2 - 0.1162414 * 0.1162414) + 0.8974794 * x2 / (x2 - 9.896161 * 9.896161));
  }
  function besJ1overJ0(x) {
    var q = x * x / 4, t0 = 1, t1 = x / 2, j0 = 1, j1 = x / 2;
    for (var m = 1; m < 60; m++) { t0 *= -q / (m * m); t1 *= -q / (m * (m + 1)); j0 += t0; j1 += t1; if (Math.abs(t0) + Math.abs(t1) < 1e-19) break; }
    return j1 / j0;
  }
  // K1(x)/K0(x) from K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt (trapezoid rule converges exponentially)
  function besK1overK0(x) {
    var h = 0.2, s0 = 0.5, s1 = 0.5;
    for (var k = 1; k < 20000; k++) {
      var ch = Math.cosh(k * h), e = Math.exp(-x * (ch - 1));
      s0 += e; s1 += e * ch;
      if (e * ch < 1e-18 * s1) break;
    }
    return s1 / s0;
  }
  var neffCache = new Map();
  function lp01(lam, a, dn) {
    var key = lam.toPrecision(15) + "|" + a + "|" + dn, hit = neffCache.get(key);
    if (hit) return hit;
    var ncl = nSilica(lam), nco = ncl + dn, k0 = 2 * Math.PI / lam, NA2 = nco * nco - ncl * ncl;
    var V = k0 * a * Math.sqrt(NA2);
    function F(U) { var W = Math.sqrt(V * V - U * U); return U * besJ1overJ0(U) - W * besK1overK0(W); }
    var lo = 1e-9, hi = Math.min(V, 2.404825557695773) * (1 - 1e-12), flo = F(lo), fhi = F(hi), side = 0, U = lo;
    for (var it = 0; it < 200; it++) {
      U = (lo * fhi - hi * flo) / (fhi - flo);
      if (!(U > lo && U < hi)) U = 0.5 * (lo + hi);
      var fu = F(U);
      if (fu === 0 || hi - lo < 1e-15 * hi) break;
      if (fu * fhi > 0) { hi = U; fhi = fu; if (side === -1) flo /= 2; side = -1; }
      else { lo = U; flo = fu; if (side === 1) fhi /= 2; side = 1; }
    }
    var b = 1 - U * U / (V * V), neff = Math.sqrt(ncl * ncl + NA2 * b);
    var res = { neff: neff, V: V, w: a * (0.65 + 1.619 / Math.pow(V, 1.5) + 2.879 / Math.pow(V, 6)) };
    if (neffCache.size > 20000) neffCache.clear();
    neffCache.set(key, res);
    return res;
  }

  // Barycentric interpolation on Chebyshev points of the first kind
  function chebInterp(f, lo, hi, N) {
    var xs = new Float64Array(N), fs = new Float64Array(N), ws = new Float64Array(N);
    for (var j = 0; j < N; j++) {
      var th = Math.PI * (j + 0.5) / N;
      xs[j] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * Math.cos(th);
      fs[j] = f(xs[j]); ws[j] = (j % 2 ? -1 : 1) * Math.sin(th);
    }
    return function (x) {
      var num = 0, den = 0;
      for (var j = 0; j < N; j++) { var d = x - xs[j]; if (d === 0) return fs[j]; var t = ws[j] / d; num += t * fs[j]; den += t; }
      return num / den;
    };
  }

  var OM_MAX = 2 * Math.PI * 15e12;
  // Exact-model dispersion over ±15 THz (56-node Chebyshev interpolant of Δk_rel), GVM, β2 sum and Δk at λs
  function fiberDispersion(process, lamPump, ls, a, dn) {
    var wP = 2 * Math.PI * C0 / lamPump, wS = 2 * Math.PI * C0 / ls, wSum = 2 * Math.PI * C0 / lamSum(process, lamPump), wI = wSum - wS;
    var kf = function (w) { return w / C0 * lp01(2 * Math.PI * C0 / w, a, dn).neff; };
    // χ2: Δk_D = k_p − k_s − k_i ; χ3: Δβ = k_s + k_i − 2k_p (both written as sg·(k_Σ − k_s − k_i))
    var sg = SGN(process), kSum = process === "chi3" ? 2 * kf(wP) : kf(wP);
    var dkD = function (O) { var ws = wS + O; return sg * (kSum - kf(ws) - kf(wSum - ws)); };
    var d0 = dkD(0), cheb = chebInterp(function (O) { return dkD(O) - d0; }, -OM_MAX, OM_MAX, 56);
    var h1 = 1e12, h2 = 1e13;
    var b1 = function (w) { return (kf(w + h1) - kf(w - h1)) / (2 * h1); };
    var b2 = function (w) { return (kf(w + h2) - 2 * kf(w) + kf(w - h2)) / (h2 * h2); };
    return { model: "fiber", dkRel: cheb, gvm: b1(wI) - b1(wS), b2: b2(wS) + b2(wI), dk0: d0,
      V: [lp01(lamPump, a, dn).V, lp01(ls, a, dn).V, lp01(2 * Math.PI * C0 / wI, a, dn).V] };
  }
  function taylorDispersion(process, gvm, b2) {
    var sg = SGN(process);
    return { model: "taylor", dkRel: function (O) { return sg * (gvm * O - 0.5 * b2 * O * O); }, gvm: gvm, b2: b2 };
  }
  // Quartic Lagrange interpolant of Δk_rel(Ω) through Ω = 0, ±1, ±2 THz·2π (error < 1e-4 rad/m out to ±4 THz)
  function fastDisp(process, lamPump, ls, a, dn) {
    var wS = 2 * Math.PI * C0 / ls, wP = 2 * Math.PI * C0 / lamPump, wSum = 2 * Math.PI * C0 / lamSum(process, lamPump), sg = SGN(process);
    var kf = function (w) { return w / C0 * lp01(2 * Math.PI * C0 / w, a, dn).neff; };
    var kp = (process === "chi3" ? 2 : 1) * kf(wP), dkD = function (O) { var ws = wS + O; return sg * (kp - kf(ws) - kf(wSum - ws)); };
    var H = 2 * Math.PI * 1e12, xs = [-2 * H, -H, 0, H, 2 * H], d0 = dkD(0);
    var ys = xs.map(function (x) { return x === 0 ? 0 : dkD(x) - d0; });
    var dkRel = function (O) {
      var sum = 0;
      for (var i = 0; i < 5; i++) { var t = ys[i]; if (t === 0) continue; for (var j = 0; j < 5; j++) if (j !== i) t *= (O - xs[j]) / (xs[i] - xs[j]); sum += t; }
      return sum;
    };
    var gvm = sg * (ys[3] - ys[1]) / (2 * H), curv = (ys[3] + ys[1]) / (H * H);
    return { model: "fiber", dkRel: dkRel, gvm: gvm, b2: -sg * curv, dk0: d0, Vs: lp01(ls, a, dn).V };
  }
  function fiberModes(lamPump, ls, li, a, dn) { return { wp: lp01(lamPump, a, dn).w, ws: lp01(ls, a, dn).w, wi: lp01(li, a, dn).w }; }

  // ---------- spectral loss profile (0..1 at an idler wavelength in nm) ----------
  // Custom profile: control points {x: idler nm, y: 0..1}, monotone cubic (Fritsch–Carlson) interpolation
  function pchip(pts) {
    var P2 = pts.slice().sort(function (a, b) { return a.x - b.x; }), n = P2.length;
    var xs = P2.map(function (p) { return p.x; }), ys = P2.map(function (p) { return p.y; }), d = [], m = [], k;
    for (k = 0; k < n - 1; k++) d.push((ys[k + 1] - ys[k]) / Math.max(1e-15, xs[k + 1] - xs[k]));
    for (k = 0; k < n; k++) {
      if (k === 0) m.push(n > 1 ? d[0] : 0);
      else if (k === n - 1) m.push(d[n - 2]);
      else m.push(d[k - 1] * d[k] <= 0 ? 0 : (d[k - 1] + d[k]) / 2);
    }
    for (k = 0; k < n - 1; k++) {
      if (d[k] === 0) { m[k] = m[k + 1] = 0; continue; }
      var a = m[k] / d[k], b = m[k + 1] / d[k], h = a * a + b * b;
      if (h > 9) { var t = 3 / Math.sqrt(h); m[k] = t * a * d[k]; m[k + 1] = t * b * d[k]; }
    }
    return function (x) {
      if (n === 0) return 0;
      if (x <= xs[0]) return ys[0];
      if (x >= xs[n - 1]) return ys[n - 1];
      var lo = 0, hi = n - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] > x) hi = mid; else lo = mid; }
      var hh = xs[hi] - xs[lo], t2 = (x - xs[lo]) / hh, t3 = t2 * t2 * t2, tt = t2 * t2;
      return (2 * t3 - 3 * tt + 1) * ys[lo] + (t3 - 2 * tt + t2) * hh * m[lo] + (-2 * t3 + 3 * tt) * ys[hi] + (t3 - tt) * hh * m[hi];
    };
  }
  function lossT(pr, lamNm) {
    if (pr.type === "flat") return 1;
    if (pr.type === "custom") return Math.max(0, Math.min(1, (pr.fn || function () { return 0; })(lamNm)));
    var a = pr.c - pr.w / 2, b = pr.c + pr.w / 2;
    var box = 0.5 * (Math.tanh((lamNm - a) / pr.e) - Math.tanh((lamNm - b) / pr.e));
    box = Math.max(0, Math.min(1, box));
    return pr.type === "pass" ? box : 1 - box;
  }
  // Loss of signal and pump from the same spectral curve (real filter); zero when disabled or the profile is flat
  function waveLoss(enabled, prof, lm, aPk, attDb, lamS, lamP) {
    if (!enabled || lm === "none" || prof.type === "flat") return { aS: 0, aP: 0, tS: 1, tP: 1, TS: 0, TP: 0 };
    var TS = lossT(prof, lamS * 1e9), TP = lossT(prof, lamP * 1e9);
    return { aS: aPk * TS, aP: aPk * TP, tS: lm === "lumped" ? Math.pow(10, -attDb * TS / 20) : 1, tP: lm === "lumped" ? Math.pow(10, -attDb * TP / 20) : 1, TS: TS, TP: TP };
  }

  // ---------- coupled-wave solvers ----------
  // χ2: RK4 in normalised amplitudes u_j = a_j / sqrt(Fp0)
  function simulate2(p, dk, record, maxSteps, sc) {
    var kap = p.kappa, a2 = p.alpha / 2, r0 = p.r0, wlz = p.wl || { aS: 0, aP: 0, tS: 1, tP: 1 }, a2s = wlz.aS / 2, a2p = wlz.aP / 2;
    var rate = Math.max(kap * Math.sqrt(1 + r0), Math.abs(dk), a2, sc ? sc.rate : 0, 1e-6);
    var N = Math.ceil(p.L * rate / 0.04);
    N = Math.max(record ? 2000 : 300, Math.min(maxSteps, N));
    var nSeg = p.nf + 1, nPer = Math.ceil(N / nSeg), h = p.L / (nSeg * nPer);
    var y = new Float64Array([1, 0, Math.sqrt(r0), 0, 0, 0, sc && sc.trackR ? sc.x0R : 0]);
    var k1 = new Float64Array(7), k2 = new Float64Array(7), k3 = new Float64Array(7), k4 = new Float64Array(7), t = new Float64Array(7);
    function f(z, Y, o) {
      var c = Math.cos(dk * z), s = Math.sin(dk * z);
      var pr = Y[0], pi = Y[1], sr = Y[2], si = Y[3], ir = Y[4], ii = Y[5], xr, xi, yr, yi;
      xr = sr * ir - si * ii; xi = sr * ii + si * ir;
      yr = xr * c + xi * s; yi = xi * c - xr * s;
      o[0] = -kap * yi; o[1] = kap * yr;
      xr = pr * ir + pi * ii; xi = pi * ir - pr * ii;
      yr = xr * c - xi * s; yi = xr * s + xi * c;
      o[2] = -kap * yi; o[3] = kap * yr;
      xr = pr * sr + pi * si; xi = pi * sr - pr * si;
      yr = xr * c - xi * s; yi = xr * s + xi * c;
      o[4] = -kap * yi - a2 * ir; o[5] = kap * yr - a2 * ii;
      if (a2s) { o[2] -= a2s * Y[2]; o[3] -= a2s * Y[3]; }
      if (a2p) { o[0] -= a2p * Y[0]; o[1] -= a2p * Y[1]; }
      if (sc) addScat(sc, z, Y, o); else o[6] = 0;
    }
    var rec = record ? { z: [], seg: [], fp: [], fs: [], fi: [], fR: [] } : null;
    var every = Math.max(1, Math.floor(nSeg * nPer / 900));
    function push(z, sgi) { rec.z.push(z); rec.seg.push(sgi); rec.fp.push(y[0] * y[0] + y[1] * y[1]); rec.fs.push(y[2] * y[2] + y[3] * y[3]); rec.fi.push(y[4] * y[4] + y[5] * y[5]); rec.fR.push(y[6]); }
    var z = 0, cnt = 0, j;
    if (record) push(0, 0);
    for (var sg = 0; sg < nSeg; sg++) {
      for (var n = 0; n < nPer; n++) {
        f(z, y, k1);
        for (j = 0; j < 7; j++) t[j] = y[j] + 0.5 * h * k1[j];
        f(z + 0.5 * h, t, k2);
        for (j = 0; j < 7; j++) t[j] = y[j] + 0.5 * h * k2[j];
        f(z + 0.5 * h, t, k3);
        for (j = 0; j < 7; j++) t[j] = y[j] + h * k3[j];
        f(z + h, t, k4);
        for (j = 0; j < 7; j++) y[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        cnt++;
        z = (sg * nPer + n + 1) * h;
        if (record && (cnt % every === 0 || n === nPer - 1)) push(z, sg);
      }
      if (sg < nSeg - 1) { y[4] *= p.tAmp; y[5] *= p.tAmp; y[2] *= wlz.tS; y[3] *= wlz.tS; y[0] *= wlz.tP; y[1] *= wlz.tP; if (record) push(z, sg + 1); }
    }
    var fp = y[0] * y[0] + y[1] * y[1], fs = y[2] * y[2] + y[3] * y[3], fi = y[4] * y[4] + y[5] * y[5];
    return { rec: rec, fp: fp, fs: fs, fi: fi, xR: y[6], steps: nSeg * nPer, capped: N === maxSteps, resid: Math.abs(fp + fs - (1 + r0)) / (1 + r0) };
  }

  function simulate3(p, dkNet, record, maxSteps, sc) {
    var wpf = p.wP, wsf = p.wS, wif = p.wI, np = p.gP, ns = np * wsf / wpf, ni = np * wif / wpf, a2 = p.alpha / 2, r0 = p.r0, wlz = p.wl || { aS: 0, aP: 0, tS: 1, tP: 1 }, a2s = wlz.aS / 2, a2p = wlz.aP / 2;
    var db = dkNet - 2 * np;
    var rate = Math.max(p.kappa * Math.sqrt(1 + r0), Math.abs(db), 3 * np * (1 + r0), a2, sc ? sc.rate : 0, 1e-6);
    var N = Math.ceil(p.L * rate / 0.04);
    N = Math.max(record ? 2000 : 300, Math.min(maxSteps, N));
    var nSeg = p.nf + 1, nPer = Math.ceil(N / nSeg), h = p.L / (nSeg * nPer);
    var y = new Float64Array([1, 0, Math.sqrt(r0 * wsf / wpf), 0, 0, 0, sc && sc.trackR ? sc.x0R : 0]);
    var k1 = new Float64Array(7), k2 = new Float64Array(7), k3 = new Float64Array(7), k4 = new Float64Array(7), t = new Float64Array(7);
    function f(z, Y, o) {
      var c = Math.cos(db * z), sn = Math.sin(db * z);
      var pr = Y[0], pi = Y[1], sr = Y[2], si = Y[3], ir = Y[4], ii = Y[5];
      var Pp2 = pr * pr + pi * pi, Ps2 = sr * sr + si * si, Pi2 = ir * ir + ii * ii;
      // pump: n_p[(|p|²+2|s|²+2|i|²) p + 2 s i p* e^{iΔβz}]
      var xr = sr * ir - si * ii, xi = sr * ii + si * ir;            // s·i
      var yr = xr * pr + xi * pi, yi = xi * pr - xr * pi;            // ·p*
      var er = yr * c - yi * sn, ei = yr * sn + yi * c;              // ·e^{+iΔβz}
      var Ar = (Pp2 + 2 * Ps2 + 2 * Pi2) * pr + 2 * er, Ai = (Pp2 + 2 * Ps2 + 2 * Pi2) * pi + 2 * ei;
      o[0] = -np * Ai; o[1] = np * Ar;
      // p² e^{−iΔβz}
      var qr = pr * pr - pi * pi, qi = 2 * pr * pi, gr = qr * c + qi * sn, gi = qi * c - qr * sn;
      // signal: n_s[(|s|²+2|p|²+2|i|²) s + p² i* e^{−iΔβz}]
      var Br = (Ps2 + 2 * Pp2 + 2 * Pi2) * sr + (gr * ir + gi * ii), Bi = (Ps2 + 2 * Pp2 + 2 * Pi2) * si + (gi * ir - gr * ii);
      o[2] = -ns * Bi; o[3] = ns * Br;
      var Cr = (Pi2 + 2 * Pp2 + 2 * Ps2) * ir + (gr * sr + gi * si), Ci = (Pi2 + 2 * Pp2 + 2 * Ps2) * ii + (gi * sr - gr * si);
      o[4] = -ni * Ci - a2 * ir; o[5] = ni * Cr - a2 * ii;
      if (a2s) { o[2] -= a2s * Y[2]; o[3] -= a2s * Y[3]; }
      if (a2p) { o[0] -= a2p * Y[0]; o[1] -= a2p * Y[1]; }
      if (sc) addScat(sc, z, Y, o); else o[6] = 0;
    }
    var cs_ = wpf / wsf, ci_ = wpf / wif;
    var rec = record ? { z: [], seg: [], fp: [], fs: [], fi: [], fR: [] } : null;
    var every = Math.max(1, Math.floor(nSeg * nPer / 900));
    function push(z, sgi) { rec.z.push(z); rec.seg.push(sgi); rec.fp.push(y[0] * y[0] + y[1] * y[1]); rec.fs.push((y[2] * y[2] + y[3] * y[3]) * cs_); rec.fi.push((y[4] * y[4] + y[5] * y[5]) * ci_); rec.fR.push(y[6]); }
    var z = 0, cnt = 0, j;
    if (record) push(0, 0);
    for (var sg = 0; sg < nSeg; sg++) {
      for (var n = 0; n < nPer; n++) {
        f(z, y, k1);
        for (j = 0; j < 7; j++) t[j] = y[j] + 0.5 * h * k1[j];
        f(z + 0.5 * h, t, k2);
        for (j = 0; j < 7; j++) t[j] = y[j] + 0.5 * h * k2[j];
        f(z + 0.5 * h, t, k3);
        for (j = 0; j < 7; j++) t[j] = y[j] + h * k3[j];
        f(z + h, t, k4);
        for (j = 0; j < 7; j++) y[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        cnt++;
        z = (sg * nPer + n + 1) * h;
        if (record && (cnt % every === 0 || n === nPer - 1)) push(z, sg);
      }
      if (sg < nSeg - 1) { y[4] *= p.tAmp; y[5] *= p.tAmp; y[2] *= wlz.tS; y[3] *= wlz.tS; y[0] *= wlz.tP; y[1] *= wlz.tP; if (record) push(z, sg + 1); }
    }
    var fp = y[0] * y[0] + y[1] * y[1], fs = (y[2] * y[2] + y[3] * y[3]) * cs_, fi = (y[4] * y[4] + y[5] * y[5]) * ci_;
    return { rec: rec, fp: fp, fs: fs, fi: fi, xR: y[6], steps: nSeg * nPer, capped: N === maxSteps, resid: Math.abs(fp + 2 * fs - (1 + 2 * r0)) / (1 + 2 * r0) };
  }
  function simulate(process, p, dk, record, maxSteps, sc) {
    return process === "chi3" ? simulate3(p, dk, record, maxSteps, sc) : simulate2(p, dk, record, maxSteps, sc);
  }

  // ---------- stimulated Brillouin and Raman scattering (silica) ----------
  // Raman response h(t) = A e^{-t/τ2} sin(t/τ1) (Blow & Wood); gain ∝ Im H(Ω), odd in Ω (Stokes gain, anti-Stokes loss)
  function ramanIm(O) { var a = 1 / RT2, b = 1 / RT1, X = a * a + b * b - O * O, Y = 2 * a * O; return 2 * a * b * O / (X * X + Y * Y); }
  var RAMAN_PEAK = (function () { var best = 0, Ob = 0; for (var k = 1; k < 4000; k++) { var O = 2 * Math.PI * 40e12 * k / 4000, v = ramanIm(O); if (v > best) { best = v; Ob = O; } } return { v: best, O: Ob }; })();
  // the single-oscillator model has long tails; silica gain is negligible beyond ~40 THz, so roll it off smoothly
  function ramanShape(O) {
    var nu = Math.abs(O) / 2 / Math.PI / 1e12, win = nu > 32 ? Math.exp(-Math.pow((nu - 32) / 5, 2)) : 1;
    return ramanIm(O) / RAMAN_PEAK.v * win;
  }
  function gRpeak(lam) { return 1e-13 * (1e-6 / lam); }
  function brillouin(lamP, lwHz) {
    var nuB = 2 * NIDX * VAC / lamP, dnuB = 20e6 * Math.pow(1550e-9 / lamP, 2);
    var g = GB0 * dnuB / (dnuB + lwHz), nth = KB * 293 / (HPL * nuB);
    return { nuB: nuB, dnuB: dnuB, g: g, seed: HPL * C0 / lamP * nth * dnuB };
  }
  // Exact CW two-wave SBS: Pp − PB = C along z, PB(L) = seed. Returns PB(z).
  function sbsSolve(P0, gA, L, seed) {
    function pbL(D) { var C = P0 - D, q = D / P0, e = q * Math.exp(-gA * C * L); return C * e / (1 - e); }
    var lo = Math.log(seed * 1e-6), hi = Math.log(P0 * (1 - 1e-9));
    for (var it = 0; it < 200; it++) { var m = 0.5 * (lo + hi); if (pbL(Math.exp(m)) > seed) hi = m; else lo = m; }
    var D = Math.exp(0.5 * (lo + hi)), C = P0 - D, q = D / P0;
    return { D: D, C: C, PL: C / (1 - q * Math.exp(-gA * C * L)), PB: function (z) { var e = q * Math.exp(-gA * C * z); return C * e / (1 - e); } };
  }
  // Scattering terms for one solver run. opts: {process, sbs, srs, linewidthHz}. lossTable: pump loss into the
  // Raman Stokes taken from a main run (spectral runs); without it the Raman Stokes wave is integrated.
  function buildScattering(p, q, lossTable, opts) {
    var sbs = !!opts.sbs, srs = !!opts.srs;
    if (!sbs && !srs) return null;
    var w = [q.wP, q.wS, q.wI], rad = [p.wp, p.ws, p.wi], P0 = p.Pp, Ap = Math.PI * p.wp * p.wp, rate = 0;
    var sc = { P0: P0, w: w, wt: opts.process === "chi3" ? [1, 1, 1] : [1, w[1] / w[0], w[2] / w[0]], pairs: [], trackR: false, sbs: null };
    if (srs) {
      [[0, 1], [0, 2], [1, 2]].forEach(function (pr) {
        var j = w[pr[0]] > w[pr[1]] ? pr[0] : pr[1], k = j === pr[0] ? pr[1] : pr[0];
        var g = gRpeak(2 * Math.PI * C0 / w[j]) * ramanShape(w[j] - w[k]) / (Math.PI * (rad[j] * rad[j] + rad[k] * rad[k]) / 2);
        sc.pairs.push({ j: j, k: k, c: g, r: w[j] / w[k] });
        rate = Math.max(rate, Math.abs(g) * P0 * w[j] / w[k]);
      });
      var wR = w[0] - RAMAN_PEAK.O;
      sc.gRp = gRpeak(2 * Math.PI * C0 / w[0]) / Ap; sc.ratioPR = w[0] / wR; sc.lamR = 2 * Math.PI * C0 / wR;
      sc.thrR = 16 * Ap / (gRpeak(2 * Math.PI * C0 / w[0]) * p.L);
      if (!lossTable) { sc.trackR = true; sc.x0R = HPL * wR / (2 * Math.PI) * 1e12 / P0; }
      rate = Math.max(rate, sc.gRp * P0);
    }
    var lossB = null;
    if (sbs) {
      var br = brillouin(2 * Math.PI * C0 / w[0], opts.linewidthHz || 0), gA = br.g / Ap, sol = sbsSolve(P0, gA, p.L, br.seed);
      sc.sbs = { br: br, sol: sol, gA: gA, thr: 21 * Ap / (br.g * p.L) };
      lossB = function (z) { return gA * sol.PB(z); };
      rate = Math.max(rate, gA * sol.D);
    }
    var tab = lossTable;
    sc.lossAt = function (z) {
      var v = lossB ? lossB(z) : 0;
      if (tab) {
        var a = 0, b = tab.z.length - 1;
        if (z <= tab.z[0]) v += tab.v[0]; else if (z >= tab.z[b]) v += tab.v[b];
        else { while (b - a > 1) { var m = (a + b) >> 1; if (tab.z[m] > z) b = m; else a = m; } var t = (z - tab.z[a]) / Math.max(1e-30, tab.z[b] - tab.z[a]); v += tab.v[a] + t * (tab.v[b] - tab.v[a]); }
      }
      return v;
    };
    sc.rate = rate;
    return sc;
  }
  function addScat(sc, z, Y, o) {
    var P0 = sc.P0, Pw = [P0 * sc.wt[0] * (Y[0] * Y[0] + Y[1] * Y[1]), P0 * sc.wt[1] * (Y[2] * Y[2] + Y[3] * Y[3]), P0 * sc.wt[2] * (Y[4] * Y[4] + Y[5] * Y[5])];
    for (var n = 0; n < sc.pairs.length; n++) {
      var pr = sc.pairs[n], j = pr.j, k = pr.k, gk = 0.5 * pr.c * Pw[j], gj = -0.5 * pr.c * pr.r * Pw[k];
      o[2 * k] += gk * Y[2 * k]; o[2 * k + 1] += gk * Y[2 * k + 1];
      o[2 * j] += gj * Y[2 * j]; o[2 * j + 1] += gj * Y[2 * j + 1];
    }
    var l = sc.lossAt(z);
    if (sc.trackR) { o[6] = sc.gRp * Pw[0] * Y[6]; l += sc.ratioPR * sc.gRp * P0 * Y[6]; } else o[6] = 0;
    o[0] -= 0.5 * l * Y[0]; o[1] -= 0.5 * l * Y[1];
  }

  // ---------- undepleted-pump analytic solution ----------
  // complex helpers [re, im]
  function cm(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]; }
  function ca(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
  function cs(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
  function cd(a, b) { var d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; }
  function ce(a) { var e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; }
  function csq(a) { var r = Math.hypot(a[0], a[1]); var re = Math.sqrt((r + a[0]) / 2), im = Math.sqrt(Math.max(0, (r - a[0]) / 2)); return [re, a[1] < 0 ? -im : im]; }
  function cr(x) { return [x, 0]; }

  // Undepleted pump: d/dz [s, c] = M [s, c], c = a_i* e^{iΔkz}, M = [[as, iκ], [−iκ, β]], β = iΔk − α/2, as = −α_s/2
  function expM(kap, beta, z, as) {
    as = as || 0;
    var tr = [beta[0] + as, beta[1]], df = [beta[0] - as, beta[1]];
    var disc = csq(ca(cm(df, df), cr(4 * kap * kap)));
    var l1 = [(tr[0] + disc[0]) / 2, (tr[1] + disc[1]) / 2], l2 = [(tr[0] - disc[0]) / 2, (tr[1] - disc[1]) / 2];
    if (Math.max(l1[0], l2[0]) * z > 600) return null;
    var M = [[cr(as), [0, kap]], [[0, -kap], beta]];
    if (Math.hypot(disc[0], disc[1]) * z < 1e-7) {
      var l = [tr[0] / 2, tr[1] / 2], el = ce([l[0] * z, l[1] * z]);
      return [[cm(el, ca(cr(1), cm(cs(M[0][0], l), cr(z)))), cm(el, cm(M[0][1], cr(z)))],
              [cm(el, cm(M[1][0], cr(z))), cm(el, ca(cr(1), cm(cs(M[1][1], l), cr(z))))]];
    }
    var e1 = ce([l1[0] * z, l1[1] * z]), e2 = ce([l2[0] * z, l2[1] * z]);
    var E = [[0, 0], [0, 0]];
    for (var r = 0; r < 2; r++) for (var q = 0; q < 2; q++) {
      var a = M[r][q], b = M[r][q];
      if (r === q) { a = cs(a, l2); b = cs(b, l1); }
      E[r][q] = cd(cs(cm(e1, a), cm(e2, b)), disc);
    }
    return E;
  }
  function apply(E, v) { return [ca(cm(E[0][0], v[0]), cm(E[0][1], v[1])), ca(cm(E[1][0], v[0]), cm(E[1][1], v[1]))]; }

  function analyticOut(p, dk) {
    var beta = [-p.alpha / 2, dk], nSeg = p.nf + 1, dz = p.L / nSeg, wl = p.wl || { aS: 0, tS: 1 };
    var E = expM(p.kappa, beta, dz, -wl.aS / 2); if (!E) return null;
    var v = [cr(Math.sqrt(p.r0)), cr(0)];
    for (var sg = 0; sg < nSeg; sg++) {
      v = apply(E, v);
      if (sg < nSeg - 1) { v[1] = cm(v[1], cr(p.tAmp)); v[0] = cm(v[0], cr(wl.tS)); }
    }
    return { fs: v[0][0] * v[0][0] + v[0][1] * v[0][1], fi: v[1][0] * v[1][0] + v[1][1] * v[1][1] };
  }
  function analyticAlongZ(p, zs, segs) {
    var beta = [-p.alpha / 2, p.dk], nSeg = p.nf + 1, dz = p.L / nSeg, wl = p.wl || { aS: 0, tS: 1 }, as = -wl.aS / 2;
    var starts = [], v = [cr(Math.sqrt(p.r0)), cr(0)], E = expM(p.kappa, beta, dz, as);
    for (var sg = 0; sg < nSeg; sg++) {
      starts.push(v);
      if (!E) break;
      var w = apply(E, v); w[1] = cm(w[1], cr(p.tAmp)); w[0] = cm(w[0], cr(wl.tS)); v = w;
    }
    var out = { fs: [], fi: [] };
    for (var k = 0; k < zs.length; k++) {
      var s = segs[k], loc = Math.max(0, Math.min(dz, zs[k] - s * dz));
      var Ez = starts[s] ? expM(p.kappa, beta, loc, as) : null;
      if (!Ez) { out.fs.push(Infinity); out.fi.push(Infinity); continue; }
      var u = apply(Ez, starts[s]);
      out.fs.push(u[0][0] * u[0][0] + u[0][1] * u[0][1]); out.fi.push(u[1][0] * u[1][0] + u[1][1] * u[1][1]);
    }
    return out;
  }

  return {
    HB: HB, C0: C0, EPS0: EPS0, NIDX: NIDX, LP: LP, KB: KB, HPL: HPL, OM_MAX: OM_MAX, RAMAN_PEAK: RAMAN_PEAK,
    lamSum: lamSum, idlerOf: idlerOf, SGN: SGN, overlapFor: overlapFor, coupling: coupling,
    nSilica: nSilica, lp01: lp01, chebInterp: chebInterp, fiberDispersion: fiberDispersion, taylorDispersion: taylorDispersion,
    fastDisp: fastDisp, fiberModes: fiberModes, pchip: pchip, lossT: lossT, waveLoss: waveLoss,
    simulate: simulate, simulate2: simulate2, simulate3: simulate3,
    ramanShape: ramanShape, gRpeak: gRpeak, brillouin: brillouin, sbsSolve: sbsSolve, buildScattering: buildScattering, addScat: addScat,
    expM: expM, analyticOut: analyticOut, analyticAlongZ: analyticAlongZ
  };
});
