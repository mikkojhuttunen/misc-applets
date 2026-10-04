/* Validation suite for nlo-engine.js.  Run:  node test/run-tests.js
   No dependencies. Exit code 1 if any check fails. */
'use strict';
const NLO = require('../nlo-engine.js');
const LEG = require('./legacy/applet_engine_a74e7d9.js');
const { analysis, nl } = NLO;

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) pass++; else fail++;
  console.log((ok ? 'PASS  ' : 'FAIL  ') + name + (detail ? '   [' + detail + ']' : ''));
}
const fmt = x => (typeof x === 'number' ? x.toExponential(2) : String(x));

/* ---------------- FFT ---------------- */
{
  const N = 64, rng = NLO.makeRng(1), re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) { re[i] = rng.randn(); im[i] = rng.randn(); }
  const r = re.slice(), m = im.slice(); NLO.fft(r, m, false);
  let err = 0;
  for (let k = 0; k < N; k++) {
    let sr = 0, si = 0;
    for (let n = 0; n < N; n++) { const a = -2 * Math.PI * k * n / N; sr += re[n] * Math.cos(a) - im[n] * Math.sin(a); si += re[n] * Math.sin(a) + im[n] * Math.cos(a); }
    err = Math.max(err, Math.hypot(sr - r[k], si - m[k]));
  }
  check('FFT matches naive DFT', err < 1e-12, 'max err ' + fmt(err));
  let e1 = 0, e2 = 0; for (let i = 0; i < N; i++) e1 += re[i] * re[i] + im[i] * im[i];
  for (let k = 0; k < N; k++) e2 += r[k] * r[k] + m[k] * m[k];
  check('FFT Parseval', Math.abs(e2 / N - e1) < 1e-10 * e1, 'rel ' + fmt(Math.abs(e2 / N - e1) / e1));
  NLO.fft(r, m, true);
  let rt = 0; for (let i = 0; i < N; i++) rt = Math.max(rt, Math.hypot(r[i] - re[i], m[i] - im[i]));
  check('FFT inverse round trip', rt < 1e-13, fmt(rt));
}

/* ---------------- ODE ---------------- */
{
  // nonlinear pendulum  y'' = -sin y  (y0 = 2, v0 = 0)
  const f = (t, y, d) => { d[0] = y[1]; d[1] = -Math.sin(y[0]); };
  const ref = NLO.integrate(f, [2, 0], { t1: 10, method: 'rk45', rtol: 1e-13, atol: 1e-14, dt: 0.01 }).y;
  const e = dt => { const y = NLO.integrate(f, [2, 0], { t1: 10, method: 'rk4', dt: dt }).y; return Math.hypot(y[0] - ref[0], y[1] - ref[1]); };
  const e1 = e(0.0125), e2 = e(0.00625), order = Math.log2(e1 / e2);
  check('RK4 is 4th order (pendulum)', order > 3.85 && order < 4.15, 'observed order ' + order.toFixed(2));
  const H = y => 0.5 * y[1] * y[1] - Math.cos(y[0]);
  const r = NLO.integrate(f, [2, 0], { t1: 50, method: 'rk45', rtol: 1e-10, atol: 1e-12, dt: 0.1 });
  check('RK45 conserves pendulum energy', Math.abs(H(r.y) - H([2, 0])) < 1e-7, 'dH ' + fmt(Math.abs(H(r.y) - H([2, 0]))) + ', ' + r.steps + ' steps');
  const ho = NLO.integrate((t, y, d) => { d[0] = y[1]; d[1] = -y[0]; }, [1, 0], { t1: 20, method: 'rk45', rtol: 1e-10, atol: 1e-12, dt: 0.1 });
  check('RK45 harmonic oscillator vs cos(t)', Math.abs(ho.y[0] - Math.cos(20)) < 1e-8, fmt(Math.abs(ho.y[0] - Math.cos(20))));
}

/* ---------------- analysis ---------------- */
{
  const g = NLO.makeGrid(1024, { dt: 1 }), T0 = 10;
  const A = analysis.sech(g, T0);
  const m = analysis.pulseMetrics(A.re, A.im, 1), exact = 2 * Math.log(1 + Math.SQRT2) * T0;
  check('FWHM of sech(t/T0) = 1.7627 T0', Math.abs(m.fwhm - exact) / exact < 3e-3, m.fwhm.toFixed(3) + ' vs ' + exact.toFixed(3));
  // pulse straddling the periodic window edge
  const sh = 1024 / 2 - 8, B = { re: new Float64Array(1024), im: new Float64Array(1024) };
  for (let i = 0; i < 1024; i++) B.re[(i + sh) % 1024] = A.re[i];   // peak moved to index 8
  const mb = analysis.pulseMetrics(B.re, B.im, 1);
  check('FWHM correct when pulse straddles window edge', Math.abs(mb.fwhm - exact) / exact < 3e-3, mb.fwhm.toFixed(3));
  // spectrum / chirp sign convention: exp(-i w0 T) carries frequency +w0
  const w0 = 0.7, C = analysis.sech(g, 30);
  for (let i = 0; i < 1024; i++) { const c = Math.cos(w0 * g.t[i]), s = -Math.sin(w0 * g.t[i]), r = C.re[i]; C.re[i] = r * c; C.im[i] = r * s; }
  const sp = analysis.spectrum(C.re, C.im, g); let ip = 0; for (let j = 0; j < 1024; j++) if (sp.power[j] > sp.power[ip]) ip = j;
  check('spectrum peak at +w0 for exp(-i w0 T)', Math.abs(sp.omega[ip] - w0) <= g.dw, 'peak ' + sp.omega[ip].toFixed(4) + ', w0 ' + w0);
  const ch = analysis.chirp(C.re, C.im, 1);
  check('chirp() returns +w0 for exp(-i w0 T)', Math.abs(ch[512] - w0) < 1e-3, ch[512].toFixed(5));
}

/* ---------------- NLSE: linear physics ---------------- */
{
  // Gaussian pulse, beta2 only: width grows as sqrt(1 + (z/LD)^2)
  const s = new NLO.NLSE({ N: 1024, T: 200, linear: { beta: [1] }, method: 'strang' });
  const A = analysis.gaussian(s.grid, 1);
  const w0 = analysis.pulseMetrics(A.re, A.im, s.grid.dt).rms;
  s.propagate(A, { dz: 0.05, steps: 40 });                      // z = 2, LD = 1
  const w1 = analysis.pulseMetrics(A.re, A.im, s.grid.dt).rms, exp = Math.sqrt(1 + 4);
  check('Gaussian dispersive broadening sqrt(1+(z/LD)^2)', Math.abs(w1 / w0 - exp) / exp < 2e-3, (w1 / w0).toFixed(4) + ' vs ' + exp.toFixed(4));

  // group delay sign: custom i*b1*w  ->  A(T,z) = A0(T - b1 z)
  const s2 = new NLO.NLSE({ N: 1024, T: 100, linear: { custom: w => [0, 0.5 * w] } });
  const B = analysis.sech(s2.grid, 1); s2.propagate(B, { dz: 1, steps: 6 });   // z = 6 -> shift +3
  let m0 = 0, m1 = 0;   // plain (un-rolled) first moment; pulse stays far from the window edge
  for (let i = 0; i < 1024; i++) { const I = B.re[i] * B.re[i] + B.im[i] * B.im[i]; m0 += I; m1 += I * s2.grid.t[i]; }
  const c = m1 / m0;
  check('odd-order term moves pulse to +T (Agrawal sign)', Math.abs(c - 3) < 1e-3, 'centroid shift ' + c.toFixed(4));

  // linear loss: E(z) = E0 exp(-alpha z), with Kerr switched on (phase only)
  const s3 = new NLO.NLSE({ N: 512, T: 40, linear: { alpha: 0.5, beta: [-1] }, gamma: 1 });
  const D = analysis.sech(s3.grid, 1, 1), E0 = analysis.energy(D.re, D.im, s3.grid.dt);
  s3.propagate(D, { dz: 0.01, steps: 300 });
  const E1 = analysis.energy(D.re, D.im, s3.grid.dt), want = E0 * Math.exp(-0.5 * 3);
  check('energy decays as exp(-alpha z)', Math.abs(E1 / want - 1) < 1e-9, 'rel ' + fmt(Math.abs(E1 / want - 1)));
}

/* ---------------- NLSE: solitons ---------------- */
{
  const s = new NLO.NLSE({ N: 1024, T: 60, linear: { beta: [-1] }, gamma: 1 });
  const A = analysis.sech(s.grid, 1), E0 = analysis.energy(A.re, A.im, s.grid.dt);
  s.propagate(A, { dz: 0.01, steps: 300 });                         // z = 3
  const ex = { re: new Float64Array(1024), im: new Float64Array(1024) };
  for (let i = 0; i < 1024; i++) { const a = 1 / Math.cosh(s.grid.t[i]); ex.re[i] = a * Math.cos(1.5); ex.im[i] = a * Math.sin(1.5); }   // sech * exp(i z/2)
  const d = analysis.maxDiff(A, ex);
  check('fundamental soliton: shape AND phase exp(iz/2) preserved', d < 1e-4, 'max |dA| ' + fmt(d));
  check('lossless NLSE conserves energy', Math.abs(analysis.energy(A.re, A.im, s.grid.dt) / E0 - 1) < 1e-12, fmt(Math.abs(analysis.energy(A.re, A.im, s.grid.dt) / E0 - 1)));

  // N = 2 soliton recurs after z0 = pi/2 * LD
  const s2 = new NLO.NLSE({ N: 1024, T: 40, linear: { beta: [-1] }, gamma: 1 });
  const B = analysis.sech(s2.grid, 1, 2), I0 = Float64Array.from(B.re, x => x * x);
  let Imax = 0, zAtMax = 0;
  s2.propagate(B, { dz: Math.PI / 2 / 800, steps: 800, every: 10, onSample: (z, re, im) => { let p = 0; for (let i = 0; i < re.length; i++) p = Math.max(p, re[i] * re[i] + im[i] * im[i]); if (z > 0.1 && z < Math.PI / 2 - 0.1 && p > Imax) { Imax = p; zAtMax = z; } } });
  let dI = 0; for (let i = 0; i < 1024; i++) dI = Math.max(dI, Math.abs(B.re[i] * B.re[i] + B.im[i] * B.im[i] - I0[i]));
  check('N=2 soliton returns to initial intensity at z0 = pi/2 LD', dI < 2e-3, 'max |dI| ' + fmt(dI));
  check('N=2 soliton compresses mid-period (peak intensity > 1.5x initial 4)', Imax > 1.5 * 4, 'peak ' + Imax.toFixed(2) + ' at z=' + zAtMax.toFixed(3));
}

/* ---------------- NLSE: convergence order and merged-Strang equivalence ---------------- */
{
  const mk = method => new NLO.NLSE({ N: 512, T: 40, linear: { beta: [-1] }, gamma: 1, method: method });
  const run = (method, n) => { const s = mk(method), A = analysis.sech(s.grid, 1, 2); s.propagate(A, { dz: 0.4 / n, steps: n }); return A; };
  const ref = run('strang', 4000);
  const err = (m, n) => analysis.maxDiff(run(m, n), ref);
  const sOrd = Math.log2(err('strang', 40) / err('strang', 80)), lOrd = Math.log2(err('lie', 40) / err('lie', 80));
  check('Strang split-step is 2nd order', sOrd > 1.9 && sOrd < 2.2, 'observed ' + sOrd.toFixed(2));
  check('Lie split-step is 1st order', lOrd > 0.9 && lOrd < 1.2, 'observed ' + lOrd.toFixed(2));

  const s = mk('strang'), A = analysis.sech(s.grid, 1, 2), B = analysis.copy(A);
  s.propagate(A, { dz: 0.01, steps: 50 });
  for (let n = 0; n < 50; n++) s.step(B, 0.01, n * 0.01);
  check('merged Strang == plain 2-FFT-pair Strang', analysis.maxDiff(A, B) < 1e-11, fmt(analysis.maxDiff(A, B)));

  // sampling callback agrees with a separately propagated state
  let samp = null; const C = analysis.sech(s.grid, 1, 2);
  s.propagate(C, { dz: 0.01, steps: 50, every: 25, onSample: (z, re, im, n) => { if (n === 25) samp = { re: Float64Array.from(re), im: Float64Array.from(im) }; } });
  const Dd = analysis.sech(s.grid, 1, 2); s.propagate(Dd, { dz: 0.01, steps: 25 });
  check('mid-run sample equals state propagated to that z', analysis.maxDiff(samp, Dd) < 1e-11, fmt(analysis.maxDiff(samp, Dd)));

  // custom pointwise nonlinearity: two-photon absorption on a flat-ish field decays as 1/(1+beta I z)
  const t2 = new NLO.NLSE({ N: 16, T: 16, nonlinear: nl.pointwise((I, z, i, o) => { o[0] = -0.5 * I; o[1] = 0; }), method: 'lie' });
  const F = { re: new Float64Array(16).fill(1), im: new Float64Array(16) };
  t2.propagate(F, { dz: 1e-4, steps: 10000 });   // z = 1; dI/dz = -I^2 (|A|^2 = I, amplitude rate -I/2 -> intensity rate -I^2... first order)
  const Iend = F.re[0] * F.re[0], exact = 1 / (1 + 1);
  check('pointwise nonlinearity: two-photon absorption vs 1/(1+z)', Math.abs(Iend - exact) < 2e-3, Iend.toFixed(5) + ' vs ' + exact);
}

/* ---------------- Haus model vs the frozen applet engine ---------------- */
{
  const compare = (tauA, rounds) => {
    const realRandom = Math.random, seedLeg = NLO.makeRng(42);
    Math.random = seedLeg.uniform;
    const p = { g0: 0.6, l: 0.1, Esat: 400, Dg: 2, D: -0.02, gamma: 0.005, q0: 0.3, EsatA: 40, tauA: tauA, noise: 1e-3 };
    const Nt = LEG.ML_Nt, re = new Float64Array(Nt), im = new Float64Array(Nt), q = new Float64Array(Nt).fill(p.q0);
    for (let i = 0; i < Nt; i++) { re[i] = 0.01 * LEG.mlRandn(); im[i] = 0.01 * LEG.mlRandn(); }
    const pl = Object.assign({}, p, { D: -p.D });   // frozen applet code used exp(-iDw^2/2): opposite sign
    const mine = new NLO.HausModelocking({ rng: NLO.makeRng(42), params: p, absorberMode: 'legacy' });
    let worst = 0, scale = 0;
    for (let r = 0; r < rounds; r++) {
      LEG.mlStep(re, im, q, pl); mine.step();
      for (let i = 0; i < Nt; i++) { worst = Math.max(worst, Math.abs(re[i] - mine.re[i]), Math.abs(im[i] - mine.im[i])); scale = Math.max(scale, Math.abs(re[i]), Math.abs(im[i])); }
    }
    Math.random = realRandom;
    return worst / scale;
  };
  const a = compare(0, 80), b = compare(25, 80);
  check('Haus model == frozen applet mlStep with D sign flipped (instantaneous SA, 80 RT)', a < 1e-9, 'rel dev ' + fmt(a));
  check('Haus model == frozen applet mlStep with D sign flipped (tauA=25, legacy mode, 80 RT)', b < 1e-9, 'rel dev ' + fmt(b));

  // adaptive stopping reproduces the applet's runSelfStart round count
  const realRandom = Math.random; Math.random = NLO.makeRng(7).uniform;
  const p = { g0: 0.6, l: 0.1, Esat: 400, Dg: 2, D: -0.02, gamma: 0.005, q0: 0.3, EsatA: 40, tauA: 0, noise: 1e-3 };
  const pl = Object.assign({}, p, { D: -p.D });
  const Nt = LEG.ML_Nt, re = new Float64Array(Nt), im = new Float64Array(Nt), q = new Float64Array(Nt).fill(p.q0);
  for (let i = 0; i < Nt; i++) { re[i] = 0.01 * LEG.mlRandn(); im[i] = 0.01 * LEG.mlRandn(); }
  let lastE = null, stable = 0, finalRT = 4000, conv = false;
  for (let r = 0; r <= 4000; r++) {
    LEG.mlStep(re, im, q, pl);
    if (r >= 100 && r % 5 === 0) {
      let E = 0; for (let i = 0; i < Nt; i++) E += re[i] * re[i] + im[i] * im[i];
      if (lastE !== null) { const rel = Math.abs(E - lastE) / Math.max(E, lastE, 1e-9); stable = (rel < 0.001 || (E < 0.01 && lastE < 0.01)) ? stable + 1 : 0; if (stable >= 16) { conv = true; finalRT = r; break; } }
      lastE = E;
    }
  }
  Math.random = realRandom;
  const mine = new NLO.HausModelocking({ rng: NLO.makeRng(7), params: p }), res = mine.run({ snapshotEvery: 1000 });
  check('ConvergenceMonitor reproduces applet stopping round', res.converged === conv && Math.abs(res.rounds - finalRT) <= 1, 'applet ' + finalRT + ', engine ' + res.rounds);
  const m = analysis.pulseMetrics(mine.re, mine.im, 1);
  check('default KLM-like case converges to a single short pulse', res.converged && m.fwhm < 40 && m.peak > 1, 'rounds ' + res.rounds + ', FWHM ' + m.fwhm.toFixed(1) + ' bins, peak ' + m.peak.toFixed(1));
}

/* ---------------- Haus model: the two corrected behaviours ---------------- */
{
  // (1) dispersion sign: gain, loss, absorber, noise OFF -> pure SPM + dispersion per round trip.
  //     D = beta2 L < 0 (anomalous) must carry a fundamental soliton unchanged; D > 0 must spread it.
  const off = { g0: 0, l: 0, Dg: 0, q0: 0, noise: 0, Esat: 1e30, gamma: 0.005, tauA: 0 };
  const run = D => {
    const h = new NLO.HausModelocking({ seed: 1, params: Object.assign({}, off, { D: D }), initAmp: 0 });
    const T0 = 8, P0 = Math.abs(D) / (off.gamma * T0 * T0);
    for (let i = 0; i < 256; i++) { h.re[i] = Math.sqrt(P0) / Math.cosh((i - 128) / T0); h.im[i] = 0; }
    const f0 = analysis.pulseMetrics(h.re, h.im, 1).fwhm;
    for (let r = 0; r < 2000; r++) h.step();
    return analysis.pulseMetrics(h.re, h.im, 1).fwhm / f0;
  };
  const anom = run(-0.02), norm = run(+0.02);
  check('Haus D<0 is anomalous: soliton FWHM preserved over 2000 round trips', Math.abs(anom - 1) < 0.01, 'FWHM ratio ' + anom.toFixed(4));
  check('Haus D>0 is normal: same pulse spreads', norm > 1.3, 'FWHM ratio ' + norm.toFixed(3));

  // (2) SESAM recovery is in time bins, along the fast-time axis (exact exponential solution)
  const tau = 20, q0 = 0.3;
  const h = new NLO.HausModelocking({ seed: 1, absorberMode: 'fastTime', initAmp: 0, params: { tauA: tau, q0: q0, noise: 0, g0: 0, l: 0, Dg: 0, D: 0, gamma: 0 } });
  h.qCarry = 0.5; h.step();                                             // empty field, I = 0 everywhere
  const err1 = Math.abs(h.q[9] - (q0 + (0.5 - q0) * Math.exp(-10 / tau)));
  check('fastTime absorber relaxes to q0 with e-fold = tauA bins (I = 0)', err1 < 1e-12, 'err ' + fmt(err1));
  const h2 = new NLO.HausModelocking({ seed: 1, absorberMode: 'fastTime', initAmp: 0, params: { tauA: tau, q0: q0, noise: 0, g0: 0, l: 0, Dg: 0, D: 0, gamma: 0, EsatA: 40 } });
  h2.re[100] = 20; h2.step();                                           // one bright bin bleaches the absorber
  const dq0 = h2.q[100] - q0, dq1 = h2.q[100 + tau] - q0;
  check('bleached absorber recovers with e-fold tauA bins after the pulse', dq0 < -0.1 && Math.abs(dq1 / dq0 - Math.exp(-1)) < 1e-9, 'ratio ' + (dq1 / dq0).toFixed(6) + ' vs ' + Math.exp(-1).toFixed(6));
}

/* ---------------- GNLSE: self-steepening ---------------- */
{
  const N = 1024, T = 24, gamma = 1, sS = 0.05;
  const nlse = new NLO.NLSE({ N: N, T: T, nonlinear: nl.gnlse({ N: N, T: T, gamma: gamma, tShock: sS, substeps: 1 }) });
  const A = analysis.gaussian(nlse.grid, 1);                               // A = exp(-T^2/2), |A|^2 = exp(-T^2)
  const E0 = analysis.energy(A.re, A.im, nlse.grid.dt);
  const z = 2; nlse.propagate(A, { dz: 0.002, steps: z / 0.002 });
  const I = Float64Array.from(A.re, (r, i) => r * r + A.im[i] * A.im[i]);
  let worst = 0;                                                         // dispersionless: dI/dz = -3 gamma s I dI/dT  -> characteristics T = T0 + 3 gamma s I0(T0) z
  for (let i = 0; i < N; i++) {
    const Tn = nlse.grid.t[i]; let lo = Tn - 3 * gamma * sS * z - 1, hi = Tn + 1;
    for (let it = 0; it < 80; it++) { const m = 0.5 * (lo + hi), f = m + 3 * gamma * sS * z * Math.exp(-m * m) - Tn; if (f > 0) hi = m; else lo = m; }
    worst = Math.max(worst, Math.abs(I[i] - Math.exp(-lo * lo)));
  }
  check('self-steepening, no dispersion: intensity follows the exact characteristics (pre-shock)', worst < 2e-3, 'max |dI| ' + fmt(worst));
  check('self-steepening conserves energy', Math.abs(analysis.energy(A.re, A.im, nlse.grid.dt) / E0 - 1) < 1e-6, fmt(Math.abs(analysis.energy(A.re, A.im, nlse.grid.dt) / E0 - 1)));
  let pk = 0, ip = 0; for (let i = 0; i < N; i++) if (I[i] > pk) { pk = I[i]; ip = i; }
  check('self-steepening moves the peak to +T (trailing-edge steepening)', nlse.grid.t[ip] > 0.2, 'peak at T = ' + nlse.grid.t[ip].toFixed(3));

  // with s = 0 and no Raman the GNLSE operator must reduce to Kerr SPM
  const a = new NLO.NLSE({ N: 256, T: 20, nonlinear: nl.gnlse({ N: 256, T: 20, gamma: 1 }) }), b = new NLO.NLSE({ N: 256, T: 20, gamma: 1 });
  const U = analysis.sech(a.grid, 1, 1.3), V = analysis.copy(U);
  a.propagate(U, { dz: 0.01, steps: 20 }); b.propagate(V, { dz: 0.01, steps: 20 });
  check('GNLSE with s = 0, no Raman == Kerr (RK4 truncation only)', analysis.maxDiff(U, V) < 1e-7, fmt(analysis.maxDiff(U, V)));
}

/* ---------------- GNLSE: Raman soliton self-frequency shift ---------------- */
{
  // Gordon: d(Omega)/dz = -8 TR |beta2| / (15 T0^4)  for a fundamental soliton (beta2 = -1, gamma = 1, T0 = 1)
  const sh = nlseRaman => {
    const nlse = nlseRaman, A = analysis.sech(nlse.grid, 1, 1), c0 = analysis.spectralCentroid(A.re, A.im, nlse.grid);
    nlse.propagate(A, { dz: 0.02, steps: 750 });                           // z = 15
    return (analysis.spectralCentroid(A.re, A.im, nlse.grid) - c0) / 15;
  };
  const N = 4096, T = 40, TR = 0.01, pred = -8 * TR / 15;
  const intr = sh(new NLO.NLSE({ N: N, T: T, linear: { beta: [-1] }, nonlinear: nl.gnlse({ N: N, T: T, gamma: 1, raman: { TR: TR } }) }));
  check('Raman (intrinsic): soliton redshift rate vs Gordon -8 TR |b2| / (15 T0^4)', Math.abs(intr / pred - 1) < 0.05, 'rate ' + intr.toExponential(3) + ' vs ' + pred.toExponential(3) + ' (ratio ' + (intr / pred).toFixed(3) + ')');

  // full causal response: same shift when the pulse is long compared with the response time (TR_eff = fR * int t hR dt)
  const fR = 0.18, t1 = 0.05, t2 = 0.13;
  let m1 = 0; { const dt = 1e-4; for (let t = 0; t < 2; t += dt) m1 += t * (t1 * t1 + t2 * t2) / (t1 * t2 * t2) * Math.exp(-t / t2) * Math.sin(t / t1) * dt; }
  const full = sh(new NLO.NLSE({ N: N, T: T, linear: { beta: [-1] }, nonlinear: nl.gnlse({ N: N, T: T, gamma: 1, raman: { fR: fR, tau1: t1, tau2: t2 } }) }));
  const predFull = -8 * fR * m1 / 15;
  check('Raman (full hR response): redshift rate vs Gordon with TR = fR * int t hR dt', Math.abs(full / predFull - 1) < 0.12, 'rate ' + full.toExponential(3) + ' vs ' + predFull.toExponential(3) + ' (TR_eff ' + (fR * m1).toFixed(4) + ')');
}

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
