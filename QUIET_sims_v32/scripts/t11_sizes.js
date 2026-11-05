/* Task 11: the grating-coupler design (one input on facet 0, one output on a chosen facet) at 5, 10, 15 and 20 mm cell
   diameter, with the Gaussian-beamlet model of Tasks 9-10. One set of beamlet ray traces per layout serves every input
   dither design of the list, so the fringe can be plotted against the number of input states N for each cell size.

   node scripts/t11_sizes.js --band nir|mir --R 0.97 --dBcm 0.3 --cells d5:16:25e-6:11,d10:... --dithers all|design
        --out results/t11/<name>.json [--seeds 30] [--c_cal 1.406]
   cells: name:n_inc:inc_radius:out_facet[:diameter_mm] (diameter from the name dNN if omitted)

   Per layout and dither: static (0 Hz) fringe of the modes none / input / spectral / both (eps 1e-6; input = angle and
   position dither of the input beam; spectral = triangle wavelength dither), at +-20, +-35, +-81 pm; T, Gamma<L>, M
   (engine definition, calibrated, as Task 10); the exit-path distribution (geometric path, power) of the design dither for
   the noise traces of Fig. 3a. Drift averaging is applied in the analysis: static x 1/sqrt(1 + 2 (0.05 T)^2) (the engine's
   envelope for 0.1 rad/s drift). */
const fs = require('fs');
const path = require('path');
const { loadEngine } = require('./engine_path.js');
const { config, args } = require('./cells.js');
const { DitherLab: D, GasEngine: G, PlanarEngine: Pl } = loadEngine();
const a = args(process.argv.slice(2));
const band = a.band || 'nir', R = +(a.R || 0.97), dBcm = +(a.dBcm || 0.3), nSeed = +(a.seeds || 30), sigma = 10e-6;
const B = band === 'nir' ? { lam: 1.5317e-6, n: 1.5391, gamma: 0.4532 } : { lam: 3.3151e-6, n: 1.4253, gamma: 0.5526 };
const DLAMS = [10 ** 1.3 * 1e-12, 35e-12, 81e-12];
const k0 = 2 * Math.PI / B.lam, kap = k0 * B.n, XMAX = 250e-6, PORTW = 150e-6, NF = 24;
const DITHERS_ALL = [
  ['w40 A2', 40e-6, 2e-3, 0], ['w40 A15', 40e-6, 15e-3, 0], ['w40 A30', 40e-6, 30e-3, 0], ['w40 A60', 40e-6, 60e-3, 0],
  ['w75 A15', 75e-6, 15e-3, 0], ['w75 A30', 75e-6, 30e-3, 0], ['w75 A60', 75e-6, 60e-3, 0],
  ['w150 A15', 150e-6, 15e-3, 0], ['w150 A30', 150e-6, 30e-3, 0], ['w150 A60', 150e-6, 60e-3, 0],
  ['w40 X100', 40e-6, 0, 100e-6], ['w40 X150', 40e-6, 0, 150e-6],
  ['w75 A15 X100', 75e-6, 15e-3, 100e-6], ['w40 A30 X150', 40e-6, 30e-3, 150e-6], ['w75 A30 X100', 75e-6, 30e-3, 100e-6],
  ['w75 A45 X100', 75e-6, 45e-3, 100e-6], ['w75 A60 X100', 75e-6, 60e-3, 100e-6],
];
const DESIGN = 'w75 A60 X100';
const DITHERS = a.dithers === 'design' ? DITHERS_ALL.filter(d => d[0] === DESIGN || d[0] === 'w40 A2') : DITHERS_ALL;
const CELLS = (a.cells || 'd5:16:25e-6:11,d10:16:25e-6:11,d15:16:25e-6:11,d20:16:25e-6:11').split(',').map(s => {
  const [name, n, r, f, dmm] = s.split(':');
  return { name, n_inc: +n, inc_r: +r, out_facet: +f, r: (dmm ? +dmm : +name.replace(/^d/, '').split('_')[0]) * 0.5e-3 };
});

/* Gabor lattice and dither-averaged coefficients (as Task 9b) */
const rr_ = Math.sqrt(Math.PI), dx = rr_ * sigma, dphi = rr_ / (2 * kap * sigma), PHIMAX = 60e-3 + 3 / (kap * 40e-6) + 3 / (2 * kap * sigma);
const xs = [], phis = [];
for (let x = -XMAX; x <= XMAX + 1e-12; x += dx) xs.push(x);
for (let p = -Math.ceil(PHIMAX / dphi) * dphi; p <= PHIMAX + 1e-12; p += dphi) phis.push(p);
const sineS = n => Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * (i + 0.5) / n));
const nAng = (A, w) => A > 0 ? Math.max(48, Math.ceil(2 * Math.PI * A * kap * w)) : 1, nPos = (X, w) => X > 0 ? Math.max(48, Math.ceil(4 * Math.PI * X / w)) : 1;
function coef(xj, pj, w, X, Th) {
  const A = 1 / (4 * sigma * sigma) + 1 / (w * w);
  const Br = xj / (2 * sigma * sigma) + 2 * X / (w * w), Bi = kap * (Th - pj);
  const Cr = xj * xj / (4 * sigma * sigma) + X * X / (w * w), Ci = kap * Th * X;
  const er = (Br * Br - Bi * Bi) / (4 * A) - Cr, ei = (2 * Br * Bi) / (4 * A) - Ci, m = Math.sqrt(Math.PI / A) * Math.exp(er);
  return [m * Math.cos(ei), m * Math.sin(ei)];
}
const COEF = {};
for (const [name, w, A, Xa] of DITHERS) {
  const sA = A > 0 ? sineS(nAng(A, w)) : [0], sX = Xa > 0 ? sineS(nPos(Xa, w)) : [0], c0 = [], cb = [];
  for (const xj of xs) for (const pj of phis) {
    c0.push(coef(xj, pj, w, 0, 0));
    let cr = 0, ci = 0;
    for (const u of sA) for (const v of sX) { const c = coef(xj, pj, w, Xa * v, A * u); cr += c[0]; ci += c[1]; }
    cb.push([cr / (sA.length * sX.length), ci / (sA.length * sX.length)]);
  }
  COEF[name] = { c0, cb };
}

function opts(cell, seed) {
  const base = config('circle', { n_eff: B.n, R, dBcm, wavelength: B.lam, gamma: B.gamma, n_inc: cell.n_inc, inc_dn: 0.01, inc_radius: cell.inc_r });
  const o = Object.assign({}, D.DEFAULTS, base, { c_radius: cell.r, inc_seed: seed || 1 });
  return seed === null ? D.regular(o) : o;
}

/* trace every beamlet once: exit (amp, opl, geometric L, y, u) or null */
function trace(cell, seed) {
  const o = opts(cell, seed), S = D.setup(o), c = S.mc.cell, P = c.perimeter, th0 = S.theta, cosT = Math.cos(th0);
  G.setAlpha(o.alpha_m); G.setRTable(null);
  const sOut = c.sIn + cell.out_facet * P / NF, nB = Math.min(4000, Math.ceil(Math.log(1e-3) / Math.log(Math.min(o.R_mirror, 0.99999))));
  const out = [];
  for (const xj of xs) for (const pj of phis) {
    const r = G.marchRay(S.mc, Pl.launch(c, c.sIn + xj / cosT, th0 + pj), nB, c.sIn, sOut, PORTW, 'out');
    if (r.exit !== 1) { out.push(null); continue; }
    const h = r.nWalls, sc = r.WC[h - 1];
    let ds = r.WS[h - 1] - sOut; ds -= P * Math.round(ds / P);
    out.push({ amp: Math.pow(o.R_mirror, (h - 1) / 2) * Math.exp(-0.5 * o.alpha_m * r.WG[h - 1]), opl: S.mc.n0 * r.WL[h - 1], L: r.WG[h - 1],
      y: ds * Math.sqrt(Math.max(1 - sc * sc, 0)), u: Math.asin(Math.max(-1, Math.min(1, sc))) });
  }
  return { g: out, lamM: B.lam / B.n };
}

function evaluate(tr, dname, cal) {
  const K = COEF[dname], g = tr.g;
  let P = 0, PL = 0, Pin = 0, my = 0, mu = 0, yy = 0, uu = 0, yu = 0;
  const F = DLAMS.map(() => new Float64Array(8));
  for (let j = 0; j < g.length; j++) {
    const c0 = K.c0[j], cb = K.cb[j], q0 = c0[0] * c0[0] + c0[1] * c0[1];
    Pin += q0;
    const b = g[j];
    if (!b) continue;
    const p = q0 * b.amp * b.amp, ph = k0 * b.opl, gr = b.amp * Math.cos(ph), gi = b.amp * Math.sin(ph);
    P += p; PL += p * b.L; my += p * b.y; mu += p * b.u; yy += p * b.y * b.y; uu += p * b.u * b.u; yu += p * b.y * b.u;
    DLAMS.forEach((dl, q) => {
      const x = k0 * b.opl * dl / B.lam, s = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x, f = F[q];
      f[0] += c0[0] * gr - c0[1] * gi; f[1] += c0[0] * gi + c0[1] * gr; f[2] += cb[0] * gr - cb[1] * gi; f[3] += cb[0] * gi + cb[1] * gr;
      f[4] += s * (c0[0] * gr - c0[1] * gi); f[5] += s * (c0[0] * gi + c0[1] * gr); f[6] += s * (cb[0] * gr - cb[1] * gi); f[7] += s * (cb[0] * gi + cb[1] * gr);
    });
  }
  if (!(P > 0)) return null;
  my /= P; mu /= P;
  const M2 = 4 * Math.PI / tr.lamM * Math.sqrt(Math.max((yy / P - my * my) * (uu / P - mu * mu) - (yu / P - my * mu) ** 2, 0)), M = cal * M2;
  const fr = {};
  DLAMS.forEach((dl, q) => {
    const f = F[q], k = `${(dl * 1e12).toFixed(0)}pm`, z = (r_, i_) => 2e-3 * Math.hypot(r_, i_) / Math.sqrt(P) / Math.sqrt(M);
    fr[k] = { none: z(f[0], f[1]), input: z(f[2], f[3]), spectral: z(f[4], f[5]), both: z(f[6], f[7]) };
  });
  return { T: P / Pin, GL_cm: B.gamma * PL / P * 100, L_cm: PL / P * 100, M, fringe: fr };
}

let cal = a.c_cal ? +a.c_cal : null;
if (cal === null) {      /* calibrate M to the engine's grains (Task 2 geometry: 4 mm, 16 x 50 um, facet 11, 40 um beam) */
  const t2 = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 't2', `circle_${band}_R${R}_${dBcm}dB.json`)));
  const ref = { name: 'ref', n_inc: 16, inc_r: 50e-6, out_facet: 11, r: 2e-3 }, rat = [];
  for (let s = 1; s <= 30; s++) { const e = evaluate(trace(ref, s), 'w40 A2', 1); if (e && e.M > 0) rat.push(t2.chaotic[s - 1].grains / e.M); }
  rat.sort((x, y) => x - y); cal = rat[Math.floor(rat.length / 2)];
}
console.log('c_cal', cal.toFixed(3));
const res = { band, R, dBcm, c_cal: cal, dlams: DLAMS, design: DESIGN, dithers: DITHERS.map(d => ({ name: d[0], w: d[1], A: d[2], Xa: d[3] })), cells: {} };
const t0 = Date.now();
for (const cell of CELLS) {
  const rows = [];
  for (const seed of [null].concat(Array.from({ length: nSeed }, (_, i) => i + 1))) {
    try {
      const tr = trace(cell, seed), row = { seed, d: {} };
      for (const [dn] of DITHERS) row.d[dn] = evaluate(tr, dn, cal);
      /* exit paths of the design dither (undithered coefficients): geometric path and power, the 120 strongest */
      const K = COEF[DESIGN], ex = [];
      tr.g.forEach((b, j) => { if (b) { const c0 = K.c0[j]; ex.push([b.L, (c0[0] * c0[0] + c0[1] * c0[1]) * b.amp * b.amp]); } });
      ex.sort((x, y) => y[1] - x[1]);
      row.paths = ex.slice(0, 120).map(([L, w]) => [+L.toPrecision(5), +w.toPrecision(4)]);
      rows.push(row);
    } catch (e) { rows.push({ seed, error: String(e.message) }); }
  }
  res.cells[cell.name] = { cell, regular: rows[0], chaotic: rows.slice(1) };
  console.log(cell.name, ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
fs.mkdirSync(path.dirname(a.out), { recursive: true });
fs.writeFileSync(a.out, JSON.stringify(res));
console.log('wrote', a.out);
