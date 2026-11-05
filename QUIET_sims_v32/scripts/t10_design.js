/* Task 10: design levers for sensitivity, with the wave-consistent Gaussian-beamlet model of Task 9 (input angle and
   position dither) generalised to several input ports, several (optionally randomised) output ports, larger cells and
   other inclusion sets.

   node scripts/t10_design.js --band nir|mir --R 0.97 --dBcm 0.3 --variants a,b,c --out results/t10/<name>.json [--seeds 30]

   Per variant and layout:
   - each input port i (a facet centre) launches the dithered Gaussian beam of the variant, expanded on the Gabor lattice of
     Task 9 (sigma 10 um, oversampling 4); each beamlet is one engine ray traced with all ports closed, and its exit is the
     first wall hit inside any output window (width 150 um); input windows reflect (as in Tasks 2-9);
   - per input i and output k: incoherent power P_ik = sum |c0_j|^2 |g_j|^2 and the static fringe phasor
     F_ik = sum cbar_j gbar_j (cbar: dither-averaged coefficient; gbar: g sinc(k0 OPL dlam / lam), triangle wavelength dither);
   - the inputs' relative phase is dithered, so at detector k only the light of input i interferes with the stray copy of
     input i (cross terms average out); each (input, output) pair carries its normalised phasor v_ik = F_ik / sqrt(P_ik), and
     the summed photocurrents give dP/P = 2 sqrt(eps) |sum_ik P_ik v_ik / sqrt(M)| / sum_ik P_ik (Task 10 first run summed
     all inputs into one phasor, i.e. one stray copy of the combined input: no gain from extra inputs; fixed, documented);
   - M = c_cal * 4 pi / lam_M * sqrt(det cov(y, u)) of the power-weighted exits (the engine's M2 definition; y = offset in
     the port times cos chi, u = chi), the within-port covariance pooled over all output ports (same width, same cell), with c_cal fitted once to the engine's grains on the Task 2 geometry
     (results/t2, set B R 97 %);
   - transmission T = sum_k P_k / sum_i sum_j |c0_ij|^2 (laser power split equally over the inputs), Gamma<L> = Gamma times the
     power-weighted mean geometric path over all detectors.
*/
const fs = require('fs');
const path = require('path');
const { loadEngine } = require('./engine_path.js');
const { config, args } = require('./cells.js');
const { DitherLab: D, GasEngine: G, PlanarEngine: Pl } = loadEngine();
const a = args(process.argv.slice(2));
const band = a.band || 'nir', R = +(a.R || 0.97), dBcm = +(a.dBcm || 0.3), nSeed = +(a.seeds || 30), sigma = 10e-6;
const B = band === 'nir' ? { lam: 1.5317e-6, n: 1.5391, gamma: 0.4532 } : { lam: 3.3151e-6, n: 1.4253, gamma: 0.5526 };
const DLAMS = [10 ** 1.3 * 1e-12, 35e-12, 81e-12];
const k0 = 2 * Math.PI / B.lam, kap = k0 * B.n, XMAX = 250e-6, PORTW = 150e-6, NF = 24, C_CAL = a.c_cal ? +a.c_cal : null;
const IN_FACETS = [0, 6, 15, 19], OUT_FIXED = 11;

/* variants: cell radius r (m), inclusions (count, radius, dn), input ports, output ports (fixed at facet 11 when one and
   not randomised; otherwise random facets, and random offsets inside the facet), input dither (waist, angle, position) */
const V0 = { r: 2e-3, n_inc: 16, inc_r: 50e-6, inc_dn: 0.01, n_in: 1, n_out: 1, rand: 0, w: 75e-6, A: 60e-3, Xa: 100e-6 };
const VARIANTS = {
  base: {}, old_dither: { w: 40e-6, A: 2e-3, Xa: 0 },
  in2: { n_in: 2 }, in3: { n_in: 3 }, in4: { n_in: 4 },
  ...Object.fromEntries(Array.from({ length: 23 }, (_, i) => [`of${i + 1}`, { out_facet: i + 1 }])),
  out1r: { rand: 1 }, out2: { n_out: 2, rand: 1 }, out4: { n_out: 4, rand: 1 }, out8: { n_out: 8, rand: 1 },
  inc8: { n_inc: 8 }, inc32: { n_inc: 32 }, inc64: { n_inc: 64 }, r25: { inc_r: 25e-6 }, r25n64: { inc_r: 25e-6, n_inc: 64 }, r100: { inc_r: 100e-6 },
  dn005: { inc_dn: 0.005 }, dn02: { inc_dn: 0.02 }, dn04: { inc_dn: 0.04 },
  d7: { r: 3.5e-3 }, d7s: { r: 3.5e-3, n_inc: 49 }, d10: { r: 5e-3 }, d10s: { r: 5e-3, n_inc: 100 },
};
for (const [k, v] of Object.entries(JSON.parse(a.extra || '{}'))) VARIANTS[k] = v;
const names = (a.variants || Object.keys(VARIANTS).join(',')).split(',');

/* lattice and dither-averaged coefficients (Task 9) */
const r_ = Math.sqrt(Math.PI), dx = r_ * sigma, dphi = r_ / (2 * kap * sigma), PHIMAX = 60e-3 + 3 / (kap * 40e-6) + 3 / (2 * kap * sigma);
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
function coefOf(w, A, Xa) {
  const key = `${w}|${A}|${Xa}`;
  if (COEF[key]) return COEF[key];
  const sA = A > 0 ? sineS(nAng(A, w)) : [0], sX = Xa > 0 ? sineS(nPos(Xa, w)) : [0], c0 = [], cb = [];
  for (const xj of xs) for (const pj of phis) {
    c0.push(coef(xj, pj, w, 0, 0));
    let cr = 0, ci = 0;
    for (const u of sA) for (const v of sX) { const c = coef(xj, pj, w, Xa * v, A * u); cr += c[0]; ci += c[1]; }
    cb.push([cr / (sA.length * sX.length), ci / (sA.length * sX.length)]);
  }
  return (COEF[key] = { c0, cb });
}

function opts(v, seed) {
  const base = config('circle', { n_eff: B.n, R, dBcm, wavelength: B.lam, gamma: B.gamma, n_inc: v.n_inc, inc_dn: v.inc_dn, inc_radius: v.inc_r });
  const o = Object.assign({}, D.DEFAULTS, base, { c_radius: v.r, inc_seed: seed || 1 });
  return seed === null ? D.regular(o) : o;
}

function run(v, seed) {
  const o = opts(v, seed), S = D.setup(o), c = S.mc.cell, P = c.perimeter, th0 = S.theta, cosT = Math.cos(th0), lamM = B.lam / B.n;
  G.setAlpha(o.alpha_m); G.setRTable(null);
  const flen = P / NF, rnd = Pl.mulberry32(1000 + (seed || 0) * 7 + v.n_out * 131);
  const inF = v.in_facets || IN_FACETS.slice(0, v.n_in), ins = inF.map(f => c.sIn + f * flen);
  let outs;
  if (v.out_facet !== undefined) outs = [c.sIn + v.out_facet * flen];
  else if (!v.rand && v.n_out === 1) outs = [c.sIn + OUT_FIXED * flen];
  else {
    const pool = []; for (let f = 1; f < NF; f++) if (!inF.includes(f)) pool.push(f);
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    outs = pool.slice(0, v.n_out).map(f => c.sIn + f * flen + (rnd() - 0.5) * (flen - PORTW) * 0.9);
  }
  const nB = Math.min(3000, Math.ceil(Math.log(1e-3) / Math.log(Math.min(o.R_mirror, 0.99999))));
  const K = coefOf(v.w, v.A, v.Xa), nO = outs.length;
  const nI = ins.length, Pk = new Float64Array(nO * nI), Fk = DLAMS.map(() => Array.from({ length: nO * nI }, () => [0, 0, 0, 0, 0, 0, 0, 0]));   /* per (input, output): none r,i; input r,i; spec r,i; both r,i */
  const mom = Array.from({ length: nO }, () => ({ w: 0, y: 0, u: 0, yy: 0, uu: 0, yu: 0 }));
  let PL = 0, Pin = 0;
  ins.forEach((s0, ii) => {
    let j = 0;
    for (const xj of xs) for (const pj of phis) {
      const c0 = K.c0[j], cb = K.cb[j]; j++;
      Pin += c0[0] * c0[0] + c0[1] * c0[1];
      const rr = G.marchRay(S.mc, Pl.launch(c, s0 + xj / cosT, th0 + pj), nB, 0, 0, 0, 'none');
      let hit = -1, port = -1, ds = 0;
      for (let h = 0; h < rr.nWalls && hit < 0; h++) {
        for (let k = 0; k < nO; k++) { let d = rr.WS[h] - outs[k]; d -= P * Math.round(d / P); if (Math.abs(d) <= PORTW / 2) { hit = h; port = k; ds = d; break; } }
      }
      if (hit < 0) continue;
      const amp = Math.pow(o.R_mirror, hit / 2) * Math.exp(-0.5 * o.alpha_m * rr.WG[hit]), opl = S.mc.n0 * rr.WL[hit], ph = k0 * opl;
      const gr = amp * Math.cos(ph), gi = amp * Math.sin(ph), p = (c0[0] * c0[0] + c0[1] * c0[1]) * amp * amp, ik = ii * nO + port;
      Pk[ik] += p; PL += p * rr.WG[hit];
      const sc = rr.WC[hit], y = ds * Math.sqrt(Math.max(1 - sc * sc, 0)), u = Math.asin(Math.max(-1, Math.min(1, sc))), M = mom[port];
      M.w += p; M.y += p * y; M.u += p * u; M.yy += p * y * y; M.uu += p * u * u; M.yu += p * y * u;
      DLAMS.forEach((dl, q) => {
        const x = k0 * opl * dl / B.lam, sc2 = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x, F = Fk[q][ik];
        F[0] += c0[0] * gr - c0[1] * gi; F[1] += c0[0] * gi + c0[1] * gr;
        F[2] += cb[0] * gr - cb[1] * gi; F[3] += cb[0] * gi + cb[1] * gr;
        F[4] += sc2 * (c0[0] * gr - c0[1] * gi); F[5] += sc2 * (c0[0] * gi + c0[1] * gr);
        F[6] += sc2 * (cb[0] * gr - cb[1] * gi); F[7] += sc2 * (cb[0] * gi + cb[1] * gr);
      });
    }
  });
  /* within-port covariance pooled over the ports (same width, same cell): one M per layout, robust for many small ports */
  let W = 0, Cyy = 0, Cuu = 0, Cyu = 0;
  for (const M of mom) { if (!(M.w > 0)) continue; const my = M.y / M.w, mu = M.u / M.w;
    Cyy += M.yy - M.w * my * my; Cuu += M.uu - M.w * mu * mu; Cyu += M.yu - M.w * my * mu; W += M.w; }
  const Mpool = W > 0 ? 4 * Math.PI / lamM * Math.sqrt(Math.max((Cyy / W) * (Cuu / W) - (Cyu / W) ** 2, 0)) : NaN;
  const M2 = Array.from({ length: nO * nI }, () => Mpool);
  const Ptot = Pk.reduce((s, x) => s + x, 0);
  return { Pk: Array.from(Pk), M2, T: Ptot / Pin, GL_cm: Ptot > 0 ? B.gamma * PL / Ptot * 100 : NaN, L_cm: Ptot > 0 ? PL / Ptot * 100 : NaN, Fk, outs: outs.length };
}

/* combine the detectors with a calibration constant for M */
function fringe(rw, cal) {
  const out = {};
  DLAMS.forEach((dl, q) => {
    const key = `${(dl * 1e12).toFixed(0)}pm`;
    out[key] = {};
    ['none', 'input', 'spectral', 'both'].forEach((m, mi) => {
      let zr = 0, zi = 0, pt = 0;
      rw.Pk.forEach((p, k) => {
        if (!(p > 0) || !(rw.M2[k] > 0)) return;
        const F = rw.Fk[q][k], s = 1 / Math.sqrt(p), M = cal * rw.M2[k];
        zr += p * F[2 * mi] * s / Math.sqrt(M); zi += p * F[2 * mi + 1] * s / Math.sqrt(M); pt += p;
      });
      out[key][m] = pt > 0 ? 2e-3 * Math.hypot(zr, zi) / pt : NaN;     /* eps = 1e-6 */
    });
  });
  return out;
}

let cal = C_CAL;
if (cal === null) {    /* calibrate M to the engine's grains on the Task 2 geometry (base variant, 30 layouts) */
  const t2 = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 't2', `circle_${band}_R${R}_${dBcm}dB.json`)));
  const ratios = [];
  for (let s = 1; s <= Math.min(nSeed, 30); s++) { const rw = run(Object.assign({}, V0, { w: 40e-6, A: 2e-3, Xa: 0 }), s); if (rw.M2[0] > 0) ratios.push(t2.chaotic[s - 1].grains / rw.M2[0]); }
  ratios.sort((x, y) => x - y); cal = ratios[Math.floor(ratios.length / 2)];
  console.log('M calibration c_cal =', cal.toFixed(3));
}
const res = { band, R, dBcm, c_cal: cal, dlams: DLAMS, variants: {} };
const t0 = Date.now();
for (const nm of names) {
  const v = Object.assign({}, V0, VARIANTS[nm]);
  const rows = [];
  for (const seed of [null].concat(Array.from({ length: nSeed }, (_, i) => i + 1))) {
    try { const rw = run(v, seed); rows.push({ seed, T: rw.T, GL_cm: rw.GL_cm, L_cm: rw.L_cm, M: rw.M2.map(m => cal * m), Pk: rw.Pk, fringe: fringe(rw, cal) }); }
    catch (e) { rows.push({ seed, error: String(e.message) }); }
  }
  res.variants[nm] = { v, regular: rows[0], chaotic: rows.slice(1) };
  console.log(nm, ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
fs.mkdirSync(path.dirname(a.out), { recursive: true });
fs.writeFileSync(a.out, JSON.stringify(res));
console.log('wrote', a.out);
