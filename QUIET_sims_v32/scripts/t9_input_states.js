/* Task 9: wave-consistent input dither (angle, position, both) with Gaussian beamlets.

   node scripts/t9_input_states.js --band nir|mir --R 0.97 --dBcm 0.3 --out results/t9/<name>.json [--sigma 10e-6] [--seeds 30]

   The engine's angle-dither fringe follows one zero-width ray per dither angle; Task 5b showed that this resolves tilts
   far below the beam's diffraction angle. Here the input field E_in(x; X, Theta) = exp(-(x-X)^2/w^2) exp(i k n Theta (x-X))
   (x transverse at the input facet, w the 1/e^2 intensity radius, X the position offset, Theta the tilt) is expanded on a
   Gabor lattice of Gaussian beamlets psi_j = exp(-(x-x_j)^2/(4 sigma^2)) exp(i k n phi_j x) (oversampling 4). Each beamlet is
   traced as one engine ray from the facet point x_j (arc offset x_j / cos theta0) at angle theta0 + phi_j; its exit phasor
   is g_j = R^((h-1)/2) exp(-alpha L/2) exp(i k0 OPL_j) if it leaves through the output port, else 0. The fringe phasor of
   an input state is F = sum_j c_j g_j with c_j = <psi_j, E_in> (closed-form Gaussian integral). Under the dither the static
   (0 Hz) fringe is F_static = sum_j cbar_j gbar_j: cbar_j = time average of c_j over the input dither (sine angle and sine
   position at incommensurate rates: average over the product of the two waveforms' samples), gbar_j = g_j sinc(x_j) for
   the triangle wavelength dither (x_j = k0 OPL_j dlam / lam, uniform sweep). The fringe factor of a dither mode is
   V = |F_static| / sqrt(sum_j |c_j(0)|^2 |g_j|^2) (V = 1 without dither, as the engine's V); the residual is
   2 sqrt(eps) V / sqrt(M) with the engine's speckle grains M of the same layout (results/t2). Tilts below the beamlet's
   angular width only re-weight the c_j, so sub-diffraction dithers average correctly; beamlets of one diffraction cell
   are traced with independent rays, which treats the cell as fully mixing at that scale (checked with sigma 7 / 10 / 14 um).
   Launch positions are kept on the input facet (|x| <= 250 um, facet 522 um).
*/
const fs = require('fs');
const path = require('path');
const { loadEngine } = require('./engine_path.js');
const { config, args } = require('./cells.js');
const { DitherLab: D, GasEngine: G, PlanarEngine: Pl } = loadEngine();
const a = args(process.argv.slice(2));
const band = a.band || 'nir', R = +(a.R || 0.97), dBcm = +(a.dBcm || 0.3), nSeed = +(a.seeds || 30), sigma = +(a.sigma || 10e-6);
const B = band === 'nir' ? { lam: 1.5317e-6, n: 1.5391, gamma: 0.4532 } : { lam: 3.3151e-6, n: 1.4253, gamma: 0.5526 };
const DLAMS = [10 ** 1.3 * 1e-12, 35e-12, 81e-12, 160e-12];   /* +-20, +-35, +-81, +-160 pm (1.53 um: +-2.6, +-4.5, +-10, +-20 GHz; 3.3 um: +-0.55, +-0.96, +-2.2, +-4.4 GHz) */
const k0 = 2 * Math.PI / B.lam, kap = k0 * B.n, XMAX = 250e-6;
/* dither configurations: waist w (1/e^2 intensity radius), angle amplitude A (rad, sine), position amplitude Xa (m, sine) */
const CONF = [
  ['w40 A2', 40e-6, 2e-3, 0], ['w40 A10', 40e-6, 10e-3, 0], ['w40 A15', 40e-6, 15e-3, 0],
  ['w75 A10', 75e-6, 10e-3, 0], ['w75 A15', 75e-6, 15e-3, 0],
  ['w150 A2', 150e-6, 2e-3, 0], ['w150 A10', 150e-6, 10e-3, 0], ['w150 A15', 150e-6, 15e-3, 0],
  ['w40 X50', 40e-6, 0, 50e-6], ['w40 X100', 40e-6, 0, 100e-6], ['w40 X150', 40e-6, 0, 150e-6],
  ['w40 A15 X100', 40e-6, 15e-3, 100e-6], ['w40 A15 X150', 40e-6, 15e-3, 150e-6], ['w75 A15 X100', 75e-6, 15e-3, 100e-6],
  ['w40 A2 X100', 40e-6, 2e-3, 100e-6],
  /* Task 9b: wide angle sweeps and combinations */
  ['w40 A30', 40e-6, 30e-3, 0], ['w40 A45', 40e-6, 45e-3, 0], ['w40 A60', 40e-6, 60e-3, 0],
  ['w75 A30', 75e-6, 30e-3, 0], ['w75 A45', 75e-6, 45e-3, 0], ['w75 A60', 75e-6, 60e-3, 0],
  ['w150 A30', 150e-6, 30e-3, 0], ['w150 A45', 150e-6, 45e-3, 0], ['w150 A60', 150e-6, 60e-3, 0],
  ['w40 A30 X150', 40e-6, 30e-3, 150e-6], ['w40 A45 X150', 40e-6, 45e-3, 150e-6], ['w40 A60 X150', 40e-6, 60e-3, 150e-6],
  ['w75 A30 X100', 75e-6, 30e-3, 100e-6], ['w75 A45 X100', 75e-6, 45e-3, 100e-6], ['w75 A60 X100', 75e-6, 60e-3, 100e-6],
];
/* Gabor lattice, oversampling 4, balanced: dx = r sigma, dphi = r / (2 kappa sigma), r^2 = 4 pi / 4 */
const r = Math.sqrt(Math.PI), dx = r * sigma, dphi = r / (2 * kap * sigma), PHIMAX = Math.max(...CONF.map(c => c[2])) + 3 / (kap * 40e-6) + 3 / (2 * kap * sigma);
const xs = [], phis = [];
for (let x = -XMAX; x <= XMAX + 1e-12; x += dx) xs.push(x);
for (let p = -Math.ceil(PHIMAX / dphi) * dphi; p <= PHIMAX + 1e-12; p += dphi) phis.push(p);
/* samples per sine period: the tilt step 2 pi A / NA must stay below half the beam divergence 2 / (kappa w), the shift step
   2 pi Xa / NX below w / 2 (Task 9 used 48 / 48, too coarse for the wide sweeps of large beams; see REPORT Task 9b) */
const nAng = (A, w) => A > 0 ? Math.max(48, Math.ceil(2 * Math.PI * A * kap * w)) : 1, nPos = (X, w) => X > 0 ? Math.max(48, Math.ceil(4 * Math.PI * X / w)) : 1;
const sineS = n => Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * (i + 0.5) / n));

/* c = int conj(psi_j) E_in dx, closed form: sqrt(pi/A) exp(B^2/(4A) - C), complex B, C */
function coef(xj, pj, w, X, Th) {
  const A = 1 / (4 * sigma * sigma) + 1 / (w * w);
  const Br = xj / (2 * sigma * sigma) + 2 * X / (w * w), Bi = kap * (Th - pj);
  const Cr = xj * xj / (4 * sigma * sigma) + X * X / (w * w), Ci = kap * Th * X;
  const er = (Br * Br - Bi * Bi) / (4 * A) - Cr, ei = (2 * Br * Bi) / (4 * A) - Ci, m = Math.sqrt(Math.PI / A) * Math.exp(er);
  return [m * Math.cos(ei), m * Math.sin(ei)];
}

function beamletPhasors(o) {
  const S = D.setup(o), c = S.mc.cell, pw = c.meta.portW, th0 = S.theta, cosT = Math.max(Math.cos(th0), 1e-6);
  G.setAlpha(o.alpha_m); G.setRTable(null);
  const nB = Math.min(3000, Math.ceil(Math.log(1e-4) / Math.log(Math.min(o.R_mirror, 0.99999))));
  const out = [];
  for (const xj of xs) for (const pj of phis) {
    const rr = G.marchRay(S.mc, Pl.launch(c, c.sIn + xj / cosT, th0 + pj), nB, c.sIn, c.sOut, pw, 'out');
    if (rr.exit !== 1) { out.push(null); continue; }
    const h = rr.nWalls, L = rr.WL[h - 1], amp = Math.pow(o.R_mirror, (h - 1) / 2) * Math.exp(-0.5 * o.alpha_m * rr.WG[h - 1]), ph = k0 * S.mc.n0 * L;
    out.push({ amp, ph, opl: S.mc.n0 * L });
  }
  return out;
}

/* undithered and dither-averaged beamlet coefficients of each configuration (independent of the layout) */
const COEF = {};
for (const cf of CONF) {
  const [name, w, Aang, Xa] = cf, sA = Aang > 0 ? sineS(nAng(Aang, w)) : [0], sX = Xa > 0 ? sineS(nPos(Xa, w)) : [0];
  const c0 = [], cb = [];
  for (const xj of xs) for (const pj of phis) {
    c0.push(coef(xj, pj, w, 0, 0));
    let cr = 0, ci = 0;
    for (const u of sA) for (const v of sX) { const c = coef(xj, pj, w, Xa * v, Aang * u); cr += c[0]; ci += c[1]; }
    cb.push([cr / (sA.length * sX.length), ci / (sA.length * sX.length)]);
  }
  COEF[name] = { c0, cb, nA: sA.length, nX: sX.length };
}

function factors(g, conf, dlam) {
  const K = COEF[conf[0]];
  let F0r = 0, F0i = 0, n0 = 0, Fa = [0, 0], Fl = [0, 0], Fb = [0, 0];
  for (let j = 0; j < g.length; j++) {
    const gj = g[j];
    if (!gj) continue;
    const c0 = K.c0[j], cr = K.cb[j][0], ci = K.cb[j][1];
    const gr = gj.amp * Math.cos(gj.ph), gi = gj.amp * Math.sin(gj.ph), x = k0 * gj.opl * dlam / B.lam, sc = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x;
    n0 += (c0[0] * c0[0] + c0[1] * c0[1]) * gj.amp * gj.amp;
    F0r += c0[0] * gr - c0[1] * gi; F0i += c0[0] * gi + c0[1] * gr;
    Fa[0] += cr * gr - ci * gi; Fa[1] += cr * gi + ci * gr;
    Fl[0] += sc * (c0[0] * gr - c0[1] * gi); Fl[1] += sc * (c0[0] * gi + c0[1] * gr);
    Fb[0] += sc * (cr * gr - ci * gi); Fb[1] += sc * (cr * gi + ci * gr);
  }
  const nn = Math.sqrt(n0), V = z => Math.hypot(z[0], z[1]) / nn;
  return { none: Math.hypot(F0r, F0i) / nn, input: V(Fa), spectral: V(Fl), both: V(Fb) };
}

const t2 = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 't2', `circle_${band}_R${R}_${dBcm}dB.json`)));
const res = { band, R, dBcm, sigma, lattice: { nx: xs.length, nphi: phis.length, dx, dphi }, conf: CONF.map(c => ({ name: c[0], w: c[1], A: c[2], Xa: c[3], nA: COEF[c[0]].nA, nX: COEF[c[0]].nX })),
  dlams: DLAMS, layouts: [] };
const t0 = Date.now();
const runs = [['regular', null]].concat(Array.from({ length: nSeed }, (_, i) => [`seed${i + 1}`, i + 1]));
for (const [label, seed] of runs) {
  const base = config('circle', { n_eff: B.n, R, dBcm, wavelength: B.lam, gamma: B.gamma });
  const o = seed === null ? D.regular(Object.assign({}, base, { inc_seed: 1 })) : Object.assign({}, base, { inc_seed: seed });
  const g = beamletPhasors(Object.assign({}, D.DEFAULTS, o));
  const ref = seed === null ? t2.regular : t2.chaotic[seed - 1];
  const row = { label, grains: ref.grains, V_lambda_engine: ref.V_lambda, V_theta_engine: ref.V_theta, V_joint_engine: ref.V_joint,
    n_exit: g.filter(Boolean).length, V: {} };
  for (const dl of DLAMS) for (const cf of CONF) row.V[`${cf[0]} | ${(dl * 1e12).toFixed(0)}pm`] = factors(g, cf, dl);
  res.layouts.push(row);
  if ((seed || 0) % 5 === 0) console.log(label, ((Date.now() - t0) / 1000).toFixed(0) + 's', 'exit beamlets', row.n_exit, '/', g.length);
}
fs.mkdirSync(path.dirname(a.out), { recursive: true });
fs.writeFileSync(a.out, JSON.stringify(res));
console.log('wrote', a.out);
