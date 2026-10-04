// Check the JavaScript ports against the shared JSON vectors from the Python engines.
// Run from the repository root:  node math-engines/tools/check_js_ports.mjs
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const V = JSON.parse(fs.readFileSync(path.join(repo, 'math-engines/test_vectors/vectors.json'), 'utf8'));
let fails = 0, checks = 0;
const close = (label, got, want, rel, abs = 0) => {
  checks++;
  const ok = Number.isFinite(got) && Math.abs(got - want) <= Math.max(abs, rel * Math.abs(want));
  if (!ok) { fails++; console.error(`FAIL ${label}: got ${got}, want ${want}`); }
};

/* ---- dbr-structures/dbr-engine.js (lengths in µm inside the port) ---- */
{
  const ctx = vm.createContext({ Math, Float64Array, Array, Number, Map, Set, Object, JSON });
  vm.runInContext(fs.readFileSync(path.join(repo, 'dbr-structures/dbr-engine.js'), 'utf8') +
    '\n;globalThis.__e = { nIdx, slabNeff, cmtR, layerMat, mmul, mpow, rtFromM, TAU };', ctx);
  const e = ctx.__e, um = x => x * 1e6;
  const key = { sio2: 'sio2', si3n4: 'si3n4', ln_e: 'lne', ln_o: 'lno', lt_e: 'lte' };
  for (const r of V.materials) close(`dbr n(${r.material}, ${r.wavelength})`, e.nIdx({ mat: key[r.material] }, um(r.wavelength), {}), r.n, 1e-12);
  for (const r of V.slab) close(`dbr slabNeff ${JSON.stringify(r)}`, e.slabNeff(um(r.wavelength), r.n_sub, r.n_core, r.n_clad, um(r.thickness), r.polarization, r.order), r.neff, 1e-10);
  for (const r of V.cmt) close(`dbr cmtR κ=${r.kappa} δ=${r.half_detuning}`, e.cmtR(r.kappa * 1e-6, r.half_detuning * 1e-6, um(r.length)), r.R, 1e-9, 1e-12);
  for (const r of V.stack) {
    const k = e.TAU / um(r.wavelength);
    const P = e.mmul(e.layerMat(r.n_high, 0, k, um(r.d_high)), e.layerMat(r.n_low, 0, k, um(r.d_low)));
    const o = e.rtFromM(e.mpow(P, r.periods), r.n_in, r.n_out);
    close(`dbr stack R λ=${r.wavelength} N=${r.periods}`, o.R, r.R, 1e-8, 1e-10);
    close(`dbr stack T λ=${r.wavelength} N=${r.periods}`, o.T, r.T, 1e-8, 1e-10);
  }
}

/* ---- parametric-amplifier.html: physics functions extracted from the page (SI inside) ---- */
{
  const html = fs.readFileSync(path.join(repo, 'parametric-amplifier.html'), 'utf8');
  const grab = name => {
    const i = html.indexOf(`function ${name}(`);
    if (i < 0) throw new Error(`function ${name} not found in parametric-amplifier.html`);
    let j = html.indexOf('{', i), depth = 0;
    for (; j < html.length; j++) { if (html[j] === '{') depth++; else if (html[j] === '}' && --depth === 0) break; }
    return html.slice(i, j + 1);
  };
  const src = ['nSilica', 'besJ1overJ0', 'besK1overK0', 'lp01'].map(grab).join('\n');
  const ctx = vm.createContext({ Math, Map });
  vm.runInContext('var neffCache = new Map();\n' + src + '\n;globalThis.__p = { nSilica, lp01 };', ctx);
  const p = ctx.__p;
  for (const r of V.materials.filter(r => r.material === 'sio2')) close(`pa nSilica(${r.wavelength})`, p.nSilica(r.wavelength), r.n, 1e-12);
  for (const r of V.lp01) {
    const o = p.lp01(r.wavelength, r.core_radius, r.delta_n);
    close(`pa lp01 neff λ=${r.wavelength}`, o.neff, r.neff, 1e-11);
    close(`pa lp01 V λ=${r.wavelength}`, o.V, r.V, 1e-12);
    close(`pa lp01 w λ=${r.wavelength}`, o.w, r.mode_radius, 1e-9);
  }
  /* χ2 mode: overlapFor and the RK4 in simulate(), normalised amplitudes, no scattering */
  const ctx2 = vm.createContext({ Math, Float64Array });
  vm.runInContext('var MODE = "chi2"; function addScat() {}\n' + ['overlapFor', 'simulate'].map(grab).join('\n') +
    '\n;globalThis.__q = { overlapFor, simulate };', ctx2);
  const q = ctx2.__q;
  for (const r of V.chi2_overlap) close(`pa overlapFor ${r.w_pump},${r.w_signal},${r.w_idler}`, q.overlapFor(r.w_pump, r.w_signal, r.w_idler), r.overlap, 1e-13);
  for (const r of V.chi2_propagation) {
    const pp = { kappa: r.kappa, alpha: r.alpha_idler, r0: r.flux_ratio, L: r.length, nf: 0, tAmp: 1,
      wl: { aS: r.alpha_signal, aP: r.alpha_pump, tS: 1, tP: 1 } };
    const o = q.simulate(pp, r.delta_k, false, 1e9, null);
    const tag = `κ=${r.kappa} Δk=${r.delta_k} r0=${r.flux_ratio}`;
    checks++; if (o.steps !== r.steps) { fails++; console.error(`FAIL pa simulate steps ${tag}: got ${o.steps}, want ${r.steps}`); }
    close(`pa simulate fp ${tag}`, o.fp, r.fp, 1e-10);
    close(`pa simulate fs ${tag}`, o.fs, r.fs, 1e-10);
    close(`pa simulate fi ${tag}`, o.fi, r.fi, 1e-10, 1e-20);
  }
  /* χ3 mode: overlapFor (A_eff = 1/θ) and the RK4 in simulate3(), power amplitudes / √P_p0 */
  const ctx3 = vm.createContext({ Math, Float64Array });
  vm.runInContext('var MODE = "chi3"; function addScat() {}\n' + ['overlapFor', 'simulate3'].map(grab).join('\n') +
    '\n;globalThis.__q = { overlapFor, simulate3 };', ctx3);
  const q3 = ctx3.__q;
  for (const r of V.chi3_area) close(`pa overlapFor χ3 ${r.w_pump},${r.w_signal},${r.w_idler}`, 1 / q3.overlapFor(r.w_pump, r.w_signal, r.w_idler), r.area_eff, 1e-13);
  for (const r of V.chi3_propagation) {
    const C = 299792458, wP = 2 * Math.PI * C / r.wavelength_pump, wS = 2 * Math.PI * C / r.wavelength_signal, wI = 2 * Math.PI * C / r.wavelength_idler;
    const pp = { wP, wS, wI, gP: r.gamma_power, kappa: r.gamma_power * Math.sqrt(wS * wI) / wP, alpha: r.alpha_idler, r0: r.flux_ratio,
      L: r.length, nf: 0, tAmp: 1, wl: { aS: r.alpha_signal, aP: r.alpha_pump, tS: 1, tP: 1 } };
    const o = q3.simulate3(pp, r.delta_beta + 2 * r.gamma_power, false, 1e9, null);
    const tag = `γP=${r.gamma_power} Δβ=${r.delta_beta} λs=${r.wavelength_signal}`;
    checks++; if (o.steps !== r.steps) { fails++; console.error(`FAIL pa simulate3 steps ${tag}: got ${o.steps}, want ${r.steps}`); }
    close(`pa simulate3 fp ${tag}`, o.fp, r.fp, 1e-10);
    close(`pa simulate3 fs ${tag}`, o.fs, r.fs, 1e-10);
    close(`pa simulate3 fi ${tag}`, o.fi, r.fi, 1e-10, 1e-20);
  }
}

/* ---- parametric-amplifier.html: dissipative idler (lossT, pchip, analyticOut, simulate/simulate3 with dumps) ---- */
{
  const html = fs.readFileSync(path.join(repo, 'parametric-amplifier.html'), 'utf8');
  const grab = name => {
    const i = html.indexOf(`function ${name}(`);
    if (i < 0) throw new Error(`function ${name} not found in parametric-amplifier.html`);
    let j = html.indexOf('{', i), depth = 0;
    for (; j < html.length; j++) { if (html[j] === '{') depth++; else if (html[j] === '}' && --depth === 0) break; }
    return html.slice(i, j + 1);
  };
  const names = ['cm', 'ca', 'cs', 'cd', 'ce', 'csq', 'cr', 'expM', 'apply', 'analyticOut', 'pchip', 'lossT', 'simulate', 'simulate3'];
  const ctx = vm.createContext({ Math, Float64Array });
  vm.runInContext('var MODE = "chi2"; var customFn = function () { return 0; }; function addScat() {}\n' + names.map(grab).join('\n') +
    '\n;globalThis.__q = { analyticOut, pchip, lossT, simulate, simulate3, setMode: function (m) { MODE = m; } };', ctx);
  const q = ctx.__q, nm = x => x * 1e9;
  for (const r of V.idler_linear) {
    const pp = { kappa: r.gamma, alpha: r.alpha_idler, L: r.length, nf: r.dumps, tAmp: Math.pow(10, -r.dump_loss_db / 20), r0: 1,
      wl: { aS: r.alpha_signal, tS: Math.pow(10, -r.dump_loss_signal_db / 20) } };
    const o = q.analyticOut(pp, r.delta_k);
    const tag = `Γ=${r.gamma} Δk=${r.delta_k} α=${r.alpha_idler} N=${r.dumps}`;
    close(`pa analyticOut fs ${tag}`, o.fs, r.fs, 1e-9);
    close(`pa analyticOut fi ${tag}`, o.fi, r.fi, 1e-9, 1e-300);
  }
  for (const r of V.idler_weight) {
    const pr = { type: r.profile, c: nm(r.band_center), w: nm(r.band_width), e: nm(r.band_edge),
      fn: r.points ? q.pchip(r.points.map(([x, y]) => ({ x: nm(x), y }))) : null };
    close(`pa lossT ${r.profile} λ=${r.wavelength}`, q.lossT(pr, nm(r.wavelength)), r.weight, 1e-12, 1e-12);
  }
  const db = x => Math.pow(10, -x / 20);
  for (const r of V.chi2_dumps) {
    q.setMode('chi2');
    const pp = { kappa: r.kappa, alpha: r.alpha_idler, r0: r.flux_ratio, L: r.length, nf: r.dumps, tAmp: db(r.dump_loss_db),
      wl: { aS: 0, aP: 0, tS: db(r.dump_loss_signal_db), tP: db(r.dump_loss_pump_db) } };
    const o = q.simulate(pp, r.delta_k, false, 1e9, null), tag = `N=${r.dumps} dB=${r.dump_loss_db}`;
    checks++; if (o.steps !== r.steps) { fails++; console.error(`FAIL pa simulate+dumps steps ${tag}: got ${o.steps}, want ${r.steps}`); }
    for (const k of ['fp', 'fs', 'fi']) close(`pa simulate+dumps ${k} ${tag}`, o[k], r[k], 1e-10, 1e-20);
  }
  for (const r of V.chi3_dumps) {
    q.setMode('chi3');
    const C = 299792458, wP = 2 * Math.PI * C / r.wavelength_pump, wS = 2 * Math.PI * C / r.wavelength_signal, wI = 2 * Math.PI * C / r.wavelength_idler;
    const pp = { wP, wS, wI, gP: r.gamma_power, kappa: r.gamma_power * Math.sqrt(wS * wI) / wP, alpha: r.alpha_idler, r0: r.flux_ratio,
      L: r.length, nf: r.dumps, tAmp: db(r.dump_loss_db), wl: { aS: 0, aP: 0, tS: 1, tP: 1 } };
    const o = q.simulate3(pp, r.delta_beta + 2 * r.gamma_power, false, 1e9, null), tag = `N=${r.dumps} dB=${r.dump_loss_db}`;
    checks++; if (o.steps !== r.steps) { fails++; console.error(`FAIL pa simulate3+dumps steps ${tag}: got ${o.steps}, want ${r.steps}`); }
    for (const k of ['fp', 'fs', 'fi']) close(`pa simulate3+dumps ${k} ${tag}`, o[k], r[k], 1e-10, 1e-20);
  }
}

console.log(`${checks - fails}/${checks} JS port checks passed`);
process.exit(fails ? 1 : 0);
