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

/* ---- parametric-amplifier/index.html: physics functions extracted from the page (SI inside) ---- */
{
  const html = fs.readFileSync(path.join(repo, 'parametric-amplifier/index.html'), 'utf8');
  const grab = name => {
    const i = html.indexOf(`function ${name}(`);
    if (i < 0) throw new Error(`function ${name} not found in parametric-amplifier/index.html`);
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
}

/* ---- er-waveguide-amplifier/er-engine.js: WGCORE (µm inside) and ER (SI) ---- */
{
  const ctx = vm.createContext({ Math, Float64Array, Float32Array, Uint8Array, Array, Number, Object, Error, JSON, isFinite });
  vm.runInContext(fs.readFileSync(path.join(repo, 'er-waveguide-amplifier/er-engine.js'), 'utf8') + '\n;globalThis.__C = WGCORE(); globalThis.__ER = ER;', ctx);
  const C = ctx.__C, ER = ctx.__ER, um = x => x * 1e6;
  for (const r of V.materials_er) close(`er n(${r.material}, ${r.wavelength})`, C.nMat({ id: r.material }, um(r.wavelength)), r.n, 1e-12);
  for (const r of V.er_cross_sections) {
    close(`er σa(${r.wavelength})`, ER.sigmaAbsorption(r.wavelength, r.sigma_a_peak), r.sigma_a, 1e-12);
    close(`er σe(${r.wavelength}, ${r.temperature} K)`, ER.sigmaEmission(r.wavelength, r.sigma_a_peak, r.temperature), r.sigma_e, 1e-12);
  }
  for (const r of V.er_pump_sigmas) {
    const o = ER.pumpSigmas(r.pump_wavelength, r.sigma_a_peak, r.sigma_a_980, r.temperature);
    close(`er pump σa(${r.pump_wavelength})`, o.a, r.sigma_a, 1e-12); close(`er pump σe(${r.pump_wavelength})`, o.e, r.sigma_e, 1e-12, 1e-40);
  }
  for (const r of V.er_concentration) {
    const o = ER.concentrationEffects(r.n_er, r.c_up, r.upconversion, r.quenching, r.k_q, r.f_q);
    close(`er C_up ${r.upconversion}/${r.quenching}`, o.cupEff, r.c_up_eff, 1e-12); close(`er f_q ${r.upconversion}/${r.quenching}`, o.fq, r.f_q_eff, 1e-12, 1e-15);
  }
  for (const r of V.er_upper_population) close(`er N2 ru=${r.rate_up} C=${r.c_up}`, ER.upperPopulation(r.rate_up, r.rate_down, r.n_active, r.tau, r.c_up), r.n2, 1e-12, 1);
  for (const r of V.er_propagate) {
    const o = ER.propagate({ pumpPower: r.pump_power, signalPower: r.signal_power, length: r.length, steps: r.steps, wp: r.weight_pump, ws: r.weight_signal,
      area: r.cell_area, nEr: r.n_er, tau: r.tau, cupEff: r.c_up_eff, fq: r.f_q, sap: r.sigma_ap, sep: r.sigma_ep, sas: r.sigma_as, ses: r.sigma_es,
      lp: r.pump_wavelength, ls: r.signal_wavelength, alpha: r.alpha });
    const tag = `er propagate Pp=${r.pump_power} λp=${r.pump_wavelength}`;
    close(`${tag} gain`, o.gainLn, r.gain_ln, 1e-10, 1e-12); close(`${tag} pump out`, o.pumpPower[r.steps], r.pump_out, 1e-10, 1e-15);
    close(`${tag} I1`, o.I1, r.I1, 1e-10); close(`${tag} I2`, o.I2, r.I2, 1e-10, 1e-6);
    close(`${tag} inv in`, o.inversion[0], r.inversion_in, 1e-10, 1e-14); close(`${tag} inv out`, o.inversion[r.steps], r.inversion_out, 1e-10, 1e-14);
    close(`${tag} probe 1550`, ER.probeGainLn(1.55e-6, o.I1, o.I2, r.length, r.alpha, 1, 5.7e-25, 295), r.probe_1550, 1e-10, 1e-12);
  }
  const cst = n => ({ id: 'const', n });
  for (const r of V.eim) {
    const G = { geo: r.geometry, t: um(r.film_thickness), w: um(r.width), h: um(r.strip_height), tout: um(r.slab_thickness), d: um(r.depth),
      mats: { sup: cst(r.n_sup), strip: cst(r.n_strip), film: cst(r.n_film), sub: cst(r.n_sub) } };
    const pols = {}; pols[r.polarization] = true;
    const got = C.solveEIM(G, um(r.wavelength), pols).modes.map(m => m.N);
    r.neff.forEach((n, i) => close(`er EIM ${r.geometry} ${r.polarization} #${i}`, got[i], n, 1e-9));
  }
  for (const r of V.fd) {
    const xe = r.x_edges.map(um), ye = r.y_edges.map(um), Nx = xe.length - 1, Ny = ye.length - 1;
    const grid = { Nx, Ny, xc: new Float64Array(Nx), dx: new Float64Array(Nx), yc: new Float64Array(Ny), dy: new Float64Array(Ny),
      reg: Uint8Array.from(r.regions), cb: { y0: um(r.core_y[0]), y1: um(r.core_y[1]) } };
    for (let i = 0; i < Nx; i++) { grid.xc[i] = 0.5 * (xe[i] + xe[i + 1]); grid.dx[i] = xe[i + 1] - xe[i]; }
    for (let j = 0; j < Ny; j++) { grid.yc[j] = 0.5 * (ye[j] + ye[j + 1]); grid.dy[j] = ye[j + 1] - ye[j]; }
    const got = C.fdSolve(grid, r.indices, um(r.wavelength), r.polarization, r.neff.length).map(m => m.N);
    r.neff.forEach((n, i) => close(`er FD ${r.polarization} #${i}`, got[i], n, 1e-9));
  }
}

console.log(`${checks - fails}/${checks} JS port checks passed`);
process.exit(fails ? 1 : 0);
