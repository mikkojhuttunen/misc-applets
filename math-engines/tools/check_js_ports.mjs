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
}

/* ---- cmpc-ray-tracer.html: segmented-cell engine block (billiard_cell.SegmentedCell port) ---- */
{
  const html = fs.readFileSync(path.join(repo, 'cmpc-ray-tracer.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  if (a < 0 || b < 0) throw new Error('engine block markers not found in cmpc-ray-tracer.html');
  const ctx = vm.createContext({ Math, Number, Float64Array, Infinity });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__c = CellEngine;', ctx);
  const E = ctx.__c;
  for (const [i, r] of V.segmented_cell.entries()) {
    const c = E.makeCell(r.radius, r.n_facets, r.tilts, r.curvatures, r.offsets);
    close(`cell[${i}] mean chord`, c.meanChord, r.mean_chord, 1e-12);
    const st = E.launch(c, c.h, r.theta);
    st.forEach((v, j) => close(`cell[${i}] start[${j}]`, v, r.start[j], 1e-12, 1e-15));
    const tr = E.trace(c, st, r.hits.length, null);
    r.hits.forEach(([x, y, s, sc], j) => {
      close(`cell[${i}] hit ${j} x`, tr.xs[j + 1], x, 0, 1e-9 * r.radius);
      close(`cell[${i}] hit ${j} y`, tr.ys[j + 1], y, 0, 1e-9 * r.radius);
      close(`cell[${i}] hit ${j} s`, tr.s[j], s, 0, 1e-9 * r.radius);
      close(`cell[${i}] hit ${j} sinchi`, tr.sinchi[j], sc, 0, 1e-8);
    });
  }
}

{
  const html = fs.readFileSync(path.join(repo, 'cmpc-ray-tracer.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  const ctx = vm.createContext({ Math, Number, Float64Array, Infinity, NaN, Array, Set });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__c = CellEngine;', ctx);
  const E = ctx.__c;
  for (const [i, r] of V.reflection_path.entries()) {
    const o = E.reflectionWeightedPath(r.chords, r.R);
    close(`reflpath[${i}] L`, o.L, r.L, 1e-12);
    close(`reflpath[${i}] L_eff`, o.Leff, r.L_eff, 1e-12);
    close(`reflpath[${i}] I_end`, o.Iend, r.I_end, 1e-12);
  }
  for (const [i, r] of V.ray_phase.entries()) {
    const c = E.makeCell(r.radius, r.n_facets, null, Array(r.n_facets).fill(r.curvature), null);
    const d = E.ditherOffsets(r.amplitude, r.n_samples, r.waveform);
    r.offsets.forEach((v, j) => close(`phase[${i}] offset ${j}`, d[j], v, 1e-12, 1e-18));
    const sc = E.ditherScan(c, r.theta_c, r.amplitude, r.n_pass, r.wavelength, r.n_index, r.waveform, r.n_samples);
    r.L0.forEach((v, j) => close(`phase[${i}] L0 pass ${j + 1}`, sc.L0[j], v, 1e-12));
    r.V.forEach((v, j) => close(`phase[${i}] V pass ${j + 1}`, sc.V[j], v, 0, 1e-6));
  }
  const nan = v => v === null ? NaN : v;
  const same = (label, got, want, rel, abs) => (want === null ? (checks++, Number.isNaN(got) || (fails++, console.error(`FAIL ${label}: got ${got}, want NaN`))) : close(label, got, want, rel, abs));
  for (const [i, r] of V.ray_phase_analysis.entries()) {
    const c = E.makeCell(r.radius, r.n_facets, r.tilts, null, null);
    const o = E.ditherAnalysis(c, r.theta_c, r.amplitude, r.n_pass, r.wavelength, r.n_index, r.waveform, r.n_min, r.n_max);
    close(`analysis[${i}] n_used`, o.nUsed, r.n_used, 0, 0);
    for (let j = 0; j < r.n_pass; j++) {
      same(`analysis[${i}] V p${j + 1}`, o.V[j], r.V[j], 0, 1e-6);
      /* a, b are finite differences (steps 1 nrad, 1 µrad): rounding of L gives ~1e-6 (a) and ~1e-3 (b) relative noise */
      same(`analysis[${i}] V_model p${j + 1}`, o.Vmodel[j], r.V_model[j], 0, 2e-5);
      same(`analysis[${i}] a p${j + 1}`, o.a[j], r.a[j], 2e-5, 1e-3);
      same(`analysis[${i}] b p${j + 1}`, o.b[j], r.b[j], 3e-3, 2e3);
      same(`analysis[${i}] same_path p${j + 1}`, o.same[j], r.same_path[j], 0, 1e-12);
      checks++; if (o.resolved[j] !== r.resolved[j]) { fails++; console.error(`FAIL analysis[${i}] resolved p${j + 1}`); }
    }
    void nan;
  }
}

console.log(`${checks - fails}/${checks} JS port checks passed`);
process.exit(fails ? 1 : 0);
