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
    '\n;HANDLE.__vec = () => globalThis.__handle;' +
    '\n;globalThis.__e = { nIdx, slabNeff, cmtR, layerMat, mmul, mpow, rtFromM, TAU, profileFn, sampleProfile, tableAt, neffProfile, fourier, radiation, farField, crigfAt, findResonance, tmWeights, mean };', ctx);
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
  /* grating_coupler: profile, n_eff table, Fourier κ, radiation per order, guided Bragg order, far field.
     Custom layers with n at λ0 = probe λ (no dispersion); the handle index is injected as HANDLE.__vec */
  for (const r of V.grating_coupler) {
    const C = n => ({ mat: 'custom', n1: n, n2: n }), lam = um(r.wavelength), Lam = um(r.period);
    const st = { lam0: lam, lamP: lam, layers: { sub: C(r.n_sub), core: C(r.n_core), clad: C(r.n_clad) }, t: um(r.thickness), h: um(r.etch_depth),
      f: r.fill, profile: r.profile, sw: r.sidewall_angle * 180 / Math.PI, sig: um(r.edge_sigma), pol: r.polarization,
      under: r.handle ? { mode: '__vec', tbox: um(r.box_thickness) } : { mode: 'none' } };
    ctx.__handle = r.handle;
    const id = `dbr grating ${r.profile} f=${r.fill} ${r.polarization} handle=${JSON.stringify(r.handle)}`;
    const smooth = r.profile === 'smooth', rel = smooth ? 1e-5 : 1e-9, tight = smooth ? 1e-8 : 1e-12;   /* the port's erf is a 1.5e-7 approximation */
    const pf = e.profileFn(st, Lam), gS = e.sampleProfile(pf), tab = e.tableAt(st, lam, 0, !!pf.rect), nP = e.neffProfile(tab, gS);
    close(`${id} n_high`, tab.vals[tab.K], r.n_high, 1e-12);
    close(`${id} n_low`, tab.vals[0], r.n_low, 1e-12);
    const rhoTM = Math.abs(e.tmWeights(st, lam, Lam, e.mean(gS), 0).rho);   /* TM tooth model, 1 for TE */
    r.kappa.forEach((kq, i) => close(`${id} κ${i + 1}`, e.TAU * e.fourier(nP, i + 1) / lam * 1e6 * rhoTM, kq, rel, 1e-2));
    const rad = e.radiation(st, lam, 0, gS, Lam, r.periods, !!pf.rect);
    close(`${id} N0`, rad.N0, r.N0, tight);
    close(`${id} α total`, rad.alphaTot * 1e6, r.alpha_total, rel, 1e-9);
    for (const [q, med, a, th] of r.orders) {
      const o = rad.orders.find(o => o.q === q && o.med === med);
      if (!o) { checks++; fails++; console.error(`FAIL ${id}: order ${q} ${med} missing`); continue; }
      close(`${id} α q=${q} ${med}`, o.alpha * 1e6, a, rel);
      close(`${id} θ q=${q} ${med}`, o.theta * Math.PI / 180, th, tight * 100, 1e-12);
    }
    const b = rad.bragg;
    close(`${id} Bragg q`, b.q, r.bragg.q, 0);
    close(`${id} Bragg κ`, b.kap * 1e6, r.bragg.kappa, rel, 1e-2);
    close(`${id} Bragg δ`, b.dh * 1e6, r.bragg.half_detuning, 1e-10, 1e-3);
    close(`${id} Bragg R`, b.R, r.bragg.R, rel, 1e-12);
    close(`${id} λ_B`, b.lamB * 1e-6, r.bragg.lambda_B, tight);
    if (rad.beta) {
      const ff = e.farField(rad, gS, Lam, r.periods, rad.alphaTot, r.n_clad, r.theta.map(x => x * 180 / Math.PI));
      r.far_up.forEach((v, i) => close(`${id} far field θ=${r.theta[i]}`, ff[i], v, rel, 1e-12 * Math.max(...r.far_up)));
    }
  }
  /* crigf: DBR | spacer | coupler | spacer | DBR fed by a Gaussian beam (crigfAt, findResonance); the port takes
     lengths in µm, the angle in degrees and the propagation loss in 1/µm */
  const crKey = { R: 'R', T: 'T', Rd: 'Rd', escL: 'escL', escR: 'escR', lat_loss: 'latLoss', U_max: 'Umax', alpha_rad: 'alphaRad' };
  for (const r of V.crigf) {
    const C = n => ({ mat: 'custom', n1: n, n2: n }), lam0 = um(1.55e-6);
    const st = { lam0, lamP: lam0, layers: { sub: C(r.n_sub), core: C(r.n_core), clad: C(r.n_clad) }, t: um(r.thickness),
      h: um(r.dbr_etch_depth), f: r.dbr_fill, profile: r.profile, sw: r.sidewall_angle * 180 / Math.PI, sig: 0, pol: r.polarization,
      under: r.handle ? { mode: '__vec', tbox: um(r.box_thickness) } : { mode: 'none' } };
    const g = { ND: r.dbr_periods, LamD: um(r.dbr_period), NG: r.gc_periods, LamG: um(r.gc_period), hG: um(r.gc_etch_depth), fG: r.gc_fill,
      sp: um(r.spacer), Ls: um(r.straight), theta: r.theta * 180 / Math.PI, w0: um(r.w0), alphaProp: r.alpha_prop * 1e-6,
      oy: r.overlap_y, etaB: r.bounce_eta };
    ctx.__handle = r.handle;
    const id = `dbr crigf ${r.profile} ${r.polarization} ND=${r.dbr_periods} NG=${r.gc_periods} θ=${r.theta.toFixed(3)}`;
    r.wavelengths.forEach((l, i) => {
      const o = e.crigfAt(st, g, um(l), false);
      for (const [k, jk] of Object.entries(crKey)) close(`${id} ${k} λ=${l}`, o[jk], r.out[k][i], 1e-8, 1e-13);
    });
    if (r.resonance) {
      const q = r.resonance, o = e.findResonance(st, g, um(q.lam_c), um(q.fsr), q.n, false);
      close(`${id} resonance λ`, o.lam * 1e-6, q.wavelength, 0, 1e-13);
      close(`${id} resonance U`, o.Umax, q.U_max, 1e-8);
      close(`${id} resonance FWHM`, o.fwhm * 1e-6, q.fwhm, 1e-6, 1e-13);
    }
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
  for (const [i, r] of V.oblique_stack.entries())
    close(`oblique[${i}] R`, E.stackROblique(r.wavelength, r.sin_in, r.n_in, r.layers, r.n_out, r.polarization), r.R, 1e-9, 1e-14);
  const dbrOf = r => E.trenchDBR({ nTooth: r.n_tooth, lam: r.wavelength || 1.55e-6, N: r.periods, mGap: r.m_gap || 1, mTooth: r.m_tooth || 1,
    loss: r.bounce_loss || 0, slabPol: r.slab_pol, sinDesign: r.sin_design });
  for (const [i, r] of V.trench_dbr.entries()) {
    const d = dbrOf(r);
    close(`trench[${i}] d_gap`, d.dGap, r.d_gap, 1e-12);
    close(`trench[${i}] d_tooth`, d.dTooth, r.d_tooth, 1e-12);
    r.sin.forEach((sv, j) => close(`trench[${i}] R(${sv})`, d.R(r.wavelength, sv), r.R[j], 1e-9, 1e-14));
  }
  for (const [i, r] of V.cell_mirror.entries()) {
    const c = E.makeCell(5e-3, r.n_facets, r.tilts, Array(r.n_facets).fill(r.curvature), null);
    const st = E.cellMirrorStats(c, r.theta_c, r.fan, r.n_beams, r.n_hits, dbrOf(r), 1.55e-6);
    for (const k of ["R_mean", "R_eff", "I_end", "L_eff", "L_geom", "R_design", "R_uniform"]) close(`cellmirror[${i}] ${k}`, st[k], r[k], 1e-9, 1e-14);
    /* case 1 is a chaotic (curved-facet) cell: rounding grows ~e^0.6 per bounce, hence 1e-6 on angles and histogram */
    for (const k of ["chi_50", "chi_95", "chi_max"]) close(`cellmirror[${i}] ${k}`, st[k], r[k], 0, 1e-6);
    r.hist.forEach((hv, j) => close(`cellmirror[${i}] hist ${j}`, st.hist[j], hv, 1e-6, 1e-14));
  }
}

/* ---- fringe-washout.html: fringe_averaging engine block ---- */
{
  const html = fs.readFileSync(path.join(repo, 'fringe-washout.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  if (a < 0 || b < 0) throw new Error('engine block markers not found in fringe-washout.html');
  const ctx = vm.createContext({ Math, Number, Float64Array, Uint8Array, Infinity, NaN, Array, Map, Error });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__f = FringeEngine;', ctx);
  const F = ctx.__f;
  for (const r of V.bessel) {
    const J = F.besselJAll(r.x, 40);
    r.J.forEach((v, k) => close(`bessel J${k}(${r.x})`, J[k], v, 1e-12, 1e-15));
  }
  for (const [i, r] of V.fringe.entries()) {
    const c = F.fringeComponents(r.x_angle, r.f_angle, r.wave_angle, r.x_freq, r.f_freq, r.wave_freq, r.phase);
    close(`fringe[${i}] groups`, c.f.length, r.n_groups, 0, 0);
    const Vs = F.residualVisibility(c, r.T, r.drift, r.coherence, r.filter, false);
    r.V.forEach((v, k) => close(`fringe[${i}] V(T=${r.T[k]})`, Vs[k], v, 1e-9, 1e-14));
    close(`fringe[${i}] floor`, F.staticFloor(c, r.drift, r.coherence), r.floor, 1e-9, 1e-15);
  }
}

console.log(`${checks - fails}/${checks} JS port checks passed`);
process.exit(fails ? 1 : 0);
