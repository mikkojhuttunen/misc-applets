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


/* ---- planar-mpc-ray-tracer.html: planar_cell engine block (with ray_phase, cell_mirror, bragg_grating) ---- */
/* a = dφ/dθ by central difference (h = 1 nrad): rounding of L (~1e-16 L) gives an absolute noise of ~k0 n L 1e-16 / 1e-9 */
const aTol = (r, j) => Math.max(0.3, 4 * (2 * Math.PI * r.n_index / r.wavelength) * Math.abs(r.L0[j]) * 2.2e-16 / 2e-9);
const sameOrNaN = (label, got, want, rel, abs) => (want === null ? (checks++, Number.isNaN(got) || (fails++, console.error(`FAIL ${label}: got ${got}, want NaN`))) : close(label, got, want, rel, abs));
{
  const html = fs.readFileSync(path.join(repo, 'planar-mpc-ray-tracer.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  if (a < 0 || b < 0) throw new Error('engine block markers not found in planar-mpc-ray-tracer.html');
  const ctx = vm.createContext({ Math, Number, Float64Array, Infinity, NaN, Array, Set, Object, Error, JSON });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__p = PlanarEngine;', ctx);
  const E = ctx.__p, same = sameOrNaN;
  function build(c) {
    let cell;
    if (c.build === 'stadium') cell = E.stadiumCell(c.r, c.a, c.cap_facets, c.straight_segments);
    else if (c.build === 'circle') cell = E.circleCell(c.r);
    else if (c.build === 'polygon') cell = E.polygonCell(c.r, c.n_facets);
    else cell = E.herriottPlanarCell(c.R, c.N, c.M, c.A, { portW: c.port_w ?? null, R2: c.R2 ?? null, phase: c.phase, d: E.reentrantSpacing(c.R, c.N, c.M) });
    if (c.tilts || c.offsets || c.curvatures) cell = E.perturb(cell, c.tilts || null, c.offsets || null, c.curvatures || null);
    return cell;
  }
  for (const [i, c] of V.planar_cell.entries()) {
    const cell = build(c);
    c.elements.forEach((e, k) => { ['Ax', 'Ay', 'Bx', 'By', 'kappa'].forEach((f, q) => close(`pc[${i}] el${k} ${f}`, cell[f][k], e[q], 1e-12, 1e-17)); });
    close(`pc[${i}] area`, cell.area, c.area, 1e-12); close(`pc[${i}] P`, cell.perimeter, c.perimeter, 1e-12);
    close(`pc[${i}] meanChord`, cell.meanChord, c.mean_chord, 1e-12); close(`pc[${i}] sIn`, cell.sIn, c.s_in, 1e-12);
    const st = E.launch(cell, cell.sIn, c.theta_used);
    st.forEach((v, j) => close(`pc[${i}] start${j}`, v, c.start[j], 1e-12, 1e-15));
    const tr = E.trace(cell, st, c.hits.length + (c.leaked ? 1 : 0), null);
    checks++; if ((tr.exit === -2) !== c.leaked || tr.s.length !== c.hits.length) { fails++; console.error(`FAIL pc[${i}] hit count ${tr.s.length} vs ${c.hits.length}`); }
    const sc = (c.build === 'stadium' && c.cap_facets === null) || c.curvatures ? 1e-7 : 1e-9;
    c.hits.forEach(([x, y, s, sn, k], j) => {
      close(`pc[${i}] hit${j} x`, tr.xs[j + 1], x, 0, sc * 1e-2); close(`pc[${i}] hit${j} y`, tr.ys[j + 1], y, 0, sc * 1e-2);
      close(`pc[${i}] hit${j} s`, tr.s[j], s, 0, sc * 1e-2); close(`pc[${i}] hit${j} sinchi`, tr.sinchi[j], sn, 0, sc);
      checks++; if (tr.elem[j] !== k) { fails++; console.error(`FAIL pc[${i}] hit${j} element ${tr.elem[j]} vs ${k}`); }
    });
    if (c.port_exit) {
      const h = E.herriottTrace(cell, c.dtheta || 0, 200);
      checks++; if (h.exit !== c.port_exit || h.nHits !== c.port_n_hits) { fails++; console.error(`FAIL pc[${i}] herriott exit ${h.exit} ${h.nHits}`); }
      close(`pc[${i}] port path`, h.path, c.port_path, 1e-12); close(`pc[${i}] exit offset`, h.exitOffset, c.exit_offset, 0, 1e-12);
      close(`pc[${i}] exit angle`, h.exitAngle, c.exit_angle, 0, 1e-10); close(`pc[${i}] theta launch`, cell.meta.thetaLaunch, c.meta.theta_launch, 1e-12, 1e-18);
    }
  }
  for (const [i, r] of V.planar_phase.entries()) {
    const cell = build(V.planar_cell[r.case]);
    const o = E.ditherAnalysis(E.cellPaths(cell), r.theta_c, r.amplitude, r.n_pass, r.wavelength, r.n_index, r.waveform, r.n_min, r.n_max);
    close(`pphase[${i}] n_used`, o.nUsed, r.n_used, 0, 0);
    for (let j = 0; j < r.n_pass; j++) {
      same(`pphase[${i}] V p${j + 1}`, o.V[j], r.V[j], 0, 1e-6); same(`pphase[${i}] Vm p${j + 1}`, o.Vmodel[j], r.V_model[j], 0, 2e-5);
      same(`pphase[${i}] a p${j + 1}`, o.a[j], r.a[j], 2e-5, 0.3); same(`pphase[${i}] b p${j + 1}`, o.b[j], r.b[j], 3e-3, 2e3);
      same(`pphase[${i}] same p${j + 1}`, o.same[j], r.same_path[j], 0, 1e-12); same(`pphase[${i}] L0 p${j + 1}`, o.L0[j], r.L0[j], 1e-12);
    }
  }
  for (const [i, r] of V.planar_mirror.entries()) {
    const cell = build(V.planar_cell[r.case]);
    const d = E.trenchDBR({ nTooth: r.n_tooth, lam: 1.55e-6, N: r.periods, slabPol: r.slab_pol });
    const st = E.cellMirrorStats(cell, r.theta_c, r.fan, r.n_beams, r.n_hits, d, 1.55e-6);
    for (const k of ["R_mean", "R_eff", "I_end", "L_eff", "L_geom", "R_design", "R_uniform"]) close(`pmirror[${i}] ${k}`, st[k], r[k], 1e-9, 1e-14);
    for (const k of ["chi_50", "chi_95", "chi_max"]) close(`pmirror[${i}] ${k}`, st[k], r[k], 0, 1e-6);
    r.hist.forEach((hv, j) => close(`pmirror[${i}] hist ${j}`, st.hist[j], hv, 1e-6, 1e-14));
  }
}

/* ---- herriott-ray-tracer.html: herriott_cell engine block ---- */
{
  const html = fs.readFileSync(path.join(repo, 'herriott-ray-tracer.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  if (a < 0 || b < 0) throw new Error('engine block markers not found in herriott-ray-tracer.html');
  const ctx = vm.createContext({ Math, Number, Float64Array, Infinity, NaN, Array, Set, Object, Error, JSON });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__h = HerriottEngine;', ctx);
  const H = ctx.__h, same = sameOrNaN;
  function build(c) {
    if (c.kind === 'design' || c.kind === 'build')
      return H.buildCell({ R: c.R, N: c.N, M: c.M, A: c.A, tilt2: c.tilt2 || 0, decentre2: c.decentre2 || 0, dR: c.dR || 0, dR2: c.dR2 || 0,
        astig2: c.astig2 || 0, spacingError: c.spacing_error || 0, conic: c.conic || 0 });
    if (c.kind === 'astig') { const [Rx, Ry, d] = H.astigmaticReentrant(c.R_mean, c.N, c.Mx, c.My); return H.astigmaticCell(Rx, Ry, d, c.A, c.B, c.hole); }
    const cc = H.herriottCell(c.R, c.N, c.M, c.A);
    const m1 = H.perturbMirror(cc.mirrors[0], { poly: c.poly }); m1.holes = [];
    const st = H.startOnMirror(m1, [c.A, 0], [0, 0, 1]);
    return { mirrors: [m1, cc.mirrors[1]], p0: st.p, d0: cc.d0, wMode: cc.wMode };
  }
  for (const [i, c] of V.herriott3d.entries()) {
    const cc = build(c);
    c.mirrors.forEach((m, k) => {
      const g = cc.mirrors[k];
      m.vertex.forEach((v, q) => close(`h[${i}] m${k} vertex${q}`, g.vertex[q], v, 1e-12, 1e-15));
      m.rot.forEach((r, a) => r.forEach((v, b) => close(`h[${i}] m${k} rot${a}${b}`, g.rot[a][b], v, 1e-12, 1e-15)));
      ['Rx', 'Ry', 'kx', 'ky', 'aperture'].forEach(f => close(`h[${i}] m${k} ${f}`, g[f], m[f], 1e-12, 1e-15));
      m.holes.forEach((h, q) => h.forEach((v, z) => close(`h[${i}] m${k} hole${q}.${z}`, g.holes[q][z], v, 1e-12, 1e-15)));
    });
    c.p0.forEach((v, q) => close(`h[${i}] p0.${q}`, cc.p0[q], v, 1e-12, 1e-15)); c.d0.forEach((v, q) => close(`h[${i}] d0.${q}`, cc.d0[q], v, 1e-12, 1e-15));
    const tr = H.trace3d(cc.mirrors, cc.p0, cc.d0, c.n_max);
    checks++; if (tr.exit !== c.exit || tr.nHits !== c.n_hits || tr.reflections !== c.reflections) { fails++; console.error(`FAIL h[${i}] exit ${tr.exit} ${tr.nHits} vs ${c.exit} ${c.n_hits}`); }
    c.hits.forEach(([x, y, z, m, ci, lx, ly], j) => {
      [x, y, z].forEach((v, q) => close(`h[${i}] hit${j} ${q}`, tr.hits[j][q], v, 0, 1e-12));
      close(`h[${i}] hit${j} cos`, tr.cosInc[j], ci, 0, 1e-12); close(`h[${i}] hit${j} lx`, tr.local[j][0], lx, 0, 1e-12); close(`h[${i}] hit${j} ly`, tr.local[j][1], ly, 0, 1e-12);
      checks++; if (tr.mirror[j] !== m) { fails++; console.error(`FAIL h[${i}] hit${j} mirror`); }
    });
    close(`h[${i}] path`, tr.path, c.path, 1e-13);
    const re = H.reentrance(tr);
    close(`h[${i}] exit offset`, re.exitOffset, c.exit_offset, 1e-7, 1e-13);
    if (c.reentry_angle !== null && c.reentry_angle !== undefined) close(`h[${i}] reentry angle`, re.reentryAngle, c.reentry_angle, 1e-6, 1e-11);
    if (c.w_mode) {
      const sm = H.spotMetrics(tr, cc.mirrors, cc.wMode);
      close(`h[${i}] w`, cc.wMode, c.w_mode, 1e-12);
      c.min_spacing.forEach((v, k) => close(`h[${i}] spacing${k}`, sm[k].minSpacing, v, 1e-9));
      if (c.hole_clearance !== null && c.hole_clearance !== undefined) close(`h[${i}] hole clearance`, sm[0].holeClearance, c.hole_clearance, 1e-9);
    }
    if (c.d !== undefined) { close(`h[${i}] d`, cc.d, c.d, 1e-14); close(`h[${i}] hole`, cc.holeRadius, c.hole_radius, 1e-12); }
  }
  for (const [i, r] of V.herriott_phase.entries()) {
    const cc = build(V.herriott3d[r.case]);
    const hl = H.launcher(cc.mirrors, cc.p0, cc.d0, r.plane);
    const o = H.ditherAnalysis(hl.paths, 0, r.amplitude, r.n_pass, r.wavelength, r.n_index, r.waveform, r.n_min, r.n_max);
    close(`hphase[${i}] n_used`, o.nUsed, r.n_used, 0, 0);
    for (let j = 0; j < r.n_pass; j++) {
      same(`hphase[${i}] V p${j + 1}`, o.V[j], r.V[j], 0, 1e-6); same(`hphase[${i}] a p${j + 1}`, o.a[j], r.a[j], 2e-5, aTol(r, j));
      same(`hphase[${i}] same p${j + 1}`, o.same[j], r.same_path[j], 0, 1e-12); same(`hphase[${i}] L0 p${j + 1}`, o.L0[j], r.L0[j], 1e-12);
    }
  }
}


/* ---- onchip-herriott.html: onchip_herriott engine block (on planar_cell and bragg_grating) ---- */
{
  const html = fs.readFileSync(path.join(repo, 'onchip-herriott.html'), 'utf8');
  const a = html.indexOf('/* engine:begin'), b = html.indexOf('/* engine:end */');
  if (a < 0 || b < 0) throw new Error('engine block markers not found in onchip-herriott.html');
  const ctx = vm.createContext({ Math, Number, Float64Array, Infinity, NaN, Array, Set, Object, Error, JSON });
  vm.runInContext(html.slice(a, b) + '\n;globalThis.__o = { P: PlanarEngine, O: OnchipEngine };', ctx);
  const { P: E, O } = ctx.__o;
  /* erf against Python's math.erf; footprint is a grid count (2e-3 absolute) */
  [0, 0.3, -1.1, 1.9, 2.1, 3.5, -5].forEach(x => close(`erf(${x})`, O.erf(x), { 0: 0, 0.3: 0.3286267594591274, '-1.1': -0.8802050695740817, 1.9: 0.9927904292352575, 2.1: 0.997020533343667, 3.5: 0.9999992569016276, '-5': -0.9999999999984626 }[x], 1e-14, 1e-16));
  for (const [i, c] of V.onchip_herriott.entries()) {
    const d = E.reentrantSpacing(c.R, c.N, c.M);
    let cell = E.herriottPlanarCell(c.R, c.N, c.M, c.A, { portW: c.port_w, phase: c.phase });
    const th = cell.meta.thetaLaunch;
    cell = E.herriottPlanarCell(c.R, c.N, c.M, c.A, { portW: c.port_w, phase: c.phase, R2: c.R + (c.dR2 || 0), d });
    cell.meta.thetaLaunch = th; cell.meta.d = d;
    if (c.tilt2) cell = E.perturb(cell, [0, c.tilt2], null, null);
    const mode = O.modeProfile(cell.meta.R, d, c.lam, c.n_eff);
    close(`oh[${i}] w0`, mode.w0, c.w0, 1e-13); close(`oh[${i}] zR`, mode.zR, c.zR, 1e-13); close(`oh[${i}] theta0`, mode.theta0, c.theta0, 1e-13);
    const dbr = c.mirror === 'dbr' ? E.trenchDBR({ nTooth: c.n_eff, lam: c.lam, N: c.periods, slabPol: c.pol }) : null;
    const Rf = O.mirrorFunction({ model: c.mirror, R: c.R_mirror ?? 1, dbr, lam: c.lam, theta0: mode.theta0, beamAverage: !!c.avg, scatter: c.scatter || 0 });
    const b = O.budget(cell, { lam: c.lam, nEff: c.n_eff, hEff: 0.3e-6, Rf, exitMode: c.exit, clip: c.clip });
    checks++; if (b.nHits !== c.n_hits || b.reflections !== c.reflections || b.exits !== c.exits) { fails++; console.error(`FAIL oh[${i}] hits ${b.nHits}/${c.n_hits}`); }
    for (const [k, kk] of [["T_out", "T_out"], ["eta_window", "etaWindow"], ["I_end", "I_end"], ["L_geom", "L_geom"], ["L_eff", "L_eff"], ["V_eff", "V_eff"],
      ["chi_mean", "chi_mean"], ["chi_rms", "chi_rms"], ["chi_max", "chi_max"], ["R_mean", "R_mean"]]) close(`oh[${i}] ${k}`, b[kk], c[k], 1e-9, 1e-15);
    close(`oh[${i}] clip_loss`, b.clip_loss, c.clip_loss, 1e-7, 1e-13);
    c.I.forEach((v, j) => close(`oh[${i}] I${j}`, b.I[j], v, 1e-10, 1e-16));
    c.chi.forEach((v, j) => close(`oh[${i}] chi${j}`, b.chi[j], v, 0, 1e-9));
    c.w_int.forEach((v, j) => close(`oh[${i}] wint${j}`, b.wInt[j], v, 1e-11));
    if (c.footprint !== null) close(`oh[${i}] footprint`, O.footprintFraction(cell, c.lam, c.n_eff), c.footprint, 0, 2e-3);
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
