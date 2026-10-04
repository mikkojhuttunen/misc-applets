// Check the JavaScript ports against the shared JSON vectors from the Python engines.
// Run from the repository root:  node math-engines/tools/check_js_ports.mjs
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

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

/* ---- amplifiers/parametric-amplifiers/opa_engine.js (SI units inside the port) ---- */
{
  const E = createRequire(import.meta.url)(path.join(repo, 'amplifiers/parametric-amplifiers/opa_engine.js'));
  const W = lam => 2 * Math.PI * E.C0 / lam, thz4 = 2 * Math.PI * 4e12;
  for (const r of V.materials.filter(r => r.material === 'sio2')) close(`pa nSilica(${r.wavelength})`, E.nSilica(r.wavelength), r.n, 1e-12);
  for (const r of V.lp01) {
    const o = E.lp01(r.wavelength, r.core_radius, r.delta_n);
    close(`pa lp01 neff λ=${r.wavelength}`, o.neff, r.neff, 1e-11);
    close(`pa lp01 V λ=${r.wavelength}`, o.V, r.V, 1e-12);
    close(`pa lp01 w λ=${r.wavelength}`, o.w, r.mode_radius, 1e-9);
  }
  for (const r of V.pa_dispersion) {
    const tag = `${r.process} λp=${r.wavelength_pump}`;
    const d = E.fiberDispersion(r.process, r.wavelength_pump, r.wavelength_signal, r.core_radius, r.delta_n);
    close(`pa dispersion dk0 ${tag}`, d.dk0, r.dk0, 1e-12, 1e-7);
    close(`pa dispersion gvm ${tag}`, d.gvm, r.gvm, 1e-9);
    close(`pa dispersion beta2 ${tag}`, d.b2, r.beta2_sum, 1e-6);
    r.detuning.forEach((O, k) => close(`pa dispersion dk_rel ${tag} Ω=${O}`, d.dkRel(O), r.dk_rel[k], 0, 1e-6));
    const f = E.fastDisp(r.process, r.wavelength_pump, r.wavelength_signal, r.core_radius, r.delta_n);
    close(`pa fastDisp gvm ${tag}`, f.gvm, r.gvm, 2e-3);
    r.detuning.forEach((O, k) => { if (Math.abs(O) <= thz4) close(`pa fastDisp dk_rel ${tag} Ω=${O}`, f.dkRel(O), r.dk_rel[k], 0, 2e-4); });
  }
  for (const r of V.pa_coupling) {
    const tag = `${r.process} λp=${r.lambda_pump}`;
    const o = E.coupling(r.process, r.lambda_pump, r.lambda_signal, r.w_pump, r.w_signal, r.w_idler, r.pump_power, r.d_eff, r.n2);
    close(`pa coupling Γ ${tag}`, o.kappa, r.gain_coefficient, 1e-12);
    close(`pa coupling A_eff ${tag}`, o.Aeff, r.A_eff, 1e-12);
    close(`pa coupling λi ${tag}`, o.li, r.lambda_idler, 1e-13);
    close(`pa coupling 2γP ${tag}`, 2 * o.gP, r.nonlinear_phase, 1e-12, 1e-300);
  }
  for (const r of V.pa_small_signal_gain) {
    const tag = JSON.stringify({ G: r.gain_coefficient, dk: r.phase_mismatch, ai: r.idler_loss, nd: r.n_dumps });
    const q = { kappa: r.gain_coefficient, alpha: r.idler_loss, L: r.length, nf: r.n_dumps, tAmp: Math.sqrt(r.idler_dump_transmission), r0: 1,
      wl: { aS: r.signal_loss, tS: Math.sqrt(r.signal_dump_transmission) } };
    const a = E.analyticOut(q, r.phase_mismatch);
    close(`pa analytic gain ${tag}`, a.fs, r.gain, 1e-9);
    close(`pa analytic idler ${tag}`, a.fi, r.idler_conversion, 1e-9, 1e-14);
  }
  for (const r of V.pa_amplify) {
    const i = r.inputs, li = E.idlerOf(i.process, i.lambda_pump, i.lambda_signal);
    const wP = W(i.lambda_pump), wS = W(i.lambda_signal), wI = W(li), Fp0 = i.pump_power / (E.HB * wP);
    const p = { kappa: i.gain_coefficient, alpha: i.idler_loss || 0, r0: i.signal_power / (E.HB * wS) / Fp0, L: i.length, nf: i.n_dumps || 0,
      tAmp: Math.sqrt(i.idler_dump_transmission ?? 1), wP, wS, wI, gP: i.gain_coefficient * wP / Math.sqrt(wS * wI),
      wl: { aS: i.signal_loss || 0, aP: i.pump_loss || 0, tS: Math.sqrt(i.signal_dump_transmission ?? 1), tP: Math.sqrt(i.pump_dump_transmission ?? 1) } };
    const sc = (i.sbs || i.srs) ? E.buildScattering({ wp: i.mode_radii[0], ws: i.mode_radii[1], wi: i.mode_radii[2], Pp: i.pump_power, L: i.length },
      { wP, wS, wI }, null, { process: i.process, sbs: !!i.sbs, srs: !!i.srs, linewidthHz: i.pump_linewidth || 0 }) : null;
    const o = E.simulate(i.process, p, i.phase_mismatch, false, 400000, sc);
    const tag = `${i.process} ${JSON.stringify({ G: i.gain_coefficient, L: i.length, sbs: i.sbs, srs: i.srs })}`;
    close(`pa amplify steps ${tag}`, o.steps, r.steps, 0);
    close(`pa amplify pump ${tag}`, o.fp, r.pump_fraction, 1e-9);
    close(`pa amplify signal ${tag}`, o.fs, r.signal_fraction, 1e-9);
    close(`pa amplify idler ${tag}`, o.fi, r.idler_fraction, 1e-9, 1e-15);
    close(`pa amplify Raman ${tag}`, o.xR * i.pump_power, r.P_raman_out, 1e-9, 1e-20);
  }
  for (const r of V.brillouin) {
    const b = E.brillouin(r.lambda_pump, r.pump_linewidth), tag = `λp=${r.lambda_pump}`;
    close(`pa brillouin shift ${tag}`, b.nuB, r.shift, 1e-13);
    close(`pa brillouin linewidth ${tag}`, b.dnuB, r.linewidth, 1e-13);
    close(`pa brillouin g_B ${tag}`, b.g, r.g_B, 1e-13);
    close(`pa brillouin seed ${tag}`, b.seed, r.seed_power, 1e-12);
  }
  for (const r of V.sbs) {
    const s = E.sbsSolve(r.P0, r.gA, r.length, r.seed), tag = `P0=${r.P0}`;
    close(`pa sbs reflected ${tag}`, s.D, r.reflected, 1e-9);
    close(`pa sbs transmitted ${tag}`, s.PL, r.transmitted, 1e-9);
    close(`pa sbs PB(L/2) ${tag}`, s.PB(r.length / 2), r.PB_mid, 1e-9);
  }
  close('pa Raman peak shift', E.RAMAN_PEAK.O, V.raman_peak_shift, 1e-13);
  for (const r of V.raman) close(`pa Raman shape(${r.shift})`, E.ramanShape(r.shift), r.shape, 1e-12, 1e-15);
}

console.log(`${checks - fails}/${checks} JS port checks passed`);
process.exit(fails ? 1 : 0);
