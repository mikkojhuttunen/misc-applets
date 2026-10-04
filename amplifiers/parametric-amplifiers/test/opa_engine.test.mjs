// Checks the JavaScript port (opa_engine.js) against vectors produced by the Python
// reference engines (math-engines/tools/make_test_vectors.py). Run: node --test amplifiers/parametric-amplifiers/test/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const E = require("../opa_engine.js");
const V = JSON.parse(readFileSync(new URL("../../../math-engines/engines/parametric_amplifier/test_vectors.json", import.meta.url)));

function close(actual, expected, { rel = 1e-9, abs = 0 } = {}, what = "") {
  if (expected === null) { assert.ok(!Number.isFinite(actual), `${what}: expected non-finite, got ${actual}`); return; }
  const tol = Math.max(abs, rel * Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tol, `${what}: ${actual} vs ${expected} (tol ${tol})`);
}
const C0 = E.C0, W = (lam) => 2 * Math.PI * C0 / lam;

test("silica index", () => { for (const c of V.silica_index) close(E.nSilica(c.wavelength), c.n, { rel: 1e-13 }, `n(${c.wavelength})`); });

test("LP01 mode", () => {
  for (const c of V.lp01) {
    const r = E.lp01(c.wavelength, c.core_radius, c.delta_n);
    close(r.neff, c.n_eff, { rel: 1e-13 }, "n_eff"); close(r.V, c.V, { rel: 1e-13 }, "V"); close(r.w, c.w_mode, { rel: 1e-12 }, "w");
  }
});

test("fiber dispersion (Chebyshev interpolant) matches the exact reference", () => {
  for (const c of V.dispersion) {
    const d = E.fiberDispersion(c.process, c.lambda_pump, c.lambda_signal, c.core_radius, c.delta_n);
    // dk0 is a difference of ~1e7 rad/m wavenumbers: allow rounding at k * 1e-15
    close(d.dk0, c.dk0, { rel: 1e-12, abs: 1e-7 }, "dk0"); close(d.gvm, c.gvm, { rel: 1e-9 }, "gvm"); close(d.b2, c.beta2_sum, { rel: 1e-6 }, "beta2_sum");
    c.detuning.forEach((O, k) => close(d.dkRel(O), c.dk_rel[k], { abs: 1e-6 }, `${c.process} dk_rel(${O})`));
  }
});

test("fast quartic dispersion stays within 2e-4 rad/m out to ±4 THz", () => {
  for (const c of V.dispersion) {
    const d = E.fastDisp(c.process, c.lambda_pump, c.lambda_signal, c.core_radius, c.delta_n);
    close(d.gvm, c.gvm, { rel: 2e-3 }, "gvm");
    c.detuning.forEach((O, k) => { if (Math.abs(O) <= 2 * Math.PI * 4e12) close(d.dkRel(O), c.dk_rel[k], { abs: 2e-4 }, `fast dk_rel(${O})`); });
  }
});

test("coupling", () => {
  for (const c of V.coupling) {
    const r = E.coupling(c.process, c.lambda_pump, c.lambda_signal, c.w_pump, c.w_signal, c.w_idler, c.pump_power, c.d_eff, c.n2);
    close(r.kappa, c.gain_coefficient, { rel: 1e-12 }, "Gamma"); close(r.Aeff, c.A_eff, { rel: 1e-12 }, "A_eff"); close(r.li, c.lambda_idler, { rel: 1e-13 }, "lambda_i");
    close(2 * r.gP, c.nonlinear_phase, { rel: 1e-12, abs: 1e-300 }, "2 gamma P");
  }
});

test("undepleted analytic gain with idler/signal loss and dumps", () => {
  for (const c of V.small_signal_gain) {
    const q = { kappa: c.gain_coefficient, alpha: c.idler_loss, L: c.length, nf: c.n_dumps, tAmp: Math.sqrt(c.idler_dump_transmission), r0: 1,
      wl: { aS: c.signal_loss, tS: Math.sqrt(c.signal_dump_transmission) } };
    const a = E.analyticOut(q, c.phase_mismatch);
    close(a.fs, c.gain, { rel: 1e-9 }, "gain"); close(a.fi, c.idler_conversion, { rel: 1e-9, abs: 1e-14 }, "idler");
  }
});

test("RK4 coupled-wave solvers (chi2, chi3, losses, dumps, SBS/SRS)", () => {
  for (const c of V.amplify) {
    const i = c.inputs, lp = i.lambda_pump, ls = i.lambda_signal, li = E.idlerOf(i.process, lp, ls);
    const wP = W(lp), wS = W(ls), wI = W(li), Fp0 = i.pump_power / (E.HB * wP);
    const p = { kappa: i.gain_coefficient, alpha: i.idler_loss || 0, r0: i.signal_power / (E.HB * wS) / Fp0, L: i.length, nf: i.n_dumps || 0,
      tAmp: Math.sqrt(i.idler_dump_transmission ?? 1), wP, wS, wI, gP: i.gain_coefficient * wP / Math.sqrt(wS * wI),
      wl: { aS: i.signal_loss || 0, aP: i.pump_loss || 0, tS: Math.sqrt(i.signal_dump_transmission ?? 1), tP: Math.sqrt(i.pump_dump_transmission ?? 1) } };
    const sc = (i.sbs || i.srs) ? E.buildScattering({ wp: i.mode_radii[0], ws: i.mode_radii[1], wi: i.mode_radii[2], Pp: i.pump_power, L: i.length },
      { wP, wS, wI }, null, { process: i.process, sbs: !!i.sbs, srs: !!i.srs, linewidthHz: i.pump_linewidth || 0 }) : null;
    const r = E.simulate(i.process, p, i.phase_mismatch, false, 400000, sc);
    const tag = `${i.process} ${JSON.stringify({ G: i.gain_coefficient, L: i.length, sbs: i.sbs, srs: i.srs })}`;
    assert.equal(r.steps, c.steps, `${tag}: step count`);
    close(r.fp, c.pump_fraction, { rel: 1e-9 }, `${tag} pump`); close(r.fs, c.signal_fraction, { rel: 1e-9 }, `${tag} signal`);
    close(r.fi, c.idler_fraction, { rel: 1e-9, abs: 1e-15 }, `${tag} idler`);
    close(r.xR * i.pump_power, c.P_raman_out, { rel: 1e-9, abs: 1e-20 }, `${tag} Raman`);
  }
});

test("Brillouin parameters and exact two-wave SBS", () => {
  for (const c of V.brillouin) {
    const b = E.brillouin(c.lambda_pump, c.pump_linewidth);
    close(b.nuB, c.shift, { rel: 1e-13 }); close(b.dnuB, c.linewidth, { rel: 1e-13 }); close(b.g, c.g_B, { rel: 1e-13 }); close(b.seed, c.seed_power, { rel: 1e-12 });
  }
  for (const c of V.sbs) {
    const s = E.sbsSolve(c.P0, c.gA, c.length, c.seed);
    close(s.D, c.reflected, { rel: 1e-9 }, "reflected"); close(s.PL, c.transmitted, { rel: 1e-9 }, "transmitted"); close(s.PB(c.length / 2), c.PB_mid, { rel: 1e-9 }, "PB(L/2)");
  }
});

test("Raman gain shape", () => {
  close(E.RAMAN_PEAK.O, V.raman_peak_shift, { rel: 1e-13 }, "peak shift");
  for (const c of V.raman) close(E.ramanShape(c.shift), c.shape, { rel: 1e-12, abs: 1e-15 }, `shape(${c.shift})`);
});
