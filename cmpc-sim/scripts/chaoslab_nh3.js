/* Residual stray-light fringe of the Multipass Chaos Lab artifact (claude.ai/artifact/TLZGwP3Q27qcztDTy5qJGT) for the NH3
   line, for fig6_spectra_noise.py. Runs the artifact's own DitherLab engine (the code between "engine:begin" and
   "engine:end" in its HTML) with the rail defaults, gas NH3; both cells and their regular references.
   Usage (from cmpc-sim/):  node scripts/chaoslab_nh3.js <saved artifact .html>      -> data/chaoslab_nh3.json  (~20 s) */
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(process.argv[2], 'utf8');
const src = html.slice(html.indexOf('/* engine:begin'), html.indexOf('/* engine:end */'));
const D = new Function(src + '\nreturn DitherLab;')();
const common = { wavelength: 1531.6844239947554e-9, species: 'NH3', n_eff: 2, R_mirror: 0.995, gamma: 0.2, P_in: 1e-3,
  dither: 2e-3, waveform: 'sine', f_theta: 1000, dlam: 10 ** 1.3 * 1e-12, lwave: 'triangle', f_lambda: 1297, stray: 1,
  f_parasitic: 1e-6, speckle_avg: 1, ctr_dn: 0, n_samp: 8192, n_u: 24, n_v: 72, n_side: 12, poin_s: 40, poin_p: 14, poin_hits: 180, map_n: 7 };
const inc = { n_inc: 16, inc_radius: 50e-6, inc_dn: 0.01, inc_sign: 'both', inc_seed: 1 };
const circle = Object.assign({}, common, { cell: 'circle', c_radius: 2e-3, n_facets: 24, facet_curv: 0, c_theta: 20 * Math.PI / 180, c_port: 150e-6, c_waist: 40e-6,
  out_frac: 0.208, same_port: 0, f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1 }, inc);
const herriott = Object.assign({}, common, { cell: 'herriott', L: 10e-3, W: 1.5e-3, N: 14, M: 3, distinct: 1, port_width: 0, walls: 1, retune: 1,
  dR1: 0, dR2: 0, tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0, ripple_period: 0.5e-3 }, inc);
const drift = 0.1, Ts = []; for (let i = 0; i <= 160; i++) Ts.push(10 ** (-5 + 7 * i / 160));
const out = { settings: { circle, herriott, drift_rad_s: drift, source: 'Multipass Chaos Lab (claude.ai artifact TLZGwP3Q27qcztDTy5qJGT), DitherLab engine' }, T: Ts, cells: {} };
for (const [name, o] of [['circle', circle], ['herriott', herriott]]) for (const [v, oo] of [['current', o], ['regular', D.regular(o)]]) {
  const t0 = Date.now(), r = D.analyse(oo, { full: false, keepSpec: true });
  const sp = r.spec, amp = 2 * Math.sqrt(oo.f_parasitic) / Math.sqrt(r.grains || 1), sc = {};
  for (const k of ['none', 'angle', 'spectral', 'both']) if (sp[k]) sc[k] = { floor: amp * sp[k].floor, env: Ts.map(T => amp * D.residual(sp[k], T, drift, true)), exact: Ts.map(T => amp * D.residual(sp[k], T, drift, false)) };
  out.cells[name + '_' + v] = { T_trans: r.T, L_mean_m: r.L_mean, M2: r.M2, grains: r.grains, gamma: oo.gamma, n_pass: r.n_pass, stretch: r.stretch, lam_ftle: r.lam_ftle,
    samp_floor: amp * sp.sampFloor, samp_floor_both: amp * sp.sampFloorBoth, A_ppm_artifact: r.A_ppm, nec_shot_artifact_ppm: r.nec_shot, scen: sc };
  console.error(name, v, ((Date.now() - t0) / 1e3).toFixed(1) + ' s', 'T=', r.T.toFixed(3), 'L=', (r.L_mean * 100).toFixed(1), 'cm M2=', r.M2.toFixed(1), 'floors', Object.entries(sc).map(([k, x]) => k + ':' + x.floor.toExponential(2)).join(' '));
}
fs.writeFileSync(path.join(__dirname, '..', 'data', 'chaoslab_nh3.json'), JSON.stringify(out));
