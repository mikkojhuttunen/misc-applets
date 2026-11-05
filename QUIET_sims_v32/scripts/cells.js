/* Cell configurations shared by the QUIET scripts: the Fig. 2 settings of scripts/ens2.js (circle 4 mm, 24 facets, 16
   inclusions dn 0.01 radius 50 um, +-2 mrad sine angle dither, +-20 pm triangle wavelength dither, eps 1e-6, 1 mW) with
   the QUIET options: n_eff, membrane loss (dB/cm), constant R or an R(|sin chi|) table, and the launch of the 24-facet
   star orbit with m facets between input and exit: chi = 90 deg - 180 deg m / 24 from the wall normal, out_frac = m/24
   (m = 11: the baseline 7.5 deg; the baseline out_frac 0.458 = 11/24). */
const LAM_155 = 1653.7262586923987e-9;
const BUDn = { n_samp: 8192, n_u: 24, n_v: 72, n_side: 12, poin_s: 16, poin_p: 10, poin_hits: 700, map_n: 7 };
const DB = 4.3429448190325175;

function launchOfM(m, nFacets = 24) {
  return { c_theta: (90 - 180 * m / nFacets) * Math.PI / 180, out_frac: m / nFacets };
}

function config(cell, { n_eff = 2, R = 0.99, R_table = null, dBcm = 0.1, m = null, wavelength = LAM_155, gamma = 0.2, n_inc = 16, inc_dn = 0.01, inc_radius = 50e-6,
  dither = 2e-3, dlam = 10 ** 1.3 * 1e-12, budget = {} } = {}) {
  const common = Object.assign({ wavelength, species: 'CH4', n_eff, R_mirror: R, R_table, alpha_m: dBcm / DB / 0.01, gamma, P_in: 1e-3, dither,
    waveform: 'sine', f_theta: 1000, dlam, lwave: 'triangle', f_lambda: 1297, stray: 1, f_parasitic: 1e-6, speckle_avg: 1, ctr_dn: 0 },
  BUDn, budget);
  const inc = { n_inc, inc_radius, inc_dn, inc_sign: 'both' };
  if (cell === 'circle') {
    const ln = m === null ? { c_theta: 7.5 * Math.PI / 180, out_frac: 0.458 } : launchOfM(m);
    /* the inclusions go into the ring the launch rays cross (between the caustic a sin(theta) and the wall); a grazing
       launch leaves no room for a 50 um disk there (m = 1, 2): then they are placed anywhere in the cell (inc_ring 0),
       where the rays do not reach them */
    const r = 2e-3, apo = r * Math.cos(Math.PI / 24), R = inc.inc_radius, ringFits = 0.97 * apo - 1.5 * R > apo * Math.sin(Math.abs(ln.c_theta)) + 1.5 * R;
    return Object.assign({}, common, { cell: 'circle', c_radius: r, n_facets: 24, facet_curv: 0, c_port: 150e-6, c_waist: 40e-6, same_port: 0,
      f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: ringFits ? 1 : 0 }, ln, inc);
  }
  if (cell === 'herriott') {
    return Object.assign({}, common, { cell: 'herriott', L: 10e-3, W: 1.5e-3, N: 14, M: 3, distinct: 1, port_width: 0, walls: 1, retune: 1, dR1: 0, dR2: 0,
      tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0, ripple_period: 0.5e-3 }, inc);
  }
  throw new Error(`unknown cell ${cell}`);
}

/* tiny argv parser: --key value pairs */
function args(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const k = argv[i].slice(2), v = argv[i + 1];
    o[k] = v === undefined || v.startsWith('--') ? true : v;
    if (o[k] !== true) i++;
  }
  return o;
}
module.exports = { config, launchOfM, args, LAM_155, DB };
