/* a constant R table must give the same ensemble as the constant R (engine hook check). node scripts/check_rtable.js */
const { config } = require('./cells.js');
const { DitherLab: D } = require('../engine/quiet_engine.js');
const o = Object.assign(config('circle', { R: 0.99, dBcm: 0.1, budget: { n_samp: 256, n_u: 8, n_v: 16, n_side: 6 } }), { inc_seed: 3 });
const a = D.analyse(o, { full: false }), b = D.analyse(Object.assign({}, o, { R_table: { s: [0, 0.5, 1], R: [0.99, 0.99, 0.99] } }), { full: false });
const keys = ['T', 'L_mean', 'M2', 'V_joint', 'V_theta', 'V_lambda', 'grains'];
console.log(JSON.stringify(Object.fromEntries(keys.map(k => [k, [a[k], b[k]]]))));
