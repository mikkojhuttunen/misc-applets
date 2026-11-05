/* Task 9b ray-trace illustration: ray paths of the circle CMPC (set B near-IR, R 97 %, 0.3 dB/cm) under wide angle and
   position dithers of the input beam, and the exit-port phase space of the whole beam per input state.
   node scripts/t9_raytrace_dump.js -> results/t9b/raytrace.json
   - traces: centre ray of each input state (angle theta0 + A u, u in -1..1; or position offset X), first NW wall hits,
     chaotic layout (seed 1) and the regular cell;
   - exit: for each input state a Gaussian beam bundle (12 x 12 rays across waist and divergence, as GasEngine.beamRays)
     traced to the output port; exit position along the port and exit sin(chi), weight R^(h-1) exp(-alpha L) times the
     ray's Gaussian weight; old dither (40 um, +-2 mrad) against wide (75 um, +-45 mrad, +-100 um). */
const fs = require('fs');
const path = require('path');
const { loadEngine } = require('./engine_path.js');
const { config } = require('./cells.js');
const { DitherLab: D, GasEngine: G, PlanarEngine: Pl } = loadEngine();
const lam = 1.5317e-6, n = 1.5391, R = 0.97, dBcm = 0.3, NW = 28;
const kap = 2 * Math.PI * n / lam;
const base = config('circle', { n_eff: n, R, dBcm, wavelength: lam, gamma: 0.4532 });
const out = { lam, n, R, dBcm, cells: {} };

function trace(S, x, th, rec, nB = 2000) {
  const c = S.mc.cell, cosT = Math.cos(S.theta);
  return G.marchRay(S.mc, Pl.launch(c, c.sIn + x / cosT, th), nB, c.sIn, c.sOut, c.meta.portW, 'out', rec);
}

for (const [label, o] of [['chaotic', Object.assign({}, base, { inc_seed: 1 })], ['regular', D.regular(Object.assign({}, base, { inc_seed: 1 }))]]) {
  const S = D.setup(Object.assign({}, D.DEFAULTS, o)), c = S.mc.cell, P = c.perimeter, th0 = S.theta;
  G.setAlpha(o.alpha_m); G.setRTable(null);
  const outline = Array.from({ length: 721 }, (_, i) => Pl.pointAt(c, P * i / 720).slice(0, 2));
  const inc = D.inclusions(c, Object.assign({}, D.DEFAULTS, o));
  const ports = { in: Pl.pointAt(c, c.sIn).slice(0, 2), out: Pl.pointAt(c, c.sOut).slice(0, 2), w: c.meta.portW };
  const cut = r => { const k = Math.min(r.pts.length, NW + 1); return r.pts.slice(0, k); };
  const angle = [-1, -0.5, 0, 0.5, 1].map(u => { const r = trace(S, 0, th0 + 45e-3 * u, true, NW); return { u, mrad: 45 * u, pts: cut(r) }; });
  const pos = [-1, 0, 1].map(u => { const r = trace(S, 100e-6 * u, th0, true, NW); return { u, um: 100 * u, pts: cut(r) }; });
  /* exit phase space for input states of two dither designs */
  const qs = [-2.1, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.1].map(v => v / 2.1);
  const exitOf = (w, A, Xa, nStates) => {
    const states = [];
    for (let i = 0; i < nStates; i++) {
      const u = Math.sin(2 * Math.PI * (i + 0.5) / nStates), v = Math.sin(2 * Math.PI * 0.6180339887 * (i + 0.5) * 3);
      const X = Xa * v, Th = A * u, pts = [];
      for (const a of qs) for (const b of qs) {
        const x = X + w * 0.5 * a * 2.1, th = th0 + Th + (2 / (kap * w)) * 0.5 * b * 2.1, wt = Math.exp(-0.5 * ((a * 2.1) ** 2 + (b * 2.1) ** 2) / 1);
        if (Math.abs(x) > 250e-6) continue;
        const r = trace(S, x, th, false);
        if (r.exit !== 1) continue;
        const h = r.nWalls, p = wt * Math.pow(R, h - 1) * Math.exp(-o.alpha_m * r.WG[h - 1]);
        let ds = r.WS[h - 1] - c.sOut; ds -= P * Math.round(ds / P);
        pts.push([ds * 1e6, r.WC[h - 1], p, r.WG[h - 1] * 100]);
      }
      states.push({ X: X * 1e6, A: Th * 1e3, pts });
    }
    return states;
  };
  out.cells[label] = { outline, inc: { cx: inc.cx, cy: inc.cy, r: inc.r }, ports, theta0: th0, angle, pos,
    exit_old: exitOf(40e-6, 2e-3, 0, 9), exit_wide: exitOf(75e-6, 45e-3, 100e-6, 9) };
}
const f = path.join(__dirname, '..', 'results', 't9b', 'raytrace.json');
fs.writeFileSync(f, JSON.stringify(out));
console.log('wrote', f);
