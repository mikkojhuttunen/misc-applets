// Checks web/ngrc-engine.js against the Python reference outputs in tools/vectors.json.  node tools/check_js.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const N = require('../web/ngrc-engine.js');
const V = JSON.parse(readFileSync(new URL('./vectors.json', import.meta.url)));
const cell = V.cell, shapes = V.shapes.map(N.makeShape);
const opt = { nPos: V.source.n_pos, nAng: V.source.n_ang };
const k0 = 2 * Math.PI / cell.wavelength;
let fails = 0;
const check = (name, ok, info) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${info ? '  ' + info : ''}`); if (!ok) fails++; };
const corr = (A, B) => {   // A: [[re, im]...], B: {re, im}
  let r = 0, i = 0, na = 0, nb = 0;
  A.forEach(([ar, ai], k) => { const br = B.re[k], bi = B.im[k]; r += ar * br + ai * bi; i += ar * bi - ai * br; na += ar * ar + ai * ai; nb += br * br + bi * bi; });
  return Math.hypot(r, i) / Math.sqrt(na * nb);
};
for (const [mode, ref] of Object.entries(V.modes)) {
  const tr = N.trace(cell, 1, shapes, Object.assign({ mode }, opt));
  const ex = tr.exits.slice().sort((a, b) => a.ray - b.ray);
  const same = ex.length === ref.exits.length && ex.every((e, i) => e.ray === ref.exits[i].ray && e.port === ref.exits[i].port && e.nb === ref.exits[i].nb);
  check(`${mode}: same exits (ray, port, bounces)`, same, `${ex.length} vs ${ref.exits.length}`);
  if (same) {
    const dL = Math.max(...ex.map((e, i) => Math.abs(e.L - ref.exits[i].L))) * k0;
    const dQ = Math.max(...ex.map((e, i) => Math.hypot(e.Qr - ref.exits[i].Q[0], e.Qi - ref.exits[i].Q[1]) / Math.hypot(...ref.exits[i].Q)));
    const dA = Math.max(...ex.map((e, i) => Math.abs(e.argQ - ref.exits[i].argQ)));
    check(`${mode}: optical path`, dL < 1e-5, `max |k0 ΔL| = ${dL.toExponential(2)} rad`);
    check(`${mode}: beamlet Q, arg Q`, dQ < 1e-8 && dA < 1e-8, `rel ΔQ ${dQ.toExponential(2)}, Δarg ${dA.toExponential(2)}`);
  }
  const F = N.detectorFields(cell, tr);
  for (const p of Object.keys(ref.fields)) {
    const c = corr(ref.fields[p], F[p]);
    check(`${mode}: detector field port ${p}`, c > 1 - 1e-9, `corr ${c.toFixed(12)}`);
  }
}
const ps = new N.PhaseScreen(cell, opt).fields(shapes);
for (const k of Object.keys(V.phase_screen)) {
  const c = corr(V.phase_screen[k], ps[k]);
  check(`phase screen ${k}`, c > 1 - 1e-7, `corr ${c.toFixed(10)}`);
}
const s = shapes[1];
const chd = N.chdContour((x, y) => N.shapeValue(s, x, y), s.x0, s.y0, s.bound, 6);
const dc = Math.max(...chd.a.map((v, i) => Math.abs(v - V.chd.a[i])), ...chd.b.map((v, i) => Math.abs(v - V.chd.b[i])));
check('contour CHD', dc < 2e-3, `max |Δa, Δb| = ${dc.toExponential(2)}`);
console.log(fails ? `${fails} check(s) failed` : 'all checks passed');
process.exit(fails ? 1 : 0);
