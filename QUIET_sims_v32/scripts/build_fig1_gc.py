"""Fig. 1 artifact with the grating-coupler design (QUIET Task 11) as a selectable input design.

python scripts/build_fig1_gc.py    (artifacts/fig1_src_live.html -> artifacts/fig1_v32.html)

- Device panel: "Input design" buttons, waveguide input with on-chip heater dither (as before) or the new design: a large-area
  grating coupler on the membrane fed by a free-space beam from a scanning mirror (angle + position dither); default: new.
- 3D chip (memChip): with the new design the input waveguide is replaced by a grating patch on a wider input window, the
  free-space beam fans in from the scanning mirror above the chip, and the drawn ray orbit uses the design's 8 inclusions of
  r 25 um.
- Figure panel (e): with the new design the dither system is a separate block between the source and the chip (scanning
  mirror and scan lens, driven by the electronics); the chip holds the grating coupler, the CMPC and the output ports.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def rep(s, a, b, cnt=1):
    assert a in s, a[:100]
    return s.replace(a, b, cnt)


def main():
    s = (ART / "fig1_src_live.html").read_text(encoding="utf-8")
    # ---------------------------------------------------------------- controls
    s = rep(s, '        <label class="check" for="pillarOn"><input id="pillarOn" type="checkbox" checked> CMPC: central support pillar</label>',
            '        <label class="check" for="pillarOn"><input id="pillarOn" type="checkbox" checked> CMPC: central support pillar</label>\n'
            '        <div class="ctl"><label>Input design (CMPC chip and panel e)</label>\n'
            '        <div class="seg" role="group" aria-label="Input design" id="inSeg">\n'
            '          <button type="button" data-indes="wg" aria-pressed="false">Waveguide + on-chip dither</button>\n'
            '          <button type="button" data-indes="gc" aria-pressed="true">Grating coupler + free-space dither (new)</button>\n'
            '        </div>\n'
            '        <p class="hint">New design (QUIET Task 11): a free-space beam on a large-area input grating coupler, its angle (±60 mrad) and position (±100 µm) swept by a scanning mirror off the chip; 8 index inclusions of r 25 µm, output on the facet the launch orbit aims at.</p></div>')
    s = rep(s, '  const S = { dev: "mem", mview: "chip", cview: "gc", pol: "TM" };', '  const S = { dev: "mem", mview: "chip", cview: "gc", pol: "TM", indes: "gc" };')
    s = rep(s, "  for (const b of document.querySelectorAll(\"[data-pol]\")) b.addEventListener(\"click\", () => { S.pol = b.dataset.pol; press(\"[data-pol]\", \"pol\", S.pol); update(); });",
            "  for (const b of document.querySelectorAll(\"[data-pol]\")) b.addEventListener(\"click\", () => { S.pol = b.dataset.pol; press(\"[data-pol]\", \"pol\", S.pol); update(); });\n"
            "  for (const b of document.querySelectorAll(\"[data-indes]\")) b.addEventListener(\"click\", () => { S.indes = b.dataset.indes; press(\"[data-indes]\", \"indes\", S.indes); needFit = true; update(); if (typeof liveDraw === \"function\") { liveDraw(); setTimeout(refreshPanels, 300); } });")
    s = rep(s, 'gc: "#e87a29", gcSec: "#f4b787" };', 'gc: "#e87a29", gcSec: "#f4b787", mirror: "#9aa3ad" };')
    # ---------------------------------------------------------------- 3D: orbit with the design's inclusions
    s = rep(s, '  function memOrbit(N, chiDeg, nInc) {', '  function memOrbit(N, chiDeg, nInc, incR = 50e-6) {')
    s = rep(s, 'f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1, n_inc: nInc, inc_radius: 50e-6, inc_dn: 0.01, inc_sign: "both",',
            'f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 5, f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1, n_inc: nInc, inc_radius: incR, inc_dn: 0.01, inc_sign: "both",')
    s = rep(s, '    for (const nInc of ($("mInc").checked ? [16, 0] : [0])) { try { orbit = memOrbit(N, chiDeg, nInc); break; } catch (e) { orbit = null; } }',
            '    const GCD = S.indes === "gc";\n'
            '    for (const nInc of ($("mInc").checked ? [GCD ? 8 : 16, 0] : [0])) { try { orbit = memOrbit(N, chiDeg, nInc, GCD ? 25e-6 : 50e-6); break; } catch (e) { orbit = null; } }')
    # wider input window for the grating-coupler beam
    s = rep(s, '    const inWin = Q => clip(clip(clip(Q, nb[0], nb[1], c0 + wp / 2), -nb[0], -nb[1], -c0 + wp / 2), d0[0], d0[1], cb + 1e-6);',
            '    const wpIn = GCD ? 3 * wp : wp;\n'
            '    const inWin = Q => clip(clip(clip(Q, nb[0], nb[1], c0 + wpIn / 2), -nb[0], -nb[1], -c0 + wpIn / 2), d0[0], d0[1], cb + 1e-6);')
    # grating patch instead of the input waveguide, the free-space fan from the scanning mirror
    s = rep(s, '    guide(sIn, uIn);\n',
            '    let free = null, gcC = null;\n'
            '    if (GCD) {                                         /* large-area grating coupler: teeth across the propagation direction */\n'
            '      const wv = [-uIn[1], uIn[0]], Wg = 3 * wp, gp = 0.024, tooth = 0.012, nT = 9;\n'
            '      const P = (s_, h) => [sIn[0] + s_ * uIn[0] + h * wv[0], sIn[1] + s_ * uIn[1] + h * wv[1]];\n'
            '      for (let i = 0; i < nT; i++) { const a_ = 0.01 + i * gp; B.add(sq([P(a_, -Wg / 2), P(a_ + tooth, -Wg / 2), P(a_ + tooth, Wg / 2), P(a_, Wg / 2)]), t, t + ts, COL.gc, COL.gcSec, cut); }\n'
            '      gcC = P(0.01 + nT * gp / 2, 0);\n'
            '      free = freeFan(gcC, uIn, wv, Wg, t + ts);\n'
            '    } else guide(sIn, uIn);\n')
    s = rep(s, '    const inSegs = [toEdge(sIn, uIn), sIn], inBand = [sIn, p0];',
            '    const inSegs = GCD ? [] : [toEdge(sIn, uIn), sIn], inBand = GCD ? [gcC, p0] : [sIn, p0];')
    s = rep(s, '      { text: "tapered strip-loaded waveguide", p: [sIn[0] + uIn[0] * 0.2, sIn[1] + uIn[1] * 0.2, t + ts] },',
            '      GCD ? { text: "large-area grating coupler (free-space input)", p: [gcC[0], gcC[1], t + ts] } : { text: "tapered strip-loaded waveguide", p: [sIn[0] + uIn[0] * 0.2, sIn[1] + uIn[1] * 0.2, t + ts] },\n'
            '      ...(GCD ? [{ text: "free-space beam: angle + position dither", p: free.top }] : []),')
    s = rep(s, '    return { B, isSec, segs, inSegs, inBand, outSegs, glow, mols, incs, cloud: { pieces: cloudPieces, z: t, reach }, beamZ: t + 0.002, beamW: 0.012, labels, info: { M, N, Np, m, t, band, nOut, nOutPorts } };',
            '    return { B, isSec, segs, inSegs, inBand, outSegs, glow, mols, incs, free, cloud: { pieces: cloudPieces, z: t, reach }, beamZ: t + 0.002, beamW: 0.012, labels, info: { M, N, Np, m, t, band, nOut, nOutPorts, gc: GCD } };')
    # draw the free-space fan and the mirror
    s = rep(s, '      if (desc.glow && $("glowOn").checked) for (const g of desc.glow) group.add(new THREE.Mesh(g, glowMaterial()));',
            '      if (desc.free && showBeam) {                     /* free-space beam: source -> scanning mirror -> fan of dither states on the grating */\n'
            '        const fg = new THREE.Group(), V = a_ => new THREE.Vector3(...a_);\n'
            '        const rod = (a_, b_, r_, op) => { const A_ = V(a_), B_ = V(b_), L_ = A_.distanceTo(B_), m_ = new THREE.Mesh(new THREE.CylinderGeometry(r_, r_, L_, 10), new THREE.MeshBasicMaterial({ color: COL.beam, transparent: op < 1, opacity: op, depthWrite: op >= 1 }));\n'
            '          m_.position.copy(A_.clone().add(B_).multiplyScalar(0.5)); m_.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B_.clone().sub(A_).normalize()); fg.add(m_); };\n'
            '        for (const [a_, b_, r_, op] of desc.free.rods) rod(a_, b_, r_, op);\n'
            '        group.add(fg);\n'
            '      }\n'
            '      if (desc.glow && $("glowOn").checked) for (const g of desc.glow) group.add(new THREE.Mesh(g, glowMaterial()));')
    s = rep(s, '<span><i class="sw" style="background:${COL.port}"></i>input beam</span><span><i class="sw" style="background:${COL.strip}"></i>input waveguide</span><span><i class="sw" style="background:${COL.outStrip}"></i>output ports and waveguides</span>',
            '${desc.info.gc ? `<span><i class="sw" style="background:${COL.gc}"></i>input grating coupler</span><span><i class="sw" style="background:${COL.beam}"></i>free-space beam (dither fan)</span>` : `<span><i class="sw" style="background:${COL.port}"></i>input beam</span><span><i class="sw" style="background:${COL.strip}"></i>input waveguide</span>`}<span><i class="sw" style="background:${COL.outStrip}"></i>output ports and waveguides</span>')
    s = rep(s, 'port: "#1f8a4c"', 'port: "#d0352b"')
    # Herriott DBRs: mechanical bridges across the trenches, outside the spot pattern (|y| = 1.2 A, the aperture is 1.4 A), as on the CMPC facets
    s = rep(s, '    ys.add(0); if (Math.abs(y0 - wp / 2) < D)',
            '    const yBr = 1.2 * A, bwH = Math.max(0.03, 0.04 * A), brOn = yBr + bwH / 2 < D;\n'
            '    if (brOn) for (const sg of [-1, 1]) for (const e_ of [-1, 1]) ys.add(+(sg * yBr + e_ * bwH / 2).toFixed(12));\n'
            '    ys.add(0); if (Math.abs(y0 - wp / 2) < D)')
    s = rep(s, '      const ya = Ys[i], yb = Ys[i + 1], ym = (ya + yb) / 2, port = Math.abs(ym - y0) <= wp / 2;',
            '      const ya = Ys[i], yb = Ys[i + 1], ym = (ya + yb) / 2, port = Math.abs(ym - y0) <= wp / 2, br = brOn && Math.abs(Math.abs(ym) - yBr) <= bwH / 2;')
    s = rep(s, '      if (port) add(cur(0, bandOff), cur(0, 0));', '      if (port || br) add(cur(0, bandOff), cur(0, 0));')
    s = rep(s, '      for (let j = 0; j < Np; j++) add(cur(1, j * (g + tw) + g), cur(1, (j + 1) * (g + tw)));',
            '      if (br) add(cur(1, 0), cur(1, bandOff)); else for (let j = 0; j < Np; j++) add(cur(1, j * (g + tw) + g), cur(1, (j + 1) * (g + tw)));')
    s = rep(s, 'the corner bridges hold the inner membrane.</p>', 'the corner bridges hold the inner membrane. The Herriott DBRs have bridges at |y| = 1.2 A, outside the spot pattern.</p>')
    # (c): as tall as the (d) axis box, so (d) and (e) set the row height
    s = rep(s, 'const cx0 = 0, cy0 = 164, cw = 74, ch = Math.round(COMP_CH * 0.88), mid = cy0 + ch / 2;',
            'const cx0 = 0, cy0 = 164, cw = 74, ch = Math.round(COMP_CH - 2.35 * fs), mid = cy0 + ch / 2;')
    # short free-space fan shared by (a) and (b): only the last part of the beam above the grating is drawn (the scanning mirror is in e)
    s = rep(s, '  function memChip() {',
            '  function freeFan(gcC, u, wv, Wg, z, H = 0.22, run = 0.1) {   /* dither states: parallel shifts (position) with a small angle spread, faded upper ends */\n'
            '    const rods = [], F = [-1, -0.5, 0, 0.5, 1];\n'
            '    for (const f of F) {\n'
            '      const e = [gcC[0] + wv[0] * f * Wg * 0.38, gcC[1] + wv[1] * f * Wg * 0.38, z];\n'
            '      const a = [e[0] + u[0] * run + wv[0] * f * Wg * 0.25, e[1] + u[1] * run + wv[1] * f * Wg * 0.25, z + H];\n'
            '      const m = [e[0] + (a[0] - e[0]) * 0.6, e[1] + (a[1] - e[1]) * 0.6, z + H * 0.6], c = f === 0;\n'
            '      rods.push([m, e, c ? 0.008 : 0.005, c ? 1 : 0.45], [a, m, c ? 0.008 : 0.005, c ? 0.35 : 0.15]);\n'
            '    }\n'
            '    return { rods, top: [gcC[0] + u[0] * run, gcC[1] + u[1] * run, z + H] };\n'
            '  }\n'
            '  function memChip() {')
    # (b) integrated Herriott cell: grating coupler at the input port
    s = rep(s, '''    for (const [u, kind] of [[[-h.inDir[0], -h.inDir[1]], "in"], [h.exitDir, "out"]]) {
      if (kind === "out" && h.exit !== "port") continue;
      const w = [-u[1], u[0]], st = [p0[0] + u[0] * (band + 0.03), p0[1] + u[1] * (band + 0.03)];
      const P = (s_, hh) => [st[0] + s_ * u[0] + hh * w[0], st[1] + s_ * u[1] + hh * w[1]], w1 = wp, w2 = 0.35 * wp, lt = 0.3;''',
            '''    const GCD = S.indes === "gc"; let free = null, gcC = null;
    for (const [u, kind] of [[[-h.inDir[0], -h.inDir[1]], "in"], [h.exitDir, "out"]]) {
      if (kind === "out" && h.exit !== "port") continue;
      const w = [-u[1], u[0]], st = [p0[0] + u[0] * (band + 0.03), p0[1] + u[1] * (band + 0.03)];
      const P = (s_, hh) => [st[0] + s_ * u[0] + hh * w[0], st[1] + s_ * u[1] + hh * w[1]], w1 = wp, w2 = 0.35 * wp, lt = 0.3;
      if (GCD && kind === "in") {                      /* grating coupler on the port axis, fed from above by the free-space beam */
        const gp = 0.024, tooth = 0.012, nT = 9, Wg = Math.max(wp, 0.12);
        for (let i = 0; i < nT; i++) { const a_ = 0.02 + i * gp; B.add(sq([P(a_, -Wg / 2), P(a_ + tooth, -Wg / 2), P(a_ + tooth, Wg / 2), P(a_, Wg / 2)]), t, t + ts, COL.gc, COL.gcSec, cut); }
        gcC = P(0.02 + nT * gp / 2, 0); free = freeFan(gcC, u, w, Wg, t + ts, 0.4, 0.16);
        io.push([kind, gcC, u]); continue;
      }''')
    s = rep(s, '    for (const [kind, st, u] of io) { (kind === "in" ? inSegs : outSegs).push(toEdge(st, u), st); inBand.push(st, p0); }',
            '    for (const [kind, st, u] of io) { if (!(GCD && kind === "in")) (kind === "in" ? inSegs : outSegs).push(toEdge(st, u), st); inBand.push(st, p0); }')
    s = rep(s, '      { text: "tapered strip-loaded waveguides", p: io.length',
            '      { text: GCD ? "input grating coupler, output waveguide" : "tapered strip-loaded waveguides", p: io.length')
    s = rep(s, '    return { B, isSec, segs, inSegs, inBand, outSegs, glow, mols, cloud: { pieces: cloudPieces, z: t, reach }, framing:',
            '    return { B, isSec, segs, inSegs, inBand, outSegs, glow, mols, free, cloud: { pieces: cloudPieces, z: t, reach }, framing:')
    s = rep(s, '<span><i class="sw" style="background:${COL.port}"></i>input beam</span><span><i class="sw" style="background:${COL.strip}"></i>input waveguide</span><span><i class="sw" style="background:${COL.outStrip}"></i>output waveguide</span>',
            '${S.indes === "gc" ? `<span><i class="sw" style="background:${COL.gc}"></i>input grating coupler</span><span><i class="sw" style="background:${COL.beam}"></i>free-space beam (dither fan)</span>` : `<span><i class="sw" style="background:${COL.port}"></i>input beam</span><span><i class="sw" style="background:${COL.strip}"></i>input waveguide</span>`}<span><i class="sw" style="background:${COL.outStrip}"></i>output waveguide</span>')
    # ---------------------------------------------------------------- panel (e) with the dither system off the chip
    egc = r'''
  // ---- panel (e), grating-coupler design: the dither system (scanning mirror, scan lens) is its own block between the source and the chip
  function eCompactGC(p, fs, WE) {
    const Pl = PlanarEngine, G = GasEngine, D = DitherLab;
    const n2 = x => Math.round(x * 100) / 100, pt = q => `${n2(q[0])} ${n2(q[1])}`, esc = t => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    const cx_ = document.createElement("canvas").getContext("2d");
    const tw = (s_, b) => { cx_.font = `${b ? "700 " : ""}${fs}px ${LIVE_FONT}`; return cx_.measureText(s_).width; };
    const COL = { gold: "#c8962a", blue: "#1f3f93", grey: "#c3c9cd", beam: "#ef6f6c", ray: "#d9342b", elec: "#5a6a6f", gc: "#e87a29", inc: "#e8590c", ink: "#142026", mut: "#5a6a6f" };
    const out = [], lh = 1.2 * fs;
    const txt = (x, y, str, a = {}) => out.push(`<text x="${n2(x)}" y="${n2(y)}" font-size="${n2(fs)}"${a.b ? ' font-weight="700"' : ""}${a.anchor ? ` text-anchor="${a.anchor}"` : ""} fill="${a.fill || COL.ink}"${a.halo ? ` stroke="${a.halo}" stroke-width="${n2(fs * 0.3)}" stroke-linejoin="round" paint-order="stroke"` : ""}>${a.raw || esc(str)}</text>`);
    const line = (d, col, w, extra = "") => out.push(`<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"${extra}/>`);
    const g = 5, HB = 104, y0 = 48, ELEC = ' stroke-dasharray="3.5 2.5"';
    const bwE = Math.ceil(tw("electronics") + 10), xs_ = Math.max(56, 5 + bwE + 9);   // driver box, then the reference pick-off column right of it
    const wI = Math.max(Math.ceil(tw("(e)") + 5 + tw("I: Source") + 16), xs_ + 10);
    const wD = Math.max(58, Math.ceil(Math.max(tw("Dither"), tw("±θ, ±x") + 22) + 16));
    const wIII = Math.max(88, Math.ceil(tw("III: Detection") + 8)), xD = wI + g, xII = xD + wD + g, xIII = WE - wIII, xE = xIII - g;
    // the design cell of Task 11 (8 inclusions r 25 um when shown); centre ray and an edge state of the input dither
    const o = { cell: "circle", c_radius: 2e-3, n_facets: 24, facet_curv: 0, c_theta: p.th * Math.PI / 180, out_frac: 0.458, same_port: 0, c_port: 150e-6, c_waist: 75e-6, f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 5,
      f_sel_tilt: 0, f_sel_curv: 0, inc_ring: 1, n_inc: p.inc ? 8 : 0, inc_radius: 25e-6, inc_dn: 0.01, inc_sign: "both", inc_seed: 1, wavelength: 1.6537e-6, n_eff: 2, ctr_dn: 0 };
    const S_ = D.setup(o), mc = S_.mc, c = mc.cell, N = c.meta.nFacets;
    const RC = 20, PITCH = 2.5, ROUT = RC + 4 * PITCH, CX = xE - 24 - ROUT, CY = y0, k = RC / c.meta.r, PIN = Math.PI;
    const pin = Pl.pointAt(c, c.sIn), pout = Pl.pointAt(c, c.sOut), rot = PIN - Math.atan2(pin[1], pin[0]), cs = Math.cos(rot), sn = Math.sin(rot);
    const Tf = (x, y) => [CX + (x * cs - y * sn) * k, CY - (x * sn + y * cs) * k];
    const angOut = Math.atan2(pout[1], pout[0]) + rot, angs = [angOut + 0.75, angOut, angOut - 0.75];
    const onRing = (a, r) => [CX + r * Math.cos(a), CY - r * Math.sin(a)];
    const qp = (a, r, b) => [CX + r * Math.cos(a) - b * Math.sin(a), CY - (r * Math.sin(a) + b * Math.cos(a))];
    const poly = pts => "M" + pts.map(pt).join("L") + "Z", quad = (a, r1, r2, w1, w2) => poly([qp(a, r1, -w1), qp(a, r2, -w2), qp(a, r2, w2), qp(a, r1, w1)]);
    const vert = rad => Array.from({ length: N }, (_, i) => { const a = rot + 2 * Math.PI * i / N - Math.PI / N; return [CX + rad * Math.cos(a), CY - rad * Math.sin(a)]; });
    const st = p.beams == 1 ? [[0, 0]] : [[0, 0], [1, 1]];
    const beams = st.map(([ua, ux]) => G.marchRay(mc, Pl.launch(c, c.sIn + ux * 100e-6, S_.theta + ua * 45e-3), p.hits, c.sIn, c.sOut, c.meta.portW, "out", true));
    const bp = beams.map(r => r.pts); for (let i = 1; i < bp.length; i++) bp[i] = bp[i].slice(0, bp[0].length);
    out.push(`<defs><linearGradient id="gLas2" x1="0" x2="1"><stop offset="0" stop-color="#5e0b0d"/><stop offset="0.5" stop-color="#d4151b"/><stop offset="1" stop-color="#5e0b0d"/></linearGradient>
<linearGradient id="gIso2" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#1c1c20"/><stop offset="0.5" stop-color="#8c8c92"/><stop offset="1" stop-color="#1c1c20"/></linearGradient>
<radialGradient id="gDet2" cx="0.3" cy="0.4" r="0.8"><stop offset="0" stop-color="#ffd36b"/><stop offset="0.55" stop-color="#f08a24"/><stop offset="1" stop-color="#c92a12"/></radialGradient>
<radialGradient id="gDetG2" cx="0.3" cy="0.4" r="0.8"><stop offset="0" stop-color="#e6f58a"/><stop offset="0.55" stop-color="#6cc04a"/><stop offset="1" stop-color="#2c8a3a"/></radialGradient></defs>`);
    out.push(`<rect x="1.2" y="1.2" width="${wI - 2.4}" height="${HB - 2.4}" rx="5" fill="#fde5e2" stroke="#c0392b" stroke-width="2.4"/>`);
    out.push(`<rect x="${xD}" y="1.2" width="${wD}" height="${HB - 2.4}" rx="5" fill="#fff1dc" stroke="#e08a00" stroke-width="2.4"/>`);
    out.push(`<rect x="${xII}" y="1.2" width="${xE - xII}" height="${HB - 2.4}" rx="8" fill="#eef1f4" stroke="#8a949a" stroke-width="2.8"/>`);
    out.push(`<rect x="${xIII}" y="1.2" width="${wIII - 1.2}" height="${HB - 2.4}" rx="5" fill="#dde3f4" stroke="#27408b" stroke-width="2.4"/>`);
    const ty = 5 + fs; txt(8, ty, "(e)"); txt(8 + tw("(e)") + 5, ty, "I: Source"); txt(xD + 6, ty, "Dither"); txt(xII + 8, ty, tw("II: Gas sensor chip") + 6 < xE - 24 - 2 * ROUT - xII - 8 ? "II: Gas sensor chip" : "II: Chip"); txt(xIII + 8, ty, "III: Detection");
    // source: laser, isolator, reference pick-off; electronics drive the laser and the scanner
    const refY = HB - 12, BY = y0 + 10, BH = 36;
    txt(8, y0 - 12, "Laser");
    out.push(`<path d="M30 ${y0}L${xD + 15} ${y0}" stroke="${COL.beam}" stroke-width="3.2" fill="none"/>`);
    out.push(`<rect x="6" y="${y0 - 7}" width="22" height="14" rx="1.5" fill="#2b2a33"/><rect x="6" y="${y0 - 4}" width="22" height="8" fill="url(#gLas2)"/><rect x="28" y="${y0 - 5.5}" width="3.5" height="11" fill="#000"/>`);
    out.push(`<rect x="36" y="${y0 - 4.5}" width="12" height="9" fill="#c9cdd0" stroke="#8a9296" stroke-width="0.5"/><rect x="38.5" y="${y0 - 6}" width="7" height="12" fill="url(#gIso2)"/>`);
    const REF = ' stroke-dasharray="5 2.6"';   // reference beam: optical (beam colour), dashed to tell it from the signal path
    out.push(`<path d="M${xs_ - 3.5} ${y0 + 3.5}L${xs_ + 3.5} ${y0 - 3.5}" stroke="#5a6a6f" stroke-width="1.6"/>`);
    out.push(`<rect x="5" y="${BY}" width="${bwE}" height="${BH}" rx="5" fill="#fff" stroke="#222" stroke-width="2"/>`);
    txt(5 + bwE / 2, BY + BH / 2 - lh / 2 + 0.35 * fs, "Driver", { anchor: "middle" }); txt(5 + bwE / 2, BY + BH / 2 + lh / 2 + 0.35 * fs, "electronics", { anchor: "middle" });
    line(`M18 ${y0 + 7}L18 ${BY}`, COL.elec, 1.3, ELEC);
    line(`M${xs_} ${y0 + 3}L${xs_} ${refY}L${xE} ${refY}`, COL.beam, 2.2, REF);
    // dither block: scanning mirror (angle + position) and scan lens
    const xm = xD + 15, xl = xD + wD - 9;
    line(`M${5 + bwE} ${BY + BH - 6}L${n2(xm)} ${BY + BH - 6}L${n2(xm)} ${y0 + 9}`, COL.elec, 1.3, ELEC);
    out.push(`<rect x="${n2(xm - 2)}" y="${y0 - 9}" width="4" height="18" fill="#9aa3ad" stroke="#5a6470" stroke-width="0.6" transform="rotate(-35 ${n2(xm)} ${y0})"/>`);
    line(`M${n2(xm - 7)} ${y0 - 12}A9 9 0 0 1 ${n2(xm + 7)} ${y0 - 12}`, COL.mut, 1, ' marker-end="none"');
    out.push(`<path d="M${n2(xm + 7)} ${y0 - 12}l-2.6 -1.6l0.4 2.8z M${n2(xm - 7)} ${y0 - 12}l2.6 -1.6l-0.4 2.8z" fill="${COL.mut}"/>`);
    for (const [dyy, op] of [[-6, 0.5], [0, 1], [6, 0.5]]) out.push(`<path d="M${n2(xm)} ${y0}L${n2(xl)} ${y0 + dyy * 0.6}L${n2(onRing(PIN, ROUT)[0] - 11)} ${y0 + dyy * 0.35}" stroke="${COL.beam}" stroke-width="${op === 1 ? 3 : 1.6}" stroke-opacity="${op}" fill="none"/>`);
    out.push(`<path d="M${n2(xl)} ${y0 - 9}Q${n2(xl + 3.6)} ${y0} ${n2(xl)} ${y0 + 9}Q${n2(xl - 3.6)} ${y0} ${n2(xl)} ${y0 - 9}Z" fill="#e3e7ea" stroke="#8b9498" stroke-width="0.6"/>`);
    txt(xD + wD - 4, y0 + 15 + 0.35 * fs, "±θ, ±x", { anchor: "end", fill: COL.mut });
    // chip: grating coupler at the input port, the cell, output ports
    const Ein = onRing(PIN, ROUT), gx1 = Ein[0] - 2, gx0 = gx1 - 18;
    out.push(`<path d="M${n2(gx1)} ${y0 - 7}L${n2(Ein[0] + 1)} ${y0 - 4.4}L${n2(Ein[0] + 1)} ${y0 + 4.4}L${n2(gx1)} ${y0 + 7}Z" fill="#f6c9a3" fill-opacity="0.7"/>`);
    out.push(`<rect x="${n2(gx0)}" y="${y0 - 9}" width="${n2(gx1 - gx0)}" height="18" fill="#fde3cc" stroke="#b5561a" stroke-width="0.8"/>`);
    for (let x = gx0 + 2; x < gx1 - 1; x += 2.6) out.push(`<path d="M${n2(x)} ${y0 - 8}L${n2(x)} ${y0 + 8}" stroke="${COL.gc}" stroke-width="1.3"/>`);
    for (let i = 8; i >= 1; i--) out.push(`<path d="${poly(vert(RC + i * PITCH / 2))}" fill="${i % 2 === 0 ? "#3f6283" : "#d9dfe6"}" stroke-linejoin="round"/>`);
    out.push(`<path d="${poly(vert(ROUT))}" fill="none" stroke="#2f4d68" stroke-width="0.7" stroke-linejoin="round"/>`);
    out.push(`<g stroke="#eef1f4" stroke-width="1.1">${vert(RC).map((q, i) => { const a = rot + 2 * Math.PI * i / N - Math.PI / N; return `<path d="M${pt(q)}L${pt(onRing(a, ROUT))}"/>`; }).join("")}</g>`);
    out.push(`<path d="${poly(vert(RC))}" fill="#c9d6ea" stroke="#2b5b86" stroke-width="1.3" stroke-linejoin="round"/>`);
    const port = (a, fill, edge, WH = 4.4, wide = false) => {
      out.push(`<path d="${quad(a, RC - 0.5, ROUT + 0.7, WH, WH)}" fill="#c9d6ea"/>`);
      out.push(`<path d="M${pt(qp(a, RC, -WH))}L${pt(qp(a, ROUT + 0.7, -WH))}M${pt(qp(a, RC, WH))}L${pt(qp(a, ROUT + 0.7, WH))}" fill="none" stroke="#2f4d68" stroke-width="0.7" stroke-linecap="round"/>`);
      if (!wide) out.push(`<path d="${quad(a, RC - 0.3, ROUT + 1.2, 3.5, 1.3)}" fill="${fill}" stroke="${edge}" stroke-width="0.5" stroke-linejoin="round"/>`);
    };
    port(PIN, COL.gc, "#b5561a", 6.6, true); for (const a of angs) port(a, COL.blue, "#14296a");
    for (let j = 0; j < mc.nInc; j++) { const q = Tf(mc.cx[j], mc.cy[j]); out.push(`<circle cx="${n2(q[0])}" cy="${n2(q[1])}" r="${n2(Math.max(mc.r[j] * k, 0.9))}" fill="${COL.inc}"/>`); }
    bp.forEach((pts, i) => out.push(`<path d="M${pts.map(q => pt(Tf(q[0], q[1]))).join("L")}" fill="none" stroke="${COL.ray}" stroke-width="${p.rw}" stroke-opacity="${i ? 0.45 : 0.95}" stroke-linejoin="round" stroke-linecap="round"/>`));
    txt(CX, CY + 0.35 * fs, "CMPC", { b: 1, anchor: "middle", fill: "#1b3a5a", halo: "#c9d6ea" });
    const dyy = [32, 49, 74], xT = xE - 11;
    angs.forEach((a, i) => {
      const E = onRing(a, ROUT), y = dyy[i];
      line(`M${pt(E)}C${n2(E[0] + 14 * (1 + Math.cos(a)) / 2)} ${n2((y < E[1] ? Math.min : Math.max)(E[1] - 14 * Math.sin(a) / 2, E[1]))} ${n2(xT - 16)} ${n2(y)} ${n2(xT)} ${n2(y)}`, COL.blue, 3);
      out.push(`<path d="M${n2(xT)} ${n2(y)}L${n2(xE)} ${n2(y - 6)}L${n2(xE)} ${n2(y + 6)}Z" fill="${COL.blue}"/>`);
    });
    const det = (y, raw, green) => {
      const xb = xIII + 20;
      out.push(`<path d="M${xE} ${n2(y)}L${xE + 9} ${n2(y - 2.2)}L${xb - 8} ${n2(y - 2.2)}L${xb - 1} ${n2(y - 0.9)}L${xb - 1} ${n2(y + 0.9)}L${xb - 8} ${n2(y + 2.2)}L${xE + 9} ${n2(y + 2.2)}Z" fill="${COL.beam}"/>`);
      out.push(`<path d="M${xIII + 9} ${n2(y - 3.8)}Q${xIII + 11.2} ${n2(y)} ${xIII + 9} ${n2(y + 3.8)}Q${xIII + 6.8} ${n2(y)} ${xIII + 9} ${n2(y - 3.8)}Z" fill="#e3e7ea" stroke="#8b9498" stroke-width="0.6"/>`);
      out.push(`<path d="M${xb + 2.5} ${n2(y - 4.6)}A6.2 4.6 0 0 1 ${xb + 2.5} ${n2(y + 4.6)}Z" fill="url(#${green ? "gDetG2" : "gDet2"})" stroke="#6a3a1a" stroke-width="0.5"/><rect x="${xb - 0.5}" y="${n2(y - 5.4)}" width="2.8" height="10.8" fill="#000"/>`);
      txt(xb + 11, y + 0.35 * fs, "", { raw });
    };
    det(dyy[0], "Det. 1"); det(dyy[1], "Det. 2"); det(dyy[2], `Det. <tspan font-style="italic">N</tspan>`); det(refY, "Det. R", true);
    for (const d of [-3.6, 0, 3.6]) out.push(`<circle cx="${xIII + 20 + 11 + 8}" cy="${n2((dyy[1] + dyy[2]) / 2 + d)}" r="1.2" fill="${COL.ink}"/>`);
    const leg = [["grating coupler", COL.gc, 5, ""], ["output WGs", COL.blue, 3, ""], ["ref. beam", COL.beam, 2.2, REF], ["free-space beam", COL.beam, 3, ""], ["electrical", COL.elec, 1.3, ELEC]];
    if (p.inc) leg.push(["index inclusions", COL.inc, 0, "dot"]);
    let lx = 0, ly = HB + 4 + fs * 0.85;
    for (const [lab, col, w, ex] of leg) {
      const iw = 17 + tw(lab) + 11;
      if (lx > 0 && lx + iw - 11 > WE) { lx = 0; ly += 1.25 * fs; }
      if (ex === "dot") out.push(`<circle cx="${n2(lx + 7)}" cy="${n2(ly - 0.35 * fs)}" r="2.4" fill="${col}"/>`);
      else line(`M${n2(lx)} ${n2(ly - 0.35 * fs)}L${n2(lx + 14)} ${n2(ly - 0.35 * fs)}`, col, w, ex);
      txt(lx + 17, ly, lab); lx += iw;
    }
    return { body: out.join("\n"), h: Math.ceil(ly + 0.3 * fs + 2), nWalls: beams[0].nWalls, exit: beams[0].exit };
  }
'''
    s = rep(s, '\n  const COMP_W = 595, E_X = 220,', egc + '\n  const COMP_W = 595, E_X = 220,')
    s = rep(s, '    const eRes = eCompact(liveP, fs, COMP_W - E_X), COMP_CH = eRes.h,', '    const eRes = (S.indes === "gc" ? eCompactGC : eCompact)(liveP, fs, COMP_W - E_X), COMP_CH = eRes.h,')
    s = rep(s, '      const r = liveBuild(liveP);\n      liveRes = r;',
            '      const r = liveBuild(liveP);\n      liveRes = r;\n'
            '      if (liveView === "e" && S.indes === "gc") {    /* panel (e) alone, new design: the compact drawing at the full width */\n'
            '        const e = eCompactGC(liveP, r.fsU, LIVE_W);\n'
            '        liveRes = Object.assign({}, r, { svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${LIVE_W} ${e.h}" font-family=\'${LIVE_FONT}\' text-rendering="geometricPrecision" role="img" aria-label="System schematic, grating-coupler design: source, dither optics, gas sensor chip, detection"><rect width="${LIVE_W}" height="${e.h}" fill="#fff"/>${e.body}</svg>`, hMm: liveP.mm * e.h / LIVE_W });\n'
            '      }')
    s = rep(s, '      $("figNote").textContent = (liveView === "full" ? FIG_NOTE_FULL : FIG_NOTE_E) + tail',
            '      $("figNote").textContent = (liveView === "full" ? FIG_NOTE_FULL : FIG_NOTE_E) + (S.indes === "gc" ? FIG_NOTE_GC : "") + tail')
    s = rep(s, '  const liveSvgText = () =>',
            '  const FIG_NOTE_GC = " New input design (QUIET Task 11): (e) shows the dither system off the chip, a scanning mirror and scan lens sweeping the angle (±60 mrad) and position (±100 µm) of a free-space beam on a large-area grating coupler at the input port of the CMPC; the chip has no heater section. In (a) and (b) only the last part of the free-space beam above the input grating coupler is drawn (the dither states as parallel shifts with a small angle spread). The ray trace in (e) uses the design\'s 8 index inclusions of r 25 µm and shows the centre ray and an edge state of the input dither (+45 mrad, +100 µm).";\n'
            '  const liveSvgText = () =>')
    s = rep(s, '<title>Fig. 1 MPC sensor schematics</title>', '<title>Fig. 1 MPC sensor schematics</title>')
    (ART / "fig1_v32.html").write_text(s, encoding="utf-8")
    print("ok", len(s))


if __name__ == "__main__":
    main()
