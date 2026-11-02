/* engine:begin  JS ports of math-engines/engines/gas_cell (GasEngine) and planar_cell (Cell2D with smooth profile mirrors,
   builders, tracing) with ray_phase (dither analysis); checked by math-engines/tools/check_js_ports.mjs */
const PlanarEngine = (() => {
  /* planar_cell.Cell2D: elements [{A: [x, y], B: [x, y], kappa, ext, label}] in counter-clockwise order */
  /* planar_cell.Profile: smooth profile mirror ξ = S(η) in its own frame (vertex V, axis u into the cell, e = (−u_y, u_x)) */
  const PROFILE_TABLE = 129;
  /* ripple ra sin(q η + rp), q = 2π / rl (planar_cell.Profile ra, rl, rp) */
  const prQ = p => p.ra && p.rl ? 2 * Math.PI / p.rl : 0;
  const prS = (p, eta) => { const o = p.c * eta * eta / (1 + Math.sqrt(Math.max(1 - (1 + p.k) * p.c * p.c * eta * eta, 0))) + p.a3 * eta ** 3 + p.a4 * eta ** 4, q = prQ(p); return q ? o + p.ra * Math.sin(q * eta + p.rp) : o; };
  const prdS = (p, eta) => { const o = p.c * eta / Math.sqrt(Math.max(1 - (1 + p.k) * p.c * p.c * eta * eta, 1e-300)) + 3 * p.a3 * eta ** 2 + 4 * p.a4 * eta ** 3, q = prQ(p); return q ? o + p.ra * q * Math.cos(q * eta + p.rp) : o; };
  const prd2S = (p, eta) => { const o = p.c / Math.max(1 - (1 + p.k) * p.c * p.c * eta * eta, 1e-300) ** 1.5 + 6 * p.a3 * eta + 12 * p.a4 * eta ** 2, q = prQ(p); return q ? o - p.ra * q * q * Math.sin(q * eta + p.rp) : o; };
  const prPoint = (p, eta) => { const S = prS(p, eta), ux = p.axis[0], uy = p.axis[1]; return [p.V[0] + S * ux - eta * uy, p.V[1] + S * uy + eta * ux]; };
  const prCurv = (p, eta) => prd2S(p, eta) / (1 + prdS(p, eta) ** 2) ** 1.5;
  function makeProfile({ V, axis, c, k = 0, a3 = 0, a4 = 0, eta0 = 0, eta1 = 0, ra = 0, rl = 0, rp = 0 }) { return { V, axis, c, k, a3, a4, eta0, eta1, ra, rl, rp }; }
  function interpInc(x, xs, ys) {                     /* numpy.interp for increasing xs */
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    return ys[lo] + (x - xs[lo]) * (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]);
  }

  function makeCell(els, opts = {}) {
    const E = els.length;
    const c = { els, E, name: opts.name || "cell", meta: Object.assign({}, opts.meta || {}),
      Ax: new Float64Array(E), Ay: new Float64Array(E), Bx: new Float64Array(E), By: new Float64Array(E),
      kappa: new Float64Array(E), ext: new Float64Array(E), labels: els.map(e => e.label || ""),
      h: new Float64Array(E), tx: new Float64Array(E), ty: new Float64Array(E), nx: new Float64Array(E), ny: new Float64Array(E),
      Mx: new Float64Array(E), My: new Float64Array(E), R: new Float64Array(E).fill(Infinity), Cx: new Float64Array(E).fill(NaN),
      Cy: new Float64Array(E).fill(NaN), L: new Float64Array(E), sense: new Float64Array(E).fill(1), s0: new Float64Array(E),
      prof: els.map(e => e.profile || null), ptab: new Array(E).fill(null) };
    for (let k = 0; k < E; k++) {
      const e = els[k];
      c.Ax[k] = e.A[0]; c.Ay[k] = e.A[1]; c.Bx[k] = e.B[0]; c.By[k] = e.B[1]; c.kappa[k] = e.kappa || 0; c.ext[k] = e.ext || 0;
      const chx = c.Bx[k] - c.Ax[k], chy = c.By[k] - c.Ay[k];
      c.h[k] = 0.5 * Math.hypot(chx, chy);
      c.tx[k] = chx / (2 * c.h[k]); c.ty[k] = chy / (2 * c.h[k]);
      c.nx[k] = c.ty[k]; c.ny[k] = -c.tx[k];
      c.Mx[k] = 0.5 * (c.Ax[k] + c.Bx[k]); c.My[k] = 0.5 * (c.Ay[k] + c.By[k]);
      c.L[k] = 2 * c.h[k];
      const pr = c.prof[k];
      if (pr) {
        const n = PROFILE_TABLE, eta = new Float64Array(n), sl = new Float64Array(n), step = (pr.eta1 - pr.eta0) / (n - 1);
        for (let i = 0; i < n; i++) eta[i] = i === n - 1 ? pr.eta1 : pr.eta0 + i * step;
        let dPrev = Math.sqrt(1 + prdS(pr, eta[0]) ** 2);
        for (let i = 1; i < n; i++) { const d = Math.sqrt(1 + prdS(pr, eta[i]) ** 2); sl[i] = sl[i - 1] + 0.5 * (d + dPrev) * Math.abs(eta[i] - eta[i - 1]); dPrev = d; }
        const inc = eta[n - 1] > eta[0];
        c.ptab[k] = { eta, sl, etaInc: inc ? eta : Float64Array.from(eta, v => -v), sign: inc ? 1 : -1 };
        c.L[k] = sl[n - 1];
        continue;
      }
      if (c.kappa[k] !== 0) {
        const R = 1 / Math.abs(c.kappa[k]);
        if (R < c.h[k] * (1 - 1e-12)) throw new Error(`element ${k}: radius smaller than half its chord`);
        const d0 = Math.sqrt(Math.max(R * R - c.h[k] ** 2, 0));
        c.R[k] = R;
        const sg = c.kappa[k] < 0 ? -1 : 1;
        c.Cx[k] = c.Mx[k] + sg * d0 * c.nx[k]; c.Cy[k] = c.My[k] + sg * d0 * c.ny[k];
        c.L[k] = 2 * R * Math.asin(Math.min(c.h[k] / R, 1));
        const vAx = c.Ax[k] - c.Cx[k], vAy = c.Ay[k] - c.Cy[k], vBx = c.Bx[k] - c.Cx[k], vBy = c.By[k] - c.Cy[k];
        c.sense[k] = Math.sign(vAx * vBy - vAy * vBx) || 1;
      }
    }
    let acc = 0;
    for (let k = 0; k < E; k++) { c.s0[k] = acc; acc += c.L[k]; }
    c.perimeter = acc;
    let a = 0, gapMax = 0;
    for (let k = 0; k < E; k++) {
      const j = (k + 1) % E;
      a += 0.5 * (c.Ax[k] * c.By[k] - c.Bx[k] * c.Ay[k]);
      gapMax = Math.max(gapMax, Math.hypot(c.Ax[j] - c.Bx[k], c.Ay[j] - c.By[k]));
    }
    for (let k = 0; k < E; k++) { const j = (k + 1) % E; a += 0.5 * (c.Bx[k] * c.Ay[j] - c.Ax[j] * c.By[k]); }
    for (let k = 0; k < E; k++) {
      if (!c.prof[k]) continue;
      const eta = c.ptab[k].eta, pts = Array.from(eta, v => prPoint(c.prof[k], v));
      for (let i = 0; i + 1 < pts.length; i++) a += 0.5 * (pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]);
      a -= 0.5 * (c.Ax[k] * c.By[k] - c.Bx[k] * c.Ay[k]);
    }
    for (let k = 0; k < E; k++) {
      if (c.kappa[k] === 0) continue;
      const R = c.R[k], al = Math.asin(Math.min(c.h[k] / R, 1)), seg = R * R * (al - Math.sin(al) * Math.cos(al));
      a += c.kappa[k] < 0 ? seg : -seg;
    }
    c.area = a;
    c.closed = gapMax <= 1e-6 * c.perimeter;
    c.meanChord = Math.PI * a / c.perimeter;
    c.sIn = opts.sIn === undefined || opts.sIn === null ? c.L[0] / 2 : opts.sIn;
    c.sOut = opts.sOut === undefined ? null : opts.sOut;
    return c;
  }

  function arcElements(cx, cy, r, phi0, phi1, n, faceted, ext, label) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = i === 0 ? phi0 : phi0 + i * ((phi1 - phi0) / n), b = i + 1 === n ? phi1 : phi0 + (i + 1) * ((phi1 - phi0) / n);
      out.push({ A: [cx + r * Math.cos(a), cy + r * Math.sin(a)], B: [cx + r * Math.cos(b), cy + r * Math.sin(b)],
        kappa: faceted ? 0 : -1 / r, ext: faceted ? ext : 0, label });
    }
    return out;
  }

  function polygonCell(r, N, ext = 0.15) {
    const els = [];
    for (let k = 0; k < N; k++) {
      const a = 2 * Math.PI * k / N - Math.PI / N, b = 2 * Math.PI * k / N + Math.PI / N;
      els.push({ A: [r * Math.cos(a), r * Math.sin(a)], B: [r * Math.cos(b), r * Math.sin(b)], kappa: 0, ext, label: `facet${k}` });
    }
    return makeCell(els, { name: `polygon N=${N}` });
  }

  function circleCell(r, nArcs = 4) {
    return makeCell(arcElements(0, 0, r, -Math.PI / nArcs, 2 * Math.PI - Math.PI / nArcs, nArcs, false, 0, "circle"), { name: "circle" });
  }

  function stadiumCell(r, a, capFacets = null, straightSegments = 1, ext = 0.15) {
    const ns = straightSegments, els = [];
    const xs = Array.from({ length: ns + 1 }, (_, i) => i === ns ? a / 2 : -a / 2 + i * (a / ns));
    const segExt = ns > 1 || capFacets ? ext : 0;
    if (a > 0) for (let i = 0; i < ns; i++) els.push({ A: [xs[i], -r], B: [xs[i + 1], -r], kappa: 0, ext: segExt, label: "straight_bottom" });
    const faceted = capFacets !== null && capFacets !== undefined, npc = faceted ? capFacets : 2;
    els.push(...arcElements(a / 2, 0, r, -Math.PI / 2, Math.PI / 2, npc, faceted, ext, "cap_right"));
    if (a > 0) for (let i = 0; i < ns; i++) els.push({ A: [xs[ns - i], r], B: [xs[ns - i - 1], r], kappa: 0, ext: segExt, label: "straight_top" });
    els.push(...arcElements(-a / 2, 0, r, Math.PI / 2, 3 * Math.PI / 2, npc, faceted, ext, "cap_left"));
    return makeCell(els, { name: "stadium", sIn: a > 0 ? a / 2 : null });
  }

  /* planar_cell integrated Herriott design helpers */
  const thetaOf = (R, d) => { const c = 1 - d / R; return c > -1 && c < 1 ? Math.acos(c) : NaN; };
  const reentrantSpacing = (R, N, M) => R * (1 - Math.cos(2 * Math.PI * M / N));
  const modeRadius = (R, d, lam, n = 1) => { const g = 1 - d / R; return Math.sqrt(lam / n * d / Math.PI / Math.sqrt(1 - g * g)); };

  function herriottPlanarCell(R, N, M, A, { portW = null, aperture = null, R2 = null, d = null, phase = 0 } = {}) {
    if (N < 4 || N % 2) throw new Error("N must be even and >= 4");
    if (!(M > 0 && M < N / 2)) throw new Error("need 0 < M < N/2");
    d = d === null ? reentrantSpacing(R, N, M) : d;
    R2 = R2 === null ? R : R2;
    const w = portW === null ? A / 10 : portW, a = aperture === null ? A + w : aperture, y0 = A * Math.cos(phase);
    if (a >= Math.min(R, R2)) throw new Error("aperture must be smaller than the mirror radii");
    if (a < Math.max(A, Math.abs(y0) + w / 2)) throw new Error("aperture must hold the spot pattern and the window");
    const p1 = Math.asin(a / R), p2 = Math.asin(a / R2);
    const els = [...arcElements(-d / 2 + R, 0, R, Math.PI - p1, Math.PI + p1, 1, false, 0, "M1"), ...arcElements(d / 2 - R2, 0, R2, -p2, p2, 1, false, 0, "M2")];
    const sIn = R * (p1 - Math.asin(y0 / R));
    const thLaunch = phase ? Math.atan(-A * Math.sin(phase) * Math.sqrt((2 * R - d) / d) / R) : 0;
    return makeCell(els, { name: "integrated Herriott", sIn, sOut: sIn,
      meta: { R, R2, d, N, M, A, aperture: a, portW: w, theta: thetaOf(R, d), phase, y0, thetaLaunch: thLaunch } });
  }

  /* planar_cell.mulberry32 / mulberry_normals (identical streams in Python) */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function mulberryNormals(seed, n) {
    const u = mulberry32(seed), out = [];
    for (let i = 0; i < n; i++) { const u1 = u(), u2 = u(); out.push(Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2)); }
    return out;
  }
  /* planar_cell._segment_mirror: n arcs with their ends on the profile, curvature segCurv × the profile's at mid-segment */
  function segmentMirror(p, n, segCurv, label) {
    const step = (p.eta1 - p.eta0) / n, eta = Array.from({ length: n + 1 }, (_, i) => i === n ? p.eta1 : p.eta0 + i * step), els = [];
    for (let i = 0; i < n; i++) els.push({ A: prPoint(p, eta[i]), B: prPoint(p, eta[i + 1]), kappa: -segCurv * prCurv(p, 0.5 * (eta[i] + eta[i + 1])), ext: 0, label: `${label} seg${i}` });
    return els;
  }
  /* planar_cell.deformed_herriott_cell: smooth profile mirrors (conic + cubic/quartic deformation), optional side walls,
     optional curved segmentation with tilt jitter */
  function deformedHerriottCell(R, N, M, A, { portW = null, aperture = null, R2 = null, d = null, phase = 0, conic = 0, eps3 = 0, eps4 = 0, deform = "both", walls = false,
    nSeg = 0, segCurv = 1, segTilt = 0, segSeed = 1, rippleA = 0, ripplePeriod = 0.5e-3, ripplePhase = 0, rippleOn = "both" } = {}) {
    if (N < 4 || N % 2) throw new Error("N must be even and >= 4");
    if (!(M > 0 && M < N / 2)) throw new Error("need 0 < M < N/2");
    d = d === null ? reentrantSpacing(R, N, M) : d;
    R2 = R2 === null ? R : R2;
    const w = portW === null ? A / 10 : portW, a = aperture === null ? A + w : aperture, y0 = A * Math.cos(phase);
    if (a >= Math.min(R, R2)) throw new Error("aperture must be smaller than the mirror radii");
    if (a < Math.max(A, Math.abs(y0) + w / 2)) throw new Error("aperture must hold the spot pattern and the window");
    const profs = [[R, [-d / 2, 0], [1, 0]], [R2, [d / 2, 0], [-1, 0]]].map(([Rm, V, ax], i) => {
      const on = deform === "both" || deform === `M${i + 1}`, c = 1 / Rm;
      const p = makeProfile({ V, axis: ax, c, k: conic, a3: on ? eps3 * c / (2 * a) : 0, a4: on ? eps4 * c / (2 * a * a) : 0, eta0: a, eta1: -a });
      if (1 - (1 + conic) * c * c * a * a <= 0) throw new Error("the conic mirror is not defined out to the aperture");
      for (let i = 0; i < 257; i++) if (prd2S(p, -a + 2 * a * i / 256) <= 0) throw new Error("the deformation makes a mirror convex somewhere: reduce eps3 / eps4");
      if (rippleA && (rippleOn === "both" || rippleOn === `M${i + 1}`)) {
        if (nSeg) throw new Error("ripple applies to smooth mirrors (n_seg = 0)");
        if (!(ripplePeriod > 0)) throw new Error("ripple_period must be positive");
        p.ra = rippleA; p.rl = ripplePeriod; p.rp = ripplePhase;
      }
      return p;
    });
    let A1 = prPoint(profs[0], a), B1 = prPoint(profs[0], -a), A2 = prPoint(profs[1], a), B2 = prPoint(profs[1], -a), m1, m2;
    if (nSeg) {
      if (segCurv < 0) throw new Error("seg_curv must be >= 0 (0: flat facets)");
      m1 = segmentMirror(profs[0], nSeg, segCurv, "M1"); m2 = segmentMirror(profs[1], nSeg, segCurv, "M2");
      if (segTilt) {
        const tl = mulberryNormals(segSeed, 2 * nSeg).map(v => segTilt * v);
        m1.concat(m2).forEach((e, j) => {
          const Mx = 0.5 * (e.A[0] + e.B[0]), My = 0.5 * (e.A[1] + e.B[1]), co = Math.cos(tl[j]), si = Math.sin(tl[j]);
          const rot = P => [Mx + co * (P[0] - Mx) - si * (P[1] - My), My + si * (P[0] - Mx) + co * (P[1] - My)];
          e.A = rot(e.A); e.B = rot(e.B); e.ext = 0.02;
        });
      }
      A1 = m1[0].A; B1 = m1[nSeg - 1].B; A2 = m2[0].A; B2 = m2[nSeg - 1].B;
    } else {
      m1 = [{ A: A1, B: B1, kappa: 0, ext: 0, label: "M1", profile: profs[0] }]; m2 = [{ A: A2, B: B2, kappa: 0, ext: 0, label: "M2", profile: profs[1] }];
    }
    const els = m1.slice();
    if (walls) els.push({ A: B1, B: A2, kappa: 0, ext: 0, label: "side_bottom" });
    els.push(...m2);
    if (walls) els.push({ A: B2, B: A1, kappa: 0, ext: 0, label: "side_top" });
    const thLaunch = phase ? Math.atan(-A * Math.sin(phase) * Math.sqrt((2 * R - d) / d) / R) : 0;
    const meta = { R, R2, d, N, M, A, aperture: a, portW: w, theta: thetaOf(R, d), phase, y0, thetaLaunch: thLaunch, conic, eps3, eps4, deform, walls: !!walls,
      nSeg, segCurv, segTilt, segSeed, rippleA, ripplePeriod, ripplePhase, rippleOn, kRipple: rippleA ? 8 * Math.PI ** 2 * rippleA * d / ripplePeriod ** 2 : 0 };
    const c = makeCell(els, { name: "deformed Herriott", meta });
    if (nSeg) {
      let lo = 0, hi = c.s0[nSeg - 1] + c.L[nSeg - 1];
      for (let i = 0; i < 80; i++) { const mid = 0.5 * (lo + hi); if (pointAt(c, mid)[1] > y0) lo = mid; else hi = mid; }
      c.sIn = c.sOut = 0.5 * (lo + hi);
    } else c.sIn = c.sOut = slocOf(c, 0, y0);
    return c;
  }

  /* planar_cell.herriott_planar_spots: design spots per mirror, closest spots, window clearance */
  function herriottSpots(m) {
    const y1 = [], y2 = [];
    for (let n = 1; n < m.N; n++) (n % 2 ? y2 : y1).push(m.A * Math.cos(n * m.theta + m.phase));
    const minSep = arr => { let s = Infinity; for (let i = 0; i < arr.length; i++) for (let j = i + 1; j < arr.length; j++) s = Math.min(s, Math.abs(arr[i] - arr[j])); return s; };
    let clear = Infinity; for (const y of y1) clear = Math.min(clear, Math.abs(y - m.y0));
    return { y1, y2, spacing: Math.min(minSep(y1.concat([m.y0])), minSep(y2)), clearance: clear - m.portW / 2 };
  }

  /* planar_cell.perturb with explicit per-element arrays (null = 0): tilt about the chord midpoint, shift along the
     outward normal, curvature change; ports and meta kept */
  function perturb(c, tilts, offsets, curvatures) {
    const els = c.els.map((e, k) => {
      const tl = tilts ? tilts[k] : 0, of = offsets ? offsets[k] : 0, cu = curvatures ? curvatures[k] : 0;
      const M0x = c.Mx[k], M0y = c.My[k], Mx = M0x + of * c.nx[k], My = M0y + of * c.ny[k];
      const co = Math.cos(tl), si = Math.sin(tl);
      const rot = P => [Mx + co * (P[0] - M0x) - si * (P[1] - M0y), My + si * (P[0] - M0x) + co * (P[1] - M0y)];
      if (e.profile) {
        const p = e.profile, ax = [co * p.axis[0] - si * p.axis[1], si * p.axis[0] + co * p.axis[1]];
        const pn = makeProfile({ V: rot(p.V), axis: ax, c: p.c + cu, k: p.k, a3: p.a3, a4: p.a4, eta0: p.eta0, eta1: p.eta1, ra: p.ra, rl: p.rl, rp: p.rp });
        return { A: prPoint(pn, p.eta0), B: prPoint(pn, p.eta1), kappa: 0, ext: e.ext || 0, label: e.label, profile: pn };
      }
      return { A: rot(e.A), B: rot(e.B), kappa: (e.kappa || 0) + cu, ext: e.ext || 0, label: e.label };
    });
    return makeCell(els, { name: c.name + " (perturbed)", sIn: c.sIn, sOut: c.sOut, meta: c.meta });
  }

  const modP = (s, P) => ((s % P) + P) % P;
  function segmentOf(c, s) {
    s = modP(s, c.perimeter);
    let k = 0; while (k + 1 < c.E && c.s0[k + 1] <= s) k++;
    return k;
  }

  const etaOf = (c, k, sloc) => interpInc(sloc, c.ptab[k].sl, c.ptab[k].eta);
  const slocOf = (c, k, eta) => interpInc(c.ptab[k].sign * eta, c.ptab[k].etaInc, c.ptab[k].sl);
  function profileNormal(p, eta) { const d = prdS(p, eta), nn = Math.sqrt(1 + d * d), ux = p.axis[0], uy = p.axis[1]; return [(-ux - d * uy) / nn, (-uy + d * ux) / nn]; }
  /* planar_cell.Cell2D._profile_hit: t (Infinity: none) and η */
  function profileHit(c, k, px, py, dx, dy, eps) {
    const p = c.prof[k], ux = p.axis[0], uy = p.axis[1], ex = -uy, ey = ux;
    const xi0 = (px - p.V[0]) * ux + (py - p.V[1]) * uy, et0 = (px - p.V[0]) * ex + (py - p.V[1]) * ey;
    const dxi = dx * ux + dy * uy, det = dx * ex + dy * ey;
    let lo = Math.min(p.eta0, p.eta1), hi = Math.max(p.eta0, p.eta1);
    const pad = c.ext[k] * (hi - lo) / 2; lo -= pad; hi += pad;
    const tmax = det > 0 ? (hi - et0) / det : det < 0 ? (lo - et0) / det : Infinity;
    const tlin = dxi < 0 ? (prS(p, et0) - xi0) / dxi : Infinity;
    let t = Number.isFinite(tmax) ? tmax : tlin;
    const g = tt => xi0 + tt * dxi - prS(p, et0 + tt * det);
    if (prQ(p)) return rippledHit(p, g, t, tmax, xi0, dxi, et0, det, eps, lo, hi);
    if (!(Number.isFinite(t) && t > eps && g(t) <= 0)) return { t: Infinity, eta: et0 };
    const scale = 2 * c.h[k] + 1e-300;
    for (let it = 0; it < 80; it++) {
      const gp = dxi - prdS(p, et0 + t * det) * det, step = gp < 0 ? g(t) / gp : 0;
      t -= step;
      if (Math.abs(step) <= 1e-15 * scale) break;
    }
    return t > eps ? { t, eta: et0 + t * det } : { t: Infinity, eta: et0 };
  }
  /* planar_cell.Profile.sag_band: (S_min, S_max) over [lo, hi] from 257 samples, widened by 2 |ripple| + 1e-3 of the range */
  function sagBand(p, lo, hi) {
    const B = p._band;
    if (!B || B.lo !== lo || B.hi !== hi || B.ra !== p.ra || B.rl !== p.rl || B.rp !== p.rp) {
      let a = Infinity, b = -Infinity;
      for (let i = 0; i < 257; i++) { const v = prS(p, lo + (hi - lo) * i / 256); if (v < a) a = v; if (v > b) b = v; }
      const m = 2 * Math.abs(p.ra) + 1e-3 * (b - a);
      p._band = { lo, hi, ra: p.ra, rl: p.rl, rp: p.rp, v: [a - m, b + m] };
    }
    return p._band.v;
  }
  /* planar_cell.Cell2D._rippled_hit: within the sag band, sample g every Λ/8 of η, bracket the first drop to ≤ 0, safeguarded Newton */
  function rippledHit(p, g, t, tmax, xi0, dxi, et0, det, eps, eLo, eHi) {
    if (!(Number.isFinite(t) && t > eps)) return { t: Infinity, eta: et0 };
    if (!Number.isFinite(tmax)) return { t, eta: et0 + t * det };
    const [smin, smax] = sagBand(p, eLo, eHi);
    let ta = dxi < 0 ? Math.max((smax - xi0) / dxi, 0) : 0;
    let tb = dxi < 0 ? Math.min((smin - xi0) / dxi, t) : dxi > 0 ? Math.min((smax - xi0) / dxi, t) : t;
    ta = Math.min(ta, t);
    if (!(tb > ta)) return { t: Infinity, eta: et0 };
    const ns = Math.ceil(Math.abs((tb - ta) * det) / (p.rl / 8)) + 2;
    let lo = 0, hi = 0, found = false;
    for (let j = 1; j <= ns; j++) {
      const tt = ta + (tb - ta) * j / ns;
      if (g(tt) <= 0) { hi = tt; lo = j > 1 ? ta + (tb - ta) * (j - 1) / ns : ta; found = true; break; }
    }
    if (!found) return { t: Infinity, eta: et0 };
    let x = 0.5 * (lo + hi);
    for (let it = 0; it < 100; it++) {
      const et = et0 + x * det, gx = xi0 + x * dxi - prS(p, et), gd = dxi - prdS(p, et) * det;
      if (gx > 0) lo = x; else hi = x;
      let xn = x - gx / gd;
      if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
      const tol = 1e-13 * hi, stop = Math.abs(xn - x) <= tol || hi - lo <= tol || gx === 0;
      if (gx !== 0) x = xn;
      if (stop) break;
    }
    return x > eps ? { t: x, eta: et0 + x * det } : { t: Infinity, eta: et0 };
  }
  function curvatureAt(c, s) {
    s = modP(s, c.perimeter);
    const k = segmentOf(c, s);
    if (c.prof[k]) return prCurv(c.prof[k], etaOf(c, k, s - c.s0[k]));
    return c.kappa[k] !== 0 ? -c.kappa[k] : 0;
  }

  function pointAt(c, s) {
    s = modP(s, c.perimeter);
    const k = segmentOf(c, s), loc = s - c.s0[k];
    if (c.prof[k]) { const eta = etaOf(c, k, loc), P = prPoint(c.prof[k], eta), n = profileNormal(c.prof[k], eta); return [P[0], P[1], n[0], n[1]]; }
    if (c.kappa[k] === 0) {
      const f = loc / c.L[k];
      return [c.Ax[k] + f * (c.Bx[k] - c.Ax[k]), c.Ay[k] + f * (c.By[k] - c.Ay[k]), c.nx[k], c.ny[k]];
    }
    const ang = loc / c.R[k] * c.sense[k], vx = c.Ax[k] - c.Cx[k], vy = c.Ay[k] - c.Cy[k];
    const co = Math.cos(ang), sn = Math.sin(ang), px = vx * co - vy * sn, py = vx * sn + vy * co, sg = c.kappa[k] < 0 ? 1 : -1;
    return [c.Cx[k] + px, c.Cy[k] + py, sg * px / c.R[k], sg * py / c.R[k]];
  }

  /* planar_cell.Cell2D.hit_k */
  function hit(c, px, py, dx, dy) {
    const eps = 1e-12 * c.perimeter;
    let bt = Infinity, bnx = 0, bny = 0, bs = 0, bk = -1;
    for (let k = 0; k < c.E; k++) {
      if (c.prof[k]) {
        const r = profileHit(c, k, px, py, dx, dy, eps);
        if (Number.isFinite(r.t) && r.t < bt) { const n = profileNormal(c.prof[k], r.eta); bt = r.t; bnx = n[0]; bny = n[1]; bs = c.s0[k] + slocOf(c, k, r.eta); bk = k; }
        continue;
      }
      const hh = c.h[k] * (1 + c.ext[k]), Mx = c.Mx[k], My = c.My[k], tx = c.tx[k], ty = c.ty[k], nkx = c.nx[k], nky = c.ny[k];
      let t, ok, nx_, ny_, sl;
      if (c.kappa[k] === 0) {
        const dn = dx * nkx + dy * nky;
        t = ((Mx - px) * nkx + (My - py) * nky) / dn;
        const u = (px + t * dx - Mx) * tx + (py + t * dy - My) * ty;
        ok = dn > 1e-12 && t > eps && Math.abs(u) <= hh;
        nx_ = nkx; ny_ = nky; sl = Math.min(Math.max(u, -c.h[k]), c.h[k]) + c.h[k];
      } else {
        const R = c.R[k], cx = c.Cx[k], cy = c.Cy[k], ox = px - cx, oy = py - cy;
        const b = ox * dx + oy * dy, disc = b * b - (ox * ox + oy * oy - R * R), sq = Math.sqrt(Math.max(disc, 0));
        t = c.kappa[k] < 0 ? -b + sq : -b - sq;
        const hx = px + t * dx, hy = py + t * dy;
        const side = ((hx - cx) * nkx + (hy - cy) * nky) * (-Math.sign(c.kappa[k])) > 0;
        const u = (hx - Mx) * tx + (hy - My) * ty;
        ok = disc >= 0 && t > eps && side && Math.abs(u) <= hh;
        if (!ok) continue;
        const sg = c.kappa[k] < 0 ? 1 : -1;
        nx_ = sg * (hx - cx) / R; ny_ = sg * (hy - cy) / R;
        const vAx = c.Ax[k] - cx, vAy = c.Ay[k] - cy, vHx = hx - cx, vHy = hy - cy;
        const ang = Math.atan2(vAx * vHy - vAy * vHx, vAx * vHx + vAy * vHy) * c.sense[k];
        sl = Math.min(Math.max(R * ang, 0), c.L[k]);
      }
      if (ok && t < bt) { bt = t; bnx = nx_; bny = ny_; bs = c.s0[k] + sl; bk = k; }
    }
    return { t: bt, nx: bnx, ny: bny, s: bs, k: bk };
  }

  function reflect(dx, dy, nx, ny) {
    const dn = dx * nx + dy * ny;
    const rx = dx - 2 * dn * nx, ry = dy - 2 * dn * ny, nn = Math.hypot(rx, ry);
    return [rx / nn, ry / nn];
  }

  function launch(c, s, theta) {
    const [x, y, nx, ny] = pointAt(c, s);
    const ux = -nx, uy = -ny;
    return [x, y, ux * Math.cos(theta) - uy * Math.sin(theta), ux * Math.sin(theta) + uy * Math.cos(theta)];
  }

  function inWindow(s, s0, w, P) { const d = modP(s - s0 + P / 2, P) - P / 2; return Math.abs(d) <= w / 2; }

  /* trace up to nHits; ports = {sIn, sOut, w} or null (output window checked first, as trace_rays).
     exit: 1 output, 0 input, -1 still inside, -2 left through a gap or past an open wall (dir: the leaving direction) */
  function trace(c, start, nHits, ports) {
    let [x, y, dx, dy] = start;
    const xs = [x], ys = [y], S = [], SC = [], K = [];
    let L = 0, exit = -1;
    for (let j = 0; j < nHits; j++) {
      const h = hit(c, x, y, dx, dy);
      if (!Number.isFinite(h.t)) { exit = -2; break; }
      x += h.t * dx; y += h.t * dy; L += h.t;
      xs.push(x); ys.push(y); S.push(h.s); SC.push(dx * h.ny - dy * h.nx); K.push(h.k);
      if (ports) {
        if (inWindow(h.s, ports.sOut, ports.w, c.perimeter)) { exit = 1; break; }
        if (inWindow(h.s, ports.sIn, ports.w, c.perimeter)) { exit = 0; break; }
      }
      [dx, dy] = reflect(dx, dy, h.nx, h.ny);
    }
    return { xs, ys, s: S, sinchi: SC, elem: K, L, exit, dir: [dx, dy] };
  }

  /* planar_cell.herriott_planar_trace: the centre ray from the window until it leaves */
  function herriottTrace(c, dtheta = 0, nMax = 2000) {
    const m = c.meta, theta = m.thetaLaunch + dtheta, w = m.portW;
    const tr = trace(c, launch(c, c.sIn, theta), nMax, { sIn: c.sIn, sOut: c.sIn, w });
    const n = tr.s.length, out = { tr, nHits: n, path: tr.L, theta };
    if (tr.exit === 1) {
      out.exit = "port"; out.reflections = n - 1;
      const P = c.perimeter; out.exitOffset = modP(tr.s[n - 1] - c.sIn + P / 2, P) - P / 2;
      out.exitAngle = Math.asin(Math.min(Math.max(tr.sinchi[n - 1], -1), 1)) - theta;
    } else { out.exit = tr.exit === -2 ? "leak" : "trapped"; out.reflections = n; }
    return out;
  }

  function reflectionWeightedPath(chords, R) {
    let I = 1, L = 0, Leff = 0;
    for (let j = 0; j < chords.length; j++) { const c = chords[j]; L += c; Leff += I * c; I *= Array.isArray(R) ? R[j] : R; }
    return { L, Leff, Iend: I };
  }

  /* ray_phase.path_lengths: cumulative path per hit and a path id (hash of the element sequence, -1 once gone) */
  const HASH_M = 2147483647;
  function pathLengths(c, thetas, nPass, sIn, withIds) {
    const Ls = [], IDs = [];
    for (const th of thetas) {
      let [x, y, dx, dy] = launch(c, sIn === undefined ? c.sIn : sIn, th);
      const L = new Float64Array(nPass), ids = new Float64Array(nPass);
      let acc = 0, pid = 0;
      for (let j = 0; j < nPass; j++) {
        if (pid < 0) { L[j] = NaN; ids[j] = -1; continue; }
        const h = hit(c, x, y, dx, dy);
        if (!Number.isFinite(h.t)) { pid = -1; L[j] = NaN; ids[j] = -1; continue; }
        pid = (pid * 1000003 + h.k + 1) % HASH_M; ids[j] = pid;
        acc += h.t; L[j] = acc;
        x += h.t * dx; y += h.t * dy;
        [dx, dy] = reflect(dx, dy, h.nx, h.ny);
      }
      Ls.push(L); IDs.push(ids);
    }
    return withIds ? { L: Ls, ids: IDs } : Ls;
  }

  function ditherOffsets(A, n, waveform) {
    const d = new Float64Array(n);
    for (let j = 0; j < n; j++) { const t = (j + 0.5) / n; d[j] = waveform === "sine" ? A * Math.sin(2 * Math.PI * t) : A * (2 * t - 1); }
    return d;
  }

  function visibility(dphi) {
    let re = 0, im = 0, n = 0;
    for (const v of dphi) if (Number.isFinite(v)) { re += Math.cos(v); im += Math.sin(v); n++; }
    return n ? Math.hypot(re, im) / n : NaN;
  }

  const X_V09 = { sine: 0.6406308771588489, triangle: 0.786683072049212 };
  const X_V05 = { sine: 1.521144057668765, triangle: 1.8954942670340231 };
  const X_V0 = { sine: 2.404825557695807, triangle: Math.PI };
  function modelVisibility(x, waveform) {
    if (!Number.isFinite(x)) return NaN;
    if (waveform !== "sine") return x === 0 ? 1 : Math.abs(Math.sin(x) / x);
    if (Math.abs(x) > 1e4) {                           /* ray_phase._j0_large: Hankel asymptote */
      const z = Math.abs(x);
      return Math.abs(Math.sqrt(2 / (Math.PI * z)) * ((1 - 9 / (128 * z * z)) * Math.cos(z - Math.PI / 4) + 1 / (8 * z) * Math.sin(z - Math.PI / 4)));
    }
    const n = Math.max(64, Math.ceil(Math.abs(x)) + 64);
    let s = 0;
    for (let j = 0; j < n; j++) s += Math.cos(x * Math.sin(2 * Math.PI * ((j + 0.5) / n)));
    return Math.abs(s / n);
  }

  /* ray_phase.phase_derivatives and dither_analysis for any path function pf(thetas, nPass) -> {L, ids} */
  function phaseDerivatives(pf, thetaC, nPass, lam, nIdx, h1 = 1e-9, h2 = 1e-6) {
    const k = 2 * Math.PI * nIdx / lam;
    const { L, ids } = pf([thetaC - h2, thetaC - h1, thetaC, thetaC + h1, thetaC + h2], nPass);
    const a = new Float64Array(nPass), b = new Float64Array(nPass);
    for (let p = 0; p < nPass; p++) {
      const s1 = ids[1][p] === ids[2][p] && ids[3][p] === ids[2][p], s2 = ids[0][p] === ids[2][p] && ids[4][p] === ids[2][p];
      a[p] = s1 ? k * (L[3][p] - L[1][p]) / (2 * h1) : NaN;
      b[p] = s2 ? k * (L[4][p] - 2 * L[2][p] + L[0][p]) / (h2 * h2) : NaN;
    }
    return { a, b };
  }

  function ditherAnalysis(pf, thetaC, A, nPass, lam, nIdx, waveform, nMin = 401, nMax = 12801, stepTol = Math.PI / 2, onProgress) {
    const k = 2 * Math.PI * nIdx / lam;
    const c0 = pf([thetaC], nPass), L0 = c0.L[0], id0 = c0.ids[0];
    const { a, b } = phaseDerivatives(pf, thetaC, nPass, lam, nIdx);
    const x = Array.from(a, v => Math.abs(v) * A);
    const base = {
      a, b, x, L0, Vmodel: x.map(v => modelVisibility(v, waveform)), q: Array.from(b, v => 0.5 * Math.abs(v) * A * A),
      A09: Array.from(a, v => X_V09[waveform] / Math.abs(v)), A05: Array.from(a, v => X_V05[waveform] / Math.abs(v)),
      A0: Array.from(a, v => X_V0[waveform] / Math.abs(v)),
    };
    let n = nMin, res;
    for (;;) {
      const d = ditherOffsets(A, n, waveform);
      const { L, ids } = pf(Array.from(d, v => thetaC + v), nPass);
      const order = Array.from(d.keys()).sort((i, j) => d[i] - d[j]);
      const V = new Float64Array(nPass), maxStep = new Float64Array(nPass), same = new Float64Array(nPass);
      const ph = new Float64Array(n);
      let worst = 0;
      for (let p = 0; p < nPass; p++) {
        let cnt = 0;
        for (let i = 0; i < n; i++) { ph[i] = k * (L[i][p] - L0[p]); if (ids[i][p] === id0[p]) cnt++; }
        V[p] = visibility(ph); same[p] = cnt / n;
        let m = 0;
        for (let q = 1; q < n; q++) {
          const i = order[q], j = order[q - 1];
          if (ids[i][p] === ids[j][p] && ids[i][p] >= 0) { const st = Math.abs(ph[i] - ph[j]); if (st > m) m = st; }
        }
        maxStep[p] = m; if (m > worst) worst = m;
      }
      res = Object.assign({}, base, { V, maxStep, same, resolved: Array.from(maxStep, v => v <= stepTol), nUsed: n,
        noiseFloor: Math.sqrt(Math.PI) / (2 * Math.sqrt(n)) });
      const done = worst <= stepTol || n >= nMax;
      if (onProgress) onProgress(res, done);
      if (done) return res;
      n = Math.min(nMax, Math.max(2 * n - 1, Math.ceil(n * worst / stepTol * 1.25)));
      n += 1 - n % 2;
    }
  }
  /* ray_phase._filon: ∫ e^{iφ} over length h with φ linear from a to b, as [re, im] */
  function filon(h, a, b) {
    const x = 0.5 * (b - a), sc = Math.abs(x) < 1e-8 ? 1 - x * x / 6 : Math.sin(x) / x, m = 0.5 * (a + b);
    return [h * sc * Math.cos(m), h * sc * Math.sin(m)];
  }
  /* ray_phase.dither_filon: adaptive Filon integral of the dither average (exact where the phase is resolvable),
     V_err = Σ kept errors + sqrt(Σ open half-widths²); passes: 1-based passes that drive the refinement (null: all) */
  function ditherFilon(pf, thetaC, A, nPass, lam, nIdx, waveform, { tol = 1e-7, nMin = 256, nMax = 24000, passes = null } = {}) {
    const P = nPass, k = 2 * Math.PI * nIdx / lam;
    const c0 = pf([thetaC], P), L0 = c0.L[0], id0 = c0.ids[0];
    const delta = waveform === "sine" ? s => A * Math.sin(Math.PI * (s - 0.5)) : s => A * (2 * s - 1);
    const ev = sv => { const r = pf(sv.map(v => thetaC + delta(v)), P); return { ph: r.L.map(row => Array.from(row, (v, p) => k * (v - L0[p]))), ids: r.ids }; };
    const tgt = new Array(P).fill(passes === null);
    if (passes !== null) for (const q of passes) tgt[Math.min(Math.max(q - 1, 0), P - 1)] = true;
    const n0 = nMin, nodes = Array.from({ length: n0 + 1 }, (_, i) => i / n0), e0 = ev(nodes);
    let used = n0 + 1, iv = [];
    for (let i = 0; i < n0; i++) iv.push({ lo: nodes[i], hi: nodes[i + 1], pl: e0.ph[i], pr: e0.ph[i + 1], il: e0.ids[i] });
    const Ire = new Float64Array(P), Iim = new Float64Array(P), meas = new Float64Array(P), errS = new Float64Array(P), errQ = new Float64Array(P);
    const openM = new Float64Array(P), sameM = new Float64Array(P), maxStep = new Float64Array(P);
    while (iv.length) {
      const em = ev(iv.map(t => 0.5 * (t.lo + t.hi)));
      used += iv.length;
      const info = iv.map((t, j) => {
        const H = t.hi - t.lo, mp = em.ph[j], r = { H, mp, mi: em.ids[j], ok: [], dead: [], If: [], e: [], a: [], m: [], b: [], f: [], done: true };
        for (let p = 0; p < P; p++) {
          const fa = Number.isFinite(t.pl[p]), fm = Number.isFinite(mp[p]), fb = Number.isFinite(t.pr[p]);
          const a = fa ? t.pl[p] : 0, m = fm ? mp[p] : 0, b = fb ? t.pr[p] : 0;
          const f1 = filon(H / 2, a, m), f2 = filon(H / 2, m, b), fc = filon(H, a, b), If = [f1[0] + f2[0], f1[1] + f2[1]];
          const e = Math.hypot(fc[0] - If[0], fc[1] - If[1]);
          const ok = fa && fm && fb && e <= tol * H && Math.abs(a - 2 * m + b) <= 1, dead = !fa && !fm && !fb;
          r.ok.push(ok); r.dead.push(dead); r.If.push(If); r.e.push(e); r.a.push(a); r.m.push(m); r.b.push(b); r.f.push([fa, fm, fb]);
          if (tgt[p] && !(ok || dead)) r.done = false;
        }
        return r;
      });
      const open = []; info.forEach((f, j) => { if (!f.done) open.push(j); });
      const budget = nMax - used, refine = new Uint8Array(iv.length);
      if (open.length > Math.max(budget, 0)) {
        if (budget > 0) open.slice().sort((x, y) => (iv[y].hi - iv[y].lo) - (iv[x].hi - iv[x].lo)).slice(0, budget).forEach(j => { refine[j] = 1; });
      } else open.forEach(j => { refine[j] = 1; });
      iv.forEach((t, j) => {
        if (refine[j]) return;
        const f = info[j], half = f.H / 2;
        for (let p = 0; p < P; p++) {
          if (f.ok[p]) {
            Ire[p] += f.If[p][0]; Iim[p] += f.If[p][1]; meas[p] += f.H; errS[p] += f.e[p];
            if (t.il[p] === id0[p]) sameM[p] += f.H;
            const st = Math.abs(f.b[p] - f.a[p]); if (st > maxStep[p]) maxStep[p] = st;
            continue;
          }
          const [fa, fm, fb] = f.f[p], a = f.a[p], m = f.m[p], b = f.b[p];
          const left = fa && fm ? filon(half, a, m) : (fa || fm) ? [half * Math.cos(fa ? a : m), half * Math.sin(fa ? a : m)] : [0, 0];
          const right = fm && fb ? filon(half, m, b) : (fm || fb) ? [half * Math.cos(fb ? b : m), half * Math.sin(fb ? b : m)] : [0, 0];
          Ire[p] += left[0] + right[0]; Iim[p] += left[1] + right[1];
          const lit = (fa || fm ? 1 : 0) + (fm || fb ? 1 : 0);
          meas[p] += half * lit;
          if (!f.dead[p]) { errQ[p] += half * half * lit; openM[p] += half * lit; }
        }
      });
      const L = [], Rr = [];
      iv.forEach((t, j) => { if (refine[j]) { const m = 0.5 * (t.lo + t.hi); L.push({ lo: t.lo, hi: m, pl: t.pl, pr: info[j].mp, il: t.il }); Rr.push({ lo: m, hi: t.hi, pl: info[j].mp, pr: t.pr, il: info[j].mi }); } });
      if (!L.length) break;
      iv = L.concat(Rr);
    }
    const V = new Float64Array(P), Verr = new Float64Array(P), frag = new Float64Array(P), same = new Float64Array(P);
    let nf = 0;
    for (let p = 0; p < P; p++) {
      if (!(meas[p] > 0)) { V[p] = NaN; Verr[p] = NaN; continue; }
      V[p] = Math.hypot(Ire[p], Iim[p]) / meas[p]; Verr[p] = (errS[p] + Math.sqrt(errQ[p])) / meas[p];
      frag[p] = openM[p] / meas[p]; same[p] = sameM[p] / meas[p];
      if (Verr[p] > nf) nf = Verr[p];
    }
    const { a, b } = phaseDerivatives(pf, thetaC, P, lam, nIdx);
    const x = Array.from(a, v => Math.abs(v) * A);
    return { V, Verr, frag, same, maxStep, resolved: Array.from(V, (v, p) => Verr[p] <= Math.max(10 * tol, 0.2 * v)), nUsed: used, noiseFloor: nf,
      a, b, x, L0, Vmodel: x.map(v => modelVisibility(v, waveform)), q: Array.from(b, v => 0.5 * Math.abs(v) * A * A) };
  }
  const cellPaths = (c, sIn) => (th, n) => pathLengths(c, th, n, sIn, true);

  /* bragg_grating.stack_R_oblique for real indices: tilted admittances (s: n cos θ, p: n / cos θ), Im(cos θ) >= 0 */
  const cx = (re, im) => ({ re, im });
  const cmul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
  const cadd = (a, b) => cx(a.re + b.re, a.im + b.im);
  const csub = (a, b) => cx(a.re - b.re, a.im - b.im);
  const cdiv = (a, b) => { const d = b.re * b.re + b.im * b.im; return cx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d); };
  function stackROblique(lam, sinIn, nIn, layers, nOut, pol) {
    const s = nIn * Math.min(Math.abs(sinIn), 1 - 1e-9), k = 2 * Math.PI / lam;
    const cosOf = n => { const q = 1 - (s / n) ** 2; return q >= 0 ? cx(Math.sqrt(q), 0) : cx(0, Math.sqrt(-q)); };
    const adm = n => { const c = cosOf(n); return pol === "s" ? cx(n * c.re, n * c.im) : cdiv(cx(n, 0), c); };
    let A = cx(1, 0), B = cx(0, 0), C = cx(0, 0), D = cx(1, 0);
    for (const [n, d] of layers) {
      const c = cosOf(n), q = adm(n);
      const dr = k * n * d * c.re, di = k * n * d * c.im;
      const cs = cx(Math.cos(dr) * Math.cosh(di), -Math.sin(dr) * Math.sinh(di));
      const sn = cx(Math.sin(dr) * Math.cosh(di), Math.cos(dr) * Math.sinh(di));
      const isn = cx(sn.im, -sn.re);
      const L01 = cdiv(isn, q), L10 = cmul(isn, q);
      const nA = cadd(cmul(A, cs), cmul(B, L10)), nB = cadd(cmul(A, L01), cmul(B, cs));
      const nC = cadd(cmul(C, cs), cmul(D, L10)), nD = cadd(cmul(C, L01), cmul(D, cs));
      const m = Math.max(Math.hypot(nA.re, nA.im), Math.hypot(nB.re, nB.im), Math.hypot(nC.re, nC.im), Math.hypot(nD.re, nD.im)) || 1;
      A = cx(nA.re / m, nA.im / m); B = cx(nB.re / m, nB.im / m); C = cx(nC.re / m, nC.im / m); D = cx(nD.re / m, nD.im / m);
    }
    const qi = adm(nIn), qo = adm(nOut);
    const x = cmul(qi, cadd(A, cmul(B, qo))), y = cadd(C, cmul(D, qo));
    const r = cdiv(csub(x, y), cadd(x, y));
    return r.re * r.re + r.im * r.im;
  }

  /* bragg_grating.TrenchDBR: membrane | [gap, tooth] x N | gap, quarter-wave orders at the design angle */
  const LATERAL_POL = { TE: "p", TM: "s" };
  function trenchDBR({ nTooth, lam, N, mGap = 1, mTooth = 1, nGap = 1, loss = 0, slabPol = "TM", sinDesign = 0 }) {
    const sCrit = nGap / nTooth;
    if (!(sinDesign >= 0 && sinDesign < sCrit)) throw new Error(`design angle beyond the critical angle (sin < ${sCrit.toFixed(4)})`);
    const dGap = mGap * lam / (4 * nGap * Math.sqrt(1 - (nTooth * sinDesign / nGap) ** 2));
    const dTooth = mTooth * lam / (4 * nTooth * Math.sqrt(1 - sinDesign * sinDesign));
    const layers = []; for (let i = 0; i < N; i++) layers.push([nGap, dGap], [nTooth, dTooth]);
    const pol = LATERAL_POL[slabPol];
    return { dGap, dTooth, sCrit, sinDesign, R: (l, s) => stackROblique(l, s, nTooth, layers, nGap, pol) * (1 - loss) };
  }

  /* cell_mirror.hit_angles: |sin χ| and chord per hit, NaN once a ray has left */
  function hitAngles(c, thetas, nHits, sIn) {
    const S = [], C = [];
    for (const th of thetas) {
      let [x, y, dx, dy] = launch(c, sIn === undefined ? c.sIn : sIn, th);
      const s = new Float64Array(nHits).fill(NaN), ch = new Float64Array(nHits).fill(NaN);
      for (let j = 0; j < nHits; j++) {
        const h = hit(c, x, y, dx, dy);
        if (!Number.isFinite(h.t)) break;
        s[j] = Math.abs(dx * h.ny - dy * h.nx); ch[j] = h.t;
        x += h.t * dx; y += h.t * dy;
        [dx, dy] = reflect(dx, dy, h.nx, h.ny);
      }
      S.push(s); C.push(ch);
    }
    return { S, C };
  }

  function weightedQuantile(values, weights, q) {
    const o = Array.from(values.keys()).sort((i, j) => values[i] - values[j]);
    let tot = 0; for (const w of weights) tot += w;
    let cw = 0;
    for (const i of o) { cw += weights[i]; if (cw >= q * tot) return values[i]; }
    return values[o[o.length - 1]];
  }

  /* cell_mirror.weighted_stats: exact R at every hit; I_j = product of the R met before hit j */
  function weightedStats(S, C, Rf, binDeg = 1) {
    const nb = Math.round(90 / binDeg), hist = new Float64Array(nb);
    const chis = [], ws = [], perRay = [];
    let W = 0, WR = 0, nHit = 0, sumIend = 0, sumLeff = 0, sumL = 0, chiMax = -Infinity;
    for (let r = 0; r < S.length; r++) {
      let cum = 0, Leff = 0, L = 0;
      const Iray = [];
      for (let j = 0; j < S[r].length; j++) {
        const s = S[r][j];
        if (!Number.isFinite(s)) continue;
        const R = Rf(s), lr = Math.log(Math.min(Math.max(R, 1e-300), 1));
        const I = Math.exp(cum);
        W += I; WR += I * Math.exp(lr); nHit++;
        Leff += I * C[r][j]; L += C[r][j];
        const chi = Math.asin(Math.min(Math.max(s, 0), 1)) * 180 / Math.PI;
        chis.push(chi); ws.push(I); if (chi > chiMax) chiMax = chi;
        hist[Math.min(Math.floor(chi / binDeg), nb - 1)] += I;
        Iray.push(I);
        cum += lr;
      }
      sumIend += Math.exp(cum); sumLeff += Leff; sumL += L;
      perRay.push({ I: Iray, Iend: Math.exp(cum) });
    }
    const nR = S.length;
    return { R_mean: WR / W, R_eff: S[0] && S[0].length ? (sumIend / nR) ** (nR / Math.max(nHit, 1)) : 1, I_end: sumIend / nR, L_eff: sumLeff / nR, L_geom: sumL / nR,
      chi_50: weightedQuantile(chis, ws, 0.5), chi_95: weightedQuantile(chis, ws, 0.95), chi_max: chiMax,
      hist: Array.from(hist, v => v / W), n_hits: nHit, perRay };
  }

  function uniformAverage(Rf, n = 4097) {
    let acc = 0, prev = Rf(0);
    for (let i = 1; i < n; i++) { const s = i / (n - 1), R = Rf(s); acc += 0.5 * (R + prev) / (n - 1); prev = R; }
    return acc;
  }

  function launchAngles(thetaC, fan, nBeams) {
    return nBeams === 1 ? [thetaC] : Array.from({ length: nBeams }, (_, i) => thetaC + fan * (i / (nBeams - 1) - 0.5));
  }

  function cellMirrorStats(c, thetaC, fan, nBeams, nHits, dbr, lam, withUniform = true) {
    const Rf = s => dbr.R(lam, s);
    const { S, C } = hitAngles(c, launchAngles(thetaC, fan, nBeams), nHits);
    const out = weightedStats(S, C, Rf);
    out.R_design = dbr.R(lam, dbr.sinDesign);
    out.R_uniform = withUniform ? uniformAverage(Rf) : NaN;
    return out;
  }

  /* planar_cell.trace_rays + evaluate with a constant R (mulberry32 draws, not numpy's): T, mean path, histogram */
  function portStatistics(c, { portW, sIn, sOut, nRays, nBounce, theta0, thetaC = 0, R = 1, seed = 1, alpha = 0, nBins = 40 }) {
    let a = seed >>> 0;
    const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const ports = { sIn, sOut, w: portW };
    const det = [], cnt = { 1: 0, 0: 0, [-1]: 0, [-2]: 0 };
    for (let i = 0; i < nRays; i++) {
      const s = sIn + (u() - 0.5) * portW, th = thetaC + (u() * 2 - 1) * theta0;
      const tr = trace(c, launch(c, s, th), nBounce, ports);
      cnt[tr.exit]++;
      if (tr.exit === 1) det.push({ L: tr.L, W: R ** (tr.s.length - 1) * Math.exp(-alpha * tr.L) });
    }
    let SW = 0, SWL = 0, SWLL = 0, SW2 = 0, Lmax = 0;
    for (const d of det) { SW += d.W; SWL += d.W * d.L; SWLL += d.W * d.L * d.L; SW2 += d.W * d.W; Lmax = Math.max(Lmax, d.L); }
    const Lm = SWL / SW, varL = SWLL / SW - Lm * Lm, nk = SW * SW / SW2;
    const hist = new Float64Array(nBins), top = Lmax * 1.0001 || 1;
    for (const d of det) hist[Math.min(Math.floor(d.L / top * nBins), nBins - 1)] += d.W / Math.max(SW, 1e-300);
    const ell = c.meanChord, eta = portW / c.perimeter, surv = (1 - 2 * eta) * R * Math.exp(-alpha * ell);
    return { T: SW / nRays, Lmean: Lm, Lerr: Math.sqrt(Math.max(varL, 0) / Math.max(nk, 1)), nDet: det.length, counts: cnt, hist, Ltop: top,
      mf: { T: eta * Math.exp(-alpha * ell) / (1 - surv), L: ell / (1 - surv) } };
  }

  /* planar_cell.lyapunov_fit (scale: sqrt(area)) */
  function lyapunovFit(sep, scale, lo = 1e-10, hi = 1e-2, minPts = 6) {
    const pts = sep.map((v, j) => [j + 1, v]).filter(([, v]) => v > lo * scale && v < hi * scale);
    if (pts.length < minPts) return null;
    const n = pts.length, mx = pts.reduce((a, [x]) => a + x, 0) / n, my = pts.reduce((a, [, v]) => a + Math.log(v), 0) / n;
    let sxy = 0, sxx = 0; pts.forEach(([x, v]) => { sxy += (x - mx) * (Math.log(v) - my); sxx += (x - mx) ** 2; });
    return { lam: sxy / sxx, a: my - sxy / sxx * mx, x0: pts[0][0], x1: pts[n - 1][0] };
  }

  return { mulberry32, mulberryNormals, makeProfile, deformedHerriottCell, curvatureAt, profileHit, makeCell, arcElements, polygonCell, circleCell, stadiumCell, herriottPlanarCell, herriottSpots, thetaOf, reentrantSpacing,
    modeRadius, perturb, segmentOf, pointAt, hit, reflect, launch, inWindow, trace, herriottTrace, reflectionWeightedPath, pathLengths,
    cellPaths, ditherOffsets, visibility, modelVisibility, phaseDerivatives, ditherAnalysis, ditherFilon, filon, X_V09, X_V05, X_V0, stackROblique, trenchDBR,
    hitAngles, weightedQuantile, weightedStats, uniformAverage, launchAngles, cellMirrorStats, portStatistics, lyapunovFit };
})();
/* JS port of math-engines/engines/gas_cell: deformed elongated Herriott cell against the CMPC (uses PlanarEngine) */
const GasEngine = (() => {
  const Pl = PlanarEngine;
  const Q_E = 1.602176634e-19;
  const SPECIES = { CH4: [1.9404499757559918e-07, 1653.7262586923987e-9], NH3: [9.070328505920917e-07, 1531.6844239947554e-9],
    CO2: [1.8544907877951942e-09, 1572.3344606971416e-9], H2O: [2.9303442119426514e-09, 1528.9584734878601e-9] };
  const A_ = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const B_ = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const C_ = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const D_ = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  function invNormal(p) {
    if (p < 0.02425) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (((((C_[0] * q + C_[1]) * q + C_[2]) * q + C_[3]) * q + C_[4]) * q + C_[5]) / ((((D_[0] * q + D_[1]) * q + D_[2]) * q + D_[3]) * q + 1);
    }
    if (p > 1 - 0.02425) return -invNormal(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((A_[0] * r + A_[1]) * r + A_[2]) * r + A_[3]) * r + A_[4]) * r + A_[5]) * q / (((((B_[0] * r + B_[1]) * r + B_[2]) * r + B_[3]) * r + B_[4]) * r + 1);
  }
  const quantiles = n => Array.from({ length: n }, (_, i) => invNormal((i + 0.5) / n));
  /* erf: W. J. Cody's rational approximations (as OnchipEngine.erf) */
  function erf(x) {
    const ax = Math.abs(x);
    if (ax < 0.5) {
      const t = x * x;
      const p = [3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1];
      const q = [2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03];
      let xn = p[4] * t, xd = t;
      for (let i = 0; i < 3; i++) { xn = (xn + p[i]) * t; xd = (xd + q[i]) * t; }
      return x * (xn + p[3]) / (xd + q[3]);
    }
    let r;
    if (ax < 4) {
      const p = [5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8];
      const q = [1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03];
      let xn = p[8] * ax, xd = ax;
      for (let i = 0; i < 7; i++) { xn = (xn + p[i]) * ax; xd = (xd + q[i]) * ax; }
      r = (xn + p[7]) / (xd + q[7]);
    } else {
      const p = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
      const q = [2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
      const z = 1 / (ax * ax);
      let xn = p[5] * z, xd = z;
      for (let i = 0; i < 4; i++) { xn = (xn + p[i]) * z; xd = (xd + q[i]) * z; }
      r = (1 / Math.sqrt(Math.PI) - z * (xn + p[4]) / (xd + q[4])) / ax;
    }
    const xsq = Math.trunc(ax * 16) / 16, del = (ax - xsq) * (ax + xsq);
    r *= Math.exp(-xsq * xsq) * Math.exp(-del);
    return x < 0 ? r - 1 : 1 - r;
  }
  const gfrac = (lo, hi, s, w) => { const k = Math.SQRT2 / w; return 0.5 * (erf(k * (hi - s)) - erf(k * (lo - s))); };

  /* ---------------------------------------------------------------- cells */
  const designRadius = (L, N, M) => L / (1 - Math.cos(2 * Math.PI * M / N));
  const mirrorRadiusW = (L, R, lamM) => { const g = 1 - L / R; return Math.sqrt(lamM * L / (Math.PI * Math.sqrt(1 - g * g))); };
  function baselineCell({ L = 15e-3, W = 1.5e-3, N = 14, M = 3, distinct = true, portW = 0, eps4 = 0, eps3 = 0, conic = 0, deform = "both", walls = false, lamM = 1.6537e-6 / 2,
    nSeg = 0, segCurv = 1, segTilt = 0, segSeed = 1, rippleA = 0, ripplePeriod = 0.5e-3, ripplePhase = 0, rippleOn = "both" } = {}) {
    const R = designRadius(L, N, M), pw = portW > 0 ? portW : 2.5 * mirrorRadiusW(L, R, lamM), a = W / 2, A = a - pw;
    if (A <= 0) throw new Error("the window does not fit: make the cell wider or the window narrower");
    const c = Pl.deformedHerriottCell(R, N, M, A, { portW: pw, aperture: a, d: L, phase: distinct ? Math.PI / N : 0, conic, eps3, eps4, deform, walls, nSeg, segCurv, segTilt, segSeed,
      rippleA, ripplePeriod, ripplePhase, rippleOn });
    Object.assign(c.meta, { kind: "herriott", L, W });
    return c;
  }
  function circleReference(area, nFacets = 24, portW = 150e-6, outFrac = 0.37, smooth = false) {
    let c, r;
    if (smooth) { r = Math.sqrt(area / Math.PI); c = Pl.circleCell(r); }
    else { r = Math.sqrt(2 * area / (nFacets * Math.sin(2 * Math.PI / nFacets))); c = Pl.polygonCell(r, nFacets); }
    c.sOut = c.sIn + outFrac * c.perimeter;
    Object.assign(c.meta, { kind: "circle", r, portW, nFacets: smooth ? 0 : nFacets });
    return c;
  }

  /* ---------------------------------------------------------------- index inclusions and the medium tracer */
  const HASH_M = 2147483647;
  /* gas_cell.place_inclusions (same mulberry32 stream and order of draws) */
  function placeInclusions(c, nInc, radius = 50e-6, dn = 0.01, sign = "both", seed = 1, gap = null) {
    gap = gap === null ? 0.5 * radius : gap;
    const m = c.meta, u = Pl.mulberry32(seed), out = { cx: [], cy: [], r: [], dn: [] };
    let box, inside;
    if (m.kind === "circle") {
      const rmax = m.r * (m.nFacets ? Math.cos(Math.PI / m.nFacets) : 1) - radius - gap;
      box = [-rmax, rmax, -rmax, rmax]; inside = (x, y) => x * x + y * y <= rmax * rmax;
    } else {
      let xe = -Infinity, xb = Infinity;
      for (let k = 0; k < c.E; k++) if (c.labels[k].startsWith("M")) { xe = Math.max(xe, Math.abs(c.Ax[k])); xb = Math.min(xb, Math.abs(c.Bx[k])); }
      xe = Math.min(xe, xb);
      const xm = xe - radius - gap, ym = m.aperture - radius - gap;
      box = [-xm, xm, -ym, ym]; inside = () => true;
    }
    if (box[1] <= box[0] || box[3] <= box[2]) return out;
    let tries = 0;
    while (out.cx.length < nInc && tries < 400 * Math.max(nInc, 1)) {
      tries++;
      const x = box[0] + (box[1] - box[0]) * u(), y = box[2] + (box[3] - box[2]) * u();
      if (!inside(x, y)) continue;
      let clash = false;
      for (let i = 0; i < out.cx.length; i++) if ((x - out.cx[i]) ** 2 + (y - out.cy[i]) ** 2 < (2 * radius + gap) ** 2) { clash = true; break; }
      if (clash) continue;
      const sg = sign === "both" ? (u() < 0.5 ? -1 : 1) : sign === "plus" ? 1 : -1;
      out.cx.push(x); out.cy.push(y); out.r.push(radius); out.dn.push(sg * dn);
    }
    return out;
  }
  const medium = (cell, inc = null, n0 = 1) => ({ cell, cx: inc ? inc.cx : [], cy: inc ? inc.cy : [], r: inc ? inc.r : [], dn: inc ? inc.dn : [], n0, nInc: inc ? inc.cx.length : 0 });
  const asMedium = (c, n0 = 1) => (c && c.cell ? c : medium(c, null, n0));
  /* gas_cell.march for one ray: wall hits (optical path / n0, geometric path, s, sin χ, element, point, id), exit code */
  function marchRay(mc, start, maxWalls, sIn = 0, sOut = 0, portW = 0, ports = "none", record = false) {
    const c = mc.cell, Pm = c.perimeter, eps = 1e-12 * Pm, maxEv = maxWalls * 40 + 200;
    let [x, y, dx, dy] = start, inside = -1, Lo = 0, Lg = 0, pid = 0, exit = -1, nw = 0;
    const WL = [], WG = [], WS = [], WC = [], WK = [], WX = [], WY = [], ID = [], pts = record ? [[x, y]] : null;
    for (let ev = 0; ev < maxEv; ev++) {
      let tw = Infinity, hw = null;
      if (inside < 0) { hw = Pl.hit(c, x, y, dx, dy); tw = hw.t; }
      let ti = Infinity, ji = -1;
      if (mc.nInc) {
        if (inside < 0) {
          for (let j = 0; j < mc.nInc; j++) {
            const bx = x - mc.cx[j], by = y - mc.cy[j], b = bx * dx + by * dy, cc = bx * bx + by * by - mc.r[j] * mc.r[j], disc = b * b - cc;
            if (disc > 0) { const t = -b - Math.sqrt(disc); if (t > eps && t < ti) { ti = t; ji = j; } }
          }
        } else {
          const j = inside, bx = x - mc.cx[j], by = y - mc.cy[j], b = bx * dx + by * dy, cc = bx * bx + by * by - mc.r[j] * mc.r[j], disc = b * b - cc;
          ti = -b + Math.sqrt(Math.max(disc, 0)); ji = j;
        }
      }
      const incEv = ti < tw, leak = !incEv && !Number.isFinite(tw);
      if (leak) { exit = -2; pid = -1; break; }
      const t = incEv ? ti : tw, nmed = mc.n0 + (inside >= 0 ? mc.dn[inside] : 0);
      x += t * dx; y += t * dy; Lg += t; Lo += t * nmed / mc.n0;
      if (record) pts.push([x, y]);
      if (incEv) {
        const j = ji, nxn = (x - mc.cx[j]) / mc.r[j], nyn = (y - mc.cy[j]) / mc.r[j], ent = inside < 0;
        const n1 = ent ? mc.n0 : mc.n0 + mc.dn[j], n2 = ent ? mc.n0 + mc.dn[j] : mc.n0, Nx = ent ? nxn : -nxn, Ny = ent ? nyn : -nyn;
        const cosi = -(dx * Nx + dy * Ny), eta = n1 / n2, kk = 1 - eta * eta * (1 - cosi * cosi), tir = kk < 0, sk = Math.sqrt(Math.max(kk, 0));
        const rx = tir ? dx + 2 * cosi * Nx : eta * dx + (eta * cosi - sk) * Nx, ry = tir ? dy + 2 * cosi * Ny : eta * dy + (eta * cosi - sk) * Ny, nn = Math.hypot(rx, ry);
        dx = rx / nn; dy = ry / nn;
        if (!tir) { if (ent && pid >= 0) pid = (pid * 1000003 + 100000 + j) % HASH_M; inside = ent ? j : -1; }
        continue;
      }
      WL.push(Lo); WG.push(Lg); WS.push(hw.s); WC.push(dx * hw.ny - dy * hw.nx); WK.push(hw.k); WX.push(x); WY.push(y);
      if (pid >= 0) pid = (pid * 1000003 + hw.k + 1) % HASH_M;
      ID.push(pid); nw++;
      if (ports === "both" || ports === "out") {
        if (Pl.inWindow(hw.s, sOut, portW, Pm)) { exit = 1; break; }
        if (ports === "both" && Pl.inWindow(hw.s, sIn, portW, Pm)) { exit = 0; break; }
      }
      [dx, dy] = Pl.reflect(dx, dy, hw.nx, hw.ny);
      if (nw >= maxWalls) break;
    }
    return { WL, WG, WS, WC, WK, WX, WY, ID, nWalls: nw, exit, pts };
  }
  /* path function for Pl.ditherAnalysis: optical path / n0 and ids per wall hit (NaN, -1 after a leak) */
  const mediumPaths = (mc, sIn) => (thetas, nPass) => {
    const Ls = [], IDs = [];
    for (const th of thetas) {
      const r = marchRay(mc, Pl.launch(mc.cell, sIn, th), nPass);
      const L = new Float64Array(nPass).fill(NaN), ids = new Float64Array(nPass).fill(-1);
      for (let j = 0; j < r.nWalls; j++) { L[j] = r.WL[j]; ids[j] = r.ID[j]; }
      Ls.push(L); IDs.push(ids);
    }
    return { L: Ls, ids: IDs };
  };

  /* ---------------------------------------------------------------- Gaussian path along the centre ray */
  function centrePath(cell, sIn, theta, sOut, portW, nMax = 2000) {
    const mc = asMedium(cell), c = mc.cell, st = Pl.launch(c, sIn, theta), r = marchRay(mc, st, nMax, sIn, sOut, portW, "out", true);
    const n = r.nWalls, chords = r.WG.map((g, j) => g - (j ? r.WG[j - 1] : 0));
    return { xs: r.pts.map(p => p[0]), ys: r.pts.map(p => p[1]), hx: [st[0]].concat(r.WX), hy: [st[1]].concat(r.WY), s: r.WS, sinchi: r.WC, element: r.WK,
      chords, kappa: r.WS.map(v => Pl.curvatureAt(c, v)), cos: r.WC.map(v => Math.sqrt(Math.min(Math.max(1 - v * v, 1e-12), 1))), exits: r.exit === 1, nHits: n,
      L_opt: n ? r.WL[n - 1] : 0 };
  }

  /* ---------------------------------------------------------------- Gaussian-beam two-mirror model */
  const cdivr = (a, b) => { const d = b.re * b.re + b.im * b.im; return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d }; };
  const inv = q => cdivr({ re: 1, im: 0 }, q);
  function eigenQ(Mt) {
    const [A, B, C, D] = Mt, m = 0.5 * (A + D);
    if (Math.abs(m) >= 1 || B === 0) return null;
    return inv({ re: (D - A) / (2 * B), im: -Math.sqrt(1 - m * m) / Math.abs(B) });
  }
  const mul = (P, Q) => [P[0] * Q[0] + P[1] * Q[2], P[0] * Q[1] + P[1] * Q[3], P[2] * Q[0] + P[3] * Q[2], P[2] * Q[1] + P[3] * Q[3]];
  function inputQ(o) {
    const c0 = baselineCell(Object.assign({}, o, { eps4: 0, eps3: 0, conic: 0 }));
    const p = centrePath(c0, c0.sIn, c0.meta.thetaLaunch, c0.sOut, c0.meta.portW, 3);
    let Mt = [1, 0, 0, 1];
    for (let j = 0; j < 2; j++) { Mt = mul([1, p.chords[j], 0, 1], Mt); Mt = mul([1, 0, -2 * p.kappa[j] / p.cos[j], 1], Mt); }
    const q = eigenQ(Mt);
    if (!q) throw new Error("the undeformed baseline is not a stable two-mirror cell");
    return { q, c0 };
  }
  function qWidth(q, lamM) { const iq = inv(q); return [Math.sqrt(-lamM / (Math.PI * iq.im)), iq.re !== 0 ? 1 / iq.re : Infinity]; }
  function gaussianPath(c, path, q0, lamM, ports, Rm = 1) {
    const n = path.nHits, wIn = qWidth(q0, lamM)[0];
    let q = { re: q0.re, im: q0.im }, ell = 1;
    if (n) { let s = 0; for (const v of path.chords) s += v; ell = s / n; }
    let Mt = [1, 0, 0, 1], T = 1;
    const W = new Float64Array(n), share = new Float64Array(n).fill(1), steps = [];
    for (let j = 0; j < n; j++) {
      q = { re: q.re + path.chords[j], im: q.im };
      const F = [1, path.chords[j] / ell, 0, 1];
      Mt = mul(F, Mt); steps.push(F);
      W[j] = qWidth(q, lamM)[0];
      const last = j === n - 1 && path.exits;
      if (!last) {
        const k = path.element[j], s = path.s[j], wf = W[j] / path.cos[j], lo = c.s0[k], hi = c.s0[k] + c.L[k], P = c.perimeter;
        let sh = gfrac(lo, hi, s, wf);
        for (const [s0, pw] of ports) {
          const d = ((s0 - s + P / 2) % P + P) % P - P / 2;
          if (s + d + pw / 2 > lo && s + d - pw / 2 < hi) sh -= gfrac(Math.max(s + d - pw / 2, lo), Math.min(s + d + pw / 2, hi), s, wf);
        }
        share[j] = Math.max(sh, 0); T *= share[j] * Rm;
        const Cm = -2 * path.kappa[j] / path.cos[j];
        q = cdivr(q, { re: Cm * q.re + 1, im: Cm * q.im });
        const Mm = [1, 0, Cm * ell, 1];
        Mt = mul(Mm, Mt); steps[steps.length - 1] = mul(Mm, steps[steps.length - 1]);
      }
    }
    const tr = Mt[0] ** 2 + Mt[1] ** 2 + Mt[2] ** 2 + Mt[3] ** 2, det = Mt[0] * Mt[3] - Mt[1] * Mt[2];
    const sv = n ? Math.sqrt((tr + Math.sqrt(Math.max(tr * tr - 4 * det * det, 0))) / 2) : 1;
    let mMax = steps.length > 1 ? 0 : 1;
    for (let j = 0; j + 1 < steps.length; j++) { const P2 = mul(steps[j + 1], steps[j]); mMax = Math.max(mMax, Math.abs(0.5 * (P2[0] + P2[3]))); }
    let wmax = 0; for (const v of W) wmax = Math.max(wmax, v);
    return { w: W, share, T_G: T, w_max_ratio: n ? wmax / wIn : 1, lam_G: n ? Math.log(sv) / n : 0, m_max: mMax, w_in: wIn, q_end: q };
  }

  /* ---------------------------------------------------------------- ray ensemble, chaos, figures */
  function beamRays(c, sIn, thetaC, q0, lamM, nSide = 24) {
    const [w, Rw] = qWidth(q0, lamM), thw = lamM / (Math.PI * w), z = quantiles(nSide), cosT = Math.max(Math.cos(thetaC), 1e-6), out = [];
    for (let i = 0; i < nSide; i++) for (let j = 0; j < nSide; j++) {
      const off = w / 2 * z[i], ang = thw / 2 * z[j] + (Number.isFinite(Rw) ? off / Rw : 0);
      out.push(Pl.launch(c, sIn + off / cosT, thetaC + ang));
    }
    return out;
  }
  /* gas_cell.ensemble: port-to-port statistics and the M² of the detected light */
  let ALPHA_M = 0;
  function setAlpha(a) { ALPHA_M = a; }
  function ensemble(mc, starts, portW, sIn, sOut, nBounce, Rm, gamma, lamM) {
    const n = starts.length, Pm = mc.cell.perimeter, Ls = [], Ws = [], ys = [], us = [];
    let leaked = 0, back = 0, trapped = 0;
    for (const st of starts) {
      const r = marchRay(mc, st, nBounce, sIn, sOut, portW, "both");
      if (r.exit === 1) {
        const nh = r.nWalls, sc = r.WC[nh - 1], ds = (((r.WS[nh - 1] - sOut + Pm / 2) % Pm) + Pm) % Pm - Pm / 2;
        Ls.push(r.WG[nh - 1]); Ws.push((Rm < 1 ? Rm ** (nh - 1) : 1) * Math.exp(-ALPHA_M * r.WG[nh - 1]));
        ys.push(ds * Math.sqrt(Math.min(Math.max(1 - sc * sc, 0), 1))); us.push(Math.asin(Math.min(Math.max(sc, -1), 1)));
      } else if (r.exit === -2) leaked++; else if (r.exit === 0) back++; else trapped++;
    }
    const out = { leaked: leaked / n, back: back / n, trapped: trapped / n, L: Ls, W: Ws };
    let sw = 0; for (const w of Ws) sw += w;
    if (!(sw > 0)) return Object.assign(out, { T: 0, L_mean: NaN, L_rms: NaN, S1: 0, M2: NaN });
    let swl = 0, my = 0, mu = 0;
    for (let i = 0; i < Ls.length; i++) { swl += Ws[i] * Ls[i]; my += Ws[i] * ys[i]; mu += Ws[i] * us[i]; }
    const Lm = swl / sw; my /= sw; mu /= sw;
    let vl = 0, cyy = 0, cuu = 0, cyu = 0;
    for (let i = 0; i < Ls.length; i++) { vl += Ws[i] * (Ls[i] - Lm) ** 2; cyy += Ws[i] * (ys[i] - my) ** 2; cuu += Ws[i] * (us[i] - mu) ** 2; cyu += Ws[i] * (ys[i] - my) * (us[i] - mu); }
    cyy /= sw; cuu /= sw; cyu /= sw;
    return Object.assign(out, { T: sw / n, L_mean: Lm, L_rms: Math.sqrt(vl / sw), S1: gamma * sw / n * Lm, M2: 4 * Math.PI / lamM * Math.sqrt(Math.max(cyy * cuu - cyu * cyu, 0)) });
  }
  function twinDivergence(mc, sIn, theta, nHits = 300, dtheta = 1e-9) {
    const c = mc.cell, a = marchRay(mc, Pl.launch(c, sIn, theta), nHits), b = marchRay(mc, Pl.launch(c, sIn, theta + dtheta), nHits);
    const n = Math.min(a.nWalls, b.nWalls), sep = [];
    for (let j = 0; j < n; j++) sep.push(Math.hypot(a.WX[j] - b.WX[j], a.WY[j] - b.WY[j]));
    return { sep, fit: Pl.lyapunovFit(sep, Math.sqrt(Math.abs(c.area))) };
  }
  /* gas_cell.tangent_chaos: FTLE and MEGNO of one ray from the exact tangent map of the walls (no ports, no inclusions) */
  function tangentChaos(c, start, nHits = 300) {
    const mc = asMedium(c && c.cell ? c.cell : c), r = marchRay(mc, start, nHits), nh = r.nWalls, ell = Math.sqrt(Math.abs(mc.cell.area));
    let u = Math.sqrt(0.5), v = Math.sqrt(0.5), acc = 0, Sk = 0, Yb = 0;
    for (let j = 0; j < nh; j++) {
      const ch = r.WG[j] - (j ? r.WG[j - 1] : 0), kap = Pl.curvatureAt(mc.cell, r.WS[j]), cs = Math.sqrt(Math.min(Math.max(1 - r.WC[j] ** 2, 1e-12), 1));
      u = u + ch / ell * v;
      v = v - 2 * kap * ell / cs * u;
      const nn = Math.hypot(u, v), g = Math.log(nn);
      acc += g; Sk += (j + 1) * g; Yb += 2 * Sk / (j + 1);
      u /= nn; v /= nn;
    }
    const k = Math.max(nh, 1);
    return { lam: acc / k, Y: Yb / k, nHits: nh };
  }
  const median = a => { if (!a.length) return NaN; const b = Float64Array.from(a).sort(), m = b.length >> 1; return b.length % 2 ? b[m] : 0.5 * (b[m - 1] + b[m]); };
  const VERDICTS = { 2: "launches into the chaotic sea", 1: "launches into a regular island", 0: "regular phase space (no chaotic sea)", [-1]: "launch escapes the cell" };
  /* gas_cell.chaos_map: chaotic share of the launch beam and of a section grid on the input mirror, verdict */
  function chaosMap(cell, thetaC, q0, lamM, { nHits = 300, nGrid = 0, nSide = 8, Ychaos = 4, minHits = 30, onRow = null } = {}) {
    const c = cell && cell.cell ? cell.cell : cell, sIn = c.sIn;
    const [w, Rw] = qWidth(q0, lamM), thw = lamM / (Math.PI * w), z = quantiles(nSide), cosT = Math.max(Math.cos(thetaC), 1e-6);
    const tb = beamRays(c, sIn, thetaC, q0, lamM, nSide).map((st, i) => {
      const off = w / 2 * z[Math.floor(i / nSide)], ang = thw / 2 * z[i % nSide] + (Number.isFinite(Rw) ? off / Rw : 0);
      return Object.assign(tangentChaos(c, st, nHits), { s: sIn + off / cosT, p: Math.sin(thetaC + ang) });
    });
    let nst = 0, nch = 0; const Ys = [], lams = [];
    for (const t of tb) { Ys.push(t.Y); if (t.nHits >= minHits) { nst++; lams.push(t.lam); if (t.Y > Ychaos) nch++; } }
    const out = { f_launch: nch / tb.length, esc_launch: (tb.length - nst) / tb.length, Y_launch: median(Ys), lam_ftle: nst ? median(lams) : NaN,
      sea: NaN, island: NaN, esc_sea: NaN, grid: null, launch: tb };
    if (nGrid) {
      const m = c.meta, sg = [], pg = [];
      let pm;
      if (m.kind === "herriott" || "aperture" in m) {
        const ks = c.labels.map((l, k) => (l === "M1" || l.startsWith("M1 ") ? k : -1)).filter(k => k >= 0), sLo = c.s0[ks[0]], sHi = c.s0[ks[ks.length - 1]] + c.L[ks[ks.length - 1]];
        for (let i = 0; i < nGrid; i++) sg.push(sLo + (sHi - sLo) * (i + 0.5) / nGrid);
        pm = 2 * m.aperture / m.d;
      } else { for (let i = 0; i < nGrid; i++) sg.push(c.perimeter * (i + 0.5) / nGrid); pm = 0.95; }
      for (let i = 0; i < nGrid; i++) pg.push(pm * (2 * (i + 0.5) / nGrid - 1));
      const Y = [], lam = [], nH = [];
      let gs = 0, gc = 0;
      for (let ip = 0; ip < nGrid; ip++) {
        const yr = [], lr = [], hr = [];
        for (let is = 0; is < nGrid; is++) {
          const t = tangentChaos(c, Pl.launch(c, sg[is], Math.asin(pg[ip])), nHits);
          yr.push(t.Y); lr.push(t.lam); hr.push(t.nHits);
          if (t.nHits >= minHits) { gs++; if (t.Y > Ychaos) gc++; }
        }
        Y.push(yr); lam.push(lr); nH.push(hr);
        if (onRow) onRow(ip + 1, nGrid);
      }
      const N2 = nGrid * nGrid;
      Object.assign(out, { sea: gc / N2, island: (gs - gc) / N2, esc_sea: (N2 - gs) / N2, grid: { s: sg, p: pg, Y, lam, n_hits: nH, pmax: pm } });
    }
    let v;
    if (out.esc_launch >= 0.5) v = -1;
    else if (nch >= 0.5 * nst) v = 2;
    else if (nGrid && out.sea >= 0.05 * (out.sea + out.island) && out.sea > 0) v = 1;
    else v = 0;
    out.verdict = v;
    return out;
  }
  function cellFigures(cell, thetaC, q0, lamM, { nEff = 2, gamma = 0.2, alphaPpm = SPECIES.CH4[0], Rm = 0.995, dither = 2e-3, waveform = "sine", fPar = 1e-6,
    Pin = 1e-3, bandwidth = 1, nSide = 24, nBounce = 2000, nLyap = 300, nPassMax = 60, nDitherMax = 24000, chaosGrid = 12, nFtle = 300, speckleAvg = 0 } = {}) {
    const mc = asMedium(cell, nEff), c = mc.cell, sIn = c.sIn, sOut = c.sOut, pw = c.meta.portW, lam = lamM * nEff;
    const ports = [[sIn, pw]].concat(Math.abs(sOut - sIn) < 1e-15 ? [] : [[sOut, pw]]);
    const ev = ensemble(mc, beamRays(c, sIn, thetaC, q0, lamM, nSide), pw, sIn, sOut, nBounce, Rm, gamma, lamM);
    const Appm = ev.T > 0 ? 100 * alphaPpm * gamma * ev.L_mean : 0;
    const sigA = ev.T > 0 ? Math.sqrt(2 * Q_E * bandwidth / (Pin * ev.T)) : Infinity;
    const cp = centrePath(mc, sIn, thetaC, sOut, pw, nBounce);
    const gb = gaussianPath(c, cp, q0, lamM, ports, Rm);
    const tw = twinDivergence(mc, sIn, thetaC, nLyap);
    const nPass = cp.nHits ? Math.min(cp.nHits, nPassMax) : 1;
    const da = Pl.ditherFilon(mediumPaths(mc, sIn), thetaC, dither, nPass, lam, nEff, waveform, { nMax: nDitherMax, passes: [nPass] });
    const V = da.V[nPass - 1], Verr = da.Verr[nPass - 1];
    const nGr = speckleAvg && Number.isFinite(ev.M2) ? Math.max(1, ev.M2) : 1, Vdet = V / Math.sqrt(nGr);
    const cm = chaosMap(mc, thetaC, q0, lamM, { nHits: nFtle, nGrid: chaosGrid });
    const necShot = Appm > 0 ? sigA / Appm : Infinity, necFringe = Appm > 0 ? 2 * Math.sqrt(fPar) * Vdet / Appm : Infinity;
    const necFringeHi = Appm > 0 ? 2 * Math.sqrt(fPar) * (V + Verr) / Math.sqrt(nGr) / Appm : Infinity;
    return { V_floor: Verr, V_err: Verr, V_err_pass: da.Verr, V_det: Vdet, n_grains: nGr, nec_fringe_hi: necFringeHi, frag: da.frag[nPass - 1], T: ev.T, L_mean: ev.L_mean, L_rms: ev.L_rms, S1: ev.S1, M2: ev.M2, leaked: ev.leaked, back: ev.back, trapped: ev.trapped, A_ppm: Appm,
      nec_shot: necShot, nec_fringe: necFringe, nec: Math.hypot(necShot, necFringe), lam: tw.fit ? tw.fit.lam : 0, lam_fit: !!tw.fit, V,
      V_pass: da.V, same_path: da.same[nPass - 1], n_pass: nPass, exit_pass: cp.exits ? cp.nHits : -1, L_centre: cp.chords.reduce((x, y) => x + y, 0),
      T_G: gb.T_G, w_max_ratio: gb.w_max_ratio, lam_G: gb.lam_G, m_max: gb.m_max, theta: thetaC, w_hits: gb.w, share: gb.share, sep: tw.sep,
      centre: cp, area: c.area, dither_n: da.nUsed, n_inc: mc.nInc, ens: ev, K_ripple: c.meta.kRipple || 0, f_launch: cm.f_launch, esc_launch: cm.esc_launch,
      Y_launch: cm.Y_launch, lam_ftle: cm.lam_ftle, sea: cm.sea, island: cm.island, verdict: cm.verdict, chaos: cm };
  }
  function retune(cell, theta0, q0, lamM, { span = null, nTheta = 41, Rm = 0.995, gamma = 0.2, nSide = 8, nBounce = 2000, nEff = 2 } = {}) {
    const mc = asMedium(cell, nEff), c = mc.cell, w = qWidth(q0, lamM)[0];
    span = span === null ? 4 * lamM / (Math.PI * w) : span;
    const pw = c.meta.portW, step = 2 / (nTheta - 1), scan = [];
    let best = [-1, theta0];
    for (let i = 0; i < nTheta; i++) {
      const th = theta0 + span * (i === nTheta - 1 ? 1 : -1 + i * step);
      const ev = ensemble(mc, beamRays(c, c.sIn, th, q0, lamM, nSide), pw, c.sIn, c.sOut, nBounce, Rm, gamma, lamM);
      scan.push([th, ev.S1]);
      if (ev.S1 > best[0] * (1 + 1e-12)) best = [ev.S1, th];
    }
    return { theta: best[1], S1: best[0], scan };
  }
  const PERT = { eps4: 0, eps3: 0, conic: 0, deform: "both", n_seg: 0, seg_curv: 1, seg_tilt: 0, seg_seed: 1, n_inc: 0, inc_radius: 50e-6, inc_dn: 0.01,
    inc_sign: "both", inc_seed: 1, inc_circle: 0, ripple_a: 0, ripple_period: 0.5e-3, ripple_phase: 0, ripple_on: "both" };
  /* gas_cell._common: input beam, baseline (deformed / segmented, with inclusions) and the CMPC as medium cells */
  function common(o) {
    const lamM = o.wavelength / o.n_eff;
    const g = { L: o.L, W: o.W, N: o.N, M: o.M, distinct: !!o.distinct, portW: o.port_width, walls: !!o.walls, lamM };
    const { q, c0 } = inputQ(g);
    const pw = c0.meta.portW, p = Object.assign({}, PERT, o);
    const base = baselineCell(Object.assign({}, g, { portW: pw, eps4: p.eps4, eps3: p.eps3, conic: p.conic, deform: p.deform, nSeg: p.n_seg, segCurv: p.seg_curv,
      segTilt: p.seg_tilt, segSeed: p.seg_seed, rippleA: p.ripple_a, ripplePeriod: p.ripple_period, ripplePhase: p.ripple_phase, rippleOn: p.ripple_on }));
    const circ = circleReference(base.area, o.n_facets, pw, o.out_frac, !!o.smooth_circle);
    const inc = p.n_inc ? placeInclusions(base, p.n_inc, p.inc_radius, p.inc_dn, p.inc_sign, p.inc_seed) : null;
    const incc = p.n_inc && p.inc_circle ? placeInclusions(circ, p.n_inc, p.inc_radius, p.inc_dn, p.inc_sign, p.inc_seed) : null;
    return { lamM, q0: q, base: medium(base, inc, o.n_eff), circ: medium(circ, incc, o.n_eff), g, pw };
  }
  const DEFAULTS = Object.assign({ L: 15e-3, W: 1.5e-3, N: 14, M: 3, distinct: 1, port_width: 0, walls: 0, n_facets: 24, circle_theta: 0.35, out_frac: 0.37,
    smooth_circle: 0, wavelength: 1.6537e-6, n_eff: 2.0, gamma: 0.2, species: "CH4", R_mirror: 0.995, dither: 2e-3, waveform: "sine", f_parasitic: 1e-6, P_in: 1e-3,
    bandwidth: 1, n_side: 24, n_bounce: 2000, retune_launch: 1, chaos_grid: 20, n_ftle: 300, n_dither_max: 24000, speckle_avg: 0 }, PERT);
  const KEYS = ["T", "L_mean", "L_rms", "S1", "M2", "A_ppm", "nec_shot", "nec_fringe", "nec", "lam", "V", "V_floor", "V_det", "n_grains",
    "nec_fringe_hi", "frag", "same_path", "n_pass", "exit_pass", "T_G", "w_max_ratio",
    "lam_G", "m_max", "leaked", "area", "theta", "n_inc", "K_ripple", "f_launch", "esc_launch", "Y_launch", "lam_ftle", "sea", "island", "verdict"];
  const figOpts = o => ({ nEff: o.n_eff, gamma: o.gamma, alphaPpm: SPECIES[o.species][0], Rm: o.R_mirror, dither: o.dither, waveform: o.waveform,
    fPar: o.f_parasitic, Pin: o.P_in, bandwidth: o.bandwidth, nSide: o.n_side, nBounce: o.n_bounce, chaosGrid: o.chaos_grid, nFtle: o.n_ftle,
    nDitherMax: o.n_dither_max, speckleAvg: o.speckle_avg });
  /* gas_cell.compare: values as the Python Result, plus the two figure sets (h, c), the medium cells and the retune scan */
  function compare(opts = {}) {
    const o = Object.assign({}, DEFAULTS, opts);
    const { lamM, q0, base, circ } = common(o);
    const rt = o.retune_launch ? retune(base, base.cell.meta.thetaLaunch, q0, lamM, { Rm: o.R_mirror, gamma: o.gamma, nEff: o.n_eff }) : null;
    const th = rt ? rt.theta : base.cell.meta.thetaLaunch;
    const h = cellFigures(base, th, q0, lamM, figOpts(o)), c = cellFigures(circ, o.circle_theta, q0, lamM, figOpts(o));
    const v = {};
    for (const k of KEYS) { v[`h_${k}`] = h[k]; v[`c_${k}`] = c[k]; }
    Object.assign(v, { path_gain: c.L_mean > 0 ? h.L_mean / c.L_mean : NaN, sensitivity_gain: c.S1 > 0 ? h.S1 / c.S1 : NaN, chaos_lost: c.lam - h.lam,
      washout_lost: h.V - c.V, nec_ratio: c.nec > 0 ? h.nec / c.nec : NaN, R_design: base.cell.meta.R, port_width: base.cell.meta.portW, w_in: qWidth(q0, lamM)[0],
      circle_radius: circ.cell.meta.r, verdict_text: VERDICTS[h.verdict] });
    return Object.assign(v, { h, c, base, circ, q0, lamM, retune: rt });
  }
  const SWEEPABLE = ["eps4", "eps3", "conic", "n_seg", "seg_curv", "seg_tilt", "n_inc", "inc_dn", "inc_radius", "ripple_a", "ripple_period", "ripple_phase"];
  /* gas_cell.perturbation_sweep: baseline rows against one perturbation parameter; the CMPC (no inclusions) as reference */
  function perturbationSweep(param, values, opts = {}, onRow = null) {
    if (!SWEEPABLE.includes(param)) throw new Error(`cannot sweep ${param}`);
    const o = Object.assign({}, DEFAULTS, { n_side: 16, chaos_grid: 10 }, opts);
    const ref = common(Object.assign({}, o, { n_inc: 0 }));
    const rows = [];
    for (const val of values) {
      const { base: b } = common(Object.assign({}, o, { [param]: val }));
      const th = o.retune_launch ? retune(b, b.cell.meta.thetaLaunch, ref.q0, ref.lamM, { Rm: o.R_mirror, gamma: o.gamma, nEff: o.n_eff }).theta : b.cell.meta.thetaLaunch;
      const r = cellFigures(b, th, ref.q0, ref.lamM, figOpts(o));
      rows.push(r); if (onRow) onRow(val, r);
    }
    return { param, values, rows, c: cellFigures(ref.circ, o.circle_theta, ref.q0, ref.lamM, figOpts(o)) };
  }
  const deformationSweep = (eps4Values, opts = {}, onRow = null) => perturbationSweep("eps4", eps4Values, opts, onRow);
  return { setAlpha, SPECIES, invNormal, quantiles, erf, gfrac, designRadius, mirrorRadiusW, baselineCell, circleReference, placeInclusions, medium, asMedium, marchRay,
    mediumPaths, tangentChaos, chaosMap, VERDICTS, centrePath, eigenQ, inputQ, qWidth, gaussianPath, beamRays, ensemble, twinDivergence, cellFigures, retune, common, compare, perturbationSweep,
    deformationSweep, DEFAULTS, KEYS, SWEEPABLE, PERT };
})();
/* DitherLab (new in this applet, no Python reference yet): one cell, either a circular multipass cell (segmented or smooth
   wall) or an integrated Herriott-type cell, with geometry perturbations (facet or mirror tilt and curvature, mirror
   deformation and ripple) and refractive-index perturbations (a disk at the cell centre, random inclusions). Gives the
   chaos indicators (twin rays, finite-time Lyapunov exponent and MEGNO from renormalised twins through walls and
   inclusions), the fringe visibility under a joint launch-angle and wavelength dither, and the speckle diversity the
   dither buys: the exit field on phase-space channels of one mode each, its coherence matrix over the dither states and
   the effective number of independent speckle patterns N_eff = (tr G)² / ‖G‖²_F. Built on PlanarEngine and GasEngine. */
const DitherLab = (() => {
  const Pl = PlanarEngine, G = GasEngine;
  const TAU = 2 * Math.PI;
  const DEFAULTS = {
    cell: "herriott",
    c_radius: 2e-3, n_facets: 24, facet_curv: 0, c_theta: 20 * Math.PI / 180, out_frac: 0.37, same_port: 0, c_port: 150e-6, c_waist: 40e-6,
    f_tilt_rms: 0, f_curv_rms: 0, f_seed: 1, f_sel: 0, f_sel_tilt: 0, f_sel_curv: 0,
    L: 10e-3, W: 1.5e-3, N: 14, M: 3, distinct: 1, port_width: 0, walls: 1,
    dR1: 0, dR2: 0, tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0, ripple_period: 0.5e-3, ripple_phase: 0, retune: 1,
    ctr_dn: 0, ctr_r: 150e-6, ctr_x: 0, ctr_y: 0, n_inc: 0, inc_ring: 1, inc_radius: 50e-6, inc_dn: 0.005, inc_sign: "both", inc_seed: 1,
    wavelength: 1.6537e-6, n_eff: 2, R_mirror: 0.995, gamma: 0.2, species: "CH4", P_in: 1e-3, f_parasitic: 1e-6, bandwidth: 1,
    dither: 2e-3, waveform: "sine", f_theta: 1000, dlam: 20e-12, lwave: "triangle", f_lambda: 1297, stray: 1, speckle_avg: 1,
    n_u: 24, n_v: 72, n_samp: 8192, n_ftle: 300, chaos_grid: 0, map_n: 7, n_side: 12, pass_cap: 200, poin_s: 36, poin_p: 12, poin_hits: 150,
  };
  const SPECIES = G.SPECIES;

  /* ---------------------------------------------------------------- cells */
  /* circular cell: regular polygon of n_facets flat facets (0: smooth circle as 24 arcs) with circumradius c_radius; every
     facet gets curvature facet_curv / r (concave), random tilt and curvature (rms, seeded) and an optional single-facet defect */
  function circularCell(o) {
    const N = Math.round(o.n_facets), r = o.c_radius, smooth = N < 3, n = smooth ? 24 : N;
    let c;
    if (smooth) {
      const els = Pl.arcElements(0, 0, r, -Math.PI / n, 2 * Math.PI - Math.PI / n, n, false, 0, "arc");
      els.forEach((e, k) => { e.ext = 0.15; e.label = `arc${k}`; });
      c = Pl.makeCell(els, { name: "circle" });
    } else c = Pl.polygonCell(r, N);
    const E = c.E, tilts = new Float64Array(E), curv = new Float64Array(E), z = Pl.mulberryNormals(o.f_seed, 2 * E);
    for (let k = 0; k < E; k++) {
      tilts[k] = o.f_tilt_rms * z[k];
      curv[k] = (smooth ? 0 : -o.facet_curv / r) - o.f_curv_rms / r * z[E + k];
    }
    if (o.f_sel_tilt || o.f_sel_curv) { const k = ((Math.round(o.f_sel) % E) + E) % E; tilts[k] += o.f_sel_tilt; curv[k] -= o.f_sel_curv / r; }
    for (let k = 0; k < E; k++) {                       /* a facet cannot bend tighter than its half chord */
      const kap = c.kappa[k] + curv[k], lim = 0.98 / c.h[k];
      if (Math.abs(kap) > lim) curv[k] = Math.sign(kap) * lim - c.kappa[k];
    }
    const p = Pl.perturb(c, tilts, null, curv);
    p.sIn = p.L[0] / 2;
    p.sOut = o.same_port ? p.sIn : p.sIn + o.out_frac * p.perimeter;
    p.meta = { kind: "circle", r, portW: o.c_port, nFacets: smooth ? 0 : N, smooth };
    return p;
  }

  /* Herriott-type cell (planar_cell.deformed_herriott_cell): design radius R for re-entrance after N passes at spacing L,
     radius errors dR1, dR2 (relative), quartic deformation, ripple, mirror tilts about the vertex, optional side walls */
  function herriottCell(o, lamM) {
    const R = G.designRadius(o.L, o.N, o.M), pw = o.port_width > 0 ? o.port_width : 2.5 * G.mirrorRadiusW(o.L, R, lamM), a = o.W / 2, A = a - pw;
    if (A <= 0) throw new Error("the window does not fit: make the cell wider or the window narrower");
    const phase = o.distinct ? Math.PI / o.N : 0;
    const c0 = Pl.deformedHerriottCell(R * (1 + o.dR1), o.N, o.M, A, { portW: pw, aperture: a, d: o.L, R2: R * (1 + o.dR2), phase, eps4: o.eps4,
      walls: !!o.walls, rippleA: o.ripple_a, ripplePeriod: o.ripple_period, ripplePhase: o.ripple_phase });
    const meta = Object.assign({}, c0.meta, { kind: "herriott", L: o.L, W: o.W, R, tilt1: o.tilt1, tilt2: o.tilt2, dR1: o.dR1, dR2: o.dR2,
      thetaLaunch: phase ? Math.atan(-A * Math.sin(phase) * Math.sqrt((2 * R - o.L) / o.L) / R) : 0 });
    const rot = (P, V, t) => { const co = Math.cos(t), si = Math.sin(t); return [V[0] + co * (P[0] - V[0]) - si * (P[1] - V[1]), V[1] + si * (P[0] - V[0]) + co * (P[1] - V[1])]; };
    const els = c0.els.map(e => {
      const t = e.label === "M1" ? o.tilt1 : e.label === "M2" ? o.tilt2 : 0;
      if (!t || !e.profile) return Object.assign({}, e);
      const p = e.profile, ax = [Math.cos(t) * p.axis[0] - Math.sin(t) * p.axis[1], Math.sin(t) * p.axis[0] + Math.cos(t) * p.axis[1]];
      const pn = Pl.makeProfile({ V: p.V, axis: ax, c: p.c, k: p.k, a3: p.a3, a4: p.a4, eta0: p.eta0, eta1: p.eta1, ra: p.ra, rl: p.rl, rp: p.rp });
      return { A: rot(e.A, p.V, t), B: rot(e.B, p.V, t), kappa: 0, ext: e.ext, label: e.label, profile: pn };
    });
    const k1 = els.findIndex(e => e.label === "M1"), k2 = els.findIndex(e => e.label === "M2");
    for (const e of els) {
      if (e.label === "side_bottom") { e.A = els[k1].B; e.B = els[k2].A; }
      if (e.label === "side_top") { e.A = els[k2].B; e.B = els[k1].A; }
    }
    const c = Pl.makeCell(els, { name: "Herriott", meta, sIn: c0.sIn, sOut: c0.sOut });
    const q0 = G.inputQ({ L: o.L, W: o.W, N: o.N, M: o.M, distinct: !!o.distinct, portW: pw, walls: !!o.walls, lamM }).q;
    return { cell: c, q0 };
  }

  /* index perturbations: the disk at the cell centre (ctr_dn, ctr_r, offset ctr_x, ctr_y) and random inclusions
     (GasEngine.placeInclusions); random disks touching the central one are dropped */
  /* circular cell: random disks only in the ring the launch rays cross, between the caustic r_c = a sin θ (a: apothem,
     θ: launch angle) and the wall, uniform in area, non-overlapping (mulberry32 stream as GasEngine.placeInclusions) */
  function ringInclusions(cell, o) {
    const m = cell.meta, R = o.inc_radius, gap = 0.5 * R, a = m.r * (m.nFacets ? Math.cos(Math.PI / m.nFacets) : 1);
    const rIn = a * Math.sin(Math.abs(o.c_theta)) + R + gap, rOut = 0.97 * a - R - gap, u = Pl.mulberry32(o.inc_seed), out = { cx: [], cy: [], r: [], dn: [] };
    if (rOut <= rIn) throw new Error("no room for inclusions between the caustic and the wall: lower the launch angle or the inclusion radius");
    for (let tries = 0; out.cx.length < o.n_inc && tries < 400 * o.n_inc; tries++) {
      const rho = Math.sqrt(rIn * rIn + (rOut * rOut - rIn * rIn) * u()), ph = 2 * Math.PI * u(), x = rho * Math.cos(ph), y = rho * Math.sin(ph);
      let clash = false;
      for (let i = 0; i < out.cx.length; i++) if ((x - out.cx[i]) ** 2 + (y - out.cy[i]) ** 2 < (2 * R + gap) ** 2) { clash = true; break; }
      if (clash) continue;
      const sg = o.inc_sign === "both" ? (u() < 0.5 ? -1 : 1) : o.inc_sign === "plus" ? 1 : -1;
      out.cx.push(x); out.cy.push(y); out.r.push(R); out.dn.push(sg * o.inc_dn);
    }
    return out;
  }
  function inclusions(cell, o) {
    const inc = !o.n_inc ? { cx: [], cy: [], r: [], dn: [] } : o.cell === "circle" && o.inc_ring ? ringInclusions(cell, o)
      : G.placeInclusions(cell, o.n_inc, o.inc_radius, o.inc_dn, o.inc_sign, o.inc_seed);
    if (o.ctr_dn && o.ctr_r > 0) {
      const m = cell.meta, x0 = o.ctr_x, y0 = o.ctr_y, R = o.ctr_r;
      const fits = m.kind === "circle" ? Math.hypot(x0, y0) + R < 0.97 * m.r * (m.nFacets ? Math.cos(Math.PI / m.nFacets) : 1)
        : Math.abs(x0) + R < 0.9 * (m.d / 2 - m.aperture ** 2 / (2 * m.R)) && Math.abs(y0) + R < m.aperture;
      if (!fits) throw new Error("the central index disk does not fit inside the cell");
      const keep = inc.cx.map((x, i) => Math.hypot(x - x0, inc.cy[i] - y0) > inc.r[i] + R + 0.25 * inc.r[i]);
      for (const key of ["cx", "cy", "r", "dn"]) inc[key] = inc[key].filter((_, i) => keep[i]);
      inc.cx.push(x0); inc.cy.push(y0); inc.r.push(R); inc.dn.push(o.ctr_dn);
    }
    return inc;
  }

  function setup(opts) {
    const o = Object.assign({}, DEFAULTS, opts), lamM = o.wavelength / o.n_eff;
    let cell, q0, theta;
    if (o.cell === "circle") { cell = circularCell(o); q0 = { re: 0, im: Math.PI * o.c_waist ** 2 / lamM }; theta = o.c_theta; }
    else { const h = herriottCell(o, lamM); cell = h.cell; q0 = h.q0; theta = cell.meta.thetaLaunch; }
    const inc = inclusions(cell, o);
    return { o, lamM, cell, mc: G.medium(cell, inc.cx.length ? inc : null, o.n_eff), q0, theta };
  }

  /* ---------------------------------------------------------------- chaos: renormalised twin rays */
  /* one wall hit through walls and inclusions (as GasEngine.marchRay): the state after the reflection; null if the ray leaves */
  function step(mc, st) {
    const c = mc.cell, eps = 1e-12 * c.perimeter;
    let [x, y, dx, dy] = st, inside = -1, Lo = 0;
    for (let ev = 0; ev < 400; ev++) {
      let tw = Infinity, hw = null;
      if (inside < 0) { hw = Pl.hit(c, x, y, dx, dy); tw = hw.t; }
      let ti = Infinity, ji = -1;
      if (mc.nInc) {
        if (inside < 0) {
          for (let j = 0; j < mc.nInc; j++) {
            const bx = x - mc.cx[j], by = y - mc.cy[j], b = bx * dx + by * dy, cc = bx * bx + by * by - mc.r[j] * mc.r[j], disc = b * b - cc;
            if (disc > 0) { const t = -b - Math.sqrt(disc); if (t > eps && t < ti) { ti = t; ji = j; } }
          }
        } else {
          const j = inside, bx = x - mc.cx[j], by = y - mc.cy[j], b = bx * dx + by * dy, cc = bx * bx + by * by - mc.r[j] * mc.r[j], disc = b * b - cc;
          ti = -b + Math.sqrt(Math.max(disc, 0)); ji = j;
        }
      }
      const incEv = ti < tw;
      if (!incEv && !Number.isFinite(tw)) return null;
      const t = incEv ? ti : tw, nmed = mc.n0 + (inside >= 0 ? mc.dn[inside] : 0);
      x += t * dx; y += t * dy; Lo += t * nmed / mc.n0;
      if (incEv) {
        const j = ji, nxn = (x - mc.cx[j]) / mc.r[j], nyn = (y - mc.cy[j]) / mc.r[j], ent = inside < 0;
        const n1 = ent ? mc.n0 : mc.n0 + mc.dn[j], n2 = ent ? mc.n0 + mc.dn[j] : mc.n0, Nx = ent ? nxn : -nxn, Ny = ent ? nyn : -nyn;
        const cosi = -(dx * Nx + dy * Ny), eta = n1 / n2, kk = 1 - eta * eta * (1 - cosi * cosi), tir = kk < 0, sk = Math.sqrt(Math.max(kk, 0));
        const rx = tir ? dx + 2 * cosi * Nx : eta * dx + (eta * cosi - sk) * Nx, ry = tir ? dy + 2 * cosi * Ny : eta * dy + (eta * cosi - sk) * Ny, nn = Math.hypot(rx, ry);
        dx = rx / nn; dy = ry / nn;
        if (!tir) inside = ent ? j : -1;
        continue;
      }
      const sc = dx * hw.ny - dy * hw.nx, [rx, ry] = Pl.reflect(dx, dy, hw.nx, hw.ny);
      return { st: [x, y, rx, ry], s: hw.s, sc, k: hw.k, nx: hw.nx, ny: hw.ny, Lo };
    }
    return null;
  }
  /* finite-time Lyapunov exponent (per hit) and MEGNO ⟨Y⟩ from a twin renormalised at every wall hit: offset along the wall
     (units of √area) and in direction; includes inclusions, ripple and facet curvature (flat facets: no stretching) */
  function ftle(mc, start, nHits, eps = 1e-8) {
    const ell = Math.sqrt(Math.abs(mc.cell.area));
    let r = step(mc, start);
    if (!r) return { lam: 0, Y: 0, nHits: 0 };
    let a = Math.SQRT1_2, b = Math.SQRT1_2, acc = 0, Sk = 0, Yb = 0, n = 0;
    for (let j = 0; j < nHits; j++) {
      const [x, y, dx, dy] = r.st, tx = -r.ny, ty = r.nx, co = Math.cos(eps * b), si = Math.sin(eps * b);
      const r1 = step(mc, r.st);
      if (!r1) break;
      const r2 = step(mc, [x + eps * ell * a * tx, y + eps * ell * a * ty, co * dx - si * dy, si * dx + co * dy]);
      let g = 0;
      if (r2) {
        const ux = -r1.ny, uy = r1.nx;
        const dp = ((r2.st[0] - r1.st[0]) * ux + (r2.st[1] - r1.st[1]) * uy) / (eps * ell);
        const da = Math.atan2(r1.st[2] * r2.st[3] - r1.st[3] * r2.st[2], r1.st[2] * r2.st[2] + r1.st[3] * r2.st[3]) / eps;
        const nn = Math.hypot(dp, da);
        if (nn > 0 && Number.isFinite(nn)) { g = Math.log(nn); a = dp / nn; b = da / nn; }
      }
      n++; acc += g; Sk += n * g; Yb += 2 * Sk / n;
      r = r1;
    }
    const k = Math.max(n, 1);
    return { lam: acc / k, Y: Yb / k, nHits: n };
  }
  const median = a => { if (!a.length) return NaN; const b = Float64Array.from(a).sort(), m = b.length >> 1; return b.length % 2 ? b[m] : 0.5 * (b[m - 1] + b[m]); };
  /* chaotic share of the launch beam and of a section grid (input mirror for a Herriott cell, the whole wall for a circle) */
  function chaosMap(mc, theta, q0, lamM, { nHits = 300, nGrid = 16, nSide = 8, Ychaos = 4, minHits = 30 } = {}) {
    const c = mc.cell, sIn = c.sIn, [w, Rw] = G.qWidth(q0, lamM), thw = lamM / (Math.PI * w), z = G.quantiles(nSide), cosT = Math.max(Math.cos(theta), 1e-6);
    const launch = [];
    for (let i = 0; i < nSide; i++) for (let j = 0; j < nSide; j++) {
      const off = w / 2 * z[i], ang = thw / 2 * z[j] + (Number.isFinite(Rw) ? off / Rw : 0), s = sIn + off / cosT;
      launch.push(Object.assign(ftle(mc, Pl.launch(c, s, theta + ang), nHits), { s, p: Math.sin(theta + ang) }));
    }
    let nst = 0, nch = 0;
    const lams = [];
    for (const t of launch) if (t.nHits >= minHits) { nst++; lams.push(t.lam); if (t.Y > Ychaos) nch++; }
    const out = { f_launch: nch / launch.length, esc_launch: (launch.length - nst) / launch.length, Y_launch: median(launch.map(t => t.Y)),
      lam_ftle: nst ? median(lams) : NaN, launch, grid: null, sea: NaN, island: NaN };
    if (nGrid) {
      const m = c.meta, sg = [], pg = [];
      let pm, sLo, sHi;
      if (m.kind === "herriott") { const k = c.labels.indexOf("M1"); sLo = c.s0[k]; sHi = c.s0[k] + c.L[k]; pm = Math.min(0.96, (m.walls ? 4 : 2.5) * m.aperture / m.d); }
      else { sLo = 0; sHi = c.perimeter; pm = 0.95; }
      for (let i = 0; i < nGrid; i++) { sg.push(sLo + (sHi - sLo) * (i + 0.5) / nGrid); pg.push(pm * (2 * (i + 0.5) / nGrid - 1)); }
      const Y = [], nH = [];
      let gs = 0, gc = 0;
      for (let ip = 0; ip < nGrid; ip++) {
        const yr = [], hr = [];
        for (let is = 0; is < nGrid; is++) {
          const t = ftle(mc, Pl.launch(c, sg[is], Math.asin(pg[ip])), nHits);
          yr.push(t.Y); hr.push(t.nHits);
          if (t.nHits >= minHits) { gs++; if (t.Y > Ychaos) gc++; }
        }
        Y.push(yr); nH.push(hr);
      }
      const N2 = nGrid * nGrid;
      Object.assign(out, { sea: gc / N2, island: (gs - gc) / N2, grid: { s: sg, p: pg, Y, n_hits: nH, pmax: pm, sLo, sHi } });
    }
    return out;
  }
  function twin(mc, sIn, theta, nHits = 300, dtheta = 1e-9) {
    const c = mc.cell, a = G.marchRay(mc, Pl.launch(c, sIn, theta), nHits), b = G.marchRay(mc, Pl.launch(c, sIn, theta + dtheta), nHits);
    const n = Math.min(a.nWalls, b.nWalls), sep = [];
    for (let j = 0; j < n; j++) sep.push(Math.hypot(a.WX[j] - b.WX[j], a.WY[j] - b.WY[j]));
    return { sep, fit: Pl.lyapunovFit(sep, Math.sqrt(Math.abs(c.area))) };
  }
  function poincare(mc, { nS = 36, nP = 12, nHits = 120 } = {}) {
    const c = mc.cell, m = c.meta, S = [], C = [];
    const pmax = m.kind === "herriott" && !m.walls ? Math.min(0.96, 2.5 * 2 * m.aperture / m.d) : 0.96;
    for (let i = 0; i < nS; i++) for (let j = 0; j < nP; j++) {
      const s = (i + 0.5) / nS * c.perimeter, p = pmax * (-1 + 2 * (j + 0.5) / nP);
      const tr = G.marchRay(mc, Pl.launch(c, s, Math.asin(p)), nHits);
      for (let k = 0; k < tr.nWalls; k++) { S.push(tr.WS[k] / c.perimeter); C.push(tr.WC[k]); }
    }
    return { S: Float32Array.from(S), C: Float32Array.from(C), pmax };
  }

  /* ---------------------------------------------------------------- dither waveforms and kernels */
  /* time-uniform samples of a unit waveform */
  const wave = (n, form) => Float64Array.from({ length: n }, (_, i) => { const t = (i + 0.5) / n; return form === "sine" ? Math.sin(TAU * t) : 1 - 4 * Math.abs(t - 0.5); });
  /* signed time average of cos(x s(t)): J0(x) for a sine, sinc(x) for a triangle */
  function kernel(x, form) {
    x = Math.abs(x);
    if (form !== "sine") return x < 1e-8 ? 1 : Math.sin(x) / x;
    if (x > 60) return Math.sqrt(2 / (Math.PI * x)) * ((1 - 9 / (128 * x * x)) * Math.cos(x - Math.PI / 4) + 1 / (8 * x) * Math.sin(x - Math.PI / 4));
    const n = Math.ceil(x / 2) + 24;
    let s = 0;
    for (let j = 0; j < n; j++) s += Math.cos(x * Math.sin(Math.PI * (j + 0.5) / n));
    return s / n;
  }

  /* in-place radix-2 FFT, forward (e^{-2πi mk/n}) */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -TAU / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < h; k++) {
          const a = i + k, b = a + h, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  const pow2 = n => { let m = 1; while (m < n) m <<= 1; return m; };
  /* harmonics c_n of e^{-i x s(τ)} over one dither period (s: unit waveform), n = -nmax..nmax, as [re, im] arrays */
  function waveHarmonics(x, form, Nl, nmax) {
    const re = new Float64Array(Nl), im = new Float64Array(Nl), s = wave(Nl, form);
    for (let l = 0; l < Nl; l++) { re[l] = Math.cos(x * s[l]); im[l] = -Math.sin(x * s[l]); }
    fft(re, im);
    const out = [new Float64Array(2 * nmax + 1), new Float64Array(2 * nmax + 1)];
    for (let n = -nmax; n <= nmax; n++) {
      const k = (n + Nl) % Nl, ph = -Math.PI * n / Nl, cr = Math.cos(ph), ci = Math.sin(ph), r = re[k] / Nl, i = im[k] / Nl;  /* samples sit at τ = (l + ½)/Nl */
      out[0][n + nmax] = r * cr - i * ci; out[1][n + nmax] = r * ci + i * cr;
    }
    return out;
  }

  /* the exit of each launch angle: optical path / n0 and path id at the output port, hits; NaN if the ray does not
     reach the output within nB hits (it leaks or stays). Only the output port ends the path: a single ray of a regular
     circular cell would otherwise often return to the input window; the light lost there is counted in T, from the
     ray bundle across the beam */
  function exitPaths(mc, thetas, nB) {
    const c = mc.cell, pw = c.meta.portW, n = thetas.length, L = new Float64Array(n).fill(NaN), id = new Float64Array(n).fill(-1), hits = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const r = G.marchRay(mc, Pl.launch(c, c.sIn, thetas[i]), nB, c.sIn, c.sOut, pw, "out");
      if (r.exit === 1) { const k = r.nWalls - 1; L[i] = r.WL[k]; id[i] = r.ID[k]; hits[i] = r.nWalls; }
    }
    return { L, id, hits };
  }

  /* Fringe of the light leaving the output port against a stray copy, as a spectrum. At each launch angle θ of the
     dither the ray is followed to the output port, whichever pass that is, and carries the field amplitude R^((n-1)/2)
     of its n hits; the stray copy follows the wavelength but not the angle, with optical path difference
     D(θ) = OPL_exit(θ) − (1 − stray) OPL_exit(θ0). The term a e^{i k(t) D(θ(t))} under an angle dither at f_θ and a
     wavelength dither at f_λ (whole Hz) is periodic in both: an FFT over Na angle samples and the harmonics of the
     wavelength waveform at depth x = k0 D Δλ/λ0 (per sample, binned in x) give its components at m f_θ + n f_λ, equal
     frequencies summed, normalised by the mean amplitude (V = 1 without dither). The 0 Hz component is the static
     floor; a boxcar of length T weights the others by sinc(Ω T / 2). Four scenarios: no dither, angle only, spectral
     only, both. */
  function fringeSpectrum(mc, theta, o) {
    const lam = o.wavelength, k0 = TAU / lam, A = o.dither, Na = A > 0 ? pow2(o.n_samp) : 1, R = o.R_mirror;
    const fa = Math.max(1, Math.round(o.f_theta)), fl = Math.max(1, Math.round(o.f_lambda));
    const nB = Math.min(3000, Math.ceil(Math.log(1e-3) / Math.log(Math.min(R, 0.99999))));
    const c0 = exitPaths(mc, [theta], nB), s = wave(Na, o.waveform), th = Array.from(s, v => theta + A * v);
    const tr = Na > 1 ? exitPaths(mc, th, nB) : c0, amp = Float64Array.from(tr.hits, (h, i) => (h ? R ** ((h - 1) / 2) * Math.exp(-0.5 * (o.alpha_m || 0) * tr.L[i]) : 0));
    let aSum = 0, nSum = 0, lSum = 0;
    for (let i = 0; i < Na; i++) { aSum += amp[i]; nSum += amp[i] * tr.hits[i]; if (amp[i]) lSum += amp[i] * tr.L[i]; }
    if (!(aSum > 0)) throw new Error("no light of the dithered beam reaches the output port");
    const aMean = aSum / Na, nMean = nSum / aSum;
    const base = (Number.isFinite(c0.L[0]) ? c0.L[0] : lSum / aSum) * mc.n0, Lref = (1 - o.stray) * base;
    const zr = new Float64Array(Na), zi = new Float64Array(Na), xs = new Float64Array(Na), Ds = new Float64Array(Na);
    let lost = 0;
    for (let i = 0; i < Na; i++) {
      if (!amp[i]) { lost++; continue; }
      const D = tr.L[i] * mc.n0 - Lref, ph = k0 * D, a = amp[i] / aMean;
      zr[i] = a * Math.cos(ph); zi[i] = a * Math.sin(ph); xs[i] = k0 * D * o.dlam / lam; Ds[i] = D;
    }
    /* coherence of the exit phase over each sample's dither interval Δθ_i (the angle it stands for). Same exit path at
       θ_i + Δθ_i: the phase is smooth, the interval integral is Filon's, weight sinc(Δφ/2) at the midpoint phase. Path
       changes inside: the phase is scrambled beyond a coherence width h (largest probe step, bisected on a log scale
       down to 1e-13 rad, with the same exit path and |Δφ| < 1); the interval then holds Δθ/h random phasors of width h
       and contributes a random term of rms √(h/Δθ). Those terms set the sampling floor. */
    const wr = new Float64Array(Na), wi = new Float64Array(Na), H = new Float64Array(Na), dth = new Float64Array(Na), fScr = new Float64Array(Na);
    let scr = [], nScr = 0, nRes = 0;
    if (Na > 1) {
      for (let i = 0; i < Na; i++) {
        const t = (i + 0.5) / Na, dsdt = o.waveform === "sine" ? TAU * Math.abs(Math.cos(TAU * t)) : 4;
        dth[i] = Math.max(A * dsdt / Na, A * (TAU / Na) ** 2 / 2);
      }
      const pr = exitPaths(mc, Array.from(th, (v, i) => v + dth[i]), nB);
      for (let i = 0; i < Na; i++) {
        if (!amp[i]) continue;
        if (Number.isFinite(pr.L[i]) && pr.id[i] === tr.id[i]) {
          const dp = k0 * mc.n0 * (pr.L[i] - tr.L[i]), x = 0.5 * dp, sc = Math.abs(x) < 1e-8 ? 1 : Math.sin(x) / x;
          wr[i] = sc * Math.cos(x); wi[i] = sc * Math.sin(x); nRes++;
        } else scr.push(i);
      }
      const coherent = (i, L2, id2) => Number.isFinite(L2) && id2 === tr.id[i] && Math.abs(k0 * mc.n0 * (L2 - tr.L[i])) < 1;
      const lo = new Float64Array(Na).fill(-13), hi = new Float64Array(Na);
      for (const i of scr) hi[i] = Math.log10(dth[i]);
      /* bisect at most 1024 scrambled samples (evenly strided); the others take the coherent fraction log10(h/Δθ) of
         their nearest probed neighbour along the dither */
      const stride = Math.max(1, Math.ceil(scr.length / 1024)), probe = scr.filter((_, q) => q % stride === 0);
      for (let it = 0; it < 7 && probe.length; it++) {             /* invariant: coherent at 10^lo, not at 10^hi */
        const mid = probe.map(i => 0.5 * (lo[i] + hi[i])), r = exitPaths(mc, probe.map((i, q) => th[i] + 10 ** mid[q]), nB);
        probe.forEach((i, q) => { if (coherent(i, r.L[q], r.id[q])) lo[i] = mid[q]; else hi[i] = mid[q]; });
      }
      if (stride > 1) scr.forEach((i, q) => {
        if (q % stride === 0) return;
        const j = probe[Math.min(Math.round(q / stride), probe.length - 1)];
        lo[i] = Math.min(Math.log10(dth[i]) + lo[j] - Math.log10(dth[j]), Math.log10(dth[i]));
      });
      for (const i of scr) { H[i] = 10 ** lo[i]; const f = Math.min(H[i] / dth[i], 1); wr[i] = Math.sqrt(f); fScr[i] = f; nScr++; }
    }
    /* wavelength-dither harmonics c_n(x) per sample, from a table over x bins (width 0.02, coarsened to keep the table
       below 3e6 entries); the centre ray's x0 exactly */
    const D0 = base - Lref, x0 = k0 * D0 * o.dlam / lam, dl = o.dlam > 0;
    let xm = Math.abs(x0);
    for (let i = 0; i < Na; i++) if (amp[i]) xm = Math.max(xm, Math.abs(xs[i]));
    const nmax = dl ? Math.ceil(xm + 10 * Math.cbrt(xm + 1) + 8) : 0, Nl = dl ? Math.min(pow2(4 * (nmax + 8)), 32768) : 1, nmx = Math.min(nmax, Nl / 2 - 1);
    const binOf = new Int32Array(Na), tab = [];
    let bw = 0.02;
    if (dl && Na > 1) {
      let keys;
      for (;;) { keys = new Set(); for (let i = 0; i < Na; i++) if (amp[i]) keys.add(Math.round(xs[i] / bw)); if (keys.size * (2 * nmx + 1) <= 3e6) break; bw *= 2; }
      const idx = new Map();
      for (const k of keys) { idx.set(k, tab.length); tab.push(waveHarmonics(k * bw, o.lwave, Nl, nmx)); }
      for (let i = 0; i < Na; i++) if (amp[i]) binOf[i] = idx.get(Math.round(xs[i] / bw));
    }
    const tab0 = dl ? waveHarmonics(x0, o.lwave, Nl, nmx) : null;
    const collect = (useAngle, useLam) => {
      /* components up to 5 kHz summed exactly (coincident frequencies, boxcar nulls); above that their power goes straight
         into bins of 1/200 decade per sign (deep wavelength dithers give millions of components) */
      const comp = new Map(), bins = new Map();
      let tot = 0;
      const add = (F, r, i) => {
        if (Math.abs(F) <= 5000) { const c = comp.get(F); if (c) { c[0] += r; c[1] += i; } else comp.set(F, [r, i]); return; }
        const p = r * r + i * i;
        if (p < 1e-18) { tot += p; return; }
        const key = Math.sign(F) * Math.floor(200 * Math.log10(Math.abs(F))), b = bins.get(key);
        if (b) { b[0] += p; b[1] += p * F; } else bins.set(key, [p, p * F]);
      };
      let sf = 0;                                       /* sampling floor of this scenario: Σ f |y|² over the scrambled samples */
      const ns = useLam ? nmx : 0;
      for (let n = -ns; n <= ns; n++) {
        if (!useAngle) {
          const br = useLam ? tab0[0][n + nmx] : n ? 0 : 1, bi = useLam ? tab0[1][n + nmx] : 0, pr = Math.cos(k0 * D0), pi = Math.sin(k0 * D0);
          add(n * fl, pr * br - pi * bi, pr * bi + pi * br);
          continue;
        }
        /* point samples for the fast harmonics (their power is the instantaneous fringe); coherence-weighted samples for
           the angle harmonics |m| ≤ 8, where the interval integral matters (the 0 Hz static floor first of all) */
        const yr = new Float64Array(Na), yi = new Float64Array(Na), ur = new Float64Array(Na), ui = new Float64Array(Na);
        for (let i = 0; i < Na; i++) {
          if (!amp[i]) continue;
          const t = useLam ? tab[binOf[i]] : null, br = t ? t[0][n + nmx] : n ? 0 : 1, bi = t ? t[1][n + nmx] : 0;
          yr[i] = zr[i] * br - zi[i] * bi; yi[i] = zr[i] * bi + zi[i] * br;
          ur[i] = yr[i] * wr[i] - yi[i] * wi[i]; ui[i] = yr[i] * wi[i] + yi[i] * wr[i];
          if (n === 0) sf += fScr[i] * (yr[i] * yr[i] + yi[i] * yi[i]);
        }
        if (Na > 1) { fft(yr, yi); fft(ur, ui); }
        for (let k = 0; k < Na; k++) {
          const m = k < Na / 2 ? k : k - Na, ph = -Math.PI * m / Na, cr = Math.cos(ph), ci = Math.sin(ph), low = Na > 1 && Math.abs(m) <= 8;
          const r = (low ? ur[k] : yr[k]) / Na, ii = (low ? ui[k] : yi[k]) / Na;
          add(m * fa + n * fl, r * cr - ii * ci, r * ci + ii * cr);
        }
      }
      const F = [], P = [];
      for (const [f, [r, i]] of comp) { const p = r * r + i * i; tot += p; if (p >= 1e-16) { F.push(f); P.push(p); } }
      for (const [p, pf] of bins.values()) { tot += p; F.push(pf / p); P.push(p); }
      const c0c = comp.get(0);
      return { F, P, floor: c0c ? Math.hypot(c0c[0], c0c[1]) : 0, total: tot, sfloor: Math.sqrt(sf) / Na };
    };
    const useA = Na > 1, spectral = dl ? collect(false, true) : null, angle = useA ? collect(true, false) : null, both = useA || dl ? collect(useA, dl) : null;
    /* static floor (c00 term) against the wavelength-dither amplitude, 0.01 pm to 1000 pm (32 points per decade), from the same samples:
       spectral only |K(x0)|, angle + spectral |Σ z w K(x_i)|/Na with its sampling floor, K = sinc (triangle) or J0 (sine) */
    const grid = Array.from({ length: 161 }, (_, j) => 1e-14 * 10 ** (j / 32)), stat = { dl: grid, spec: [], both: [], floor: [] };
    for (const g of grid) {
      const K0 = kernel(k0 * D0 * g / lam, o.lwave);
      stat.spec.push(Math.abs(K0));
      if (!useA) { stat.both.push(Math.abs(K0)); stat.floor.push(0); continue; }
      let re = 0, im = 0, sfg = 0;
      for (let i = 0; i < Na; i++) {
        if (!amp[i]) continue;
        const K = kernel(k0 * Ds[i] * g / lam, o.lwave), yr = zr[i] * K, yi = zi[i] * K;
        re += yr * wr[i] - yi * wi[i]; im += yr * wi[i] + yi * wr[i]; sfg += fScr[i] * (yr * yr + yi * yi);
      }
      stat.both.push(Math.hypot(re, im) / Na); stat.floor.push(Math.sqrt(sfg) / Na);
    }
    /* sampling floor of each scenario: √(Σ f |y|²)/Na over the scrambled samples (y with the wavelength kernel for both) */
    return { none: { F: [0], P: [1], floor: 1, total: 1 }, angle, spectral, both, OPD: D0, OPL0: base, x_lambda: x0, Na, lost: lost / Na,
      sampFloor: angle ? angle.sfloor : 0, sampFloorBoth: useA && both ? both.sfloor : 0, n_mean: nMean, x_bin: bw, stat,
      smooth: useA ? nRes / Na : 1, scrambled: useA ? nScr / Na : 0, cohWidth: nScr ? Math.exp(Array.from(H).filter(v => v > 0).reduce((a, v) => a + Math.log(v), 0) / nScr) : NaN, fa, fl };
  }
  /* rms residual fringe after a boxcar of length T (exact, or lobe-averaged envelope), with a fringe drift ω_d (rad/s) */
  function residual(spec, T, drift = 0, envelope = false) {
    let s = 0;
    for (let i = 0; i < spec.F.length; i++) {
      const z = 0.5 * (TAU * spec.F[i] + drift) * T;
      const H = envelope ? 1 / Math.sqrt(1 + 2 * z * z) : Math.abs(z) < 1e-12 ? 1 : Math.sin(z) / z;
      s += spec.P[i] * H * H;
    }
    return Math.sqrt(s);
  }

  /* ---------------------------------------------------------------- speckle: exit field on phase-space channels */
  /* frozen-Gaussian launch on a fixed (offset u, angle v) grid around the launch, traced once; a launch-angle dither δ only
     reweights the beamlets, W(u, v; δ) = <g_{u,p}|E_in(δ)> with frozen width w (the input radius) and the input wavefront */
  function speckleTrace(mc, theta, q0, lamM, o, aMax) {
    const c = mc.cell, n0 = mc.n0, k0 = TAU / o.wavelength, [w, Rw] = G.qWidth(q0, lamM), thw = lamM / (Math.PI * w);
    const nU = o.n_u, nV = o.n_v, uMax = Math.min(3 * w, 0.5 * c.meta.portW * Math.max(Math.cos(theta), 1e-6)), vMax = 4.5 * thw + aMax, cosT = Math.max(Math.cos(theta), 1e-6);
    const pw = c.meta.portW, Pm = c.perimeter, nB = Math.min(6000, Math.ceil(Math.log(1e-4) / Math.log(Math.min(o.R_mirror, 0.99999))));
    const gam = 2 / (w * w), ar = gam / 2 + 1 / (w * w), ai = Number.isFinite(Rw) ? -k0 * n0 / (2 * Rw) : 0;
    const ad = 4 * (ar * ar + ai * ai), iar = 4 * ar / ad, iai = -4 * ai / ad;            /* 1 / (4a) */
    const dy = Math.sqrt(Math.PI) * w, dp = o.wavelength / dy;                           /* one mode per channel: Δy Δp = λ0 */
    const rays = { u: [], v: [], opl: [], lg: [], amp: [], y: [], p: [], ch: [], e1r: [], e1i: [], e2r: [], e2i: [], hits: [] };
    const keyMap = new Map(), keys = [];
    let launched = 0, leaked = 0, back = 0;
    for (let iu = 0; iu < nU; iu++) for (let iv = 0; iv < nV; iv++) {
      const u = nU > 1 ? -uMax + 2 * uMax * iu / (nU - 1) : 0, v = nV > 1 ? -vMax + 2 * vMax * iv / (nV - 1) : 0;
      launched++;
      const r = G.marchRay(mc, Pl.launch(c, c.sIn + u / cosT, theta + v), nB, c.sIn, c.sOut, pw, "both");
      if (r.exit !== 1) { if (r.exit === -2) leaked++; else if (r.exit === 0) back++; continue; }
      const nh = r.nWalls, sc = r.WC[nh - 1], ds = (((r.WS[nh - 1] - c.sOut + Pm / 2) % Pm) + Pm) % Pm - Pm / 2;
      const yy = ds * Math.sqrt(Math.max(1 - sc * sc, 0)), pp = n0 * sc;
      const key = `${Math.floor(yy / dy)},${Math.floor(pp / dp)}`;
      let ch = keyMap.get(key);
      if (ch === undefined) { ch = keys.length; keyMap.set(key, ch); keys.push([Math.floor(yy / dy), Math.floor(pp / dp)]); }
      /* exponent of W: (β + i k0 p0)²/(4a) + c0 with β = γu − i k0 p, c0 = −γu²/2 + i k0 p u; split as e1 + p0 e2 + p0² e3 */
      const p = n0 * Math.sin(v), br = gam * u, bi = -k0 * p;
      const b2r = br * br - bi * bi, b2i = 2 * br * bi;
      rays.e1r.push(b2r * iar - b2i * iai - gam * u * u / 2); rays.e1i.push(b2r * iai + b2i * iar + k0 * p * u);
      const tr = -2 * k0 * bi, ti = 2 * k0 * br;                                           /* 2 i k0 β */
      rays.e2r.push(tr * iar - ti * iai); rays.e2i.push(tr * iai + ti * iar);
      rays.u.push(u); rays.v.push(v); rays.opl.push(n0 * r.WL[nh - 1]); rays.lg.push(r.WG[nh - 1]); rays.amp.push(o.R_mirror ** (nh - 1));
      rays.y.push(yy); rays.p.push(pp); rays.ch.push(ch); rays.hits.push(nh);
    }
    return { rays, keys, nCh: keys.length, e3r: -k0 * k0 * iar, e3i: -k0 * k0 * iai, w, thw, dy, dp, launched, leaked, back, k0, n0 };
  }
  /* beamlet weights for launch dither δ, times the mirror losses: [re, im] */
  function weights(sp, delta) {
    const R = sp.rays, J = R.u.length, p0 = sp.n0 * Math.sin(delta), Wr = new Float64Array(J), Wi = new Float64Array(J);
    const fr = p0 * p0 * sp.e3r, fi = p0 * p0 * sp.e3i;
    for (let j = 0; j < J; j++) {
      const er = R.e1r[j] + p0 * R.e2r[j] + fr, ei = R.e1i[j] + p0 * R.e2i[j] + fi, m = er > -700 ? Math.exp(er) * R.amp[j] : 0;
      Wr[j] = m * Math.cos(ei); Wi[j] = m * Math.sin(ei);
    }
    return [Wr, Wi];
  }
  function phases(sp, k) {
    const R = sp.rays, J = R.u.length, Pr = new Float64Array(J), Pi = new Float64Array(J);
    for (let j = 0; j < J; j++) { const ph = k * R.opl[j]; Pr[j] = Math.cos(ph); Pi[j] = Math.sin(ph); }
    return [Pr, Pi];
  }
  function field(sp, W, P) {
    const R = sp.rays, J = R.u.length, Er = new Float64Array(sp.nCh), Ei = new Float64Array(sp.nCh);
    for (let j = 0; j < J; j++) { const c = R.ch[j]; Er[c] += W[0][j] * P[0][j] - W[1][j] * P[1][j]; Ei[c] += W[0][j] * P[1][j] + W[1][j] * P[0][j]; }
    return [Er, Ei];
  }
  /* effective number of independent speckle patterns over a set of field states: (tr G)² / ‖G‖²_F, G = Σ_t E_t E_t^† */
  function nEff(states, nCh) {
    const T = states.length;
    let tr = 0, fro = 0;
    if (T <= nCh) {
      const gr = new Float64Array(T * T), gi = new Float64Array(T * T);
      for (let a = 0; a < T; a++) for (let b = a; b < T; b++) {
        const [ar, ai] = states[a], [br, bi] = states[b];
        let sr = 0, si = 0;
        for (let c = 0; c < nCh; c++) { sr += ar[c] * br[c] + ai[c] * bi[c]; si += ai[c] * br[c] - ar[c] * bi[c]; }
        gr[a * T + b] = sr; gi[a * T + b] = si;
      }
      for (let a = 0; a < T; a++) { tr += gr[a * T + a]; fro += gr[a * T + a] ** 2; for (let b = a + 1; b < T; b++) fro += 2 * (gr[a * T + b] ** 2 + gi[a * T + b] ** 2); }
    } else {
      const N = nCh, gr = new Float64Array(N * N), gi = new Float64Array(N * N);
      for (const [er, ei] of states) for (let a = 0; a < N; a++) {
        const xr = er[a], xi = ei[a];
        if (!xr && !xi) continue;
        for (let b = a; b < N; b++) { gr[a * N + b] += xr * er[b] + xi * ei[b]; gi[a * N + b] += xi * er[b] - xr * ei[b]; }
      }
      for (let a = 0; a < N; a++) { tr += gr[a * N + a]; fro += gr[a * N + a] ** 2; for (let b = a + 1; b < N; b++) fro += 2 * (gr[a * N + b] ** 2 + gi[a * N + b] ** 2); }
    }
    return fro > 0 ? tr * tr / fro : NaN;
  }
  function statesFor(sp, A, dl, nA, nL, formA, formL, lam) {
    const sa = A > 0 ? wave(nA, formA) : [0], sl = dl > 0 ? wave(nL, formL) : [0];
    const Ws = Array.from(sa, v => weights(sp, A * v)), Ps = Array.from(sl, v => phases(sp, TAU / (lam + dl * v))), out = [];
    for (const W of Ws) for (const P of Ps) out.push(field(sp, W, P));
    return out;
  }
  const overlap = (a, b) => {
    let sr = 0, si = 0, na = 0, nb = 0;
    for (let c = 0; c < a[0].length; c++) { sr += a[0][c] * b[0][c] + a[1][c] * b[1][c]; si += a[1][c] * b[0][c] - a[0][c] * b[1][c]; na += a[0][c] ** 2 + a[1][c] ** 2; nb += b[0][c] ** 2 + b[1][c] ** 2; }
    return na > 0 && nb > 0 ? (sr * sr + si * si) / (na * nb) : NaN;
  };
  const firstBelow = (xs, ys, lev) => { for (let i = 1; i < ys.length; i++) if (ys[i] < lev) return xs[i - 1] + (xs[i] - xs[i - 1]) * (ys[i - 1] - lev) / (ys[i - 1] - ys[i]); return NaN; };

  function speckle(mc, theta, q0, lamM, o, { map = true, curves = true } = {}) {
    const lam = o.wavelength, A = o.dither, dl = o.dlam;
    const thw0 = lamM / (Math.PI * G.qWidth(q0, lamM)[0]);
    const aMap = A > 0 ? 2 * A : 4 * thw0;
    const sp = speckleTrace(mc, theta, q0, lamM, o, map ? aMap : Math.max(A, 2 * thw0));
    const R = sp.rays, J = R.u.length;
    const out = { n_ch: sp.nCh, n_exit: J, launched: sp.launched, dy: sp.dy, dp: sp.dp, thw: sp.thw };
    if (!J || !sp.nCh) return Object.assign(out, { N_eff: NaN, N_theta: NaN, N_lambda: NaN, M_spat: NaN });
    const W0 = weights(sp, 0), P0 = phases(sp, TAU / lam), E0 = field(sp, W0, P0);
    let Iin = 0, Iex = 0, sI = 0, sIL = 0, sILL = 0;
    for (let j = 0; j < J; j++) { const I = W0[0][j] ** 2 + W0[1][j] ** 2; sI += I; sIL += I * R.opl[j]; sILL += I * R.opl[j] ** 2; }
    const Lm = sIL / sI, sigL = Math.sqrt(Math.max(sILL / sI - Lm * Lm, 0));
    const I0 = Array.from(E0[0], (v, c) => v * v + E0[1][c] ** 2);
    let s1 = 0, s2 = 0; for (const v of I0) { s1 += v; s2 += v * v; }
    const nA = 24, nL = 16;
    const st = statesFor(sp, A, dl, nA, nL, o.waveform, o.lwave, lam);
    const avg = new Float64Array(sp.nCh); for (const [er, ei] of st) for (let c = 0; c < sp.nCh; c++) avg[c] += (er[c] ** 2 + ei[c] ** 2) / st.length;
    Object.assign(out, { N_eff: nEff(st, sp.nCh), N_theta: A > 0 ? nEff(statesFor(sp, A, 0, nA, 1, o.waveform, o.lwave, lam), sp.nCh) : 1,
      N_lambda: dl > 0 ? nEff(statesFor(sp, 0, dl, 1, nL, o.waveform, o.lwave, lam), sp.nCh) : 1, M_spat: s1 * s1 / s2, sigma_opl: sigL, opl_mean: Lm,
      dlam_c_est: sigL > 0 ? lam * lam / (TAU * sigL) : Infinity, keys: sp.keys, I_static: I0, I_avg: Array.from(avg) });
    if (curves) {
      const dmax = Math.max(2 * A, 4 * sp.thw, 1e-4), nd = 41, dth = [], cth = [];
      for (let i = 0; i < nd; i++) { const d = dmax * i / (nd - 1); dth.push(d); cth.push(overlap(E0, field(sp, weights(sp, Math.min(d, aMap + 4 * sp.thw)), P0))); }
      const lmax = Math.max(2 * dl, 4 * out.dlam_c_est, 1e-13), nl = 61, dls = [], cl = [], cpath = [];
      for (let i = 0; i < nl; i++) {
        const d = lmax * i / (nl - 1), k = TAU / (lam + d), dk = k - TAU / lam;
        dls.push(d); cl.push(overlap(E0, field(sp, W0, phases(sp, k))));
        let re = 0, im = 0; for (let j = 0; j < J; j++) { const I = W0[0][j] ** 2 + W0[1][j] ** 2; re += I * Math.cos(dk * R.opl[j]); im += I * Math.sin(dk * R.opl[j]); }
        cpath.push((re * re + im * im) / (sI * sI));
      }
      Object.assign(out, { corr_theta: { d: dth, c: cth }, corr_lambda: { d: dls, c: cl, path: cpath },
        dtheta_c: firstBelow(dth, cth, Math.exp(-1)), dlam_c: firstBelow(dls, cl, Math.exp(-1)) });
      const hist = { L: R.lg.slice(), W: Array.from(W0[0], (v, j) => v * v + W0[1][j] ** 2) };
      out.paths = hist;
    }
    if (map) {
      const m = o.map_n, lMap = dl > 0 ? 2 * dl : Math.max(4 * out.dlam_c_est, 1e-13), As = [], Ls = [], Nm = [];
      for (let i = 0; i < m; i++) { As.push(aMap * i / (m - 1)); Ls.push(Number.isFinite(lMap) ? lMap * i / (m - 1) : 0); }
      for (let j = 0; j < m; j++) { const row = []; for (let i = 0; i < m; i++) row.push(nEff(statesFor(sp, As[i], Ls[j], 12, 12, o.waveform, o.lwave, lam), sp.nCh)); Nm.push(row); }
      out.map = { A: As, L: Ls, N: Nm };
    }
    return out;
  }

  /* ---------------------------------------------------------------- one full analysis */
  function analyse(opts, { onStage = null, full = true, keepSpec = false, poin = false } = {}) {
    const stage = s => { if (onStage) onStage(s); };
    const S = setup(opts), { o, lamM, mc, q0 } = S, c = mc.cell, pw = c.meta.portW;
    G.setAlpha(o.alpha_m || 0);
    let theta = S.theta, rt = null;
    if (o.cell !== "circle" && o.retune) { stage("re-aiming the launch"); rt = G.retune(mc, theta, q0, lamM, { Rm: o.R_mirror, gamma: o.gamma, nEff: o.n_eff, nSide: 6 }); theta = rt.theta; }
    stage("tracing the beam");
    const nB = Math.min(6000, Math.ceil(Math.log(1e-4) / Math.log(Math.min(o.R_mirror, 0.99999))));
    const ev = G.ensemble(mc, G.beamRays(c, c.sIn, theta, q0, lamM, o.n_side), pw, c.sIn, c.sOut, nB, o.R_mirror, o.gamma, lamM);
    const cp = G.centrePath(mc, c.sIn, theta, c.sOut, pw, Math.min(nB, 3000));
    const ports = [[c.sIn, pw]].concat(Math.abs(c.sOut - c.sIn) < 1e-15 ? [] : [[c.sOut, pw]]);
    const gb = G.gaussianPath(c, cp, q0, lamM, ports, o.R_mirror);
    stage("chaos indicators");
    const tw = twin(mc, c.sIn, theta, o.n_ftle);
    const cm = chaosMap(mc, theta, q0, lamM, { nHits: o.n_ftle, nGrid: full ? o.chaos_grid : 0, nSide: 8 });
    stage("fringe under the dither");
    const fs = fringeSpectrum(mc, theta, o), fl = x => (x ? x.floor : NaN), nPass = fs.n_mean;
    stage("speckle diversity");
    const sk = speckle(mc, theta, q0, lamM, o, { map: full, curves: full });
    const alpha = SPECIES[o.species] ? SPECIES[o.species][0] : SPECIES.CH4[0];
    const Appm = ev.T > 0 ? 100 * alpha * o.gamma * ev.L_mean : 0, sigA = ev.T > 0 ? Math.sqrt(2 * 1.602176634e-19 * o.bandwidth / (o.P_in * ev.T)) : Infinity;
    const grains = o.speckle_avg && Number.isFinite(ev.M2) ? Math.max(1, ev.M2) : 1;
    const Vj = fs.both ? fs.both.floor : 1, Vdet = Vj / Math.sqrt(grains);
    const necShot = Appm > 0 ? sigA / Appm : Infinity, necFringe = Appm > 0 ? 2 * Math.sqrt(o.f_parasitic) * Vdet / Appm : Infinity;
    const res = {
      kind: o.cell, theta, T: ev.T, L_mean: ev.L_mean, L_rms: ev.L_rms, M2: ev.M2, leaked: ev.leaked, back: ev.back, trapped: ev.trapped,
      Ls: ev.L, Ws: ev.W, exit_pass: cp.exits ? cp.nHits : -1, n_pass: nPass, stretch: (cm.lam_ftle || 0) * nPass, xs: cp.xs, ys: cp.ys, w_hits: Array.from(gb.w), share: Array.from(gb.share), T_G: gb.T_G,
      lam: tw.fit ? tw.fit.lam : 0, lam_fit: !!tw.fit, sep: tw.sep, f_launch: cm.f_launch, esc_launch: cm.esc_launch, Y_launch: cm.Y_launch, lam_ftle: cm.lam_ftle,
      sea: cm.sea, island: cm.island, grid: cm.grid, launch: cm.launch.map(t => ({ s: t.s, p: t.p, Y: t.Y, n: t.nHits })),
      V_joint: Vj, V_theta: fs.angle ? fl(fs.angle) : 1, V_lambda: fs.spectral ? fl(fs.spectral) : 1, V_floor: fs.sampFloor, V_floor_both: fs.sampFloorBoth, OPD: fs.OPD, OPL_exit: fs.OPL0,
      x_lambda: fs.x_lambda, spec: full || keepSpec ? fs : null,
      grains, V_det: Vdet, A_ppm: Appm, nec_shot: necShot, nec_fringe: necFringe, nec: Math.hypot(necShot, necFringe),
      N_eff: sk.N_eff, N_theta: sk.N_theta, N_lambda: sk.N_lambda, M_spat: sk.M_spat, n_ch: sk.n_ch, n_exit: sk.n_exit, sigma_opl: sk.sigma_opl,
      dlam_c_est: sk.dlam_c_est, dtheta_c: sk.dtheta_c, dlam_c: sk.dlam_c, speckle: sk, retune: rt ? { scan: rt.scan, theta: rt.theta } : null,
      area: c.area, n_inc: mc.nInc, K_ripple: c.meta.kRipple || 0, w_in: G.qWidth(q0, lamM)[0], port: pw,
    };
    if (full || poin) { stage("phase-space sections"); res.poin = poincare(mc, { nS: o.poin_s, nP: o.poin_p, nHits: o.poin_hits }); }
    return res;
  }

  /* sweepable parameters per cell (the advanced facet / mirror ones stay available to scripts) */
  const SWEEPS = {
    circle: {
      n_inc: { values: [0, 2, 4, 8, 16, 24, 32], label: "number of inclusions", scale: 1 },
      inc_dn: { values: [0, 0.002, 0.005, 0.01, 0.015, 0.02], label: "inclusion |Δn|", scale: 1, need: { n_inc: 16 } },
      inc_radius: { values: [20e-6, 35e-6, 50e-6, 75e-6, 100e-6, 150e-6], label: "inclusion radius (µm)", scale: 1e6, need: { n_inc: 16 } },
      dither: { values: [0, 0.5e-3, 1e-3, 2e-3, 4e-3, 7e-3, 10e-3], label: "angle dither amplitude (mrad)", scale: 1e3 },
      dlam: { values: [0, 2e-12, 5e-12, 10e-12, 20e-12, 50e-12, 100e-12, 200e-12, 500e-12, 1000e-12], label: "wavelength dither amplitude (pm)", scale: 1e12, log: true },
      stray: { values: [0.01, 0.03, 0.1, 0.3, 0.6, 1], label: "stray-path difference (share of the exit path)", scale: 1, log: true },
      facet_curv: { values: [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25], label: "facet curvature / circle curvature", scale: 1, advanced: true },
      f_tilt_rms: { values: [0, 0.25e-3, 0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3], label: "facet tilt rms (mrad)", scale: 1e3, advanced: true },
    },
    herriott: {
      n_inc: { values: [0, 2, 4, 8, 16, 24, 32], label: "number of inclusions", scale: 1 },
      inc_dn: { values: [0, 0.002, 0.005, 0.01, 0.015, 0.02], label: "inclusion |Δn|", scale: 1, need: { n_inc: 16 } },
      inc_radius: { values: [20e-6, 35e-6, 50e-6, 75e-6, 100e-6, 150e-6], label: "inclusion radius (µm)", scale: 1e6, need: { n_inc: 16 } },
      dither: { values: [0, 0.5e-3, 1e-3, 2e-3, 4e-3, 7e-3, 10e-3], label: "angle dither amplitude (mrad)", scale: 1e3 },
      dlam: { values: [0, 2e-12, 5e-12, 10e-12, 20e-12, 50e-12, 100e-12, 200e-12, 500e-12, 1000e-12], label: "wavelength dither amplitude (pm)", scale: 1e12, log: true },
      stray: { values: [0.01, 0.03, 0.1, 0.3, 0.6, 1], label: "stray-path difference (share of the exit path)", scale: 1, log: true },
      ripple_a: { values: [0, 0.1e-6, 0.2e-6, 0.3e-6, 0.5e-6, 0.8e-6], label: "mirror ripple amplitude (µm)", scale: 1e6, advanced: true },
      eps4: { values: [-0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3], label: "quartic deformation ε₄", scale: 1, advanced: true },
    },
  };
  const ROW_KEYS = ["T", "L_mean", "M2", "lam", "lam_ftle", "f_launch", "Y_launch", "V_joint", "V_theta", "V_lambda", "V_floor", "V_floor_both", "V_det", "grains", "n_pass", "stretch", "N_eff", "N_theta", "N_lambda", "M_spat", "nec", "nec_shot", "exit_pass"];
  /* reference variants of a cell: the regular design (no perturbations, no index disks; flat facets) and the same cell
     with its index disks removed */
  function regular(o) {
    return Object.assign({}, o, o.cell === "circle" ? { facet_curv: 0, f_tilt_rms: 0, f_curv_rms: 0, f_sel_tilt: 0, f_sel_curv: 0 }
      : { dR1: 0, dR2: 0, tilt1: 0, tilt2: 0, eps4: 0, ripple_a: 0 }, { ctr_dn: 0, n_inc: 0 });
  }
  const noIndex = o => Object.assign({}, o, { ctr_dn: 0, n_inc: 0 });
  function sweep(param, opts, onRow) {
    const sp = SWEEPS[opts.cell === "circle" ? "circle" : "herriott"][param];
    if (!sp) throw new Error(`cannot sweep ${param}`);
    const base = Object.assign({}, DEFAULTS, opts, { n_u: Math.min(opts.n_u || 16, 16), n_v: Math.min(opts.n_v || 48, 48), n_samp: 1024, n_side: 8 });
    if (sp.need) for (const [k, v] of Object.entries(sp.need)) if (!base[k]) base[k] = v;
    const rows = [];
    for (const v of sp.values) {
      let row;
      try { const r = analyse(Object.assign({}, base, { [param]: v }), { full: false }); row = { x: v }; for (const k of ROW_KEYS) row[k] = r[k]; }
      catch (err) { row = { x: v, error: String(err.message || err) }; }
      rows.push(row); if (onRow) onRow(row);
    }
    return { param, values: sp.values, rows };
  }

  return { DEFAULTS, SWEEPS, circularCell, herriottCell, ringInclusions, inclusions, setup, step, ftle, chaosMap, twin, poincare, wave, kernel, fft, waveHarmonics, fringeSpectrum, residual,
    speckleTrace, weights, phases, field, nEff, statesFor, speckle, analyse, regular, noIndex, sweep };
})();
module.exports={PlanarEngine,GasEngine,DitherLab};
