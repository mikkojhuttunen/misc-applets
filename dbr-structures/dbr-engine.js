/* dbr-engine.js — JavaScript port of the waveguide DBR / CRIGF physics (UI-free).

   Reference implementation: math-engines/ (Python, SI units). This port exists
   because the page needs thousands of evaluations per interaction. It is checked
   against the shared vectors in math-engines/test_vectors/vectors.json by
   math-engines/tools/check_js_ports.mjs.

   Known deviation from the math-engine principles: lengths are in micrometres
   inside this port (wavelength, thicknesses, periods); propagation constants are
   in 1/µm except where a function says SI. Conversions happen at the test and UI
   boundary. Moving the port to SI is tracked as future work.
*/
/* ---------- physics engine: lengths in µm ---------- */
const TAU = 2 * Math.PI;

function sell(L, terms) { let s = 1; for (const [A, B] of terms) s += A * L / (L - B); return Math.sqrt(s); }
const MATS = {
  air:   { label: 'Air', n: () => 1 },
  sio2:  { label: 'SiO₂ (fused silica)', n: l => sell(l * l, [[0.6961663, 0.0684043 ** 2], [0.4079426, 0.1162414 ** 2], [0.8974794, 9.896161 ** 2]]) },
  si3n4: { label: 'Si₃N₄ (LPCVD)', n: l => sell(l * l, [[3.0249, 0.1353406 ** 2], [40314, 1239.842 ** 2]]) },
  lne:   { label: 'LiNbO₃ extraordinary (5% MgO)', n: l => sell(l * l, [[2.2454, 0.01242], [1.3005, 0.0513], [6.8972, 331.33]]) },
  lno:   { label: 'LiNbO₃ ordinary (5% MgO)', n: l => sell(l * l, [[2.4272, 0.01478], [1.4617, 0.05612], [9.6536, 371.216]]) },
  lte:   { label: 'LiTaO₃ extraordinary (approx. fit)', n: l => Math.sqrt(1 + 3.502 * l * l / (l * l - 0.035) - 0.025 * l * l) },
  si:    { label: 'Si (real part only)', n: l => sell(l * l, [[10.6684293, 0.301516485 ** 2], [1.54133408, 1104 ** 2]]) },
  custom:{ label: 'Custom (two-point Cauchy)' }
};

/* layer = {mat, n1, n2}; n1 at λ0, n2 at probe λ */
function nIdx(layer, lam, st) {
  if (layer.mat !== 'custom') return MATS[layer.mat].n(lam);
  const a = st.lam0, b = st.lamP, n1 = layer.n1, n2 = layer.n2;
  if (Math.abs(a - b) < 1e-6 || n1 === n2) return n1;
  const B = (n1 - n2) / (1 / (a * a) - 1 / (b * b));
  return n1 - B / (a * a) + B / (lam * lam);
}

/* asymmetric three-layer slab, mode order m; returns NaN below cut-off */
function slabNeff(lam, ns, nf, nc, d, pol, m) {
  const nlo = Math.max(ns, nc);
  if (!(d > 0) || nf <= nlo) return NaN;
  const k = TAU / lam, rs = pol === 'TM' ? nf * nf / (ns * ns) : 1, rc = pol === 'TM' ? nf * nf / (nc * nc) : 1;
  const F = ne => {
    const kx = k * Math.sqrt(Math.max(nf * nf - ne * ne, 0));
    const gs = k * Math.sqrt(Math.max(ne * ne - ns * ns, 0)), gc = k * Math.sqrt(Math.max(ne * ne - nc * nc, 0));
    return kx * d - m * Math.PI - Math.atan2(rs * gs, kx) - Math.atan2(rc * gc, kx);
  };
  let lo = nlo + 1e-12, hi = nf - 1e-12;
  if (F(lo) <= 0) return NaN;
  for (let i = 0; i < 60; i++) { const mid = 0.5 * (lo + hi); if (F(mid) > 0) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

function countModes(lam, ns, nf, nc, d, pol) {
  let c = 0; while (c < 30 && Number.isFinite(slabNeff(lam, ns, nf, nc, d, pol, c))) c++; return c;
}

/* transverse field (E_y for TE, H_y for TM); substrate x<0, core 0..d, cladding x>d */
function modeParams(lam, ns, nf, nc, d, pol, m) {
  const neff = slabNeff(lam, ns, nf, nc, d, pol, m);
  if (!Number.isFinite(neff)) return { valid: false };
  const k = TAU / lam, rs = pol === 'TM' ? nf * nf / (ns * ns) : 1, rc = pol === 'TM' ? nf * nf / (nc * nc) : 1;
  const kx = k * Math.sqrt(nf * nf - neff * neff);
  const gs = Math.max(k * Math.sqrt(neff * neff - ns * ns), 1e-9), gc = Math.max(k * Math.sqrt(neff * neff - nc * nc), 1e-9);
  const ps = Math.atan2(rs * gs, kx);
  const cs = Math.cos(ps), cc = Math.cos(kx * d - ps);
  const field = x => x < 0 ? cs * Math.exp(gs * x) : x <= d ? Math.cos(kx * x - ps) : cc * Math.exp(-gc * (x - d));
  const norm = cs * cs / (2 * gs) + cc * cc / (2 * gc) + d / 2 + (Math.sin(2 * (kx * d - ps)) + Math.sin(2 * ps)) / (4 * kx);
  return { valid: true, neff, kx, gs, gc, field, norm, d };
}

/* grating profile g(u) in [0,1], u = position within the period, 1 = unetched tooth */
function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function profileFn(st, Lam) {
  const f = st.f, type = st.profile;
  const warp = av => av <= f / 2 ? av / (2 * f) : 0.25 + (av - f / 2) / ((1 - f) / 2) * 0.25;
  if (type === 'trap') {
    const a = st.h / Math.tan(st.sw * Math.PI / 180) / Lam;
    if (a < 1e-6) return { g: u => (Math.abs(u - 0.5) < f / 2 ? 1 : 0), rect: true, f };
    return { g: u => Math.min(1, Math.max(0, (f / 2 + a / 2 - Math.abs(u - 0.5)) / a)) };
  }
  if (type === 'smooth') {
    const s = st.sig / Lam;
    if (s < 1e-6) return { g: u => (Math.abs(u - 0.5) < f / 2 ? 1 : 0), rect: true, f };
    const c = 1 / (Math.SQRT2 * s);
    return { g: u => { const v = u - 0.5; let acc = 0; for (let k = -1; k <= 1; k++) acc += 0.5 * (erf((v + k + f / 2) * c) - erf((v + k - f / 2) * c)); return Math.min(1, Math.max(0, acc)); } };
  }
  if (type === 'sine') return { g: u => 0.5 * (1 + Math.cos(TAU * warp(Math.abs(u - 0.5)))) };
  if (type === 'tri') return { g: u => 1 - 2 * warp(Math.abs(u - 0.5)) };
  if (type === 'saw') { const p = f; return { g: u => (u < p ? u / p : (1 - u) / (1 - p)) }; }
  return { g: u => (Math.abs(u - 0.5) < f / 2 ? 1 : 0), rect: true, f };
}

const MS = 1024;
/* MS samples per period: cell averages for the rectangular tooth (exact fill and Fourier coefficients to O((q/MS)²),
   midpoints would quantise f to 1/MS), midpoint values for the continuous profiles */
function sampleProfile(pf) {
  const a = new Float64Array(MS);
  if (pf.rect) { const lo = 0.5 - pf.f / 2, hi = 0.5 + pf.f / 2; for (let i = 0; i < MS; i++) a[i] = Math.max(0, Math.min(hi, (i + 1) / MS) - Math.max(lo, i / MS)) * MS; return a; }
  for (let i = 0; i < MS; i++) a[i] = pf.g((i + 0.5) / MS); return a;
}
function sliceList(pf, st) {
  if (pf.rect) return [{ w: (1 - st.f) / 2, g: 0 }, { w: st.f, g: 1 }, { w: (1 - st.f) / 2, g: 0 }];
  const Ns = 64, out = []; for (let i = 0; i < Ns; i++) out.push({ w: 1 / Ns, g: pf.g((i + 0.5) / Ns) }); return out;
}

/* neff vs local fill g (thickness t-h+h*g) at one wavelength */
function tableAt(st, lam, order, rect) {
  const ns = nIdx(st.layers.sub, lam, st), nf = nIdx(st.layers.core, lam, st), nc = nIdx(st.layers.clad, lam, st);
  const K = rect ? 1 : 24, vals = new Float64Array(K + 1), floor = Math.max(ns, nc);
  let cut = false;
  for (let i = 0; i <= K; i++) {
    const d = st.t - st.h + st.h * i / K;
    let v = d > 1e-6 ? neffFull(st, lam, ns, nf, nc, d, order) : NaN;
    if (!Number.isFinite(v)) { v = floor; cut = true; }
    vals[i] = v;
  }
  return { K, vals, cut, ns, nf, nc, hiCut: !Number.isFinite(neffFull(st, lam, ns, nf, nc, st.t, order)) };
}
/* lateral geometry: ridge of width W with side slab tSide (second effective-index step); otherwise the slab value */
function sideIndex(st, lam, ns, nf, nc) {
  const fl = Math.max(ns, nc);
  if (!(st.lat && st.lat.tSide > 1e-6)) return fl;
  const v = slabNeff(lam, ns, nf, nc, st.lat.tSide, st.pol, 0);
  return Number.isFinite(v) ? v : fl;
}
/* ridge: lateral effective-index step on the unetched ridge (thickness t), and the grating slices as a
   perturbation weighted by the lateral confinement Γ_lat (fraction of lateral power inside the ridge) */
function ridgeRef(st, lam, ns, nf, nc, order) {
  const key = lam + '|' + order + '|' + ns + '|' + nf + '|' + nc + '|' + st.t + '|' + st.lat.W + '|' + st.lat.tSide + '|' + st.pol;
  const cache = st.lat._cache || (st.lat._cache = new Map());
  if (cache.has(key)) return cache.get(key);
  const vt = slabNeff(lam, ns, nf, nc, st.t, st.pol, order), nO = sideIndex(st, lam, ns, nf, nc);
  let out = { valid: false, vt };
  if (Number.isFinite(vt) && vt > nO) {
    const m = modeParams(lam, nO, vt, nO, st.lat.W, st.pol === 'TE' ? 'TM' : 'TE', 0);
    if (m.valid) {
      const W = st.lat.W, ps = Math.atan2(m.gs, m.kx);
      const core = W / 2 + (Math.sin(2 * (m.kx * W - ps)) + Math.sin(2 * ps)) / (4 * m.kx);
      out = { valid: true, vt, Nt: m.neff, G: Math.min(1, Math.max(0, core / m.norm)), nO };
    }
  }
  if (cache.size > 4000) cache.clear();
  cache.set(key, out); return out;
}
function neffFull(st, lam, ns, nf, nc, d, order) {
  const v = slabNeff(lam, ns, nf, nc, d, st.pol, order);
  if (!(st.lat && st.lat.mode === 'ridge')) return v;
  const r = ridgeRef(st, lam, ns, nf, nc, order);
  if (!r.valid) return NaN;
  const vl = Number.isFinite(v) ? v : Math.max(ns, nc);
  return r.Nt + r.G * (vl - r.vt);
}
function tint(tab, g) { const x = g * tab.K, i = Math.min(tab.K - 1, Math.max(0, Math.floor(x))), r = x - i; return tab.vals[i] * (1 - r) + tab.vals[i + 1] * r; }
function fourier(arr, q) { let re = 0, im = 0; const n = arr.length; for (let i = 0; i < n; i++) { const ph = -TAU * q * (i + 0.5) / n; re += arr[i] * Math.cos(ph); im += arr[i] * Math.sin(ph); } return Math.hypot(re, im) / n; }
function neffProfile(tab, gS) { const a = new Float64Array(gS.length); for (let i = 0; i < gS.length; i++) a[i] = tint(tab, gS[i]); return a; }
function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

/* ---------- 1-D transfer matrix on effective indices ---------- */
function layerMat(nr, ni, k, d) {
  const xr = k * nr * d, xi = k * ni * d;
  const ch = Math.cosh(xi), sh = Math.sinh(xi), c = Math.cos(xr), s = Math.sin(xr);
  const cr = c * ch, ci = -s * sh, sr = s * ch, si = c * sh;
  const den = nr * nr + ni * ni;
  const snr = (sr * nr + si * ni) / den, sni = (si * nr - sr * ni) / den;
  const nsr = nr * sr - ni * si, nsi = nr * si + ni * sr;
  return [cr, ci, sni, -snr, nsi, -nsr, cr, ci];
}
function mmul(A, B) {
  const m = (ar, ai, br, bi) => [ar * br - ai * bi, ar * bi + ai * br];
  const out = new Array(8);
  for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
    const p = m(A[4 * r], A[4 * r + 1], B[2 * c], B[2 * c + 1]);
    const q = m(A[4 * r + 2], A[4 * r + 3], B[4 + 2 * c], B[4 + 2 * c + 1]);
    out[4 * r + 2 * c] = p[0] + q[0]; out[4 * r + 2 * c + 1] = p[1] + q[1];
  }
  return out;
}
function mpow(M, N) { let R = [1, 0, 0, 0, 0, 0, 1, 0], B = M; while (N > 0) { if (N & 1) R = mmul(R, B); B = mmul(B, B); N >>= 1; } return R; }
function rtFromM(M, n0, ns) {
  const [ar, ai, br, bi, cr, ci, dr, di] = M;
  const xr = n0 * ar + n0 * ns * br, xi = n0 * ai + n0 * ns * bi;
  const yr = cr + ns * dr, yi = ci + ns * di;
  const Dr = xr + yr, Di = xi + yi, Nr = xr - yr, Ni = xi - yi, dd = Dr * Dr + Di * Di;
  const r = [(Nr * Dr + Ni * Di) / dd, (Ni * Dr - Nr * Di) / dd];
  const t = [2 * n0 * Dr / dd, -2 * n0 * Di / dd];
  return { r, t, R: r[0] * r[0] + r[1] * r[1], T: ns / n0 * (t[0] * t[0] + t[1] * t[1]) };
}
function periodLayers(st, lam, slices, Lam, ni, rect) {
  const tab = tableAt(st, lam, 0, rect);
  return { tab, layers: slices.map(s => ({ d: s.w * Lam, nr: tint(tab, s.g), ni })) };
}
function tmmAt(st, lam, slices, Lam, N, alphaPow, rect) {
  const k = TAU / lam, ni = alphaPow / 2 / k;
  const { tab, layers } = periodLayers(st, lam, slices, Lam, ni, rect);
  let P = [1, 0, 0, 0, 0, 0, 1, 0];
  for (const L of layers) P = mmul(P, layerMat(L.nr, L.ni, k, L.d));
  const nin = tab.vals[tab.K];
  return rtFromM(mpow(P, N), nin, nin);
}

/* field |E|^2 along z through the grating, binned; incident amplitude 1 */
function fieldAlong(st, lam, slices, Lam, N, alphaPow, rect, nbins, wantAB) {
  const k = TAU / lam, ni = alphaPow / 2 / k;
  const { tab, layers } = periodLayers(st, lam, slices, Lam, ni, rect);
  let P = [1, 0, 0, 0, 0, 0, 1, 0];
  const mats = layers.map(L => layerMat(L.nr, L.ni, k, L.d));
  for (const M of mats) P = mmul(P, M);
  const nin = tab.vals[tab.K];
  const res = rtFromM(mpow(P, N), nin, nin);
  const Ltot = N * Lam, sum = new Float64Array(nbins), cnt = new Float64Array(nbins);
  let E = [res.t[0], res.t[1]], H = [nin * res.t[0], nin * res.t[1]];
  const sub = Math.max(1, Math.ceil(24 / layers.length));
  let z = Ltot;
  const AB = wantAB ? new Float64Array(4 * N) : null;
  const put = (zz, e) => { const b = Math.min(nbins - 1, Math.max(0, Math.floor(zz / Ltot * nbins))); sum[b] += e; cnt[b]++; };
  put(z, E[0] * E[0] + E[1] * E[1]);
  for (let p = N - 1; p >= 0; p--) {
    for (let j = layers.length - 1; j >= 0; j--) {
      const L = layers[j];
      for (let s = 1; s <= sub; s++) {
        const M = s === sub ? mats[j] : layerMat(L.nr, L.ni, k, L.d * s / sub);
        const e0 = M[0] * E[0] - M[1] * E[1] + M[2] * H[0] - M[3] * H[1];
        const e1 = M[0] * E[1] + M[1] * E[0] + M[2] * H[1] + M[3] * H[0];
        if (s === sub) {
          const h0 = M[4] * E[0] - M[5] * E[1] + M[6] * H[0] - M[7] * H[1];
          const h1 = M[4] * E[1] + M[5] * E[0] + M[6] * H[1] + M[7] * H[0];
          E = [e0, e1]; H = [h0, h1];
        }
        put(z - L.d * s / sub, e0 * e0 + e1 * e1);
      }
      z -= L.d;
    }
    if (AB) { const nb = layers[0].nr, i = 4 * p; AB[i] = (E[0] + H[0] / nb) / 2; AB[i + 1] = (E[1] + H[1] / nb) / 2; AB[i + 2] = (E[0] - H[0] / nb) / 2; AB[i + 3] = (E[1] - H[1] / nb) / 2; }
  }
  const I = new Float64Array(nbins);
  for (let b = 0; b < nbins; b++) I[b] = cnt[b] ? sum[b] / cnt[b] : (b ? I[b - 1] : 1 + res.R);
  return { I, R: res.R, T: res.T, nin, AB };
}

/* coupled-mode reflectance, half-detuning dh = beta - m*pi/Lam */
function cmtR(kap, dh, L) {
  const k2 = kap * kap, d2 = dh * dh;
  if (Math.abs(k2 - d2) < 1e-14) return k2 * L * L / (1 + k2 * L * L);
  if (k2 > d2) { const s = Math.sqrt(k2 - d2), sh = Math.sinh(s * L), c = Math.cosh(s * L); return k2 * sh * sh / (d2 * sh * sh + (k2 - d2) * c * c); }
  const s = Math.sqrt(d2 - k2), sn = Math.sin(s * L), c = Math.cos(s * L); return k2 * sn * sn / (d2 - k2 * c * c);
}

/* out-of-plane scattering of a guided mode by the grating (thin-sheet volume-current estimate) */
function radiation(st, lam, order, gS, Lam, N, rect) {
  const tab = tableAt(st, lam, order, rect);
  const { ns, nf, nc } = tab;
  const k = TAU / lam;
  const nP = neffProfile(tab, gS), N0 = mean(nP);
  const out = { lam, order, N0, ns, nf, nc, tab, orders: [], alphaTot: 0, valid: !tab.hiCut };
  if (!out.valid) return out;
  const beta = k * N0; out.beta = beta;
  const gbar = mean(gS);
  const mp = modeParams(lam, ns, nf, nc, st.t - st.h + st.h * gbar, st.pol, order);
  const xg = st.t - st.h / 2;
  const phi2 = mp.valid ? mp.field(xg) ** 2 / mp.norm : 0;
  const pref = k ** 4 * (nf * nf - nc * nc) ** 2 * st.h ** 2 * phi2 / (4 * beta);
  const chans = channels(st, lam), K = TAU / Lam, nmax = Math.max(ns, nc, ...chans.map(c => c[1]));
  const qmax = Math.ceil((beta + k * nmax) / K);
  for (let q = 1; q <= qmax; q++) {
    const kz = beta - q * K, G = fourier(gS, q);
    for (const [med, nj] of chans) {
      if (Math.abs(kz) >= k * nj) continue;
      const kx = Math.sqrt(k * k * nj * nj - kz * kz);
      /* slab correction by reciprocity: local field at the grating for a plane wave arriving from this medium */
      const lf = planeWave(st, lam, kz, xg, med === 'cladding' ? 'top' : 'bottom').F;
      const a = pref * G * G / kx * (lf[0] * lf[0] + lf[1] * lf[1]);
      out.orders.push({ q, med, nj, kz, theta: Math.asin(kz / (k * nj)) * 180 / Math.PI, alpha: a, G });
      out.alphaTot += a;
    }
  }
  const qB = 2 * N0 * Lam / lam, qn = Math.max(1, Math.round(qB));
  const Nq = fourier(nP, qn), kapq = TAU * Nq / lam, dh = beta - qn * Math.PI / Lam;
  out.bragg = { qB, q: qn, kap: kapq, dh, R: cmtR(kapq, dh, N * Lam), lamB: 2 * N0 * Lam / qn };
  out.mode = mp; out.hi = tab.vals[tab.K]; out.lo = tab.vals[0];
  return out;
}

/* far-field angular intensity into one medium; theta in degrees. One period as M = 128 block averages of g − ḡ */
function farField(rad, gS, Lam, N, alphaPow, nj, thetas) {
  const k = TAU / rad.lam, beta = rad.beta, M = 128, step = gS.length / M;
  const S = new Float64Array(M); let gb = 0;
  for (let i = 0; i < M; i++) { const a = Math.floor(i * step), b = Math.floor((i + 1) * step); let s = 0; for (let j = a; j < b; j++) s += gS[j]; S[i] = s / (b - a); gb += S[i]; }
  gb /= M; for (let i = 0; i < M; i++) S[i] -= gb;
  const a = Math.exp(-alphaPow * Lam / 2), aN = Math.pow(a, N);
  const out = new Float64Array(thetas.length);
  for (let j = 0; j < thetas.length; j++) {
    const D = beta - k * nj * Math.sin(thetas[j] * Math.PI / 180);
    let cr = 0, ci = 0;
    for (let i = 0; i < M; i++) { const ph = D * Lam * (i + 0.5) / M; cr += S[i] * Math.cos(ph); ci += S[i] * Math.sin(ph); }
    const ph = D * Lam;
    const nr = 1 - aN * Math.cos(N * ph), nim = -aN * Math.sin(N * ph);
    const dr = 1 - a * Math.cos(ph), dim = -a * Math.sin(ph);
    const dd = dr * dr + dim * dim;
    const af = dd < 1e-20 ? N * N : (nr * nr + nim * nim) / dd;
    out[j] = (cr * cr + ci * ci) * af;
  }
  return out;
}

/* ---------- second-harmonic generation inside the grating ----------
   Pump standing wave U(z) = A e^{iβz} + B e^{-iβz} from the transfer matrix; nonlinear core only.
   Radiated SH from the sheet K(z) = ∫J dx, guided SH from the overlap with the SH mode (coupled-mode theory).
   SI units inside; 2D powers are per unit width and multiplied by the effective width w. */
function normMode(m) { const s = 1 / Math.sqrt(m.norm); return x => m.field(x) * s; }
function overlapTables(st, lamF, lam2, sOrder, rect) {
  const K = rect ? 1 : 12, G = new Float64Array(K + 1), O = new Float64Array(K + 1);
  const nsF = nIdx(st.layers.sub, lamF, st), nfF = nIdx(st.layers.core, lamF, st), ncF = nIdx(st.layers.clad, lamF, st);
  const ns2 = nIdx(st.layers.sub, lam2, st), nf2 = nIdx(st.layers.core, lam2, st), nc2 = nIdx(st.layers.clad, lam2, st);
  for (let i = 0; i <= K; i++) {
    const d = st.t - st.h + st.h * i / K;
    const mF = modeParams(lamF, nsF, nfF, ncF, d, st.pol, 0), m2 = modeParams(lam2, ns2, nf2, nc2, d, st.pol, sOrder);
    if (!mF.valid || d <= 0) continue;
    const fF = normMode(mF), f2 = m2.valid ? normMode(m2) : null, n = 64;
    let g = 0, o = 0;
    for (let j = 0; j <= n; j++) { const x = d * j / n, wgt = (j === 0 || j === n ? 1 : j % 2 ? 4 : 2) * d / (3 * n), a = fF(x) ** 2; g += wgt * a; if (f2) o += wgt * a * f2(x); }
    G[i] = g; O[i] = o * 1e3; /* µm^-1/2 -> m^-1/2 */
  }
  return { K, G, O };
}
function shgCalc(st, lamF, fld, gS, Lam, N, rect, opt) {
  const lam2 = lamF / 2, c0 = 299792458, mu0 = 4e-7 * Math.PI, eps0 = 8.8541878128e-12;
  const om = TAU * c0 / (lamF * 1e-6), Lm = Lam * 1e-6;
  const tabF = tableAt(st, lamF, 0, rect), tab2 = tableAt(st, lam2, opt.sOrder, rect);
  const NF = mean(neffProfile(tabF, gS)), N2 = mean(neffProfile(tab2, gS));
  const kF = TAU / (lamF * 1e-6), k2 = 2 * kF, beta = kF * NF, beta2 = k2 * N2;
  const ov = overlapTables(st, lamF, lam2, opt.sOrder, rect);
  const M = 64, step = gS.length / M, Gs = new Float64Array(M), Os = new Float64Array(M);
  const ti = (arr, g) => { const x = g * ov.K, i = Math.min(ov.K - 1, Math.max(0, Math.floor(x))), r = x - i; return arr[i] * (1 - r) + arr[i + 1] * r; };
  for (let i = 0; i < M; i++) { const g = gS[Math.floor((i + 0.5) * step)]; Gs[i] = ti(ov.G, g); Os[i] = ti(ov.O, g); }
  const cell = (X, q) => { let re = 0, im = 0; for (let i = 0; i < M; i++) { const s = (i + 0.5) / M * Lm; re += X[i] * Math.cos(q * s); im += X[i] * Math.sin(q * s); } return [re * Lm / M, im * Lm / M]; };
  const Pw = opt.P / (opt.w * 1e-6);                 /* W/m */
  const AB = fld.AB, S1 = new Float64Array(2 * N), S2 = new Float64Array(2 * N), S3 = new Float64Array(2 * N);
  for (let p = 0; p < N; p++) {
    const ar = AB[4 * p], ai = AB[4 * p + 1], br = AB[4 * p + 2], bi = AB[4 * p + 3];
    S1[2 * p] = (ar * ar - ai * ai) * Pw; S1[2 * p + 1] = 2 * ar * ai * Pw;
    S2[2 * p] = (br * br - bi * bi) * Pw; S2[2 * p + 1] = 2 * br * bi * Pw;
    S3[2 * p] = 2 * (ar * br - ai * bi) * Pw; S3[2 * p + 1] = 2 * (ar * bi + ai * br) * Pw;
  }
  if (opt.pole) {
    const Lpp = TAU / Math.abs(tab2.hiCut ? 1 : (k2 * N2 - 2 * kF * NF)) * 1e6;
    for (let p = 0; p < N; p++) { const sgn = polingMean(p * Lam, Lam, Lpp, opt.duty); for (const A of [S1, S2, S3]) { A[2 * p] *= sgn; A[2 * p + 1] *= sgn; } }
  }
  const cm = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  /* Σ_p X_p e^{-i kz z_p} */
  const psum = (S, kz) => { const cr = Math.cos(kz * Lm), ci = -Math.sin(kz * Lm); let pr = 1, pi = 0, sr = 0, si = 0;
    for (let p = 0; p < N; p++) { const xr = S[2 * p], xi = S[2 * p + 1]; sr += xr * pr - xi * pi; si += xr * pi + xi * pr; const t = pr * cr - pi * ci; pi = pr * ci + pi * cr; pr = t; } return [sr, si]; };
  const source = (X, kz) => { const a = cm(cell(X, 2 * beta - kz), psum(S1, kz)), b = cm(cell(X, -2 * beta - kz), psum(S2, kz)), c = cm(cell(X, -kz), psum(S3, kz)); return [a[0] + b[0] + c[0], a[1] + b[1] + c[1]]; };
  const d = opt.d * 1e-12, Kpref = 2 * om * eps0 * d * (2 * om * mu0 / beta);     /* |K̃| = Kpref |Σ| */
  const gpref = om * eps0 * d / 2 * (2 * om * mu0 / beta) * Math.sqrt(4 * om * mu0 / beta2);
  const out = { lam2, NF, N2, beta, beta2, dk: beta2 - 2 * beta, Kg: TAU / Lm, guided2: !tab2.hiCut, Lpp: TAU / Math.abs(beta2 - 2 * beta) * 1e6 };
  if (out.guided2) {
    const f = source(Os, beta2), b = source(Os, -beta2);
    out.Pf = gpref ** 2 * (f[0] ** 2 + f[1] ** 2) * opt.w * 1e-6;
    out.Pb = gpref ** 2 * (b[0] ** 2 + b[1] ** 2) * opt.w * 1e-6;
    const Oh = ov.O[ov.K], Lt = N * Lm;
    out.Pref = (gpref * Oh * Pw * Lt) ** 2 * opt.w * 1e-6;
  }
  const n2c = nIdx(st.layers.clad, lam2, st), stk2 = stackOf(st, lam2), n2s = stk2.botN;
  const cands = [];
  for (let q = -60; q <= 60; q++) for (const base of [2 * beta, -2 * beta, 0]) cands.push(base + q * out.Kg);
  const Lt = N * Lm;
  const grid = nj => {
    const th = [], du = N > 4000 ? 0.25 : 0.1;
    for (let t = -89.95; t < 90; t += du) th.push(t);
    for (const kz of cands) { const sn = kz / (k2 * nj); if (Math.abs(sn) >= 1) continue; const t0 = Math.asin(sn) * 180 / Math.PI, wd = Math.max(lam2 * 1e-6 / (nj * Lt * Math.max(Math.cos(t0 * Math.PI / 180), 0.05)) * 180 / Math.PI, 1e-5); for (let i = -40; i <= 40; i++) { const t = t0 + i * wd / 8; if (t > -90 && t < 90) th.push(t); } }
    return Float64Array.from(th.sort((a, b) => a - b));
  };
  const far = nj => {
    const th = grid(nj), I = new Float64Array(th.length);
    const med = nj === n2c ? 'cladding' : 'substrate';
    for (let j = 0; j < th.length; j++) { const kz = k2 * nj * Math.sin(th[j] * Math.PI / 180), s = source(Gs, kz); I[j] = 2 * om * mu0 / (16 * Math.PI) * Kpref ** 2 * (s[0] ** 2 + s[1] ** 2) * opt.w * 1e-6 * shLocalFactor(st, lam2, nj, med, kz * 1e-6); }
    let P = 0; for (let j = 1; j < th.length; j++) P += 0.5 * (I[j] + I[j - 1]) * (th[j] - th[j - 1]) * Math.PI / 180;
    const peaks = [];
    let mx = 0; for (const v of I) mx = Math.max(mx, v);
    for (let j = 1; j < th.length - 1; j++) if (I[j] >= I[j - 1] && I[j] > I[j + 1] && I[j] > mx * 1e-3) {
      let a = j, b = j; while (a > 0 && I[a - 1] < I[a]) a--; while (b < th.length - 1 && I[b + 1] < I[b]) b++;
      let pw = 0; for (let k = a + 1; k <= b; k++) pw += 0.5 * (I[k] + I[k - 1]) * (th[k] - th[k - 1]) * Math.PI / 180;
      let hl = j, hr = j; while (hl > 0 && I[hl] > I[j] / 2) hl--; while (hr < th.length - 1 && I[hr] > I[j] / 2) hr++;
      peaks.push({ theta: th[j], peak: I[j], P: pw, fwhm: th[hr] - th[hl] });
    }
    peaks.sort((a, b) => b.P - a.P);
    return { th, I, P, peaks: peaks.slice(0, 6), nj };
  };
  out.up = far(n2c); out.down = stk2.botReal ? far(n2s) : { th: new Float64Array(0), I: new Float64Array(0), P: 0, peaks: [], nj: NaN };
  return out;
}

/* ---------- complex helpers ---------- */
const cx = (re, im = 0) => [re, im];
const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
const cabs2 = a => a[0] * a[0] + a[1] * a[1];
const cexpi = p => [Math.cos(p), Math.sin(p)];
const cscale = (a, s) => [a[0] * s, a[1] * s];
const csqrt = a => { const r = Math.hypot(a[0], a[1]), re = Math.sqrt((r + a[0]) / 2), im = Math.sign(a[1] || 1) * Math.sqrt(Math.max(0, (r - a[0]) / 2)); return [re, im]; };
const ccos = a => [Math.cos(a[0]) * Math.cosh(a[1]), -Math.sin(a[0]) * Math.sinh(a[1])];
const csin = a => [Math.sin(a[0]) * Math.cosh(a[1]), Math.cos(a[0]) * Math.sinh(a[1])];
const C0 = 299792458, MU0 = 4e-7 * Math.PI, EPS0 = 8.8541878128e-12, ETA0 = MU0 * C0;

/* matrix-vector on the 8-array 2x2 complex matrices of physics.js; v = [E, H] as complex pairs */
function mvec(M, E, H) {
  return [[M[0] * E[0] - M[1] * E[1] + M[2] * H[0] - M[3] * H[1], M[0] * E[1] + M[1] * E[0] + M[2] * H[1] + M[3] * H[0]],
          [M[4] * E[0] - M[5] * E[1] + M[6] * H[0] - M[7] * H[1], M[4] * E[1] + M[5] * E[0] + M[6] * H[1] + M[7] * H[0]]];
}
/* inverse of a unimodular characteristic matrix: [[d, -b], [-c, a]] */
function minv(M) { return [M[6], M[7], -M[2], -M[3], -M[4], -M[5], M[0], M[1]]; }

/* ---------- plane wave on the three-layer slab (TE), local field at depth xg ----------
   x: bottom medium x<0, core 0..t, top medium x>t. Wave incident from the top. xg measured from the bottom.
   Returns field at xg and the reflection / transmission field amplitudes. */
function slabPlaneWave(lam, nTop, nCore, nBot, t, kz, xg) {
  const k = TAU / lam;
  const kxOf = n => { const v = csqrt([k * k * n * n - kz * kz, 0]); return v[1] < 0 ? cscale(v, -1) : v; };
  const kt = kxOf(nTop), kf = kxOf(nCore), kb = kxOf(nBot);
  /* bottom: E = tau e^{-i kb x}; core: E = cos(kf x) + B sin(kf x), with tau = 1 */
  const B = cdiv(cmul([0, -1], kb), kf);
  const at = (x) => { const ph = cscale(kf, x), c = ccos(ph), s = csin(ph); return { E: cadd(c, cmul(B, s)), D: cmul(kf, csub(cmul(B, c), s)) }; };
  const top = at(t);
  const ikt = cmul([0, 1], kt);
  const tau = cdiv(cscale(ikt, 2), csub(cmul(ikt, top.E), top.D));
  const r = csub(cmul(tau, top.E), [1, 0]);
  const loc = cmul(tau, at(xg).E);
  return { F: loc, r, tau, kt, kb };
}

/* ---------- plane wave on the full vertical stack: cladding | core | BOX | handle (Si or Au) ----------
   from = 'top' (cladding side) or 'bottom' (handle or substrate side). Returns the field at the grating depth xg
   (from the core bottom) per unit incident amplitude, reflection, power-normalised transmission and the media. */
const HANDLE = { si: l => [nIdx({ mat: 'si' }, l, {}), l < 1.1 ? 0.005 : 0], au: l => (l > 1.2 ? [0.52, 10.7] : [0.15, 4.7]) };
function stackOf(st, lam) {
  const ns = nIdx(st.layers.sub, lam, st), nf = nIdx(st.layers.core, lam, st), nc = nIdx(st.layers.clad, lam, st);
  const u = st.under || { mode: 'none' };
  if (u.mode === 'none') return { layers: [{ n: [ns, 0] }, { n: [nf, 0], d: st.t, core: true }, { n: [nc, 0] }], ns, nf, nc, botN: ns, botReal: true };
  const nh = HANDLE[u.mode](lam);
  return { layers: [{ n: nh }, { n: [ns, 0], d: u.tbox }, { n: [nf, 0], d: st.t, core: true }, { n: [nc, 0] }], ns, nf, nc, botN: nh[0], botReal: u.mode !== 'au' && nh[1] < 0.05 };
}
function planeWave(st, lam, kz, xg, from) {
  const S = stackOf(st, lam), k = TAU / lam;
  let L = S.layers.slice();
  if (from === 'bottom') { if (!S.botReal) return null; L = L.reverse(); }
  const kxOf = n => { const n2 = cmul(n, n); let v = csqrt([k * k * n2[0] - kz * kz, k * k * n2[1]]); if (v[1] < 0) v = cscale(v, -1); return v; };
  const kb = kxOf(L[0].n), kt = kxOf(L[L.length - 1].n);
  let E = [1, 0], D = cmul([0, -1], kb), F = null;
  for (let i = 1; i < L.length - 1; i++) {
    const kx = kxOf(L[i].n), d = L[i].d;
    const at = x => { const ph = cscale(kx, x), c = ccos(ph), s = csin(ph); return [cadd(cmul(E, c), cmul(cdiv(D, kx), s)), csub(cmul(D, c), cmul(cmul(E, kx), s))]; };
    if (L[i].core) { const xl = from === 'bottom' ? st.t - xg : xg; F = at(xl)[0]; }
    [E, D] = at(d);
  }
  const ikt = cmul([0, 1], kt);
  const tau = cdiv(cscale(ikt, 2), csub(cmul(ikt, E), D));
  const r = csub(cmul(tau, E), [1, 0]);
  const Tn = cscale(tau, Math.sqrt(Math.max(kb[0], 0) / kt[0]));
  return { F: cmul(tau, F), r, tau: Tn, kt, kb, nIn: L[L.length - 1].n[0], nOut: L[0].n[0] };
}
/* radiation channels: cladding (up) and, unless a metal mirror closes it, the bottom medium */
function channels(st, lam) { const S = stackOf(st, lam); const ch = [['cladding', S.nc]]; if (S.botReal) ch.push(['substrate', S.botN]); return ch; }

/* ---------- periodic poling (QPM) ---------- */
/* straight unetched waveguide: Δk(λ) = β_SH(λ/2) − 2β(λ), in 1/µm */
function dkStraight(st, lam, sOrder) {
  const n = (L, l) => nIdx(st.layers[L], l, st);
  const b1 = TAU / lam * neffFull(st, lam, n('sub', lam), n('core', lam), n('clad', lam), st.t, 0);
  const l2 = lam / 2, b2 = TAU / l2 * neffFull(st, l2, n('sub', l2), n('core', l2), n('clad', l2), st.t, sOrder);
  return b2 - 2 * b1;
}
/* coupling prefactor and overlap for guided SHG in the straight guide; η = |g Ω I|² / w  [1/W] */
function qpmCoupling(st, lam, sOrder, dpm) {
  const n = (L, l) => nIdx(st.layers[L], l, st), l2 = lam / 2;
  const mF = modeParams(lam, n('sub', lam), n('core', lam), n('clad', lam), st.t, st.pol, 0);
  const m2 = modeParams(l2, n('sub', l2), n('core', l2), n('clad', l2), st.t, st.pol, sOrder);
  if (!mF.valid || !m2.valid) return null;
  const fF = normMode(mF), f2 = normMode(m2), N = 96; let o = 0;
  for (let j = 0; j <= N; j++) { const x = st.t * j / N, w = (j === 0 || j === N ? 1 : j % 2 ? 4 : 2) * st.t / (3 * N); o += w * fF(x) ** 2 * f2(x); }
  const om = TAU * C0 / (lam * 1e-6), beta = TAU / (lam * 1e-6) * mF.neff, beta2 = TAU / (l2 * 1e-6) * m2.neff;
  const g = om * EPS0 * dpm * 1e-12 / 2 * (2 * om * MU0 / beta) * Math.sqrt(4 * om * MU0 / beta2);
  return { gO: g * o * 1e3, neffF: mF.neff, neffS: m2.neff };
}
/* domain walls from a local QPM wavevector K(z) and duty D(z); returns Float64Array of walls (µm) starting at 0,
   sign of the first domain is +1 and alternates */
function buildPoling(L, K0, chirp, apod, D0) {
  const walls = [0];
  const Kz = z => K0 + chirp * (z - L / 2);
  const amp = z => { if (apod <= 0) return 1; const a = apod * L, u = Math.min(z, L - z); return u >= a ? 1 : 0.5 * (1 - Math.cos(Math.PI * Math.max(u, 0) / a)); };
  const duty = z => { const A = amp(z) * Math.sin(Math.PI * D0); return Math.asin(Math.min(1, A)) / Math.PI; };
  let z = 0, guard = 0;
  while (z < L && guard++ < 2e6) {
    const K = Math.max(Kz(z), 1e-6), P = TAU / K, D = duty(z + P / 2);
    const zb = z + Math.max(D, 0.004) * P, ze = z + P;
    if (zb < L) walls.push(zb); if (ze < L) walls.push(ze);
    z = ze;
  }
  walls.push(L);
  return Float64Array.from(walls);
}
/* overlap integral I(Δk) = ∫ s(z) e^{iΔk z} dz over the domains, in µm */
function polingIntegral(walls, dk) {
  if (Math.abs(dk) < 1e-12) { let s = 0; for (let j = 0; j < walls.length - 1; j++) s += (j % 2 ? -1 : 1) * (walls[j + 1] - walls[j]); return [s, 0]; }
  let re = 0, im = 0, sign = 1;
  for (let j = 0; j < walls.length - 1; j++) {
    const a = dk * walls[j], b = dk * walls[j + 1];
    re += sign * (Math.sin(b) - Math.sin(a)); im += sign * (Math.cos(a) - Math.cos(b));
    sign = -sign;
  }
  return [re / dk, im / dk];
}
function qpmSpectrum(walls, dks, gO, wUm) {
  const out = new Float64Array(dks.length);
  for (let i = 0; i < dks.length; i++) { const I = polingIntegral(walls, dks[i]); out[i] = gO * gO * cabs2(I) * 1e-12 / (wUm * 1e-6); }
  return out;
}
function bandMetrics(lams, eta) {
  let ip = 0; for (let i = 1; i < eta.length; i++) if (eta[i] > eta[ip]) ip = i;
  const half = eta[ip] / 2; let a = -1, b = -1;
  for (let i = 0; i < eta.length; i++) if (eta[i] >= half) { if (a < 0) a = i; b = i; }
  const edge = (i, j) => lams[i] + (half - eta[i]) / (eta[j] - eta[i]) * (lams[j] - lams[i]);
  const lo = a > 0 ? edge(a - 1, a) : lams[0], hi = b < eta.length - 1 ? edge(b + 1, b) : lams[eta.length - 1];
  const fw = hi - lo, c0 = lo + 0.15 * fw, c1 = hi - 0.15 * fw;
  let mn = Infinity, mx = 0; for (let i = 0; i < eta.length; i++) if (lams[i] >= c0 && lams[i] <= c1) { mn = Math.min(mn, eta[i]); mx = Math.max(mx, eta[i]); }
  const ripple = mx > 0 && Number.isFinite(mn) && mn > 0 ? 10 * Math.log10(mx / mn) : 0;
  return { peak: eta[ip], lamPeak: lams[ip], fwhm: fw, lo, hi, ripple, edgeHit: a <= 0 || b >= eta.length - 1 };
}
/* QPM design: uniform, or iterative chirp + apodisation search for a target FWHM (µm) */
function qpmDesign(st, opt) {
  const lamC = opt.lamC, L = opt.L, sOrder = opt.sOrder;
  const dk0 = dkStraight(st, lamC, sOrder);
  const cp = qpmCoupling(st, lamC, sOrder, opt.d);
  if (!Number.isFinite(dk0) || !cp) return { ok: false };
  const dl = 0.0005, dkdl = (dkStraight(st, lamC + dl, sOrder) - dkStraight(st, lamC - dl, sOrder)) / (2 * dl);
  const K0 = Math.abs(dk0), Lpp = TAU / K0;
  const fwU = 5.566 / (L * Math.abs(dkdl));
  const target = opt.target;
  const span = Math.min(Math.max(4 * Math.max(target || 0, fwU), 6 * fwU), 0.4 * lamC);
  const NL = 320, lams = new Float64Array(NL), dks = new Float64Array(NL);
  for (let i = 0; i < NL; i++) { lams[i] = lamC - span / 2 + span * i / (NL - 1); dks[i] = Math.sign(dk0) * dkStraight(st, lams[i], sOrder); }
  /* sign convention: poling wavevector cancels |Δk|, so use |Δk| near the centre */
  const evalD = (chirp, apod) => { const walls = buildPoling(L, K0, chirp, apod, opt.duty); const eta = qpmSpectrum(walls, dks, cp.gO, opt.w); return { walls, eta, m: bandMetrics(lams, eta), chirp, apod }; };
  const uni = evalD(0, 0);
  if (opt.fixed) { const b = evalD(opt.fixed.chirp, opt.fixed.apod); return { ok: true, dk0, dkdl, Lpp, K0, fwU, lams, uniform: uni, best: b, log: opt.fixed.log || [], gO: cp.gO, neffF: cp.neffF, neffS: cp.neffS }; }
  const res = { ok: true, dk0, dkdl, Lpp, K0, fwU, lams, uniform: uni, log: [], gO: cp.gO, neffF: cp.neffF, neffS: cp.neffS };
  if (!(target > 0) || target <= 1.1 * uni.m.fwhm) { res.best = uni; res.note = target > 0 ? 'target ≤ uniform bandwidth' : ''; return res; }
  const c0 = Math.abs(dkdl) * target / L * Math.sign(dkdl || 1);
  let best = null;
  for (const apod of opt.apods || [0, 0.15, 0.3]) {
    let s0 = 1, d0 = evalD(c0 * s0, apod), s1 = 1.3, d1 = evalD(c0 * s1, apod);
    const err = d => Math.log(d.m.fwhm / target);
    res.log.push({ apod, it: 0, scale: s0, fwhm: d0.m.fwhm, ripple: d0.m.ripple, peak: d0.m.peak });
    res.log.push({ apod, it: 1, scale: s1, fwhm: d1.m.fwhm, ripple: d1.m.ripple, peak: d1.m.peak });
    for (let it = 2; it < 10 && Math.abs(err(d1)) > 0.02; it++) {
      const e0 = err(d0), e1 = err(d1);
      let s2 = Math.abs(e1 - e0) > 1e-9 ? s1 - e1 * (s1 - s0) / (e1 - e0) : s1 * Math.exp(-e1);
      if (!(s2 > 0.05) || s2 > 20) s2 = s1 * Math.exp(-e1);
      s0 = s1; d0 = d1; s1 = s2; d1 = evalD(c0 * s1, apod);
      res.log.push({ apod, it, scale: s1, fwhm: d1.m.fwhm, ripple: d1.m.ripple, peak: d1.m.peak });
    }
    const score = d1.m.ripple + 40 * Math.abs(err(d1));
    if (!best || score < best.score) best = { ...d1, score };
  }
  res.best = best;
  return res;
}
/* sign of the poling at z for a uniform device grating; returns mean of s over [z, z+dz] */
function polingMean(z, dz, Lpp, duty) {
  if (!(Lpp > 0)) return 1;
  const F = x => { const u = x / Lpp, n = Math.floor(u), r = u - n; return Lpp * (n * (2 * duty - 1) + (r < duty ? r : duty - (r - duty))); };
  return (F(z + dz) - F(z)) / dz;
}

/* ---------- CRIGF: DBR | spacer | grating coupler | spacer | DBR, excited by a Gaussian beam ---------- */
function crigfSetup(st, g, lam) {
  /* period matrices and layers at one wavelength */
  const k = TAU / lam;
  const pfD = profileFn(st, g.LamD), gD = sampleProfile(pfD), slD = sliceList(pfD, st);
  const stG = { ...st, h: g.hG, f: g.fG, profile: 'rect' };
  const pfG = profileFn(stG, g.LamG), gG = sampleProfile(pfG), slG = sliceList(pfG, stG);
  const tabD = tableAt(st, lam, 0, pfD.rect), tabG = tableAt(stG, lam, 0, true);
  const nOut = tabD.vals[tabD.K];
  const niProp = g.alphaProp / 2 / k, niG = (g.alphaProp + g.alphaG) / 2 / k;
  let P = [1, 0, 0, 0, 0, 0, 1, 0];
  for (const s of slD) P = mmul(P, layerMat(tint(tabD, s.g), niProp, k, s.w * g.LamD));
  const MD = mpow(P, g.ND);
  const gbar = mean(gG);
  const sub = 6, gcLayers = [];
  
  const nSp = tabG.vals[tabG.K];
  return { k, MD, nOut, gcLayers, nSp, tabD, tabG, gD, gG, pfD, pfG, slD, slG, stG, gbar };
}
function cfourier(arr, q) { let re = 0, im = 0; const n = arr.length; for (let i = 0; i < n; i++) { const ph = -TAU * q * (i + 0.5) / n; re += arr[i] * Math.cos(ph); im += arr[i] * Math.sin(ph); } return [re / n, im / n]; }
/* GC section by coupled-mode theory (Kazarinov–Henry): envelopes R, S of U = R e^{iβζ} + S e^{−iβζ}.
   Radiative self- and cross-coupling, 2nd-order guided Bragg coupling, beam excitation, DBR boundary reflections. */
function crigfAt(st, g, lam, wantField) {
  const S = crigfSetup(st, g, lam), k = S.k;
  const ns = nIdx(st.layers.sub, lam, st), nf = nIdx(st.layers.core, lam, st), nc = nIdx(st.layers.clad, lam, st);
  const lamM = lam * 1e-6, om = TAU * C0 / lamM, kM = TAU / lamM;
  /* GC averages */
  const nPG = neffProfile(S.tabG, S.gG), N0G = mean(nPG);
  const beta = kM * N0G, KG = TAU / (g.LamG * 1e-6), D = KG - beta;
  const G1 = cfourier(S.gG, 1), Gm1 = [G1[0], -G1[1]], N2 = cfourier(nPG, 2), Nm2 = [N2[0], -N2[1]];
  const mGC = modeParams(lam, ns, nf, nc, st.t - g.hG + g.hG * S.gbar, st.pol, 0);
  const xg = st.t - g.hG / 2;
  const eg = mGC.valid ? Math.sqrt(2 * om * MU0 / beta) * normMode(mGC)(xg) * 1e3 : 0;
  const dEps = nf * nf - nc * nc, hM = g.hG * 1e-6;
  const aC = 0.25 * om * EPS0 * dEps * hM * eg;                  /* a = i·aC */
  const th = g.theta * Math.PI / 180, kzIn = k * nc * Math.sin(th);
  const kzR = (beta - KG) * 1e-6;                                /* radiated in-plane wavevector, 1/µm */
  let chi = 0;
  for (const [med, nj] of channels(st, lam)) { const lf = planeWave(st, lam, kzR, xg, med === 'cladding' ? 'top' : 'bottom'); const kxj = Math.sqrt(Math.max(k * k * nj * nj - kzR * kzR, 1e-12)) * 1e6; chi += 0.5 * cabs2(lf.F) / kxj; }
  const rad = 2 * om * MU0 * chi * aC * aC;                      /* 2ωμ0χ·|a|², enters with a² = −aC² */
  const alphaP = g.alphaProp * 1e6;                              /* power loss 1/m */
  /* beam */
  const pwTop = planeWave(st, lam, kzIn, xg, 'top'), pwBot = planeWave(st, lam, kzIn, xg, 'bottom') || { F: [0, 0], tau: [0, 0] };
  const nB = stackOf(st, lam).botN;
  const Lsp = g.sp + (g.Ls || 0);
  const Lg = g.NG * g.LamG * 1e-6, zGC0 = g.ND * g.LamD + Lsp, zc = (g.NG * g.LamG) / 2;
  const E0 = Math.sqrt(2 * ETA0 / (nc * g.w0 * 1e-6 * Math.sqrt(Math.PI / 2)));
  const ths = Math.asin(Math.min(1, nc * Math.sin(th) / nB)), E0s = Math.sqrt(2 * ETA0 / (nB * g.w0 * 1e-6 * Math.sqrt(Math.PI / 2)));
  const oy = g.oy === undefined ? 1 : g.oy, etaB = g.etaB === undefined ? 1 : g.etaB;
  const kzInM = kzIn * 1e6, phase0 = kzIn * zGC0;
  const beamU = (zeta, cth) => Math.exp(-((((zeta * 1e6 - zc) * cth) / g.w0) ** 2));
  const Einc = zeta => cmul(pwTop.F, cscale(cexpi(kzInM * zeta + phase0), oy * E0 * beamU(zeta, Math.cos(th))));
  /* y' = A(ζ) y + b(ζ) */
  const kap = kM;
  const deriv = (zeta, y) => {
    const R = y[0], Sb = y[1];
    const eP = cexpi(D * zeta), eM = [eP[0], -eP[1]], e2P = cexpi(2 * D * zeta), e2M = [e2P[0], -e2P[1]];
    /* radiative: dR ⊃ −rad[|G1|² R + G1² S e^{2iDζ}],  dS ⊃ +rad[G−1² R e^{−2iDζ} + |G1|² S] */
    const g2 = cabs2(G1), G1s = cmul(G1, G1), Gm1s = cmul(Gm1, Gm1);
    let dR = cadd(cscale(R, -rad * g2 - alphaP / 2), cscale(cmul(cmul(G1s, e2P), Sb), -rad));
    let dS = cadd(cscale(Sb, rad * g2 + alphaP / 2), cscale(cmul(cmul(Gm1s, e2M), R), rad));
    /* guided 2nd-order Bragg: R' = i k N2 S e^{2iDζ}, S' = −i k N−2 R e^{−2iDζ} */
    dR = cadd(dR, cmul([0, kap], cmul(cmul(N2, e2P), Sb)));
    dS = cadd(dS, cmul([0, -kap], cmul(cmul(Nm2, e2M), R)));
    return [dR, dS];
  };
  const src = zeta => { const E = Einc(zeta), eP = cexpi(D * zeta), eM = [eP[0], -eP[1]];
    return [cmul([0, aC], cmul(cmul(G1, eP), E)), cmul([0, -aC], cmul(cmul(Gm1, eM), E))]; };
  const nStep = Math.max(32, Math.ceil(g.NG * 2.5)), hS = Lg / nStep;
  const step = (zeta, y, withSrc) => {
    const f = (z, v) => { const d = deriv(z, v); if (!withSrc) return d; const b = src(z); return [cadd(d[0], b[0]), cadd(d[1], b[1])]; };
    const add = (v, d, s) => [cadd(v[0], cscale(d[0], s)), cadd(v[1], cscale(d[1], s))];
    const k1 = f(zeta, y), k2 = f(zeta + hS / 2, add(y, k1, hS / 2)), k3 = f(zeta + hS / 2, add(y, k2, hS / 2)), k4 = f(zeta + hS, add(y, k3, hS));
    return [cadd(y[0], cscale(cadd(cadd(k1[0], cscale(k2[0], 2)), cadd(cscale(k3[0], 2), k4[0])), hS / 6)), cadd(y[1], cscale(cadd(cadd(k1[1], cscale(k2[1], 2)), cadd(cscale(k3[1], 2), k4[1])), hS / 6))];
  };
  let y1 = [[1, 0], [0, 0]], y2 = [[0, 0], [1, 0]], yp = [[0, 0], [0, 0]];
  const H1 = [y1], H2 = [y2], HP = [yp];
  for (let i = 0; i < nStep; i++) { const z = i * hS; y1 = step(z, y1, false); y2 = step(z, y2, false); yp = step(z, yp, true); H1.push(y1); H2.push(y2); HP.push(yp); }
  /* boundary mirrors, seen from the GC with reference index N0G */
  const Mspace = layerMat(S.nSp, (g.alphaProp + (g.alphaExtra || 0)) / 2 / k, k, Lsp);
  let Prev = [1, 0, 0, 0, 0, 0, 1, 0]; for (const s of S.slD.slice().reverse()) Prev = mmul(Prev, layerMat(tint(S.tabD, s.g), g.alphaProp / 2 / k, k, s.w * g.LamD));
  const MLrev = mmul(Mspace, mpow(Prev, g.ND)), MR = mmul(Mspace, S.MD);
  const mL = rtFromM(MLrev, N0G, S.nOut), mR = rtFromM(MR, N0G, S.nOut), sb = Math.sqrt(etaB), rL = cscale(mL.r, sb), rR = cscale(mR.r, sb);
  const rho = cmul(rR, cexpi(2 * beta * Lg));
  const c1 = cadd(cmul(y1[1], rL), y2[1]), c2 = cadd(cmul(y1[0], rL), y2[0]);
  const S0 = cdiv(csub(cmul(rho, yp[0]), yp[1]), csub(c1, cmul(rho, c2)));
  const R0 = cmul(rL, S0);
  const yAt = (Y1, Y2, YP) => [cadd(cadd(cmul(Y1[0], R0), cmul(Y2[0], S0)), YP[0]), cadd(cadd(cmul(Y1[1], R0), cmul(Y2[1], S0)), YP[1])];
  const yL = yAt(y1, y2, yp);
  /* escape through the mirrors */
  const stL = mvec(minv(MLrev), cadd(R0, S0), cscale(csub(S0, R0), N0G));          /* walking left: forward = leftward */
  const escLamp = cscale(cadd(stL[0], cscale(stL[1], 1 / S.nOut)), 0.5);
  const eR = cmul(yL[0], cexpi(beta * Lg)), eS = cmul(yL[1], cexpi(-beta * Lg));
  const stR = mvec(minv(MR), cadd(eR, eS), cscale(csub(eR, eS), N0G));
  const escRamp = cscale(cadd(stR[0], cscale(stR[1], 1 / S.nOut)), 0.5);
  /* power leaving through each mirror, and power lost laterally at each bounce (curved/finite DBRs) */
  const PinL = cabs2(S0), PinR = cabs2(yL[0]);
  const escL = PinL * mL.T, escR = PinR * mR.T;
  const latLoss = (PinL * cabs2(mL.r) + PinR * cabs2(mR.r)) * (1 - etaB);
  /* out-coupling into the beams: r_g = −¼ ∫ K_n E_rev dζ, K_n = c[G−1 R e^{−iDζ} + G1 S e^{iDζ}], c = −4i·aC */
  let rg = [0, 0], tg = [0, 0];
  const steps = [];
  const acc = (zeta, yv, wgt) => {
    const eP = cexpi(D * zeta), eM = [eP[0], -eP[1]];
    const Kn = cmul([0, -4 * aC], cadd(cmul(cmul(Gm1, eM), yv[0]), cmul(cmul(G1, eP), yv[1])));
    const Er = cmul(pwTop.F, cscale(cexpi(-(kzInM * zeta + phase0)), oy * E0 * beamU(zeta, Math.cos(th))));
    const Et = cmul(pwBot.F, cscale(cexpi(-(kzInM * zeta + phase0)), oy * E0s * beamU(zeta, Math.cos(ths))));
    rg = cadd(rg, cscale(cmul(Kn, Er), -0.25 * wgt)); tg = cadd(tg, cscale(cmul(Kn, Et), -0.25 * wgt));
  };
  /* Simpson over the RK grid (re-integrate, keeping states) */
  const ys = H1.map((_, i) => yAt(H1[i], H2[i], HP[i]));
  for (let i = 0; i <= nStep; i++) acc(i * hS, ys[i], hS * (i === 0 || i === nStep ? 0.5 : 1));
  const rTot = cadd(pwTop.r, rg);
  const tDir = pwTop.tau;
  const tTot = cadd(tDir, tg);
  const res = { R: cabs2(rTot), T: cabs2(tTot), Rd: cabs2(pwTop.r), escL, escR, latLoss, Umax: 0, nOut: S.nOut, N0G, rL, rR, alphaRad: 2 * rad * cabs2(G1) };
  for (const yv of ys) res.Umax = Math.max(res.Umax, cabs2(yv[0]) + cabs2(yv[1]));
  if (wantField) {
    const out = [];
    /* GC: fine samples within each RK step */
    for (let i = 0; i < nStep; i++) {
      const ya = ys[i], yb = ys[i + 1];
      for (let j = 0; j < 4; j++) {
        const r = j / 4, zeta = (i + r) * hS, Rr = cadd(cscale(ya[0], 1 - r), cscale(yb[0], r)), Sr = cadd(cscale(ya[1], 1 - r), cscale(yb[1], r));
        const U = cadd(cmul(Rr, cexpi(beta * zeta)), cmul(Sr, cexpi(-beta * zeta)));
        const zu = zeta * 1e6, u = ((zu / g.LamG) % 1 + 1) % 1, gg = S.pfG.g(u);
        out.push([zGC0 + zu, cabs2(U), U[0], U[1], gg, 1]);
      }
    }
    const niP = g.alphaProp / 2 / k;
    const mirror = []; for (let p = 0; p < g.ND; p++) for (const s of S.slD) mirror.push({ d: s.w * g.LamD, nr: tint(S.tabD, s.g), ni: niP, g: s.g, reg: 0 });
    const walk = (layers, E, H, z, dir) => {
      for (const L of layers) {
        const ns_ = L.sub || 4;
        for (let s = 1; s <= ns_; s++) { const M = layerMat(L.nr, L.ni, k, -dir * L.d * s / ns_), v = mvec(M, E, H); out.push([z + dir * L.d * s / ns_, cabs2(v[0]), v[0][0], v[0][1], L.g, L.reg]); }
        const v = mvec(layerMat(L.nr, L.ni, k, -dir * L.d), E, H); E = v[0]; H = v[1]; z += dir * L.d;
      }
    };
    const sp = { d: Lsp, nr: S.nSp, ni: (g.alphaProp + (g.alphaExtra || 0)) / 2 / k, g: 1, reg: 2, sub: Math.min(4000, Math.max(4, Math.ceil(Lsp / (g.LamD * 2)))) };
    walk([sp, ...mirror.slice().reverse()], cadd(R0, S0), cscale(csub(R0, S0), N0G), zGC0, -1);
    const zEnd = zGC0 + g.NG * g.LamG;
    walk([sp, ...mirror], cadd(eR, eS), cscale(csub(eR, eS), N0G), zEnd, 1);
    out.sort((a, b) => a[0] - b[0]);
    res.field = out; res.Ltot = zEnd + Lsp + g.ND * g.LamD; res.zGC = [zGC0, zEnd];
    /* straight sections: forward/backward amplitudes at their start, complex propagation constant (1/m) */
    const bc = [kM * S.nSp, (g.alphaProp + (g.alphaExtra || 0)) * 1e6 / 2], LspM = Lsp * 1e-6;
    const ab = (E, H) => [cscale(cadd(E, cscale(H, 1 / S.nSp)), 0.5), cscale(csub(E, cscale(H, 1 / S.nSp)), 0.5)];
    const [aL, bL] = ab(cadd(R0, S0), cscale(csub(R0, S0), N0G));
    const ph = cmul([0, 1], cscale(bc, LspM)), eF = (() => { const m = Math.exp(ph[0]); return [m * Math.cos(ph[1]), m * Math.sin(ph[1])]; })();
    const eB = cdiv([1, 0], eF);
    const [aR, bR] = ab(cadd(eR, eS), cscale(csub(eR, eS), N0G));
    res.straight = { bc, sections: [{ z0: zGC0 - Lsp, L: Lsp, A: cmul(aL, eB), B: cmul(bL, eF) }, { z0: zEnd, L: Lsp, A: aR, B: bR }] };
    res.stG = S.stG; res.gG = S.gG; res.gD = S.gD; res.rectD = S.pfD.rect;
  }
  return res;
}
/* SH local-field factor (slab reciprocity) for a sheet at the core centre, per medium and angle */
function shLocalFactor(st, lam2, nj, med, kz) {
  const ns = nIdx(st.layers.sub, lam2, st), nf = nIdx(st.layers.core, lam2, st), nc = nIdx(st.layers.clad, lam2, st);
  const pw = planeWave(st, lam2, kz, st.t / 2, med === 'cladding' ? 'top' : 'bottom');
  return pw ? cabs2(pw.F) : 0;
}
/* ∫_0^L s(z) e^{κ z} dz with s = ±1 periodic poling (period P, duty D, + first) or s = 1; κ complex, lengths in m */
function cexpC(z) { const m = Math.exp(z[0]); return [m * Math.cos(z[1]), m * Math.sin(z[1])]; }
function intExpPoledOff(kap, L, P, D, off) {
  if (!(P > 0) || !off) return intExpPoled(kap, L, P, D);
  const o = off * P;
  return cmul(cexpC(cscale(kap, -o)), csub(intExpPoled(kap, L + o, P, D), intExpPoled(kap, o, P, D)));
}
function intExpPoled(kap, L, P, D) {
  const small = Math.hypot(kap[0], kap[1]) * Math.max(L, P || 0) < 1e-6;
  const seg = (a, b) => small ? [b - a, 0] : cdiv(csub(cexpC(cscale(kap, b)), cexpC(cscale(kap, a))), kap);
  if (!(P > 0)) return seg(0, L);
  const N = Math.floor(L / P), rem = L - N * P;
  const cell = csub(seg(0, D * P), seg(D * P, P));
  const q = cexpC(cscale(kap, P)), qN = cexpC(cscale(kap, N * P));
  const den = csub([1, 0], q);
  const geo = Math.hypot(den[0], den[1]) < 1e-12 ? [N, 0] : cdiv(csub([1, 0], qN), den);
  let tot = cmul(cell, geo);
  const part = rem <= D * P ? seg(0, rem) : csub(seg(0, D * P), seg(D * P, rem));
  return cadd(tot, cmul(qN, part));
}
/* SHG from the CRIGF pump field: grating regions sampled, straight sections analytic */
function shgSampled(st, lamF, cr, gDev, opt) {
  const lam2 = lamF / 2, om = TAU * C0 / (lamF * 1e-6);
  const stG = cr.stG;
  const ovD = overlapTables(st, lamF, lam2, opt.sOrder, cr.rectD), ovG = overlapTables(stG, lamF, lam2, opt.sOrder, true);
  const tabF = tableAt(st, lamF, 0, cr.rectD), tab2 = tableAt(st, lam2, opt.sOrder, cr.rectD);
  const NF = mean(neffProfile(tabF, cr.gD)), N2 = mean(neffProfile(tab2, cr.gD));
  const kF = TAU / (lamF * 1e-6), k2 = 2 * kF, beta = kF * NF, beta2 = k2 * N2;
  const LppDev = TAU / Math.abs(beta2 - 2 * beta) * 1e6;
  const bS = kF * tabF.vals[tabF.K], b2S = k2 * tab2.vals[tab2.K];
  const LppS = TAU / Math.abs(b2S - 2 * bS) * 1e6;               /* straight, unetched guide */
  const poleG = opt.poleMode === 'all', poleS = opt.poleMode === 'all' || opt.poleMode === 'straight';
  const ti = (ov, arr, g) => { const x = g * ov.K, i = Math.min(ov.K - 1, Math.max(0, Math.floor(x))), r = x - i; return arr[i] * (1 - r) + arr[i + 1] * r; };
  const secs = (cr.straight && cr.straight.sections) || [];
  /* grating regions on a uniform grid (straight sections excluded) */
  const f = cr.field, Lt = cr.Ltot, dz = Math.min(gDev.LamD, gDev.LamG) / 16;
  const regs = []; let z0 = 0;
  for (const sct of secs.slice().sort((a, b) => a.z0 - b.z0)) { if (sct.z0 > z0) regs.push([z0, sct.z0]); z0 = sct.z0 + sct.L; }
  if (Lt > z0) regs.push([z0, Lt]);
  const samp = [];
  let j = 0;
  for (const [ra, rb] of regs) {
    const n = Math.max(1, Math.ceil((rb - ra) / dz)), h = (rb - ra) / n, arr = { z0: ra, h: h * 1e-6, n, Ur: new Float64Array(n), Ui: new Float64Array(n), G: new Float64Array(n), O: new Float64Array(n) };
    for (let i = 0; i < n; i++) {
      const z = ra + (i + 0.5) * h;
      while (j < f.length - 2 && f[j + 1][0] < z) j++;
      while (j > 0 && f[j][0] > z) j--;
      const a = f[j], b = f[j + 1] || a, r = b[0] > a[0] ? Math.min(1, Math.max(0, (z - a[0]) / (b[0] - a[0]))) : 0;
      arr.Ur[i] = a[2] + (b[2] - a[2]) * r; arr.Ui[i] = a[3] + (b[3] - a[3]) * r;
      const ov = a[5] === 1 ? ovG : ovD, pol = poleG ? polingMean(z, h, LppDev, opt.duty) : 1;
      arr.G[i] = ti(ov, ov.G, a[4]) * pol; arr.O[i] = ti(ov, ov.O, a[4]) * pol;
    }
    samp.push(arr);
  }
  const Pw = opt.P / (opt.w * 1e-6);
  for (const a of samp) { a.U2r = new Float64Array(a.n); a.U2i = new Float64Array(a.n); for (let i = 0; i < a.n; i++) { a.U2r[i] = (a.Ur[i] ** 2 - a.Ui[i] ** 2) * Pw; a.U2i[i] = 2 * a.Ur[i] * a.Ui[i] * Pw; } }
  /* guided SH phase φ2(z) accumulated region by region; radiated waves use kz·z */
  const phiAt = z => { let ph = 0, zc = 0; const all = [...regs.map(r => [r[0], r[1], beta2]), ...secs.map(c => [c.z0, c.z0 + c.L, b2S])].sort((a, b) => a[0] - b[0]);
    for (const [a, b, bb] of all) { if (z <= a) break; ph += bb * (Math.min(z, b) - a) * 1e-6; } return ph; };
  const srcGr = (key, kz, guided) => { let sr = 0, si = 0;
    for (const a of samp) { const X = a[key], c_ = Math.cos(kz * a.h), s_ = -Math.sin(kz * a.h); const p0 = guided ? -Math.sign(kz) * (phiAt(a.z0) + Math.abs(kz) * a.h / 2) : -kz * (a.z0 * 1e-6 + a.h / 2); let pr = Math.cos(p0), pi = Math.sin(p0);
      for (let i = 0; i < a.n; i++) { const xr = X[i] * a.U2r[i], xi = X[i] * a.U2i[i]; sr += (xr * pr - xi * pi) * a.h; si += (xr * pi + xi * pr) * a.h; const t = pr * c_ - pi * s_; pi = pr * s_ + pi * c_; pr = t; } }
    return [sr, si]; };
  const GS = ovD.G[ovD.K], OS = ovD.O[ovD.K], bc = cr.straight ? cr.straight.bc : [bS, 0];
  let offs = [0, 0];
  const srcSt = (X, kz, guided, only) => { let tot = [0, 0];
    secs.forEach((sc, si) => { if (!(sc.L > 0) || (only !== undefined && only !== si)) return; const L = sc.L * 1e-6, P = poleS ? LppS * 1e-6 : 0, off = offs[si];
      const kzL = guided ? Math.sign(kz) * b2S : kz, ph0 = guided ? -Math.sign(kz) * phiAt(sc.z0) : -kz * sc.z0 * 1e-6;
      const A2 = cmul(sc.A, sc.A), B2 = cmul(sc.B, sc.B), AB2 = cscale(cmul(sc.A, sc.B), 2);
      const ib = cmul([0, 1], bc);
      const t1 = cmul(A2, intExpPoledOff(cadd(cscale(ib, 2), [0, -kzL]), L, P, opt.duty, off));
      const t2 = cmul(B2, intExpPoledOff(cadd(cscale(ib, -2), [0, -kzL]), L, P, opt.duty, off));
      const t3 = cmul(AB2, intExpPoledOff([0, -kzL], L, P, opt.duty, off));
      tot = cadd(tot, cmul(cexpi(ph0), cscale(cadd(cadd(t1, t2), t3), X * Pw))); });
    return tot; };
  const d = opt.d * 1e-12, Kpref = 2 * om * EPS0 * d * (2 * om * MU0 / beta);
  const gpref = om * EPS0 * d / 2 * (2 * om * MU0 / beta) * Math.sqrt(4 * om * MU0 / beta2);
  const out = { lam2, NF, N2, beta, beta2, dk: beta2 - 2 * beta, Kg: TAU / (gDev.LamD * 1e-6), guided2: !tab2.hiCut, Lpp: LppDev, LppS, dkS: b2S - 2 * bS };
  const P = s => gpref ** 2 * cabs2(s) * opt.w * 1e-6;
  if (out.guided2) {
    const fG = srcGr('O', beta2, true), bG = srcGr('O', -beta2, true);
    /* poling offset of the right section relative to the left: a free design parameter, chosen for maximum guided SH */
    if (poleS && secs.length === 2 && secs[1].L > 0 && opt.optOffset !== false) {
      let best = -1, bo = 0;
      for (let i = 0; i < 36; i++) { offs = [0, i / 36]; const v = P(cadd(srcSt(OS, beta2, true), fG)) + P(cadd(srcSt(OS, -beta2, true), bG)); if (v > best) { best = v; bo = i / 36; } }
      offs = [0, bo]; out.offset = bo;
    }
    const fS = srcSt(OS, beta2, true), bSt = srcSt(OS, -beta2, true);
    out.Pf = P(cadd(fS, fG)); out.Pb = P(cadd(bSt, bG)); out.PfStr = P(fS); out.PbStr = P(bSt); out.Pref = NaN;
    const Lp = secs.reduce((s_, c) => s_ + c.L, 0) * 1e-6; out.Psp = (gpref * OS * Pw * (poleS ? 2 / Math.PI : 0) * Lp) ** 2 * opt.w * 1e-6; out.Lpoled = Lp;
  }
  const far = (nj, med) => {
    const kmax = k2 * nj * 1e-6, th = [];
    for (let t = -89.95; t < 90; t += 0.1) th.push(t);
    const Ls = Math.max(1, ...secs.map(x => x.L)), cand = [0];
    if (poleS) { const Kp = TAU / LppS; for (let m = 1; m <= 15; m += 2) cand.push(m * Kp, -m * Kp); }
    for (const kz of cand) { const sn = kz / kmax; if (Math.abs(sn) >= 1) continue; const t0 = Math.asin(sn) * 180 / Math.PI, wd = Math.max(lam2 / (nj * Ls * Math.max(Math.cos(t0 * Math.PI / 180), 0.05)) * 180 / Math.PI, 1e-5);
      for (let i = -40; i <= 40; i++) { const t = t0 + i * wd / 6; if (t > -90 && t < 90) th.push(t); } }
    th.sort((a, b) => a - b);
    const I = new Float64Array(th.length);
    for (let q = 0; q < th.length; q++) { const kz = k2 * nj * Math.sin(th[q] * Math.PI / 180), s = cadd(srcGr('G', kz), srcSt(GS, kz)); I[q] = 2 * om * MU0 / (16 * Math.PI) * Kpref ** 2 * cabs2(s) * opt.w * 1e-6 * shLocalFactor(st, lam2, nj, med, kz * 1e-6); }
    return lobeStats(Float64Array.from(th), I, nj);
  };
  out.up = far(nIdx(st.layers.clad, lam2, st), 'cladding'); out.down = stackOf(st, lam2).botReal ? far(stackOf(st, lam2).botN, 'substrate') : { th: [], I: new Float64Array(0), P: 0, peaks: [], nj: NaN };
  const live = secs.filter(c => c.L > 0); out.c0 = live.length ? live.reduce((s_, c) => s_ + (cabs2(c.A) + cabs2(c.B)) / 2, 0) / live.length : 0;
  return out;
}
/* cavity resonance nearest lamC: maximise circulating guided power over one free spectral range */
function findResonance(st, g, lamC, fsr, opt_n, noWidth) {
  const U = l => crigfAt(st, g, l, false).Umax;
  const n = opt_n || 40, a = lamC - fsr / 2, h = fsr / n;
  let ib = 0, vb = -1;
  for (let i = 0; i <= n; i++) { const v = U(a + i * h); if (v > vb) { vb = v; ib = i; } }
  let lo = a + (ib - 1) * h, hi = a + (ib + 1) * h;
  for (let i = 0; i < 30; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (U(m1) > U(m2)) hi = m2; else lo = m1; }
  const lam = (lo + hi) / 2, Um = U(lam);
  if (noWidth) return { lam, Umax: Um, fwhm: NaN };
  const edge = (dir) => { let l0 = lam, l1 = lam + dir * fsr / 2; for (let i = 0; i < 28; i++) { const m = (l0 + l1) / 2; if (U(m) > Um / 2) l0 = m; else l1 = m; } return (l0 + l1) / 2; };
  const fw = edge(1) - edge(-1);
  return { lam, Umax: Um, fwhm: fw };
}
function lobeStats(th, I, nj) {
  let P = 0; for (let j = 1; j < th.length; j++) P += 0.5 * (I[j] + I[j - 1]) * (th[j] - th[j - 1]) * Math.PI / 180;
  const peaks = []; let mx = 0; for (const v of I) mx = Math.max(mx, v);
  for (let j = 1; j < th.length - 1; j++) if (I[j] >= I[j - 1] && I[j] > I[j + 1] && I[j] > mx * 1e-3) {
    let a = j, b = j; while (a > 0 && I[a - 1] < I[a]) a--; while (b < th.length - 1 && I[b + 1] < I[b]) b++;
    let pw = 0; for (let k = a + 1; k <= b; k++) pw += 0.5 * (I[k] + I[k - 1]) * (th[k] - th[k - 1]) * Math.PI / 180;
    let hl = j, hr = j; while (hl > 0 && I[hl] > I[j] / 2) hl--; while (hr < th.length - 1 && I[hr] > I[j] / 2) hr++;
    peaks.push({ theta: th[j], peak: I[j], P: pw, fwhm: th[hr] - th[hl] });
  }
  peaks.sort((a, b) => b.P - a.P);
  return { th, I, P, peaks: peaks.slice(0, 6), nj };
}

/* ---------- lateral optics ---------- */
/* 1D Gaussian beams u = exp(−i k y²/(2q)); power overlap of two beams at the same plane */
function gaussOverlap(q1, q2, k) {
  const a = q => { const iq = cdiv([1, 0], q); return [-k / 2 * iq[1], k / 2 * iq[0]]; };
  const A1 = a(q1), A2 = a(q2);
  const s = [A1[0] + A2[0], A1[1] - A2[1]];
  return 2 * Math.sqrt(A1[0] * A2[0]) / Math.hypot(s[0], s[1]);
}
/* bounce efficiency of a lateral Gaussian launched from a plane (waist w, flat phase), propagated a distance d
   to a mirror of radius Rc (concave > 0, Infinity = flat) with aperture half-width a, and back */
function bounce(lamM, w, d, Rc, a) {
  const k = TAU / lamM, zR = Math.PI * w * w / lamM;
  const q0 = [0, zR], qm = [d, zR];
  const inv = cdiv([1, 0], qm);
  const wm = Math.sqrt(lamM / (Math.PI * Math.max(-inv[1], 1e-30)));
  const iq2 = [inv[0] - (Number.isFinite(Rc) ? 2 / Rc : 0), inv[1]];
  const qb = cadd(cdiv([1, 0], iq2), [d, 0]);
  const ov = gaussOverlap(q0, qb, k);
  const clip = Number.isFinite(a) ? erf(Math.SQRT2 * a / wm) : 1;
  return { eta: ov * clip, ov, clip, wm };
}
/* best lateral waist for a symmetric cavity (half-length d) or a launched beam; returns the optimum */
function bestBounce(lamM, d, Rc, a) {
  let best = null;
  const wlo = Math.max(0.3 * lamM, 0.05), whi = Math.max(Number.isFinite(a) ? 2 * a : 50, 2 * Math.sqrt(lamM * d) + 1);
  for (let i = 0; i <= 120; i++) { const w = wlo * Math.pow(whi / wlo, i / 120), b = bounce(lamM, w, d, Rc, a); if (!best || b.eta > best.eta) best = { ...b, w }; }
  let lo = best.w / 1.1, hi = best.w * 1.1;
  for (let i = 0; i < 40; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (bounce(lamM, m1, d, Rc, a).eta > bounce(lamM, m2, d, Rc, a).eta) hi = m2; else lo = m1; }
  const w = (lo + hi) / 2; return { ...bounce(lamM, w, d, Rc, a), w };
}
/* ridge lateral mode: overlap with a Gaussian beam of waist w0 and SHG effective width 1/∫φ⁴ */
function ridgeLateral(st, lam, nIn, ns, nf, nc, w0) {
  const nO = sideIndex(st, lam, ns, nf, nc), polL = st.pol === 'TE' ? 'TM' : 'TE';
  const m = modeParams(lam, nO, nIn, nO, st.lat.W, polL, 0);
  if (!m.valid) return { valid: false };
  const W = st.lat.W, ext = W / 2 + Math.max(3 / m.gs, 3 * w0), N = 800, f = normMode(m);
  let ov = 0, g2 = 0, p4 = 0, mx = 0, x50 = 0;
  for (let i = 0; i <= N; i++) { const y = -ext + 2 * ext * i / N, wgt = 2 * ext / N * (i === 0 || i === N ? 0.5 : 1), phi = f(y + W / 2), gb = Math.exp(-((y / w0) ** 2));
    ov += wgt * phi * gb; g2 += wgt * gb * gb; p4 += wgt * phi ** 4; }
  return { valid: true, Oy2: ov * ov / g2, wEff: 1 / p4, neff: m.neff, nSide: nO };
}

/* tune the phase spacer (0 … λ/2n) and the wavelength for maximum circulating pump power */
function tunePhase(st, g, lamC, fsr, nSp) {
  const half = lamC / (2 * nSp), tries = [];
  for (let i = 0; i < 6; i++) { const sp = half * i / 6, r = findResonance(st, { ...g, sp }, lamC, fsr, 28, true); tries.push({ sp, ...r }); }
  tries.sort((a, b) => b.Umax - a.Umax);
  let lo = tries[0].sp - half / 6, hi = tries[0].sp + half / 6;
  const val = sp => findResonance(st, { ...g, sp: ((sp % half) + half) % half }, lamC, fsr, 20, true).Umax;
  for (let i = 0; i < 6; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (val(m1) > val(m2)) hi = m2; else lo = m1; }
  const sp = (((lo + hi) / 2) % half + half) % half;
  return { sp, ...findResonance(st, { ...g, sp }, lamC, fsr, 28, false) };
}

/* ---------- pump depletion and cavity loading (CRIGF with poled straight sections) ----------
   The undepleted solution gives E_NL (W⁻¹): guided SH per direction = E_NL·Pc² for circulating power Pc.
   Per pass a fraction tanh²(√(E_NL·Pc)) of the pump converts; that fraction is put back as an extra loss in the
   straight sections and Pc is iterated to self-consistency. Radiated SH scales with Pc². */
function depletionSweep(st, g, lam, shRef, Pref, powers) {
  const secs = shRef.secsAmp, c0 = shRef.c0;            /* circulating power per direction per watt of beam */
  const Pdir0 = c0 * Pref;
  const E = Pdir0 > 0 ? (shRef.Pf + shRef.Pb) / 2 / (Pdir0 * Pdir0) : 0;
  const U0 = crigfAt(st, g, lam, false).Umax, Ls2 = 2 * (g.Ls + g.sp);
  const radRatio = (shRef.up.P + shRef.down.P) / Math.max(shRef.Pf + shRef.Pb, 1e-300);
  return powers.map(P => {
    let Pc = c0 * P, aEx = 0;
    for (let it = 0; it < 60; it++) {
      const f = Math.min(0.999999, Math.tanh(Math.sqrt(Math.max(E * Pc, 0))) ** 2);
      const aNew = Ls2 > 0 ? -Math.log(1 - f) / Ls2 : 0;
      aEx = 0.5 * aEx + 0.5 * aNew;
      const U = crigfAt(st, { ...g, alphaExtra: aEx }, lam, false).Umax;
      const PcNew = c0 * P * U / U0;
      if (Math.abs(PcNew - Pc) < 1e-6 * Pc) { Pc = PcNew; break; }
      Pc = 0.5 * Pc + 0.5 * PcNew;
    }
    const f = Math.tanh(Math.sqrt(Math.max(E * Pc, 0))) ** 2;
    const Pg = 2 * Pc * f, PgU = 2 * E * (c0 * P) ** 2;
    return { P, Pc, Pc0: c0 * P, f, Psh: Pg * (1 + radRatio), PshU: PgU * (1 + radRatio), Pg, conv: Pg * (1 + radRatio) / P };
  });
}
/* effective area of a guided mode (vertical 1/∫φ⁴ × lateral width) in µm² */
function modeArea(st, lam, order, wLat) {
  const ns = nIdx(st.layers.sub, lam, st), nf = nIdx(st.layers.core, lam, st), nc = nIdx(st.layers.clad, lam, st);
  const m = modeParams(lam, ns, nf, nc, st.t, st.pol, order); if (!m.valid) return NaN;
  const f = normMode(m), x0 = -3 / m.gs, x1 = st.t + 3 / m.gc, N = 600; let s4 = 0;
  for (let i = 0; i <= N; i++) { const x = x0 + (x1 - x0) * i / N; s4 += (i === 0 || i === N ? 0.5 : 1) * (x1 - x0) / N * f(x) ** 4; }
  return wLat / s4;
}
