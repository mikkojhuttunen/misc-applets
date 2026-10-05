/* =========================================================================
   nlo-engine.js  --  dependency-free numerical engine for nonlinear dynamics
   Version 0.2.0

   One file, no build step, no dependencies. Works as
     - a browser global:   <script src="nlo-engine.js"></script>  ->  NLO
     - a Node module:      const NLO = require('./nlo-engine.js');
     - inlined text inside a single-file HTML applet (copy the whole file
       between <script> tags; nothing else is needed).

   Contents
     1. RNG            seedable uniform / Gaussian generator
     2. FFT            in-place radix-2 complex FFT with cached plans
     3. Grid           periodic time grid + angular-frequency axis
     4. ODE            fixed-step RK4 and adaptive Dormand-Prince RK45
     5. NLSE           generalised nonlinear Schroedinger / Ginzburg-Landau
                       solver by symmetric split-step Fourier (2nd order)
                       or first-order Lie splitting; pluggable nonlinearities
     6. nl             nonlinear-step factories: kerr, pointwise, compose, and
                       gnlse (Kerr + self-steepening + Raman, intrinsic or full response)
     7. analysis       pulse metrics, spectrum, invariants, soliton helpers
     8. ConvergenceMonitor   adaptive stopping rule for round-trip maps
     9. HausModelocking      round-trip master-equation model (extracted from
                       the FYS.501 mode-locking explorer, see README)

   CONVENTION (matters for odd-order dispersion and for sign checks)
     A(T,z) = (1/2pi) Int A~(w) exp(-i w T) dw,   A~(w) = Int A(T) exp(+i w T) dT
     dA/dz  = D(w) A  +  N(A) A,    D(w) = -alpha/2 + gain - gainBW w^2
                                            + i * Sum_{m>=2} beta_m w^m / m!
   This is the Agrawal (Nonlinear Fiber Optics) convention: beta2 < 0 is
   anomalous dispersion and supports bright solitons together with gamma > 0.
   Internally the "spectral" transform is the forward DFT exp(-2 pi i k n / N),
   so grid.omega[k] = -2 pi kk / (N dt) with kk the signed DFT index.
   ========================================================================= */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NLO = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const VERSION = '0.2.0';

  /* ---------------------------------------------------------------------
     1. RNG  (mulberry32 + Box-Muller). Pass an rng object wherever
        randomness is used so runs are reproducible.
  --------------------------------------------------------------------- */
  function makeRng(seed) {
    let s = (seed === undefined ? Math.floor(Math.random() * 4294967296) : seed) >>> 0;
    function uniform() {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function randn() {
      let u = 0, v = 0;
      while (u === 0) u = uniform();
      while (v === 0) v = uniform();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return { uniform: uniform, randn: randn };
  }

  /* ---------------------------------------------------------------------
     2. FFT  (in place, power-of-two length, cached twiddles/bit reversal)
        forward: X_k = sum_n x_n exp(-2 pi i k n / N)
        inverse: includes the 1/N factor
  --------------------------------------------------------------------- */
  const _plans = new Map();
  function _plan(N) {
    let p = _plans.get(N);
    if (p) return p;
    if (N < 2 || (N & (N - 1)) !== 0) throw new Error('FFT length must be a power of two >= 2, got ' + N);
    const logN = Math.round(Math.log2(N));
    const cos = new Float64Array(N / 2), sin = new Float64Array(N / 2);
    for (let k = 0; k < N / 2; k++) { const a = 2 * Math.PI * k / N; cos[k] = Math.cos(a); sin[k] = Math.sin(a); }
    const rev = new Uint32Array(N);
    for (let i = 1; i < N; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (logN - 1));
    p = { cos: cos, sin: sin, rev: rev };
    _plans.set(N, p);
    return p;
  }

  function fft(re, im, inverse) {
    const N = re.length, p = _plan(N), rev = p.rev;
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? 1 : -1;
    for (let len = 2; len <= N; len <<= 1) {
      const half = len >> 1, step = N / len;
      for (let i = 0; i < N; i += len) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = p.cos[t], wi = sgn * p.sin[t];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr;        im[a] += xi;
        }
      }
    }
    if (inverse) { const s = 1 / N; for (let i = 0; i < N; i++) { re[i] *= s; im[i] *= s; } }
  }

  /* ---------------------------------------------------------------------
     3. Grid  -- periodic window of N points, spacing dt (or total width T).
        t[n] = (n - N/2) dt  (window centred on 0)
        omega[k]            see CONVENTION above
  --------------------------------------------------------------------- */
  function makeGrid(N, opts) {
    opts = opts || {};
    const dt = opts.dt !== undefined ? opts.dt : (opts.T !== undefined ? opts.T / N : 1);
    const t = new Float64Array(N), omega = new Float64Array(N);
    const dw = 2 * Math.PI / (N * dt);
    for (let n = 0; n < N; n++) {
      t[n] = (n - N / 2) * dt;
      const kk = n <= N / 2 ? n : n - N;
      omega[n] = -dw * kk;
    }
    return { N: N, dt: dt, T: N * dt, dw: dw, t: t, omega: omega };
  }

  /* ---------------------------------------------------------------------
     4. ODE integrators.   f(t, y, dydt)  writes the derivative into dydt.
  --------------------------------------------------------------------- */
  function rk4Step(f, t, y, h, ws) {
    const n = y.length;
    const k1 = ws.k1, k2 = ws.k2, k3 = ws.k3, k4 = ws.k4, yt = ws.yt;
    f(t, y, k1);
    for (let i = 0; i < n; i++) yt[i] = y[i] + 0.5 * h * k1[i];
    f(t + 0.5 * h, yt, k2);
    for (let i = 0; i < n; i++) yt[i] = y[i] + 0.5 * h * k2[i];
    f(t + 0.5 * h, yt, k3);
    for (let i = 0; i < n; i++) yt[i] = y[i] + h * k3[i];
    f(t + h, yt, k4);
    for (let i = 0; i < n; i++) y[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }

  // Dormand-Prince 5(4) tableau
  const DP_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
  const DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
  ];
  const DP_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
  const DP_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];

  /**
   * integrate(f, y0, opts) -> { t, y, steps, rejected }
   *   opts.t0 (0), opts.t1 (required)
   *   opts.method   'rk4' (default, needs opts.dt) | 'rk45' (adaptive)
   *   opts.dt       step (rk4) / initial step (rk45)
   *   opts.rtol, opts.atol, opts.dtMax, opts.maxSteps   (rk45)
   *   opts.onStep(t, y)   called after every accepted step (y is live; copy it)
   * y0 is not modified.
   */
  function integrate(f, y0, opts) {
    const n = y0.length, y = Float64Array.from(y0);
    let t = opts.t0 || 0;
    const t1 = opts.t1, method = opts.method || 'rk4';
    const onStep = opts.onStep;
    let steps = 0, rejected = 0;
    if (method === 'rk4') {
      const nSteps = Math.max(1, Math.round((t1 - t) / opts.dt)), h = (t1 - t) / nSteps;
      const ws = { k1: new Float64Array(n), k2: new Float64Array(n), k3: new Float64Array(n), k4: new Float64Array(n), yt: new Float64Array(n) };
      for (let s = 0; s < nSteps; s++) { rk4Step(f, t, y, h, ws); t += h; steps++; if (onStep) onStep(t, y); }
      return { t: t, y: y, steps: steps, rejected: 0 };
    }
    if (method !== 'rk45') throw new Error('unknown method ' + method);
    const rtol = opts.rtol !== undefined ? opts.rtol : 1e-8, atol = opts.atol !== undefined ? opts.atol : 1e-10;
    const maxSteps = opts.maxSteps || 1e6, dtMax = opts.dtMax || Infinity, dir = t1 >= t ? 1 : -1;
    let h = Math.min(Math.abs(opts.dt || (t1 - t) / 100), dtMax) * dir;
    const k = []; for (let i = 0; i < 7; i++) k.push(new Float64Array(n));
    const yt = new Float64Array(n), ynew = new Float64Array(n);
    f(t, y, k[0]);
    while ((t1 - t) * dir > 1e-14 * Math.max(1, Math.abs(t1))) {
      if (steps + rejected > maxSteps) throw new Error('rk45: maxSteps exceeded');
      if ((t + h - t1) * dir > 0) h = t1 - t;
      for (let s = 1; s < 7; s++) {
        for (let i = 0; i < n; i++) {
          let acc = 0; for (let j = 0; j < s; j++) acc += DP_A[s][j] * k[j][i];
          yt[i] = y[i] + h * acc;
        }
        f(t + DP_C[s] * h, yt, k[s]);
      }
      let errN = 0;
      for (let i = 0; i < n; i++) {
        let s5 = 0, s4 = 0;
        for (let j = 0; j < 7; j++) { s5 += DP_B5[j] * k[j][i]; s4 += DP_B4[j] * k[j][i]; }
        ynew[i] = y[i] + h * s5;
        const e = h * (s5 - s4), sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(ynew[i]));
        errN = Math.max(errN, Math.abs(e) / sc);
      }
      if (errN <= 1) {
        t += h; for (let i = 0; i < n; i++) y[i] = ynew[i];
        const tmp = k[0]; k[0] = k[6]; k[6] = tmp;   // FSAL
        steps++; if (onStep) onStep(t, y);
      } else rejected++;
      const fac = errN === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * Math.pow(errN, -0.2)));
      h *= fac; if (Math.abs(h) > dtMax) h = dtMax * dir;
    }
    return { t: t, y: y, steps: steps, rejected: rejected };
  }

  /* ---------------------------------------------------------------------
     5. NLSE solver (split-step Fourier)
        dA/dz = D(w) A + N(A) A     A = { re: Float64Array, im: Float64Array }

        linear spec:  { alpha, gain, gainBW, beta:[beta2,beta3,...], custom(w)->[re,im] }
        nonlinear:    a function (re, im, h, z) that applies exp(h N(A)) in
                      place (see nl.* factories), or null for the linear problem.
  --------------------------------------------------------------------- */
  function linearSymbol(omega, spec) {
    spec = spec || {};
    const N = omega.length, re = new Float64Array(N), im = new Float64Array(N);
    const alpha = spec.alpha || 0, gain = spec.gain || 0, gainBW = spec.gainBW || 0, beta = spec.beta || [];
    for (let k = 0; k < N; k++) {
      const w = omega[k];
      let r = -alpha / 2 + gain - gainBW * w * w, ph = 0, wp = w * w, fact = 2;   // wp = w^m, fact = m!
      for (let m = 2; m < 2 + beta.length; m++) {
        ph += beta[m - 2] * wp / fact;
        wp *= w; fact *= (m + 1);
      }
      if (spec.custom) { const c = spec.custom(w); r += c[0]; ph += c[1]; }
      re[k] = r; im[k] = ph;
    }
    return { re: re, im: im };
  }

  class NLSE {
    /**
     * new NLSE({ N, T | dt, linear, nonlinear, gamma, method })
     *   gamma     shortcut: nonlinear = nl.kerr(gamma)   (ignored if nonlinear given)
     *   method    'strang' (default, 2nd order) | 'lie' (1st order)
     */
    constructor(opts) {
      this.grid = makeGrid(opts.N, { T: opts.T, dt: opts.dt });
      this.N = opts.N;
      this.linear = opts.linear || {};
      this.D = linearSymbol(this.grid.omega, this.linear);
      this.nonlinear = opts.nonlinear || (opts.gamma ? nl.kerr(opts.gamma) : null);
      this.method = opts.method || 'strang';
      this._cache = new Map();
      this._sre = new Float64Array(this.N); this._sim = new Float64Array(this.N);
    }

    // exp(s * D(w)) as a spectral multiplier; cached per s
    _prop(s) {
      const key = s.toPrecision(17);
      let p = this._cache.get(key);
      if (p) return p;
      const N = this.N, pr = new Float64Array(N), pi = new Float64Array(N);
      for (let k = 0; k < N; k++) {
        const a = Math.exp(s * this.D.re[k]), ph = s * this.D.im[k];
        pr[k] = a * Math.cos(ph); pi[k] = a * Math.sin(ph);
      }
      p = { re: pr, im: pi };
      if (this._cache.size > 8) this._cache.clear();
      this._cache.set(key, p);
      return p;
    }

    _applyL(re, im, p) {
      const N = this.N;
      fft(re, im, false);
      for (let k = 0; k < N; k++) {
        const r = re[k] * p.re[k] - im[k] * p.im[k], i = re[k] * p.im[k] + im[k] * p.re[k];
        re[k] = r; im[k] = i;
      }
      fft(re, im, true);
    }

    /** One step of size h (symmetric Strang or Lie, per this.method). */
    step(A, h, z) {
      z = z || 0;
      if (this.method === 'lie') {
        if (this.nonlinear) this.nonlinear(A.re, A.im, h, z);
        this._applyL(A.re, A.im, this._prop(h));
      } else {
        const ph = this._prop(h / 2);
        this._applyL(A.re, A.im, ph);
        if (this.nonlinear) this.nonlinear(A.re, A.im, h, z + h / 2);
        this._applyL(A.re, A.im, ph);
      }
      return A;
    }

    /**
     * propagate(A, { dz, steps, z0, every, onSample })  -- modifies A in place.
     *   Calls onSample(z, re, im, stepIndex) at z0 and then every `every` steps.
     *   The arrays handed to onSample are valid only during the call (copy to keep).
     *   Strang mode merges adjacent half linear steps (one FFT pair per step instead of two)
     *   unless the linear operator is so strongly damping that undoing the half step would
     *   over/underflow, in which case it falls back to the plain 2-FFT-pair form.
     */
    propagate(A, opts) {
      const h = opts.dz, steps = opts.steps, every = opts.every || 1, onSample = opts.onSample;
      let z = opts.z0 || 0;
      if (onSample) onSample(z, A.re, A.im, 0);
      let merge = this.method === 'strang';
      if (merge) { let minRe = Infinity; for (let k = 0; k < this.N; k++) minRe = Math.min(minRe, this.D.re[k]); if (-0.5 * h * minRe > 20) merge = false; }
      if (!merge) {
        for (let n = 1; n <= steps; n++) {
          this.step(A, h, z); z += h;
          if (onSample && n % every === 0) onSample(z, A.re, A.im, n);
        }
        return { z: z };
      }
      // merged Strang:  B_0 = L(h/2) A_0 ;  B_{n+1} = L(h) N(h) B_n ;  A_n = L(-h/2) B_n
      const pf = this._prop(h), ph = this._prop(h / 2), pinv = this._prop(-h / 2);
      this._applyL(A.re, A.im, ph);
      for (let n = 1; n <= steps; n++) {
        if (this.nonlinear) this.nonlinear(A.re, A.im, h, z + h / 2);
        this._applyL(A.re, A.im, pf); z += h;
        if (onSample && n % every === 0 && n < steps) {
          this._sre.set(A.re); this._sim.set(A.im);
          this._applyL(this._sre, this._sim, pinv);
          onSample(z, this._sre, this._sim, n);
        }
      }
      this._applyL(A.re, A.im, pinv);
      if (onSample && steps % every === 0) onSample(z, A.re, A.im, steps);
      return { z: z };
    }

    /** Convenience: propagate and collect |A|^2(T) and |A~|^2(w) snapshots. */
    evolve(A, opts) {
      const out = { z: [], I: [], S: [] };
      const self = this;
      this.propagate(A, {
        dz: opts.dz, steps: opts.steps, z0: opts.z0, every: opts.every,
        onSample: function (z, re, im) {
          const I = new Float64Array(self.N); for (let i = 0; i < self.N; i++) I[i] = re[i] * re[i] + im[i] * im[i];
          out.z.push(z); out.I.push(I); out.S.push(analysis.spectrum(re, im, self.grid).power);
        }
      });
      return out;
    }
  }

  /* ---------------------------------------------------------------------
     6. Nonlinear-step factories.  Each returns fn(re, im, h, z) that
        applies A <- exp(h * N) A in place.
  --------------------------------------------------------------------- */
  const nl = {
    /** Kerr / SPM:  dA/dz = i gamma |A|^2 A */
    kerr: function (gamma) {
      return function (re, im, h) {
        for (let i = 0; i < re.length; i++) {
          const ph = gamma * h * (re[i] * re[i] + im[i] * im[i]);
          const c = Math.cos(ph), s = Math.sin(ph);
          const r = re[i] * c - im[i] * s; im[i] = re[i] * s + im[i] * c; re[i] = r;
        }
      };
    },
    /**
     * Any local nonlinearity that depends on the local intensity:
     *   rateFn(I, z, i, out):  out[0] = Re N,  out[1] = Im N     (per unit z)
     * e.g. quintic: out[1] = g3*I + g5*I*I ;  two-photon absorption: out[0] = -b*I
     */
    pointwise: function (rateFn) {
      const out = [0, 0];
      return function (re, im, h, z) {
        for (let i = 0; i < re.length; i++) {
          rateFn(re[i] * re[i] + im[i] * im[i], z, i, out);
          const a = Math.exp(h * out[0]), ph = h * out[1], c = a * Math.cos(ph), s = a * Math.sin(ph);
          const r = re[i] * c - im[i] * s; im[i] = re[i] * s + im[i] * c; re[i] = r;
        }
      };
    },
    /**
     * Generalised NLSE nonlinear operator (Agrawal convention, see header):
     *   dA/dz = i gamma (1 + i tShock d/dT) [ A * (R (x) |A|^2)(T) ]
     *   tShock = 1/omega0   (self-steepening time, in grid time units; 0 = off)
     *   raman  = { TR }                  intrinsic (Gordon) approximation, R (x) I ~ I - TR dI/dT
     *          | { fR, tau1, tau2 }      full causal response, hR(t) = (tau1^2+tau2^2)/(tau1 tau2^2)
     *                                    exp(-t/tau2) sin(t/tau1); R = (1-fR) delta + fR hR
     *   Needs the grid: { N, dt | T }.  substeps: RK4 sub-steps per nonlinear step (default 1).
     *   The nonlinear step is integrated with RK4 (it contains a spectral derivative, so it is
     *   not a pure phase), so keep  h * 3 gamma tShock I_max * pi/dt  well below ~2 (CFL-type limit);
     *   raise `substeps` or reduce dz if the step is unstable.
     */
    gnlse: function (o) {
      const N = o.N, g = makeGrid(N, { dt: o.dt, T: o.T }), w = g.omega, gamma = o.gamma;
      const s = o.tShock || 0, sub = o.substeps || 1, ram = o.raman || null;
      const Rre = new Float64Array(N).fill(1), Rim = new Float64Array(N);   // spectral multiplier of the response
      if (ram && ram.TR) { for (let k = 0; k < N; k++) Rim[k] = ram.TR * w[k]; }   // 1 - TR*(-i w) = 1 + i TR w
      else if (ram && ram.tau1) {
        const fR = ram.fR, t1 = ram.tau1, t2 = ram.tau2, hr = new Float64Array(N), hi = new Float64Array(N);
        let sum = 0;
        for (let n = 0; n < N / 2; n++) { const t = n * g.dt; hr[n] = (t1 * t1 + t2 * t2) / (t1 * t2 * t2) * Math.exp(-t / t2) * Math.sin(t / t1); sum += hr[n] * g.dt; }
        for (let n = 0; n < N / 2; n++) hr[n] *= g.dt / sum;                  // exact unit area on the grid
        fft(hr, hi, false);
        for (let k = 0; k < N; k++) { Rre[k] = (1 - fR) + fR * hr[k]; Rim[k] = fR * hi[k]; }
      }
      const useR = !!ram, I = new Float64Array(N), Ii = new Float64Array(N), Br = new Float64Array(N), Bi = new Float64Array(N), Xr = new Float64Array(N), Xi = new Float64Array(N);
      const y = [new Float64Array(N), new Float64Array(N)], k = [0, 1, 2, 3].map(() => [new Float64Array(N), new Float64Array(N)]), yt = [new Float64Array(N), new Float64Array(N)];
      function rhs(ar, ai, outR, outI) {
        for (let n = 0; n < N; n++) I[n] = ar[n] * ar[n] + ai[n] * ai[n];
        if (useR) {
          Ii.fill(0); fft(I, Ii, false);
          for (let q = 0; q < N; q++) { const a = I[q] * Rre[q] - Ii[q] * Rim[q], b = I[q] * Rim[q] + Ii[q] * Rre[q]; I[q] = a; Ii[q] = b; }
          fft(I, Ii, true);                                                   // I now holds R (x) |A|^2 (real)
        }
        for (let n = 0; n < N; n++) { Br[n] = ar[n] * I[n]; Bi[n] = ai[n] * I[n]; }
        for (let n = 0; n < N; n++) { outR[n] = -gamma * Bi[n]; outI[n] = gamma * Br[n]; }          // i gamma B
        if (s) {
          Xr.set(Br); Xi.set(Bi); fft(Xr, Xi, false);
          for (let q = 0; q < N; q++) { const a = w[q] * Xi[q], b = -w[q] * Xr[q]; Xr[q] = a; Xi[q] = b; }   // -i w X
          fft(Xr, Xi, true);
          for (let n = 0; n < N; n++) { outR[n] -= gamma * s * Xr[n]; outI[n] -= gamma * s * Xi[n]; }       // - gamma s dB/dT
        }
      }
      return function (re, im, h) {
        const hs = h / sub;
        for (let st = 0; st < sub; st++) {
          rhs(re, im, k[0][0], k[0][1]);
          for (let n = 0; n < N; n++) { yt[0][n] = re[n] + 0.5 * hs * k[0][0][n]; yt[1][n] = im[n] + 0.5 * hs * k[0][1][n]; }
          rhs(yt[0], yt[1], k[1][0], k[1][1]);
          for (let n = 0; n < N; n++) { yt[0][n] = re[n] + 0.5 * hs * k[1][0][n]; yt[1][n] = im[n] + 0.5 * hs * k[1][1][n]; }
          rhs(yt[0], yt[1], k[2][0], k[2][1]);
          for (let n = 0; n < N; n++) { yt[0][n] = re[n] + hs * k[2][0][n]; yt[1][n] = im[n] + hs * k[2][1][n]; }
          rhs(yt[0], yt[1], k[3][0], k[3][1]);
          for (let n = 0; n < N; n++) {
            re[n] += hs / 6 * (k[0][0][n] + 2 * k[1][0][n] + 2 * k[2][0][n] + k[3][0][n]);
            im[n] += hs / 6 * (k[0][1][n] + 2 * k[1][1][n] + 2 * k[2][1][n] + k[3][1][n]);
          }
        }
      };
    },
    /** Apply several nonlinear steps one after another within the same sub-step. */
    compose: function () {
      const fns = Array.prototype.slice.call(arguments);
      return function (re, im, h, z) { for (let j = 0; j < fns.length; j++) fns[j](re, im, h, z); };
    }
  };

  /* ---------------------------------------------------------------------
     7. Analysis helpers
  --------------------------------------------------------------------- */
  const analysis = {
    energy: function (re, im, dt) {
      let E = 0; for (let i = 0; i < re.length; i++) E += re[i] * re[i] + im[i] * im[i];
      return E * (dt === undefined ? 1 : dt);
    },

    /**
     * Peak-centred pulse metrics on a periodic grid. The intensity trace is rolled so the
     * maximum sits at index N/2 before moments and FWHM are measured, so pulses straddling
     * the window edge are handled. FWHM is the width of the main peak (outward scan from the
     * maximum, linear interpolation at the half-maximum crossings); NaN if the pulse does not
     * fall below half maximum inside the window.
     * Returns { energy, peak, peakIdx, shift, Irot, fwhm, centroid, rms }  (times in units of dt)
     */
    pulseMetrics: function (re, im, dt) {
      dt = dt === undefined ? 1 : dt;
      const N = re.length, I = new Float64Array(N);
      let peak = 0, peakIdx = 0, E = 0;
      for (let i = 0; i < N; i++) { I[i] = re[i] * re[i] + im[i] * im[i]; E += I[i]; if (I[i] > peak) { peak = I[i]; peakIdx = i; } }
      const c = N >> 1, shift = c - peakIdx, Irot = new Float64Array(N);
      for (let i = 0; i < N; i++) Irot[((i + shift) % N + N) % N] = I[i];
      const half = peak / 2;
      let xl = NaN, xr = NaN;
      for (let i = c; i > 0; i--) if (Irot[i - 1] < half) { xl = (i - 1) + (half - Irot[i - 1]) / (Irot[i] - Irot[i - 1]); break; }
      for (let i = c; i < N - 1; i++) if (Irot[i + 1] < half) { xr = i + (Irot[i] - half) / (Irot[i] - Irot[i + 1]); break; }
      let m0 = 0, m1 = 0, m2 = 0;
      for (let i = 0; i < N; i++) { const x = i - c; m0 += Irot[i]; m1 += x * Irot[i]; m2 += x * x * Irot[i]; }
      const mean = m0 > 0 ? m1 / m0 : 0, varr = m0 > 0 ? m2 / m0 - mean * mean : 0;
      return {
        energy: E * dt, peak: peak, peakIdx: peakIdx, shift: shift, Irot: Irot,
        fwhm: (xr - xl) * dt, centroid: (mean) * dt, rms: Math.sqrt(Math.max(varr, 0)) * dt
      };
    },

    /** |A~(w)|^2 on an ascending frequency axis. Returns { omega, power }. */
    spectrum: function (re, im, grid) {
      const N = re.length, r = Float64Array.from(re), i2 = Float64Array.from(im);
      fft(r, i2, false);
      const power = new Float64Array(N), omega = new Float64Array(N);
      for (let j = 0; j < N; j++) {
        const kk = -(j - N / 2), k = ((kk % N) + N) % N;
        power[j] = r[k] * r[k] + i2[k] * i2[k];
        omega[j] = (j - N / 2) * grid.dw;
      }
      return { omega: omega, power: power };
    },

    /** Power-weighted mean angular frequency of the spectrum (same sign convention as spectrum()). */
    spectralCentroid: function (re, im, grid) {
      const sp = analysis.spectrum(re, im, grid); let m0 = 0, m1 = 0;
      for (let j = 0; j < sp.power.length; j++) { m0 += sp.power[j]; m1 += sp.power[j] * sp.omega[j]; }
      return m1 / m0;
    },

    /** Instantaneous frequency -dphi/dT (rad per unit T), finite difference of the unwrapped phase. */
    chirp: function (re, im, dt) {
      const N = re.length, out = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const j = (i + 1) % N, k = (i - 1 + N) % N;
        // phase difference via conj product avoids unwrapping
        const dr = re[j] * re[k] + im[j] * im[k], di = im[j] * re[k] - re[j] * im[k];
        out[i] = -Math.atan2(di, dr) / (2 * dt);
      }
      return out;
    },

    /** Sampled fields */
    sech: function (grid, T0, amp) { const A = { re: new Float64Array(grid.N), im: new Float64Array(grid.N) }; for (let i = 0; i < grid.N; i++) A.re[i] = (amp === undefined ? 1 : amp) / Math.cosh(grid.t[i] / T0); return A; },
    gaussian: function (grid, T0, amp) { const A = { re: new Float64Array(grid.N), im: new Float64Array(grid.N) }; for (let i = 0; i < grid.N; i++) A.re[i] = (amp === undefined ? 1 : amp) * Math.exp(-0.5 * Math.pow(grid.t[i] / T0, 2)); return A; },

    /** Fibre-optics soliton bookkeeping (anomalous beta2 < 0, gamma > 0). */
    soliton: function (beta2, gamma, T0, order) {
      order = order || 1;
      const LD = T0 * T0 / Math.abs(beta2), P0 = order * order * Math.abs(beta2) / (gamma * T0 * T0);
      return { LD: LD, z0: Math.PI / 2 * LD, P0: P0, LNL: 1 / (gamma * P0), order: order };
    },

    copy: function (A) { return { re: Float64Array.from(A.re), im: Float64Array.from(A.im) }; },
    maxDiff: function (A, B) { let m = 0; for (let i = 0; i < A.re.length; i++) m = Math.max(m, Math.hypot(A.re[i] - B.re[i], A.im[i] - B.im[i])); return m; }
  };

  /* ---------------------------------------------------------------------
     8. ConvergenceMonitor -- adaptive stopping for round-trip maps.
        Extracted from runSelfStart() of the mode-locking explorer:
        after minRounds, every checkEvery rounds compare the monitored scalar
        (e.g. pulse energy) with the previous check; converged once
        stableWindow consecutive checks all agree within tol (relative), or both
        values sit below noiseFloor (a stochastic near-zero state is not "moving").
  --------------------------------------------------------------------- */
  class ConvergenceMonitor {
    constructor(o) {
      o = o || {};
      this.minRounds = o.minRounds !== undefined ? o.minRounds : 100;
      this.checkEvery = o.checkEvery || 5;
      this.stableWindow = o.stableWindow || 16;
      this.tol = o.tol !== undefined ? o.tol : 1e-3;
      this.noiseFloor = o.noiseFloor !== undefined ? o.noiseFloor : 1e-2;
      this.reset();
    }
    reset() { this.last = null; this.stable = 0; this.converged = false; }
    /** Call once per round with the monitored scalar; returns true when converged. */
    update(round, value) {
      if (round >= this.minRounds && round % this.checkEvery === 0) {
        if (this.last !== null) {
          const rel = Math.abs(value - this.last) / Math.max(value, this.last, 1e-9);
          const quiet = value < this.noiseFloor && this.last < this.noiseFloor;
          this.stable = (rel < this.tol || quiet) ? this.stable + 1 : 0;
          if (this.stable >= this.stableWindow) this.converged = true;
        }
        this.last = value;
      }
      return this.converged;
    }
  }

  /* ---------------------------------------------------------------------
     9. HausModelocking -- one cavity round trip per step()
        Order inside a round trip (identical to the explorer):
          SPM -> saturable absorber -> additive noise -> gain (saturated by
          the energy at the START of the round trip) x parabolic gain filter
          x dispersion, in the frequency domain.
        params: g0, l, Esat, Dg, D, gamma, q0, EsatA, tauA, noise
        Units are normalised; dt = 1 time bin.  D is the net cavity group-delay
        dispersion beta2*L: the dispersion phase is exp(+i D w^2 / 2), so D < 0
        is anomalous and (with gamma > 0) supports solitons; D > 0 is normal.
        (v0.1.0 and the original applet had the opposite sign, exp(-i D w^2/2),
        which contradicted the applet's own "D < 0 = anomalous" label.)

        absorber modes
          'fastTime' (default) q is advanced along the fast time axis inside each
                     round trip with an exact exponential step per bin, so tauA is
                     a recovery time in time bins, as the applet's labels say
                     (a SESAM recovers within one round trip).
          'legacy'   the original applet code, kept only for regression: q_i is
                     advanced ONCE PER ROUND TRIP by an explicit Euler step, so
                     tauA acts as a recovery time in round trips (a bug).
        tauA <= 1e-9 selects the instantaneous (KLM-like) absorber in both.
  --------------------------------------------------------------------- */
  class HausModelocking {
    constructor(o) {
      o = o || {};
      this.N = o.N || 256;
      this.grid = makeGrid(this.N, { dt: 1 });
      this.rng = o.rng || makeRng(o.seed);
      this.mode = o.absorberMode || 'fastTime';
      this.p = Object.assign({ g0: 0.6, l: 0.1, Esat: 400, Dg: 2, D: -0.02, gamma: 0.005, q0: 0.3, EsatA: 40, tauA: 0, noise: 1e-3 }, o.params || {});
      this.re = new Float64Array(this.N); this.im = new Float64Array(this.N); this.q = new Float64Array(this.N);
      this.reset(o.initAmp !== undefined ? o.initAmp : 0.01);
    }
    reset(initAmp) {
      const N = this.N;
      for (let i = 0; i < N; i++) { this.re[i] = initAmp * this.rng.randn(); this.im[i] = initAmp * this.rng.randn(); }
      this.q.fill(this.p.q0);
      this.qCarry = this.p.q0;
      this.round = 0;
    }
    setParams(patch) { Object.assign(this.p, patch); }
    energy() { return analysis.energy(this.re, this.im, 1); }

    step() {
      const N = this.N, p = this.p, re = this.re, im = this.im, q = this.q, w = this.grid.omega;
      let E = 0; for (let i = 0; i < N; i++) E += re[i] * re[i] + im[i] * im[i];

      for (let i = 0; i < N; i++) {                              // SPM
        const ph = p.gamma * (re[i] * re[i] + im[i] * im[i]);
        const c = Math.cos(ph), s = Math.sin(ph);
        const nr = re[i] * c - im[i] * s, ni = re[i] * s + im[i] * c; re[i] = nr; im[i] = ni;
      }

      if (p.tauA <= 1e-9) {                                      // instantaneous saturable absorber
        for (let i = 0; i < N; i++) {
          const I = re[i] * re[i] + im[i] * im[i];
          const s = Math.sqrt(Math.max(1 - p.q0 / (1 + I / p.EsatA), 0));
          re[i] *= s; im[i] *= s;
        }
      } else if (this.mode === 'legacy') {                       // per-round-trip Euler recovery
        for (let i = 0; i < N; i++) {
          const I = re[i] * re[i] + im[i] * im[i];
          q[i] = Math.max(q[i] + (-(q[i] - p.q0) / p.tauA - q[i] * I / p.EsatA), 0);
          const s = Math.sqrt(Math.max(1 - q[i], 0));
          re[i] *= s; im[i] *= s;
        }
      } else {                                                   // fast-time exact exponential recovery
        let qp = this.qCarry;
        for (let i = 0; i < N; i++) {
          const I = re[i] * re[i] + im[i] * im[i];
          const kap = 1 / p.tauA + I / p.EsatA, qeq = (p.q0 / p.tauA) / kap;
          qp = qeq + (qp - qeq) * Math.exp(-kap);                // dt = 1 bin
          q[i] = qp;
          const s = Math.sqrt(Math.max(1 - qp, 0));
          re[i] *= s; im[i] *= s;
        }
        this.qCarry = qp;
      }

      for (let i = 0; i < N; i++) { re[i] += p.noise * this.rng.randn(); im[i] += p.noise * this.rng.randn(); }

      const gSat = p.g0 / (1 + E / p.Esat);                      // gain, gain filter, dispersion
      fft(re, im, false);
      for (let k = 0; k < N; k++) {
        const wk = w[k], a = Math.exp(gSat - p.l - p.Dg * wk * wk), ph = 0.5 * p.D * wk * wk;
        const c = a * Math.cos(ph), s = a * Math.sin(ph);
        const nr = re[k] * c - im[k] * s, ni = re[k] * s + im[k] * c; re[k] = nr; im[k] = ni;
      }
      fft(re, im, true);
      this.round++;
      return E;
    }

    /**
     * run({ maxRounds=4000, snapshotEvery=15, monitor }) -> { snapshots, rounds, converged }
     * snapshots: [{ rt, re, im }] copies, including round 0 and the final state.
     */
    run(o) {
      o = o || {};
      const maxRounds = o.maxRounds || 4000, every = o.snapshotEvery || 15;
      const monitor = o.monitor || new ConvergenceMonitor();
      const snapshots = [];
      let finalRT = maxRounds, converged = false;
      for (let r = 0; r <= maxRounds; r++) {
        if (r % every === 0) snapshots.push({ rt: r, re: this.re.slice(), im: this.im.slice() });
        this.step();
        if (monitor.update(r, this.energy())) { converged = true; finalRT = r; break; }
      }
      snapshots.push({ rt: finalRT, re: this.re.slice(), im: this.im.slice() });
      return { snapshots: snapshots, rounds: finalRT, converged: converged };
    }
  }

  return {
    VERSION: VERSION,
    makeRng: makeRng, fft: fft, makeGrid: makeGrid,
    rk4Step: rk4Step, integrate: integrate,
    linearSymbol: linearSymbol, NLSE: NLSE, nl: nl,
    analysis: analysis, ConvergenceMonitor: ConvergenceMonitor, HausModelocking: HausModelocking
  };
});
