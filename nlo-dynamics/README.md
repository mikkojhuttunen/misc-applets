# nlo-dynamics

Dependency-free numerical engine for nonlinear optics and nonlinear dynamics, extracted and generalised
from the self-starting mode-locking simulation in
[`physics-applets/laserquiz-rewards/mode-locking-explorer.html`](https://github.com/mikkojhuttunen/physics-applets/blob/main/laserquiz-rewards/mode-locking-explorer.html)
(commit `a74e7d9`). One file, no build step:

```html
<script src="nlo-engine.js"></script>      <!-- browser global NLO -->
```
```js
const NLO = require('./nlo-engine.js');    // Node
```
or paste the file between `<script>` tags to keep a single-file applet.

## What is in it

| Piece | What it does |
|---|---|
| `NLO.fft(re, im, inverse)` | in-place radix-2 FFT, cached plans |
| `NLO.NLSE` | split-step Fourier solver for `dA/dz = D(ω)A + N(A)A`: Strang (2nd order, one FFT pair per step) or Lie (1st order) |
| `NLO.linearSymbol` / `linear:{}` | loss `alpha`, gain, parabolic gain filter `gainBW`, dispersion `beta:[β2,β3,…]`, or any `custom(ω)` |
| `NLO.nl` | nonlinear steps: `kerr(γ)`, `pointwise(rate(I,z,i,out))` for any intensity-local term (quintic, two-photon absorption, saturable absorption…), `compose(...)`, and `gnlse({N, T, gamma, tShock, raman})`: Kerr + **self-steepening** (`tShock = 1/ω0`) + **Raman**, either intrinsic (`{TR}`, Gordon) or the full causal response (`{fR, tau1, tau2}`) |
| `NLO.integrate` | generic ODE solver: fixed-step RK4 or adaptive Dormand–Prince RK45 (rate equations, Lorenz, pendulum, coupled-mode equations) |
| `NLO.analysis` | peak-centred `pulseMetrics` (periodic-safe FWHM, rms, energy), `spectrum`, `spectralCentroid`, `chirp`, `sech`, `gaussian`, `soliton` (LD, z0, P0), `energy`, `maxDiff` |
| `NLO.ConvergenceMonitor` | the applet's adaptive stopping rule for round-trip maps |
| `NLO.HausModelocking` | the applet's round-trip master-equation model (SPM, saturable absorber, noise, saturated gain, gain filter, dispersion), with the two applet errors corrected (see below) |
| `NLO.makeRng(seed)` | seedable RNG, so noisy runs are reproducible |

Convention: Agrawal, `A ∝ exp(−iωT)`, `D(ω) = −α/2 + g − g_BW ω² + i Σ β_m ω^m/m!`. β2 < 0 is anomalous and, with γ > 0,
supports bright solitons.

```js
const nlse = new NLO.NLSE({ N: 1024, T: 40, linear: { beta: [-1] }, gamma: 1 });  // normalised NLSE
const A = NLO.analysis.sech(nlse.grid, 1, 2);                                       // N = 2 soliton
const map = nlse.evolve(A, { dz: 0.004, steps: 400, every: 40 });                   // z, |A|², spectrum snapshots
```
```js
const N = 4096, T = 40;                                       // generalised NLSE: Raman + self-steepening
const g = new NLO.NLSE({ N, T, linear: { beta: [-1] },
  nonlinear: NLO.nl.gnlse({ N, T, gamma: 1, tShock: 0.02, raman: { TR: 0.01 } }) });
```
`node examples/soliton.js`, `node examples/modelocking.js` and `node examples/raman_soliton.js` are runnable demos.

The `gnlse` step contains a spectral derivative, so it is integrated with RK4 (`substeps` per step) instead of an exact phase
rotation; keep `h · 3γ·tShock·I_max · π/dt` well below ~2, or raise `substeps`. For pure Kerr use `nl.kerr`, which is exact.

## Validation (`npm test`, 36 checks)

- **Numerics:** FFT against a naive DFT, Parseval and round trip; RK4 observed order 3.94 and RK45 energy conservation on the
  nonlinear pendulum; Strang order 2.00, Lie order 1.01; merged-Strang equals the plain two-FFT-pair form.
- **Linear and soliton physics:** sech FWHM = 1.7627·T0 including a pulse straddling the window edge; sign convention
  (spectrum peak and chirp of `exp(−iω0T)` at +ω0, odd-order term moving the pulse to +T); Gaussian broadening
  √(1+(z/LD)²); `E(z) = E0·e^{−αz}`; fundamental soliton keeps shape and phase `exp(iz/2)` (max error 2.5·10⁻⁵); N = 2 soliton
  peak 4 → 16 at z0/2 and recurrence at z0; two-photon absorption against its analytic solution.
- **Self-steepening:** with no dispersion the intensity obeys `∂I/∂z = −3γ·tShock·I ∂I/∂T`, whose characteristics are exact
  before shock formation; the engine matches them to 1·10⁻¹². Energy is conserved; the peak moves to +T (trailing edge steepens).
- **Raman:** soliton self-frequency shift against Gordon's `dΩ/dz = −8 T_R|β2|/(15 T0⁴)`: ratio 0.999 for the intrinsic
  response, and within 2 % for the full `hR` response when `T_R = fR∫t·hR dt`.
- **Mode-locking model:** against a frozen copy of the original applet code (`test/legacy/`, commit `a74e7d9`) the engine
  reproduces `mlStep` to ~10⁻¹³ over 80 round trips (the frozen code has the old dispersion sign, so the comparison flips D), and
  the same stopping round (215) as its `runSelfStart`. The two corrections below have their own tests.

## Extraction map

| Applet (`mode-locking-explorer.html`) | Engine |
|---|---|
| `fft` | `NLO.fft` (twiddle table instead of recurrence) |
| `mlRandn` | `NLO.makeRng().randn` |
| `mlStep` | `HausModelocking.step` |
| `mlAnalyze` | `analysis.pulseMetrics` (FWHM now measured outward from the peak with interpolation, instead of the first half-maximum crossing anywhere on the grid, so noise bumps cannot truncate it) |
| `runSelfStart` loop and `SS_*` constants | `HausModelocking.run`, `ConvergenceMonitor` |

## Corrections to the original applet (made in v0.2.0 and in the applet)

The first extraction exposed two errors in `mode-locking-explorer.html`; both are fixed in the applet and here.

1. **Dispersion sign.** The applet applied `exp(−iDω²/2)` together with SPM phase `+γI`, so `D > 0` was anomalous while the
   UI called `D < 0` anomalous (and defaulted to −0.02). With only those two operators, `D = +0.02` kept a fundamental soliton
   unchanged over 2000 round trips and `D = −0.02` spread it (FWHM 14.1 → 22.2 bins). The phase is now `exp(+iDω²/2)`, i.e.
   `D = β2·L`. Tests: `D = −0.02` preserves the soliton (FWHM ratio 0.9999), `D = +0.02` spreads it (1.57). At the default
   parameters dispersion is weak next to the gain filter and absorber, so the default pulses do not change visibly.
2. **SESAM recovery time.** The absorber state `q[i]` was advanced once per round trip, so `τ_A` was a recovery time in round
   trips although the UI says time bins. It is now integrated along the fast-time axis inside each round trip with an exact
   exponential step per bin, carrying `q` across the periodic boundary. Tests: relaxation to `q0` with e-fold `τ_A` bins, and a
   bleached bin recovering by e⁻¹ after `τ_A` bins (both exact). `absorberMode: 'legacy'` keeps the old behaviour for the
   regression only.

Consequence for the SESAM preset: with recovery now measured in bins, the old preset `τ_A = 25` mostly fails to settle in 4000
round trips (3 of 12 seeds converge, 8 of 12 end as one pulse), so the applet preset is now `τ_A = 10` (12 of 12 converge, 10 of
12 single pulse; KLM-like: 12 of 12 and 12 of 12). `τ_A = 5` gives 12 of 12 and 11 of 12. This is a judgement on the preset, not a
physics result; the slider still reaches 60.

## Not included (natural next steps)

Vector/coupled-mode NLSE, 2D transverse (beam-propagation) mode, adaptive step-size split-step, a Hamiltonian diagnostic, a
Python reference port for `math-engines/` (this folder is the JavaScript implementation).
