# cmpc-ngrc

Math engine for using an **on-chip chaotic multipass cell (CMPC)** as a physical reservoir for
**next-generation reservoir computing (NGRC)**.

Small effective-index dots Δn_eff(x, y) (round, slightly deformed) sit in the slab of a circular cell with
any number of ports in the wall. The beam injected at every input port is expanded in frozen Gaussians
(Herman–Kluk "complex rays") that ride on slab-mode rays. The rays bend in the index gradients, pick up the
phase k₀∫n ds, reflect with R(χ), and leak smoothly through the port openings. The coherent sum over each
opening gives the far-field speckle of every output port. The speckle intensities over all (input, output)
pairs are the features. A ridge (NGRC) or SVM readout is trained to return the **circular-harmonic
decomposition (CHD)** of the dot contour, r(θ) = R[1 + Σ_m (a_m cos mθ + b_m sin mθ)].

Plan and status: [TASKS.md](TASKS.md). Interactive explorer: [web/cmpc-ngrc-explorer.html](web/cmpc-ngrc-explorer.html)
(open the file directly; no server needed).

> **Status: research code.** Semiclassical (frozen-Gaussian) optics, validated against exact Gaussian beams
> through apertures and mirrors and against BPM for one pass through a dot; no full-wave check of the
> multi-bounce speckle. Results use the first-order phase screen (exact ray paths of the empty cell, dot phases
> added), which is the Δn → 0 limit. Curved-ray datasets are possible but not converged at affordable ray
> counts (E7b). Model details: [docs/MODELS.md](docs/MODELS.md).

## Layout

```
ngrc/
  geometry.py      CircularCell (model "fga" | "gbs"), WallCell (any gmpc.planar wall, e.g. stadium), Port (position,
                   width, role in/out/inout, launch, fan), symmetric_ports, ports_at, wall_ports, far-field / line
                   detectors, cell variants (launch, λ), membrane n_eff and ∂n_eff/∂t (math-engines membrane_mode)
  shapes.py        Shape: deformed tanh-edged dot, analytic Δn / ∇Δn / Hessian, Radon table, chord integrals;
                   Perturbation; random_shape
  index_map.py     GridIndex: bicubic Δn image (value, gradient, Hessian); from_image (PNG/TIFF/.npy); rasterize_shape
  harmonics.py     CHD: exact coefficients, contour of any Δn map, Fourier coefficients, power spectrum
  rays.py          tracer: straight segments + RK4 (ray equation, Q/P) in the dots; wall reflection; frozen-Gaussian
                   launch grid and Herman–Kluk prefactor; smooth port leakage; modes "curved", "straight", "none"
  field.py         aperture sampling + far-field / Rayleigh–Sommerfeld propagation; legacy beamlets; correlations
  perturbative.py  PhaseScreenModel: trace once, E = K (B · exp(i k0 ΔL)) with ΔL from the dots' Radon tables
  jsengine.py      node_fields: the JS engine through node on all cores (datasets)
  wave_ref.py      split-step BPM reference for one pass through a dot
  features.py      stacked intensities (raw / relative / log / delta), NGRC monomials, detector noise
  dataset.py       Ensemble of random dots (fixed / random position, jitter) → labels and features
  readout.py       ridge with CV, kernel ridge (linear / poly2 / RBF, × position kernel), SVR / SVC, R²
  analysis.py      sensitivity ∂I/∂a_m, ray occupancy, phase-screen validity, decorrelation
examples/run_fga.py        experiments E1–E12 with the frozen-Gaussian model → results/progress.json
examples/run_progress.py, run_next.py   the same experiments with the legacy model (results/progress_gbs.json)
web/ngrc-engine.js         JS port of the engine (same algorithms, checked against Python)
web/explorer.src.html      applet source; tools/build_applet.py → web/cmpc-ngrc-explorer.html
tools/node_fields.mjs      worker-pool batch runner for jsengine.py
tools/make_vectors.py, tools/check_js.mjs   Python → JS test vectors and the node check
docs/MODELS.md             equations, assumptions, validation, limits
tests/                     pytest suite
```

```
cd cmpc-ngrc
pip install numpy scipy scikit-learn pillow pytest
python -m pytest -q
python examples/run_fga.py --only E3        # node needed for the datasets; --quick for small runs
python tools/make_vectors.py && node tools/check_js.mjs
python tools/build_applet.py
```

## Use

```python
from ngrc.geometry import CircularCell, ports_at
from ngrc.rays import Source, trace
from ngrc.field import detector_field
from ngrc.perturbative import PhaseScreenModel
from ngrc.jsengine import node_fields
from ngrc.shapes import Perturbation, Shape
from ngrc.index_map import from_image

cell = CircularCell(radius=1e-3, n_eff=1.8, reflectance=0.97,
                    ports=ports_at([0, 67, 151, 238], inputs=(0, 1), width=20e-6, launch=0.35, fan=0.4))
dots = Perturbation([Shape(2.5e-4, 1e-4, 80e-6, 1e-3, edge=3e-6, a=[0, 0.06, 0.03], b=[0, 0, 0, 0.02]),
                     from_image("blob.png", pitch=1e-6, dn=1e-3, x0=-3e-4, y0=-2e-4, smooth=2e-6)])
src = Source(0, n_pos=None, n_ang=2000, frozen=20e-6)                 # frozen-Gaussian launch grid
ex = trace(cell, src, dots, mode="curved", amp_min=0.02, max_bounces=300)
E = detector_field(cell, ex, port=2)                                   # complex far-field speckle, 64 directions
F = PhaseScreenModel(cell, [Source(i, n_pos=None, n_ang=2000) for i in cell.inputs]).fields(dots)
F_many = node_fields(cell, [None, Perturbation([dots.components[0]])], mode="phase", n_ang=6000)   # JS, all cores
```

## Physics in short

- Rays: straight in the uniform slab; RK4 of the ray equation and of the stability (Q, P) system inside the
  dots; specular wall reflection with √R, φ_R and the mirror's tangential lens.
- Field: frozen Gaussians of fixed width (20 µm) on a phase-space grid of launch positions × directions,
  Herman–Kluk prefactor from the stability matrix. A dot only affects the rays that cross it. The earlier
  Gaussian-beam summation is kept as `model="gbs"`; its beamlets grew to millimetres in the circle.
- Ports: smooth leakage (footprint overlap); the field across each opening is propagated to 64 far-field
  directions.
- Circle: sin χ is conserved, so a launch angle χ leaves a caustic disk of radius R_c sin χ that the core of
  the beam never enters (E1).
- Phase screen: the same rays with ΔL = ∫Δn ds from each dot's Radon table, about 1 s per sample in the JS
  engine. Ray bending changes the speckle from Δn ≈ 3e-4 (E4).

## Results (frozen-Gaussian model; `results/progress.json`, charted on the applet's Progress tab)

Cell R_c = 1 mm, n_eff = 1.8, λ = 1.55 µm, wall R = 0.97, 20 µm ports, launch 20°, fan 23°. Dots: R = 60–110 µm,
Δn = 1e-3, a_m and b_m (m = 2…6) random with spread 0.08/√(m−1). Phase-screen engine, 6000 directions per input.
R² below is for a_m, b_m (averaged over the pair) on held-out dots.

- **Launch angle (E1):** sensitivity peaks near 23° and is zero at 57°: dots within 600 µm of the centre are
  then inside the caustic disk.
- **Port layout (E2, 800 dots at one position, linear readout):** more inputs and more ports help. Four
  inputs: 0.92 / 0.86 / 0.72 / 0.69 / 0.62. Eight ports, one input: 0.93 / 0.82 / 0.60 / 0.70 / 0.55. Two
  inputs: 0.86–0.87 / 0.73–0.77 / … One input, four ports: 0.80–0.82 / 0.62–0.68 / …
  Symmetric and asymmetric placements differ by less than the sampling noise.
- **Readouts (E3, two inputs, 1600 dots at one position):** NGRC ridge (linear + quadratic)
  0.96 / 0.91 / 0.83 / 0.86 / 0.84, RBF SVR 0.95 / 0.89 / 0.80 / 0.83 / 0.81, linear ridge and linear SVR
  0.92 / 0.82 / 0.76 / 0.75 / 0.64. The same readouts on the raw Δn image: 0.72 / 0.57–0.63 / 0.45 / 0.44 / 0.30.
  Rotation-invariant powers p_m need the quadratic terms: p₂ 0.65 (NGRC), 0.73 (RBF SVR); p₃ 0.47.
- **Learning curve (E6):** NGRC mean R² 0.38 / 0.59 / 0.72 / 0.84 / 0.88 with 100 / 200 / 400 / 800 / 1200
  training dots, still rising.
- **Position (E5):** a readout trained at one position tolerates a 2 µm shift (R² 0.94 / 0.84 / 0.69 / 0.69 /
  0.66) and loses the higher orders by 5–10 µm (a₂ still 0.72 at 10 µm). Training with placement jitter keeps
  a₂ at 0.80 (10 µm) and 0.75 (20 µm). Dots anywhere in the central 600 µm: nothing is learned with 1600 dots,
  even with a readout that knows the position.
- **Multiplexing (E8, one input):** three wavelengths 0.2 nm apart, or three launch angles, raise R² from
  0.85 / 0.68 / 0.57 / 0.40 / 0.45 to 0.93 / 0.84 / 0.77 / 0.71 / 0.65 and 0.90 / 0.74 / 0.71 / 0.52 / 0.58;
  3 × 3 reaches 0.95 / 0.85 / 0.83 / 0.81 / 0.77. A 0.05 nm step already decorrelates the speckle (field
  correlation 0.37): long paths make wavelength a cheap source of independent views.
- **Δn (E12, 400 dots):** best between 3e-5 and 3e-4 without noise; at 3e-3 the dot phase becomes too large
  (a₂ 0.50). With 1 % detector noise, small Δn loses signal and the sweep flattens near 0.35–0.47 for a₂ at
  this sample size.
- **Noise and drift (E11, readout trained with the operating noise):** a₂ 0.92 / 0.83 / 0.76 at 0 / 1 / 3 %
  noise. A uniform index drift up to 1e-7 (≈ 5 mK for SiN) costs nothing. At 3e-7 the higher orders fail
  unless the features use an empty-cell reference recorded at the same temperature (then m ≤ 4 hold); at 1e-6
  both fail.
- **Curved rays (E7, E7b):** at the affordable 1500 directions, curved-ray datasets read out much worse than
  the phase screen on the same rays (a₂ 0.29 vs 0.93 at Δn = 1e-4). The curved and phase-screen responses to a
  small shape change correlate only 0.16–0.25, and improve slowly with more directions. Bending moves ray
  endpoints by ~0.1 µm, a ~1e-2 effect on 20 µm frozen Gaussians that cancels only in a converged sum, against
  a ~6e-4 rad shape signal. The effect of bending on learnability is therefore still open.
- **Validation (E9, tests):** FGA vs BPM, one pass through a dot: scattered-field correlation ≥ 0.9997.
- **Stadium vs circle (E10, Python engine, 300 dots, 2000 directions, dot 180 µm from the centre):** circle
  (R = 0.97) 0.44 / 0.42 / 0.23 / −0.05 / 0.00; the same circle with R = 0.80 (short paths)
  0.94 / 0.76 / 0.64 / 0.64 / 0.62; stadium 0.99 / 0.99 / 0.93 / 0.97 / 0.93 (p₂ 0.92, p₃ 0.72). In the
  chaotic stadium the frozen-Gaussian prefactors grow exponentially and rays are cut at about the Ehrenfest
  time (~10 bounces), so its field is a short-path field too. Shorter paths read out better, and chaotic
  mixing adds to that. Long-path chaotic speckle is beyond the semiclassical model.

Earlier results with the Gaussian-beam-summation model (wide beamlets, hard ports) are archived in
`results/progress_gbs.json`; they overstated readout quality and the stadium advantage.
