# cmpc-ngrc

Math engine for using an **on-chip chaotic multipass cell (CMPC)** as a physical reservoir for
**next-generation reservoir computing (NGRC)**.

Small effective-index dots Δn_eff(x, y) (round, slightly deformed) sit in the slab of a circular cell with
any number of ports in the wall. Slab-mode rays carrying Gaussian beamlets ("complex rays") are traced
from every input port. They bend in the index gradients, pick up the phase k₀∫n ds, reflect with R(χ),
and leave through the ports. The coherent beamlet sum on a detector line outside each output port is the
speckle field. The speckle intensities over all (input, output) pairs are the features. A ridge (NGRC)
or SVM readout is trained to return the **circular-harmonic decomposition (CHD)** of the dot contour,
r(θ) = R[1 + Σ_m (a_m cos mθ + b_m sin mθ)].

Plan and status: [TASKS.md](TASKS.md). Interactive explorer: [web/cmpc-ngrc-explorer.html](web/cmpc-ngrc-explorer.html)
(open the file directly; no server needed).

> **Status: research code.** Ray and beamlet optics with a first-order phase-screen shortcut, no full-wave
> check yet (T5.2). The circular cell is integrable, so its behaviour differs from a chaotic stadium.

## Layout

```
ngrc/
  geometry.py      CircularCell, Port (angle, width, role in/out/inout, launch, fan), symmetric_ports, ports_at,
                   detector lines, membrane n_eff and ∂n_eff/∂t (math-engines membrane_mode)
  shapes.py        Shape: deformed tanh-edged dot, analytic Δn / ∇Δn / Hessian, bounding circle, safe RK4 step;
                   Perturbation (sum of components); random_shape
  index_map.py     GridIndex: bicubic Δn image (value, gradient, Hessian); from_image (PNG/TIFF/.npy); rasterize_shape
  harmonics.py     CHD: exact coefficients, contour of any Δn map, Fourier coefficients, power spectrum
  rays.py          tracer: exact straight segments + RK4 ray equation and complex Q, P (dynamic ray tracing) in the
                   perturbations; circular-mirror reflection; ports; modes "curved", "straight", "none"
  field.py         beamlet matrix G (pixels × exit rays), detector field, correlations
  perturbative.py  PhaseScreenModel: trace once, E = G @ exp(i k0 ∫Δn ds) for any set of dots
  features.py      stacked intensities (raw / relative / log / delta), NGRC monomials, detector noise
  dataset.py       Ensemble of random dots → labels (R, a_m, b_m, p_m, dominant m) and features
  readout.py       ridge with CV, linear / RBF SVR and SVC (scikit-learn), R²
  analysis.py      sensitivity ∂I/∂a_m, ray occupancy, phase-screen validity, decorrelation
examples/run_progress.py   experiments E1–E4 → results/progress.json
web/ngrc-engine.js         JS port of the engine (same algorithms, checked against Python)
web/explorer.src.html      applet source; tools/build_applet.py → web/cmpc-ngrc-explorer.html
tools/make_vectors.py, tools/check_js.mjs   Python → JS test vectors and the node check
tests/                     pytest suite
```

```
cd cmpc-ngrc
pip install numpy scipy scikit-learn pillow pytest
python -m pytest -q
python examples/run_progress.py            # or --quick
python tools/make_vectors.py && node tools/check_js.mjs
python tools/build_applet.py
```

## Use

```python
from ngrc.geometry import CircularCell, ports_at
from ngrc.rays import Source, trace
from ngrc.field import detector_field
from ngrc.perturbative import PhaseScreenModel
from ngrc.shapes import Perturbation, Shape
from ngrc.index_map import from_image

cell = CircularCell(radius=1e-3, n_eff=1.8, ports=ports_at([0, 67, 151, 238], inputs=(0, 1), width=20e-6,
                                                           launch=0.35, fan=0.4))
dots = Perturbation([Shape(2.5e-4, 1e-4, 80e-6, 1e-3, edge=3e-6, a=[0, 0.06, 0.03], b=[0, 0, 0, 0.02]),
                     from_image("blob.png", pitch=1e-6, dn=1e-3, x0=-3e-4, y0=-2e-4, smooth=2e-6)])
ex = trace(cell, Source(0, n_pos=3, n_ang=61), dots, mode="curved")    # full curved rays + beamlets
E = detector_field(cell, ex, port=2)                                   # complex speckle on 64 pixels
F = PhaseScreenModel(cell, [Source(i) for i in cell.inputs]).fields(dots)   # fast first-order fields
```

## Physics in short

- Between dots the slab is uniform: straight rays, exact beamlet propagation Q → Q + P s/n₀. Inside a
  dot's bounding circle: RK4 of dr/ds = t, dp/ds = ∇n, dL/ds = n and dQ/ds = P/n, dP/ds = (n_nn − 2n_n²/n) Q.
- Wall: specular reflection, amplitude √R, phase φ_R, tangential mirror lens P → P − 2n₀Q/(R_c cos χ).
- In the circle sin χ is conserved, so a launch angle χ leaves a caustic disk of radius R_c sin χ that no
  ray enters. Dots inside it are invisible (E1).
- Phase screen: same rays, only ΔL = ∫Δn ds. It equals the full result for Δn → 0. At Δn = 1e-4 the field
  correlation with curved rays is ≈ 0.97, at 1e-3 ≈ 0.84 (E4): ray bending is what makes the cell nonlinear.

## First results (`results/progress.json`, shown on the applet's Progress tab)

- **Launch angle (E1):** sensitivity peaks near 23° and drops to zero at 57°, where the dots (within 600 µm
  of the centre) are inside the caustic disk.
- **Port layout (E2, dot at one position, 400 samples):** two input ports beat one. Asymmetric 2-input:
  R² of a_m, b_m = 0.91 / 0.79 / 0.60 / 0.67 / 0.56 for m = 2…6. Symmetric 1-input: 0.83 / 0.61 / 0.45 / 0.42 / 0.29.
  Four inputs: 0.86 / 0.84 / 0.74 / 0.66 / 0.69. Eight symmetric ports, one input: 0.91 / 0.78 / 0.70 / 0.66 / 0.68.
- **Readouts (E3, asym4 · 2 in, 800 dots at one position):** NGRC ridge 0.96 / 0.86 / 0.83 / 0.76 / 0.73,
  linear ridge and linear SVR within 0.01–0.04 of that, RBF SVR similar. The same readouts on the raw Δn
  image reach only 0.74 / 0.64 / 0.52 / 0.43 / 0.33, so the speckle is a better feature map than the
  pixels.
- **Power p_m:** RBF SVR recovers p₂ (R² 0.62); linear readouts reach 0.3 for p₂ and nothing for m ≥ 3.
- **Random dot positions:** nothing is learned with 800 samples; the position change dominates the
  speckle. Translation-invariant features are the next task (T8.4).
