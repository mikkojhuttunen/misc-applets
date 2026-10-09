# cmpc-ngrc

Math engine for using an **on-chip chaotic multipass cell (CMPC)** as a physical reservoir for
**next-generation reservoir computing (NGRC)**.

Small effective-index perturbations Δn_eff(x, y) (roughly round shapes, possibly slightly deformed)
are placed in the slab of a CMPC. The engine traces slab-mode rays and Gaussian beamlets ("complex
rays") through the cell. The rays bend in the index gradients and pick up phase k₀∫n_eff ds, and are
reflected by the cell walls with reflectance R(χ). The beamlets are summed coherently at the output
into a speckle field. That speckle is the feature vector. A linear or kernel readout (ridge NGRC
readout, SVM/SVR) is then trained to return the **harmonic decomposition** of the shapes: their
circular-harmonic coefficients (the 2D analogue of a spherical-harmonic decomposition, SHD), or the
Y_lm coefficients when the perturbation is treated as a 3D star-shaped bump.

The question to answer: can complex ray propagation in a CMPC, plus a simple trained readout, do or
imitate SHD of the inserted shapes, and how many detectors, launch conditions and samples does it need?

> **Status: planning / scaffolding.** Nothing is implemented yet. [TASKS.md](TASKS.md) has the work
> plan, the physics choices and the acceptance tests for each module.

Lives in [`misc-applets`](https://github.com/mikkojhuttunen/misc-applets) next to `math-engines/`
(`billiard_cell`, `membrane_mode`, `slab_waveguide`, `cell_mirror`, `path_coherence`), `general-mpc/`
(`gmpc.planar` walls) and `cmpc-sim/`, and reuses them instead of re-deriving cell geometry, slab
modes and mirror reflectance.

## Planned layout

```
ngrc/
  geometry.py      cell wall (from gmpc.planar / billiard_cell), slab stack, n_eff(λ), ports, detector line
  index_map.py     Δn_eff grid: image import, shape rasterisation, smoothing, bicubic interpolation of n and ∇n
  shapes.py        round / deformed shapes r(θ) = R[1 + Σ a_m cos mθ + b_m sin mθ], 3D bumps, ground-truth harmonics
  rays.py          ray equation d/ds(n dr/ds) = ∇n (RK4 / adaptive), wall hits, R(χ), optical path, phase
  beamlets.py      complex-ray / Gaussian-beamlet (dynamic ray tracing: Q, P along the ray), field at a point
  field.py         coherent sum at the output aperture → complex field E(y_d), intensity, far field
  perturbative.py  fast first-order mode: unperturbed rays + phase screen k₀∫Δn ds (+ first-order bending)
  wave_ref.py      2D wave-optics reference (split-step BPM / FDFD Helmholtz) for small cells, validation only
  features.py      speckle → feature vectors: multiplexing over launch angle / wavelength, NGRC monomials
  harmonics.py     circular-harmonic and Y_lm decomposition of shapes and of Δn maps (labels and baselines)
  dataset.py       random shape ensembles → (Δn map, labels, speckle features), cached to .npz
  readout.py       ridge (NGRC), linear / RBF SVM and SVR, cross-validation, metrics
  analysis.py      Jacobian dE/da_m, effective rank, decorrelation vs Δn, noise and detector-count sweeps
examples/          scripted figures
tests/             pytest suite (analytic GRIN cases, reciprocity, energy, ray ↔ wave agreement)
```

```
cd cmpc-ngrc
pip install -e ".[ml,figures,images,test]"
python -m pytest -q
```

SI units, numpy/scipy core; scikit-learn only in the readout layer.
