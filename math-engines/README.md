# math-engines

Pure-function physics engines with machine-readable specs. One Python reference implementation per engine; the applets and the web calculator are front ends on top.

| Engine | Computes | Extra deps |
|---|---|---|
| [gaussian_beam](engines/gaussian_beam/) | Rayleigh range, divergence, w(z), R(z), Gouy phase, ABCD propagation, thin-lens focusing | |
| [materials](engines/materials/) | Sellmeier phase index, group index and GVD: SiO₂, Si₃N₄, Si, LiNbO₃ (e, o, temperature-dependent), LiTaO₃, KTP, BBO, AlN, GaAs, TiO₂, sapphire | |
| [slab_waveguide](engines/slab_waveguide/) | Three-layer and multilayer slab effective index, V, b, mode count | |
| [anisotropic_slab](engines/anisotropic_slab/) | TE/TM modes of slabs with anisotropic layers (x-, y-, z-cut uniaxial films, biaxial stacks), orientation helpers | |
| [bragg_grating](engines/bragg_grating/) | Grating coupling coefficient, coupled-mode peak reflectance and bandwidth, transfer-matrix stack spectra; oblique incidence (s/p, TIR), etched-trench membrane DBRs, two-wavelength DBR orders | |
| [qpm_shg](engines/qpm_shg/) | SHG phase mismatch, poling period, coherence length, sinc² acceptance bandwidth | |
| [step_index_fiber](engines/step_index_fiber/) | LP01 effective index, V, b, Marcuse mode-field radius | scipy |
| [membrane_mode](engines/membrane_mode/) | Free-standing membrane slab mode for evanescent sensing: n_eff, n_g, absorption factor Γ, penetration depth, surface-field (roughness) weight, evanescent gas volume | |
| [billiard_cell](engines/billiard_cell/) | 2D ray tracing of stadium and perturbed segmented-polygon multipass cells with ports; reusable ray tables re-weighted for any R(sin χ), loss and Γ; ergodic mean-field estimate, Poincaré sections, occupancy maps | |
| [trace_gas](engines/trace_gas/) | Voigt line shapes and absorption spectra of CH₄, NH₃, CO₂, H₂O (illustrative lines or a HITRAN export), transmission through a path-length distribution, shot-noise absorbance | scipy |
| [path_coherence](engines/path_coherence/) | Random-phase path model of speckle/etalon noise in multipass cells: contrast vs linewidth, spectral autocovariance, thermal decorrelation, simulated coherent spectra | |
| [ray_phase](engines/ray_phase/) | Optical path and phase of rays vs launch angle, pass by pass, in a segmented cell, any `planar_cell` wall or a 3D Herriott cell (`HerriottLaunch`); fringe visibility under sine or triangle angle dither | |
| [cell_mirror](engines/cell_mirror/) | Etched-trench DBR reflectance averaged self-consistently over the angles of incidence in a (perturbed) segmented cell or any `planar_cell` wall, vs the unperturbed cell; effective path and angle percentiles | |
| [planar_cell](engines/planar_cell/) | 2D ray tracing in walls of flat and curved mirror elements: circle, segmented polygon, smooth or faceted stadium, per-element perturbations, and the integrated (in-plane) Herriott cell; ports, path statistics, chaos, re-entrance; feeds `ray_phase` and `cell_mirror` | |
| [herriott_cell](engines/herriott_cell/) | Exact 3D tracing of Herriott, astigmatic and deformed Herriott cells (biconic/conic/polynomial mirrors, tilt, decentre, radius and spacing errors, holes, apertures); re-entrance, spot metrics, effective path, injection-angle dither | |
| [onchip_herriott](engines/onchip_herriott/) | Gaussian-beam budget of the integrated Herriott cell: effective path, mode volume and cross-section, power-weighted angles of incidence, throughput and clipping, with constant or trench-DBR mirrors (ray angle or beam angular spectrum), against mirror reflectance | |
| [fringe_averaging](engines/fringe_averaging/) | Residual interference-fringe visibility after angle and laser-frequency dithering, drifts, linewidth and averaging over seconds (exact harmonic expansion) | scipy (tests) |

## Use

```
pip install -e ".[test]"
python -m pytest -q
```

```python
from engines import registry
from engines.gaussian_beam.engine import beam_parameters

r = beam_parameters(wavelength=1.55e-6, w0=1.0e-3)
r["z_R"], r.units["z_R"], r.assumptions
registry.call("qpm_shg", "phase_matching", wavelength=1.55e-6, n_pump=2.138, n_sh=2.183)
```

Web calculator: `web/index.html` runs every engine in the browser through Pyodide, with forms generated from the specs. Serve the folder over HTTP (`python -m http.server` from the repository root, then open `/math-engines/web/`); on GitHub Pages it is at `https://mikkojhuttunen.github.io/misc-applets/math-engines/web/`.

## Layout

```
engines/
  common.py            Result type and input validation
  registry.py          engine discovery and JSON-safe calls (used by the web calculator)
  test_specs.py        every spec matches its engine signature and outputs
  <engine>/
    engine.py          pure functions, SI in and out
    spec.yaml          inputs (SI unit, UI unit, scale, default, range), outputs, equations, references, version
    test_engine.py     hand-checkable reference values and independent cross-checks
    README.md
test_vectors/vectors.json   reference outputs shared with the JavaScript ports
tools/build_index.py        specs → web/engines_index.json (engine files plus the engines they import)
tools/make_vectors.py       engines → test_vectors/vectors.json
tools/check_js_ports.mjs    JS ports in the applets checked against vectors.json
tools/fetch_hitran.py       HITRAN line lists (HAPI) → engines/trace_gas/hitran_lines.json
web/index.html              Pyodide calculator
```

After changing a spec or an engine, regenerate and rerun:

```
python tools/build_index.py
python tools/make_vectors.py
python -m pytest -q
node tools/check_js_ports.mjs
```

The tests fail if `web/engines_index.json` or `test_vectors/vectors.json` is out of date.

## JavaScript ports

| Port | Covers | Checked by |
|---|---|---|
| `dbr-structures/dbr-engine.js` | Sellmeier indices, three-layer slab, coupled-mode reflectance, transfer-matrix stack | `check_js_ports.mjs` |
| `parametric-amplifier.html` (inline) | silica Sellmeier, Bessel ratios, LP01 solver | `check_js_ports.mjs` (functions extracted from the page) |
| `fringe-washout.html` (inline `engine:begin/end` block) | `fringe_averaging` harmonics, components, filters, residual visibility | `check_js_ports.mjs` (block extracted from the page) |
| `cmpc-ray-tracer.html` (inline `engine:begin/end` block) | `billiard_cell.SegmentedCell` geometry, hit test, reflection, launch, `reflection_weighted_path`; `ray_phase` path lengths, dither offsets, visibility, `dither_analysis`; `bragg_grating.stack_R_oblique`, `TrenchDBR`; `cell_mirror` statistics | `check_js_ports.mjs` (block extracted from the page) |

| `planar-mpc-ray-tracer.html` (inline `engine:begin/end` block) | `planar_cell` builders, perturbations, hits, traces, integrated Herriott trace; `ray_phase` path lengths and `dither_analysis`; `cell_mirror` statistics with `TrenchDBR` | `check_js_ports.mjs` (block extracted from the page) |
| `onchip-herriott.html` (inline `engine:begin/end` block) | `onchip_herriott` mode, chord width integrals, erf and Gaussian clipping, angular-spectrum DBR average, `budget`, `footprint_fraction` (with the `planar_cell` and `bragg_grating` ports) | `check_js_ports.mjs` (block extracted from the page) |
| `herriott-ray-tracer.html` (inline `engine:begin/end` block) | `herriott_cell` surfaces, Newton intersections, `trace3d`, `build_cell`, astigmatic cells, `reentrance`, `spot_metrics`, `HerriottLaunch` paths and `dither_analysis` | `check_js_ports.mjs` (block extracted from the page) |

Known deviation: the ports work in µm inside; the check converts SI to µm at the boundary.
