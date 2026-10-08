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
| [ray_phase](engines/ray_phase/) | Optical path and phase of rays vs launch angle in a segmented cell, pass by pass; fringe visibility under sine or triangle angle dither | |
| [cell_mirror](engines/cell_mirror/) | Etched-trench DBR reflectance averaged self-consistently over the angles of incidence in a (perturbed) segmented cell, vs the unperturbed cell; effective path and angle percentiles | |
| [thermo_optic](engines/thermo_optic/) | Thermo-optic LUT (dn/dT, k, ρ, c_p, expansion, band gap, TPA) for Si, SiO₂, Si₃N₄, SiNx, TFLN, TFLT, GaAs, AlGaAs, InP, AlN, Al₂O₃, TiO₂, polymer; slab-mode dn_eff/dT and resonance drift; heat from absorption, TPA, free carriers and quantum defect; closed-form thermal resistance; pump-power budget | |
| [waveguide_thermal](engines/waveguide_thermal/) | 2D finite-volume heat conduction in a waveguide cross-section (substrate / BOX / film / ridge / cladding): ΔT, R′, energy time constant, index and resonance shifts | scipy |
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
tools/build_index.py        specs → web/engines_index.json
tools/make_vectors.py       engines → test_vectors/vectors.json
tools/check_js_ports.mjs    JS ports in the applets checked against vectors.json
tools/make_thermo_lut.py    thermo_optic LUT → engines/thermo_optic/LUT.md and thermo_optic_lut.csv
tools/fetch_hitran.py       HITRAN line lists (HAPI) → engines/trace_gas/hitran_lines.json
web/index.html              Pyodide calculator
```

After changing a spec or an engine, regenerate and rerun:

```
python tools/build_index.py
python tools/make_vectors.py
python tools/make_thermo_lut.py
python -m pytest -q
node tools/check_js_ports.mjs
```

The tests fail if `web/engines_index.json`, `test_vectors/vectors.json` or the thermo-optic LUT exports are out of date.

## JavaScript ports

| Port | Covers | Checked by |
|---|---|---|
| `dbr-structures/dbr-engine.js` | Sellmeier indices, three-layer slab, coupled-mode reflectance, transfer-matrix stack | `check_js_ports.mjs` |
| `parametric-amplifier.html` (inline) | silica Sellmeier, Bessel ratios, LP01 solver | `check_js_ports.mjs` (functions extracted from the page) |
| `fringe-washout.html` (inline `engine:begin/end` block) | `fringe_averaging` harmonics, components, filters, residual visibility | `check_js_ports.mjs` (block extracted from the page) |
| `cmpc-ray-tracer.html` (inline `engine:begin/end` block) | `billiard_cell.SegmentedCell` geometry, hit test, reflection, launch, `reflection_weighted_path`; `ray_phase` path lengths, dither offsets, visibility, `dither_analysis`; `bragg_grating.stack_R_oblique`, `TrenchDBR`; `cell_mirror` statistics | `check_js_ports.mjs` (block extracted from the page) |

Known deviation: the ports work in µm inside; the check converts SI to µm at the boundary.
