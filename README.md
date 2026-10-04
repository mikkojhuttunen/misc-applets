# misc-applets

Interactive applets, no build step, on top of a shared set of tested calculation engines in `math-engines/`. Each applet has its own folder at the root with an `index.html`; heavy physics that must run in the browser sits next to it in a UI-free `*-engine.js` port that is checked against the Python engines.

| Applet | Description |
|---|---|
| [parametric-amplifier/](parametric-amplifier/index.html) | Fiber parametric amplifier: coupled-wave RK4 solver for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) OPAs with a signal near 1550 nm. LP01 fiber dispersion, gain bandwidth and idler spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and bandwidth optimisation over loss, fiber geometry, length and pump power. |
| [dbr-structures/](dbr-structures/index.html) | Waveguide DBR and CRIGF calculator for thin-film LiNbO₃/LiTaO₃: Bragg-grating index contrast and reflection spectra, grating-coupler cavities pumped from free space, SHG with periodic poling and QPM bandwidth optimisation, poled straight sections, ridge/strip-loaded and curved-DBR lateral geometries, BOX/mirror directionality, pump depletion and photorefraction estimates. |
| [er-waveguide-amplifier/](er-waveguide-amplifier/index.html) | Er:Al₂O₃ strip-loaded waveguide amplifier on an Al₂O₃, Si₃N₄, TFLN or TFLT film with SiO₂ or air top cladding. Pump (980/1480 nm) and signal modes from the effective index method or a semi-vectorial 2D finite-difference solver, overlaps with the doped strip, steady-state rate equations with upconversion and quenching, pump/signal propagation, net gain spectrum, gain against length, pump power, Er concentration and quenching, and a design optimizer. Physics port: `er-engine.js` (engines `erbium_amplifier`, `channel_waveguide`). |
| [math-engines/](math-engines/) | Shared Python reference engines (Gaussian beams, Sellmeier materials, slab waveguides, Bragg gratings, QPM SHG, LP01 fiber) with machine-readable specs, tests and a [Pyodide web calculator](math-engines/web/index.html). |

## Engine principles

New calculation code follows these rules. The applets are front ends; the physics lives in engines.

1. **Engines are pure functions.** Inputs in, outputs out: no UI, no plotting, no file or network access, no global state. The same call always gives the same answer.
2. **SI inside, conversions at the edge.** Engines take and return SI units (m, s, W, rad, 1/m). nm, µm, mm, dB and percent exist only in front ends, through the `scale` in the spec: `si = ui × scale`. Wavelengths are vacuum wavelengths.
3. **Structured results.** A main engine function returns a `Result` with `values`, `units` and `assumptions`, so every number carries its unit and the model it came from. Small helpers used by other engines may return plain numbers or arrays.
4. **Machine-readable spec.** Each engine has a `spec.yaml` listing inputs (SI unit, UI unit, scale, default, range), outputs, equations, references and a version number. Forms, docs and checks are generated from it; a test fails if the spec and the code disagree.
5. **Hand-checkable tests.** Every engine has tests against values you can verify on paper or in a textbook, plus at least one independent route (an analytic limit, a conservation law such as R + T = 1, or a second method).
6. **Vectorize, keep dependencies light.** NumPy arrays in and out where it makes sense; NumPy first, SciPy only when needed and declared under `requires` in the spec. Everything must run in Pyodide.
7. **One reference implementation.** Python is the reference. JavaScript ports exist only for hot paths in interactive applets and are checked against shared JSON test vectors generated from the Python engines.
8. **Layers.** engines → composition (functions that combine engines) → front ends (applets, the web calculator, scripts). Lower layers never import higher ones.
9. **Validate inputs.** Reject non-physical input with a clear `ValueError` (`require_positive`, `require_range`, …) instead of returning NaN.
10. **Document conventions.** State sign conventions, field definitions (e.g. 1/e² radius) and approximations in the docstring and in `assumptions`.
11. **Version the spec.** Bump `version` when inputs, outputs or meanings change.

### Adding an engine

1. Create `math-engines/engines/<name>/` with `engine.py`, `spec.yaml`, `test_engine.py` and `README.md`.
2. Run `python tools/build_index.py` and `python tools/make_vectors.py` (if a JS port uses it) from `math-engines/`.
3. Run `python -m pytest -q` and `node tools/check_js_ports.mjs`. CI (`.github/workflows/math-engines-tests.yml`) runs both on every push that touches the engines or a port.

The engine appears in the web calculator automatically. With GitHub Pages enabled for this repository, the calculator is served at `https://mikkojhuttunen.github.io/misc-applets/math-engines/web/`.

## Keep design documentation out of this repo

This repository holds tool code only. Do not commit design reports, analysis write-ups, parameter-study results, device recommendations, measurement data or fabrication details: they can contain sensitive information. Keep them in separate, access-controlled documents and link to nothing sensitive from here. This applies to people and to AI assistants working in the repo.
