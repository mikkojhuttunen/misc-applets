# Math Engines on GitHub

*A tutorial on the main idea and core principles*

## 1. The main idea

A math engine is a small, tested, UI-free piece of code that solves one well-defined physics problem: given inputs in SI units, return the physical quantities of interest, together with their units and the assumptions behind them. Gaussian beam propagation, ABCD ray matrices and cavity stability are typical examples.

The point of keeping engines separate from everything else is reuse. The same engine can then serve an offline problem set in a notebook, a browser applet, a Telegram bot, or a larger simulator that chains several engines together. You write and verify the physics once; every front end inherits it.

A GitHub repository is the natural home: it gives version history, review, automated tests on every push, and free hosting of the browser demo through GitHub Pages.

## 2. Core principles

- Engines are pure functions. Numbers in, results out. No file access, no printing, no plotting, no global state. This is what makes them testable and composable.
- SI units inside, conversions at the edge. Metres, seconds, radians, hertz, always. Unit conversion (mm, nm, mrad) belongs in the spec file or the UI, never inside the physics.
- Return structured results. A Result carries values, units and assumptions. A bare float hides what it means; a Result tells the next tool what it received.
- Every engine has a machine-readable spec. spec.yaml lists inputs, defaults, ranges, outputs, equations and references. Tools read it to build forms and documentation automatically, so a new engine gets a basic applet nearly for free.
- Tests have hand-checkable answers. Use analytic limits and textbook or homework values (for example z_R = pi w0^2 / lambda = 4.96 m for a 1 mm HeNe waist). If a test fails, the engine is wrong, not the test.
- Vectorise with NumPy. Accept arrays where sweeps are likely so scans and plots need no Python loops.
- Keep dependencies light. NumPy first, SciPy or SymPy only when needed. This keeps the engines runnable in the browser through Pyodide.
- One reference implementation. Python is the source of truth. A JavaScript port is justified only for small, hot-path engines, and then shared JSON test vectors keep the two versions from drifting apart.

## 3. Layers

| Layer | What it does | Examples |
|---|---|---|
| Engines | Pure physics functions with a common input/output convention | gaussian_beam, abcd_matrix, cavity_stability |
| Composition | Simulators and calculators that chain engines | stability, then eigenmode, then waist, then mode matching |
| Front ends | Anything a person or bot touches | notebooks, Pyodide web page, Telegram bot |

Dependencies point one way only: front ends use composition, composition uses engines, engines use nothing but NumPy and the shared common.py.

## 4. Repository layout

```text
misc-applets/
  .github/workflows/math-engines-tests.yml   # CI: run pytest on every push
  math-engines/
    engines/
      common.py          # Result dataclass, validation helpers
      registry.py        # call(engine, function, **kwargs) -> dict
      test_specs.py      # checks every spec.yaml against its engine.py
      gaussian_beam/
        engine.py        # the physics
        spec.yaml        # inputs, units, outputs, equations, references
        test_engine.py   # golden / analytic tests
        README.md
    tools/build_index.py # all spec.yaml files -> web/engines_index.json
    web/index.html       # generic Pyodide calculator built from the index
    docs/                # this tutorial
    pyproject.toml, README.md
```

## 5. Anatomy of an engine

### engine.py

Plain functions with docstrings that state units and conventions. Functions that a person or bot calls return a Result; small helpers used by other engines may return floats or arrays.

```python
def beam_at(z, w0, wavelength, z0=0.0, n=1.0, M2=1.0) -> Result:
    zr, dz, w, roc = _radius_and_roc(z, w0, wavelength, z0, n, M2)
    return Result(
        values={"w": w, "R": roc, "gouy_phase": np.arctan(dz / zr), "z_R": zr},
        units={"w": "m", "R": "m", "gouy_phase": "rad", "z_R": "m"},
        assumptions=["Paraxial TEM00", "w is the 1/e^2 intensity radius"],
    )
```

### spec.yaml

Describes each exposed function. scale converts a UI value to SI (si = ui * scale), so a field shown in millimetres has scale: 1.0e-3. Reusable inputs such as wavelength live under common_inputs and are referenced by name.

```yaml
functions:
  beam_at:
    title: Beam at a position z
    equations: ["w(z) = w0 sqrt(1 + ((z - z0)/z_R)^2)"]
    inputs:
      - {name: z, label: "Position z", ui_unit: mm, scale: 1.0e-3, default: 1000}
      - w0
      - wavelength
    outputs:
      - {name: w, label: "Beam radius w(z)", ui_unit: mm, scale: 1.0e-3}
```

### test_engine.py

Each test states where its expected value comes from. Typical sources are analytic limits, textbook results and your own homework model solutions (cite the source in a comment). The generic test_specs.py additionally checks that every spec matches its engine, so a renamed argument cannot silently break the web page.

## 6. Worked example: Gaussian beam

The included gaussian_beam engine covers the Rayleigh range, beam radius, wavefront curvature and Gouy phase, with M-squared and refractive-index scaling. It also provides the complex beam parameter q = (z - z0) + i z_R and the ABCD law q' = (Aq + B)/(Cq + D). Together these are the building blocks for resonators and beam trains.

```python
from engines.gaussian_beam import engine as gb
 
gb.beam_parameters(w0=1e-3, wavelength=633e-9)["z_R"]      # 4.963 m
gb.thin_lens_focus(w0=1e-3, s=0.5, f=0.1, wavelength=633e-9).values
 
# compose: waist -> 2 m of free space -> lens -> 0.3 m of free space
q = gb.q_parameter(0.0, 1e-3, 633e-9)
for M in (gb.free_space(2.0), gb.thin_lens(0.5), gb.free_space(0.3)):
    q = gb.abcd_propagate(q, M)
gb.beam_from_q(q, 633e-9)["w"]
```

## 7. Adding a new engine

- Copy the template: cp -r engines/gaussian_beam engines/<new_name>.
- Rewrite engine.py with pure SI functions returning Result.
- Describe the exposed functions in spec.yaml.
- Write tests with values you can verify by hand; run python -m pytest -q.
- Regenerate the browser index: python tools/build_index.py.
- Commit and push. CI runs the same tests.
If test_specs.py complains that web/engines_index.json is stale, you changed a spec without rebuilding the index; rerun step 5.

## 8. Putting it on GitHub

From the folder that contains the delivered math-engines/ and .github/ directories:

```bash
git clone https://github.com/mikkojhuttunen/misc-applets.git
cd misc-applets
git checkout -b add-math-engines
# copy math-engines/ and .github/ into this folder (merge .github if it exists)
git add math-engines .github
git commit -m "Add math-engines skeleton with gaussian_beam engine"
git push -u origin add-math-engines
# open a pull request, or merge into main once CI is green
```

To host the browser demo, enable GitHub Pages for the main branch (root). The generic calculator then lives at https://mikkojhuttunen.github.io/misc-applets/math-engines/web/. The page loads Pyodide from a CDN, fetches the engine files listed in engines_index.json, and runs them locally in the visitor's browser.

## 9. Using engines from other front ends

- Notebooks and scripts: import the engine module directly, or install the package with pip install -e ..
- Bots: call registry.call(engine, function, **kwargs); it returns a JSON-safe dictionary (infinities become null).
- Custom applets: reuse the Pyodide loading code from web/index.html, but replace the generic form with your own controls and plots.
- Simulators: import several engines and wire their outputs together; keep the simulator itself free of physics formulas.

## 10. Conventions and pitfalls

- State sign conventions in docstrings. Example: in this engine distance_to_waist > 0 means the waist lies downstream.
- Wavelength is always the vacuum wavelength; the medium enters through the index n.
- Validate inputs at the engine boundary (require_positive) so errors read as physics, not as NaNs.
- Never put UI units in engine code; a mm inside engine.py is a bug.
- Document assumptions in the Result. Paraxial, thin-lens and TEM00 limits are the usual ones.
- Check each engine against an independent route (analytic limit, second method, textbook figure) before trusting it.
- Version the spec (version: field) and note breaking changes in the engine README.

## 11. Where to grow

- Further engines: ABCD matrix library, cavity stability and eigenmodes, Fabry-Perot, rate equations, line broadening.
- A shared units.py or pint integration if unit mistakes become a problem.
- Shared JSON test vectors, so a JavaScript port can be verified against the Python reference.
- A notebooks/ folder with worked problems that call the engines, serving as both tutorials and regression tests.

## Notes for this repository

These additions follow from the engines that live here; they do not change the principles above.

- **Choice inputs.** Some functions take a non-numeric switch, e.g. `process: chi2 | chi3`. In
  `spec.yaml` such an input is written `{name: process, label: ..., choices: [chi2, chi3], default: chi2}`;
  it has no `ui_unit` or `scale`. Integer inputs add `integer: true`.
- **YAML exponents.** PyYAML reads `1.0e6` as a string. Write `1.0e+6`. `tools/build_index.py`
  refuses non-numeric scales, defaults and limits, so the mistake cannot reach the web page.
- **The JavaScript port.** `amplifiers/parametric-amplifiers/opa_engine.js` ports `fiber_mode`,
  `parametric_amplifier` and `stimulated_scattering`, because the amplifier applet runs the
  solvers tens of thousands of times per update. `tools/make_test_vectors.py` writes
  `engines/parametric_amplifier/test_vectors.json` from the Python reference, a pytest check
  fails if that file is stale, and `node --test amplifiers/parametric-amplifiers/test/opa_engine.test.mjs`
  checks the port against it. CI runs both.
