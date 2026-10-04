# math-engines

Small, tested, UI-free physics engines: SI units in, a `Result` (values, units, assumptions) out.
The principles and layout follow [docs/math-engines-tutorial.md](docs/math-engines-tutorial.md).

| Engine | What it solves |
|---|---|
| [gaussian_beam](engines/gaussian_beam/) | TEM00 beam: Rayleigh range, w(z), R(z), Gouy phase, q parameter, ABCD law, thin-lens focusing (template engine) |
| [fiber_mode](engines/fiber_mode/) | Fused-silica index, step-index LP01 mode, χ⁽²⁾/χ⁽³⁾ phase mismatch, GVM and β₂ |
| [parametric_amplifier](engines/parametric_amplifier/) | χ⁽²⁾/χ⁽³⁾ fiber OPA: coupling, undepleted gain with idler-loss engineering, RK4 with depletion, bandwidth in nm |
| [stimulated_scattering](engines/stimulated_scattering/) | SBS (shift, gain, threshold, exact two-wave reflection) and SRS (gain spectrum, threshold) |

Dependencies point one way: front ends → composition → engines → NumPy and `engines/common.py`.
The parametric-amplifier engine uses `stimulated_scattering` for its optional scattering terms.

## Use

```bash
cd math-engines
python -m pip install numpy pyyaml pytest
python -m pytest -q                 # engine tests, spec checks, index and vector freshness
python tools/build_index.py         # after editing any spec.yaml
python tools/make_test_vectors.py   # after changing the amplifier physics; then run the JS test
node --test ../amplifiers/parametric-amplifiers/test/opa_engine.test.mjs
```

```python
from engines.gaussian_beam import engine as gb
gb.beam_parameters(w0=1e-3, wavelength=633e-9)["z_R"]          # 4.963 m

from engines.parametric_amplifier import engine as pa
c = pa.coupling("chi2", 532e-9, 1550e-9, 2.5e-6, 5e-6, 3.18e-6, pump_power=1.0, d_eff=0.08e-12)
pa.gain_bandwidth(c["gain_coefficient"], 5.0, 1550e-9, 532e-9, gvm=15.16e-12, beta2_sum=13e-27)["width_nm"]

from engines import registry                                      # bots: JSON-safe dict
registry.call("stimulated_scattering", "sbs_two_wave", pump_power=10, length=30, mode_area=10e-12, lambda_pump=1064e-9)
```

## Web calculator

`web/index.html` builds a form for every function in `web/engines_index.json` and runs the Python
engines in the browser through Pyodide (loaded from the jsDelivr CDN). With GitHub Pages enabled for
`main` (root) it is served at https://mikkojhuttunen.github.io/misc-applets/math-engines/web/.
To try it locally: `python -m http.server` from the repository root, then open
http://localhost:8000/math-engines/web/.

## Adding an engine

1. `cp -r engines/gaussian_beam engines/<new_name>`
2. Rewrite `engine.py` with pure SI functions returning `Result`.
3. Describe the exposed functions in `spec.yaml` (`version:` field; write exponents as `1.0e+6`).
4. Write tests whose expected values you can check by hand; cite the source in a comment.
5. `python tools/build_index.py`, then `python -m pytest -q`.
6. Add the package to `[tool.setuptools] packages` in `pyproject.toml`. Commit and push; CI runs the same checks.
