# Fiber parametric amplifier applet

Interactive simulator for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) fiber optical parametric
amplifiers with a signal near 1550 nm: pump depletion, LP01 fiber dispersion, gain bandwidth and idler
spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and
bandwidth optimisation over loss, fiber geometry, length and pump power.

| File | Role |
|---|---|
| `parametric-amplifier.html` | Front end: controls, plots, readouts, optimiser. No physics formulas for the solvers, dispersion, coupling or scattering. |
| `opa_engine.js` | DOM-free JavaScript port of the Python engines `materials`, `step_index_fiber` (LP01, `phase_mismatch`), `parametric_amplifier` and `stimulated_scattering` in `math-engines/`. |

Why a JavaScript port rather than Pyodide: the gain spectra, Δk scan and optimiser call the solvers tens
of thousands of times per update, which needs native JavaScript speed. Python stays the reference
implementation; `math-engines/tools/check_js_ports.mjs` checks the port against the shared vectors in
`math-engines/test_vectors/vectors.json`:

```bash
cd math-engines && python tools/make_vectors.py && cd ..
node math-engines/tools/check_js_ports.mjs
```

Open `parametric-amplifier.html` directly in a browser (it loads `opa_engine.js` from the same folder).
