# waveguide-thermal

`thermal-engine.js`: plain-JS port of the thermal math engines in `math-engines/`, in SI units. `THERMALENGINE()` returns the API and can be rebuilt inside a Blob Web Worker with `Function.prototype.toString()`, the same pattern as `WGCORE` in `waveguide-core`.

| Function | Python reference |
|---|---|
| `entry`, `indexAt`, `LUT` | `thermo_optic` LUT; Sellmeier fits from `materials` |
| `stripThermalResistance`, `maxPower`, `kirchhoff`, `thermalRunaway` | `thermo_optic` |
| `buildRidge`, `ridgeHeating`, `ridgeMode`, `modeWeightedHeating`, `gridToMode` | `waveguide_thermal` (graded finite-volume heat solve, scalar finite-volume mode) |
| `erAmplifierHeating`, `erPumpLimit` | `amplifier_thermal` |
| `qpmThermal`, `phaseMatchingFactor`, `ringThermalBistability` | `thermal_detuning` |

Linear systems use banded Cholesky. The conduction matrix is symmetric positive definite, and so is σM − A for the mode at the shift σ = (k n_max)². The fundamental mode comes from inverse iteration at that shift, which is the same eigenpair scipy's shift-invert `eigsh` returns.

Checked by `node math-engines/tools/check_js_ports.mjs` against `math-engines/test_vectors/vectors.json` (`thermal` section): LUT values to 1e-12, ridge temperatures and τ_E to 1e-8, mode n_eff, n_g, dn_eff/dT and mode-weighted ΔT to 1e-6, and the runaway, Er amplifier, QPM and ring results to 1e-8. If you change the Python engines, regenerate the vectors and re-run the check.

Used by `pump-heating-planner.html` and `er-waveguide-amplifier.html` (the heating tile).
