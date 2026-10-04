# DBR structures

Waveguide Bragg-grating and CRIGF (cavity-resonant integrated grating filter) calculator for thin-film χ⁽²⁾ waveguides. Self-contained page, no build step: open `index.html` in a browser (keep `dbr-engine.js` next to it).

## What it computes

- Slab waveguide with a surface-etched grating: local effective indices, index contrast, κ, reflection spectrum (transfer matrix), field along the grating, out-of-plane scattering at a probe wavelength.
- SHG generated inside the grating, periodic poling (QPM) with a chirp/apodisation optimiser for a target bandwidth.
- CRIGF: free-space Gaussian beam on a 2nd-order grating coupler between two DBRs (Kazarinov–Henry coupled-mode model), pump reflectance spectra, resonance lock and phase tuning.
- Poled straight sections between coupler and DBRs, with a length sweep.
- Lateral geometry: 2D slab, ridge/rib/strip-loaded guide, open slab with curved DBRs, wide coupler tapered to a rib.
- Vertical stack with BOX on a Si handle or a gold mirror (coupler directionality).
- Pump depletion and cavity loading, power sweep, photorefraction onset estimates (editable thresholds).
- Materials: SiO₂, Si₃N₄, LiNbO₃ (e, o), LiTaO₃ (approximate fit), Si, custom.

`Copy CSV` buttons export spectra and sweeps; `plot_pump_spectra.py` plots them:

```
pip install matplotlib numpy
python plot_pump_spectra.py crigf.csv dbr.csv --labels CRIGF "one DBR" --out spectra.png
python plot_pump_spectra.py crigf.csv --db --no-t --xlim 1549.5 1551
```

## Code structure

- `dbr-engine.js`: the physics (Sellmeier indices, slab effective index, coupled-mode and transfer-matrix reflectance, lateral and cavity models), UI-free. Its core functions are a port of the Python reference engines in `../math-engines/` and are checked against their test vectors with `node math-engines/tools/check_js_ports.mjs` from the repository root. Deviation: lengths are in µm inside the port.
- `index.html`: the UI, loads `dbr-engine.js`.

LiNbO₃ e and o Sellmeier coefficients were swapped in earlier versions (n_e at 1550 nm came out as about 2.21 instead of 2.13). The engine tests caught it; results for LiNbO₃ from earlier versions should be recomputed. LiTaO₃ was not affected.

## Limits

Design-stage models: effective-index method, first-order radiation coupling, undepleted-pump SHG unless the depletion panel is used, no thermal or Kerr effects. LiTaO₃ indices are an approximate fit (±0.01). Photorefraction thresholds are placeholders to be replaced with measured values. Check final designs with a mode solver and FDTD.

## Keep design documentation out of this repo

This repository holds tool code only. Do not commit design reports, analysis write-ups, parameter-study results, device recommendations, measurement data or fabrication details: they can contain sensitive information. Keep them in separate, access-controlled documents and link to nothing sensitive from here. This applies to people and to AI assistants working in the repo.
