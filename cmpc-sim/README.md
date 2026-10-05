# cmpc-sim

Chip-scale chaotic multipass cell (CMPC) simulator: effective path length, evanescent coupling, mirrors, trace-gas spectra and interference noise for free-standing membrane cells. The physics lives in [`math-engines`](../math-engines/); this folder holds the v0.2 script API on top of it and the figure script.

| File | Role |
|---|---|
| `cmpc_sim.py` | Material table (Si, SiNx, Al₂O₃), `membrane_mode`, `DBR`, ray tracing, `evaluate`, `selftest`, `ASSUMPTIONS` → `membrane_mode`, `bragg_grating`, `billiard_cell` engines |
| `coherence_model.py` | Speckle / etalon path model → `path_coherence` engine |
| `gas_spectra.py` | Voigt spectra, cell transmission, shot noise → `trace_gas` engine |
| `run_figures.py` | All design assumptions in `CFG`; writes figs 1–10 to `figs/` |

```
cd cmpc-sim
python cmpc_sim.py                 selftest
python run_figures.py              all figures (about 3 min)
python run_figures.py 3 4          selected
```

Needs numpy, scipy, matplotlib. `math-engines` is found next to this folder (or via `$MISC_APPLETS`).

Gas lines are ILLUSTRATIVE unless a HITRAN export exists: `python ../math-engines/tools/fetch_hitran.py` (needs `pip install hitran-api` and access to hitran.org), or put a `hitran_lines.json` here. Both locations are git-ignored here; HITRAN data is not committed.

Changes against the v0.2 scripts (figures 1–4 and 6–9 are pixel-identical; fig 10 differs by one grey level):

- `coherence_model.prepare(min_cell_power=...)` was accepted but ignored; it now drops mode cells below that share of the detected power. Paths whose weight underflows to zero are dropped instead of giving 0/0 path lengths.
- Pair sums in `contrast` and `autocovariance` run in blocks, so memory no longer grows as (paths per cell)².
- `evaluate` only touches columns up to the last exit, saving memory on 12000 × 2500 tables.
- `trace_path` renormalises the direction after each reflection. In the curved-facet cell a 1e-18 m difference grows past 1 mm by bounce 60, so the sample trajectories in fig 5 look different; physics unchanged.
- `occupancy_map` sizes its grid from the cell outline, so facets pushed outward by `offset_rms` are no longer clipped.
- Shot noise uses the exact elementary charge.
- `np.trapezoid` falls back to `np.trapz` on numpy < 2.
