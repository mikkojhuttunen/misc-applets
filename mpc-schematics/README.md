# mpc-schematics

Schematics of the multipass-cell structures used in the simulations, and a period-count estimate for the in-plane membrane DBR mirror. Both scripts use the simulators' own classes and engines, so the drawings show the traced geometry.

```
cd mpc-schematics
python draw_cells.py              cmpc_cells.png, trench_dbr.png, gmpc_stadiums.png, herriott_cells.png into figs/
python draw_cells.py herriott     one figure: cmpc | dbr | stadium | herriott
python dbr_periods.py             periods for R >= 0.99, Si membranes at 1.55 um
python dbr_periods.py --target 0.999 --plot --no-cell
```

## draw_cells.py

| Figure | Structure | Source |
|---|---|---|
| `cmpc_cells.png` | segmented circular cell, 24 facets, Ø 10 mm, star {24/7} launch (χ = 37.5°): regular, 1° rms tilt, curved facets ρ = 5 cm, the last two also exaggerated; stadium r = a = 5 mm | `cmpc_sim.SegmentedCell`, `cmpc_sim.Stadium` |
| `trench_dbr.png` | etched-trench DBR on a facet, cavity \| [air gap, Si tooth] × N \| air, to scale, and R(χ) | `cmpc_sim.DBR`, `cmpc_sim.membrane_mode` |
| `gmpc_stadiums.png` | smooth stadium, 16 facets per cap, facets + tilt, facets + curved caps | `gmpc.planar` |
| `herriott_cells.png` | Herriott R = 0.5 m, N = 30, M = 7 (side view, spots); astigmatic Lissajous cell, N = 50 | `gmpc.herriott` (exact 3D trace) |

Facet tilts and curvature sags are a few µm over a 1.3 mm facet, so they are invisible at true scale; the exaggerated panels are labelled.

## dbr_periods.py

Smallest number of periods N with R ≥ target for a Si membrane (n_eff from `slab_waveguide`, Salzberg–Villa Si index), deep-etched trench DBR (`bragg_grating.TrenchDBR`, exact oblique transfer matrix) and shallow partial-etch gratings (coupled-mode κ and transfer matrix). Columns: normal incidence, the star {24/7} angle, the uniform-sin χ average (fully chaotic cell), the worst angle, and the power-weighted R_eff traced in the curved 24-facet cell (`cell_mirror`). Lossless 1D effective-index model: out-of-plane radiation in the air gaps, roughness and TE↔TM conversion are not included (cmpc-sim lumps them into a 3×10⁻⁴ per-bounce loss).

The interactive version is [`../mpc-designs.html`](../mpc-designs.html).
