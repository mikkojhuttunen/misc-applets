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

## membrane_3d.py (3D, oblique view)

```
python membrane_3d.py                    membrane_chip_3d.png/.pdf and membrane_dbr_3d.png/.pdf into figs/
python membrane_3d.py chip --facets 12 --periods 3 --azim -40 --elev 26
python membrane_3d.py dbr --thickness 300 --pol TM --chi 15
python membrane_3d.py chip --outputs 3 --out-periods 1
```

| Figure | Content |
|---|---|
| `membrane_chip_3d` | schematic, not to scale: Si handle, buried oxide and the Si device layer released over a backside window; the segmented cell with a trench DBR on every facet, corner bridges, the input port (no DBR, beam entering at the star-orbit angle), output ports that keep only the first `--out-periods` DBR periods over the port width (partial transmission; default half), tapered strip-loaded waveguides from every port to the chip edge, the star orbit, and a cutaway quadrant through the stack and the DBR |
| `membrane_dbr_3d` | to scale: membrane thickness, air gaps and teeth from `cmpc_sim.membrane_mode` and `cmpc_sim.DBR`, the incident and reflected slab-mode ray, and the |H|² (TM) or |E|² (TE) profile across the membrane on the cut face |

Rendering is done by `render3d.py`, a small z-buffered renderer (orthographic camera, Lambert shading, polygons with holes, prisms extruded from shapely polygons, depth-tested lines, 2× supersampling). It needs only numpy, Pillow, shapely and matplotlib, and draws occlusion correctly where mplot3d's painter's algorithm does not. The corner bridges are a drawing assumption (something has to hold the inner membrane once the trenches are etched through); the simulations do not model them.

## herriott_planar.py (integrated Herriott cell)

In-plane Herriott cell in the membrane: two concave curved trench-DBR mirrors (radius R) at d = R (1 − cos 2πM/N), injected through a gap in mirror 1 for the spot pattern x_n = A cos(nθ + φ). The trace is exact (circular mirrors), and every hit applies the trench-DBR reflectance at its angle (`cmpc_sim.DBR`), giving the throughput and the absorption-weighted path.

```
python herriott_planar.py                                   R = 5 mm, N = 22, M = 5, A = 0.5 mm, φ = 90°
python herriott_planar.py --phase 0                         classic edge port
python membrane_3d.py herriott --R 5 --N 22 --M 5 --A 0.5   3D cutaway: curved DBR mirrors, centre port, V of tapered waveguides
```

In one transverse dimension the launch phase matters: with φ = 0 (port at the pattern edge, the usual 3D choice) the pattern is symmetric in n and the exit ray retraces the input (1.2° apart for the defaults), so the output couples back into the input waveguide. φ = 90° puts the port at the mirror centre and separates input and output by ≈ 2A sin θ / d (13° for the defaults). Spherical aberration of circular mirrors grows with (A/R)²: at A/R = 0.2 the beam misses the centre port after N passes; aspheric DBR arcs or a smaller A/R fix it.

## Gas molecules and the evanescent field

`membrane_3d.py` draws schematic gas molecules (ball and stick CH₄, CO₂, H₂O, NH₃; `molecules.py`) above and below the cells, a faint red haze above the cell for the evanescent field (stacked translucent sheets, opacity ∝ intensity, out to three intensity decay lengths), and a red halo on every molecule inside that layer: those absorb, the ones further out do not. `--no-molecules`, `--molecules 26`, `--mol-seed 3`. The CMPC chip also gets a central support pillar (handle + oxide, `--pillar 0.07`, 0 for none) in the dark core of the star orbit, standing on a bonded carrier wafer (the backside window goes through the handle). Output ports and waveguides are dark blue, the input waveguide gold.

`evanescent_sensing.py` explains the sensing on its own, in vector graphics: a to-scale cross-section through the membrane with the mode intensity, the intensity decay length δ = 1/(2γ) (71 nm for Si 250 nm TM₀ at 1.55 µm) and molecules inside and outside the field; the mode profile with Γ (α_mode = Γ α_gas, 0.64 here); Γ against membrane thickness for TE₀ and TM₀ (thinner membranes put more of the mode in the gas; thin TM₀ membranes reach Γ > 1 because E_z jumps by n_Si² at the surface).

```
python evanescent_sensing.py
python evanescent_sensing.py --thickness 220 --pol TE
```

## combined_figure.py (overview figures, panels a–e)

Two layouts at journal width (183 mm by default), each written as PDF, SVG and PNG:

- `row`: one row with (a) the CMPC chip, (b) the integrated Herriott cell, (c) the evanescent-field cross-section and (d) Γ against membrane thickness → `figs/combined_overview.*`
- `setup`: (a) and (b) on top; (c), (d) and (e), the system top view of `setup_schematic.py`, below → `figs/combined_setup.*`

(a) and (b) are the 3D renders without labels (`membrane_3d.py --bare`), embedded at full resolution; (c)–(e) are vector. Panel (e) is the full schematic drawn smaller, with text and lines scaled with it; the script prints its smallest text size (about 2 pt at 183 mm), so use a larger `--width` if it has to be read in print.

```
python combined_figure.py                     both layouts, star {24/11}
python combined_figure.py setup --width 250 --setup-kind herriott
python combined_figure.py --no-render         reuse the last 3D renders (fast)
```

## setup_schematic.py (system top view)

Whole setup in one vector figure, for the CMPC and for the integrated Herriott cell: bulk laser with its driver (frequency dither by current modulation) and an isolator, fibre to an edge coupler, an on-chip tap to a reference detector (power normalisation), the on-chip dither section (phase-dither heater on the input, a 1×2 splitter into two heated arms that recombine side by side at the input port to steer the launch angle), the cell on its released window, dark-blue output-port waveguides to edge couplers, fibres to bulk detectors, and the function generator and lock-in / DAQ with their electrical connections. Positions are schematic, not a layout. The bulk optics (laser, isolator, lenses, photodiodes) are drawn from gwoptics ComponentLibrary SVGs in `components/` when present (`components.py`, export instructions and license in `components/README.md`; the attribution line is added to the figure automatically), otherwise from vector stand-ins in the same style.

```
python setup_schematic.py                     figs/setup_cmpc.pdf, figs/setup_herriott.pdf (+ png)
python setup_schematic.py cmpc --outputs 3
python setup_schematic.py --no-modules        without the red light-source and blue detection blocks
```

The interactive version is [`../mpc-designs.html`](../mpc-designs.html).
