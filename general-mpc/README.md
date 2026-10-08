# general-mpc

General multipass-cell (MPC) simulator, grown out of the CMPC simulator (`cmpc-sim/`, `math-engines/engines/billiard_cell`): the same ray picture, ports and path statistics, extended to

- **Herriott cells** and **slightly deformed Herriott cells**: exact 3D ray tracing between real mirror surfaces (spheres, biconic/astigmatic mirrors, conic constants, polynomial surface deformations), mirror tilt, decentre, spacing and radius errors, input/output holes and clear apertures;
- **astigmatic Herriott cells** (Rx ≠ Ry): Lissajous spot patterns, re-entrant designs;
- **stadium cells**, smooth or with **faceted circular caps** and segmented straights, and any closed planar wall built from flat facets, curved facets and circular arcs (circle, regular polygon = the segmented CMPC);
- the **integrated Herriott cell**: two concave in-plane mirrors on a chip, re-entrant design, one window at (or just inside) the mirror edge for input and output (`planar.herriott_planar_cell`, `trace2d.herriott_planar_trace`);
- **perturbations** of every wall element or mirror: tilt, offset, curvature, random (rms, seed) or explicit.

numpy only (matplotlib for the figures). SI units.

The physics lives in the misc-applets math-engines, `engines/planar_cell` and `engines/herriott_cell` (with specs, tests, test vectors and the JS ports in `planar-mpc-ray-tracer.html` and `herriott-ray-tracer.html`); the `gmpc` modules re-export them under the names below, so this package must sit inside a misc-applets checkout (or set `MISC_APPLETS=/path/to/misc-applets`). The engines' `ray_phase` (path per pass, dither visibility) and `cell_mirror` (trench-DBR reflectance over the cell's angles) run on every planar cell and, through `herriott.HerriottLaunch`, on the 3D Herriott cell.

## Layout

```
gmpc/planar.py     Cell2D (chain of flat / curved mirror elements), circle_cell, polygon_cell, stadium_cell,
                   herriott_planar_cell, herriott_planar_spots, perturb, make_cell
gmpc/trace2d.py    trace_rays (ports) -> RayTable, evaluate (reflectance, loss, Γ), mean_field_estimate,
                   trace_path, poincare, twin_divergence (Lyapunov exponent), herriott_planar_trace
gmpc/herriott.py   Mirror3D (biconic + conic + polynomial sag, aperture, holes), two_mirror_cell, perturb_mirror,
                   trace3d -> Trace3D, herriott_cell, build_cell, astigmatic_cell, astigmatic_reentrant, injection,
                   paraxial_spots, mode_radius, reentrance, spot_metrics, herriott_effective_path, twin_divergence_3d,
                   HerriottLaunch (for ray_phase / cell_mirror)
gmpc/stats.py      effective_path (Σ I_j ℓ_j), lyapunov_fit, min_pairwise_distance
                   (all four: re-exports of math-engines engines/planar_cell and engines/herriott_cell)
examples/run_figures.py   overview figures (Herriott ideal / astigmatic / deformed, tolerance sweep; smooth vs
                          faceted vs perturbed stadium; path statistics vs mirror reflectance) -> figs/
tests/             API check (the physics tests are in math-engines/engines/planar_cell and herriott_cell)
```

```
cd general-mpc
python -m pytest -q
python examples/run_figures.py
```

## Planar cells

A wall is a counter-clockwise chain of elements (it may be open, as in the integrated Herriott cell); each is a chord A → B with signed curvature κ: 0 flat, κ < 0 concave from inside (a piece of a circular wall, radius 1/|κ|), κ > 0 convex (dispersing). So one element type covers smooth arcs, flat facets and curved facets. Facets get a 15 % hit-window extension past their corners (as in the segmented CMPC) so perturbed walls do not leak; rays that still escape through a gap are counted (`RayTable.leaked_fraction`).

```python
from gmpc import planar as P, trace2d as T
cell = P.stadium_cell(5e-3, 5e-3, cap_facets=16, straight_segments=4)
cell = P.perturb(cell, tilt_rms=1e-3, curvature_rms=40, seed=2, select="cap")     # caps only
tab = T.trace_rays(cell, port_w=150e-6, n_rays=2000, n_bounce=3000)
res = T.evaluate(tab, lambda s: np.full_like(s, 0.999))                            # T_det, L_mean, ...
sep, fit = T.twin_divergence(cell, 0.1 * cell.perimeter, 0.45)                    # fit["lam"]: Lyapunov / reflection
```

Cross-checks (tests): `polygon_cell` + tilts/offsets reproduces `billiard_cell.SegmentedCell` to 1e-12 m over 60 hits; the smooth `stadium_cell` reproduces `billiard_cell.Stadium` over the first chaotic bounces; the circle conserves sin χ; areas and perimeters are exact; Monte-Carlo ⟨L⟩ in the stadium matches the ergodic mean field.

Physics seen in the figures: a smooth stadium is chaotic (λ ≈ 1 per reflection); faceting its caps makes the wall polygonal and removes the exponential divergence (pseudo-integrable, quantised angles), and random facet tilts do not bring it back; curved facets do.

## Herriott-type cells

Mirror surface in its local frame (vertex at the origin, +z into the cell):
`z = (cx x² + cy y²)/(1 + sqrt(1 − (1+kx)cx²x² − (1+ky)cy²y²)) + Σ a_ij x^i y^j`, cx = 1/Rx, cy = 1/Ry. Intersections by Newton iteration (exact; a sphere matches the closed form to 1e-14 m), so spherical aberration, astigmatism, tilt and decentre are all in the trace. Rays inside a hole leave the cell; outside the clear aperture they are lost.

```python
from gmpc import herriott as H
c = H.herriott_cell(R=0.5, N=30, M=7, A=0.012)              # d = R(1 − cos 2πM/N), hole at the injection point
tr = H.trace3d(c["mirrors"], c["p0"], c["d0"], n_max=200)
H.reentrance(tr)          # exit 'hole' after 30 hits, path, re-entry offset and angle
H.spot_metrics(tr, c["mirrors"], w=c["w_mode"])            # spot spacing and hole clearance in beam radii
m2 = H.perturb_mirror(c["mirrors"][1], tilt_x=2e-4, dRx=1e-3, dRy=-1e-3, poly={(3, 0): 1e-3})
Rx, Ry, d = H.astigmatic_reentrant(0.5, 50, 11, 13)          # Lissajous cell re-entrant after 50 hits
```

Design relations (paraxial, Herriott, Kogelnik & Kompfner, Appl. Opt. 3, 523 (1964)): cos θ = 1 − d/R; spot n at x_n = x₀ cos nθ + √(d/(4f−d)) (x₀ + 2f x₀′) sin nθ, f = R/2; re-entrant when Nθ = 2πM. `injection` gives the ellipse x_n = A cos nθx, y_n = B sin nθy. `mode_radius`: w² = (λd/π)/√(1 − g²), g = 1 − d/R.

Checks (math-engines tests): spot positions follow paraxial theory for small patterns; the re-entry error of exact spheres scales as A³ (third-order spherical aberration); astigmatic mirrors give the predicted Lissajous pattern and re-enter after N hits; traces are reversible; stable cells do not diverge exponentially.

Physics: tilting or decentring a spherical mirror only moves its centre of curvature, i.e. tilts the cell axis, so the pattern shifts (by R α (R − d)/(2R − d) on the far mirror, tested) but stays re-entrant; anything that changes θ (radius or spacing errors, astigmatism) breaks re-entrance. In the figure, a 0.2–0.3 % radius error moves the re-entry point by about the hole radius; a small astigmatism partly compensates the spherical aberration.

## Integrated Herriott cell

```python
hc = P.herriott_planar_cell(R=20e-3, N=24, M=5, A=1e-3, port_w=100e-6, phase=0.0)   # d = R(1 − cos 2πM/N)
tr = T.herriott_planar_trace(hc)          # exit 'port' after N hits, exit_offset, exit_angle (0 = repeats its first pass)
P.herriott_planar_spots(hc.meta)          # design spots per mirror, closest spots, window clearance
```

In one transverse dimension the pattern y_n = A cos(nθ + φ) has N/2 spots per mirror. φ = 0 starts at the turning point (window at the mirror edge, launch along the mirror normal) and the pattern runs out and back, so spots coincide in pairs; φ = π/N separates them but puts the window just inside the edge, where the clearance to the nearest spot is A (cos π/N − cos 3π/N). The exit-angle error of exact circular mirrors grows as A³; at the edge window a period error tilts the exit ray before it moves the exit point (the window sits at a turning point of the pattern).

## Not (yet) included

Gaussian-beam propagation along the traced path (only the cell mode radius), diffraction at holes, polarisation, mirror coatings (pass a reflectance or R(cos χ) to `herriott_effective_path` / R(sin χ) to `evaluate`), cylindrical-mirror (Robert) cells and more than two mirrors in the design helpers (`trace3d` itself takes any list of mirrors).

Interactive versions: `planar-mpc-ray-tracer.html` and `herriott-ray-tracer.html` in the repository root.
