# planar_cell

2D ray tracing in planar (on-chip) multipass cells whose wall is a chain of mirror elements, each a chord with a signed curvature (flat facet, concave arc, convex arc):

- `circle_cell`, `polygon_cell` (the segmented CMPC, same convention as `billiard_cell.SegmentedCell`), `stadium_cell` (smooth caps, or caps of flat facets, with segmented straights);
- `herriott_planar_cell`: the **integrated Herriott cell**, two concave in-plane mirrors at the re-entrant spacing d = R (1 − cos 2πM/N), spot pattern y_n = A cos(nθ + φ), one window on M1 that is both input and output. φ = 0 puts the window at the mirror edge (launch along the normal; spots coincide in pairs), φ = π/N separates every spot. The wall is open: rays that pass a mirror end leave the cell (exit −2);
- `perturb`: tilt, normal offset and curvature change per element, random (rms, seed, label selection such as `"cap"`) and/or explicit arrays.

Tracing as in `billiard_cell`: `trace_rays` (ports as windows in the wall coordinate s, output checked first) → `RayTable`, `evaluate` (re-weight for any R(|sin χ|), background loss, Γ), `mean_field_estimate`, `trace_path`, `poincare`, `twin_divergence` / `lyapunov_fit`, `herriott_planar_trace` (exit through the window, exit offset and exit-angle error), `herriott_planar_spots` (design spot spacing and window clearance), `mode_radius` (in-plane, λ/n_eff).

`Cell2D` has `hit_k` (with the element index) and `hit` (the `billiard_cell` interface), so `ray_phase` (path per pass, dither visibility; path id = element sequence) and `cell_mirror` (trench-DBR reflectance over the cell's angles) run on every planar cell unchanged.

Version 1. Result front ends: `ray_statistics`, `chaos`, `planar_herriott`, `dither_visibility`, `planar_herriott_dither`, `mirror_reflectance`, `planar_herriott_mirror`.

Checks (tests): the polygon reproduces `SegmentedCell` to 1e-12 m and gives identical `ray_phase` and `cell_mirror` results; the smooth stadium reproduces `billiard_cell.Stadium`; the circle conserves sin χ; exact areas and perimeters; Monte Carlo ⟨L⟩ in the stadium matches the mean field; faceting removes chaos and curved facets restore it; the integrated Herriott cell re-enters through its window after N hits with an exit-angle error ∝ A³, follows paraxial theory, and keeps a bounded phase slope (refocusing) while the polygon's grows ∝ p; a DBR at near-normal incidence does better in a Herriott cell than in a stadium.

JS port: `planar-mpc-ray-tracer.html` (engine block), checked by `tools/check_js_ports.mjs`.
