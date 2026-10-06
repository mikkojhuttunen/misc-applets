# billiard_cell

2D ray tracing of planar multipass cells: stadium (circle at a = 0) and segmented N-gon cells with designed perturbations (random facet tilt, radial offset, facet curvature and its spread, or explicit per-facet arrays to change single facets). Rays launch from an input port window, reflect specularly, and stop at the input (lost) or output (detected) port.

The tracer is geometry only (`trace_rays` → `RayTable`: every chord length, angle of incidence and a facet-itinerary hash). `evaluate` re-weights one table for any mirror R(sin χ), background loss α_bg and evanescent factor Γ, so mirror/loss/wavelength sweeps need no re-tracing. `mean_field_estimate` is the ergodic cross-check (mean chord πA/P); `poincare` and `occupancy_map` show phase-space filling and spatial coverage.

Version 1. `mean_field` and `ray_statistics` return Results (constant R); `Stadium`, `SegmentedCell`, `make_cell`, `trace_rays`, `trace_path`, `evaluate`, `poincare`, `occupancy_map`, `mean_field_estimate`, `dB_per_cm_to_alpha` are helpers.

The JS `beam-propagation/ray-tracing-engine.js` covers the unperturbed polygon and its closed star-polygon orbits; this engine adds ports, perturbations, loss weighting and path statistics.

Rays still inside after `n_bounce` (or `max_path`) are dropped from the statistics: check `RayTable.trapped_fraction` against how much power such long paths would still carry.
