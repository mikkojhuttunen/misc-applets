# herriott_cell

Exact 3D ray tracing for Herriott-type cells between real mirror surfaces: z = (cx x² + cy y²)/(1 + √(1 − (1+kx)cx²x² − (1+ky)cy²y²)) + Σ a_ij xⁱ yʲ (spherical, biconic/astigmatic, conic, polynomial deformations), Newton intersections, clear apertures and holes. Moved here from `general-mpc` (`gmpc.herriott` now imports it).

- Design: `reentrant_spacing`, `theta_of`, `injection`, `paraxial_spots`, `mode_radius`, `herriott_cell` (spherical, hole by the design rule), `astigmatic_reentrant` / `astigmatic_cell` (Lissajous), `build_cell` (design + tilt, decentre, radius, astigmatism, spacing and conic errors; the injection keeps its design direction and starts on the deformed mirror 1).
- Tracing and figures of merit: `trace3d` → `Trace3D` (`reflections`, `last_is_exit`: a hole exit or a hit outside an aperture is not a reflection), `reentrance` (exit offset, re-entry angle), `spot_metrics` (spot spacing, hole clearance, in mode radii), `herriott_effective_path` (Σ Rʲ ℓⱼ, R constant or a function of cos χ), `twin_divergence_3d`.
- `HerriottLaunch`: the cell seen from its injection, for `ray_phase` (path per pass under an injection-angle tilt in the x–z or y–z plane, path id = mirror sequence) and `cell_mirror` (|sin χ| per hit).

Version 1. Result front ends: `design`, `astigmatic_design`, `reentrance_check`, `dither_visibility`.

Checks (tests): Newton intersections match the closed-form sphere to 1e-14 m; the design re-enters after N hits with an exit offset ∝ A³; paraxial spot positions and Lissajous patterns; tilting a spherical mirror keeps re-entrance and shifts the pattern by R α (R − d)/(2R − d); radius errors break it; traces are reversible; a stable cell does not diverge; the phase slope under injection dither stays bounded.

JS port: `herriott-ray-tracer.html` (engine block), checked by `tools/check_js_ports.mjs`.
