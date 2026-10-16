# gas_cell

Gas-sensing multipass cells on a chip: an **elongated, deformed integrated Herriott cell** (the baseline, e.g.
15 mm × 1.5 mm) against the **circular multipass cell (CMPC)** of the same footprint area, fed by the same Gaussian
beam. It answers how much path (and detection limit) the regular two-mirror cell gains, and how much chaos (fringe
washout, beam scrambling) it gives up.

- **Cells** (`planar_cell`): `deformed_herriott_cell` builds the two mirrors as smooth profile elements (conic sag +
  cubic/quartic deformation `eps3`, `eps4` as fractions of the edge sag, exact Newton intersection), optional
  reflecting side walls; `circle_reference` the polygon (or smooth circle) of equal area with input and output ports.
- **Input beam**: the eigenmode of the undeformed baseline at its window (`input_q`), sampled as an n_side × n_side
  quantile grid of rays from its Wigner function (`beam_rays`).
- **Path**: port-to-port ray statistics with mirror reflectance (`planar_cell.trace_rays/evaluate`): detected power,
  power-weighted mean path, absorbance per ppm of the strongest listed line (`SPECIES`, from `trace_gas`),
  shot-noise-limited NEC.
- **Chaos**: twin-ray Lyapunov exponent at the launch angle; dither visibility V at the exit pass
  (`ray_phase.dither_analysis`) and the fringe-limited NEC = 2 √f_par V / A_ppm.
- **Gaussian-beam two-mirror model** (`gaussian_path`): q along the traced centre ray with the local curvature and
  angle at every hit; beam radius per hit, share of the footprint left on the reflecting element (T_G), largest
  round-trip |m| (> 1: locally unstable), linearised growth λ_G.
- **Chaos makers**: curved segmented mirrors (`n_seg` arcs per mirror on the mirror profile, `seg_curv` × its curvature
  (0 flat facets, 1 no kinks), `seg_tilt` rms tilt jitter, mulberry32 seed) and **index inclusions** (`place_inclusions`:
  `n_inc` disks of radius `inc_radius`, Δn_eff = ±`inc_dn`, non-overlapping, seeded, optionally also in the CMPC).
  `MediumCell` + `march` trace rays through walls and inclusions (Snell, total internal reflection, Fresnel reflection
  neglected) with geometric and optical path; ray_phase uses its `path_lengths` (optical path), the absorption the
  geometric one. `ensemble` adds the **M²** of the detected light from its exit offsets and angles (1 for the input).
  The dither visibility is reported with its sampling floor `V_floor` = √π/(2√n); a V at the floor is unresolved.
- **Retuning** (`retune`): a deformed mirror moves the spot pattern off re-entrance; the launch angle is re-aimed
  for the largest Γ T ⟨L⟩ before the comparison.

Front ends: `compare` (both cells and their ratios), `perturbation_sweep` (baseline against one of eps4, eps3, conic,
n_seg, seg_curv, seg_tilt, n_inc, inc_dn, inc_radius; CMPC as reference), `deformation_sweep` (eps4).

Checks (tests): Φ⁻¹ against `statistics.NormalDist`; the line table against `trace_gas`; the input beam is the
resonator mode (mirror radius √(λm L/(π√(1−g²)))), stable and non-breathing in the undeformed cell; the ideal
Herriott cell exits after N hits with V ≈ 1 while the CMPC washes the fringe out (V < 0.1) and loses the Gaussian
beam; equal footprints; the quartic deformation drives round trips unstable (|m| > 1) and retuning restores the path;
mulberry32 matches the JS stream; segments with the mirror's curvature trace like the smooth mirror; a ray through an
inclusion's centre goes straight with optical path + 2rΔn/n, off centre it deviates by the ball-lens 2(θ1 − θ2), a grazing
ray on a lower-index disk is totally reflected; inclusions do not overlap; curved segments wash out part of the fringe
while keeping the light, inclusions wash it out completely but lose the light from an open cell, and closed side walls
keep it (NEC below the CMPC's) at the price of a scrambled beam (M² ≫ 1).

Strong chaos (λ ≳ 0.03 per hit) amplifies round-off: single trajectories are not reproducible beyond tens of hits
(Python and JS agree there in the ensemble numbers, not ray by ray). Limits: 2D ray optics in the slab (no diffraction
beyond the Gaussian model, no interference between rays, no scattering by inclusions smaller than a few wavelengths), one
constant mirror reflectance, the fringe model is one parasitic copy following the dither, illustrative line data.
