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
  The dither visibility comes from `ray_phase.dither_filon` (adaptive Filon integral over the dither; budget
  `n_dither_max`) with its error `V_floor` = V_err: exact for regular and mildly chaotic exits, an upper bound V + V_err
  where the exit pass is broken into unresolvable path fragments (`frag`; `nec_fringe_hi` uses V + V_err).
  `speckle_avg` = 1 lets a detector that collects all exit light average the fringe over M² speckle grains:
  `V_det` = V/√M² enters the fringe NEC. Reaching a shot-noise-limited ~10 ppb NEC needs V_det ≈ 1e-5 at f_par = 1e-6.
  **Rippled end mirrors** (`ripple_a`, `ripple_period`, `ripple_phase`, `ripple_on`): a ripple a sin(2πη/Λ + φ) on the
  mirror profile kicks the reflected angle by 2aq cos(qη); over the spacing L this is a standard map with
  K = 8π² a L / Λ² (`K_ripple`; K ≈ 1 for a ≈ 0.3 µm at Λ = 0.5 mm, L = 10 mm).
- **Chaos map** (`tangent_chaos`, `chaos_map`): the exact tangent map of the walls, unfolded (flight [[1, ℓ], [0, 1]],
  reflection [[1, 0], [−2κ/cos χ, 1]] with the local curvature κ, ripple included), renormalised every hit, gives per ray
  the finite-time Lyapunov exponent and MEGNO ⟨Y⟩ (→ 0 periodic, → 2 regular torus or island, ≈ λn/2 chaotic; > 4 after
  `n_ftle` = 300 hits counts as chaotic). Applied to the launch beam's rays (`f_launch`, `esc_launch`, `Y_launch`,
  `lam_ftle`) and to a `chaos_grid`² grid of departures from M1 (`sea`, `island`), it gives the verdict: the beam escapes
  (−1), launches into the chaotic sea (2), onto a regular island while a sea exists (1), or there is no sea (0).
  Inclusions and windows are left out of the map; flat facets have κ = 0, so a polygon reads regular there.
- **Retuning** (`retune`): a deformed mirror moves the spot pattern off re-entrance; the launch angle is re-aimed
  for the largest Γ T ⟨L⟩ before the comparison.

Front ends: `compare` (both cells and their ratios), `perturbation_sweep` (baseline against one of eps4, eps3, conic,
n_seg, seg_curv, seg_tilt, n_inc, inc_dn, inc_radius, ripple_a, ripple_period, ripple_phase; CMPC as reference), `deformation_sweep` (eps4).

Checks (tests): Φ⁻¹ against `statistics.NormalDist`; the line table against `trace_gas`; the input beam is the
resonator mode (mirror radius √(λm L/(π√(1−g²)))), stable and non-breathing in the undeformed cell; the ideal
Herriott cell exits after N hits with V ≈ 1 while the CMPC washes the fringe out (V < 0.1) and loses the Gaussian
beam; equal footprints; the quartic deformation drives round trips unstable (|m| > 1) and retuning restores the path;
mulberry32 matches the JS stream; segments with the mirror's curvature trace like the smooth mirror; a ray through an
inclusion's centre goes straight with optical path + 2rΔn/n, off centre it deviates by the ball-lens 2(θ1 − θ2), a grazing
ray on a lower-index disk is totally reflected; inclusions do not overlap; curved segments wash out part of the fringe
while keeping the light, inclusions wash it out completely but lose the light from an open cell, and closed side walls
keep it (NEC below the CMPC's) at the price of a scrambled beam (M² ≫ 1); the tangent map keeps the ideal cell's launch
bounded (MEGNO < 0.1) and its FTLE follows the twin-ray exponent in a rippled cell; K = 8π² a L / Λ²; at K ≈ 0.95
(a = 0.3 µm, Λ = 0.5 mm, L = 10 mm) the open cell's beam sits on a regular island for ripple phase 0 and in the chaotic
sea for phase π/2, while the ideal cell has no sea.

Strong chaos (λ ≳ 0.03 per hit) amplifies round-off: single trajectories are not reproducible beyond tens of hits
(Python and JS agree there in the ensemble numbers, not ray by ray). Limits: 2D ray optics in the slab (no diffraction
beyond the Gaussian model, no interference between rays, no scattering by inclusions smaller than a few wavelengths), one
constant mirror reflectance, the fringe model is one parasitic copy following the dither, illustrative line data.
