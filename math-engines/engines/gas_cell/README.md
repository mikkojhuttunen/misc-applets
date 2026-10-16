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
- **Retuning** (`retune`): a deformed mirror moves the spot pattern off re-entrance; the launch angle is re-aimed
  for the largest Γ T ⟨L⟩ before the comparison.

Front ends: `compare` (both cells and their ratios), `deformation_sweep` (baseline against eps4, CMPC as reference).

Checks (tests): Φ⁻¹ against `statistics.NormalDist`; the line table against `trace_gas`; the input beam is the
resonator mode (mirror radius √(λm L/(π√(1−g²)))), stable and non-breathing in the undeformed cell; the ideal
Herriott cell exits after N hits with V ≈ 1 while the CMPC washes the fringe out (V < 0.1) and loses the Gaussian
beam; equal footprints; the quartic deformation drives round trips unstable (|m| > 1) and retuning restores the path.

Limits: 2D ray optics in the slab (no diffraction beyond the Gaussian model, no interference between rays), one
constant mirror reflectance, the fringe model is one parasitic copy following the dither, illustrative line data.
