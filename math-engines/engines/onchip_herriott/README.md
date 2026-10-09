# onchip_herriott

Gaussian-beam budget of the integrated (on-chip, in-plane) Herriott cell from `planar_cell.herriott_planar_cell`: the cell's own mode (zR = √(d(2R − d))/2, w0² = λm zR/π, λm = λ/n_eff) rides on the traced ray, and every hit multiplies the power by the mirror reflectance and by the Gaussian share that stays on the mirror and off the window.

- Effective path L_eff = Σ I ℓ, effective mode volume V_eff = h_eff Σ I ∫ w√(π/2) ds and mean cross-section A_eff = V_eff/L_eff, power-weighted mean and RMS angle of incidence, throughput (window coupling in and out), clipping loss.
- Mirrors: constant R, or an etched-trench DBR (`bragg_grating.TrenchDBR`, exact transfer matrix) at the ray angle or averaged over the beam's angular spectrum (power ∝ exp(−2α²/θ0²), θ0 = λm/(π w0)); extra scatter loss per bounce.
- Exit: `window` (re-entrant design, out after N hits) or `closed` (circulates; L_eff → ℓ̄/(1 − R)).
- `sweep_R` (constant R list), `footprint_fraction` (share of the cell area within one beam radius of the pattern).

Version 1. Result front end: `path_budget`. Helpers: `mode_profile`, `beam_width`, `chord_width_integral`, `gauss_fraction`, `angle_weights`, `mirror_function`, `make_mirror`, `trace_hits`, `budget`, `sweep_R`, `footprint_fraction`.

Checks (tests): mode radius matches the resonator formula; the chord width integral matches the closed form; lossless and constant-R budgets match the geometric series; the closed cell reaches ℓ̄/(1 − R); angles do not depend on R; beam averaging reduces to the point reflectance for a narrow spectrum; TE (p-pol, Brewster dip) loses more than TM; clipping at a window narrower than the beam.

JS port: `onchip-herriott.html` (engine block), checked by `tools/check_js_ports.mjs`.
