# onchip_herriott

Gaussian-beam budget of the integrated (on-chip, in-plane) Herriott cell from `planar_cell.herriott_planar_cell`: the cell's own mode (zR = √(d(2R − d))/2, w0² = λm zR/π, λm = λ/n_eff) rides on the traced ray, and every hit multiplies the power by the mirror reflectance and by the Gaussian share that stays on the mirror and off the window.

- Effective path L_eff = Σ I ℓ, effective mode volume V_eff = h_eff Σ I ∫ w√(π/2) ds and mean cross-section A_eff = V_eff/L_eff, power-weighted mean and RMS angle of incidence, throughput (window coupling in and out), clipping loss.
- Mirrors: constant R, or an etched-trench DBR (`bragg_grating.TrenchDBR`, exact transfer matrix) at the ray angle or averaged over the beam's angular spectrum (power ∝ exp(−2α²/θ0²), θ0 = λm/(π w0)); extra scatter loss per bounce.
- Exit: `window` (re-entrant design, out after N hits) or `closed` (circulates; L_eff → ℓ̄/(1 − R)).
- `sweep_R` (constant R list), `footprint_fraction` (share of the cell area within one beam radius of the pattern).

Version 2 adds:

- **Waveguide loss** (`alpha` = `DB_PER_CM` × dB/cm): each pass decays as e^{−αs}; L_eff and V_eff integrate that weight; `sweep_size` compares geometrically similar cells (R, A and window scaled) at several losses.
- **Beam propagation** (`beam_on_path`): the complex beam parameter q along the exact traced chords with the in-plane mirror power 2/(R cos χ); the matched input is the eigenmode of the first round trip (`path_matrix`, `eigen_q`), the actual input is scaled (`w_ratio`) and refocused (`focus_shift`). Gives the round-trip stability (A + D)/2 and Gouy phase, the coupling into the eigenmode (`coupling`), and the breathing of a mismatched beam; clipping, volume and the DBR angular spectrum use the local beam.
- **Layer stack** (`stack.py`): substrate (SiO₂ or air) | Al₂O₃, Si₃N₄, TFLN, TFLT or Si film | optional thin Er:Al₂O₃ layer | air or SiO₂ cladding; the fundamental TE/TM slab mode by transfer matrices, n_eff, h_eff and the power share in every layer.
- **Erbium amplifier** (`erbium.py`, `amplifier`): signal and co-propagating 980/1480 nm pump on the folded path, effective two-level Er:Al₂O₃ rate equations (as in er-waveguide-amplifier.html: McCumber, upconversion, quenched ions) in the Er layer with intensity P Γ_Er/(w√(π/2) t_Er), RK4 along each pass, mirror reflectance per wavelength (`DBRAt`: the signal DBR seen by the pump).

Result front ends: `path_budget` (now with `loss_db_cm`, `beam_model`, `w_ratio`, `focus_shift`), `slab_stack`, `beam_stability`, `er_amplifier`. Helpers: `mode_profile`, `beam_width`, `chord_width_integral`, `gauss_fraction`, `angle_weights`, `mirror_function`, `make_mirror`, `trace_hits`, `budget`, `sweep_R`, `footprint_fraction`.

Checks (tests): loss reproduces the analytic e^{−αs} integrals and the size saturation; the matched beam is the round-trip eigenmode (coupling 1, periodic width, Gouy ≈ 2θ) and a mismatched one breathes; d > 2R is unstable; the stack matches `slab_waveguide.multilayer_neff` and its overlaps sum to 1; McCumber crossing at λ0, full inversion under strong 980 pumping, upconversion lowers N2; the unpumped amplifier reproduces Γσ_aN absorption and a pumped one gains; mode radius matches the resonator formula; the chord width integral matches the closed form; lossless and constant-R budgets match the geometric series; the closed cell reaches ℓ̄/(1 − R); angles do not depend on R; beam averaging reduces to the point reflectance for a narrow spectrum; TE (p-pol, Brewster dip) loses more than TM; clipping at a window narrower than the beam.

JS port: `onchip-herriott.html` (engine block), checked by `tools/check_js_ports.mjs`.
