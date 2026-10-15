# lateral_optics

Lateral (in-plane) optics of slab-waveguide gratings. Ridge, rib or strip guides by the two-step effective-index method: slab index v_t, side index n_O (side slab or substrate), ridge index N_t from the symmetric lateral slab (polarisation flipped: a vertical TE mode is lateral TM), and the lateral confinement Γ; an etched grating slice of local thickness d gets N(d) = N_t + Γ (v(d) − v_t). The lateral ridge mode against a Gaussian beam (power overlap) and its SHG width 1/∫φ⁴. Lateral Gaussian beams: power overlap of two beams, the round trip of a beam to a curved (or flat) DBR with a finite aperture, and the best launch waist (the resonator eigenmode for a stable half-cavity). SI units.

Version 1. `ridge_effective_index` and `curved_dbr_bounce` return Results; `side_index`, `ridge_reference`, `ridge_neff`, `ridge_beam_overlap`, `gauss_overlap`, `bounce` and `best_bounce` are helpers.

Python reference for `ridgeRef`, `neffFull` (ridge), `ridgeLateral`, `gaussOverlap`, `bounce` and `bestBounce` in `dbr-structures/dbr-engine.js`, checked by `tools/check_js_ports.mjs`. Writing it found an inconsistency in the JS Γ for vertical TE modes (the ridge-field phase lacked the lateral-TM factor (v_t/n_O)²; Γ was off by up to 0.2 %), fixed in the same change.

Checks: Gaussian overlaps against closed forms (waist and curvature mismatch); the stable-resonator eigenmode (z_R² = d(R_c − d)) bounces without loss and is the waist `best_bounce` finds; the bounce against the ABCD law and beam radius of `gaussian_beam`; Γ against direct integration of the lateral mode; wide-ridge and no-step limits.
