# path_coherence

Random-phase model of interference (speckle/etalon) noise in multipass cells. Detected rays from a `billiard_cell` ray table are grouped into paths (facet itinerary) per output mode cell (width λ/(n_eff w) in sin χ); from the path powers and lengths it gives the speckle contrast vs laser linewidth (one mode or all modes on a detector), the spectral autocovariance g(Δν), its width, thermal decorrelation, the resulting absorbance noise, and single random-phase realisations of the coherent output spectrum with a gas line.

Version 2 (pair_coherence now returns the fringe visibility |g(τ)| = exp(−πΔν τ) and, separately, the variance factor |g|²; version 1 returned |g|² labelled as visibility). `thermal_shift`, `pair_coherence` and `random_phase_contrast` return Results; `prepare` (→ `Paths`), `contrast`, `autocovariance`, `correlation_width`, `thermal_shift_hz`, `speckle_noise_A`, `simulate_spectrum` are helpers.

A prediction model: reflection phases neglected, not yet validated against full-wave simulations. Pair sums are O(n²) per cell and are evaluated in blocks, so cells with tens of thousands of paths fit in memory but take time.
