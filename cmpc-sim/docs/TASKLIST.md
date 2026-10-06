# CMPC path-length / evanescent-volume simulator: task list (snapshot v0.4, 5 Oct 2026; superseded in parts: see CHANGELOG and MODELS §10)

Code: `cmpc_sim.py` (geometry, modes, mirrors, tracer), `gas_spectra.py`, `coherence_model.py`, `run_figures.py`,
`fetch_hitran.py`, `run_midir.py` (5 µm, Ge/Si membranes). Built on `misc-applets/math-engines` (`materials`, `slab_waveguide`, `bragg_grating`, `gaussian_beam`).
Self-test: `python cmpc_sim.py`. Figures: `python run_figures.py`.

## Correction to v0.1
v0.1 paired the TE-mode Γ with an s-polarised mirror. For a vertical trench, slab TE is p-pol (Brewster hole at sinχ ≈ 0.27-0.36, ~7 % mean loss
per bounce for Si 220 nm) and slab TM is s-pol. All v0.1 path numbers are superseded.

## Proposal strategy
Proposal quotes the simple-model estimates below (clearly labelled as such). Full 3D numerics and full-wave coherence simulations are work
packages, scheduled early (WP1/WP2), because they replace the two largest assumptions (slot-DBR loss, membrane loss).

## A. Done (v0.2)
| # | Item | Status / validation |
|---|---|---|
| A1 | Membrane slab modes TE0/TM0: n_eff, n_g, Γ, penetration depth, surface-field proxy | Γ vs quadrature 0.2 % |
| A2 | Oblique TMM, s and p pol, TIR and frustrated TIR | equals engine at θ=0; Brewster test |
| A3 | DBR design (first order, odd orders, double-resonance order search) | Fig 3 |
| A4 | Segmented circular cell: regular N-gon, facet tilt, offset, curvature; signed sinχ, itineraries, Poincaré sections | geometry tests; Fig 5 |
| A5 | Re-weighting of one ray table for any mirror, loss, Γ: T_det, ⟨L⟩, Γ⟨L⟩, S1, p(L) | MC vs mean-field 7 % |
| A6 | TM polarisation end to end (Γ, s-pol mirror) | Figs 3, 4 |
| A7 | Gas spectra: Voigt, mixtures, p(L)-weighted transmission, shot-noise floor, HAPI loader | illustrative lines only |
| A8 | Coherence model: paths/mode, M_spat, g(Δν), contrast vs linewidth, thermal decorrelation, speckle-limited detection limit, simulated spectra | prediction model, unvalidated |

## B. To do
**Early work packages (full numerics)**
1. 3D FDTD/EME of the slot DBR: radiation loss per transition vs gap/tooth widths, TE<->TM conversion at the trench, angle dependence. Replaces `bounce_loss`. (Per your decision B2.)
2. Measured background loss of 1 cm free-standing membranes (TE and TM) vs thickness; calibrate the roughness proxy.
3. Full-wave (2D FDTD/BPM) test of the coherence model on a 1-2 mm cell: paths/mode, g(Δν), thermal decorrelation.
4. Mechanics: sag, buckling, acoustic modes vs thickness and stress (also feeds the bolometer / phase-channel idea).

**Simulator extensions**
5. HITRAN: run `fetch_hitran.py` locally, replace illustrative lines, add temperature scaling with proper partition functions, self-broadening of H2O.
6. Link speckle noise to a real line fit (wavelength-modulation 2f, baseline polynomial) and add dither / many-mode averaging models.
7. Wider-port runs for M_spat scaling (M ∝ w), more rays per cell.
8. Per-facet DBR design with chirped or multi-order mirrors for low-index membranes (Al2O3, SiNx), where first-order DBRs are weak.
9. Ports: taper / grating-coupler efficiency and angular spectrum.
10. Liquid phase: n_clad and n_gap are parameters; needs liquid spectra and a gap model.
11. 3-3.5 µm staging: materials and mode solver at those wavelengths (Sellmeier validity limits; Al2O3 and sapphire transparency), gas lines (CH4, HCl, VOCs C-H stretch).
12. Packaging into `misc-applets/cmpc-sim/` with spec.yaml entries.

## C. Figures
Made (figs/): 1 Γ, n_eff, penetration (TE and TM); 2 evanescent volume and field profiles; 3 mirrors TE (p) vs TM (s), Brewster hole; 4 thickness trade-off Si, SiNx, Al2O3; 5 segmented cells and Poincaré sections; 6 path vs port width and loss; 7 gas spectra through the cell; 8 simulated coherent spectra, static vs drifted baseline; 9 coherence metrics; 10 speckle-limited detection limit.

Suggested additions: concept schematic; footprint vs effective-path chart against spirals, rings, Herriott cells (needs literature numbers); milestone map (Fig 6 with Y1-Y4 targets); mirror map R(λ, sinχ); gas-exchange time vs footprint; dither-averaged detection-limit figure once item 6 exists.

## D. Key results (simple model, see ASSUMPTIONS in cmpc_sim.py)
- Si TM 250 nm, curved-facet 1 cm cell: Γ = 0.64, ⟨L⟩ ≈ 40 cm, Γ⟨L⟩ ≈ 26 cm at 0.1 dB/cm; ≈ 10 cm with roughness-scaled loss; 50 cm needs ≲ 0.044 dB/cm.
- Si TE 220 nm: Γ⟨L⟩ ≈ 0.9 cm (Brewster hole).
- Al2O3 / SiNx with first-order DBR: mirrors too weak below n_eff ≈ 1.7; Γ⟨L⟩ ≲ 9 cm. Needs designed mirrors (item 8) or thicker films.
- Speckle pattern decorrelates at ~1-2 mK; static baseline subtraction cannot reach ppb targets.

## E. Mid-IR (v0.3): Ge/Si free-standing membranes near 5.26 µm (NO, FeNO range)
Run: `python run_midir.py` (figures in `figs_midir/`). Ge index from a Barnes-Piltch Sellmeier fit entered from memory: verify. Gas lines are
order-of-magnitude placeholders until `python fetch_hitran.py mir`. Background loss scaled from the 1.55 µm reference assuming the same roughness
(k0², surface field, index contrast): ASSUMED, which makes the mid-IR cells mirror-limited, not loss-limited.

Ge 300 nm at 5.26 µm:
- TE0: n_eff 2.36, n_g 3.6, Γ = 0.21, z_p ≈ 200 nm, V_ev ≈ 0.031 mm³ (1 cm disc; 2.8× the 1.55 µm Si TM 250 nm case). DBR: gap 1.32 µm, tooth 0.56 µm.
- TM0: n_eff 1.02 (cut-off, ~2 µm field extent): not a usable confined mode. TM needs ≳ 700-800 nm (Ge TM 800 nm: n_eff 2.5, Γ 0.40, Γ⟨L⟩ ≈ 36 cm; Si TM 800 nm: Γ⟨L⟩ ≈ 45 cm, curved-facet cell).
- TE has the p-pol Brewster hole (sinχ ≈ 0.34-0.42): chaotic cell ⟨1-R⟩ ≈ 0.10, ⟨L⟩ ≈ 18 cm, Γ⟨L⟩ ≈ 4 cm.
- Regular 24-gon launched at 45° (incidence angles multiples of 15°, all outside the hole): T_det 0.37, ⟨L⟩ ≥ 70 cm (lower bound, path cap 8 m), Γ⟨L⟩ ≈ 15 cm, but M = 6 vs 16 output modes.
- Per-ppb peak absorbance (placeholder lines): NO at 5.26 µm is ~9× NH3 at 1.53 µm for the same chaotic cell, from stronger lines (×24) partly offset by lower Γ⟨L⟩.
- Speckle-limited NO detection limit is still 10³-10⁵ ppb at 0.1-10 mK drift (FeNO is 5-50 ppb). A phase dither equivalent to 10 GHz (~2 K at 3×10⁻⁴ /K) gains only ×4-13. Shot-noise floor 0.1-1 ppb (1 mW, 1 Hz, idealised; real mid-IR detectors are not shot-limited).

New / changed tasks:
13. Hole-free TE mirror: p-pol Brewster hole is intrinsic to a two-material stack; test a third-index layer or non-quarter-wave design (or accept TM at ≥ 700 nm).
14. Real HITRAN mid-IR lines (NO, CO, N2O, H2O, CO2); choose NO lines clear of H2O/CO2; self-broadening of H2O.
15. Ge membrane mechanics and optics: stress, sag, free-carrier absorption vs doping, thickness uniformity (VTT Ge nanomembranes), measured dn/dT of Ge at 5 µm.
16. Speckle suppression beyond dither: broadband/multimode QCL, many-mode detection (wider ports, M ∝ w), wavelength modulation with fit, active thermal stabilisation. Required total suppression ~10²-10⁴ for FeNO.
17. Contrast vs launch conditions: trade regular (4× signal, hole-free) vs chaotic (more modes) with a perturbed regular cell that stays hole-free.

## F. v0.4: hole-free TE mirror, perturbed regular cell, speckle suppression, HITRAN status
Run: `python chaos_scan.py; python budget_scan.py; python run_chaos_figs.py` (figures in `figs_chaos/`; ~3 min on one core). Mirror designs: `mirror_design.py` -> `mirror_designs.json`.

**Hole-free TE mirror (done in the 1D model).** Two-material stacks of any thickness are transparent at the Brewster angle (R_p < 1e-30 for 300 random stacks), so quarter-wave/chirped variants cannot help. A third index can: [air gap 250 nm | thinned-Ge 720 nm (Ge 130 nm, n_eff 1.49) | tooth 725 nm (Ge 300 nm)] x 16, depth 27 µm: ⟨1-R⟩ = 0.008 (two-material 0.102), worst angle 0.25. With ±20 nm layer errors the median rises to ~0.014 and the 90th percentile to ~0.05. Needs a shallow-etch step in the process and 3D validation (task 1 numerics): the thinned region is a thickness step, which the effective-index model treats only roughly.

**Perturbed regular cell.** With the two-material mirror, tilt ≲ 0.05° keeps the rays out of the hole (loss/bounce 0.0013) but gains little (M 5 -> 7). At 0.1° the hole is hit (0.009), at 1° the loss is 0.10. With the tri-index mirror any perturbation is allowed (loss/bounce 0.007-0.010) and the best cells are curved facets (rho = 5 cm) or 1° tilt: detection-limit figure of merit (contrast / Γ⟨L⟩) 0.45-0.48 of the regular + two-material cell (about 2x better). Note: flat tilted facets are not formally chaotic (Lyapunov ~ 0.005/bounce, same as the regular polygon); curved facets are (0.32, 0.46, 0.63 per bounce for rho = 20, 10, 5 cm). Both mix paths.

**Speckle vs chaoticity, several beams.** contrast² ≈ (1 - 1/N_p)/(K·M): N_p = paths per output mode, M = independent output modes, K = mutually incoherent beams (verified against the path model to ~10 %). Regular 0.43 -> curved rho=5 cm 0.14 (M 5 -> 48). K = 4 incoherent beams: regular 0.19, curved 0.083. K = 4 beams from ONE laser with no delay do not help (0.41, 0.19): paths from different beams interfere in the same output mode. Mutual incoherence needs delays longer than the coherence length (c/(π Δν): ~10 cm for a 1 GHz laser; an on-chip spiral) or separate lasers.

**Suppression budget** (curved cell, tri-index mirror, Ge 300 nm TE, NO line, placeholder line strengths): 1.7e4 ppb at start (K=1, 1 MHz laser, 1 mK drift) -> wider 400 µm port 1.4e4 (no gain: ⟨L⟩ halves while M doubles) -> K=4 1.2e4, K=8 9.8e3 (every extra port leaks light: ⟨L⟩ 1.0 -> 0.46 at K=8) -> laser 1 GHz 3.3e3 -> dither (≙10 GHz) 2.2e3 -> drift 10 µK 141 -> drift 1 µK 45 ppb. Thermal stability at the µK level is the dominant lever; the FeNO range (5-50 ppb) needs all levers together. Shot-noise floor 0.4 ppb (idealised).

**HITRAN.** hitran.org is not reachable from the sandbox and no PyPI package bundles line data. Done: HAPI (classic, TIPS-2021) installed locally to supply real partition sums for temperature scaling (e.g. NO Q(307 K)/Q(296 K) = 1.045, not the 1.037 of my power law); `select_lines.py` ranks candidate lines by selectivity against the breath mixture; `fetch_hitran.py nir|mir` fetches real lines on your machine. The line strengths and the H2O placeholders near the NO line are still NOT HITRAN: the selectivity ranking they give (0.005 at 1900.08 cm⁻¹) is an artefact of where I put the placeholder H2O lines and must not be quoted.

New tasks: 18. run `fetch_hitran.py mir` + `select_lines.py NO 1880 1920`; 19. 3D FDTD of the three-index slot mirror including TE-TM conversion at the thickness step; 20. design the beam splitter / delay network for K incoherent on-chip beams (loss budget, ports); 21. thermal stabilisation target ~1-10 µK (membrane, TEC + bolometric self-heating); 22. wavelength-modulation fit and many-mode source options beyond a single 1 GHz-wide laser.
