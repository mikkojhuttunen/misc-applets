# Prompt: bring the Task 13–14 results into the newest QUIET proposal

Paste the block below into a new Claude session that has the repository `mikkojhuttunen/misc-applets` and the newest
proposal `.tex` attached.

---

You are updating the RCF 2027 proposal "QUIET" (chaotic multipass cells, CMPC, for chip-scale gas sensing) with new
simulation results. Work on the **newest** proposal file I attach. Do not start from an older file. If I have not attached
one, ask for it before editing. The previous session saw `RCF2027_CMPC_gas_sensors_multibeam_v50_cuts.tex` and wrote
`v51_fig2f` from it, which reconciled Fig. 2f with the text and stated the honest chaotic gain. Check which of those v51
edits are already in my newest file. Keep the ones that still apply, and do not apply any of them twice.

## Sources (read before writing)

- Branch `quiet-v32-realistic`, folder `QUIET_sims_v32/`. If the newest commits are missing, apply these patches from
  branch `claude/gracious-turing-tstil7`, in order: `task13-wave-vs-ray.patch`, `task14-waist-modulation.patch`,
  `figs23-task14-default.patch`, `herriott-wave-smooth-averaging.patch`, `split-dithers-handoff.patch`.
- `REPORT.md` sections 6b (wave bound), 6c–6f (beamlet model, grating input, cell size), 6g (100 ms), **6h (Task 13: wave vs
  ray)**, **6i (Task 14: waists, membrane modulation, Herriott, averaging)**.
- `results/t14/table.md`, `results/t14/outlook.json`, `results/t14/artifact_data.json`, `results/t13/table.md`.
- `results/MANIFEST.md`. Every run file has a `meta` block with its parameters.
- Live figures (the Fig. 2 and Fig. 3 sources; the newest results are their defaults):
  Fig. 2 https://claude.ai/artifact/BiVxtNey99hyrtPLYcvhm8, Fig. 3 https://claude.ai/artifact/9sezcN5zEXTv9Som5LCpBr.
  Export the paper PDFs from them. Fig. 2 (f) has a "Newest bars" menu: dithers added one by one (default), input
  combined, or each dither alone. Fig. 3 (b) has a "Dither averaging" menu: period-locked (default), boxcar, or ideal
  whole periods.

## Design and parameters (identical in Fig. 2 and Fig. 3)

- 24-facet circle CMPC, 10 mm (5, 15, 20 mm also simulated); 8 inclusions, r 25 µm, |Δn| 0.01; output on facet 11.
- Grating-coupler input: 75 µm waist, ±60 mrad angle, ±100 µm position (sine, ≥ 1 kHz).
- Wavelength dither ±81 pm at 3.3 µm (keeps 92 % of the CH₄ peak and 82 % of C₂H₆).
- **Membrane (δn) modulation:** thermo-optic, whole membrane, δn_eff 1e-4 (≈ ±9 K), 50 Hz.
- Set B: 1000 nm SiNₓ at 3.3 µm (n_eff 1.425, Γ 0.553); R 97 % (and 99.5 %); 0.3 dB/cm; ε = 10⁻⁶; 30 inclusion layouts.
- Model: wave-consistent Gaussian-beamlet model (Tasks 9–14). Modulation is a first-order phase k₀·δn·ℓ per route with the
  exact J₀ time average.
- Herriott for comparison: same length, 1.5 mm wide, the same input, dithers and modulation.

## Results to use (cite them as simulations; medians over 30 layouts)

1. **Why the old angle-dither claim fails (Task 13).** A ±2 mrad tilt of a 40 µm beam stays inside its 7.9 mrad divergence,
   so in wave optics it gives no gain. Any input dither reduces the fringe at best by N^−½, where N is the number of
   orthogonal input states the beam sweeps. That is set by waist × tilt × shift / λ, not by chaos; chaos only lets the
   cell reach N^−½. The ray model's ≈ 1e-2 fringe factor is a single-ray artefact.
2. **Each added dimension lowers the fringe** (10 mm CMPC, static fringe ΔP/P, chaotic median):
   - 3.3 µm, R 97 %: none 3.6e-4 → λ 1.8e-5 → +angle 2.1e-5 → +position 1.6e-6 → +δn 1.5e-7.
     At 3.3 µm, angle alone sweeps only N ≈ 5 states, so it gains nothing next to λ. Angle × position sweeps N ≈ 86.
   - 1.53 µm, R 97 %: 1.7e-4 → 3.7e-6 → 1.8e-6 → 2.0e-7 → 1.8e-8 (N ≈ 10 for angle alone, ≈ 400 combined).
3. **Membrane modulation** (δn 1e-4, ±9 K) lowers the static fringe ≈ 10× (1e-3: ≈ 30×). On the fringe phases it acts
   like a very deep wavelength dither, but the laser line stays fixed, so the absorption line is not smeared.
   Modulating only the small inclusions does nothing: a route spends only 0–100 µm inside them.
4. **100 ms limits (CH₄ / C₂H₆, ppb, 1σ).** Conditions: referenced every 3 s, drift 0.1 rad/s, NEP 14 pW/√Hz, two
   couplers 8 dB, averaging locked to the dither periods.

   | Cell | 1 mW | 10 mW | 50 mW |
   |---|---|---|---|
   | 10 mm, R 97 % | 13 / 17 | 1.34 / 1.69 | 0.31 / 0.39 |
   | 10 mm, R 99.5 % | 7.0 / 8.9 | 0.74 / 0.94 | 0.19 / 0.24 |
   | 20 mm, R 97 % | | 1.90 / 2.41 | 0.39 / 0.50 |

   Without δn (10 mm, R 97 %, 10 mW): 1.77 / 2.25. With boxcar instead of locked averaging the 1 kHz remainder adds
   noise (Fig. 3: 2.6 ppb CH₄ at 100 ms, no referencing, against 1.4 locked). At 1 mW (the proposal's DFG baseline)
   the detector sets the limit.
5. **Herriott, same design** (10 mm, 3.3 µm, R 97 %): all dithers 1.4e-6 against 1.5e-7 for the CMPC; Γ⟨L⟩ 3.7 cm
   against 11 cm. Light leaves through the re-entrance port after short paths. The chaotic CMPC wins on both fringe and
   path.
6. **Caveats to state.**
   - Regular-cell numbers from the beamlet model are indicative only.
   - The modulation model is first order (route geometry fixed).
   - The thermal side effects are not modelled: gas density and line strength swing ±3 % at f_mod; film stress; thermal
     gradients deflecting routes.
   - Heating the whole 10 mm membrane by ±9 K at 20–50 Hz needs ≈ 0.1–0.25 W absorbed (gas-gap time constant
     ≈ 6–7 ms), from a CW pump or heaters kept out of the beam paths.

## Novelty framing to add (keep it accurate)

Present the noise suppression as decorrelation along independent dimensions, each one multiplying the number of
independent fringe realisations:
- the wavelength (classic dither, limited by the line width);
- the input angle;
- the input position, which together with the angle spans the input phase space through a wide grating coupler;
- a new dimension: modulating the refractive index of the guiding medium itself inside the cell (δn).

The δn dimension is the chip-specific advantage. In a free-space cell the medium is the gas, whose index cannot be
modulated usefully. Free-space path modulation needs mechanical mirror dithering (piezo), which exists and should be
acknowledged as prior practice. On a suspended membrane the whole optical path sits in a solid film with a tiny heat
capacity. Its index can be swung by ≈ 1e-4 in milliseconds, with no moving parts, by a CW pump or heaters, and it
can be patterned into zones. Do not claim that free-space cells have no path modulation at all. Claim that
modulating the medium's index along the whole path, all-optically or thermally and without moving parts, is
specific to the integrated membrane cell.

## Where to edit (adapt to the newest file's structure)

- Sec. 1.1 / Novelty paragraph: the dimensionality framing above, in 2–4 sentences.
- RO1 objective and hypothesis: angle + position (input phase space) and δn as the mechanisms. The hypothesis should
  be testable, e.g. "each added dimension lowers the residual fringe by the number of states it adds, down to …".
- Fig. 2 caption: describe panel (f) as exported (default "dithers added one by one"). Name the design and
  parameters, define V and M, and say the regular cells are indicative.
- Fig. 3 caption: the newest data set, period-locked averaging, 10 mW detector model, 100 ms marker.
- Sec. 3.2 "Preliminary results": replace the numbers with the ones above. Keep the honest caveats.
- T1.2 (grating input + position dither) and a task or subtask for the membrane modulation: heaters or a CW pump, the
  thermal design, and measuring the δn decorrelation. Place it in the existing WP structure. Do not change budgets
  without asking me.
- T1.3 / Risks (R1.3, R3.4): margins now rest on δn modulation, locked averaging and laser power. 1 mW is
  detector-limited.
- `\cite{CMPCpreprint}` placeholder: keep the `\tbd{}` until I give the arXiv number.

## Rules

- Mark new or changed text with a new revision colour macro (as the file's existing `\nc{}` / `\fc{}` pattern), and list
  every change at the end.
- No invented citations or numbers. Use only the numbers above or ones you recompute from the result files (say how).
- Keep the page budget. If text must go, propose the cut and do not delete silently.
- Deliver the edited `.tex`, a diff against my newest file, and a short list of claims that changed.
