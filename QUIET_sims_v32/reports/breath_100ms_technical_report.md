# Breath-resolved gas sensing at 100 ms with the membrane CMPC: possible measurements and setup considerations

Technical report, QUIET v32, Task 12. 6 November 2026.

Status: simulation estimates only, from the Task 11 grating-coupler design and the Task 12 noise and gas-exchange
calculations. Nothing here has been measured. Every number can be reproduced with `python scripts/t12_fast.py`
(see §10). Background: discussion with L. Lehtimäki (asthma, patient breath measurements) on measuring at 100 ms
time resolution and several compounds, in particular methane and ethane from the same patient, ideally at the same time.

---

## 1. Summary

- **100 ms is within reach for the 3.3 µm channel (CH₄, C₂H₆).** The sensing volume of an evanescent-field membrane
  is tiny (tens of µL above the chip). A shallow flow channel exchanges the gas in 10–60 ms, and diffusion to the
  surface takes 1–6 ms. The optics (≥ 1 kHz wavelength dither, kHz input scanner) average well within 100 ms.
- **What limits 100 ms is the static stray-light fringe, not the detector.** At 15–30 s, thermal drift of the fringe
  phase averages the fringe down; at 0.1 s it cannot. The fix is **breath-by-breath referencing**: pass inhaled or
  zero air through the same cartridge in every breath and subtract. With a reference every 3 s, only the fringe change
  over 3 s is left (30 % at the assumed 0.1 rad/s drift), which makes the measurement 2.2–2.6 times better.
- **0.1 s limits (1σ), 10 mm cell, R 99.5 %, 10 mW, TE-cooled detector, referenced:** CH₄ 1.35 ppb, C₂H₆ 1.71 ppb,
  NH₃ 230 ppb. With a thermally stabilised cartridge (drift 0.02 rad/s): CH₄ 0.75 ppb, C₂H₆ 0.95 ppb.
- **CH₄ and C₂H₆ from one laser, in the same 100 ms frame:** line hopping in the 2990 cm⁻¹ window (C₂H₆ 2990.06 cm⁻¹;
  CH₄ 2988.93 and 2989.02 cm⁻¹, 1.1 cm⁻¹ away). With 80 % of the time on ethane: C₂H₆ 1.9 ppb and CH₄ 5.2 ppb at
  0.1 s. Fitting a continuous scan over the whole window is 6–7 times worse for ethane, because most of the time is
  spent off the line.
- **Clinical reading:** breath CH₄ (ppm level) is easy at 100 ms. Breath C₂H₆ is low-ppb, so single 100 ms points
  resolve ethane only when it is elevated. Clinically useful ethane values (≤ 0.5 ppb) need averaging over the
  end-tidal plateau and 5–10 breaths. NH₃ at 100 ms covers only high breath levels. FeNO (5.3 µm) is not covered by
  this design (§7).
- **The biggest unknowns,** to be measured first: the fringe-phase drift rate of a real cartridge, the spectral
  correlation of the fringe, condensation of breath humidity on the membrane, and the current-tuning range of the
  laser (§9).

## 2. What is worth measuring in breath, and how well

Typical levels below are literature ranges to be confirmed with the clinical partner. The "needed" column is the
author's working assumption.

| Compound | Line used here | Typical breath level | Needed 1σ for a useful 100 ms trace | Feasible at 100 ms? |
|---|---|---|---|---|
| CH₄ | 3.31 µm (main) or 2988.9–2989.0 cm⁻¹ (with C₂H₆) | ambient ≈ 1.9–2 ppm; methane producers several ppm above ambient | ≈ 20–50 ppb | yes, by orders of magnitude |
| C₂H₆ | 2990.06 cm⁻¹ (or 2996.85 cm⁻¹) | low ppb (≈ 1 ppb in healthy subjects; reported elevated in oxidative stress and inflammation, including asthma); ambient 1–3 ppb | ≈ 0.2–0.5 ppb | elevated levels only; plateau and multi-breath averaging needed |
| C₂H₆/CH₄ ratio | both above | exhaled ethane relative to methane | ± a few % of the ratio | yes for elevated ethane (Fig. 3 readout: 5.00 ± 0.09 % for 2.5 ppm CH₄ and 125 ppb C₂H₆ at 0.1 s) |
| NH₃ | 1.53 µm | tens to thousands of ppb (oral ammonia dominates) | ≈ 20–50 ppb | only high levels; adsorption limits the response time (§6) |
| CO₂ | any weak band (not simulated) | 4–5 % end-tidal | ≈ 0.05 % | yes (percent level); needed to mark the breath phases |
| NO (FeNO) | 5.3 µm | ≈ 5–50+ ppb at 50 mL/s; clinical cut-offs in the 25–50 ppb range | ≈ 1 ppb | not with this design (§7) |

Breath time scales: a breath cycle is 3–5 s, and the end-tidal (alveolar) plateau lasts about 1–2 s. 100 ms
resolution resolves the expiratory profile (dead space, rise, phase III slope), which is what makes it interesting
beyond end-tidal averages.

## 3. Measurement model and assumptions

Source: `scripts/t12_fast.py` on the Task 11 runs (`results/t11/final_*.json`, 30 layouts per cell, medians).

- **Design:** 24-facet circle CMPC, 8 index inclusions (r 25 µm, |Δn| 0.01), output on facet 11. Free-space input on
  a large-area grating coupler (75 µm waist, ±60 mrad angle and ±100 µm position dither). Set B: SiNₓ 500 nm at
  1.53 µm, 1000 nm at 3.3 µm, 0.3 dB/cm.
- **Wavelength dither:** ±35 pm at 1.53 µm keeps 80 % of the NH₃ peak. ±81 pm (±2.2 GHz) at 3.3 µm keeps 92 % of the
  CH₄ peak and 82 % of the C₂H₆ peak.
- **Fringe residual:**
  - r(T) = √(S_both² H_d² + (S_none² − S_both²) H_ac²).
  - Drift envelope H_d(T) = 1/√(1 + 2(wT/2)²) for fringe-phase drift rate w, with w = 0.1, 0.05 and 0.02 rad/s.
    Task 11 used 0.1 rad/s, which is an assumption, not a measurement.
  - AC part averaged at the ≥ 1 kHz dither frequencies.
- **Breath referencing:** a baseline is subtracted every τ = 3 s. The static fringe is then left only as its change,
  2 sin(wτ/2): 0.30 at 0.1 rad/s and 0.06 at 0.02 rad/s.
- **Detector:** √(NEP² + 2hνP_d)/(P_d √(2T)), with P_d = P × T_cell × 10^(−0.8) (8 dB for the two grating couplers).
  - NEP 14 pW/√Hz. This is a TE-cooled InAsSb or HgCdTe photodiode, as long as its D* is high enough at 3.3 µm.
  - 1 nW/√Hz, the uncooled case, is also listed.
  - P = 10 and 20 mW.
- **Breath averaging:** σ/√n over n breaths. This assumes the referenced fringe residual is independent from breath
  to breath.
- **Line conditions:** HITRAN through HAPI on the repository's cached line lists, at breath conditions (35 °C,
  5.5 % H₂O, 4.5 % CO₂, 1 atm).

## 4. Detection limits at 0.1–2 s

1σ, ppb, median over 30 layouts. "raw" is without referencing; "ref" is referenced every 3 s.

| Cell | R | Gas | 0.1 s raw | 0.1 s ref | 0.1 s ref, 0.02 rad/s | 0.1 s ref, 20 mW | 0.1 s ref, uncooled (1 nW/√Hz) |
|---|---|---|---|---|---|---|---|
| 10 mm | 97 % | CH₄ | 4.4 | 2.0 | 1.3 | 1.5 | 89 |
| 10 mm | 97 % | C₂H₆ | 5.6 | 2.6 | 1.7 | 1.9 | 113 |
| 10 mm | 99.5 % | CH₄ | 3.5 | 1.35 | 0.75 | 1.1 | 49 |
| 10 mm | 99.5 % | C₂H₆ | 4.5 | 1.7 | 0.95 | 1.4 | 62 |
| 15 mm | 99.5 % | CH₄ | 3.0 | 1.35 | 0.92 | 0.98 | 64 |
| 20 mm | 99.5 % | CH₄ | 3.1 | 1.4 | 1.0 | 1.0 | 70 |
| 10 mm | 99.5 % | NH₃ | 316 | 230 | 219 | 133 | – |
| 20 mm | 99.5 % | NH₃ | 319 | 304 | 303 | 159 | – |

Against averaging time (10 mm, R 99.5 %, referenced, 0.1 rad/s):

| Gas | 0.1 s | 0.5 s | 1 s | 2 s | 1 s × 5 breaths | 1 s × 10 breaths |
|---|---|---|---|---|---|---|
| CH₄ | 1.35 | 1.05 | 1.02 | 0.98 | 0.46 | 0.32 |
| C₂H₆ | 1.71 | 1.33 | 1.29 | 1.25 | 0.58 | 0.41 |
| NH₃ | 230 | 106 | 79 | 59 | 35 | 25 |

Reading the tables:

- At 0.1 s and 3.3 µm, the fringe and detector parts are about equal (10 mm, R 99.5 %, referenced: 1.03 / 0.71 ppb
  for CH₄). Referencing and thermal stability therefore act on one half of the noise, and power and detector on the
  other.
- In the near-IR, NH₃ is detector-limited at 0.1 s: T_cell is only 2–5 % at 10–20 mm. 20 mW helps (230 → 133 ppb) and
  referencing barely does.
- Larger cells do not help at 100 ms. 10–15 mm is the sweet spot, because transmission falls faster than the fringe.
- An uncooled mid-IR detector is not usable at 100 ms. Use a TE-cooled one.

Full grid (cells 5–20 mm, both R, three drift rates, raw and referenced, 10 and 20 mW, both NEPs, 0.1–2 s):
`results/t12/fast.json`; summary in `results/t12/table.md`; plot in `figs/t12_fast.pdf`.

## 5. Methane and ethane together

**Line choice.** Strong C₂H₆ Q-branches lie at 2970–2997 cm⁻¹. CH₄ lines within about 1 cm⁻¹ of one exist at
2980.08 and 2990.06 cm⁻¹:

| Window | Lines | Notes |
|---|---|---|
| W2990 (recommended) | C₂H₆ 2990.06; CH₄ 2988.93 / 2989.02 (α ≈ 1.4 × 10⁻⁵ /(ppm cm), 37 % of the 3.31 µm line) | low H₂O under the C₂H₆ peak; a strong H₂O line at 2988.6 cm⁻¹ and one at 2989.6 cm⁻¹ sit next to the CH₄ pair, so breath H₂O must be tracked |
| W2980 | C₂H₆ 2980.08; CH₄ 2978.92 / 2979.00 | H₂O on the C₂H₆ peak |
| W2997 | C₂H₆ 2996.85 (cleanest, the line used in Fig. 3); CH₄ 2999.00 (weak, 20 % of the 3.31 µm line) | 2.15 cm⁻¹ apart: probably beyond current tuning of one ICL |

**How to measure, one laser (2990 cm⁻¹ ICL, current-tuned over about 1.5 cm⁻¹):**

| Mode | 0.1 s CH₄ / C₂H₆ (10 mm, R 99.5 %, ref.) | 1 s | Comment |
|---|---|---|---|
| Full scan of the window, least-squares fit (C₂H₆, CH₄, H₂O, linear baseline) | 17 / 9.4 ppb | 13 / 7.1 ppb | robust to interferents, but 6.6× (C₂H₆) and 14× (CH₄) worse than staying on the peak |
| Line hopping, 80 % on C₂H₆, 20 % on CH₄ | 5.2 / 1.9 ppb | 3.0 / 1.4 ppb | recommended; both gases in every 100 ms frame |
| Line hopping, 50 / 50 | 4.0 / 2.1 ppb | 2.8 / 1.5 ppb | |
| Two lasers (3.31 and 3.34 µm), T/2 each | 1.7 / 2.2 ppb | – | two lasers; time-shared within each frame. With frequency-multiplexed dithers both could run all the time (1.35 / 1.71 ppb), at the cost of a second laser and the demodulation |

- **Recommendation:** line hopping, with short scans across each line rather than a single point, so the line centre
  can be locked and the local baseline tracked. Add a periodic full-window scan (for example 1 in 10 frames) to fit
  H₂O and the baseline.
- **Caveat:** the fit and hopping numbers assume white noise from point to point. The real fringe is correlated over
  about 0.05–0.1 cm⁻¹ (similar to the line width), so it fits partly as a false line; referencing removes most of it.
- **Small inconsistency:** the dither is modelled here as a boxcar, while the earlier smear factors used a triangle
  wave. This gives about a 5 % difference in the C₂H₆ peak.

## 6. Cartridge and gas handling (single-use part)

- **Flow channel above the membrane:** 0.2–0.3 mm high. Volume and exchange time:
  - 10 × 10 mm × 0.3 mm = 30 µL, flushed in 60 / 30 / 15 ms at 0.5 / 1 / 2 mL/s.
  - 20 × 20 mm × 0.3 mm = 120 µL, flushed in 240 / 120 / 60 ms at the same flows.
  - Diffusion across the channel takes 2–3 ms for CH₄ and C₂H₆.
  - A side stream of 1–2 mL/s, of an exhalation at 50 mL/s, gives < 100 ms response for cells up to 15 mm.
- **Backside of the membrane:**
  - The guided mode also senses the gas under the membrane. The limits in §4 assume breath on both sides (Γ counts
    both claddings).
  - Either flow breath on both sides, or seal the back with dry N₂ and accept half the signal (about 2× worse limits).
  - A stagnant backside cavity would add a slow, lagging component to the response, so it must not be left open to
    breath without flow.
- **Humidity and condensation:**
  - Exhaled breath is saturated at about 35 °C, so the cartridge must be held above the dew point, for example at
    40 °C, and stabilised.
  - Condensed water on the membrane changes Γ and adds loss.
  - Thermal stabilisation also lowers the fringe drift (§4: 0.1 → 0.02 rad/s halves the 100 ms limit), so one
    heater and controller serve both.
- **Sticky gases:** NH₃ (and NO) adsorb on surfaces. Use short heated lines, passivated or PTFE-like wetted surfaces,
  and a hydrophobic coating on the holder. Expect an NH₃ response time of seconds unless this is engineered.
- **Referencing valve:**
  - Switch inhaled air, or scrubbed zero air, through the same cartridge in each breath cycle (at 1–2 mL/s the switch
    is complete in < 100 ms).
  - Ambient C₂H₆ (1–3 ppb) and CH₄ (≈ 2 ppm) are then measured too. That is needed for exhaled − inhaled differences
    in any case.
- **Pressure and temperature sensors in the cartridge flow path:** absorbance scales with pressure and line strength
  with temperature. A 1 % pressure error is a 1 % concentration error.
- **Single-use aspects:** the chip holder has windows (for the free-space input beam and the output couplers), gas
  ports and a filter at the inlet against saliva and particles. There are no electrical parts. Because a fresh
  cartridge has a different fringe pattern, each one needs a zero-air baseline at the start of a session.

## 7. What this design does not cover

- **FeNO at 5.3 µm.** The strongest NO line in breath is at 1903.13 cm⁻¹, α = 1.7 × 10⁻⁵ /(ppm cm) (0.44 × the CH₄
  line at 3.31 µm). It sits on 3.5 × 10⁻⁵ /cm of H₂O and CO₂ background; a cleaner line is at 1929.02 cm⁻¹
  (α = 6.6 × 10⁻⁶, background 1.7 × 10⁻⁶ /cm).
  - On line strength alone, the 3.3 µm cell would give about 3 ppb NO in 0.1 s (1.35 ppb ÷ 0.44), so 1 ppb needs about 3× more Γ⟨L⟩ or less noise.
  - The SiNₓ membrane, the DBR and the grating couplers have not been modelled at 5.3 µm. SiNₓ absorption rises towards
    the mid-IR and is the first thing to check.
  - FeNO stays a separate study.
- **CO₂ for breath phases.** Not in the current optics. A low-cost option is a separate NDIR CO₂ sensor in the reader's
  gas path; alternatively, a weak CO₂ band could be added to one of the lasers.

## 8. Reader setup (reusable part)

| Block | Requirement from this study | Notes |
|---|---|---|
| Laser | ICL at 2990 cm⁻¹ (3.344 µm), 10–20 mW, current tuning ≥ 1.5 cm⁻¹ at kHz rates, ±81 pm (±2.2 GHz) wavelength dither | check the specific device's current-tuning range; a second laser at 3.31 µm is the fallback for CH₄ |
| Beam scanner | angle ±60 mrad and position ±100 µm at the grating coupler; all input states (10²–10³) covered within each 100 ms frame, so ≥ 1–10 kHz state rate | resonant MEMS or two-axis MEMS with a scan lens; galvos are too slow; synchronise frames to the scan period |
| Detectors | TE-cooled InAsSb or HgCdTe, NEP ≤ 14 pW/√Hz, bandwidth ≥ 10× the dither frequency | uncooled detectors fail at 100 ms (§4) |
| Reference channel | on-chip reference tap to Det. R | normalises laser power and coupler transmission (window fouling, cartridge-to-cartridge variation) |
| Acquisition | 100 ms frames; demodulation of the wavelength dither; line-hopping control; averaging over scanner states | log flow, pressure, temperature and CO₂ on the same time base |
| Gas path | side-stream sampler, referencing valve, heated lines (≥ 40 °C), flow control 1–2 mL/s | FeNO-style breathing manoeuvre at a fixed flow (for example 50 mL/s) if comparing with clinical FeNO practice |

## 9. Risks, unknowns and the first experiments

1. **Fringe-phase drift rate** of a mounted, heated cartridge. Measure the fringe residual against time with zero air.
   This decides whether 0.1 s gives 1.35 ppb (0.1 rad/s) or 0.75 ppb (0.02 rad/s) for CH₄.
2. **Spectral correlation of the fringe**, and how much of it a two-line hop or a fit absorbs. Measure the zero-air
   spectrum across the 2990 cm⁻¹ window.
3. **Laser:** the current-tuning range and the hop settling time of the ICL at 2990 cm⁻¹ (line hopping within one
   100 ms frame needs settling in about 1 ms).
4. **Condensation and adsorption** on the membrane at breath humidity: response time for CH₄ (fast) against NH₃ (slow).
5. **Scanner speed and dither-state coverage** within 100 ms. Check against the N^−½ law of Task 9b with a measured
   fringe.
6. **Model validity:** all limits come from the beamlet model on simulated layouts (Task 11). Fabrication tolerances
   and real coupler losses (the 8 dB is an assumption) shift them directly.

## 10. Reproduction

```
cd QUIET_sims_v32
python scripts/t11_analysis.py
python scripts/t12_fast.py
python scripts/build_fig23_v32.py
```

Outputs:

- `results/t12/fast.json` (all limits, window fits, line hopping, gas exchange, NO).
- `results/t12/table.md`.
- `figs/t12_fast.pdf`:
  - (a) limits against averaging time for CH₄, C₂H₆ and NH₃: raw, referenced, and referenced at 0.02 rad/s;
  - (b) the 2990 cm⁻¹ window at breath conditions.
- Fig. 3 artifact: analysis times 0.01–60 s, a "breath referencing every 3 s" checkbox, and the C₂H₆/CH₄ ratio in
  line-hopping mode (2990 cm⁻¹, 80/20) or in the earlier mode (main lines, T/2 each).
- Fig. 2 artifact: 0.1 s column in the design table.
