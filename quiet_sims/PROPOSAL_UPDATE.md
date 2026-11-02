# Prompt for the proposal chat: update QUIET with the new simulation results

Paste everything below this line into the proposal chat, and attach the three figure PDFs
(`fig1d_SiNx.pdf`, `fig2f_static_fringe.pdf`, `fig3_spectroscopy.pdf` from `quiet_sims/results/figs/`).

---

Please update the QUIET proposal text with the new simulation results below. Rules:

- Change only numbers, figure captions and short supporting sentences. Keep the structure, the tone and the length.
- Replace Fig. 1d, Fig. 2f and Fig. 3 (a, b) with the attached PDFs. They are Liberation Sans 10 pt, at most 16.8 cm
  wide, in the existing colours.
- Report medians and quartiles over the inclusion layouts, never single layouts.
- Where a value below is marked "to verify", keep the wording cautious ("about", "expected", "to be confirmed").
- At the end, list every sentence you changed (old → new) so I can check it.

## Decision needed first: definition of Γ

The existing Fig. 1d plots the absorption confinement factor Γ_abs = n_g ∫_gas ε|E|² / ∫ ε|E|² (Robinson et al.,
Opt. Express 16, 4296 (2008)), which exceeds 1 for thin Si. The ppb numbers below use the |E|² fraction in the gas,
Γ_E. Both give 0.64 for Si 250 nm, so the baseline is unaffected, but they differ for SiNx:

| design point | n_eff (TM0) | Γ_E (used below) | Γ_abs (Fig. 1d definition) |
|---|---|---|---|
| SiNx 600 nm, 1.55 µm | 1.657 | 0.31 | 0.23 |
| SiNx 1.2 µm, 3.3 µm | 1.588 | 0.37 | 0.30 |

Use one definition throughout. If the text adopts Γ_abs (consistent with Fig. 1d), multiply every SiNx ppb value
below by 1.36 (600 nm, 1.55 µm) or 1.26 (1.2 µm, 3.3 µm), and state the definition once in the text.

## Numbers to replace

### Near-IR SiNx cell (600 nm membrane, TM0, 1.55 µm)

Settings: circle 4 mm, 24 facets, 16 inclusions (Δn 0.01, 50 µm), launch 7.5°, ±2 mrad angle and ±20 pm wavelength
dither, 1 mW, ε = 1e-6, 0.1 dB/cm, angle-dependent trench-mirror reflectivity.

| quantity | old text | new value |
|---|---|---|
| NH3 limit at 1 s (Fig. 3b) | 52 ppb | 35 ppb (median of 21 layouts) |
| Γ⟨L⟩ | 2.5 cm | 4.7 cm |
| transmission | 0.23 | 0.34 |
| static fringe, both dithers, chaotic (Fig. 2f) | 4.2e-8 | 3.4e-8 (median of 30 layouts); regular 4.0e-8 |
| layouts below the 5e-7 target | — | 100 % |
| Herriott (C-Herriott), both dithers | — | 2.1e-7 (median), 67 % of layouts below 5e-7; NH3 150–180 ppb |
| Si 250 nm baseline (R 0.99) | 110 ppb | unchanged, 110 ppb (reproduced) |

Fig. 2f values (none / spectral / angle / both):
R-Herriott 8.7e-4 / 7.7e-5 / 8.7e-4 / 7.7e-5; C-Herriott 2.2e-4 / 6.0e-6 / 3.0e-6 / 2.1e-7;
R-CMPC 2.3e-4 / 1.9e-5 / 3.4e-6 / 4.0e-8; C-CMPC 1.3e-4 / 5.4e-6 / 2.6e-7 / 3.4e-8.

Film choice: 600 nm gives the lowest limit (34–35 ppb); 400 nm 41 ppb (its mirror reaches only R ≈ 0.993); 500 nm 36 ppb; 700 nm 71 ppb.

### Trench mirrors

- **Reflectivity:** a first-order quarter-wave trench DBR with 20 periods gives a bounce-averaged R ≈ 0.9998–1.0000 for n_eff 1.53–1.74 (0.993 for n_eff 1.36).
- **Total internal reflection:** 13–23 % of the CMPC bounces (34 % in the Herriott cell) are above the critical angle and reflect totally.
- **Roughness:** 5 nm rms sidewall roughness limits R to about 0.996; 10 nm to about 0.985.
- **Risk to state:** vertical diffraction in the air gaps is not yet included and needs a 2D FDTD or mode-matching study. An isolated-gap estimate bounds R at 0.5–0.76, which is too pessimistic for a first-order mirror but must be checked.

### Launch angle

Keep the 7.5° launch (star orbit m = 11): 33–48 ppb NH3.
- **Grazing launches** (75–82.5°) reflect 89–93 % of the bounces totally, but the path is short and the light misses the inclusions: 200–1100 ppb.
- **52.5° (m = 5)** is the fallback if the mirrors underperform: 55–65 % of its bounces reflect totally, at 66 ppb.

### Mid-infrared SiNx cell (1.2 µm membrane, 3.3 µm, Γ_E 0.37)

| quantity | old text | new value |
|---|---|---|
| line-strength gain 3.3 µm / 1.6 µm | 150–230× | 88× between the strongest CH4 lines; 237× between the Fig. 3 lines |
| HITRAN peak, 1 ppm in breath (1 atm, 35 °C, 5.5 % H2O, 4.5 % CO2) | — | CH4 3.3151 µm: 3.79e-5 /cm; C2H6 3.348 µm: 3.69e-5 /cm; CH4 1.651 µm: 4.3e-7 /cm; NH3 1.532 µm: 2.28e-7 /cm |
| CH4 and C2H6 limit at 1 s (1 mW, shot + fringe) | — | 1.5–7 ppb over films 1.0–1.5 µm and 0.1–0.5 dB/cm; 3.1 ppb for 1.2 µm at 0.3 dB/cm |
| time to 1 ppb | — | 16–95 s (fringe-limited) |
| Herriott at 3.3 µm | — | fringe-limited, 21–58 ppb (220–650 ppb at 0.5 dB/cm) |

### Noise budget, milestone M7 (10 ppb at 1 s)

Model: detector power = P_in·10^(−IL/10), shot noise √(q/P_det), detector noise NEP/P_det, plus the fringe residual. 1.2 µm SiNx circle, 0.3 dB/cm:
- **20 dB insertion loss:** 10 ppb is met with NEP ≤ 1e-11 W/√Hz from 1 mW idler, and with NEP 1e-10 only at 10 mW (margins 1.2–3.2×).
- **35 dB insertion loss:** only 10 mW with NEP 1e-12 meets 10 ppb at 1 s (2.4×); at 10 s, 1 mW with 1e-12 or 10 mW with 1e-11 also meet it.
- **1 ppb:** not reached within 10 s in any combination. The fringe floor is about 3 ppb; about 40 s of averaging reaches 1 ppb.
- **Herriott cell:** fringe-limited at about 30 ppb; it meets neither target.

### DFG source (MgO:PPLN, pump 1064 nm)

Literature parameters to verify are marked in the code.
- **Signal wavelengths:** 1566.9 nm for CH4 (3016.49 cm⁻¹; 1568.1 nm for the nominal 3.31 µm) and 1562.1 nm for the C2H6 Q-branch (2996.85 cm⁻¹).
- **Poling period:** about 30.6 µm at 60 °C. Covering 3.06–3.66 µm over 25–200 °C needs 29.2–31.3 µm, as a multi-period grating in 0.2 µm steps or a fan-out.
- **Acceptance:** for a 50 mm crystal, 4.1 °C in temperature (to verify) and 1.7 nm (about 200 GHz) at the signal; no spatial walk-off for type-0 QPM along x.
- **Idler power:** Boyd–Kleinman, d_eff 14 pm/V, 1 W pump: 36–45 mW with 5 W signal at 50 mm, 7–9 mW with 1 W signal. That is about 1.8 mW/(W²·cm) at the theoretical optimum; experiments typically reach 30–60 % of it.
- **Dither mapping:** ±20 pm at the idler is ±0.55 GHz, i.e. ±4.5 pm on the signal laser. DFB current modulation (about ±0.5–1 mA) reaches this at 1–100 kHz, and the excursion is under 1 % of the phase-matching acceptance.

### Thermo-optic angle dither (fan-in of K beams across the 30 µm port)

- **Phase:** ±2 mrad needs ±0.20 rad at the edge of the port for SiNx at 1.55 µm (64 port modes), ±0.09 rad at 3.3 µm, and ±0.25 rad for Si (78 modes).
- **SiNx index change:** dn/dT 2.45e-5 /K (Arbabi & Goddard 2013; your 2.5e-5 is confirmed), against 1.8e-4 /K for Si. This means ±10 K with a 200 µm heater.
- **Power and speed:** heater power does not depend on heater length. It scales as 1/ℓ and the time constant as ℓ², where ℓ is the distance to the heat-sunk anchors. SiNx needs ℓ ≤ about 20 µm for a 1 kHz dither, at about 0.3 mW per edge heater (0.6 mW at 3.3 µm). Si reaches tens of kHz.
- **Feasibility:** 1 s measurements are feasible, since Fig. 3b assumes a 1 kHz angle dither.
- **Assumptions:** SiNx film conductivity 2.5 W/mK and Si film 60 W/mK.

## Captions to update

- **Fig. 1d:** "Gas overlap Γ and effective index of free-standing SiN_x membranes (TM0 navy, TE0 light blue) at 1.55 µm (solid, n = 2.0) and 3.0 µm (dashed, n = 1.97). Γ = n_g∫_gas ε|E|²/∫ε|E|². Dots: design points, 600 nm at 1.55 µm and 1.2 µm."
- **Fig. 2f:** "Static residual fringe of the 600 nm SiN_x cells (angle-dependent mirror reflectivity, 0.1 dB/cm). Chaotic cells: median and interquartile range of 30 inclusion layouts. Regular cells: single values. Dashed: target 5×10⁻⁷."
- **Fig. 3a:** "1 s spectra with each dither, on the breath background (6 % H2O, 4.5 % CO2): NH3 10 ppm, CH4 and C2H6 1 ppm (HITRAN, 35 °C, 1 atm). Near-IR strips: 600 nm SiN_x circle; 3.3 µm strips: 1.2 µm SiN_x circle at 0.3 dB/cm."
- **Fig. 3b:** "Noise-equivalent absorbance against averaging time (medians and min–max bands of 21 layouts, 600 nm SiN_x circle, 1 mW). NH3 limit 35 ppb at 1 s."

## Limitations to mention (one or two sentences in the risk section)

- **Vertical diffraction in the mirror gaps:** not yet modelled (2D FDTD planned); this is the main risk to R.
- **Spectral fitting:** the limits use the peak absorption without fitting the H2O/CO2 background. At 3.31 µm, breath water absorbs as much as 0.6 ppm of CH4 would, so spectral fitting is required.
- **Layout scatter:** the medians over 21–30 chaotic layouts scatter by tens of percent between ensembles.
- **Parameters still to verify:** MgO:PPLN temperature tuning, d_eff, and SiNx thermal conductivity.
