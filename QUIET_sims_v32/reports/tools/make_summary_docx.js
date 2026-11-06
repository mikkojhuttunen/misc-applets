const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  AlignmentType, LevelFormat, PageBreak, Footer, PageNumber, TableOfContents, BorderStyle } = require("docx");

const FIG = "/tmp/claude-0/-home-user-misc-applets/dcb7d7e6-8665-5d69-a2fd-bb9f54a2d33e/scratchpad/docfig/";
const OUT = "/home/user/misc-applets/QUIET_sims_v32/reports/QUIET_simulation_summary.docx";
const FONT = "Arial", W = 9638;   // A4, 2 cm margins: 11906 - 2 * 1134

const t = (s, o = {}) => new TextRun({ text: s, font: FONT, size: o.size || 20, bold: o.b, italics: o.i, color: o.color });
const runs = s => {   // **bold** and `code` inline markup
  const out = []; const re = /(\*\*[^*]+\*\*|`[^`]+`)/g; let i = 0, m;
  while ((m = re.exec(s))) { if (m.index > i) out.push(t(s.slice(i, m.index)));
    const x = m[0]; out.push(x.startsWith("**") ? t(x.slice(2, -2), { b: true }) : new TextRun({ text: x.slice(1, -1), font: "Consolas", size: 18 })); i = m.index + x.length; }
  if (i < s.length) out.push(t(s.slice(i))); return out; };
const P = (s, o = {}) => new Paragraph({ children: runs(s), spacing: { after: o.after ?? 100, before: o.before ?? 0 }, alignment: o.align });
const H1 = s => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(s, { size: 30, b: true })] });
const H2 = s => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(s, { size: 24, b: true })] });
const H3 = s => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [t(s, { size: 21, b: true })] });
const B = (s, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(s), spacing: { after: 60 } });
const N = s => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(s), spacing: { after: 60 } });
const EQ = s => new Paragraph({ children: [new TextRun({ text: s, font: "Cambria Math", size: 20 })], indent: { left: 567 }, spacing: { after: 80 } });
const CODE = s => new Paragraph({ children: [new TextRun({ text: s, font: "Consolas", size: 17 })], indent: { left: 340 }, spacing: { after: 0 } });
const IMG = (f, w, h, cap) => [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + f), transformation: { width: 640, height: Math.round(640 * h / w) },
      altText: { title: f, description: cap, name: f } })] }),
  new Paragraph({ children: [t(cap, { size: 18, i: true })], spacing: { after: 200 } })];
const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B7BC" };
const TBL = (head, rows, fr) => {
  const tot = fr.reduce((a, b) => a + b, 0), cw = fr.map(x => Math.round(W * x / tot)); cw[cw.length - 1] += W - cw.reduce((a, b) => a + b, 0);
  const cell = (s, i, h) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: h ? { type: ShadingType.CLEAR, color: "auto", fill: "E4EAF2" } : undefined,
    children: [new Paragraph({ children: h ? [t(s, { size: 17, b: true })] : runs(s).map(r => { r.root && 0; return r; }), spacing: { after: 0 } })] });
  const small = s => new Paragraph({ children: [t(s, { size: 17 })], spacing: { after: 0 } });
  const cellS = (s, i) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    borders: { top: border, bottom: border, left: border, right: border }, children: [small(s)] });
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: cw,
    rows: [new TableRow({ tableHeader: true, children: head.map((s, i) => cell(s, i, true)) }), ...rows.map(r => new TableRow({ children: r.map((s, i) => cellS(s, i)) }))] });
};
const GAP = () => new Paragraph({ children: [], spacing: { after: 120 } });

const C = [];
// ------------------------------------------------------------------ title
C.push(new Paragraph({ children: [t("QUIET membrane CMPC gas sensors", { size: 40, b: true })], spacing: { after: 80 } }));
C.push(new Paragraph({ children: [t("Summary of the simulations behind the Fig. 1–3 artifacts: models, data, results, realism and next steps", { size: 26 })], spacing: { after: 200 } }));
C.push(P("Repository `mikkojhuttunen/misc-applets`, branch `quiet-v32-realistic`, folder `QUIET_sims_v32/` (plus `quiet_sims/` for the older step 10–11 data). State of 6 November 2026 (commit 45990e3 and this report). Proposal: RCF2027_CMPC_gas_sensors_multibeam_v32. All results are simulations; nothing here has been measured."));
C.push(P("Artifacts: Fig. 1 https://claude.ai/artifact/VFiq7T1cwXerfD6nQ8L54p · Fig. 2 https://claude.ai/artifact/BiVxtNey99hyrtPLYcvhm8 · Fig. 3 https://claude.ai/artifact/9sezcN5zEXTv9Som5LCpBr (private to the owner)."));
C.push(P("**For a future session:** §2 says which model and data file each figure panel uses; §3 gives the formulas; §6 ranks how realistic each ingredient is; §8 maps every result file and its JSON keys. The detailed per-task tables are in `QUIET_sims_v32/REPORT.md` (854 lines); the 100 ms breath study is in `QUIET_sims_v32/reports/breath_100ms_technical_report.md`."));

// ------------------------------------------------------------------ 1 summary
C.push(H1("1. Summary"));
C.push(P("The project simulates the static stray-light fringe (the noise floor of multipass absorption) and the gas detection limits of membrane multipass cells: a 24-facet circular CMPC (chaotic multipass cell, with weak index inclusions) and an integrated Herriott cell, both on a free-standing SiNₓ membrane with trench-DBR facets. The fringe is suppressed by dithering the wavelength and the input state (angle, position) and by averaging."));
C.push(B("**Two generations of model.** (i) A single-ray 2D engine (Fig. 2 a–f, Fig. 3 data sets B03/B05/S10/A, robustness panels). (ii) A wave-consistent Gaussian-beamlet model (Tasks 9–12; Fig. 2 j–k, Fig. 3 data sets G5–G20, Task 12). Task 5b showed that the single-ray engine is optimistic for the angle dither by 6–170× because a ray resolves sub-diffraction tilts; the beamlet model removes this."));
C.push(B("**Realistic set B** (500 nm SiNₓ at 1.53 µm, 1000 nm at 3.3 µm, constant R 95–99.5 %, 0.3–0.5 dB/cm, ε = 1e-6, NEP 14 pW/√Hz) moves most proposal targets from PASS to marginal or FAIL for the 4 mm cell; the grating-coupler design in 10–20 mm cells (Task 11) recovers them."));
C.push(B("**Current best design (Task 11):** free-space beam on a large-area input grating coupler (75 µm waist, ±60 mrad angle and ±100 µm position dither, N ≈ 400 input states), 8 inclusions r 25 µm |Δn| 0.01, output on facet 11, ±35 pm (1.53 µm) / ±81 pm (3.3 µm) wavelength dither, 10 mW, two grating couplers (8 dB)."));
C.push(GAP());
C.push(TBL(["quantity (median of 30 layouts)", "4 mm cell, old launch", "10 mm design", "15 mm design", "20 mm design"], [
  ["NH₃ 1.53 µm, 30 s, R 97 / 99.5 %", "353 / 176 ppb", "61 / 46 ppb", "35 / 24 ppb", "32 / 19 ppb"],
  ["CH₄ 3.31 µm, 30 s, R 97 / 99.5 %", "13.5 / 7.7 ppb", "1.6 / 1.4 ppb", "0.91 / 0.86 ppb", "1.1 / 0.81 ppb"],
  ["C₂H₆ 3.34 µm, 30 s, R 97 / 99.5 %", "17 / 9.7 ppb", "2.0 / 1.8 ppb", "1.2 / 1.1 ppb", "1.4 / 1.0 ppb"],
  ["Cell transmission T, R 97 %", "0.111", "0.027", "0.018", "0.010"],
  ["Γ⟨L⟩ (cm)", "3.8", "10.5", "15.5", "20.7"],
  ["CH₄ / C₂H₆ at 0.1 s, R 99.5 %, breath-referenced", "–", "1.35 / 1.71 ppb", "1.35 / 1.71 ppb", "1.39 / 1.77 ppb"],
], [3.2, 1.6, 1.6, 1.6, 1.6]));
C.push(P("These assume fringe drift 0.1 rad/s (averaging the static fringe at T ≳ 10 s), ε = 1e-6, NEP 14 pW/√Hz also in the mid-IR (TE-cooled detector), 8 dB coupler loss and the 2D beamlet model. With an uncooled mid-IR detector (1 nW/√Hz) CH₄ at 30 s is ≈ 6–7 ppb.", { before: 80 }));
C.push(P("**Realism in one sentence:** the relative trends (N^−½ input-dither law, cell-size lever, facet choice, sparse weak scatterers) are robust within the model; absolute limits hinge on four unmeasured inputs: the stray fraction ε, the fringe drift rate, the real coupler and membrane losses, and out-of-plane scattering at inclusions and trenches (none in the 2D model)."));

// ------------------------------------------------------------------ 2 figures
C.push(H1("2. The figures and where their numbers come from"));
C.push(H2("2.1 Fig. 1: schematics (no simulation data except the ray traces)"));
C.push(P("Built by `scripts/build_fig1_gc.py` from `artifacts/fig1_src_live.html` → `artifacts/fig1_v32.html`. Three input designs selectable with the “Input design” buttons; the default is the reader + cartridge layout."));
C.push(...IMG("fig1_cart.png", 1996, 658, "Fig. 1, input design “Reader + cartridge” (default). (a) CMPC chip and (b) integrated Herriott cell, 3D renders with the grating input port (gold), no free-space beam drawn, bottom 20 % cropped; (c) evanescent field; (d) Γ against SiNₓ thickness at 1.55 / 3.3 µm (TM₀/TE₀; dots: 500 nm and 1000 nm designs); (e) system: I source + beam scanner and III detection (reusable reader), II single-use cartridge with windows, gas tubing, grating port, CMPC, output grating couplers and an on-chip reference tap."));
C.push(...IMG("fig1_gc.png", 1996, 758, "Fig. 1, input design “Grating coupler + free-space dither”: (e) with separate blocks I source, II dither (scanning mirror and scan lens), III chip (ray trace of the Task 11 design: 8 inclusions r 25 µm, centre ray and a +45 mrad / +100 µm edge state), IV detection."));
C.push(P("Data inside Fig. 1: (a) the 3D CMPC orbit is a live engine trace (`memOrbit`, 8 × 25 µm inclusions in the new designs, 16 × 50 µm in the waveguide design); (b) the Herriott trace is an exact 2D circular-mirror trace with the trench-DBR reflectance per hit (TM₀, lossless 1D model); (d) Γ from the in-page slab solver (`gammaOf`, Si₃N₄ Sellmeier); (e) ray trace from DitherLab (`G.marchRay`). The Herriott DBRs carry mechanical bridges at |y| = 1.2 A (outside the spot pattern; drawn only)."));

C.push(H2("2.2 Fig. 2: ray tracing, residual fringe, robustness, design scaling"));
C.push(P("Built by `scripts/build_fig23_v32.py` (functions `build2`, `build2_design`) from `artifacts/fig2_src_live.html` → `artifacts/fig2_v32.html`; the source also carries the other session's labels-in-schematics patch. A “Data set” selector switches B03 (default) / B05 / S10 / A."));
C.push(...IMG("fig2_main.png", 2048, 1038, "Fig. 2 main sheet, data set B03 (set B, 500 nm SiNₓ, 1.53 µm, 0.3 dB/cm). (a)–(d) live engine ray traces of regular and chaotic CMPC and Herriott cells; (e) Poincaré sections; (f) static residual fringe ΔP/P per dither mode (none, spectral ±20 pm, angle ±2 mrad, both), medians and quartiles over 30 layouts, target 5×10⁻⁷. Source: `results/t2/*_nir_*` (single-ray engine). The angle and both bars are optimistic by the Task 5b factor (wave-bounded both-dither value ≈ 3.4e-6 at R 97 %)."));
C.push(...IMG("fig2_rob.png", 1996, 716, "Fig. 2 robustness panels (g)–(i): static fringe against wavelength-dither depth and SiNₓ thickness 500–700 nm, regular vs chaotic CMPC. Source: older QUIET step 11 (`quiet_sims/results/step11_artifact_data.json`; 1.55 µm, designed R(χ) tables, 0.1 dB/cm, ±2 mrad angle dither, single-ray engine). Not updated to set B or the beamlet model; the angle-dither parts are optimistic (Task 5b)."));
C.push(...IMG("fig2_design.png", 1996, 762, "Fig. 2 design add-on (Task 11, beamlet model). (j) static fringe per dither mode for the selected cell size (buttons 5 / 10 / 15 / 20 mm), chaotic median and min–max band, regular (indicative only), dashed: the old 4 mm cell with ±2 mrad, 40 µm launch; (k) both-dither fringe against the number of input states N for 17 input-dither designs and four sizes, dashed N^−½. Source: `results/t11/artifact_data.json` (from `final_nir_R0.97.json` etc.). A table under the panel gives limits at 0.1, 1, 15, 30 s."));

C.push(H2("2.3 Fig. 3: spectra and noise floor"));
C.push(P("Built by `build3`, `build3_more`, `build3_species`, `build3_gc`, then `apply_external` (re-applies `artifacts/fig3_external_edits.patch`: edits made on the published artifact by another session: CH₄ default line, 0.1 s default, NH₃ strip optional, axis tweaks) and `build3_fast` (Task 12 options) → `artifacts/fig3_v32.html`."));
C.push(...IMG("fig3_main.png", 2048, 564, "Fig. 3 main sheet (as published, data set G10: grating-coupler design, 10 mm). (a) HITRAN absorption strips at breath conditions (35 °C, 1 atm, 5.5 % H₂O, 4.5 % CO₂) with noise realisations from the exit paths of a representative layout; (b) noise-equivalent absorbance (NEA) against averaging time for none / λ / angle / both dithers, the detector term and the ppb axis; marker at the analysis time (here 0.1 s: 4.5 ppb CH₄, R 97 %, not referenced). Options: data sets G5–G20 and B03/B05/S10/A, noise model, drift toggle, breath referencing every 3 s, C₂H₆/CH₄ ratio readout (line hopping at 2990 cm⁻¹ or two lines in turn)."));
C.push(...IMG("fig3_rob.png", 1996, 714, "Fig. 3 robustness panels (c)–(d): NH₃ limit at 1 s against film thickness, both dithers and angle only. Source: step 11 (as Fig. 2 g–i), older noise model of the step 10–11 figures, single-ray engine: optimistic (compare v32 set B: 270–480 ppb at 1 s for the 4 mm cell with the T1.3 detector budget)."));

// ------------------------------------------------------------------ 3 models
C.push(H1("3. Models and formulas"));
C.push(H2("3.1 Single-ray engine (Tasks 1–8; Fig. 2 a–f, i; Fig. 3 B03/B05/S10/A)"));
C.push(P("`engine/quiet_engine.js` = PlanarEngine / GasEngine / DitherLab of the Fig. 2 artifact, with membrane loss and optional R(|sin χ|) tables. 2D in-plane rays, TM₀ slab mode with phase index n_eff, specular DBR facets with constant power reflectance R per bounce (set B), membrane loss exp(−α_m L) with α_m = dB/cm·100/4.343 m⁻¹. Cells: 24-facet circle, 4 mm, launch 7.5° from the wall normal, output port at 0.458 of the perimeter, 150 µm ports; Herriott 10 × 1.5 mm, N = 14, M = 3. Chaotic = 16 index discs r 50 µm, Δn_eff ±0.01, layouts = mulberry32 seeds 1–30."));
C.push(EQ("ΔP/P_m = 2 √ε · V_m / √M,     V_none = 1,   V_m = |time average of the fringe phasor over dither m|"));
C.push(P("M = speckle grains (engine phase-space count). Dithers: sine angle ±2 mrad at 1 kHz, triangle wavelength ±20 pm at 1.297 kHz. Each dither angle is one centre ray: this is what makes the angle dither optimistic (§6)."));
C.push(H2("3.2 Gaussian-beamlet model (Tasks 9–12; Fig. 2 j–k; Fig. 3 G5–G20)"));
C.push(P("`scripts/t9_input_states.js`, `t10_design.js`, `t11_sizes.js`. The input field exp(−(x−X)²/w²)·exp(i k n Θ (x−X)) is expanded on a Gabor lattice of beamlets (σ = 10 µm, oversampling 4, Δx = √π σ, Δφ = √π/(2κσ)); each beamlet is traced as one engine ray to the output port:"));
C.push(EQ("g_j = R^((h−1)/2) · exp(−α L/2) · exp(i k₀ OPL_j),     c_j = √(π/A) · exp(B²/(4A) − C)   (closed-form overlap)"));
C.push(EQ("static fringe phasor = Σ c̄_j ḡ_j,     ḡ_j = g_j · sinc(k₀ OPL_j Δλ/λ)   (triangle wavelength dither)"));
C.push(P("c̄_j is the time average over the input dither (sine angle and sine position at incommensurate rates; samples max(48, ⌈2π A κ w⌉) and max(48, ⌈4π X_a/w⌉) per period). V = |static| / rms(undithered); residual 2√ε V/√M. Number of input states N = |E|²/|⟨E⟩_t|²; the input-dither factor follows N^−½ up to N ≈ 400. M per geometry: M = 4π/λ_M · √det cov(y, u), within-port covariance pooled, calibrated to the engine's grains (c_cal = 1.41 / 1.36 at 1.53 µm, R 97 / 99.5 %; 1.38 / 1.29 at 3.3 µm). Validity: beamlets of one diffraction cell are treated as independent rays, which needs a fully mixing cell (chaotic; FTLE checked with `D.chaosMap`); regular-cell numbers are indicative only. σ = 7 / 10 / 14 µm agree within ±30 %."));
C.push(H2("3.3 Averaging, drift and referencing"));
C.push(EQ("r(T) = √( S_both² H_d(T)² + (S_none² − S_both²) H_ac(T)² )"));
C.push(EQ("H_d(T) = 1/√(1 + 2 (w T/2)²)   (w = fringe-phase drift; 0.1 rad/s → 0.69 at 15 s, 0.43 at 30 s),     H_ac(T) = 1/√(1 + 2 (π·1000·T)²)"));
C.push(EQ("breath referencing every τ:   r → r · 2 sin(w τ/2)   (0.30 at w = 0.1 rad/s, τ = 3 s; 0.06 at 0.02 rad/s)"));
C.push(H2("3.4 Detection"));
C.push(EQ("NEA(T) = √( r(T)² + δ_det(T)² ),     δ_det = √(NEP² + 2hνP_d) / (P_d √(2T)),     P_d = P · T_cell · 10^(−0.8)"));
C.push(EQ("c_min = NEA / (α_peak,1ppm · s_line · Γ⟨L⟩)"));
C.push(P("Older data sets use IL = 14 / 20 / 25 dB instead of T_cell·8 dB, or shot noise only √(q/(P T_cell T)) (published Fig. 3, set A). s_line = line peak kept under the wavelength dither (uniform average): NH₃ 0.896 / 0.804 / 0.518 / 0.434 at ±20 / 35 / 81 / 160 pm; CH₄ 3.31 µm 0.993 / 0.979 / 0.921 / 0.785; C₂H₆ 0.986 / 0.959 / 0.824 / 0.609. Line data: HAPI on cached HITRAN tables (`../HITRAN/hitran_data`), Voigt, air broadening, 35 °C, 1 atm, 5.5 % H₂O, 4.5 % CO₂: NH₃ 6528.77 cm⁻¹ 2.28e-7 /(ppm cm); CH₄ 3016.49 cm⁻¹ 3.79e-5; C₂H₆ Q 2996.85 cm⁻¹ 3.34e-5 (`results/t7/lines.json`). Detection is peak absorbance only: no line-shape fit, no WMS demodulation, no background subtraction (except the Task 12 window fits)."));
C.push(H2("3.5 Gas exchange (Task 12)"));
C.push(P("Flush time V/Q of a channel of height h above the membrane and diffusion time h²/(2D) (D ≈ 0.22 / 0.15 / 0.24 cm²/s for CH₄ / C₂H₆ / NH₃)."));

// ------------------------------------------------------------------ 4 parameter sets
C.push(H1("4. Parameter sets"));
C.push(TBL(["set", "film / λ", "n_eff, Γ", "mirrors, loss", "dithers", "used in"], [
  ["A (published)", "– / 1.6537 µm", "2.0, 0.2", "R 99.5 %, no loss", "±2 mrad, ±20 pm", "Task 1 regression; Fig. 2/3 data set A"],
  ["step 10 (S10)", "600 nm / 1.55 µm", "1.66, ≈0.31", "R(|sin χ|) tables ideal/target/min.", "±2 mrad, ±20 pm", "Fig. 2/3 data set S10"],
  ["step 11", "500–700 nm / 1.55 µm", "1.53–1.74, 0.46–0.22", "designed R(χ), 0.1 dB/cm", "±2 mrad, 5–60 pm", "robustness panels Fig. 2 g–i, Fig. 3 c–d"],
  ["B (v32 realistic)", "500 nm / 1.5317 µm; 1000 nm / 3.3151 µm", "1.5391, 0.4532; 1.4253, 0.5526", "R 95 / 97 / 99.5 %, 0.3 / 0.5 dB/cm", "±2 mrad, ±20 pm", "Tasks 2–8; data sets B03 / B05"],
  ["Task 11 design", "set B films", "as B", "R 97 / 99.5 %, 0.3 dB/cm", "75 µm, ±60 mrad, ±100 µm; ±35 / ±81 pm", "Fig. 2 j–k, Fig. 3 G5–G20, Task 12"],
], [1.3, 1.6, 1.3, 1.6, 1.6, 2.0]));
C.push(P("Common: ε = 1e-6 (placeholder), 30 inclusion layouts (seeds 1–30; Task 8: 100), 24-facet circle, ports 150 µm (4 mm cell). Γ is the E-field fraction Γ_E; the n_g-weighted Γ_abs of Fig. 1d is ≈ 0.39 at 500 nm.", { before: 80 }));

// ------------------------------------------------------------------ 5 campaign
C.push(H1("5. Simulation campaign (Tasks 1–12)"));
C.push(TBL(["task", "question", "model", "main result", "files"], [
  ["1", "reproduce published Fig. 2f/3", "engine", "bit-for-bit; Herriott gain median 55× (not 80×); NH₃ 56 ppb shot-noise", "results/t1, tests/"],
  ["2", "realistic Fig. 2f", "engine", "near-IR C-CMPC 0.8–2.1e-7 (PASS); mid-IR 1.1–1.6e-6 (FAIL); chaotic gain ≈ 2×", "results/t2, figs/fig2f_*"],
  ["3", "ε sweep", "engine", "ΔP/P ∝ √ε exactly; set-B ε thresholds 25–30× tighter", "results/t3"],
  ["4", "path, N_λ for 2 cm", "engine", "N_λ ≈ 1.2–1.5 at ±0.55 GHz; n_g would give 1.5× more", "results/t4"],
  ["5", "angle-dither validity", "engine", "K_eff ≈ 1.9 for ±2 mrad (≪ M ≈ 300)", "results/t5"],
  ["5b", "wave-optics bound", "analytic", "V_wave ≥ 0.98 at ±2 mrad, 40 µm waist: engine angle dither optimistic 6–11× (both dithers)", "results/t5/wave_bound.json"],
  ["6", "inclusion strength", "engine", "Δn ≥ 0.005 needed; small discs (25 µm) best", "results/t6"],
  ["7", "detection limits set B", "engine", "NH₃ 270–480 ppb at 1 s; CH₄/C₂H₆ fringe-limited 7–22 ppb", "results/t7, figs/fig3_setB"],
  ["8", "layout spread (100)", "engine", "fringe spans 3 decades; n_pass ≥ 36 picks the better half", "results/t8"],
  ["9 / 9b", "input states, wide sweeps", "beamlet", "factor N^−½ to N ≈ 400; near-IR best 6e-7 (±35 pm); 3.3 µm best 5e-6", "results/t9, t9b, figs/input_states, dither_raytrace"],
  ["10", "design levers", "beamlet", "cell size is the lever (3–30×); extra/random ports do not help; sparse weak scatterers", "results/t10, figs/t10_cell_size"],
  ["11", "5–20 mm design, facet, drift", "beamlet", "facet 11 best (holdout); limits in §1; N^−½ scaling per size", "results/t11"],
  ["12", "100 ms breath", "beamlet + analytic", "0.1 s CH₄ 1.35, C₂H₆ 1.71 ppb (ref.); line hopping 2990 cm⁻¹", "results/t12, reports/breath_100ms_*"],
], [0.6, 1.6, 0.9, 3.6, 2.0]));
C.push(H2("5.1 Selected numbers"));
C.push(P("Task 9b, 4 mm circle, R 97 %, static both-dither fringe, median (share of layouts < 5e-7):"));
C.push(TBL(["input dither", "N", "factor (N^−½)", "±20 pm", "±35 pm", "±81 pm", "3.3 µm ±81 pm, CH₄ static"], [
  ["40 µm, ±2 mrad", "1.0", "0.98 (0.98)", "2.3e-5", "8.6e-6 (0 %)", "4.7e-6", "4.3e-5, 317 ppb"],
  ["150 µm, ±60 mrad", "18.5", "0.24 (0.23)", "6.6e-6", "2.4e-6 (3 %)", "1.4e-6", "1.8e-5, 132 ppb"],
  ["75 µm, ±60 mrad, ±100 µm", "401.9", "0.06 (0.05)", "1.3e-6", "6.0e-7 (33 %)", "3.4e-7", "5.0e-6, 37 ppb"],
], [2.2, 0.7, 1.1, 0.9, 1.2, 0.9, 1.8]));
C.push(P("Task 12 (10 mm, R 99.5 %, 10 mW), 1σ ppb at 0.1 s:", { before: 120 }));
C.push(TBL(["gas", "raw (w 0.1)", "referenced 3 s", "ref., w 0.02", "ref., 20 mW", "uncooled 1 nW/√Hz", "1 s ×10 breaths"], [
  ["CH₄", "3.5", "1.35", "0.75", "1.1", "49", "0.32"],
  ["C₂H₆", "4.5", "1.71", "0.95", "1.4", "62", "0.41"],
  ["NH₃", "316", "230", "219", "133", "–", "25"],
], [0.9, 1.1, 1.3, 1.1, 1.1, 1.5, 1.3]));
C.push(P("One-laser CH₄ + C₂H₆ (2990 cm⁻¹ window: C₂H₆ 2990.06, CH₄ 2988.93/2989.02 cm⁻¹): line hopping 80/20 gives 5.2 / 1.9 ppb at 0.1 s; a full-window least-squares fit (C₂H₆, CH₄, H₂O, linear baseline) costs 6.6× (C₂H₆) and 14× (CH₄). Gas exchange of a 10 × 10 × 0.3 mm channel: 30 µL, 30 ms at 1 mL/s; diffusion 2–3 ms.", { before: 80 }));
C.push(...IMG("t12_fast.png", 992, 425, "figs/t12_fast.pdf: (a) limits against averaging time (10 mm, R 99.5 %, 10 mW): raw, breath-referenced, referenced at 0.02 rad/s drift; (b) the 2990 cm⁻¹ window at breath conditions (absorption coefficients, ±81 pm dither)."));

// ------------------------------------------------------------------ 6 realism
C.push(H1("6. How realistic are the calculations?"));
C.push(P("Direction: “optimistic” means real limits are likely worse. Impact: rough factor on the limits where it can be estimated."));
C.push(TBL(["ingredient", "status / evidence", "direction", "impact on limits", "affected outputs"], [
  ["Angle dither as single ray per angle (engine)", "Task 5b: wave bound V ≥ 0.98 at ±2 mrad, 40 µm; beamlet model confirms", "optimistic", "6–170× on angle / both fringe", "Fig. 2 f (angle, both), Fig. 2 g–i, Fig. 3 B03/B05/S10/A and c–d, Tasks 2–8"],
  ["Beamlet independence (one ray per beamlet)", "needs full mixing; FTLE 0.15–0.6 per hit, ≥ 92 % chaotic rays; σ-convergence ±30 %", "uncertain", "±30 %; regular cells not valid", "Tasks 9–12 (G sets, Fig. 2 j–k)"],
  ["2D in-plane model", "no out-of-plane scattering at inclusion steps, trenches, DBR; no vertical radiation; no polarisation coupling", "optimistic", "loss and extra stray light unknown; ε may rise", "all"],
  ["Stray fraction ε = 1e-6", "placeholder of the proposal; fringe ∝ √ε exactly", "unknown", "×√(ε/1e-6)", "all fringe numbers"],
  ["Stray path model", "stray copy follows the wavelength, not the input state; OPD = exit path", "possibly optimistic", "if stray light passes part of the cell, the λ dither is weaker", "all"],
  ["Fringe drift 0.1 rad/s", "assumed (engine default); helps at T ≳ 10 s", "unknown", "fringe part ×1.45 (15 s) to ×2.3 (30 s) higher with no drift", "Fig. 3 (b) curves, Task 11/12"],
  ["Constant R per bounce (set B)", "the designed trench DBR reflects totally above ≈ 40°; ignored", "pessimistic", "higher effective R", "set B, Tasks 2–12"],
  ["Phase index n_eff for λ → phase", "group index n_g = 2.28 / 2.15 would decorrelate 1.5× faster", "pessimistic", "λ-dither fringe somewhat lower", "all"],
  ["Membrane loss 0.3 dB/cm in 10–20 mm films", "assumed; large suspended films untested", "possibly optimistic", "Γ⟨L⟩ and T fall", "Tasks 10–12"],
  ["Coupler loss 8 dB (two grating couplers)", "assumed", "unknown", "detector term ∝ loss (NH₃ 15–20 mm is detector-limited)", "Tasks 11–12"],
  ["NEP 14 pW/√Hz in the mid-IR", "plausible for TE-cooled InAsSb/HgCdTe; uncooled 1 nW/√Hz gives 5–70× worse", "optimistic for uncooled", "CH₄ 30 s 1.6 → ≈ 6 ppb (10 mm)", "3.3 µm limits"],
  ["Peak-only detection", "no line fit / WMS; Task 12 adds fits only for the 2990 cm⁻¹ window", "mixed", "fits gain from line shape; WMS adds demodulation constraints", "all limits"],
  ["Input-dither amplitude modulation", "transmission changes with input state not modelled", "optimistic", "adds AM at dither frequency", "Tasks 9–12"],
  ["Line hopping / referencing", "white-noise assumption; fringe correlated over ≈ 0.05–0.1 cm⁻¹; breaths independent", "optimistic", "unknown until a measured zero-air spectrum", "Task 12, Fig. 3 ratio readout"],
  ["Facet / inclusion choice", "holdout-validated (layouts 1–15 vs 16–30)", "robust", "–", "Task 11"],
  ["Regression", "bit-for-bit rerun of published ensembles (5 tests pass)", "validated", "–", "Task 1"],
], [2.0, 2.6, 1.0, 1.7, 2.3]));
C.push(P("Overall: within the 2D beamlet model the Task 11/12 numbers are internally consistent (model limitations flagged, no tuning after seeing results, failing cases kept). The likely real-world shortfall comes from ε, out-of-plane scattering and coupler/membrane losses; a fair expectation is that first devices land 3–10× above the Task 11 limits until ε and the drift are measured and controlled.", { before: 100 }));

// ------------------------------------------------------------------ 7 improvements
C.push(H1("7. How to improve: models and experiments"));
C.push(H2("7.1 Model improvements (in order of value)"));
[ "**Validate the beamlet model with a full-wave or beam-propagation calculation** on a small chaotic cell (2–4 mm, a few inclusions): compare V(N) and M against Tasks 9–11. Other branches of the repository have a full-wave CMPC study (`claude/cmpc-ray-propagation-engine-*`, E24) and a Tidy3D setup export (`ccr-a94be260-*`) to build on.",
  "**Model the stray light explicitly:** where ε comes from (facet back-reflection, coupler second order, scattering at inclusion steps), its OPD relative to the exit paths and its dependence on the input state; replace the single ε = 1e-6 by a distribution.",
  "**3D / out-of-plane loss at inclusion steps and trench DBRs** (FDTD or eigenmode expansion): gives a per-hit scattering loss and a stray contribution; feed both into the ray model as R and ε per event.",
  "**Angle-dependent DBR reflectance including total internal reflection** for set B (the step-1 R(χ) tables exist for 600 nm films; regenerate for 500 / 1000 nm).",
  "**Group index for the wavelength → phase conversion** (n_g instead of n_eff in the beamlet phases).",
  "**Detection chain:** line-shape fitting and WMS demodulation in the noise model; correlated (spectral) fringe instead of white per-point noise; amplitude modulation from the input dither; the 8 dB coupler loss from a coupler simulation.",
  "**Intracavity modulation** (local heaters over a few inclusions): the only untested route to N ≫ 400 independent realisations.",
  "**Layout screening:** use n_pass (Task 8: Spearman −0.47 with the fringe) or a direct beamlet run to pick fabricated layouts.",
].forEach(s => C.push(N(s)));
C.push(H2("7.2 First measurements that would calibrate the model"));
[ "Zero-air fringe spectrum and its time drift on a mounted, heated chip (sets ε, the drift rate w and the fringe correlation length).",
  "Cell transmission and Γ⟨L⟩ (cut-back or ring-down) against R and membrane loss for 10 mm cells.",
  "Grating-coupler efficiency and its dependence on the input angle/position dither (AM).",
  "Scattering loss and back-reflection of single inclusion steps (test structures).",
  "Fringe against the number of input states N with a MEMS scanner (checks the N^−½ law of Task 9b).",
].forEach(s => C.push(B(s)));

// ------------------------------------------------------------------ 8 data map
C.push(H1("8. Data and code map"));
C.push(TBL(["path (QUIET_sims_v32/ unless noted)", "content", "key structure"], [
  ["engine/quiet_engine.js", "single-ray engine (PlanarEngine, GasEngine, DitherLab)", "–"],
  ["qv/ (analysis.py, gas.py, slabmodes, style.py)", "analysis helpers, HAPI wrapper, TM₀ solver, figure style (Liberation Sans 10 pt, 16.8 cm)", "gas.alpha(table, x, ν_min, ν_max)"],
  ["configs/setA.json, setB.json", "parameter sets", "–"],
  ["results/t1 … t8/", "engine ensembles and summaries (Tasks 1–8)", "per case JSON; table.md"],
  ["results/t9, t9b/", "beamlet input-state sweeps", "table.md; summary.json"],
  ["results/t10/", "design levers", "variants[name].chaotic[i]: fringe[dl][mode], T, GL_cm, M"],
  ["results/t11/final_{nir,mir}_R{0.97,0.995}.json", "Task 11 runs, 5 cells × 30 layouts × 17 dithers", "cells[c].chaotic[i].d[design]: T, GL_cm, M, fringe[20pm|35pm|81pm][none|input|spectral|both], paths"],
  ["results/t11/summary.json", "limits per band/R/cell", "\"nir_R0.97_d10\": T, GL_cm, M, static, limits[\"NH3_15s_NEP14pW\"] = [Q1, med, Q3] (T = 0.1, 1, 15, 30, 60 s)"],
  ["results/t11/artifact_data.json", "data embedded in Fig. 2 j–k and Fig. 3 G sets", "cells[...].cur/reg: env (vs T grid), floor, lay.fr[T] per layout"],
  ["results/t12/fast.json", "100 ms limits, windows, hopping, exchange, NO", "limits[\"mir|R0.995|d10|CH4|NEP14pW|P10mW|w0.1|ref|T0.1\"] = {total: [Q1, med, Q3], fringe, det}"],
  ["results/t7/lines.json", "HITRAN line peaks at breath conditions", "alpha_peak_per_ppm_cm, nu0_cm, bg_at_peak_per_cm"],
  ["../quiet_sims/results/step10, step11_*", "older 600 nm / R(χ) data (S10, robustness panels)", "step11_artifact_data.json"],
  ["../HITRAN/hitran_data/", "cached line lists (CH₄, C₂H₆, NH₃, NO, H₂O, CO₂)", "HAPI tables"],
  ["artifacts/fig{1,2,3}_src_live.html → fig{1,2,3}_v32.html", "artifact sources and built pages", "build_fig1_gc.py, build_fig23_v32.py"],
  ["artifacts/fig3_external_edits.patch", "edits made on the published Fig. 3 by another session, re-applied on build", "unified diff"],
  ["REPORT.md, reports/breath_100ms_technical_report.md", "full per-task report; 100 ms technical report", "–"],
], [3.0, 3.3, 3.4]));
C.push(H2("8.1 Reproduction"));
["cd QUIET_sims_v32", "python run_all.py              (all ensembles, ≈ 2 h on 4 cores)", "python run_all.py --analysis   (analyses, figures, tests from stored results)",
 "sh scripts/run_t11.sh          (Task 11 runs)", "python scripts/t11_analysis.py && python scripts/t12_fast.py",
 "python scripts/build_fig23_v32.py && python scripts/build_fig1_gc.py", "python -m pytest -q tests"].forEach(s => C.push(CODE(s)));
C.push(P("Rules kept throughout (from the project brief): proposal .tex untouched; no tuning after seeing results; failing cases kept; bugs documented in REPORT.md (Task 9b sampling, Task 10 multi-input normalisation and port placement, t11 smear lookup, Fig. 3 mode naming).", { before: 120 }));

const doc = new Document({
  creator: "Claude", title: "QUIET CMPC simulation summary", description: "Summary of the simulations behind the Fig. 1-3 artifacts",
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "1F3F93" }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "1F3F93" }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 21, bold: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } }] },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } },
                                 { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 260 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("QUIET simulation summary · page ", { size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
    children: C }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("written", OUT, b.length); });
