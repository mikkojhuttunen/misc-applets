# algaas-shg

Design tool for **layer-poled AlGaAs-on-insulator rib waveguides**: flat core with χ⁽²⁾ > 0, rib with χ⁽²⁾ < 0,
TE pump fundamental at 1500–1600 nm, higher-order vertical TM second harmonic at 750–800 nm (modal phase matching).
Python, matplotlib widgets; solver is `math-engines/engines/algaas_rib`.

```
pip install numpy scipy matplotlib
python algaas-shg/layerpoled_shg_applet.py                    # interactive
python algaas-shg/layerpoled_shg_applet.py --png out.png --sweep width 1000 1800
```

Default design (x = 0.30 core and rib, 150 nm core, 200 nm rib, 680 nm wide, SiO₂ below, air above, vertical walls):
TE(0,0) at 1550 nm → TM(0,1) at 775 nm, n_eff ≈ 2.696 for both, Γ ≈ 1.49 µm⁻¹ (95 % of the best case; uniform-sign
χ⁽²⁾: 0.13 µm⁻¹), A_eff ≈ 0.45 µm². Δn changes by 3e-4 per nm of width: about ±3 nm tolerance for 1e-3.
Thicker alternative: 200/500/1450 nm with TM(0,2), Γ ≈ 0.49 µm⁻¹ (`--h-core 200 --h-rib 500 --width 1450 --sh-order 2`).

Panels: n_eff(λ) of TE (solid) and TM (dashed) modes, 750–1600 nm; n_p(λ_p), n_SH(λ_p/2) and Δk(λ_p) with the
phase-matching wavelength and sinc² bandwidth; pump, SH and integrand s·Ex²·Ey maps; Δk, L_coh, Γ, A_eff, η_norm.
Buttons: Dispersion (≈20 s), Tune (solves Δn = 0 for width, rib or core height), Sweep window.

Read before trusting numbers: the AlGaAs index model is accurate to ~1e-2 (use `--dn-algaas` to calibrate); FD
discretisation leaves ~5e-4 in Δn at the default 30 nm step (use 10–15 nm to confirm); semi-vectorial, no loss,
MQW treated as its average-index alloy, d_eff is an input.

## Engineered rib index (`dn_rib`, `dn_rib_sh`)

The χ⁽²⁾ < 0 rib can carry an extra bulk index relative to the core (MQW / composition engineering), independent of the
Al fraction and so of the 775 nm absorption limit. `dn_rib` is the offset at 1550 nm, `dn_rib_sh` at 775 nm (linear in 1/λ
between; default equal). Applet: sliders "Δn rib" and "Δn rib extra at SH", `--dn-rib`, `--dn-rib-sh`; the tuner and the
sweep window accept `dn_rib`.

Grid results (x = 0.30, TM(0,1) at 775 nm, 40 nm step, linear interpolation along width; coarse):
offset ±0.1 does not raise the best Γ (about 1.5 µm⁻¹ in the thin design at all three offsets) but multiplies the
phase-matched geometries (4 at 0, 8 at −0.1, 6 at +0.1). Refined (20 nm step), dn_rib = −0.1: 200 nm core / 200 nm rib / 1630 nm
wide gives n = 2.846, Γ = 1.11 µm⁻¹ (99 % of the best case), A_eff = 0.81 µm², η_norm ≈ 660 /(W cm²), a rib 2.4× wider than the
zero-offset thin design; its nearest other TM mode is a lateral order 0.008 away in index, so expect mixing.

## Exporting the dispersion figure and its data

`Export figure + data` in the applet, or from the command line:

```
python algaas-shg/layerpoled_shg_applet.py --export mydesign                      # defaults: 750-1600 nm, png+pdf+svg
python algaas-shg/layerpoled_shg_applet.py --export mydesign --h-core 150 --h-rib 200 --width 680 --sh-order 1 \
       --extra TM:0,0 --extra "TE:1,0@1000" --figsize 5.4,3.4 --fontsize 12 --ylim 1.5,3.4 --formats pdf,svg
```

Figure: TE(0,0) pump (red, solid, marker at λ_p) and TM(0,m) SH (blue, dotted, marker at λ_p/2) over the whole range, dashed
lines at both marker indices, Δn label (and a bracket when Δn is large), field insets with dashed core/rib outlines, dotted
connectors and a scale bar (200 nm for narrow ribs, 500 nm for wide; also written to `_meta.json`). `--extra POL:MX,MY[@nm]`
adds more mode curves; with `@nm` that curve also gets a marker and inset. Curves stop where a mode is cut off.
PDF and SVG keep text editable (Type 42 / real text); the field maps are rasterised inside the vector files.

Files written: `<prefix>.png/.pdf/.svg`; `<prefix>_dispersion.csv` (wavelength_nm, `neff_TE00`, `neff_TM01`, ..., `ng_*`);
`<prefix>_fields.npz` (x, y in µm, signed field map, χ² sign map and ε map per inset); `<prefix>_meta.json` (geometry,
index offsets, marker indices, Δn, Δk, Γ, solver step). Export takes about a minute at 36 wavelengths and a 30 nm step.

## Second-harmonic absorption (material parameters)

Source: the project note `AlGaAs_SHG_absorption.docx` (plain text). The SH photon is 1.60 eV at 775 nm and 1.65 eV at the
750 nm end of the range; the note asks for every absorbing edge to sit at least 1.65-1.7 eV, and warns that GaAs-like MQW wells
absorb near 800-850 nm.

* Recommended default: `x_core = x_rib = 0.35` (Eg = 1.86 eV by the note's formula 1.424 + 1.247 x): gap margin 0.26 eV at 775 nm
  and 0.21 eV at 750 nm, intrinsic tail loss below 1e-4 dB/cm (assumed Urbach energy 10 meV). Stay at x <= 0.45 (indirect gap above).
* `engine.bandgap`, `absorption_report`, `urbach_alpha`, `min_al_fraction`: gap models (note's linear one by default), margins, tail loss.
  The index model's own gap is 1.424 + 1.266 x + 0.266 x²; the two differ by 27 meV at x = 0.3 and the code warns against the lower.
* `engine.MQW`, `mqw_edge`, `mqw_max_well`, `mqw_indices`: e1-hh1 edge of a finite well (envelope approximation, Qc = 0.65,
  m_e = 0.067 + 0.083 x, m_hh = 0.34 + 0.42 x, 8 meV exciton binding; all approximate and overridable), maximum well width for a target edge,
  and the form-birefringent indices (TE pump sees n_par, TM SH sees n_perp).
* `RibStack(core_mqw=..., rib_mqw=...)` or `--core-mqw LW,LB,XW,XB [--rib-mqw ...]`: use MQW layers; the applet and `shg_design` then report the
  SH margin from the e1-hh1 edge.

Maximum well width (nm) for an edge of at least 1.75 eV (775 nm + 0.15 eV), 10 nm barriers:
GaAs wells 1.4 (x_b 0.40) / 1.6 (0.45); Al0.1 wells 2.1 / 2.4; Al0.2 wells 4.5 / 4.8; Al0.3 wells: not limited below 30 nm.
