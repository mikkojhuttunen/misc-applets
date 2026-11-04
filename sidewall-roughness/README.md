# sidewall-roughness

How much propagation loss etched-sidewall roughness adds in thin-film waveguides (TFLN, TFLT, SiNx, Al₂O₃, sapphire; 400–1000 nm and 1550 nm), and how it scales with platform, wavelength and the modal field at the sidewall. The comparison against slab modes (no etched sidewall) motivates slab-mode architectures for integrated liquid and nutrient sensing. Results are meant to be qualitative to semi-quantitative.

Local work: geometry, material models, mode solving, simulation building and post-processing. 3D FDTD runs on the Tidy3D cloud. **Nothing in this package starts a cloud run without an estimate printed first and the task name typed back** (`swr/runners/cloud.py`).

```
pip install -e ".[tidy3d,plot,test]"
python -m pytest -q                         # fast tests (~10 s)
SWR_SLOW=1 python -m pytest -q              # + local Tidy3D mode-solver checks (~2 min)
python scripts/m1_smooth_modes.py           # smooth-guide mode tables -> results/m1/
python scripts/m1_probe_coarse.py           # one coarse corrugation run, built and checked locally
python scripts/m1_probe_coarse.py --estimate   # + upload, print web.estimate_cost(), delete draft
python scripts/estimate_sweep_cost.py [--production] [--modes] [--cloud --limit N]
python scripts/export_tidy3d_setups.py --out setups/m1 --platforms A_rib_LN --modes TE0 --wavelengths 1.55 --periods 3
python scripts/analyse_probe_results.py setups/m1      # after the *_data.hdf5 files come back
```

**Runs by people.** `export_tidy3d_setups.py` writes one `td.Simulation` `.hdf5` per run, a `manifest.csv` (cheapest group first, smooth reference first in each group), the merged config with the git commit, and `RUN_ME.md` (estimate, run, download, naming). Whoever runs them returns the `*_data.hdf5` files; `analyse_probe_results.py` pairs each corrugated run with its smooth reference and writes alpha and H(K) per frequency to `kernel.csv`.

## Layout

| Path | Content |
|---|---|
| `config/default.yaml` | every physical and numerical parameter (versioned); `config/milestone1.yaml` coarse overrides |
| `data/materials/` | tabulated indices with their sources (LT, PEALD Al₂O₃) |
| `src/swr/config.py` | YAML loader; overrides may only change existing keys |
| `src/swr/roughness.py` | FFT roughness fields h(x, z), PSD/ACF conventions, probe sinusoid |
| `src/swr/materials.py` | lossless dispersive material models, Sellmeier fits, provenance table |
| `src/swr/orientation.py` | the one crystal-orientation helper (cut, propagation angle) → sim-frame tensor and medium |
| `src/swr/slab.py` | exact 1D anisotropic slab modes (TE/TM) |
| `src/swr/geometry/` | rib, strip-loaded and slab guides on a membrane or substrate; sidewall offsets h(x) |
| `src/swr/modes.py` | mode bookkeeping with the local Tidy3D mode solver |
| `src/swr/sim_builders.py` | FDTD probe and smooth-reference simulations, local size proxy |
| `src/swr/analysis/probe.py` | radiation K band, H(K) from a probe run, α = ∫ S·H dK |
| `src/swr/analysis/budget.py` | power budget of a run: same mode / other modes / reflection / radiation |
| `src/swr/runners/cloud.py` | estimate (always printed) and confirmed run |
| `scripts/` | milestone drivers, the sweep cost estimate, setup-file export for Tidy3D users, analysis of returned runs |

## Conventions

* **Frame.** x = propagation, y = lateral in the film plane, z = film normal; the film bottom is z = 0. Lengths in µm (config keys ending `_nm` are in nm).
* **Sidewall angle** is measured from the film plane (90° = vertical) in the config. Tidy3D's `PolySlab.sidewall_angle` is measured from the vertical, so the code passes 90° − angle. The given width is the top width.
* **Wall offsets** h(x) are top-view, positive outwards, applied over the whole wall height (columnar). z-dependent defects (striations, angle variation, footing, stitching) will be stacked PolySlabs.
* **Roughness statistics.** R(u) = σ² exp(−u²/Lc²) (Gaussian) or σ² exp(−|u|/Lc) (exponential), so R(Lc) = σ²/e for both. The two-sided PSD S(K) = (1/2π) ∫ R(u) e^{−iKu} du satisfies ∫ S dK = σ². Gaussian: S = σ² Lc/(2√π) · exp(−K²Lc²/4). Exponential: S = σ² Lc/[π(1 + K²Lc²)]. A 2D field has the separable ACF R(u, v) = σ² r_x(u/Lc) r_z(v/Lz); Lz = null is columnar. Payne–Lacey-type papers often use the one-sided PSD or Lc defined at a different level, so check the factor of 2 and the Lc definition when comparing with older work.
* **Crystal orientation.** `cut` = crystal axis along z. The propagation direction is cos θ · ref + sin θ · (cut × ref), with ref from `orientation_defaults` (z-cut: ref = X; x-cut: ref = Y, so θ = 0 is along Y and θ = 90° along Z). Uniaxial optic axis = crystal c.

## Method

1. **Roughness generator** (`roughness.py`). Spectral filtering of white noise on a periodic grid. The discrete spectrum is renormalised so the expected variance is exactly σ². Tested: ensemble σ, ACF shape, fitted Lc and Lz, periodogram against the analytic PSD, columnar limit, seeds.
2. **Spectral probe** (main tool). A straight guide carries a sinusoidal corrugation A sin(Kx) on one wall over N periods (smooth input and output sections). The corrugation's spectrum is two lines at ±K with weight A²/4 (σ² = A²/2). To first order the loss coefficient is linear in the PSD, α = ∫ S(K) H(K) dK with H even. So one run gives H(K) = 2α_probe/A², and any PSD follows without new simulations: α_one wall = 2∫₀^∞ S H dK. Two walls with uncorrelated roughness add (×2 for a mirror-symmetric guide). Radiation exists only for K in (k₀(n_eff − n_rad), k₀(n_eff + n_rad)), where n_rad is the highest index the light can radiate into. Conversion into other guided modes happens at discrete K = k₀|n_eff − n_m| (forward) and k₀(n_eff + n_m) (backward). Those lines are bookkept as mode conversion, never as radiation. Linearity is checked with `probe.linearity_amplitudes_nm` (α/A² constant).
3. **Power budget** per frequency (`analysis/budget.py`), normalised to the excited mode's input power:
   * same mode forward;
   * each other guided mode forward;
   * reflection (same and other modes);
   * radiation through the top and bottom planes (free space);
   * radiation through the side planes (free space plus scattering into slab modes of a rib's remaining film);
   * radiation by deficit (1 − all guided power, which includes what leaves through the box ends).

   A smooth run of the same layout is subtracted. The source is broadband (`fdtd.source_fwidth_fraction`), so one run gives H at several wavelengths around its centre.
4. **Perturbation-theory cross-check** (next milestone). Payne–Lacey / Marcuse volume-current scaling α ∝ σ² |E_sw|² (Δε)² / λ², with |E_sw|² from the mode solver (`sidewall_e2_per_length_um`) and Δε from `Guide.eps_interface`.
5. **Brute-force validation** (later; 1–2 cases). Random realizations of several lengths; α from the slope of ln T vs L, normalised to smooth guides of the same length and averaged over seeds.
6. **Slab reference** (platform E). Same stack, no sidewall; top and bottom surface roughness only.

## Mode bookkeeping (`modes.py`)

At each wavelength and geometry the local mode solver lists the modes with:

* n_eff;
* k_eff → leakage in dB/cm (all media are lossless, so k_eff is pure leakage);
* TE fraction and a TE/TM label;
* sidewall power fraction and |E|² fraction within ±`sidewall_band_um` of each wall;
* flags: `near-cutoff` (n_eff − n_clad,max < `near_cutoff_dneff`) and `lateral-leak` (n_eff below a slab mode of the region beside the guide, from the exact 1D solver).

Spurious PML modes (more than `max_pml_energy_fraction` of |E|² inside the PML cells) are dropped. The cross-section grows until every guided mode has `clearance_decay_lengths` field decay lengths to the boundary. The probe sweep skips flagged modes by default.

**Smooth-leakage check.** Leakage of the smooth guide (k_eff, local) comes before any roughness run, so leakage is never read as scattering.

Results so far (coarse 20 nm grid, z-cut LN membrane in air):

* **Shallow rib** (500 nm film, 50 % etch, 1.2 µm top width, 70° walls, 1550 nm): TM0 leaks laterally at about 240 dB/cm into the TE slab mode of the remaining film.
* **SiNx strip-loaded** (300 nm LN, 1.5 × 0.3 µm SiNx strip): TM0 leaks at about 70 dB/cm.
* **TE0** is leakage-free in both.

`results/m1/modes.txt` has the tables.

## Materials (all lossless in the loss runs; absorption is a separate budget item)

| Key | Model | Source | Valid range |
|---|---|---|---|
| `LN` | tidy3d `LiNbO3/Zelmon1997`, uniaxial | Zelmon et al., JOSA B 14, 3319 (1997), congruent LN | 0.4–5 µm |
| `LT` | 2-term lossless Sellmeier fit, o and e | W. L. Bond, J. Appl. Phys. 36, 1674 (1965), tabulated n (via refractiveindex.info, CC0) | data 0.45–4 µm, fit 0.45–2 µm |
| `SiNx` | constant n = 2.0 | PECVD, n = 1.95–2.2 with Si ratio | – |
| `Al2O3` | 2-term lossless Sellmeier fit | PEALD Al₂O₃, Rodriguez de Marcos et al. (database lists doi:10.1364/OME.562660; journal and year in the entry are inconsistent, to verify) | 0.38–1.69 µm |
| `sapphire` | uniaxial Sellmeier | Malitson & Dodge, JOSA 62, 1405 (1972); Dodge, CRC Handbook of Laser Science IV (1986) | 0.2–5 µm |
| `SiO2` | tidy3d `FusedSilica/ZemaxPMLStable` | Malitson 1965 (doi:10.1364/JOSA.55.001205) | 0.405–2 µm |
| `water` | tidy3d `H2O/Horiba` | no reference in the library | **0.21–0.83 µm** |
| `Si` | 2-term lossless fit to Re n of tidy3d `cSi/Green2008` | Green 2008 | max \|Δn\| 0.15 at 405 nm |

Library names were checked in tidy3d 2.12.2. Not used, and why:

* `SiO2/Palik_NoLoss`: starts at 0.55 µm.
* `Si3N4/Luke2015*`: LPCVD and stops at 0.414 µm; PECVD SiNx differs.
* `Al2O3/Horiba` and `Sapphire/Horiba`: end at 0.83 µm and have no reference; Sapphire/Horiba is also isotropic.
* `SiN/Horiba`: lossy.

There is no LT entry. `swr.materials.provenance(cfg)` prints the table above from the code.

## Numerics

* **Local mode solver, no subpixel averaging.** Tidy3D's local mode solver needs `tidy3d-extras` (licensed) for subpixel averaging. Without it an interface on a grid line makes the film one cell thicker for tangential E. The grid is therefore built with every horizontal interface at a cell centre (`interface_centered_coords`):
  * TE slab n_eff error is 1.4·10⁻⁴ at 10 nm.
  * TM (normal E across the interface) keeps an O(dl) error: +3.4·10⁻³ at 20 nm.
  * Slanted walls are staircased.

  For production mode data, set the grid to 10 nm or less, or install `tidy3d-extras` (on a machine with the API key) to enable local subpixel.
* **FDTD.** Auto grid (`min_steps_per_wvl`) plus a `MeshOverrideStructure` box around each sidewall (`sidewall_dl_um` = 5 nm production, 20 nm debug). Convergence test: halve it on one realization. Subpixel averaging stays on. A 2–5 nm corrugation on a 5 nm mesh enters only through the averaged permittivity of the boundary cells. Subpixel averaging is first-order correct for the dipole strength of a thin perturbation, but it does not resolve the near field of features smaller than a cell, so absolute α at σ ≤ 2 nm must come from the σ² linearity check, not from a single run.
* **PML.** On all sides. The probe domain has `pml_clearance_decay_lengths` decay lengths of the guided tail before the PML. For the membrane stacks there is no substrate, so the only leakage is lateral (into slab modes) and is caught by the mode report.
* **Substrate S2 (bulk SiO₂).** The SiO₂ is semi-infinite (into the PML). The clearance from the guide to the bottom PML uses the same decay-length rule, taken from the mode's n_eff in SiO₂. S1 (BOX on Si) leakage shows as k_eff of the smooth mode.
* **Off-axis crystal cuts** (e.g. x-cut at 45°) need `td.FullyAnisotropicMedium`, which Tidy3D supports only without dispersion. Those runs use the tensor at the centre wavelength, so keep their bandwidth small. z-cut and x-cut along Y or Z stay fully dispersive (`AnisotropicMedium`).
* **Local Tidy3D runs.** Only the mode solver runs locally. Any FDTD run, 2D included, is a cloud job (cheap in 2D), so "local debugging" here means building the simulation, `validate_pre_upload`, plots and the cell/time-step proxy.

## Approximations that can change conclusions qualitatively

1. **First-order (Born) scattering in the probe method.** This ignores multiple scattering and depletion; it is fine while α·L ≪ 1 and kσ ≪ 1. It also assumes the two walls are uncorrelated; correlated walls (one lithography mask) can add or cancel coherently.
2. **Lateral leakage of TM-like modes** in shallow ribs and strip-loaded z-cut guides is often far larger than roughness loss. It is reported, not mixed in, but it decides which polarization is usable at all.
3. **Multimode guides.** Above roughly 0.6 µm film at VIS, scattering into other guided modes is a separate channel. Counting it as loss or not depends on the application (a single-mode detector after the guide: yes).
4. **LT data.** Bond's 1965 points scatter by about 8·10⁻⁴, and LT birefringence is only about 0.004. TE/TM differences in LT are therefore uncertain at that level, and below 450 nm the index is extrapolated.
5. **Constant SiNx index.** With n = 2.0 below LN's index, the strip-loaded mode sits mostly in the LN film. A Si-rich SiNx (n → 2.2) moves the field into the strip and up the roughness loss.
6. **Lossless Si (S1 only).** Below 1.1 µm, real Si absorbs and does not reflect leaked light back up. The lossless model is only a leakage indicator there.

## Open parameters (placeholders marked TENTATIVE in the config)

* **Rib (A, C):** top width and etch-depth sweep. Membrane thicknesses are 300/400/500/1000 nm, and the defaults use 500 nm because 600 nm is not in that list.
* **Strip-loaded (B):** SiNx strip width, height and sidewall angle.
* **D platforms (SiNx, Al₂O₃, sapphire membranes):** guide type (rib or full-etch strip), widths and angles. Sapphire defaults to c-plane (optic axis normal).
* **VIS wavelength points.**
* **LT type:** Bond's congruent LT for now. Use Mg:LT or stoichiometric LT if that is the film.

## Cost and sweep order

`scripts/estimate_sweep_cost.py` builds every run of `config: sweep`, sorts them by cells × time steps, and with `--cloud` prices them one by one with `web.estimate_cost()` (drafts deleted, nothing started). Suggested order:

1. smooth references and 1550 nm (largest cells);
2. one platform's full H(K) at one wavelength;
3. the σ² linearity check;
4. the remaining platforms and wavelengths;
5. VIS last (finest grid).

The first full count (coarse mesh, slab-estimated domains) is 1120 runs. Ways to reduce it:

* fewer periods per band;
* in-phase and antiphase two-wall runs with a y-symmetry plane (each half the size);
* wider-band sources covering several VIS points per run.

## Future work (not implemented)

Slab-mode Herriott-type multipass cells. The propagation direction changes from pass to pass, so `orientation.crystal_to_sim` is called once per direction. The dominant losses are top/bottom surface roughness (slab reference) and reflector scattering; a per-pass loss budget then gives the effective path length against the pass count.

## Status

Milestone 1 is in place:

* roughness generator and tests;
* config, materials and orientation;
* slab solver;
* smooth mode bookkeeping;
* probe simulation builder and budget;
* cost script.

No cloud job has been run.
