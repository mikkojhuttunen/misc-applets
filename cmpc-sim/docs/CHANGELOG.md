# Changelog
* **v0.5 (packaging)** — moved into `misc-applets/cmpc-sim/` (engines found in `../math-engines`), package layout, `scripts/` pipelines writing to `results/`, `design_mirror.py`, unit tests, docs, Makefile.
* **v0.4** — hole-free three-index TE mirror (`mirror_design`); speckle vs chaos and K beams; suppression budget; `select_lines`, HAPI partition sums; Lyapunov exponent; `extra_ports`, `launch_w`, `launch_off`, `s_out` in the tracer; single-port multi-beam scans; proposal figures (10 pt, 17 cm).
* **v0.3** — mid-IR (Ge/Si 5.26 µm): Ge Sellmeier, NO/CO/N₂O line placeholders, `run_midir.py`; phase-dither (sweep) kernel; `fetch_hitran.py nir|mir`.
* **v0.2** — corrected polarisation mapping (TE→p, TM→s); TM end to end; `SegmentedCell` (tilt/offset/curvature), signed sinχ, itineraries; gas spectra; path-resolved coherence model.
* **v0.1** — Si/SiNx/Al₂O₃ membrane modes, s-pol DBR, stadium tracer (superseded; contained the polarisation error).
