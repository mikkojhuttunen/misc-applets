# trace_gas

Voigt absorption spectra of trace gases (CH₄, NH₃, CO₂, H₂O near 1.5–1.7 µm), line-strength temperature scaling, air broadening, transmission through a cell with a distribution of path lengths, and the shot-noise-limited absorbance. Needs scipy (Faddeeva function).

Line data: the built-in `ILLUSTRATIVE_LINES` are hand-entered approximations (±30 % on strengths), not HITRAN, and every Result says so in its assumptions. For real data run `python tools/fetch_hitran.py` (needs `pip install hitran-api` and access to hitran.org); `load_lines` then reads the json passed to it, named by `$TRACE_GAS_LINES`, or saved as `engines/trace_gas/hitran_lines.json`.

Units: the spectral helpers use HITRAN units (cm⁻¹, cm/molecule, 1/cm); the Result functions `line_peak` and `shot_noise` are SI.

Version 1.
