# misc-applets

Self-contained interactive applets, one `.html` file each, no build step.

| Applet | Description |
|---|---|
| [parametric-amplifier.html](parametric-amplifier.html) | Fiber parametric amplifier: coupled-wave RK4 solver for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) OPAs with a signal near 1550 nm. LP01 fiber dispersion, gain bandwidth and idler spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and bandwidth optimisation over loss, fiber geometry, length and pump power. |
| [dbr-structures/](dbr-structures/index.html) | Waveguide DBR and CRIGF calculator for thin-film LiNbO₃/LiTaO₃: Bragg-grating index contrast and reflection spectra, grating-coupler cavities pumped from free space, SHG with periodic poling and QPM bandwidth optimisation, poled straight sections, ridge/strip-loaded and curved-DBR lateral geometries, BOX/mirror directionality, pump depletion and photorefraction estimates. |

## Keep design documentation out of this repo

This repository holds tool code only. Do not commit design reports, analysis write-ups, parameter-study results, device recommendations, measurement data or fabrication details: they can contain sensitive information. Keep them in separate, access-controlled documents and link to nothing sensitive from here. This applies to people and to AI assistants working in the repo.
