# misc-applets

Self-contained interactive applets, one `.html` file each, no build step, on top of a shared set of tested calculation engines in `math-engines/`.

| Applet | Description |
|---|---|
| [parametric-amplifier.html](parametric-amplifier.html) | Fiber parametric amplifier: coupled-wave RK4 solver for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) OPAs with a signal near 1550 nm. LP01 fiber dispersion, gain bandwidth and idler spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and bandwidth optimisation over loss, fiber geometry, length and pump power. |

## Shared engines

Plain, dependency-free JS modules (not standalone applets) meant to be
reused across multiple applets instead of re-derived each time.

| Folder | Contents |
|---|---|
| [beam-propagation/](beam-propagation/) | Polygon-billiard ray tracing and periodic-orbit solving for circular segmented multipass cells, plus an ABCD-matrix Gaussian-beam propagation engine (oblique-incidence astigmatism, periodic-system eigenmodes). Originally factored out of three `fys501-laser-physics` applets in [`mikkojhuttunen/physics-applets`](https://github.com/mikkojhuttunen/physics-applets). See its own README for the physics and API. |
