# misc-applets

Self-contained interactive applets, one `.html` file each, no build step, on top of a shared set of tested calculation engines in `math-engines/`.

| Applet | Description |
|---|---|
| [parametric-amplifier.html](parametric-amplifier.html) | Fiber parametric amplifier: coupled-wave RK4 solver for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) OPAs with a signal near 1550 nm. LP01 fiber dispersion, gain bandwidth and idler spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and bandwidth optimisation over loss, fiber geometry, length and pump power. |
| [er-waveguide-amplifier.html](er-waveguide-amplifier.html) | Er:Al₂O₃ strip-loaded waveguide amplifier on Al₂O₃, Si₃N₄, TFLN or TFLT. Er concentration, strip and film dimensions, length, pump 980/1480 nm; pump and signal overlaps with the doped strip (2D finite-difference modes), steady-state rate equations with upconversion, pump/signal propagation, net gain spectrum, gain vs length, pump power and Er concentration, optional concentration-dependent upconversion and ion quenching, and an optimizer for strip/film dimensions, concentration and length. Moved here from `physics-applets`. |

| [cmpc-ray-tracer.html](cmpc-ray-tracer.html) | Multi-beam ray tracer for a segmented circular multipass cell: number of beams, centre angle (slider or pointer steering in the cell view) and fan width, star-orbit aiming, global facet curvature, random tilt and curvature perturbations, single-facet tilt/curvature, optional input/output ports; mirror reflectance with intensity Rʲ and effective path Σ Rʲ ℓⱼ; phase vs launch angle per pass and fringe visibility under sine/triangle angle dither, explained pass by pass by the dither depth x = a_p A (J₀/sinc model), phase chirp and facet-path switching, with adaptive sampling and the random-phase noise floor (`ray_phase` engine, run in a background worker); etched-trench DBR mirrors with R(χ) applied at every hit, angle-of-incidence distribution and the reflectance the light actually sees compared with the unperturbed cell (`cell_mirror` engine); figure view with a tight scale bar; phase-space (Poincaré) plot and twin-ray Lyapunov estimate. Geometry is a JS port of `billiard_cell.SegmentedCell`, checked against the Python test vectors. |
| [cmpc-sim/](cmpc-sim/) | Chip-scale chaotic multipass cell simulator (Python): membrane evanescent coupling Γ, etched-trench DBR mirrors at oblique incidence, stadium and segmented-cell ray statistics, trace-gas Voigt spectra, speckle/etalon noise model; ten design figures. Built on the `membrane_mode`, `bragg_grating`, `billiard_cell`, `trace_gas` and `path_coherence` engines in `math-engines/`. |

## Shared engines

Plain, dependency-free JS modules (not standalone applets) meant to be
reused across multiple applets instead of re-derived each time.

| Folder | Contents |
|---|---|
| [beam-propagation/](beam-propagation/) | Polygon-billiard ray tracing and periodic-orbit solving for circular segmented multipass cells, plus an ABCD-matrix Gaussian-beam propagation engine (oblique-incidence astigmatism, periodic-system eigenmodes). Originally factored out of three `fys501-laser-physics` applets in [`mikkojhuttunen/physics-applets`](https://github.com/mikkojhuttunen/physics-applets). See its own README for the physics and API. |
| [waveguide-core/](waveguide-core/) | Photonic waveguide mode solver: Sellmeier materials, 1D slab solver, effective index method, 2D semi-vectorial finite-difference solver (usable inside a Web Worker). Used by the Er waveguide amplifier. |
| [nlo-dynamics/](nlo-dynamics/) | Nonlinear dynamics engine (`nlo-engine.js`): split-step Fourier solver for the nonlinear Schrödinger equation and its generalisation (Kerr, self-steepening, Raman, user-defined local nonlinearities, dispersion to any order, loss, gain filter), RK4/RK45 ODE integrators, pulse and spectrum analysis, and the Haus mode-locking round-trip model from the FYS.501 mode-locking explorer. 36-check validation suite, runnable examples. |
