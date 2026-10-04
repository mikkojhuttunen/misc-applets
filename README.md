# misc-applets

Self-contained interactive applets, no build step. Each applet has its own folder at the root with an `index.html` that carries all of its CSS and JavaScript.

| Applet | Description |
|---|---|
| [parametric-amplifier/](parametric-amplifier/) | Fiber parametric amplifier: coupled-wave RK4 solver for χ⁽²⁾ (532 nm pump) and χ⁽³⁾ (degenerate pump) OPAs with a signal near 1550 nm. LP01 fiber dispersion, gain bandwidth and idler spectrum, continuous/lumped and spectrally engineered idler loss, Brillouin and Raman scattering, and bandwidth optimisation over loss, fiber geometry, length and pump power. |
| [er-waveguide-amplifier/](er-waveguide-amplifier/) | Er:Al₂O₃ strip-loaded waveguide amplifier on an Al₂O₃, Si₃N₄, TFLN or TFLT film with SiO₂ or air top cladding. Pump (980/1480 nm) and signal modes from the effective index method or a semi-vectorial 2D finite-difference solver, overlaps with the doped strip, steady-state rate equations with upconversion and adjustable quenching, pump/signal propagation, net gain spectrum, gain along the device and against pump power, Er concentration and quenching, and an optimizer for strip/film dimensions, concentration and length. |

The mode solver in `er-waveguide-amplifier/` is a copy of `assets/js/waveguide-core.js` from [physics-applets](https://github.com/mikkojhuttunen/physics-applets), which also hosts the [waveguide mode explorer](https://mikkojhuttunen.github.io/physics-applets/photonics/waveguide-mode-explorer.html).
