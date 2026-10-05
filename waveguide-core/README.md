# waveguide-core

Shared photonic waveguide mode solver (plain JS, no build step): materials (Sellmeier / Cauchy),
1D slab solver, effective index method and a 2D semi-vectorial finite-difference solver.
`WGCORE()` and `WGWORKER()` are plain functions so a page can rebuild them inside a Blob Web Worker
with `Function.prototype.toString()`.

Used by [`er-waveguide-amplifier.html`](../er-waveguide-amplifier.html). Copied from
[`mikkojhuttunen/physics-applets`](https://github.com/mikkojhuttunen/physics-applets) (`assets/js/waveguide-core.js`),
where the photonic waveguide mode explorer still uses its own copy. If you fix the solver, fix both.
