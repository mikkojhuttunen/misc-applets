# beam-propagation

Shared, dependency-free math engines factored out of three applets in
[`mikkojhuttunen/physics-applets`](https://github.com/mikkojhuttunen/physics-applets)
(`fys501-laser-physics/multipass-cell-ray-tracer`, `.../cmpc-gaussian-beam`):
a polygon-billiard ray tracer for circular segmented multipass cells (CMPCs),
and an ABCD-matrix Gaussian-beam propagation engine. Pulled out here so the
geometry/physics has one tested, documented home instead of being
copy-pasted (and re-derived from scratch) into every new applet that needs it.

Both files are plain vanilla JS, no build step, no dependencies — exactly
the house style of this repo. Each works two ways:

```html
<script src="ray-tracing-engine.js"></script>
<script>
  const orbit = RayTracingEngine.findStableOrbit(32, 5, 200, 0, 0);
</script>
```

```js
// or in Node (e.g. for the test scripts, or server-side verification)
const { findStableOrbit } = require('./ray-tracing-engine.js');
```

## Files

| File | What it does |
|---|---|
| `ray-tracing-engine.js` | Regular-N-gon billiard geometry: general ray tracing (`simulateBilliard`) and an exact closed-form periodic-orbit solver (`findStableOrbit`) for the "star polygon" paths a circular segmented multipass cell is designed around. |
| `abcd-gaussian-engine.js` | Paraxial Gaussian-beam propagation via the complex beam parameter `q`, including oblique-incidence spherical-mirror astigmatism (tangential vs. sagittal focal length) and periodic-system eigenmode solving. |
| `test/*.test.js` | Plain-Node (no framework) verification scripts. Run with `node test/<name>.test.js`; each prints a check count and throws (non-zero exit) on any failure. |

## The physics, briefly

**Ray tracing.** A circular segmented multipass cell approximates a circular
mirror with `N` flat facets arranged in a regular polygon. A beam bouncing
around the inside generally does *not* retrace itself — but for special
entry conditions it does, tracing a closed "star polygon" {N/m}. Because the
whole configuration is invariant under rotation by `Delta = 2*pi*m/N`
(a symmetry of the regular polygon itself), the reflection law needs to be
solved at only *one* point; every other bounce is just that point rotated.
`findStableOrbit` does this with a bisection root-find and closes to ~1e-13
in practice (see the test suite). The resulting path is periodic with period
`N/gcd(N,m)` — past that many bounces, a locked orbit adds no new coverage
or path length, which matters when you're also tracking beam divergence
(see below) against the finite width of each facet.

**ABCD / Gaussian beam.** Flat mirrors don't focus (their ABCD matrix is the
identity), so a beam bouncing between flat facets just diverges monotonically
like it would in open space. Giving each facet a (weak) concave spherical
curvature `R_m` lets it refocus the beam every bounce — but at the oblique
incidence angle `theta` this ring geometry requires, a spherical mirror's
focusing power splits into a tangential focal length `f_t = R_m*cos(theta)/2`
and a weaker sagittal one `f_s = R_m/(2*cos(theta))`. A mirror curvature
chosen to refocus one plane generally doesn't refocus the other — astigmatism
that grows with incidence angle. `solveMatchedQ` finds the self-consistent
("eigenmode") beam parameter for a periodic sequence of identical unit cells,
analogous to solving for a laser-resonator mode; `propagateUnitCellSequence`
runs an arbitrary (possibly mismatched) injected beam through the same
sequence, which is what you want for showing a beam settle into a bounded
oscillation (stable, mismatched), stay exactly constant (stable, matched), or
diverge outright (unstable, `|g| >= 1`).

Both engines were originally built and verified interactively (see commit
history / the physics-applets repo for the three applets that consume them);
the `test/` scripts here are the ported, standalone versions of those
verification checks — self-consistency to machine precision, no NaNs, and
sane limiting cases (normal incidence, the degenerate `m = N/2` "diameter"
orbit, sub-star orbits where `gcd(N,m) > 1`).

## Units

Pick one consistent length unit and use it everywhere in a given call graph;
the applets use millimeters throughout (`lambdaMm`, `waistMm`, `LMm`,
`RmMm`, canvas pixels converted to mm via a `scaleMmPerPx` factor computed
from the cell's physical size). Angles taken in degrees are named `...Deg`;
everything else is radians.

## Using this from a published, self-contained HTML artifact

A fully self-contained artifact (e.g. a Claude-published page) can't load an
arbitrary `<script src>` from GitHub directly, but **jsDelivr's GitHub proxy
is usually on the allowed script-host list**, so this works:

```html
<script src="https://cdn.jsdelivr.net/gh/mikkojhuttunen/misc-applets@main/beam-propagation/ray-tracing-engine.js"></script>
<script src="https://cdn.jsdelivr.net/gh/mikkojhuttunen/misc-applets@main/beam-propagation/abcd-gaussian-engine.js"></script>
```

Pin to a commit or tag instead of `@main` for anything you don't want to
shift under you later (e.g. `@<commit-sha>`).
