# cmpc-ngrc models

Equations, assumptions and validity limits of the engine. SI units throughout; code references in brackets.

## 1. Slab and perturbations

The membrane guides one TE slab mode with effective index n₀ = n_eff (e.g. 1.8 for a ~300 nm Si₃N₄-like
film at 1.55 µm, `geometry.membrane_neff`). Everything is 2D in the slab plane. A local change of the film
(thickness Δt, cladding, analyte) changes the mode index by Δn(x, y) = (∂n_eff/∂t) Δt
(`geometry.dneff_dthickness`). Mode conversion, radiation and scattering out of the slab at the dot
edges are neglected; they are small when Δn ≪ n₀ and the edges are smooth on the scale of λ/n₀.

**Dots** (`shapes.Shape`). Contour r(θ) = R [1 + Σ_m (a_m cos m(θ − θ_r) + b_m sin m(θ − θ_r))] around
(x₀, y₀), Δn(x, y) = Δn · ½[1 + tanh((r(θ) − ρ)/w)], with ρ, θ polar coordinates about the centre and w the
edge width. Value, gradient and Hessian are analytic. The bounding circle R(1 + Σ|a_m| + |b_m|) + 7w holds
all of Δn to < 1e-6 of the peak.

**Images** (`index_map.GridIndex`). Δn on a pixel grid, optionally Gaussian-blurred (the blur sets the
edge width), interpolated with a bicubic spline, so ∇Δn and its Hessian are continuous.

**Circular-harmonic decomposition (CHD)** (`harmonics`). For a dot: the a_m, b_m above (rotated into the
cell frame). For any Δn map: the half-maximum contour r(θ_k) along N rays from the |Δn|-weighted
centroid, then c_m = (1/N) Σ_k r_k e^{−imθ_k}, R₀ = c₀, a_m = 2 Re c_m / R₀, b_m = −2 Im c_m / R₀.
The rotation-invariant power is p_m = a_m² + b_m².

## 2. Rays (`rays.trace`)

Between perturbations the slab is uniform and rays are straight. Inside the dots' bounding circles the ray
equation and the paraxial (dynamic) ray system are integrated in arclength s with RK4:

    dr/ds = t,  dp/ds = ∇n  (p = n t, |p| = n),  dL/ds = n
    dQ/ds = P / n,  dP/ds = (n_nn − 2 n_n² / n) Q

n_n and n_nn are the first and second derivatives of n along the ray normal (Červený's 2D dynamic ray tracing
written for n). The step is ds = w/2 in the dots' edge zones; flat interiors are crossed in larger steps.
At the wall: specular reflection, amplitude × √R, phase + φ_R, and the mirror acts on (Q, P) as the tangential
lens P → P − 2 n₀ Q κ / cos χ (κ = 1/R_c in the circle, −κ_element for `gmpc.planar` walls).

## 3. Field: frozen Gaussians (Herman–Kluk), the default (`cell.model = "fga"`)

The input beam at a port (waist w_in = λ/(π n₀ tan(fan/2)), launch angle φ₀, seen on the port line with width
w_in / cos φ₀) is expanded in coherent states g_{q,p}(x) = (γ/π)^¼ exp(−γ(x − q)²/2 + i k₀ p (x − q)) on a grid of
positions q (automatic, spacing w_f/2) × directions φ (p = n₀ sin φ), with γ = 2/w_f² (w_f = `Source.frozen`,
20 µm by default). Each grid point launches one ray with weight

    W = ⟨g_{q,p} | E₀⟩ Δq Δp k₀ / 2π.

The ray carries its optical path L and the stability matrix, read from Q = A + iβB, P = C + iβD (two real
solutions of the dynamic system packed into one complex one). At a port opening the coherent state is
re-expressed on the opening (A/cos χ, B/cos χ, C cos χ, D cos χ) and contributes

    W · R · e^{i k₀ L + iφ} · (γ/π)^¼ exp(−γ (x − x_t)²/2) · e^{i k₀ n₀ t·(r − r_hit)},
    R = sqrt(½ (A + D − i(γ/k₀) B + i(k₀/γ) C)),

the Herman–Kluk prefactor, with its branch followed continuously along the ray. The frozen Gaussians keep their
width, so a dot only changes the rays that cross it. That locality is what the earlier Gaussian-beam
summation lacked: in the circle, a degenerate (concentric-type) mirror system, its beamlets grew to millimetres.

**Ports.** Every wall hit within w/2 + 3w_f of an opening is an exit record. The ray reflects with amplitude
× sqrt(1 − T), where T = ½[erf(√2(w/2 − d)/w_f) + erf(√2(w/2 + d)/w_f)] is the part of its footprint that falls
in the opening. The response is therefore smooth in the ray geometry, and wide-angle leakage is included.
The field across each opening (samples every λ/3n₀) is propagated to the detectors. The default detector is the
far field: E(θ) = sqrt(k/2πi) Σ_k A_k e^{−ik d̂(θ)·r_k} cos θ du, with 64 directions at sin θ uniform in ±0.9. A
line of points at a given distance is also available (2D Rayleigh–Sommerfeld, kR ≫ 1).

**Sampling.** Convergence needs rays landing within ~w_f of every point of the openings after the last
relevant bounce. ∂q/∂p grows with the path, so the number of launch directions grows with the bounce count.
At R = 0.97, w_f = 20 µm and a 1 mm cell: 4000 directions give field correlation 0.90–0.99 with 8000. Records
weaker than 2 % of the strongest are pruned (correlation ≥ 0.998).

**Chaotic cells.** In the stadium the stability matrix grows like e^{λn} (λ ≈ 1 per bounce). The prefactors
explode while their contributions should cancel, and the semiclassical field is only reliable up to the
Ehrenfest time ln(kL)/λ ≈ 9 bounces. Rays are dropped once |z| > 10⁴. In the circle |z| grows linearly (a few
hundred after 300 bounces) and the cut-off never acts.

**Validation** (`tests/test_fga.py`, E9). Against an exact Gaussian beam, the clipped single pass gives field
correlation 1.0000 and power within 0.1 %, independent of w_f. Triangle and pentagon orbits through 2 and 4 oblique
mirror reflections give correlation ≥ 0.9997 and power within 0.2 %. For a dot (Δn up to 1e-2) crossed by a
beam and observed after 1.4 mm, the scattered field agrees with split-step BPM to correlation 0.9997–0.9998.

The legacy model (`cell.model = "gbs"`): evolving Gaussian beamlets about each ray, hard ports (a ray leaves when
its centre hits an opening), beamlets continued straight to the detectors. It is kept for comparison; its
results are archived in results/progress_gbs.json.

## 4. Phase-screen model (`perturbative.PhaseScreenModel`, JS `PhaseScreen`)

Trace the unperturbed cell once and keep every chord. A set of dots only adds ΔL = ∫Δn ds along the chords
travelled before each exit record (prefix sums per ray). Then E = K (B · e^{i k₀ ΔL}), where B holds the
records' aperture fields and K the propagation to the detectors. The line integrals come from each dot's
Radon table, ∫Δn along direction α at offset p, about 260 × 270 entries. Those entries are computed by Newton
crossings of the contour plus 12-point Gauss–Legendre windows over the tanh edges, or 32-point
Gauss–Legendre for grazing lines. Every chord crossing reads the table by bilinear interpolation
(|k₀ ΔL| error ≲ 2e-3 rad). This is identical to the straight-ray tracer, and the Δn → 0 limit of curved
rays. Ray bending matters from Δn ≈ 3e-4 (field correlation with curved rays: 0.99 at 1e-4, 0.95 at 3e-4,
0.84 at 1e-3, E4).

Cost (JS, circle, R = 0.97, 6000 directions, two inputs): about 1 s per sample after a one-off 30–60 s trace
per worker. Datasets run through node on all cores (`ngrc.jsengine.node_fields`).

## 4b. Wave reference (`wave_ref`)

Split-step Fourier BPM of the paraxial scalar equation ∂E/∂z = (i/2k) ∂²E/∂y² + i k₀ Δn E with absorbing edges,
for one pass through a dot. A full-wave model of the whole 1 mm cell (≈ 2300 wavelengths across) is not
attempted; the multi-bounce physics is checked against exact Gaussian beams through the mirrors (above).

## 5. Features and readouts (`features`, `readout`)

The features are relative intensities I/I₀ − 1 on every detector pixel, for every (input, output) pair
and optional cell variant (launch angle or wavelength steps: `CircularCell.variant`,
`dataset.build_multiplexed`). The NGRC feature vector is [1, x, x_i x_j]. The readouts are:

- ridge regression (closed form, penalty by cross-validation);
- kernel ridge with linear, poly2 (= NGRC quadratic in kernel form) or RBF kernels, optionally
  multiplied by a position kernel exp(−|p − p'|²/2ℓ²);
- linear and RBF SVR / SVC (scikit-learn).

Targets are a_m, b_m (R² averaged over the pair), p_m and the dominant m.

## 6. Known limits

- 2D scalar TE model; no polarisation or mode conversion; wall reflectance constant or R(cos χ).
- Ray (semiclassical) model: valid while dots, edges and openings are many wavelengths; edge widths ≳ λ/n.
  Chaotic cells only to about the Ehrenfest time (prefactor cut-off).
- FGA needs many rays for long paths (the launch grid scales with the number of bounces); results at
  finite sampling are a consistent but not fully converged ray field (stated per experiment).
- Curved rays vs the phase screen (E7b): ray bending moves the frozen Gaussians' endpoints (~0.1 µm after a
  0.15 m path at Δn = 1e-4). In a converged sum that cancels, but at a few thousand directions the residual
  (~δx/w_f ≈ 1e-2) exceeds the shape signal (~6e-4 rad for Δa₂ = 0.02). Curved-ray datasets are therefore
  dominated by sampling noise, and the phase screen (exact Δn → 0 limit) is used for the readout studies.
- Openings are treated as soft apertures (Gaussian footprint overlap for the reflected power, exact
  sampling of the transmitted field); wall curvature across an opening is ignored in the far-field kernel.
- The circle is integrable: a launch angle χ leaves a caustic disk of radius R_c sin χ that no ray
  enters. The stadium is chaotic and has no such disk.
- Centred dot (the study's default, D9): only rays with R_c |sin χ| below the dot radius cross it, so the launch
  must be near 0. A chord through the centre in direction α integrates r(α) + r(α + π) at first order, so odd
  orders vanish for chords through the centre. Worse, a ray keeps its offset p = R_c sin χ, and the chord after
  a bounce is nearly the point reflection of the one before, which flips the sign of every odd harmonic: odd
  phases cancel chord by chord while even ones accumulate. The circle reads the even orders of a centred dot
  almost perfectly and the odd ones weakly (E3, E13, E14); the stadium reads all of them (E15). Every chord
  crosses the dot, so Δn must be small (≲ 3e-5, E12).

## References

- M. Born, E. Wolf, *Principles of Optics*, ch. 3 (ray equation in inhomogeneous media).
- V. Červený, *Seismic Ray Theory* (Cambridge, 2001), ch. 4 (dynamic ray tracing), ch. 5.8 (Gaussian beam summation).
- H. Kogelnik, T. Li, "Laser beams and resonators", Appl. Opt. 5, 1550 (1966) (oblique-incidence mirror ABCD).
- D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, "Next generation reservoir computing", Nat. Commun. 12, 5564 (2021).
- G. Van der Sande, D. Brunner, M. C. Soriano, "Advances in photonic reservoir computing", Nanophotonics 6, 561 (2017).
