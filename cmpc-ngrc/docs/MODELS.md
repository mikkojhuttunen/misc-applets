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

## 2. Rays and beamlets (`rays.trace`)

Between perturbations the slab is uniform and rays are straight lines. Inside the bounding circles the
ray equation and the paraxial (dynamic) ray system are integrated in arclength s with RK4:

    dr/ds = t,  dp/ds = ∇n  (p = n t, |p| = n),  dL/ds = n
    dQ/ds = P / n,  dP/ds = (n_nn − 2 n_n² / n) Q

n_n and n_nn are the first and second derivatives of n along the ray normal e = (−t_y, t_x). This is
Červený's 2D dynamic ray tracing written for n instead of the velocity v = 1/n (dQ/ds = vP,
dP/ds = −v⁻² v_nn Q). The step is ds = w/2 in the edge zones; flat interiors are crossed in larger steps
that stop short of the next edge zone. |p| − n converges at 4th order (tests).

Each ray carries a Gaussian beamlet. Its field at normal offset q from the ray is

    E = A sqrt(Q₀/Q) exp(i k₀ [L + ½ (P/Q) q²]),   P₀/Q₀ = i n₀ / z_R,  z_R = k₀ n₀ w_b² / 2

with arg Q followed continuously, which gives the Gouy phase. In uniform slab Q → Q + P s/n₀.

**Wall.** Specular reflection t → t − 2(t·n)n, amplitude × √R(χ), phase + φ_R. The curved mirror acts on the
beamlet as the tangential oblique-incidence lens P → P − 2 n₀ Q κ / cos χ, with κ = 1/R_c for the circle
and −κ_element for `gmpc.planar` walls (`geometry.WallCell`). A hit inside a port aperture ends the ray
with an exit record. Rays are dropped after a bounce limit or below an amplitude floor (both counted).

**Source.** Each input port launches a grid of n_pos positions across 90 % of the aperture × n_ang
angles (launch ± fan/2 about the inward normal). Amplitudes follow a Gaussian aperture weight and are
normalised to Σ A² = 1.

**Detector field** (`field.beamlet_matrix`). The exit rays of a port are continued straight to a
64-pixel line at 100 µm outside the port, and the beamlets are summed coherently: E = G @ 1, with
G[pixel, ray] the beamlet field.

## 3. Phase-screen model (`perturbative.PhaseScreenModel`)

Trace the unperturbed cell once and keep every chord. A set of dots only adds ΔL_j = ∫Δn ds along the
unperturbed chords of ray j (midpoint rule; the integrand is smooth and zero at both ends, so it
converges spectrally). Then E = G @ exp(i k₀ ΔL). This is identical to the tracer with straight rays
(`mode="straight"`), and it is the Δn → 0 limit of the curved result.

It neglects ray bending and beamlet focusing by the dots. For multipass speckle that matters from
Δn ≈ 1e-4: the field correlation with curved rays is 0.97 at 1e-4 and 0.84 at 1e-3 (E4). Over a single
pass both agree with BPM (E9), so the difference builds up over the many passes.

## 4. Wave reference (`wave_ref`)

Split-step Fourier BPM of the paraxial scalar equation ∂E/∂z = (i/2k) ∂²E/∂y² + i k₀ Δn E, with absorbing
edges. It is compared with a Gaussian-beam summation of the same beam (parallel beamlets spaced
w_b/1.5) traced through one dot with the engine, curved or straight. The metric is the scattered field
ΔE = E(dot) − E(no dot). A full-wave model of a whole 1 mm cell (≈ 2300 wavelengths across) is not
attempted.

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
- Ports are holes in the ray picture: a ray exits or reflects depending on where its centre hits, so
  single rays switch discontinuously at port edges, and aperture diffraction is missing.
- Beamlet summation is asymptotic: beamlets much wider than the dots or the mirror curvature scale lose
  accuracy (E9 error ≈ 10 % at 6 mm, 2–5 % at 20–60 mm).
- The circle is integrable: a launch angle χ leaves a caustic disk of radius R_c sin χ that no ray
  enters. The stadium is chaotic and has no such disk.

## References

- M. Born, E. Wolf, *Principles of Optics*, ch. 3 (ray equation in inhomogeneous media).
- V. Červený, *Seismic Ray Theory* (Cambridge, 2001), ch. 4 (dynamic ray tracing), ch. 5.8 (Gaussian beam summation).
- H. Kogelnik, T. Li, "Laser beams and resonators", Appl. Opt. 5, 1550 (1966) (oblique-incidence mirror ABCD).
- D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, "Next generation reservoir computing", Nat. Commun. 12, 5564 (2021).
- G. Van der Sande, D. Brunner, M. C. Soriano, "Advances in photonic reservoir computing", Nanophotonics 6, 561 (2017).
