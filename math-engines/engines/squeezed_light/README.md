# squeezed_light

Squeezed light from χ⁽²⁾ processes in the Hamiltonian formalism instead of classical coupled-wave equations. The interaction Hamiltonian (momentum generator for propagation) is the input; Heisenberg equations, Bogoliubov transformations and noise follow from it.

| Function | Model |
|---|---|
| `degenerate_squeezing` | Single-pass degenerate OPA: G/ħ = (Δk/2)a†a + i(γ/2)(a†² − a²) → drift A = 2ΩM → covariance with distributed loss, detection loss and phase jitter |
| `two_mode_squeezing` | Non-degenerate OPA: G/ħ = (Δk/2)(n_s + n_i) + iγ(a_s†a_i† − a_s a_i); EPR variance, thermal marginals, log-negativity |
| `opo_spectrum` | Degenerate OPO below threshold, H/ħ = i(ε/2)(a†² − a²) with input–output (Collett–Gardiner spectrum) |
| `squeezed_vacuum_statistics` | ⟨n⟩, Var n, g⁽²⁾, P(n) of S(r)\|0⟩ and of the two-mode squeezed vacuum; quadratures after loss |
| `infer_squeezing` | r and total efficiency from a measured squeezing/anti-squeezing pair |
| `fock_degenerate` | Exact evolution of H/ħ = i(g/2)(a†²b − a²b†) from \|α⟩\|β⟩: quantum pump, depletion, SHG squeezing of the fundamental |
| `fock_two_mode` | Exact evolution of H/ħ = ig(a_s†a_i†b − a_s a_i b†) from \|0,0⟩\|β⟩ |

Helpers: `hamiltonian_degenerate`, `hamiltonian_two_mode` (matrix M of ½RᵀMR), `drift_matrix`, `gaussian_evolution` (Van Loan solution of dV/dz = AV + VAᵀ + D), `lossy_channel`, `bogoliubov`, `log_negativity`, `expm` (NumPy-only, runs in Pyodide), `fock_degenerate_moments`, `fock_two_mode_moments` (vectorised over τ).

Conventions: x = a + a†, p = −i(a − a†), vacuum variance 1 (shot noise); coupling γ = √(ηP) with η = 8π²d²/(ε₀c n_s n_i n_p λ_s λ_i A_eff), the normalised SHG efficiency when degenerate, so r = γL at Δk = 0. Fock routes use τ = gt with da/dτ = b a†, so the parametric limit is r = |β|τ. Fock inputs are limited to |α|, |β| ≤ 12 and |α|² + |β|² ≤ 150 to keep calls to a few seconds.

Cross-checks in `test_engine.py`: covariance route vs closed-form Bogoliubov coefficients (including imaginary g), distributed-loss closed form, two-mode EPR = degenerate supermode, OPO formula vs Fourier-domain Langevin equations, photon distributions summed directly, Fock evolution → parametric approximation as β grows, conserved numbers, short-time SHG.

Front end: [`squeezed-light/`](../../../squeezed-light/index.html). See `spec.yaml` for inputs, outputs and equations.
