# cwe_shg

Coupled-wave second-harmonic generation in a χ² waveguide with an arbitrary poling profile g(z). First Tier-1 engine of the `chi2-waveguide-sim` plan; `cwe_cascade` (THG) and `cwe_dfg` (DFG/OPA) will reuse its domain representation and integrator (`from ..cwe_shg.engine import ...`).

Version 1. SI in and out.

## Model

Amplitudes normalised to power (|A|² in W):

```
dA1/dz = -α1/2 A1 + i κ g(z) A2 A1* e^{-iφ(z)}
dA2/dz = -α2/2 A2 + i κ g(z) A1²    e^{+iφ(z)},   φ(z) = ∫ Δk dz,   Δk = β_SH - 2 β_p
```

The grating vector is not subtracted from Δk: g(z) carries it. κ² = η0 = 8π² d² / (ε0 c n_p² n_SH λ² A_eff) is the efficiency of a full-strength, phase-matched medium, so d is the bulk coefficient for the cut and polarisation (d33 for x-cut TE or z-cut TM) and the first-order factor 2/π comes out of g(z).

Poling: boundaries `z_0 = 0 < ... < z_N = L` plus one value `g_k ∈ [-1, 1]` per domain. Δk is a scalar, an array over wavelengths `(M,)`, or a table `(M, N)` with one value per domain.

## Functions

| Function | Kind | What |
|---|---|---|
| `coupling_shg` | spec | κ and η0 from d and the overlap area |
| `ordered_shg` | spec | ordered grating at the matched period (plus offset): closed form, exact undepleted and coupled-wave SH power, diagnostics |
| `periodic_domains`, `domains_from_lengths` | helper | build the domain representation |
| `poling_spectrum` | helper | G(Δk) = ∫ g e^{iφ} dz, exact per domain, vectorised over Δk |
| `shg_undepleted` | helper | undepleted SH amplitude with loss (exact per domain) |
| `shg_coupled` | helper | full RK4; `stepping="walls"` never straddles a wall, `"uniform"` is the naive reference; power residual and optional step-halving error |
| `shg_depleted_matched` | helper | P tanh²(κ√P L) |

## Tests

Hand value of η0; period agrees with `qpm_shg` and `phase_matching`; undepleted ordered spectrum vs sinc² and `grating_strength` for D = 0.3, 0.5, 0.7; FWHM vs `qpm_shg`; depleted Δk = 0 vs tanh²; QPM vs tanh² with κ_eff = 2κ/π; power conservation for random domains, shrinking with the step; wall-aware vs uniform stepping (identical on aligned walls, uniform much worse for random walls); undepleted vs coupled-wave below 1 % conversion; loss.

## Not included

Pulses and group-velocity effects, Kerr, thermo-optic and photorefractive effects, domain-wall scattering. CW only.
