# cwe_dfg

Coupled-wave difference-frequency generation and optical parametric amplification (CW) in a χ² waveguide with an arbitrary poling profile g(z). Reuses the domain representation, Δk tables and grating helpers of [`cwe_shg`](../cwe_shg/).

Version 1. SI in and out.

## Model

ω_p = ω_s + ω_i (the `phase_matching` convention ω3 = ω1 + ω2 with 1 = signal, 2 = idler, 3 = pump), |A|² in W:

```
dA_s/dz = -α_s/2 A_s + i κ_s g A_p A_i* e^{-iφ}
dA_i/dz = -α_i/2 A_i + i κ_i g A_p A_s* e^{-iφ}
dA_p/dz = -α_p/2 A_p + i κ_p g A_s A_i  e^{+iφ},    φ = ∫ Δk dz,  Δk = β_p - β_s - β_i
```

κ_j ∝ ω_j, κ_p² = η_SFG = 8π² d² / (ε0 c n_s n_i n_p λ_p² A_eff). Parametric gain coefficient Γ² = κ_s κ_i P_p, the same as the χ² mode of `parametric-amplifier.html`. SHG is the degenerate limit: η_SFG = 4 η0(SHG at 2λ_p).

## Functions

| Function | Kind | What |
|---|---|---|
| `coupling_three_wave` | spec | κ_s, κ_i, κ_p, η_SFG, Γ/√P |
| `ordered_opa` | spec | ordered grating: closed-form gain, exact transfer matrix, depleted RK4, idler power, DFG efficiency, diagnostics, degeneracy and depletion warnings |
| `opa_transfer` | helper | undepleted pump, per-domain 2×2 matrix exponentials, signal/idler loss, vectorised over Δk |
| `three_wave` | helper | full depleted RK4, wall-aware, losses on all waves, power and Manley-Rowe residuals |
| `opa_gain_uniform` | helper | 1 + (Γ/g)² sinh²(gL) |
| `idler_wavelength` | helper | 1/(1/λ_p - 1/λ_s) |

## Tests

κ_p = κ_s + κ_i and the SHG limit; uniform gain = cosh²(ΓL) at Δk = 0 and the g formula on both sides of the band edge, for the transfer matrix and RK4; idler photons = signal photons gained; QPM gain vs Γ_eff = 2Γ/π; transfer matrix vs RK4 for random domains with loss; depleted power and Manley-Rowe conservation and gain saturation; low-gain DFG = κ_i² P_p P_s |G|²; per-domain Δk table; summary and warnings. `tools/check_js_ports.mjs` checks the `expM` transfer matrix of `parametric-amplifier.html` against the `cwe_dfg_uniform` vectors.

## Not included

Pulses, group-velocity effects, phase-sensitive degenerate OPA, Kerr and thermal effects.
