# cwe_cascade

Cascaded third-harmonic generation in a χ² waveguide: ω + ω → 2ω (SHG, mismatch Δk_a = β2 − 2β1) and ω + 2ω → 3ω (SFG, Δk_b = β3 − β2 − β1), both driven by the same poling g(z). Builds on [`cwe_shg`](../cwe_shg/) and [`cwe_dfg`](../cwe_dfg/) (same normalisation, domains and Δk tables).

Version 1. SI in and out.

## Model

```
dA1/dz = -α1/2 A1 + i κ_a g A2 A1* e^{-iφa} + i κ_1 g A3 A2* e^{-iφb}
dA2/dz = -α2/2 A2 + i κ_a g A1²   e^{+iφa} + i κ_2 g A3 A1* e^{-iφb}
dA3/dz = -α3/2 A3 + i κ_3 g A1 A2  e^{+iφb}
```

κ_a² is the SHG η0, κ_3² the SFG efficiency of ω + 2ω, and κ_1 : κ_2 : κ_3 = 1 : 2 : 3. Each process has its own d and overlap area (the 3ω mode can be a higher-order mode). Undepleted ω, lowest order: A3(L) = −κ_3 κ_a A1³ J with the nested integral J = ∫ g e^{iφb} ∫ g e^{iφa}, which a single Fourier transform does not give; it is evaluated exactly per domain, with loss as complex rates.

## Functions

| Function | Kind | What |
|---|---|---|
| `coupling_thg` | spec | κ_a, κ_1..3, SHG and SFG efficiencies, double-matched THG coefficient (P3 = η P³ L⁴) |
| `two_section_thg` | spec | SHG grating then SFG grating: periods, closed form, exact undepleted, coupled waves, diagnostics |
| `cascade_integral`, `cascade_undepleted` | helper | exact nested integral, A2 and A3, vectorised over Δk rows, κ per row allowed |
| `cascade_coupled` | helper | full RK4, wall-aware, depletion and back-conversion, loss, power residual, step halving |
| `two_section_domains` | helper | two ordered sections with a split fraction |
| `commensurate_orders` | helper | single periods matching both processes with orders (m_a, m_b), residual mismatch, best duty cycle |

## Tests

Couplings equal those of `cwe_shg` and `cwe_dfg`; uniform double-matched L⁴/4 law (exact and RK4); nested integral against brute-force quadrature for random domains, continuous g and loss; two-section closed form (2/π)⁴ κ3² κa² P³ La² Lb² and the optimum split 1/2; undepleted vs RK4 at low power; power conservation and saturation at high power; loss; commensurate-order search; per-domain tables; summary and depletion warning.

## Not included

Direct χ³ THG, pulses, group-velocity effects, Kerr and thermal effects.
