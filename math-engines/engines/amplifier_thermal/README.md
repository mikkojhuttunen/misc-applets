# amplifier_thermal

Heating along an Er-doped waveguide amplifier. Same effective two-level Er model as `er-waveguide-amplifier.html`: 980 nm pump only absorbs, 1480 nm also stimulates emission, upconversion C_up N₂², and a quenched fraction that only absorbs. Intensities are overlap-averaged over the doped area here (I = Γ P / A_d), whereas the applet uses the 2D mode profile. Pump and signal are integrated with RK4.

Heat per length is everything the ions and the absorbing part of the background loss take out of the beams, minus the spontaneous emission that is radiated away:

    q'(z) = Γ_p(σ_a,p n₁ − σ_e,p N₂)P_p + Γ_s(σ_a,s n₁ − σ_e,s N₂)P_s − η_rad (N₂/τ) hν_s A_d + f_abs α (P_p + P_s)

The equivalent form is quantum defect + non-radiative decay + upconversion + quenched-ion absorption (tested), so the heat balance closes along the whole amplifier (tested to 1e-4). ΔT(z) = R′ q′(z), with R′ taken from `waveguide_thermal` (local 2D resistance, no heat flow along z).

- `er_amplifier_heating(pump_power, signal_power, length, R_th, dneff_dT, …)`: gain, absorbed pump, total heat and heat fraction, peak and mean ΔT, the hot-spot position, peak Δn_eff, the thermal phase of the signal, inversion; arrays z, P_pump, P_signal, q, dT, inversion.
- `er_pump_limit(dT_max, dneff_max, P_search, …)`: the largest launched pump power within the ΔT or Δn_eff budget (bisection), and the gain there.

Checks: energy conservation, the heat decomposition, the weak-pump limit (heat fraction → 1 − λ_p/λ_s), a fully quenched film (all of it becomes heat, Beer–Lambert pump decay), and in-band pumping heating less.

What it shows for the default Al₂O₃:Er strip on TFLN (`examples/pump_heating.py`): once the pump bleaches the Er absorption, the ions add little heat. The absorbing part of the background loss and quenched ions dominate, and the input reaches 5 K only at several watts of pump.

Version 1.
