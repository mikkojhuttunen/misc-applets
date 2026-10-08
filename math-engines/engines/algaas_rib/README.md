# algaas_rib

AlGaAs-on-insulator rib waveguide with a **layer-poled** χ⁽²⁾ profile: flat core with χ⁽²⁾ > 0 and rib with
χ⁽²⁾ < 0, for modally phase-matched SHG with a TE pump fundamental and a higher-order vertical TM second harmonic
(the only d14 channel for propagation along [110] in (001) AlGaAs). SI units.

- `algaas_n`, `algaas_index`: Afromowitz index (absolute accuracy ~1e-2: calibrate with `RibStack.dn_algaas`).
- `RibStack`: h_core, h_rib, width (at the rib base), x_core, x_rib, SiO₂ or air claddings, side-wall angle.
- `slab_modes`, `eim_neff`, `eim_dispersion`: transfer-matrix slab modes and the effective index method (fast, ~1e-2 off).
- `rib_modes`: semi-vectorial FD (Ex for TE, Ey for TM) on an interface-aligned non-uniform grid, Richardson-extrapolated.
- `overlap`, `shg_design`, `shg_modal_pm`: Γ = ∫ s Ex_p² Ey_SH dA, A_eff, Δk, coherence length, η_norm.
- `delta_n`, `tune_parameter`, `sweep`, `dispersion`, `group_index`: design helpers.

Modes are labelled (lateral nodes, vertical nodes). Limitations: semi-vectorial (no hybrid components), isotropic
homogeneous layers (MQW treated by its average index), no sidewall roughness or loss, Dirichlet box 2 µm from the rib.
Needs scipy. Version 1.
