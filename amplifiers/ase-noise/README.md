# ase-noise

Noise build-up in optical amplifiers on top of the [`ase_noise`](../../math-engines/engines/ase_noise/) engine (level-C core: signal amplitude vector and ASE correlation matrix per mode, propagated through linear phase-insensitive channels) and the [`multimode_amplifier`](../../math-engines/engines/multimode_amplifier/) engine (highly multimode waveguides, mode-dependent gain, mode coupling, saturation).

| Script | Simulation |
|---|---|
| `cascade_osnr.py` | N × (span + amplifier): OSNR in 0.1 nm and chain noise figure vs N for 15/20/25 dB spans, checked against 58 + P − L − NF − 10 log N |
| `multimode_mdg.py` | 100 µm / NA 0.2 core at 1064 nm, 881 step-index or 435 graded-index modes: modal gain vs U/V for confined doping ρ = 1, 0.7, 0.4, how the output ASE is spread over the modes (M_eff), and MDG growth with length uncoupled (∝ L) vs randomly coupled (∝ √L) |
| `saturated_multimode.py` | Cladding-pumped Yb-like amplifier in the same 881-mode fibre: signal and ASE along z, spatial hole burning that gives higher-order modes up to ~12 dB more gain than the LP01 signal, and how confined doping suppresses it (gain advantage and output ASE fraction vs seed power) |
| `multimode_vs_pinhole.py` | Equal-gain multimode amplifier (LG modes up to order N_max), signal in LG_00: SNR vs mode count for no filter, pinholes and a single-mode fibre; SNR vs pinhole radius; shot / s-sp / sp-sp noise budget |

```
pip install numpy scipy matplotlib
python cascade_osnr.py [output_dir]
python multimode_vs_pinhole.py [output_dir]
python multimode_mdg.py [output_dir]
python saturated_multimode.py [output_dir]
```

`multimode_mdg.py` takes about 25 s and `saturated_multimode.py` about 1 min; both also need scipy.

PNGs are written next to the scripts (or to `output_dir`) and are not committed.

## Model notes

- Signal–spontaneous beat noise sees only ASE that overlaps the signal's mode after the filter. Spontaneous–spontaneous noise scales with the effective number of detected modes, M_eff = tr(C)²/tr(C²).
- The pinhole is a lens, an aperture in the focal plane and a recollimating lens. The aperture couples radial orders p ↔ p' of equal l. With a small pinhole, ASE from LG_p0 modes leaks into the signal's transmitted field, so s-sp noise rises when the first p = 1 group appears (the a = 0.7 w_f step).
- For a large-area detector the pinhole uses T = A^½ (the polar form of the aperture map). This is exact for total-power quantities, with no basis-truncation error.
- Multimode amplifiers: each mode's gain comes from its overlap with the doped region and the local inversion. Spontaneous emission enters each mode at the rate γ_e,k (one polarisation, per Hz), so n_sp per mode follows from the saturated inversion. Random coupling unitaries between sections mix the modes (coupling strength set by a Δβ correlation).
- The saturated model is co-pumped and steady-state, uses one ASE spectral bin and has no backward ASE. Off-diagonal gain overlaps between non-degenerate modes are neglected.
- Not yet included: spectral resolution and bidirectional ASE (full level B), counter-pumping, thermal and Kerr effects, Monte Carlo photon statistics (level D).
