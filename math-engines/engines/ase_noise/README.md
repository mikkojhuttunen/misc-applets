# ase_noise

Level-C ASE noise core: linear, phase-insensitive multimode channels acting on Gaussian states. A state is a coherent signal amplitude vector `alpha` (sqrt(photons/s)) and a normally ordered ASE correlation matrix `C` (photons per mode per s per Hz) for one frequency bin and polarisation. A channel `a -> T a + f` maps them as

```
alpha' = T alpha,      C' = T C T† + D
```

| Building block | T | D |
|---|---|---|
| `gain_channel(G, n_sp, unitary=U)` | U diag(√G) U† | U diag(n_sp (G-1)) U† |
| `loss_channel(η or matrix)` | √η, or any matrix with singular values ≤ 1 | 0 |
| `coupling_channel(U)` | U | 0 |
| `aperture_channel(lg_modes(N), a/w)` | circular-aperture overlap matrix in the LG basis | 0 |

`compose`, `propagate`, `noise_figure_of`, `osnr_of` and `detect` (shot, signal-spontaneous and spontaneous-spontaneous beat noise behind a mode projector W) work on any of these. Friis cascades, the 3 dB quantum limit and the mode-count scaling of sp-sp noise follow from the matrix algebra rather than being put in by hand.

Spec (calculator) functions: `amplifier_noise`, `span_cascade`, `multimode_snr`. Demo scripts with plots are in [`amplifiers/ase-noise/`](../../../amplifiers/ase-noise/).

See `spec.yaml` for inputs, outputs and equations.
