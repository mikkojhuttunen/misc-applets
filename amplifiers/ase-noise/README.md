# ase-noise

Noise build-up in optical amplifiers on top of the [`ase_noise`](../../math-engines/engines/ase_noise/) engine (level-C core: signal amplitude vector and ASE correlation matrix per mode, propagated through linear phase-insensitive channels).

| Script | Simulation |
|---|---|
| `cascade_osnr.py` | N × (span + amplifier): OSNR in 0.1 nm and chain noise figure vs N for 15/20/25 dB spans, checked against 58 + P − L − NF − 10 log N |
| `multimode_vs_pinhole.py` | Equal-gain multimode amplifier (LG modes up to order N_max), signal in LG_00: SNR vs mode count for no filter, pinholes and a single-mode fibre; SNR vs pinhole radius; shot / s-sp / sp-sp noise budget |

```
pip install numpy matplotlib
python cascade_osnr.py [output_dir]
python multimode_vs_pinhole.py [output_dir]
```

PNGs are written next to the scripts (or to `output_dir`) and are not committed.

## Model notes

- Signal–spontaneous beat noise sees only ASE that overlaps the signal's mode after the filter. Spontaneous–spontaneous noise scales with the effective number of detected modes, M_eff = tr(C)²/tr(C²).
- The pinhole is a lens, an aperture in the focal plane and a recollimating lens. The aperture couples radial orders p ↔ p' of equal l. With a small pinhole, ASE from LG_p0 modes leaks into the signal's transmitted field, so s-sp noise rises when the first p = 1 group appears (the a = 0.7 w_f step).
- For a large-area detector the pinhole uses T = A^½ (the polar form of the aperture map). This is exact for total-power quantities, with no basis-truncation error.
- Not yet included: gain saturation, spectral dependence and bidirectional ASE (level B), mode-dependent gain and gain guiding, Monte Carlo photon statistics (level D).
