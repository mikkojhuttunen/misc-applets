#!/bin/sh
# Task 9b: wide angle sweeps (+-30, +-45, +-60 mrad), combinations with position, adaptive dither sampling
cd "$(dirname "$0")/.."
for R in 0.95 0.97 0.995; do node scripts/t9_input_states.js --band nir --R $R --dBcm 0.3 --out results/t9b/nir_R${R}_0.3dB.json; done
node scripts/t9_input_states.js --band mir --R 0.97 --dBcm 0.3 --out results/t9b/mir_R0.97_0.3dB.json
node scripts/t9_input_states.js --band nir --R 0.97 --dBcm 0.3 --sigma 14e-6 --out results/t9b/nir_R0.97_0.3dB_sigma14.json
