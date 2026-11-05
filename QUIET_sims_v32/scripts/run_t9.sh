#!/bin/sh
# Task 9: wave-consistent input dithers (Gaussian-beamlet model), circle cell, set B, 0.3 dB/cm, 30 layouts + regular
cd "$(dirname "$0")/.."
for R in 0.95 0.97 0.995; do node scripts/t9_input_states.js --band nir --R $R --dBcm 0.3 --out results/t9/nir_R${R}_0.3dB.json; done
node scripts/t9_input_states.js --band mir --R 0.97 --dBcm 0.3 --out results/t9/mir_R0.97_0.3dB.json
node scripts/t9_input_states.js --band nir --R 0.97 --dBcm 0.3 --sigma 7e-6 --out results/t9/nir_R0.97_0.3dB_sigma7.json
node scripts/t9_input_states.js --band nir --R 0.97 --dBcm 0.3 --sigma 14e-6 --out results/t9/nir_R0.97_0.3dB_sigma14.json
