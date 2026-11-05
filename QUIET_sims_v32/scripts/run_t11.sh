#!/bin/sh
# Task 11: grating-coupler design (8 x 25 um inclusions, output facet 11) at 5-20 mm, all dither designs, drift in the analysis
cd "$(dirname "$0")/.."
FAM='d4ref:16:50e-6:11:4,d5:8:25e-6:11,d10:8:25e-6:11,d15:8:25e-6:11,d20:8:25e-6:11'
for B in nir mir; do for R in 0.97 0.995; do
  node scripts/t11_sizes.js --band $B --R $R --dBcm 0.3 --cells "$FAM" --dithers all --out results/t11/final_${B}_R${R}.json
done; done
