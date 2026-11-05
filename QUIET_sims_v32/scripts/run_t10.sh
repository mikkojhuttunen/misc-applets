#!/bin/sh
# Task 10: design levers (input / output ports, inclusions, cell size) with the beamlet model, set B, 0.3 dB/cm, 30 layouts
cd "$(dirname "$0")/.."
EXTRA='{"d10_in2":{"r":5e-3,"n_in":2},"d10_inc8":{"r":5e-3,"n_inc":8},"d10_r25":{"r":5e-3,"inc_r":25e-6},"d14":{"r":7e-3},"d20":{"r":10e-3},"d20_r25":{"r":10e-3,"inc_r":25e-6},"d10_old":{"r":5e-3,"w":40e-6,"A":2e-3,"Xa":0},"d20_old":{"r":10e-3,"w":40e-6,"A":2e-3,"Xa":0}}'
MAIN=base,old_dither,in2,in3,in4,out1r,out2,out4,out8,inc8,inc32,inc64,r25,r25n64,r100,dn005,dn02,dn04,d7,d7s,d10,d10s,d10_in2,d10_inc8,d10_r25,d14,d20,d20_r25,d10_old,d20_old
for R in 0.97 0.995; do node scripts/t10_design.js --band nir --R $R --dBcm 0.3 --variants $MAIN --extra "$EXTRA" --out results/t10/nir_R${R}_0.3dB.json; done
PORTS=$(node -e "console.log(Array.from({length:23},(_,i)=>'of'+(i+1)).join(','))")
node scripts/t10_design.js --band nir --R 0.97 --dBcm 0.3 --variants $PORTS --out results/t10/ports_nir_R0.97_0.3dB.json
node scripts/t10_design.js --band mir --R 0.97 --dBcm 0.3 --variants base,old_dither,d10,d14,d20 --extra "$EXTRA" --out results/t10/mir_R0.97_0.3dB.json
# inputs at facets whose separation to the output (facet 11) is good in the port scan (11, 16, 7 facets)
GOOD='{"in2g":{"in_facets":[0,19]},"in3g":{"in_facets":[0,19,4]},"d20r25_in2g":{"r":10e-3,"inc_r":25e-6,"in_facets":[0,19]},"d20r25_in3g":{"r":10e-3,"inc_r":25e-6,"in_facets":[0,19,4]},"d20_r25":{"r":10e-3,"inc_r":25e-6}}'
for R in 0.97 0.995; do node scripts/t10_design.js --band nir --R $R --dBcm 0.3 --variants base,in2g,in3g,d20_r25,d20r25_in2g,d20r25_in3g --extra "$GOOD" --out results/t10/inputs_nir_R${R}_0.3dB.json; done
