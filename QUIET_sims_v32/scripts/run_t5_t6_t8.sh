#!/usr/bin/env bash
# Tasks 5, 6, 8 (set B near-IR working point: 500 nm SiNx, 1531.7 nm, n_eff 1.5391, Gamma 0.4532, R 0.97, 0.3 dB/cm)
#  t5: angle-dither range 2, 5, 10, 20, 50 mrad (+-), circle and Herriott, 30 layouts + regular, speckle correlation
#      curves (--curves) -> results/t5/<cell>_A<mrad>.json
#  t6: inclusions one parameter at a time around 16 x dn 0.01 x r 50 um (that case = results/t2/circle_nir_R0.97_0.3dB.json):
#      dn 0.001 0.002 0.005; number 4 8 32; radius 25 100 um; + Poincare coverage of every layout -> results/t6/
#  t8: layouts 31-100 of the working point -> results/t8/circle_seeds31_100.json, + coverage of all 100
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/t5 results/t6 results/t8
B="--wavelength 1.5317e-6 --n_eff 1.5391 --gamma 0.4532 --R 0.97 --dBcm 0.3"
{
for A in 2 5 10 20 50; do for cell in circle herriott; do
  echo "node scripts/ensemble.js --cell $cell $B --seeds 30 --curves --dither ${A}e-3 --out results/t5/${cell}_A${A}.json"
done; done
for dn in 0.001 0.002 0.005; do echo "node scripts/ensemble.js --cell circle $B --seeds 30 --paths --inc_dn $dn --out results/t6/circle_dn${dn}.json"; done
for n in 4 8 32; do echo "node scripts/ensemble.js --cell circle $B --seeds 30 --paths --n_inc $n --out results/t6/circle_n${n}.json"; done
for r in 25e-6 100e-6; do echo "node scripts/ensemble.js --cell circle $B --seeds 30 --paths --inc_radius $r --out results/t6/circle_r${r}.json"; done
echo "node scripts/ensemble.js --cell circle $B --seeds 70 --first 31 --paths --stat --out results/t8/circle_seeds31_100.json"
for dn in 0.001 0.002 0.005 0.01; do echo "node scripts/chaos_metrics.js --cell circle --inc_dn $dn --seeds 30 --out results/t6/cov_dn${dn}.json"; done
for n in 4 8 32; do echo "node scripts/chaos_metrics.js --cell circle --n_inc $n --seeds 30 --out results/t6/cov_n${n}.json"; done
for r in 25e-6 100e-6; do echo "node scripts/chaos_metrics.js --cell circle --inc_radius $r --seeds 30 --out results/t6/cov_r${r}.json"; done
echo "node scripts/chaos_metrics.js --cell circle --seeds 100 --out results/t8/cov_100.json"
echo "node scripts/chaos_metrics.js --cell herriott --seeds 30 --out results/t6/cov_herriott.json"
} | xargs -P "$J" -I{} sh -c '{} > /dev/null 2>&1; echo "finished: {}" | sed "s/.*--out //"'
