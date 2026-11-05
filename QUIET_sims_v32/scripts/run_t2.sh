#!/usr/bin/env bash
# Task 2 (set B, realistic) and the mid-IR ensembles for Task 7: constant R per bounce, membrane loss, 30 layouts +
# regular, fringe against dither depth (--stat) and exit-path lists (--paths).
#   near-IR: 500 nm SiNx TM0 at the NH3 line 1531.7 nm: n_eff 1.5391, Gamma_E 0.4532; circle and Herriott
#   mid-IR : 1000 nm SiNx TM0 at the CH4 line 3315.1 nm: n_eff 1.4253, Gamma_E 0.5526; circle
# bash scripts/run_t2.sh [jobs] -> results/t2/<cell>_<band>_R<R>_<dB>dB.json
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/t2
{
for R in 0.95 0.97 0.995; do for dB in 0.3 0.5; do
  for cell in circle herriott; do
    echo "--cell $cell --wavelength 1.5317e-6 --n_eff 1.5391 --gamma 0.4532 --R $R --dBcm $dB --seeds 30 --stat --paths --out results/t2/${cell}_nir_R${R}_${dB}dB.json"
  done
  echo "--cell circle --wavelength 3.3151e-6 --n_eff 1.4253 --gamma 0.5526 --R $R --dBcm $dB --seeds 30 --stat --paths --out results/t2/circle_mir_R${R}_${dB}dB.json"
done; done
} | xargs -P "$J" -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1; echo "finished: {}" | sed "s/.*--out //"'
