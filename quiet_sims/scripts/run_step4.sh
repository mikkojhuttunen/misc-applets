#!/usr/bin/env bash
# Step 4b: mid-infrared SiNx cells at the CH4 line 3.3151 um: films 1.0 / 1.2 / 1.5 um (n_eff 1.46 / 1.59 / 1.71, Gamma
# from design_calcs/slab.py with n = 1.97), membrane loss 0.1 / 0.3 / 0.5 dB/cm, R(sin chi) tables of step 1, circle and
# Herriott cells, 21 layouts + regular.  bash scripts/run_step4.sh [jobs] -> results/step4/<cell>_neff<n>_<dB>dB.json
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/step4
declare -A GAMMA=([1.46]=0.533 [1.59]=0.372 [1.71]=0.220)
jobs=()
for cell in circle herriott; do for ne in 1.46 1.59 1.71; do for db in 0.1 0.3 0.5; do
  jobs+=("--cell $cell --n_eff $ne --gamma ${GAMMA[$ne]} --wavelength 3.3151e-6 --rtable results/step1/Rtable_${cell}_neff${ne}.json --dBcm $db --seeds 21 --out results/step4/${cell}_neff${ne}_${db}dB.json")
done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$J" -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1 && echo "done: {}" | sed "s/.*--out //"'
