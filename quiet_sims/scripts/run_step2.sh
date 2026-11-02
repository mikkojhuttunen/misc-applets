#!/usr/bin/env bash
# Step 2 ensembles: circle and Herriott cells, SiNx n_eff 1.36 / 1.53 / 1.66 / 1.74 (400 - 700 nm at 1.55 um), R(sin chi)
# tables of step 1, 0.1 dB/cm, 30 layouts + regular; plus the 600 nm cells with 5 nm sidewall roughness.
# bash scripts/run_step2.sh [jobs]  -> results/step2/<cell>_neff<n>[_rough5nm].json
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/step2
declare -A GAMMA=([1.36]=0.670 [1.53]=0.463 [1.66]=0.313 [1.74]=0.215)
jobs=()
for cell in circle herriott; do
  for ne in 1.36 1.53 1.66 1.74; do
    jobs+=("--cell $cell --n_eff $ne --gamma ${GAMMA[$ne]} --rtable results/step1/Rtable_${cell}_neff${ne}.json --dBcm 0.1 --seeds 30 --out results/step2/${cell}_neff${ne}.json")
  done
  jobs+=("--cell $cell --n_eff 1.66 --gamma 0.313 --rtable results/step1/Rtable_${cell}_neff1.66_rough5nm.json --dBcm 0.1 --seeds 30 --out results/step2/${cell}_neff1.66_rough5nm.json")
done
printf '%s\n' "${jobs[@]}" | xargs -P "$J" -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1 && echo "done: {}" | sed "s/.*--out //"'
