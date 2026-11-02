#!/usr/bin/env bash
# Step 3: launch-angle study of the circular CMPC. Star orbit with m facets between launch and exit: chi = 90 - 180 m/24
# deg from the wall normal, out_frac = m/24, for m = 11, 8, 5, 2, 1 and SiNx n_eff 1.66 (600 nm) and 1.36 (400 nm).
# Per case: the bounce-angle histogram (8 layouts), an N = 20 trench DBR optimised for it, the ensemble (21 layouts and
# the regular cell, 0.1 dB/cm).  bash scripts/run_step3.sh [jobs] -> results/step3/
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/step3
declare -A GAMMA=([1.36]=0.670 [1.66]=0.313)
cases=()
for m in 11 8 5 2 1; do for ne in 1.66 1.36; do cases+=("$m $ne"); done; done
printf '%s\n' "${cases[@]}" | xargs -P "$J" -L1 sh -c '
  m=$0; ne=$1; d=results/step3
  node scripts/angle_hist.js --cell circle --m $m --n_eff $ne --seeds 8 --out $d/anglehist_m${m}_neff${ne}.json > /dev/null
  python3 -W ignore scripts/make_rtable.py --hist $d/anglehist_m${m}_neff${ne}.json --n_eff $ne --N 20 --out $d/Rtable_m${m}_neff${ne}.json
  echo done'
G1=${GAMMA[1.66]}; G2=${GAMMA[1.36]}
printf '%s\n' "${cases[@]}" | xargs -P "$J" -L1 sh -c '
  m=$0; ne=$1; d=results/step3; g='"$G1"'; [ "$ne" = "1.36" ] && g='"$G2"'
  node scripts/ensemble.js --cell circle --m $m --n_eff $ne --gamma $g --rtable $d/Rtable_m${m}_neff${ne}.json --dBcm 0.1 --seeds 21 --out $d/circle_m${m}_neff${ne}.json > /dev/null 2>&1
  echo "ensemble m=$m n_eff=$ne done"'
