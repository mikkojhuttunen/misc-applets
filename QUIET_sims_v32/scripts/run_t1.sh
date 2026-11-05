#!/usr/bin/env bash
# Task 1 (regression, set A): circle CMPC and Herriott, 30 inclusion layouts + regular, n_eff 2, Gamma 0.2, R 0.995,
# no membrane loss, CH4 1653.7 nm, +-2 mrad / +-20 pm, with the exit-path lists and the fringe against dither depth.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/t1
for cell in circle herriott; do
  echo "--cell $cell --n_eff 2 --gamma 0.2 --R 0.995 --dBcm 0 --seeds 30 --stat --paths --out results/t1/${cell}_A.json"
done | xargs -P 2 -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1 && echo "done: {}" | sed "s/.*--out //"'
