#!/usr/bin/env bash
# Step 11: robustness of the static fringe against film thickness (n_eff, Gamma) and wavelength-dither depth.
# Circle CMPC, target spec (designed R(chi) table of the 600 nm film, 0.1 dB/cm), films 500 - 700 nm in 25 nm steps
# (n_eff, Gamma_E of TM0 at 1.55 um from qs/slabmodes.py), 30 layouts + the regular cell each, with --stat (fringe
# floor against the wavelength-dither depth). bash scripts/run_step11.sh [jobs] -> results/step11/circle_t<nm>.json
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/step11
python3 - > results/step11/films.txt <<'PY'
import sys
sys.path.insert(0, '.')
from qs.slabmodes import mode, n_luke
n = float(n_luke(1.55))
for t in range(500, 701, 25):
    m = mode(n, t * 1e-9, 1.55e-6)
    print(t, round(m['n_eff'], 4), round(m['gamma_E'], 4))
PY
while read -r t ne g; do
  echo "--cell circle --n_eff $ne --gamma $g --rtable results/step1/Rtable_circle_neff1.66.json --dBcm 0.1 --seeds 30 --stat --out results/step11/circle_t$t.json"
done < results/step11/films.txt | xargs -P "$J" -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1 && echo "done: {}" | sed "s/.*--out //"'
