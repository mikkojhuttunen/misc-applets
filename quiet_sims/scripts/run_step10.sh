#!/usr/bin/env bash
# Step 10: spec cases of the 600 nm SiNx cell (n_eff 1.66, Gamma 0.313) for the Fig. 2 / Fig. 3 artifacts, 30 layouts + regular.
#   ideal  : designed R(chi) table, 0 dB/cm
#   target : designed R(chi) table, 0.1 dB/cm  (= results/step2/<cell>_neff1.66.json, copied)
#   minimum: R(chi) capped at 0.95 below the critical angle (TIR kept), 0.3 dB/cm
# bash scripts/run_step10.sh [jobs]  -> results/step10/<cell>_<case>.json
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
mkdir -p results/step10
python3 - <<'PY'
import json
for cell in ('circle', 'herriott'):
    t = json.load(open(f'results/step1/Rtable_{cell}_neff1.66.json'))
    sc = 1 / 1.66
    t['R'] = [min(r, 0.95) if s < sc else r for s, r in zip(t['s'], t['R'])]
    t['design'] = dict(t['design'], R_avg=0.2 + 0.8 * 0.95,
                       note='minimum spec: design R(chi) capped at 0.95 below the critical angle (sin chi < 1/n_eff), TIR kept above')
    json.dump(t, open(f'results/step1/Rtable_{cell}_neff1.66_min95.json', 'w'))
PY
for cell in circle herriott; do cp results/step2/${cell}_neff1.66.json results/step10/${cell}_target.json; done
jobs=()
for cell in circle herriott; do
  jobs+=("--cell $cell --n_eff 1.66 --gamma 0.313 --rtable results/step1/Rtable_${cell}_neff1.66.json --dBcm 0 --seeds 30 --out results/step10/${cell}_ideal.json")
  jobs+=("--cell $cell --n_eff 1.66 --gamma 0.313 --rtable results/step1/Rtable_${cell}_neff1.66_min95.json --dBcm 0.3 --seeds 30 --out results/step10/${cell}_minimum.json")
done
printf '%s\n' "${jobs[@]}" | xargs -P "$J" -I{} sh -c 'node scripts/ensemble.js {} > /dev/null 2>&1 && echo "done: {}" | sed "s/.*--out //"'
