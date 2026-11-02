"""Step 10 summary: ideal / target / minimum spec of the 600 nm SiNx cells (30 layouts each), circle and Herriott.

python scripts/step10_specs.py  -> results/step10_summary.json and a markdown table on stdout
Static fringe V_det = 2 sqrt(eps) V / sqrt(grains) (eps 1e-6) per dither mode; NH3 limit at 1 s from the median noise of
the 30 layouts (shot noise at 1 mW and the fringe residual, as step 2), alpha_NH3 = max of the Fig. 3 NH3 strip.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
CASES = {'ideal': 'designed R(χ), 0 dB/cm', 'target': 'designed R(χ), 0.1 dB/cm', 'minimum': 'R ≤ 0.95 below θc, 0.3 dB/cm'}


def main():
    fig3 = json.load(open(os.path.join(RES, 'baseline', 'fig3data.json')))
    Tg, aNH3 = fig3['T'], max(fig3['strips'][0]['analyte_per_ppm'])
    lg = [math.log(t) for t in Tg]

    def at1(arr):
        i = 0
        while i < len(lg) - 2 and lg[i + 1] < 0:
            i += 1
        f = -lg[i] / (lg[i + 1] - lg[i])
        return math.exp(math.log(arr[i]) * (1 - f) + math.log(arr[i + 1]) * f)

    q = lambda a, p: float(np.quantile(a, p))
    out = {}
    print('| cell | spec | static fringe none / spectral / angle / both (chaotic median) | both: quartiles | both, regular | layouts < 5e-7 | Γ⟨L⟩ | T | ⟨R⟩ bounce | NH3 1 s median [quartiles] |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for cell in ('circle', 'herriott'):
        for case, lab in CASES.items():
            d = json.load(open(os.path.join(RES, 'step10', f'{cell}_{case}.json')))
            c = [x for x in d['chaotic'] if 'error' not in x]
            g = d['gamma']
            V = {m: [2e-3 / math.sqrt(x['grains']) * (1 if m == 'none' else x[k]) for x in c]
                 for m, k in (('none', None), ('spectral', 'V_lambda'), ('angle', 'V_theta'), ('both', 'V_joint'))}
            reg = d['regular']
            reg_both = 2e-3 / math.sqrt(reg['grains']) * reg['V_joint']
            nea = [math.hypot(at1(x['env']['both']), math.sqrt(1.602176634e-19 / (1e-3 * x['T_trans']))) for x in c]
            ppb = [n / (g * x['L_mean_m'] * 100 * aNH3) * 1e3 for n, x in zip(nea, c)]
            r = {'spec': lab, 'n': len(c), 'fringe_median': {m: q(v, .5) for m, v in V.items()},
                 'fringe_both_q1_q3': [q(V['both'], .25), q(V['both'], .75)], 'fringe_both_regular': reg_both,
                 'below_5e-7': sum(v < 5e-7 for v in V['both']) / len(c), 'GL_cm': g * q([x['L_mean_m'] for x in c], .5) * 100,
                 'T': q([x['T_trans'] for x in c], .5), 'R_bounce': q([x['R_bounce'] for x in c], .5),
                 'nh3_ppb_1s': {'median': q(ppb, .5), 'q1': q(ppb, .25), 'q3': q(ppb, .75)}}
            out[f'{cell}_{case}'] = r
            fm = r['fringe_median']
            print(f"| {cell} | {case} ({lab}) | {fm['none']:.1e} / {fm['spectral']:.1e} / {fm['angle']:.1e} / {fm['both']:.1e} | "
                  f"{r['fringe_both_q1_q3'][0]:.1e}–{r['fringe_both_q1_q3'][1]:.1e} | {reg_both:.1e} | {100 * r['below_5e-7']:.0f} % | "
                  f"{r['GL_cm']:.2f} cm | {r['T']:.2f} | {r['R_bounce']:.4f} | {r['nh3_ppb_1s']['median']:.0f} ppb [{r['nh3_ppb_1s']['q1']:.0f}–{r['nh3_ppb_1s']['q3']:.0f}] |")
    json.dump(out, open(os.path.join(RES, 'step10_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
