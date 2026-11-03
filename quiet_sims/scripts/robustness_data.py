"""Compact step-11 robustness data for the Fig. 2 / Fig. 3 robustness artifacts.

from robustness_data import build; build() -> dict (also written to results/step11_artifact_data.json by __main__)
Per film (500 - 700 nm): static fringe at the nominal dither for no / angle / wavelength / both dithers (regular cell,
chaotic median and quartiles of 30 layouts) and the NH3 limit at 1 s for angle-only and both dithers (per layout:
sqrt(r(1 s)^2 + q/(P T)) / (Gamma <L> alpha_NH3), then median and quartiles); the fringe against the wavelength-dither
depth (5 - 60 pm) for the 600 nm film; the window statistics of results/step11_summary.json.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FILMS = list(range(500, 701, 25))
MODES = ('none', 'angle', 'spectral', 'both')


def build():
    fig3 = json.load(open(os.path.join(RES, 'baseline', 'fig3data.json')))
    lg, aNH3 = np.log(fig3['T']), max(fig3['strips'][0]['analyte_per_ppm'])
    at1 = lambda arr: float(np.exp(np.interp(0.0, lg, np.log(arr))))
    shot = lambda T: math.sqrt(1.602176634e-19 / (1e-3 * T))
    ppb = lambda x, m: math.hypot(at1(x['env'][m]), shot(x['T_trans'])) / (x['gamma'] * x['L_mean_m'] * 100 * aNH3) * 1e3
    r3 = lambda v: float(f'{v:.4g}')
    out = {'films': FILMS, 'n_eff': [], 'gamma': [], 'n_layouts': [], 'fringe': {m: {'reg': [], 'med': [], 'q1': [], 'q3': []} for m in MODES},
           'nh3': {m: {'reg': [], 'med': [], 'q1': [], 'q3': []} for m in ('angle', 'both')}}
    for t in FILMS:
        d = json.load(open(os.path.join(RES, 'step11', f'circle_t{t}.json')))
        c = [x for x in d['chaotic'] if 'error' not in x]
        out['n_eff'].append(d['n_eff']); out['gamma'].append(d['gamma']); out['n_layouts'].append(len(c))
        for m in MODES:
            v = [x['floor'][m] for x in c]
            F = out['fringe'][m]
            F['reg'].append(r3(d['regular']['floor'][m])); F['med'].append(r3(np.median(v)))
            F['q1'].append(r3(np.quantile(v, .25))); F['q3'].append(r3(np.quantile(v, .75)))
        for m in ('angle', 'both'):
            v = [ppb(x, m) for x in c]
            N = out['nh3'][m]
            N['reg'].append(r3(ppb(d['regular'], m))); N['med'].append(r3(np.median(v)))
            N['q1'].append(r3(np.quantile(v, .25))); N['q3'].append(r3(np.quantile(v, .75)))
        if t == 600:
            dl = np.array(d['stat_dl']) * 1e12
            k = (dl >= 5) & (dl <= 60)
            cb = np.array([x['stat']['both'] for x in c])
            out['curve600'] = {'dl_pm': [r3(v) for v in dl[k]], 'reg': [r3(v) for v in np.array(d['regular']['stat']['both'])[k]],
                               'med': [r3(v) for v in np.median(cb, 0)[k]], 'q1': [r3(v) for v in np.quantile(cb, .25, 0)[k]],
                               'q3': [r3(v) for v in np.quantile(cb, .75, 0)[k]]}
    s = json.load(open(os.path.join(RES, 'step11_summary.json')))
    out['window_pm'] = s['window_pm']
    out['window'] = {'regular': s['regular_device'], 'chaotic_typical': s['chaotic_device_typical'], 'chaotic_q1_q3': s['chaotic_device_q1_q3'],
                     'chaotic_ensemble_median': s['chaotic_ensemble_median'], 'layouts_beating_regular_worst_case': s['layouts_beating_regular_worst_case']}
    out['target'] = 5e-7
    out['source'] = ('QUIET step 11: circle CMPC, 600 nm-class SiNx membranes 500-700 nm (TM0 n_eff and Gamma at 1.55 um), target spec '
                     '(designed R(chi), 0.1 dB/cm), +-2 mrad angle dither, +-20 pm wavelength dither, eps 1e-6, 1 mW, 30 inclusion layouts per film')
    return out


if __name__ == '__main__':
    json.dump(build(), open(os.path.join(RES, 'step11_artifact_data.json'), 'w'))
    print('ok')
