"""Step 11: robustness of the static fringe (both dithers) of the regular and the chaotic circular CMPC against film
thickness (500 - 700 nm SiNx, n_eff and Gamma of TM0) and wavelength-dither depth, target spec (designed R(chi),
0.1 dB/cm), 30 inclusion layouts per film (results/step11/circle_t<nm>.json from scripts/run_step11.sh).

python scripts/step11_robustness.py -> results/step11_summary.json, results/figs/fig_robustness.pdf/.png, tables on stdout

Operating window: every film x every grid depth within +-20 % of the nominal 20 pm (the engine grid, 32 points per
decade: 16.5 - 23.7 pm), 9 x 6 = 54 operating points. Also each dither alone at the nominal point, every film. One device = the regular cell, or one inclusion layout of the
chaotic cell; each is judged over the whole window. NH3 at 1 s: sqrt(fringe^2 + q/(P T)) / (Gamma <L> alpha_NH3), with
the static fringe as the residual at 1 s (it sets r(1 s) to within a few % for these cells).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qs import style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FILMS = list(range(500, 701, 25))
TGT = 5e-7
NOM = 10 ** 1.3                       # nominal wavelength dither, pm


def load():
    fig3 = json.load(open(os.path.join(RES, 'baseline', 'fig3data.json')))
    aNH3 = max(fig3['strips'][0]['analyte_per_ppm'])
    D = {}
    for t in FILMS:
        d = json.load(open(os.path.join(RES, 'step11', f'circle_t{t}.json')))
        D[t] = d
    return D, np.array(D[FILMS[0]]['stat_dl']) * 1e12, aNH3


def ppb(fr, x, aNH3):
    return math.hypot(fr, math.sqrt(1.602176634e-19 / (1e-3 * x['T_trans']))) / (x['gamma'] * x['L_mean_m'] * 100 * aNH3) * 1e3


def main():
    D, dl, aNH3 = load()
    win = np.where((dl >= 0.8 * NOM) & (dl <= 1.2 * NOM))[0]
    out = {'window_pm': [float(dl[win[0]]), float(dl[win[-1]])], 'films_nm': FILMS, 'n_points': len(FILMS) * len(win), 'films': {}}
    reg_pts, reg_ppb, lay_pts, lay_ppb, med_pts = [], [], {}, {}, []   # over the window, both dithers
    for t in FILMS:
        d = D[t]
        r = d['regular']
        c = [x for x in d['chaotic'] if 'error' not in x and 'stat' in x]
        rb = np.array(r['stat']['both'])
        cb = np.array([x['stat']['both'] for x in c])
        reg_pts += list(rb[win])
        reg_ppb += [ppb(v, r, aNH3) for v in rb[win]]
        med_pts += list(np.median(cb[:, win], axis=0))
        for x, row in zip(c, cb):
            lay_pts.setdefault(x['seed'], []).extend(row[win])
            lay_ppb.setdefault(x['seed'], []).extend(ppb(v, x, aNH3) for v in row[win])
        out['films'][t] = {'n_eff': d['n_eff'], 'gamma': d['gamma'], 'n': len(c),
                           'nominal': {'regular': r['floor']['both'], 'chaotic_median': float(np.median([x['floor']['both'] for x in c])),
                                       'chaotic_q1': float(np.quantile([x['floor']['both'] for x in c], .25)),
                                       'chaotic_q3': float(np.quantile([x['floor']['both'] for x in c], .75)),
                                       'nh3_regular': ppb(r['floor']['both'], r, aNH3),
                                       'nh3_chaotic_median': float(np.median([ppb(x['floor']['both'], x, aNH3) for x in c]))},
                           'curve_dl_pm': dl.tolist(), 'curve_regular': rb.tolist(), 'curve_chaotic_median': np.median(cb, 0).tolist(),
                           'curve_chaotic_q1': np.quantile(cb, .25, 0).tolist(), 'curve_chaotic_q3': np.quantile(cb, .75, 0).tolist()}

    def dev(v, p):
        v, p = np.array(v), np.array(p)
        return {'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)), 'max': float(v.max()), 'min': float(v.min()),
                'max_over_min': float(v.max() / v.min()), 'share_below_5e-7': float(np.mean(v < TGT)), 'share_below_1e-7': float(np.mean(v < 1e-7)),
                'nh3_median': float(np.median(p)), 'nh3_p90': float(np.quantile(p, .9)), 'nh3_max': float(p.max())}
    out['regular_device'] = dev(reg_pts, reg_ppb)
    lays = {s: dev(lay_pts[s], lay_ppb[s]) for s in lay_pts}
    out['chaotic_layouts'] = lays
    keys = ['median', 'p90', 'max', 'max_over_min', 'share_below_5e-7', 'share_below_1e-7', 'nh3_median', 'nh3_p90', 'nh3_max']
    out['chaotic_device_typical'] = {k: float(np.median([lays[s][k] for s in lays])) for k in keys}
    out['chaotic_device_q1_q3'] = {k: [float(np.quantile([lays[s][k] for s in lays], .25)), float(np.quantile([lays[s][k] for s in lays], .75))] for k in keys}
    out['chaotic_ensemble_median'] = dev(med_pts, [0] * len(med_pts))
    for k in ('nh3_median', 'nh3_p90', 'nh3_max'):
        out['chaotic_ensemble_median'].pop(k)
    # share of chaotic layouts whose worst point over the window beats the regular cell's worst point
    out['layouts_beating_regular_worst_case'] = float(np.mean([lays[s]['max'] < out['regular_device']['max'] for s in lays]))
    out['layouts_beating_regular_p90'] = float(np.mean([lays[s]['p90'] < out['regular_device']['p90'] for s in lays]))
    # single dithers at the nominal operating point, every film: the chaotic cell against the regular one
    single = {}
    for t in FILMS:
        d = D[t]
        c = [x for x in d['chaotic'] if 'error' not in x]
        single[t] = {}
        for md in ('none', 'angle', 'spectral', 'both'):
            v = [x['floor'][md] for x in c]
            single[t][md] = {'regular': d['regular']['floor'][md], 'chaotic_median': float(np.median(v)),
                             'chaotic_q1': float(np.quantile(v, .25)), 'chaotic_q3': float(np.quantile(v, .75)),
                             'layouts_below_5e-7': float(np.mean(np.array(v) < TGT)), 'regular_below_5e-7': bool(d['regular']['floor'][md] < TGT)}
    out['single_dithers'] = single
    for md in ('angle', 'spectral', 'both'):
        rr = np.array([single[t][md]['regular'] for t in FILMS]); cc = np.array([single[t][md]['chaotic_median'] for t in FILMS])
        out.setdefault('single_dither_summary', {})[md] = {'regular_range': [float(rr.min()), float(rr.max())], 'chaotic_median_range': [float(cc.min()), float(cc.max())],
                                                           'ratio_regular_over_chaotic': [float((rr / cc).min()), float((rr / cc).max())],
                                                           'films_regular_below_5e-7': int((rr < TGT).sum()), 'films_chaotic_median_below_5e-7': int((cc < TGT).sum()),
                                                           'layouts_below_5e-7_all_films': float(np.mean([single[t][md]['layouts_below_5e-7'] for t in FILMS]))}
    json.dump(out, open(os.path.join(RES, 'step11_summary.json'), 'w'), indent=1)

    # ---- tables
    print('Nominal operating point (20 pm, exact), per film: static fringe both dithers and NH3 at 1 s\n')
    print('| film | n_eff | Γ | regular fringe | chaotic median [quartiles] | regular / chaotic | NH3 regular | NH3 chaotic median |')
    print('|---|---|---|---|---|---|---|---|')
    for t in FILMS:
        f = out['films'][t]
        n = f['nominal']
        print(f"| {t} nm | {f['n_eff']:.3f} | {f['gamma']:.2f} | {n['regular']:.1e} | {n['chaotic_median']:.1e} [{n['chaotic_q1']:.1e}–{n['chaotic_q3']:.1e}] | "
              f"{n['regular'] / n['chaotic_median']:.1f} | {n['nh3_regular']:.0f} ppb | {n['nh3_chaotic_median']:.0f} ppb |")
    rd, cd, iq = out['regular_device'], out['chaotic_device_typical'], out['chaotic_device_q1_q3']
    print(f"\nOver the window ({out['n_points']} operating points: films 500–700 nm × wavelength dither {out['window_pm'][0]:.1f}–{out['window_pm'][1]:.1f} pm):\n")
    print('| device | fringe median | 90th percentile | worst | worst / best | points < 5e-7 | points < 1e-7 | NH3 1 s median | NH3 90th pct | NH3 worst |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    print(f"| regular cell | {rd['median']:.1e} | {rd['p90']:.1e} | {rd['max']:.1e} | {rd['max_over_min']:.0f}× | {100 * rd['share_below_5e-7']:.0f} % | "
          f"{100 * rd['share_below_1e-7']:.0f} % | {rd['nh3_median']:.0f} ppb | {rd['nh3_p90']:.0f} ppb | {rd['nh3_max']:.0f} ppb |")
    print(f"| chaotic, typical layout (median of 30) | {cd['median']:.1e} | {cd['p90']:.1e} | {cd['max']:.1e} | {cd['max_over_min']:.0f}× | "
          f"{100 * cd['share_below_5e-7']:.0f} % | {100 * cd['share_below_1e-7']:.0f} % | {cd['nh3_median']:.0f} ppb | {cd['nh3_p90']:.0f} ppb | {cd['nh3_max']:.0f} ppb |")
    print(f"| chaotic, layout quartiles | {iq['median'][0]:.1e}–{iq['median'][1]:.1e} | {iq['p90'][0]:.1e}–{iq['p90'][1]:.1e} | {iq['max'][0]:.1e}–{iq['max'][1]:.1e} | "
          f"{iq['max_over_min'][0]:.0f}–{iq['max_over_min'][1]:.0f}× | {100 * iq['share_below_5e-7'][0]:.0f}–{100 * iq['share_below_5e-7'][1]:.0f} % | "
          f"{100 * iq['share_below_1e-7'][0]:.0f}–{100 * iq['share_below_1e-7'][1]:.0f} % | {iq['nh3_median'][0]:.0f}–{iq['nh3_median'][1]:.0f} ppb | "
          f"{iq['nh3_p90'][0]:.0f}–{iq['nh3_p90'][1]:.0f} ppb | {iq['nh3_max'][0]:.0f}–{iq['nh3_max'][1]:.0f} ppb |")
    em = out['chaotic_ensemble_median']
    print(f"| chaotic, ensemble median at each point | {em['median']:.1e} | {em['p90']:.1e} | {em['max']:.1e} | {em['max_over_min']:.0f}× | "
          f"{100 * em['share_below_5e-7']:.0f} % | {100 * em['share_below_1e-7']:.0f} % | | | |")
    print(f"\nChaotic layouts whose worst point beats the regular cell's worst point: {100 * out['layouts_beating_regular_worst_case']:.0f} %; "
          f"whose 90th percentile beats the regular cell's: {100 * out['layouts_beating_regular_p90']:.0f} %.")

    print('\nEach dither alone at its nominal amplitude (±2 mrad angle, ±20 pm wavelength), every film: regular / chaotic median, chaotic layouts below 5e-7\n')
    print('| film | angle only: regular | chaotic [quartiles] | layouts < 5e-7 | wavelength only: regular | chaotic | both: regular | chaotic |')
    print('|---|---|---|---|---|---|---|---|')
    for t in FILMS:
        a_, l_, b_ = single[t]['angle'], single[t]['spectral'], single[t]['both']
        print(f"| {t} nm | {a_['regular']:.1e} | {a_['chaotic_median']:.1e} [{a_['chaotic_q1']:.1e}–{a_['chaotic_q3']:.1e}] | {100 * a_['layouts_below_5e-7']:.0f} % | "
              f"{l_['regular']:.1e} | {l_['chaotic_median']:.1e} | {b_['regular']:.1e} | {b_['chaotic_median']:.1e} |")
    for md, nm in (('angle', 'angle only'), ('spectral', 'wavelength only'), ('both', 'both')):
        x = out['single_dither_summary'][md]
        print(f"{nm}: regular {x['regular_range'][0]:.1e}–{x['regular_range'][1]:.1e}, chaotic median {x['chaotic_median_range'][0]:.1e}–{x['chaotic_median_range'][1]:.1e}, "
              f"regular/chaotic {x['ratio_regular_over_chaotic'][0]:.1f}–{x['ratio_regular_over_chaotic'][1]:.1f}×, films below 5e-7: regular {x['films_regular_below_5e-7']}/{len(FILMS)}, "
              f"chaotic median {x['films_chaotic_median_below_5e-7']}/{len(FILMS)}, chaotic layouts below 5e-7 {100 * x['layouts_below_5e-7_all_films']:.0f} %")

    # ---- figure
    plt = style.setup()
    C = style.COL
    fig, ax = plt.subplots(1, 3, figsize=(16.8 * style.CM, 6.0 * style.CM), constrained_layout=True)
    f6 = out['films'][600]
    m = (dl >= 5) & (dl <= 60)
    a = ax[0]
    a.fill_between(dl[m], np.array(f6['curve_chaotic_q1'])[m], np.array(f6['curve_chaotic_q3'])[m], color=C['c_cmpc'], alpha=0.22, lw=0)
    a.plot(dl[m], np.array(f6['curve_regular'])[m], color=C['r_cmpc'], lw=1.6, label='regular')
    a.plot(dl[m], np.array(f6['curve_chaotic_median'])[m], color=C['c_cmpc'], lw=1.6, label='chaotic')
    a.axvspan(dl[win[0]], dl[win[-1]], color='#000', alpha=0.06, lw=0)
    a.axhline(TGT, color=C['target'], ls='--', lw=1)
    a.set(xscale='log', yscale='log', xlabel='Wavelength dither (pm)', ylabel='Static fringe ΔP/P', ylim=(5e-9, 2e-6))
    a.set_xticks([5, 10, 20, 50]); a.set_xticklabels(['5', '10', '20', '50'])
    a.legend(loc='lower left', handlelength=1.2)
    a.text(0.03, 0.97, '(a) both dithers, 600 nm', transform=a.transAxes, va='top')
    b = ax[1]
    tt = np.array(FILMS)
    nr = [out['films'][t]['nominal']['regular'] for t in FILMS]
    nc = [out['films'][t]['nominal']['chaotic_median'] for t in FILMS]
    b.fill_between(tt, [out['films'][t]['nominal']['chaotic_q1'] for t in FILMS], [out['films'][t]['nominal']['chaotic_q3'] for t in FILMS],
                   color=C['c_cmpc'], alpha=0.22, lw=0)
    b.plot(tt, nr, 'o-', color=C['r_cmpc'], ms=3.5, lw=1.4)
    b.plot(tt, nc, 'o-', color=C['c_cmpc'], ms=3.5, lw=1.4)
    b.axhline(TGT, color=C['target'], ls='--', lw=1)
    b.set(yscale='log', xlabel='SiN$_x$ film thickness (nm)', ylim=(5e-9, 2e-6))
    b.set_xticks([500, 600, 700])
    c = ax[2]
    sa = out['single_dithers']
    c.fill_between(tt, [sa[t]['angle']['chaotic_q1'] for t in FILMS], [sa[t]['angle']['chaotic_q3'] for t in FILMS], color=C['c_cmpc'], alpha=0.22, lw=0)
    c.plot(tt, [sa[t]['angle']['regular'] for t in FILMS], 'o-', color=C['r_cmpc'], ms=3.5, lw=1.4)
    c.plot(tt, [sa[t]['angle']['chaotic_median'] for t in FILMS], 'o-', color=C['c_cmpc'], ms=3.5, lw=1.4)
    c.axhline(TGT, color=C['target'], ls='--', lw=1)
    c.set(yscale='log', xlabel='SiN$_x$ film thickness (nm)', ylim=(3e-8, 2e-5))
    c.set_xticks([500, 600, 700])
    c.text(0.03, 0.97, '(c) angle dither only', transform=c.transAxes, va='top')
    b.text(0.03, 0.97, '(b) both dithers, 20 pm', transform=b.transAxes, va='top')
    os.makedirs(os.path.join(RES, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(RES, 'figs', 'fig_robustness.pdf'))
    fig.savefig(os.path.join(RES, 'figs', 'fig_robustness.png'), dpi=150)


if __name__ == '__main__':
    main()
