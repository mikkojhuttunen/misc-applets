"""Rank candidate absorption lines of a target gas by selectivity against the rest of the mixture.

    python select_lines.py NO 1880 1920         # uses hitran_lines*.json if present, else the illustrative list

score = target absorption at the line centre / absorption of everything else at that frequency (same mixture, 1 atm).
Run it on real HITRAN data (python fetch_hitran.py mir) before choosing a line.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import sys

import numpy as np

import gas_spectra as gs


def rank(target, numin, numax, mix=None, p=gs.P_REF, T=gs.T_REF, x_target=None, top=8):
    lines, src = gs.load_lines()
    mix = dict(gs.MIDIR_BREATH if mix is None else mix)
    x_t = mix.get(target, 25e-9) if x_target is None else x_target
    nu0, S, gam, nair, el = lines[target]
    sel = np.nonzero((nu0 >= numin) & (nu0 <= numax))[0]
    rows = []
    others = {k: v for k, v in mix.items() if k != target}
    for k in sel:
        one = tuple(np.array([c[k]]) for c in lines[target])
        a_t = gs.alpha_species(np.array([nu0[k]]), one, gs.SPECIES[target][1], gs.SPECIES[target][2], x_t, p, T)[0]
        a_o = sum(gs.alpha_species(np.array([nu0[k]]), lines[sp], gs.SPECIES[sp][1], gs.SPECIES[sp][2], x, p, T)[0] for sp, x in others.items() if sp in lines)
        rows.append((nu0[k], 1e7 / nu0[k], S[k], a_t, a_o, a_t / max(a_o, 1e-30)))
    rows.sort(key=lambda r: -r[5])
    return rows[:top], src


if __name__ == "__main__":
    tgt, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    rows, src = rank(tgt, lo, hi)
    print("line data:", src)
    print("nu (cm-1)  lambda (nm)   S (cm/molec)   alpha_target (1/cm)  alpha_others (1/cm)  selectivity")
    for r in rows:
        print("%9.3f  %10.1f   %.2e      %.2e            %.2e           %.2g" % r)
