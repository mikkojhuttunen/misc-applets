"""mirror_design — hole-free in-plane mirror for the TE (p-polarised) slab mode.

Why a hole exists: at the Brewster angle every interface of a stack made of only TWO materials (membrane / air) has r_p = 0,
whatever the layer thicknesses, so the stack is transparent there (checked: R < 1e-30 for 300 random stacks). Quarter-wave
or chirped designs cannot cure it. A THIRD index breaks the coincidence of the Brewster angles of the individual interfaces.
In a membrane platform the third index is a partly etched (thinned) region of the same membrane, with its own n_eff.

Stack (from the cell side):  [ air gap g | thinned membrane c | membrane tooth a ] x N  into air.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

import cmpc_sim as cs

SG = np.linspace(0.002, 0.998, 500)


def _layers(nA, nC, g, c, a, N):
    return [(1.0, g), (nC, c), (nA, a)] * N


def R_tri(lam, s, nA, nC, g, c, a, N, bounce_loss=0.0):
    return cs.stack_R_oblique(lam, s, nA, _layers(nA, nC, g, c, a, N), 1.0, "p") * (1 - bounce_loss)


def design(lam, nA, nC, N=12, min_feat=250e-9, max_feat=3e-6, seed=1, maxiter=80):
    def loss(x):
        L = 1 - cs.stack_R_oblique(lam, SG, nA, _layers(nA, nC, *x, N), 1.0, "p")
        return L.mean() + 0.1 * L.max()
    r = differential_evolution(loss, [(min_feat, max_feat)] * 3, seed=seed, maxiter=maxiter, popsize=14, tol=1e-7, polish=False)
    g, c, a = r.x
    L = 1 - R_tri(lam, SG, nA, nC, g, c, a, N)
    return dict(g=g, c=c, a=a, nC=nC, nA=nA, N=N, mean_loss=float(L.mean()), max_loss=float(L.max()))


def tolerance(lam, d, sigma=20e-9, n=200, seed=0):
    """Monte-Carlo over independent Gaussian thickness errors of every layer (etch/lithography), returns <1-R> samples."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lay = [(m, max(t + rng.normal(0, sigma), 20e-9)) for m, t in _layers(d["nA"], d["nC"], d["g"], d["c"], d["a"], d["N"])]
        out.append(float((1 - cs.stack_R_oblique(lam, SG, d["nA"], lay, 1.0, "p")).mean()))
    return np.array(out)


def R_function(lam, d, bounce_loss=3e-4):
    return lambda s: R_tri(lam, s, d["nA"], d["nC"], d["g"], d["c"], d["a"], d["N"], bounce_loss)
