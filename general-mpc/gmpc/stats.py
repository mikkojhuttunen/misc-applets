"""Statistics shared by the planar and the Herriott-type tracers."""
from __future__ import annotations

import numpy as np


def effective_path(chords, R):
    """(geometric path, absorption-weighted path Σ I_j ℓ_j, intensity left) of one ray: chord j carries the product of
    the reflectances of the hits before it. R: scalar or one value per reflection (len(chords) values)."""
    ch = np.asarray(chords, float)
    Rj = np.broadcast_to(np.asarray(R, float), ch.shape)
    I = np.concatenate([[1.0], np.cumprod(Rj)])
    return float(ch.sum()), float((I[:-1] * ch).sum()), float(I[-1])


def lyapunov_fit(sep, scale, lo_rel=1e-10, hi_rel=1e-2, min_points=6):
    """Least-squares slope of ln(separation) against hit number over the stretch between lo_rel·scale and hi_rel·scale.
    Returns dict(lam = exponent per hit, intercept, first, last) or None. Exponential growth (lam clearly > 0 over a
    long stretch) means chaos; regular and pseudo-integrable cells grow linearly (lam -> small and falling)."""
    sep = np.asarray(sep, float)
    j = np.nonzero((sep > lo_rel * scale) & (sep < hi_rel * scale))[0]
    if j.size < min_points:
        return None
    x, y = j + 1.0, np.log(sep[j])
    A = np.vstack([x, np.ones_like(x)]).T
    (lam, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(lam=float(lam), intercept=float(c), first=int(x[0]), last=int(x[-1]))


def min_pairwise_distance(P):
    """Smallest distance between distinct points of an (n, d) array (O(n²), fine for spot patterns)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.inf
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return float(D.min())
