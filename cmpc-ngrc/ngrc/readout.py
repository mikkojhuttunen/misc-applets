"""Readouts: ridge regression (the NGRC readout, closed form with k-fold CV of the penalty), linear / RBF support
vector regression and classification (scikit-learn), and metrics."""
from __future__ import annotations

import numpy as np


def standardize(Xtr, Xte, floor=1e-3):
    """z-score with the training statistics; columns with sd < floor · mean sd (e.g. image pixels that are
    almost always zero) are scaled by that floor instead, so they cannot blow up on the test set."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.maximum(sd, floor * (sd.mean() or 1.0))
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, Y, alpha):
    """W minimising |XW − Y|² + α|W|² (X standardised, Y centred inside); returns (W, b)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    n, d = Xc.shape
    if d <= n:
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    else:
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(n), Yc)
    return W, ym - xm @ W


def r2(Y, P):
    Y, P = np.atleast_2d(Y.T).T, np.atleast_2d(P.T).T
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - P) ** 2).sum(0) / np.where(ss > 0, ss, 1)


def ridge_cv(X, Y, alphas=np.logspace(-3, 4, 15), k=5, seed=0):
    """Best α by k-fold CV (mean R² over targets)."""
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    scores = []
    for a in alphas:
        s = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            W, b = ridge_fit(X[tr], Y[tr], a)
            s.append(np.mean(r2(Y[f], X[f] @ W + b)))
        scores.append(np.mean(s))
    return alphas[int(np.argmax(scores))], np.array(scores)


def split(n, test=0.25, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    nt = int(round(test * n))
    return idx[nt:], idx[:nt]


def evaluate_ridge(X, Y, test=0.25, seed=0, alphas=np.logspace(-3, 4, 15)):
    tr, te = split(len(X), test, seed)
    Xtr, Xte = standardize(X[tr], X[te])
    a, _ = ridge_cv(Xtr, Y[tr], alphas, seed=seed)
    W, b = ridge_fit(Xtr, Y[tr], a)
    return dict(alpha=a, r2=r2(Y[te], Xte @ W + b), pred=Xte @ W + b, test=te)


def evaluate_svr(X, Y, kernel="linear", test=0.25, seed=0, C=None, epsilon=None):
    """One SVR per target column; C and ε chosen by a small grid with 3-fold CV."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR, LinearSVR
    tr, te = split(len(X), test, seed)
    Xtr, Xte = standardize(X[tr], X[te])
    out_r2, preds = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        ys = y[tr].std() or 1
        if kernel == "linear":
            est = LinearSVR(dual="auto", max_iter=20000, random_state=seed)
            grid = {"C": [0.01, 0.1, 1.0] if C is None else [C], "epsilon": [0.0, 0.1] if epsilon is None else [epsilon]}
        else:
            est = SVR(kernel=kernel)
            grid = {"C": [1.0, 10.0, 100.0] if C is None else [C], "gamma": ["scale"],
                    "epsilon": [0.05, 0.2] if epsilon is None else [epsilon]}
        g = GridSearchCV(est, grid, cv=3).fit(Xtr, (y[tr] - y[tr].mean()) / ys)
        p = g.predict(Xte) * ys + y[tr].mean()
        preds.append(p)
        out_r2.append(r2(y[te][:, None], p[:, None])[0])
    return dict(r2=np.array(out_r2), pred=np.array(preds).T, test=te)


def evaluate_svc(X, y, kernel="linear", test=0.25, seed=0):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC, LinearSVC
    tr, te = split(len(X), test, seed)
    Xtr, Xte = standardize(X[tr], X[te])
    if kernel == "linear":
        g = GridSearchCV(LinearSVC(dual="auto", max_iter=20000), {"C": [0.01, 0.1, 1.0]}, cv=3)
    else:
        g = GridSearchCV(SVC(kernel=kernel), {"C": [1.0, 10.0, 100.0]}, cv=3)
    g.fit(Xtr, y[tr])
    p = g.predict(Xte)
    return dict(accuracy=float(np.mean(p == y[te])), pred=p, test=te,
                chance=float(np.max(np.bincount(y[te].astype(int))) / len(te)))


# ---------------- kernel ridge (KRR) with optional position kernel ----------------

def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


def krr_fit_predict(Ktr, Ytr, Kte, alpha):
    ym = Ytr.mean(0)
    A = np.linalg.solve(Ktr + alpha * np.eye(len(Ktr)), Ytr - ym)
    return Kte @ A + ym


def _kernels(Xa, Xb, kind, gamma, Pa=None, Pb=None, ell=None):
    d = Xa.shape[1]
    if kind == "linear":
        K = Xa @ Xb.T / d
    elif kind == "rbf":
        K = np.exp(-gamma * sqdist(Xa, Xb) / d)
    elif kind == "poly2":                           # = linear + quadratic monomials (NGRC), kernel form
        L = Xa @ Xb.T / d
        K = (1 + L) ** 2
    else:
        raise ValueError(kind)
    if ell is not None:
        K = K * np.exp(-sqdist(Pa, Pb) / (2 * ell * ell))
    return K


def evaluate_krr(X, Y, kind="linear", pos=None, test=0.25, seed=0, alphas=np.logspace(-4, 2, 7),
                 gammas=(0.1, 0.3, 1.0), ells=None, val=0.2, train_idx=None, test_idx=None):
    """Kernel ridge regression. kind: "linear", "poly2" or "rbf" kernel on the standardised features; with
    pos (n, 2), the kernel is multiplied by exp(−|p − p'|²/2ℓ²): a readout whose weights vary smoothly with the
    dot position (position known or estimated). Hyper-parameters by a validation split of the training set."""
    Y = np.atleast_2d(np.asarray(Y, float).T).T
    tr, te = split(len(X), test, seed) if train_idx is None else (np.asarray(train_idx), np.asarray(test_idx))
    Xtr, Xte = standardize(X[tr], X[te])
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(tr))
    nv = max(2, int(round(val * len(tr))))
    iv, it = perm[:nv], perm[nv:]
    P = None if pos is None else np.asarray(pos, float)
    Ptr = None if P is None else P[tr]
    ells = [None] if P is None else (ells if ells is not None else np.std(Ptr, 0).mean() * np.array([0.1, 0.2, 0.4, 0.8]))
    gl = gammas if kind == "rbf" else [None]
    best = (-np.inf, None)
    for g in gl:
        for ell in ells:
            Kt = _kernels(Xtr[it], Xtr[it], kind, g, None if P is None else Ptr[it], None if P is None else Ptr[it], ell)
            Kv = _kernels(Xtr[iv], Xtr[it], kind, g, None if P is None else Ptr[iv], None if P is None else Ptr[it], ell)
            for a in alphas:
                sc = np.mean(r2(Y[tr][iv], krr_fit_predict(Kt, Y[tr][it], Kv, a)))
                if sc > best[0]:
                    best = (sc, (g, ell, a))
    g, ell, a = best[1]
    Ptest = None if P is None else P[te]
    K = _kernels(Xtr, Xtr, kind, g, Ptr, Ptr, ell)
    Kte = _kernels(Xte, Xtr, kind, g, Ptest, Ptr, ell)
    pred = krr_fit_predict(K, Y[tr], Kte, a)
    return dict(r2=r2(Y[te], pred), pred=pred, test=te, train=tr, alpha=a, gamma=g, ell=ell)
