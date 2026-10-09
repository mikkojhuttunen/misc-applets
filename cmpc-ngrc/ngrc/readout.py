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
