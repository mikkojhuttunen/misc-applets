import numpy as np

from ngrc.features import ngrc_features, stack_intensities
from ngrc.readout import evaluate_ridge, r2, ridge_fit


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    W = rng.normal(size=(20, 3))
    Y = X @ W + 0.01 * rng.normal(size=(400, 3))
    res = evaluate_ridge(X, Y)
    assert np.all(res["r2"] > 0.99)


def test_quadratic_features_make_a_quadratic_target_linear():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] * X[:, 3] + 0.5 * X[:, 2] ** 2)[:, None]
    lin = evaluate_ridge(X, y)["r2"][0]
    quad = evaluate_ridge(ngrc_features(X, constant=False), y)["r2"][0]
    assert lin < 0.2 and quad > 0.99
    assert ngrc_features(X).shape[1] == 1 + 6 + 21


def test_stack_relative():
    f = {(0, 1): np.array([1.0, 2.0]), (0, 2): np.array([1j, 0.0])}
    ref = {(0, 1): np.array([1.0, 1.0]), (0, 2): np.array([1.0, 1.0])}
    assert np.allclose(stack_intensities(f, reference=ref), [0, 3, 0, -1])
