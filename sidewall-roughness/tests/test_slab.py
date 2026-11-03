import numpy as np
import pytest
from scipy.optimize import brentq

from swr.slab import Layer, decay_length, slab_modes, symmetric_slab

LAM = 1.55
K0 = 2 * np.pi / LAM


def analytic_even(n1, n2, d, ratio):
    """Fundamental even mode of a symmetric slab: tan(kappa d/2) = ratio * gamma / kappa."""
    v = K0 * np.sqrt(n1**2 - n2**2)

    def f(kap):
        return np.tan(kap * d / 2) - ratio * np.sqrt(v**2 - kap**2) / kap

    kap = brentq(f, 1e-9 * v, min(np.pi / d * (1 - 1e-9), v))
    return np.sqrt(n1**2 - (kap / K0) ** 2)


@pytest.mark.parametrize("d", [0.3, 0.5, 1.0])
def test_te_and_tm_match_analytic_isotropic(d):
    n1, n2 = 2.21, 1.0
    layers, below, above = symmetric_slab([n1**2] * 3, [n2**2] * 3, d)
    te = slab_modes(layers, below, above, LAM, "TE", 4000)
    tm = slab_modes(layers, below, above, LAM, "TM", 4000)
    assert te[0] == pytest.approx(analytic_even(n1, n2, d, 1.0), abs=1e-8)
    assert tm[0] == pytest.approx(analytic_even(n1, n2, d, n1**2 / n2**2), abs=1e-8)
    assert te[0] > tm[0]


def test_te_sees_only_eps_y():
    d = 0.4
    a = slab_modes(*symmetric_slab([4.0, 4.8, 4.5], [1.0] * 3, d), LAM, "TE", 4000)
    b = slab_modes(*symmetric_slab([4.8] * 3, [1.0] * 3, d), LAM, "TE", 4000)
    assert a[0] == pytest.approx(b[0], abs=1e-10)


def test_multilayer_split_layer_is_same_mode():
    eps = (4.8, 4.8, 4.5)
    one = slab_modes([Layer(eps, 0.6)], (1.0,) * 3, (1.0,) * 3, 0.633, "TM", 4000)
    two = slab_modes([Layer(eps, 0.25), Layer(eps, 0.35)], (1.0,) * 3, (1.0,) * 3, 0.633, "TM", 4000)
    assert np.allclose(one, two, atol=1e-9)


def test_mode_count_and_decay():
    layers, below, above = symmetric_slab([2.21**2] * 3, [1.0] * 3, 1.0)
    te = slab_modes(layers, below, above, 0.633, "TE", 4000)
    v = np.pi * 1.0 / 0.633 * np.sqrt(2.21**2 - 1)
    assert len(te) == int(np.ceil(2 * v / np.pi))
    assert decay_length((1.0,) * 3, te[0], 0.633, "TE") == pytest.approx(
        0.633 / (2 * np.pi * np.sqrt(te[0] ** 2 - 1)))


def test_graded_coords_keep_interfaces_centred_and_grow():
    from swr.modes import graded_coords
    c = graded_coords(-3.0, 4.0, -0.5, 1.0, [0.0, 0.5], 0.02, 0.08, 1.15)
    assert c[0] == -3.0 and c[-1] == 4.0 and np.all(np.diff(c) > 0)
    centres = 0.5 * (c[1:] + c[:-1])
    for z in (0.0, 0.5):
        assert np.min(np.abs(centres - z)) < 1e-12
    d = np.diff(c)
    assert d.max() <= 0.08 + 1e-12 and np.all(d[(centres > -0.5) & (centres < 1.0)] <= 0.02 + 1e-12)
