import numpy as np
import pytest

from engines.slab_waveguide import engine as sw


def test_symmetric_slab_second_mode_cutoff_at_V_equal_pi():
    # analytic: symmetric slab, TE1 cut-off at V = k d sqrt(n1^2 - n2^2) = pi
    lam, n1, n2 = 1.55e-6, 1.50, 1.45
    d_cut = lam / (2 * np.sqrt(n1**2 - n2**2))
    assert sw.slab_neff(lam, n2, n1, n2, 0.99 * d_cut)["n_modes"] == 1
    assert sw.slab_neff(lam, n2, n1, n2, 1.01 * d_cut)["n_modes"] == 2


def test_symmetric_slab_has_no_te0_cutoff():
    assert sw.slab_neff(1.55e-6, 1.45, 1.46, 1.45, 50e-9)["guided"]


def test_te0_satisfies_its_dispersion_relation():
    lam, ns, nf, nc, d = 1.55e-6, 1.444, 2.138, 1.0, 0.6e-6
    ne = sw.slab_neff(lam, ns, nf, nc, d)["neff"]
    assert abs(sw._dispersion(ne, 2 * np.pi / lam, ns, nf, nc, d, "TE", 0)) < 1e-9


def test_te_above_tm():
    # TM has the stronger boundary factor, so its neff is lower for the same order
    a = sw.slab_neff(1.55e-6, 1.444, 2.138, 1.0, 0.6e-6, "TE")["neff"]
    b = sw.slab_neff(1.55e-6, 1.444, 2.138, 1.0, 0.6e-6, "TM")["neff"]
    assert a > b


def test_multilayer_reduces_to_three_layer():
    # independent route: transfer-matrix multilayer solver against the analytic three-layer relation
    lam, ns, nf, nc, d = 1.55e-6, 1.444, 2.12, 1.0, 0.5e-6
    for pol in ("TE", "TM"):
        ml = sw.multilayer_neff(lam, [ns, nf, nc], [d], pol)["neff"][0]
        assert ml == pytest.approx(sw.neff_three_layer(lam, ns, nf, nc, d, pol), abs=1e-6)


def test_multilayer_fractions_sum_below_one():
    r = sw.multilayer_neff(1.55e-6, [1.444, 2.12, 2.0, 1.0], [0.4e-6, 0.1e-6])
    fr = r["layer_fraction"][0]
    assert 0.5 < fr[0] < 1 and fr.sum() < 1
