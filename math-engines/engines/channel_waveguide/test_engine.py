import numpy as np
import pytest

from engines.channel_waveguide import engine as cw
from engines.slab_waveguide import engine as slab

LAM = 1.55e-6


def test_eim_without_strip_reduces_to_the_slab():
    # strip index equal to the superstrate: no lateral confinement, EIM = three-layer slab
    r = cw.eim_modes("strip", 3e-6, 0.6e-6, 1.0, 1.6, 1.45, LAM, "TE", strip_height=0.4e-6, n_strip=1.0)
    assert not r["laterally_confined"]
    assert r["neff"][0] == pytest.approx(slab.neff_three_layer(LAM, 1.45, 1.6, 1.0, 0.6e-6, "TE"), abs=1e-9)


def test_wide_ridge_approaches_the_slab_from_below():
    # lateral confinement always lowers neff; a 40 µm ridge is within 1e-3 of the slab value
    ns = slab.neff_three_layer(LAM, 1.45, 1.6, 1.45, 0.6e-6, "TE")
    ne = cw.eim_modes("ridge", 40e-6, 0.6e-6, 1.45, 1.6, 1.45, LAM, "TE")["neff"][0]
    assert ns - 1e-3 < ne < ns


def test_eim_uses_the_other_polarisation_laterally():
    # by hand: quasi-TE = TE vertical slab, then a TM lateral slab (N_out | N_in | N_out)
    r = cw.eim_modes("ridge", 1.2e-6, 0.4e-6, 1.444, 1.996, 1.444, LAM, "TE")
    n_in = slab.neff_three_layer(LAM, 1.444, 1.996, 1.444, 0.4e-6, "TE")
    assert r["n_in"][0] == pytest.approx(n_in, abs=1e-9)
    assert r["neff"][0] == pytest.approx(slab.neff_three_layer(LAM, 1.444, n_in, 1.444, 1.2e-6, "TM"), abs=1e-9)


def test_strip_loaded_leaky_flag():
    # higher vertical orders below the outside slab fundamental are flagged leaky
    r = cw.eim_modes("strip", 2e-6, 0.3e-6, 1.0, 2.138, 1.444, 1.0e-6, "TE", strip_height=1.2e-6, n_strip=1.65)
    assert not r["leaky"][0]
    assert np.all(r["neff"][r["leaky"]] < r["outside_slab_neff"])


def test_fd_wide_ridge_matches_the_slab():
    # independent route: 2D FD on a 30 µm wide ridge against the analytic slab (discretisation ~1e-3)
    ns = slab.neff_three_layer(LAM, 1.45, 1.6, 1.45, 0.6e-6, "TE")
    r = cw.fd_fundamental("ridge", 30e-6, 0.6e-6, 1.45, 1.6, 1.45, LAM, "TE", cell=50e-9, margin=1.5e-6)
    assert r["neff"] == pytest.approx(ns, abs=2e-3)


def test_fd_and_eim_agree_for_a_silicon_nitride_ridge():
    e = cw.eim_modes("ridge", 1.2e-6, 0.4e-6, 1.444, 1.996, 1.444, LAM, "TE")["neff"][0]
    f = cw.fd_fundamental("ridge", 1.2e-6, 0.4e-6, 1.444, 1.996, 1.444, LAM, "TE", cell=25e-9, margin=1.2e-6)
    # EIM ignores the field change at the ridge corners and overestimates neff slightly
    assert e - 1e-2 < f["neff"] < e
    assert f["share_sub"] + f["share_film"] + f["share_strip"] + f["share_sup"] == pytest.approx(1.0)


def test_fd_te_above_tm_for_a_wide_flat_core():
    a = cw.fd_fundamental("ridge", 2e-6, 0.3e-6, 1.444, 1.996, 1.444, LAM, "TE", cell=30e-9, margin=1e-6)["neff"]
    b = cw.fd_fundamental("ridge", 2e-6, 0.3e-6, 1.444, 1.996, 1.444, LAM, "TM", cell=30e-9, margin=1e-6)["neff"]
    assert a > b


def test_region_codes():
    r = cw.region_at("strip", [0, 0, 0, 2e-6], [-1e-7, 1e-7, 0.5e-6, 0.5e-6], 2e-6, 0.3e-6, 0.4e-6)
    assert list(r) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="geometry"):
        cw.eim_modes("slot", 1e-6, 0.2e-6, 1.0, 2.0, 1.44, LAM)
