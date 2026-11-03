import numpy as np
import pytest

from swr.orientation import crystal_to_sim, is_axis_permutation, rotate_tensor

EPS = np.array([4.8, 4.8, 4.5])  # (eo, eo, ee), LN-like
TOL = 1e-9


def test_rotation_is_proper_orthogonal():
    for cut, ref in (("z", "x"), ("x", "y"), ("y", "z"), ([1, 1, 0], "z")):
        for th in (0, 17, 45, 90, 133):
            r = crystal_to_sim(cut, th, ref)
            assert np.allclose(r @ r.T, np.eye(3))
            assert np.linalg.det(r) == pytest.approx(1.0)


def test_zcut_is_in_plane_isotropic():
    ref = rotate_tensor(EPS, crystal_to_sim("z", 0, "x"))
    assert np.allclose(ref, np.diag(EPS))
    for th in (10, 45, 77, 180):
        assert np.allclose(rotate_tensor(EPS, crystal_to_sim("z", th, "x")), ref)


def test_xcut_along_y_puts_optic_axis_lateral():
    r = crystal_to_sim("x", 0, "y")
    assert is_axis_permutation(r, TOL)
    assert np.allclose(np.diag(rotate_tensor(EPS, r)), [EPS[0], EPS[2], EPS[0]])


def test_xcut_along_z_puts_optic_axis_along_propagation():
    r = crystal_to_sim("x", 90, "y")
    assert is_axis_permutation(r, TOL)
    assert np.allclose(np.diag(rotate_tensor(EPS, r)), [EPS[2], EPS[0], EPS[0]])


def test_xcut_45_has_xy_coupling_and_keeps_eigenvalues():
    r = crystal_to_sim("x", 45, "y")
    assert not is_axis_permutation(r, TOL)
    t = rotate_tensor(EPS, r)
    assert np.allclose(t, t.T)
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), np.sort(EPS))
    assert np.trace(t) == pytest.approx(EPS.sum())
    assert abs(t[0, 1]) == pytest.approx(abs(EPS[2] - EPS[0]) / 2)
    assert t[0, 2] == pytest.approx(0) and t[1, 2] == pytest.approx(0)


def test_dispersive_batch_shape():
    eps = np.stack([EPS, EPS * 1.01])
    assert rotate_tensor(eps, crystal_to_sim("x", 30, "y")).shape == (2, 3, 3)


def test_reference_parallel_to_cut_rejected():
    with pytest.raises(ValueError):
        crystal_to_sim("z", 0, "z")
