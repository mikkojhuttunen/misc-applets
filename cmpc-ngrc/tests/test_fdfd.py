import numpy as np
import pytest

from ngrc.fdfd import WaveCell, metal_eps  # noqa: E402  (sets PYPARDISO_MKL_RT)

pytest.importorskip("pypardiso")
from ngrc.field import field_correlation
from ngrc.shapes import Perturbation, Shape


def test_metal_eps_reflectance():
    n0, R = 1.8, 0.97
    e = metal_eps(n0, R)
    r = (n0 - np.sqrt(e)) / (n0 + np.sqrt(e))
    assert abs(abs(r) ** 2 - R) < 1e-6


@pytest.fixture(scope="module")
def small_cell():
    return WaveCell(radius=6e-6, facets=12, port_width=1.5e-6, ppw=8).setup()


def test_weak_dot_matches_born(small_cell):
    c = small_cell
    F0 = c.fields(None)
    p = Perturbation([Shape(0, 0, 1.5e-6, 1e-4, 0.2e-6, a=[0, 0.06, 0.03])])
    F, Fb = c.fields(p), c.fields(p, born=True)
    for k in F0:
        dE, dB = F[k] - F0[k], Fb[k] - F0[k]
        assert np.linalg.norm(dE) > 0
        assert field_correlation(dE, dB) > 0.999
    assert all(flag == 0 for _, flag in c.last_info.values())


def test_gmres_solution_matches_direct_refactorisation(small_cell):
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    c = small_cell
    p = Perturbation([Shape(0, 0, 1.5e-6, 1e-2, 0.2e-6, a=[0, 0.06, 0.03])])
    F = c.fields(p, tol=1e-10)
    X, Y = np.meshgrid(c.x, c.x, indexing="xy")
    dn = p.value(X, Y) * (c.boundary_distance(X, Y) < 0)
    A = c.A0 + sp.diags((c.k0 ** 2 * ((c.n_eff + dn) ** 2 - c.n_eff ** 2)).ravel())
    i = c.inputs[0]
    x = sla.spsolve(A.tocsc(), c._sources[i])
    for o, (K, P) in c.detector_operators().items():
        ref = K @ (P @ x)
        assert np.linalg.norm(F[(i, o)] - ref) < 1e-6 * np.linalg.norm(ref)
