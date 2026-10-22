"""gmpc re-exports the math-engines (planar_cell, herriott_cell); the physics tests live there
(math-engines/engines/planar_cell/test_engine.py, math-engines/engines/herriott_cell/test_engine.py)."""
import numpy as np
import pytest

from gmpc import herriott as H
from gmpc import planar as P
from gmpc import stats as S
from gmpc import trace2d as T


def test_gmpc_names_are_the_engine_objects():
    from engines.herriott_cell import engine as he
    from engines.planar_cell import engine as pe
    assert P.Cell2D is pe.Cell2D and T.trace_rays is pe.trace_rays and S.lyapunov_fit is pe.lyapunov_fit
    assert H.trace3d is he.trace3d and H.HerriottLaunch is he.HerriottLaunch


def test_planar_and_herriott_smoke():
    st = P.stadium_cell(5e-3, 5e-3, cap_facets=16, straight_segments=4)
    tab = T.trace_rays(P.perturb(st, tilt_rms=1e-3, seed=2, select="cap"), 150e-6, 200, 500)
    assert 0 <= T.evaluate(tab, lambda s: np.full_like(s, 0.999)).T_det <= 1
    c = H.herriott_cell(R=0.5, N=30, M=7, A=0.012)
    r = H.reentrance(H.trace3d(c["mirrors"], c["p0"], c["d0"], 200))
    assert r["exit"] == "hole" and r["n_hits"] == 30
    hp = P.herriott_planar_cell(20e-3, 24, 5, 1e-3)
    assert T.herriott_planar_trace(hp)["n_hits"] == 24
    assert S.effective_path([1.0, 1.0], 0.5) == pytest.approx((2.0, 1.5, 0.25))
