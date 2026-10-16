import numpy as np
import pytest

from ngrc.field import detector_field, field_correlation
from ngrc.geometry import CircularCell, ports_at
from ngrc.jsengine import available, node_fields
from ngrc.rays import Source, trace
from ngrc.shapes import Perturbation, Shape

pytestmark = pytest.mark.skipif(not available(), reason="node not installed")


def test_node_curved_fields_match_python():
    cell = CircularCell(ports=ports_at([0, 67, 151, 238], inputs=(0, 1), width=20e-6, launch=0.35, fan=0.5))
    p = Perturbation([Shape(2e-4, 1e-4, 80e-6, 1e-3, 3e-6, a=[0, 0.05, 0.02])])
    F = node_fields(cell, [None, p], n_pos=2, n_ang=15, workers=2)
    for s, pert in enumerate((None, p)):
        for i in cell.inputs:
            ex = trace(cell, Source(i, n_pos=2, n_ang=15), pert, "curved")
            for o in cell.outputs:
                assert field_correlation(F[s][(i, o)], detector_field(cell, ex, o)) > 1 - 1e-9


@pytest.mark.parametrize("kind", ["polygon", "stadium"])
def test_node_wall_cells_match_python(kind):
    from ngrc.geometry import WallCell, polygon, ports_by_angle, stadium
    from ngrc.perturbative import PhaseScreenModel
    w = polygon(1e-3, 24) if kind == "polygon" else stadium(0.95e-3, 0.1e-3)
    cell = WallCell(wall=w, n_eff=1.8, wavelength=1.55e-6, reflectance=0.97,
                    ports=ports_by_angle(w, [0, 67, 151, 238], inputs=(0, 1), width=20e-6, launch=0.05, fan=0.2))
    assert isinstance(CircularCell.from_dict(cell.to_dict()), WallCell)
    p = Perturbation([Shape(0, 0, 80e-6, 3e-5, 3e-6, a=[0, 0.06, 0.03], b=[0, 0, 0, 0.02])])
    m = PhaseScreenModel(cell, [Source(i, n_pos=None, n_ang=200) for i in cell.inputs], max_bounces=100, amp_min=0.05)
    F = node_fields(cell, [None, p], mode="phase", n_ang=200, max_bounces=100, amp_min=0.05, workers=2)
    for s, pert in enumerate((None, p)):
        Fp = m.fields(pert)
        for k in Fp:
            assert field_correlation(F[s][k], Fp[k]) > 1 - 1e-6
            assert abs(np.linalg.norm(F[s][k]) / np.linalg.norm(Fp[k]) - 1) < 1e-4
