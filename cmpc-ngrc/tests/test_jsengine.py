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
