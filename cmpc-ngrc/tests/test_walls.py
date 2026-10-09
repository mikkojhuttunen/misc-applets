import sys
from pathlib import Path

import numpy as np

from ngrc.field import detector_field, field_correlation
from ngrc.geometry import CircularCell, WallCell, ports_at, stadium, wall_ports
from ngrc.perturbative import PhaseScreenModel
from ngrc.rays import Source, trace
from ngrc.shapes import Perturbation, Shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "general-mpc"))
from gmpc.planar import circle_cell  # noqa: E402

ANG = (0, 67, 151, 238)
KW = dict(width=20e-6, launch=0.35, fan=0.4)


def test_gmpc_circle_wall_reproduces_the_analytic_circle():
    c1 = CircularCell(ports=ports_at(ANG, inputs=(0,), **KW))
    w = circle_cell(1e-3)                            # starts at angle −π/4
    s = [(np.deg2rad(a) + np.pi / 4) % (2 * np.pi) * 1e-3 for a in ANG]
    c2 = WallCell(wall=w, ports=wall_ports(w, s, inputs=(0,), fractions=False, **KW))
    p = Perturbation([Shape(2e-4, 1e-4, 80e-6, 1e-3, 3e-6, a=[0, 0.05])])
    e1 = trace(c1, Source(0, n_pos=3, n_ang=21), p, "curved")
    e2 = trace(c2, Source(0, n_pos=3, n_ang=21), p, "curved")
    assert len(e1.L) == len(e2.L)
    for o in range(4):
        assert field_correlation(detector_field(c1, e1, o), detector_field(c2, e2, o)) > 1 - 1e-9


def test_stadium_traces_without_leaks_and_fills_the_centre():
    st = stadium(0.5e-3, 1.0e-3)
    cell = WallCell(wall=st, ports=wall_ports(st, [0.1, 0.33, 0.55, 0.8], inputs=(0,), **KW))
    ex = trace(cell, Source(0, n_pos=3, n_ang=41))
    assert ex.lost.get("leaked", 0) == 0
    assert ex.port_power(4).sum() > 0.5
    I = np.abs(detector_field(cell, ex, 2)) ** 2
    assert 0.5 < I.std() / I.mean() < 1.5
    # a dot at the centre is seen in the stadium; in the circle it sits in the caustic disk (R_c sin χ ≥ 150 µm)
    dot = Perturbation([Shape(0.0, 0.0, 60e-6, 1e-3, 3e-6)])
    corr = []
    for c in (cell, CircularCell(ports=ports_at(ANG, inputs=(0,), **KW))):
        m = PhaseScreenModel(c, [Source(0, n_pos=3, n_ang=41)])
        F0, F1 = m.fields(None), m.fields(dot)
        corr.append(np.mean([field_correlation(F0[k], F1[k]) for k in F0]))
    assert corr[0] < 0.999 and corr[1] > 1 - 1e-12
