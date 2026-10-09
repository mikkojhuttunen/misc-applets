"""Reference outputs of the Python engine for the JS port (web/ngrc-engine.js) → tools/vectors.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngrc import harmonics as H                                  # noqa: E402
from ngrc.field import detector_field                            # noqa: E402
from ngrc.geometry import CircularCell, ports_at                 # noqa: E402
from ngrc.perturbative import PhaseScreenModel                   # noqa: E402
from ngrc.rays import Source, trace                              # noqa: E402
from ngrc.shapes import Perturbation, Shape                      # noqa: E402

cell = CircularCell(ports=ports_at([0, 67, 151, 238], inputs=(0, 1), width=20e-6, launch=0.35, fan=0.5))
shapes = [Shape(2e-4, 1e-4, 80e-6, 1e-3, 3e-6, a=[0, 0.05, 0.02]), Shape(-3e-4, -2e-4, 60e-6, 1e-3, 3e-6, b=[0, 0.03, 0, 0.01])]
pert = Perturbation(shapes)
src = dict(n_pos=3, n_ang=21)
c = lambda z: [float(np.real(z)), float(np.imag(z))]

out = dict(cell=cell.to_dict(), shapes=[s.to_dict() for s in shapes], source=src, modes={})
for mode in ("none", "straight", "curved"):
    ex = trace(cell, Source(1, **src), pert, mode)
    o = np.argsort(ex.ray)
    out["modes"][mode] = dict(
        exits=[dict(ray=int(ex.ray[i]), port=int(ex.port[i]), nb=int(ex.nb[i]), L=float(ex.L[i]), x=float(ex.x[i]),
                    y=float(ex.y[i]), Q=c(ex.Q[i]), P=c(ex.P[i]), argQ=float(ex.argQ[i]), amp=float(ex.amp[i])) for i in o],
        fields={str(p): [c(z) for z in detector_field(cell, ex, p)] for p in cell.outputs})
m = PhaseScreenModel(cell, [Source(i, **src) for i in cell.inputs])
F = m.fields(pert)
out["phase_screen"] = {f"{a},{b}": [c(z) for z in v] for (a, b), v in F.items()}
R0, a, b, cc = H.map_chd(shapes[1], m_max=6)
out["chd"] = dict(R0=float(R0), a=a.tolist(), b=b.tolist())
(ROOT / "tools" / "vectors.json").write_text(json.dumps(out))
print("wrote tools/vectors.json")
