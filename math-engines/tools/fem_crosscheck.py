"""Independent finite-element cross-check of waveguide_thermal.ridge_heating (scikit-fem, quadratic triangles).

    pip install scikit-fem
    python tools/fem_crosscheck.py

Same cross-sections, different discretisation: P2 Lagrange elements on a triangulated tensor mesh with its own
grading, element-wise conductivity, isothermal bottom, adiabatic top and sides. Prints R' from both solvers.
Also used by engines/waveguide_thermal/test_engine.py (skipped when scikit-fem is not installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.waveguide_thermal import engine as wt  # noqa: E402

CASES = {
    "SOI wire 500 x 220 nm, 2 µm BOX, oxide clad": {},
    "Si3N4 1.6 x 0.8 µm, 4 µm BOX": dict(core_material="si3n4", core_width=1.6e-6, core_height=0.8e-6, box_thickness=4e-6,
                                          clad_thickness=3e-6),
    "TFLN rib 1.2 x 0.3 µm on 0.3 µm film, air clad": dict(core_material="ln_e", slab_material="ln_e", core_width=1.2e-6,
                                                          core_height=0.3e-6, slab_thickness=0.3e-6, box_thickness=4.7e-6,
                                                          clad_material="air", clad_thickness=0.0),
}


def fem_R(geometry: dict, refine: float = 1.4) -> float:
    """R' (K m/W) of the ridge mean temperature from a P2 FEM solve of the half cross-section."""
    from skfem import Basis, BilinearForm, ElementTriP0, ElementTriP2, Functional, LinearForm, MeshTri, asm, condense, solve
    from skfem.helpers import dot, grad

    g = wt.build_ridge(resolution=refine, **geometry)
    lv = g["levels"]
    # own node axes: the FV cell edges of a different resolution, then midpoints added (not the FV cell centres)
    x = np.unique(np.concatenate([g["xe"], 0.5 * (g["xe"][1:] + g["xe"][:-1])]))
    y = np.unique(np.concatenate([g["ye"], 0.5 * (g["ye"][1:] + g["ye"][:-1])]))
    mesh = MeshTri.init_tensor(x, y)
    basis = Basis(mesh, ElementTriP2())
    b0 = basis.with_element(ElementTriP0())
    cx, cy = mesh.p[:, mesh.t].mean(axis=1)
    k = np.full(cx.shape, g["kk"]["clad"])
    k[cy < lv["y_box"]] = g["kk"]["box"]
    k[cy < lv["H"]] = g["kk"]["substrate"]
    if lv["y_film"] > lv["y_box"]:
        k[(cy > lv["y_box"]) & (cy < lv["y_film"])] = g["kk"]["slab"]
    core = (cy > lv["y_film"]) & (cy < lv["y_core"]) & (cx < lv["w2"])
    k[core] = g["kk"]["core"]
    area_core = lv["w2"] * (lv["y_core"] - lv["y_film"])
    q = np.where(core, 0.5 / area_core, 0.0)          # q' = 1 W/m over the full ridge (half modelled)

    @BilinearForm
    def a(u, v, w):
        return w.k * dot(grad(u), grad(v))

    @LinearForm
    def f(v, w):
        return w.q * v

    A = asm(a, basis, k=b0.interpolate(k))
    F = asm(f, basis, q=b0.interpolate(q))
    D = basis.get_dofs(lambda p: np.isclose(p[1], 0.0))
    T = solve(*condense(A, F, D=D))

    @Functional
    def tc(w):
        return w.T * w.c

    mean = asm(tc, basis, T=basis.interpolate(T), c=b0.interpolate(core.astype(float))) / area_core
    return float(mean)


def main():
    for name, geo in CASES.items():
        fv = wt.ridge_heating(h_top=0.0, **geo)["R_th"]
        fe = fem_R(geo)
        print(f"{name:50s} FV R' = {fv:.4f}   FEM R' = {fe:.4f}   diff {100 * (fv / fe - 1):+.2f} %")


if __name__ == "__main__":
    main()
