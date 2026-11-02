"""Step 8: the robust slab solver, both Gamma definitions and the Fig. 1d outputs."""
import csv
from pathlib import Path

import pytest

from design_calcs import slab
from qs import slabmodes as SM

ROOT = Path(__file__).resolve().parent.parent


def test_solver_matches_slab_py_where_single_mode_and_fixes_multimode():
    for n, t, lam in ((3.48, 250e-9, 1.55e-6), (2.0, 600e-9, 1.55e-6), (1.97, 1.2e-6, 3.3e-6)):
        m = SM.mode(n, t, lam, "TM")
        a = slab.gam(n, t, lam, "TM")
        assert m["n_eff"] == pytest.approx(a[0], abs=1e-6) and m["gamma_E"] == pytest.approx(float(a[1]), abs=2e-3)
    assert SM.neff(2.0, 1500e-9, 1.55e-6, "TM") > 1.9            # fundamental mode (slab.py returns a higher one, 1.39)


def test_gamma_definitions():
    si = SM.mode(3.48, 250e-9, 1.55e-6, "TM")
    assert si["gamma_abs"] == pytest.approx(0.64, abs=0.01) and si["gamma_E"] == pytest.approx(0.64, abs=0.01)
    assert SM.mode(3.42, 400e-9, 3.0e-6, "TM")["gamma_abs"] > 1.2     # thin Si at 3 um: above 1, as in the old Fig. 1d
    sin = SM.mode(2.0, 600e-9, 1.55e-6, "TM")
    assert sin["gamma_abs"] == pytest.approx(0.231, abs=0.003) and sin["gamma_E"] == pytest.approx(0.313, abs=0.003)
    assert float(SM.n_luke(1.55)) == pytest.approx(1.996, abs=0.002)


def test_outputs():
    rows = list(csv.DictReader(open(ROOT / "results" / "step8_fig1d.csv")))
    assert {(r["lambda_um"], r["pol"], r["index_model"]) for r in rows} == {(l, p, s) for l in ("1.55", "3.0") for p in ("TE", "TM") for s in ("const", "Luke2015")}
    assert min(int(r["t_nm"]) for r in rows) == 50 and max(int(r["t_nm"]) for r in rows) == 1500
    assert (ROOT / "results" / "figs" / "fig1d_SiNx.pdf").exists()
