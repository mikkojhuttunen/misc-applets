"""Step 1: trench-DBR reflectivity. The vectorised transfer matrix equals design_calcs/tmm.py, reflects totally above
the critical angle (thick gaps) and tunnels through thin ones, stays lossless, and the engine's R(sin chi) table
reproduces the constant-R engine when the table is flat."""
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from design_calcs import tmm
from qs import mirror as Mi

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("ns,mg,mt,sd", [(1.36, 1, 1, 0.0), (1.66, 3, 1, 0.3), (2.0, 1, 5, 0.4)])
def test_vectorised_tmm_equals_original(ns, mg, mt, sd):
    dg, dt = Mi.quarter(ns, sd, mg, mt)
    s = np.linspace(0, 0.98 / ns, 41)
    for N in (1, 3, 8, 20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = tmm.refl(ns, N, dg, dt, s)
        assert np.allclose(Mi.refl(ns, Mi.stack(ns, dg, dt, N), s), a, atol=1e-12)


def test_total_internal_reflection_and_tunnelling():
    ns = 1.66
    s = np.array([0.65, 0.8, 0.95])
    dg, dt = Mi.quarter(ns, 0.2, 1, 1)
    assert np.all(Mi.refl(ns, Mi.stack(ns, dg, dt, 20), s) > 1 - 1e-12)          # TIR beyond 1/n_eff
    assert np.all(Mi.refl(ns, [(1.0, 0.02), (ns, 1.0)], s) > 1 - 1e-12)          # air behind: nowhere to tunnel to
    thin = Mi.refl(ns, [(1.0, 0.02)], s, n_exit=ns)                                # one thin gap in a continuing membrane
    thick = Mi.refl(ns, [(1.0, 0.5)], s, n_exit=ns)
    assert np.all(thin < 0.9) and np.all(thick > thin) and thin[0] < thin[-1]      # frustrated TIR
    big = Mi.refl(ns, Mi.stack(ns, *Mi.quarter(ns, 0.2, 5, 5), 20), s)            # deep evanescence: no overflow
    assert np.all(np.isfinite(big)) and np.all(big > 1 - 1e-12)


def test_stop_band_and_losslessness():
    ns = 1.66
    dg, dt = Mi.quarter(ns, 0.0, 1, 1)
    R = [Mi.refl(ns, Mi.stack(ns, dg, dt, N), [0.0])[0] for N in (1, 2, 4, 8, 16)]
    assert all(np.diff(R) > 0) and R[-1] > 0.9999 and all(0 <= r <= 1 for r in R)


def test_roughness_and_gap_coupling():
    f = Mi.roughness_factor(1.66, np.array([0.0, 0.6]), 5e-9, 1.55e-6)
    assert f[0] == pytest.approx(np.exp(-(4 * np.pi * 1.66 * 5e-9 / 1.55e-6) ** 2)) and f[1] > f[0]
    eta = Mi.gap_coupling(2.0, 600e-9, 1.55e-6)
    z = np.array([0.0, 0.2e-6, 0.4e-6, 1.2e-6])
    e = eta(z)
    assert e[0] == pytest.approx(1.0, abs=1e-6) and np.all(np.diff(e) < 0)


def test_engine_flat_table_equals_constant_R():
    r = subprocess.run(["node", str(ROOT / "scripts" / "check_rtable.js")], check=True, capture_output=True, text=True)
    d = json.loads(r.stdout)
    for k, (a, b) in d.items():
        assert a == pytest.approx(b, rel=1e-12), k


def test_step1_results_present():
    d = json.loads((ROOT / "results" / "step1_mirrors.json").read_text())
    for cell in ("circle", "herriott"):
        for ne in ("1.36", "1.53", "1.66", "1.74", "1.46", "1.59", "1.71"):
            f = d["cells"][cell]["films"][ne]
            Rs = [v["R_avg"] for v in f["by_N"].values()]
            assert 0 < Rs[0] < Rs[-1] <= 1                                   # more periods help
            assert f["best"]["R_avg"] >= max(Rs) - 1e-12
            assert f["roughness_R_avg"]["10.0"] < f["roughness_R_avg"]["0.0"]
            tab = json.loads((ROOT / f["R_table"]).read_text())
            assert len(tab["s"]) == len(tab["R"]) and tab["R"][-1] == pytest.approx(1.0, abs=1e-9)   # grazing: TIR
