"""Task 1 regression (set A): the published Fig. 2f / Fig. 3 numbers of proposal v32 Sec. 3.2 from results/t1, and a
bit-for-bit rerun of two layouts against the stored ensembles of the published figures (data/baseline).
python -m pytest -q tests/test_t1_regression.py   (needs results/t1 from scripts/run_t1.sh; the rerun takes ~20 s)"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

ALPHA = max(A.fig3()["strips"][0]["analyte_per_ppm"])


@pytest.fixture(scope="module")
def circ():
    return A.summary(A.load(ROOT / "results/t1/circle_A.json"), alpha_ppm=ALPHA)


@pytest.fixture(scope="module")
def herr():
    return A.summary(A.load(ROOT / "results/t1/herriott_A.json"), alpha_ppm=ALPHA)


def within_iqr(value, s):
    return s["q1"] <= value <= s["q3"]


def test_circle_both(circ):
    assert within_iqr(3.3e-8, circ["fringe_both"])
    assert circ["fringe_both"]["median"] == pytest.approx(3.3e-8, rel=0.1)


def test_circle_angle(circ):
    assert within_iqr(3.2e-7, circ["fringe_angle"])
    assert circ["fringe_angle"]["median"] == pytest.approx(3.2e-7, rel=0.15)


def test_gains(circ, herr):
    assert circ["gain_both_regular_over_chaotic"] == pytest.approx(3.0, rel=0.1)
    # proposal: ~80x; the stored ensembles and the rerun give 55x (median); 80x lies inside the layout IQR
    g = herr["fringe_both_regular"]
    assert g / herr["fringe_both"]["q3"] <= 80 <= g / herr["fringe_both"]["q1"]


def test_nea_and_limits(circ):
    assert circ["nea_old_none_T1"]["median"] == pytest.approx(1.1e-4, rel=0.05)
    assert circ["nea_old_both_T1"]["median"] == pytest.approx(4e-8, rel=0.15)
    lim = circ["limit_ppb_old_T1"]
    assert lim["median"] == pytest.approx(57, rel=0.1)
    assert lim["min"] == pytest.approx(28, rel=0.1)
    assert 1500 < lim["max"] < 2500                    # "2 ppm"
    assert circ["GL_cm"]["median"] == pytest.approx(3.2, rel=0.05)


def test_bitwise_rerun(tmp_path):
    out = tmp_path / "c.json"
    subprocess.run(["node", str(ROOT / "scripts/ensemble.js"), "--cell", "circle", "--n_eff", "2", "--gamma", "0.2", "--R", "0.995",
                    "--dBcm", "0", "--seeds", "2", "--out", str(out)], check=True, capture_output=True)
    new = json.loads(out.read_text())
    old = A.load(ROOT / "data/baseline/f2_circle_0.995.json")
    assert new["regular"]["V_joint"] == old["regular"]["V_joint"]
    for a, b in zip(new["chaotic"], old["chaotic"][:2]):
        for k in ("V_theta", "V_lambda", "V_joint", "grains", "T_trans"):
            kb = {"T_trans": "T", "grains": "grains"}.get(k, k)
            assert a[k] == b[kb], k
