"""Step 0: the baselines. The stored ensembles give the quoted numbers, the repo engine reproduces stored layouts bit
for bit, the full rerun (results/step0, made by scripts/ens*.js) gives the same ensemble numbers, and the slab solver
gives Si 250 nm TM0."""
import json
import math
import subprocess
from pathlib import Path

import pytest

from design_calcs import slab
from qs import metrics as M

ROOT = Path(__file__).resolve().parent.parent
TARGET = dict(fringe_both_chaotic=4.2e-8, fringe_both_regular=1.7e-7, gammaL_cm=2.5, nh3_ppb=110.0)


def _numbers(f2, ens):
    fs = M.fringe_stats(f2)
    nh3 = M.detection_limit(ens, M.fig3_data()["T"], M.peak_alpha("NH3"), 0.2, T=1.0)
    return fs, nh3


def _check(fs, nh3):
    assert fs["n_layouts"] == 30
    assert fs["modes"]["both"]["chaotic_median"] == pytest.approx(TARGET["fringe_both_chaotic"], rel=0.03)
    assert fs["modes"]["both"]["regular"] == pytest.approx(TARGET["fringe_both_regular"], rel=0.03)
    assert fs["gammaL_cm"] == pytest.approx(TARGET["gammaL_cm"], rel=0.03)
    assert nh3["ppm"] * 1e3 == pytest.approx(TARGET["nh3_ppb"], rel=0.03)


def test_stored_baselines_give_the_quoted_numbers():
    b = M.RESULTS / "baseline"
    ens = M.load(b / "ens_tgt.json")
    assert len(ens["chaotic"]) == 21
    _check(*_numbers(M.load(b / "f2_circle_0.99.json"), ens))


def test_full_rerun_reproduces_the_baselines():
    d = M.RESULTS / "step0"
    f2, ens = M.load(d / "f2_circle_0.99.json"), M.load(d / "ens_tgt.json")
    assert len(ens["chaotic"]) == 21
    _check(*_numbers(f2, ens))
    sb, se = M.load(M.RESULTS / "baseline" / "f2_circle_0.99.json"), M.load(M.RESULTS / "baseline" / "ens_tgt.json")
    for a, b in zip(f2["chaotic"], sb["chaotic"]):
        assert a["V_joint"] == b["V_joint"] and a["grains"] == b["grains"] and a["L_mean"] == b["L_mean"]
    for a, b in zip(ens["chaotic"], se["chaotic"]):
        assert a["env"]["both"] == b["env"]["both"] and a["T_trans"] == b["T_trans"]


@pytest.mark.parametrize("script,args,stored", [
    ("ens2.js", ["0.99", "0.1", "circle", "1"], "f2_circle_0.99.json"),
    ("ens.js", ["0.99", "0.1", "1"], "ens_tgt.json"),
])
def test_engine_reproduces_stored_layouts(tmp_path, script, args, stored):
    out = tmp_path / "o.json"
    subprocess.run(["node", str(ROOT / "scripts" / script), *args, str(out)], check=True, capture_output=True)
    a, b = json.loads(out.read_text()), M.load(M.RESULTS / "baseline" / stored)
    for x, y in ((a["regular"], b["regular"]), (a["chaotic"][0], b["chaotic"][0])):
        for k in x:
            if k in ("label",):
                continue
            assert x[k] == y[k], k


def test_slab_si_250nm_tm0():
    ne, g = slab.gam(3.48, 250e-9, 1.55e-6, "TM")
    assert round(ne, 1) == 2.0 and round(float(g), 2) == 0.64
    assert slab.gam(2.0, 600e-9, 1.55e-6, "TM")[1] == pytest.approx(0.31, abs=0.01)
    assert slab.gam(2.0, 700e-9, 1.55e-6, "TM")[1] == pytest.approx(0.22, abs=0.01)
