"""Step 5: the noise model reduces to the baseline shot term and adds the detector term in quadrature."""
import math

import numpy as np
import pytest

from qs import noise as Nz


def test_white_terms():
    w = Nz.white(1e-3, 0.0, 0.0)
    assert w["shot"] == pytest.approx(math.sqrt(1.602176634e-19 / 1e-3)) and w["detector"] == 0
    w = Nz.white(1e-3, 20.0, 1e-11)
    assert w["P_det"] == pytest.approx(1e-5) and w["detector"] == pytest.approx(1e-6)
    assert w["white"] == pytest.approx(math.hypot(w["shot"], w["detector"]))


def test_time_to_reach():
    T = np.logspace(-3, 2, 101)
    r = np.full(T.size, 1e-7)
    t = Nz.time_to_reach(2e-7, T, 1e-6, r)                 # sqrt(1e-12/t + 1e-14) = 4e-14 -> t = 1e-12 / 3e-14
    assert t == pytest.approx(1e-12 / 3e-14, rel=0.02)
    assert math.isinf(Nz.time_to_reach(5e-8, T, 1e-6, r))  # below the fringe floor: never


def test_step5_results():
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parent.parent / "results" / "step5_noise.json").read_text())
    for b in d["budgets"].values():
        assert len(b["rows"]) == 18
        for r in b["rows"]:
            assert r["P_det_W"] == pytest.approx(r["P_in_mW"] * 1e-3 * 10 ** (-r["IL_dB"] / 10))
            a = r["limits"]["CH4"]
            assert a["ppb_10s"] <= a["ppb_1s"] and a["M7_margin_10ppb_1s"] == pytest.approx(10 / a["ppb_1s"])
