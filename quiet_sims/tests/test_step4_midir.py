"""Step 4: mid-infrared ensembles complete; more membrane loss shortens the path; limits follow the HITRAN peaks."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "step4_summary.json").read_text())
G = json.loads((ROOT / "results" / "step4_gas.json").read_text())


def test_cases():
    keys = {(v["cell"], v["film_nm"], v["dBcm"]) for v in S["cases"].values()}
    assert keys == {(c, f, d) for c in ("circle", "herriott") for f in (1000, 1200, 1500) for d in (0.1, 0.3, 0.5)}
    assert S["alpha_peak_per_ppm_cm"]["CH4"] == pytest.approx(G["lines"]["CH4_3.31"]["alpha_peak_per_ppm_cm"])


@pytest.mark.parametrize("cell", ["circle", "herriott"])
@pytest.mark.parametrize("film", [1000, 1200, 1500])
def test_loss_shortens_path(cell, film):
    cs = sorted((v for v in S["cases"].values() if v["cell"] == cell and v["film_nm"] == film), key=lambda v: v["dBcm"])
    assert cs[0]["gammaL_cm"] > cs[1]["gammaL_cm"] > cs[2]["gammaL_cm"]
    for v in cs:
        a = v["limits"]
        assert a["C2H6"]["ppb_1s"] / a["CH4"]["ppb_1s"] == pytest.approx(S["alpha_peak_per_ppm_cm"]["CH4"] / S["alpha_peak_per_ppm_cm"]["C2H6"], rel=1e-9)
