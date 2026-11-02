"""Step 2: the membrane-cell ensembles are complete and self-consistent."""
import json
from pathlib import Path

import pytest

from qs import metrics as M

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "step2_summary.json").read_text())


def test_all_cases_present():
    names = set(S["cases"])
    for cell in ("circle", "herriott"):
        for ne in ("1.36", "1.53", "1.66", "1.74"):
            assert f"{cell}_neff{ne}" in names
        assert f"{cell}_neff1.66_rough5nm" in names
    for c in S["cases"].values():
        assert c["n_layouts"] == 30 and c["detection_limit"]["n_layouts"] == 21


@pytest.mark.parametrize("name", sorted(S["cases"]))
def test_case_consistency(name):
    c = S["cases"][name]
    res = M.load(ROOT / "results" / "step2" / f"{name}.json")
    assert c["gammaL_cm"] == pytest.approx(c["gamma"] * c["L_median_m"] * 100)
    dl = c["detection_limit"]
    assert dl["ppm"] == pytest.approx(dl["nea"] / (100 * S["alpha_NH3_per_cm_ppm"] * c["gamma"] * dl["L_median_m"]))   # ppb ∝ 1/(Γ<L>)
    assert c["R_bounce_median"] == pytest.approx(res["design"]["R_avg"], abs=3e-3)       # the table acts as designed
    m = c["modes"]
    assert m["both"]["chaotic_median"] < min(m["angle"]["chaotic_median"], m["spectral"]["chaotic_median"]) < m["none"]["chaotic_median"]


def test_roughness_costs_path_and_sensitivity():
    for cell in ("circle", "herriott"):
        a, b = S["cases"][f"{cell}_neff1.66"], S["cases"][f"{cell}_neff1.66_rough5nm"]
        assert b["R_bounce_median"] < a["R_bounce_median"] and b["T_median"] < a["T_median"]
        assert b["detection_limit"]["ppm"] > a["detection_limit"]["ppm"]
