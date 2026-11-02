"""Step 3: the launch-angle study is complete; grazing launches put more bounces above the critical angle."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "step3_summary.json").read_text())


def test_cases_and_angles():
    got = {(c["m"], f"{c['n_eff']:.2f}") for c in S["cases"].values()}
    assert got == {(m, ne) for m in (11, 8, 5, 2, 1) for ne in ("1.66", "1.36")}
    for c in S["cases"].values():
        assert c["chi_deg"] == pytest.approx(90 - 180 * c["m"] / 24)
        assert c["n_layouts"] == 21


@pytest.mark.parametrize("ne", ["1.66", "1.36"])
def test_grazing_launch_raises_the_tir_share(ne):
    cs = sorted((c for c in S["cases"].values() if f"{c['n_eff']:.2f}" == ne), key=lambda c: -c["m"])
    shares = [c["share_above_critical_hist"] for c in cs]
    assert shares[-1] > 0.8 > shares[0] and shares[0] < 0.25
    assert S["best"][ne]["NH3_ppb"] == pytest.approx(min(1e3 * c["detection_limit"]["ppm"] for c in cs))
