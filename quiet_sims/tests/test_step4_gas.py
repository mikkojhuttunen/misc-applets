"""Step 4: HITRAN absorption with HAPI (offline line files) reproduces the Fig. 3 strips on the same lines."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
G = json.loads((ROOT / "results" / "step4_gas.json").read_text())


@pytest.mark.parametrize("key", ["CH4_3.31", "CH4_1.65", "C2H6_3.34", "NH3_1.53"])
def test_fig3_lines_reproduced(key):
    f = G["fig3_lines"][key]
    assert f["alpha_peak_per_ppm_cm"] == pytest.approx(f["fig3_alpha_peak_per_ppm_cm"], rel=0.02)
    s = G["lines"][key]
    assert s["alpha_peak_per_ppm_cm"] >= f["alpha_peak_per_ppm_cm"] * (1 - 1e-3)    # strongest near the target (grid: 0.002 /cm)


def test_targets_and_ratio():
    lam = {k: v["lambda0_um"] for k, v in G["lines"].items()}
    assert lam["CH4_3.31"] == pytest.approx(3.31, abs=0.03) and lam["CH4_1.65"] == pytest.approx(1.65, abs=0.03)
    assert lam["C2H6_3.34"] == pytest.approx(3.34, abs=0.03) and lam["NH3_1.53"] == pytest.approx(1.53, abs=0.03)
    r = G["ratio_CH4_3p31_over_1p65"]
    assert 50 < r["strongest_lines"] < 300 and 150 < r["fig3_lines"] < 300


def test_live_hapi_nh3():
    from qs import gas
    r = gas.target_line("NH3_1.53", nu_center=6528.774, half_width=0.2, span=0.3)
    assert r["alpha_peak_per_ppm_cm"] == pytest.approx(G["lines"]["NH3_1.53"]["alpha_peak_per_ppm_cm"], rel=1e-3)
