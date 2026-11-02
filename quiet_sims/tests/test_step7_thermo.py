"""Step 7: thermo-optic dither scalings."""
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "results" / "step7_thermo.json").read_text())


def test_modes_and_phase():
    si = D["cases"]["Si 250 nm, 1.55 um"]
    assert si["modes_in_port"] == pytest.approx(78, abs=1.5)                    # the brief's ~78 modes
    sin = D["cases"]["SiNx 600 nm, 1.55 um"]
    assert sin["phase_edge_rad"] == pytest.approx(2 * math.pi / 1.55e-6 * sin["n_eff"] * 15e-6 * 2e-3)


@pytest.mark.parametrize("name", list(D["cases"]))
def test_power_and_speed_trade(name):
    r = D["cases"][name]["by_anchor_distance"]
    for a, b in zip(r, r[1:]):
        k = b["l_um"] / a["l_um"]
        assert a["P_edge_mW"] / b["P_edge_mW"] == pytest.approx(k) and b["tau_us"] / a["tau_us"] == pytest.approx(k * k)
