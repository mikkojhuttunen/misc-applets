"""Step 6: DFG design numbers are physically consistent."""
import json
import math
from pathlib import Path

import pytest

from qs import dfg

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "results" / "step6_dfg.json").read_text())


def test_sellmeier_and_qpm_checks():
    assert dfg.n_e(1.064, 25) == pytest.approx(2.148, abs=0.002)          # MgO:CLN e-ray at 1064 nm
    shg = 1 / (dfg.n_e(0.532, 40) / 0.532 - 2 * dfg.n_e(1.064, 40) / 1.064) / dfg.expansion(40)
    assert 6.85 < shg < 7.0                                                 # 1064 nm SHG grating ~ 6.9 um


def test_energy_conservation_and_brief_signals():
    assert dfg.signal_for_idler(1.064, 3.31) * 1e3 == pytest.approx(1568, abs=1)
    assert dfg.signal_for_idler(1.064, 1e4 / 2996.852) * 1e3 == pytest.approx(1562, abs=1)
    for v in D["lines"].values():
        assert 1 / v["idler_um"] == pytest.approx(1 / 1.064 - 1 / v["signal_um"])


def test_scalings():
    for v in D["lines"].values():
        r = v["by_length_mm"]
        assert r["20"]["T_fwhm_C"] / r["50"]["T_fwhm_C"] == pytest.approx(2.5, rel=0.02)          # ∝ 1/L
        assert r["50"]["P_idler_mW"]["Ps_5W_h1.0"] / r["20"]["P_idler_mW"]["Ps_5W_h1.0"] == pytest.approx(2.5, rel=1e-9)   # ∝ L (BK)
        assert r["50"]["P_idler_mW"]["Ps_5W_h1.0"] == pytest.approx(5 * r["50"]["P_idler_mW"]["Ps_1W_h1.0"])
    d = D["dither"]["by_line"]["CH4_3.31 nominal (brief)"]
    assert d["idler_dnu_GHz"] == pytest.approx(0.547, abs=0.002) and d["signal_dlambda_pm"] == pytest.approx(4.49, abs=0.02)
