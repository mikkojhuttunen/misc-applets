"""Local Tidy3D mode-solver checks (about a minute; set SWR_SLOW=1 to run)."""
import os

import numpy as np
import pytest
import yaml

td = pytest.importorskip("tidy3d")
pytestmark = pytest.mark.skipif(not os.environ.get("SWR_SLOW"), reason="slow: set SWR_SLOW=1")

from swr import config  # noqa: E402
from swr.modes import mode_report  # noqa: E402


def _cfg(tmp_path, platform_overrides):
    over = {"modes": {"grid_dl_um": 0.02}, "platforms": {"A_rib_LN": platform_overrides}}
    p = tmp_path / "o.yaml"
    p.write_text(yaml.safe_dump(over))
    return config.load(p)


def test_wide_full_etch_strip_approaches_slab(tmp_path):
    """TE0 of a 6 um wide, fully etched, vertical-wall membrane strip vs the effective-index estimate."""
    w = 6.0
    cfg = _cfg(tmp_path, {"top_width_um": w, "etch_fraction": 1.0, "sidewall_angle_deg": 90.0})
    rep = mode_report(cfg, "A_rib_LN", 1.55)
    te0 = rep.find("TE0")
    n_slab = rep.info["core_slab_neff"]["TE"]
    k0 = 2 * np.pi / 1.55
    n_ei = np.sqrt(n_slab**2 - (np.pi / (k0 * w)) ** 2)
    assert n_ei - 2e-3 < te0.n_eff < n_slab
    assert te0.leak_db_per_cm < 1e-2
    assert te0.te_fraction > 0.95
    tm0 = rep.find("TM0")
    n_slab_tm = rep.info["core_slab_neff"]["TM"]
    # Ez (normal E) samples sit on the film interfaces without subpixel averaging: O(dl) error,
    # measured +3.4e-3 at dl = 20 nm for this case (README, numerics); TE has no such error.
    assert n_slab_tm - 0.02 < tm0.n_eff < n_slab_tm + 5e-3


def test_lateral_leakage_flag_for_shallow_rib_tm(tmp_path):
    cfg = _cfg(tmp_path, {"etch_fraction": 0.5})
    rep = mode_report(cfg, "A_rib_LN", 1.55)
    assert not rep.find("TE0").lateral_leak_risk
    tm0 = rep.find("TM0")
    assert tm0.lateral_leak_risk and tm0.leak_db_per_cm > 1.0
