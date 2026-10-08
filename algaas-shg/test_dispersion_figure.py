"""Smoke test of the figure/data export (about 15 s). Run: python -m pytest algaas-shg/test_dispersion_figure.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
from dispersion_figure import export  # noqa: E402
from engines.algaas_rib import engine as E  # noqa: E402


def test_export_writes_consistent_files(tmp_path):
    st = E.RibStack(0.15e-6, 0.2e-6, 0.68e-6)
    pre = str(tmp_path / "d")
    export(pre, st, 1550.0, 1, extra=("TM:0,0",), n_points=5, formats=("png", "svg"), step=40e-9, dpi=80)
    for suffix in (".png", ".svg", "_dispersion.csv", "_fields.npz", "_meta.json"):
        assert Path(pre + suffix).stat().st_size > 0
    rows = np.genfromtxt(pre + "_dispersion.csv", delimiter=",", names=True)
    assert rows["wavelength_nm"].min() == 750 and rows["wavelength_nm"].max() == 1600
    assert 775.0 in rows["wavelength_nm"] and 1550.0 in rows["wavelength_nm"]
    meta = json.loads(Path(pre + "_meta.json").read_text())
    i = np.argmin(abs(rows["wavelength_nm"] - 1550))
    j = np.argmin(abs(rows["wavelength_nm"] - 775))
    assert rows["neff_TE00"][i] == np.float64(f"{meta['markers']['TE(0,0)']['neff']:.6f}")
    dn = rows["neff_TM01"][j] - rows["neff_TE00"][i]
    assert abs(dn - meta["delta_n"]) < 2e-6
    f = np.load(pre + "_fields.npz")
    assert f["TE00_field"].shape == (len(f["TE00_y_um"]), len(f["TE00_x_um"]))
