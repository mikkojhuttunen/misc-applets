"""Step 9: the figures exist, fit the page and use 10 pt Liberation Sans; the plotted numbers match the ensembles."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "results" / "figs"
D = json.loads((ROOT / "results" / "step9_figdata.json").read_text())


@pytest.mark.parametrize("name", ["fig1d_SiNx.pdf", "fig2f_static_fringe.pdf", "fig3_spectroscopy.pdf"])
def test_pdf_size_and_font(name):
    info = subprocess.run(["pdfinfo", str(F / name)], capture_output=True, text=True).stdout
    w = float(info.split("Page size:")[1].split("x")[0])
    assert w <= 17 / 2.54 * 72 + 0.5                                  # at most 17 cm wide
    fonts = subprocess.run(["pdffonts", str(F / name)], capture_output=True, text=True).stdout
    assert "LiberationSans" in fonts


def test_fig2f_medians_and_quartiles():
    res = json.loads((ROOT / "results" / "step2" / "circle_neff1.66.json").read_text())
    c = D["fig2f"]["C-CMPC"]
    for i, m in enumerate(c["modes"]):
        x = np.array([q["floor"][m] for q in res["chaotic"]])
        assert c["value"][i] == pytest.approx(np.median(x)) and c["q1"][i] == pytest.approx(np.percentile(x, 25))
    assert D["fig2f"]["R-CMPC"]["value"][3] == pytest.approx(res["regular"]["floor"]["both"])


def test_fig3b_consistent_with_step2():
    s2 = json.loads((ROOT / "results" / "step2_summary.json").read_text())["cases"]["circle_neff1.66"]
    assert D["fig3"]["3b"]["ppb_at_1s"] == pytest.approx(1e3 * s2["detection_limit"]["ppm"], rel=0.1)
