"""Task 1: summary of the set-A regression runs -> results/t1/summary.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qv import analysis as A  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results" / "t1"
a = max(A.fig3()["strips"][0]["analyte_per_ppm"])
A.dump(R / "summary.json", {c: A.summary(A.load(R / f"{c}_A.json"), alpha_ppm=a) for c in ("circle", "herriott")} | {"alpha_NH3_per_ppm_cm": a})
print("ok")
