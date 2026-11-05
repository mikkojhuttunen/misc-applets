"""Reproduce every QUIET v32 result and figure with one command (seeds fixed: inclusion layouts 1..30 (1..100 for task 8),
mulberry32 streams in the engine, numpy default_rng(7) for the Fig. 3a noise realisations and (1) for the exponential
p(L) reference).

    python run_all.py            # everything: ensembles (about 1.5-2 h on 4 cores), analyses, figures, tests
    python run_all.py --analysis # analyses, figures and tests only, from the stored results/

Steps: t1 regression (set A) -> t2 set-B ensembles (near-IR circle/Herriott, mid-IR circle) -> t5/t6/t8 ensembles
(angle-dither range, inclusions, 100 layouts) -> HITRAN lines -> analyses t1, t2, t3/t4, t7, t5/t6/t8 -> Fig. 2e data and
Fig. 2 (e, f) -> regression test."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [["bash", "scripts/run_t1.sh"], ["bash", "scripts/run_t2.sh", "4"], ["bash", "scripts/run_t5_t6_t8.sh", "4"],
        ["node", "scripts/poincare_dump.js", "results/fig2e_poincare.json"], [sys.executable, "scripts/t7_lines.py"]]
ANALYSES = [[sys.executable, "scripts/t1_summary.py"], [sys.executable, "scripts/t2_analysis.py"], [sys.executable, "scripts/t3_t4_analysis.py"],
            [sys.executable, "scripts/t7_analysis.py"], [sys.executable, "scripts/t5_t6_t8_analysis.py"], [sys.executable, "scripts/fig2ef.py"],
            [sys.executable, "-m", "pytest", "-q", "tests"]]


def main():
    steps = ANALYSES if "--analysis" in sys.argv else RUNS + ANALYSES
    for cmd in steps:
        print("::", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
