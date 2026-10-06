"""Fetch real HITRAN line lists with classic HAPI (hitran.org/hapi); gas_spectra.load_lines() merges all hitran_lines*.json.

    pip install hitran-api
    python fetch_hitran.py nir      # 1500-1700 nm  -> hitran_lines.json
    python fetch_hitran.py mir      # 4.4-5.6 um    -> hitran_lines_mir.json   (NO, CO, N2O, H2O, CO2)

Main isotopologue only (id 1). Needs internet access to hitran.org. HITRAN molecule ids: H2O 1, CO2 2, N2O 4, CO 5, CH4 6, NO 8, NH3 11.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import json
import sys
from pathlib import Path

from hapi import db_begin, fetch, getColumn

WINDOWS = {
    "nir": ("hitran_lines.json", 5880.0, 6670.0, {"CH4": 6, "NH3": 11, "CO2": 2, "H2O": 1}),
    "mir": ("hitran_lines_mir.json", 1800.0, 2300.0, {"NO": 8, "CO": 5, "N2O": 4, "H2O": 1, "CO2": 2}),
}
which = sys.argv[1] if len(sys.argv) > 1 else "nir"
fname, NUMIN, NUMAX, SPECIES = WINDOWS[which]
db_begin("hitran_data")
out = {}
for name, mol in SPECIES.items():
    fetch(name, mol, 1, NUMIN, NUMAX)
    out[name] = {c: [float(v) for v in getColumn(name, c)] for c in ("nu", "sw", "gamma_air", "n_air", "elower")}
    print(name, len(out[name]["nu"]), "lines")
Path("data").mkdir(exist_ok=True); (Path("data")/fname).write_text(json.dumps(out))
print("wrote", fname)
