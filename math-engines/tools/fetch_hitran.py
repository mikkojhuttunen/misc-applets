"""Fetch HITRAN line lists with classic HAPI and write the json that engines.trace_gas.load_lines() reads.

    pip install hitran-api
    python tools/fetch_hitran.py                         writes engines/trace_gas/hitran_lines.json
    python tools/fetch_hitran.py 5880 6670 out.json      wavenumber window in cm^-1, output path

Default window 5880-6670 cm^-1 (1500-1700 nm). Main isotopologue only (HITRAN strengths include its natural
abundance); add isotopologues (13CO2, 13CH4, ...) to SPECIES if needed. Needs internet access to hitran.org.
HITRAN molecule ids: H2O 1, CO2 2, CO 5, CH4 6, NH3 11, C2H2 26.
"""
import json
import sys
import tempfile
from pathlib import Path

from hapi import db_begin, fetch, getColumn

SPECIES = {"CH4": 6, "NH3": 11, "CO2": 2, "H2O": 1}
COLUMNS = ("nu", "sw", "gamma_air", "n_air", "elower")
ROOT = Path(__file__).resolve().parent.parent


def main(numin=5880.0, numax=6670.0, out=ROOT / "engines" / "trace_gas" / "hitran_lines.json"):
    db_begin(tempfile.mkdtemp(prefix="hitran_"))
    data = {}
    for name, mol in SPECIES.items():
        fetch(name, mol, 1, numin, numax)
        data[name] = {c: [float(v) for v in getColumn(name, c)] for c in COLUMNS}
        print(name, len(data[name]["nu"]), "lines", file=sys.stderr)
    Path(out).write_text(json.dumps(data))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(float(x) for x in a[:2]), *a[2:3]) if a else main()
