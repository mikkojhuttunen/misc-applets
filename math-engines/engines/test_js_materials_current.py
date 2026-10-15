"""The material table in dbr-structures/dbr-engine.js must be the one generated from engines/materials."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_js_materials  # noqa: E402


def test_js_material_table_is_current():
    src = gen_js_materials.TARGET.read_text(encoding="utf-8")
    assert src == gen_js_materials.render(src), "dbr-engine.js material table is stale: run python tools/gen_js_materials.py"


def test_every_material_is_in_the_js_table():
    from engines.materials.engine import MATERIALS
    py = {v[0] for v in gen_js_materials.KEYS.values()}
    assert set(MATERIALS) - py == {"lt_e"}                     # the approximate LiTaO3 fit stays Python-only
