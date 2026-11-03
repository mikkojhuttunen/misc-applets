import numpy as np
import pytest
import yaml

from swr import config

td = pytest.importorskip("tidy3d")
from swr import materials  # noqa: E402
from swr.orientation import oriented_medium, rotation_for  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def test_missing_key_raises(cfg):
    with pytest.raises(KeyError, match="fdtd.no_such_key"):
        cfg.fdtd.no_such_key


def test_override_with_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"fdtd": {"pml_layer": 3}}))
    with pytest.raises(KeyError, match="fdtd.pml_layer"):
        config.load(p)


def test_override_changes_value(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump({"fdtd": {"pml_layers": 3}}))
    assert config.load(p).fdtd.pml_layers == 3


def test_every_material_builds_lossless(cfg):
    for row in materials.provenance(cfg):
        m = materials.build(row["material"], cfg)
        lams = np.linspace(0.45, 1.55, 12)
        eps = m.eps_principal(lams)
        assert np.all(np.isreal(eps)) and np.all(eps >= 1.0)


@pytest.mark.parametrize("name,lam,n_o,n_e,tol", [
    ("LN", 1.55, 2.2111, 2.1376, 2e-3),        # Zelmon 1997
    ("LT", 1.60, 2.1174, 2.1213, 1e-3),        # Bond 1965 table
    ("sapphire", 0.6328, 1.7659, 1.7579, 1e-3),  # Malitson-Dodge
    ("Al2O3", 0.6328, 1.6674, 1.6674, 2e-3),   # PEALD
])
def test_reference_indices(cfg, name, lam, n_o, n_e, tol):
    n = materials.build(name, cfg).n_principal(lam)
    assert n[0] == pytest.approx(n_o, abs=tol)
    assert n[2] == pytest.approx(n_e, abs=tol)


def test_oriented_media_types(cfg):
    ln = materials.build("LN", cfg)
    z = oriented_medium(ln, rotation_for("zcut", cfg), cfg)
    assert isinstance(z, td.AnisotropicMedium)
    f = materials.freq(1.55)
    assert np.real(z.zz.eps_model(f)) < np.real(z.yy.eps_model(f))
    xy = oriented_medium(ln, rotation_for("xcut_y", cfg), cfg)
    assert np.real(xy.yy.eps_model(f)) == pytest.approx(np.real(z.zz.eps_model(f)))
    with pytest.raises(ValueError):
        oriented_medium(ln, rotation_for("xcut_45", cfg), cfg)
    full = oriented_medium(ln, rotation_for("xcut_45", cfg), cfg, lam_um=1.55)
    assert isinstance(full, td.FullyAnisotropicMedium)
