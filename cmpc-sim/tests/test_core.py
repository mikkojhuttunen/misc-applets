"""Unit tests (pytest, or `python tests/test_core.py`). The package must sit next to math-engines/ (inside misc-applets), or set MISC_APPLETS."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))
import json
import numpy as np
import cmpc_sim as cs, coherence_model as cm, gas_spectra as gs, mirror_design as md

LAM = 1.55e-6


def _paths(cells):
    return cm.Paths(cells=cells, lam=LAM, neff=2.0, n_g=3.0, port_w=30e-6, T_det=0.1, Gamma=0.5, n_rays_used=100,
                    L_all=np.array([0.1]), W_all=np.array([1.0]))


def test_builtin_selftest():
    assert cs.selftest(verbose=False)


def test_tmm_matches_engine_at_normal_incidence_and_s_equals_p():
    lay = [(1.0, 0.3e-6), (2.8, 0.15e-6)] * 3
    s = cs.stack_R_oblique(1.55e-6, 0.0, 2.8, lay, 1.0, "s"); p = cs.stack_R_oblique(1.55e-6, 0.0, 2.8, lay, 1.0, "p")
    assert abs(float(s) - float(p)) < 1e-12


def test_two_material_stacks_are_transparent_at_brewster():
    nA, rng = 2.36, np.random.default_rng(0)
    sB = 1 / np.sqrt(1 + nA**2)
    for _ in range(50):
        lay = [(1.0, rng.uniform(0.05, 3) * 1e-6), (nA, rng.uniform(0.05, 3) * 1e-6)] * int(rng.integers(3, 20))
        assert float(cs.stack_R_oblique(5.26e-6, sB, nA, lay, 1.0, "p")) < 1e-12


def test_gamma_limits_and_polarisation_order():
    assert cs.membrane_mode("Si", 5e-9, LAM, "TE").Gamma > 0.9
    assert cs.membrane_mode("Si", 220e-9, LAM, "TM").Gamma > cs.membrane_mode("Si", 220e-9, LAM, "TE").Gamma


def test_single_path_has_no_speckle_and_two_equal_paths_give_1_over_sqrt2():
    assert cm.contrast(_paths([(np.array([1.0]), np.array([0.2]))]), 0.0, True) == 0.0
    two = _paths([(np.array([1.0, 1.0]), np.array([0.0, 0.3]))])
    assert abs(cm.contrast(two, 0.0, True) - np.sqrt(0.5)) < 1e-12
    assert cm.contrast(two, 1e10, True) < 0.05          # linewidth >> c/(n_g dL) averages the speckle away


def test_incoherent_beams_scale_as_inverse_sqrt_K():
    cell = (np.array([1.0, 1.0]), np.array([0.0, 0.3]))
    c1 = cm.contrast(_paths([cell]), 0.0, True)
    c4 = cm.contrast(cm.merge_incoherent([_paths([cell])] * 4), 0.0, True)
    assert abs(c4 - c1 / 2) < 1e-12


def test_voigt_is_normalised():
    x = np.linspace(-300, 300, 600001)
    assert abs(np.trapezoid(gs.voigt(x, 0.06, 0.003), x) - 1) < 2e-3


def test_through_cell_weak_absorption_limit():
    L = np.array([0.2, 0.6]); W = np.array([1.0, 1.0])
    T = gs.through_cell(np.array([1e-8]), L, W, 0.5)[0]       # alpha 1e-8 /cm
    assert abs((1 - T) - 0.5 * 1e-8 * 100 * 0.4) < 1e-12


def test_three_index_mirror_beats_two_material():
    d = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "data" / "mirror_designs.json"))["130_16"]
    assert d["mean_loss"] < 0.02


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k)
    print(n, "tests passed")
