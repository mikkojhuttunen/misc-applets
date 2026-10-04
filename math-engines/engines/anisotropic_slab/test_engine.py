import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh

from engines.anisotropic_slab import engine as an
from engines.slab_waveguide import engine as sw

LN_O, LN_E, SIO2 = 2.211, 2.138, 1.444


def fd_modes(wavelength, layers, thicknesses, pol, dx=2e-9, pad=6e-6, n_want=3):
    """Independent finite-difference solver of  d/dx[(1/q) dF/dx] + k0² F = β² F / b' ,
    with b' = n_normal² (TM) or 1 (TE) and k0² n_trans² added for TE; cell-centred grid, Dirichlet walls."""
    k0 = 2 * np.pi / wavelength
    lay = [an._triple(l) for l in layers]
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    x = np.arange(-pad, edges[-1] + pad, dx) + dx / 2
    idx = np.searchsorted(edges, x, side="right")            # 0 = substrate, len(t)+1 = cladding
    nn = np.array([lay[i][0] for i in idx]); nt = np.array([lay[i][1] for i in idx]); npr = np.array([lay[i][2] for i in idx])
    if pol == "TE":
        q, pot, mass = np.ones_like(x), (k0 * nt) ** 2, np.ones_like(x)
        # F'' + (k0² nt² - β²) F = 0 -> (F'' + k0² nt² F) = β² F
    else:
        q, pot, mass = npr**2, np.full_like(x, k0**2), nn**-2.0  # (F'/q)' + k0² F = β² F / nn²
    inv = 2 / (q[:-1] + q[1:])   # coefficient 1/q at each cell face: harmonic mean of 1/q (flux continuity)
    main = -(np.concatenate([[1 / q[0]], inv]) + np.concatenate([inv, [1 / q[-1]]])) / dx**2 + pot
    A = sp.diags([inv / dx**2, main, inv / dx**2], [-1, 0, 1], format="csc")
    M = sp.diags(mass, format="csc")
    sigma = (k0 * max(max(l) for l in lay)) ** 2 * 1.01
    w = eigsh(A, k=n_want + 2, M=M, sigma=sigma, which="LM", return_eigenvectors=False)
    w = np.sort(w)[::-1]
    lim = max(an._limit(lay[0], pol), an._limit(lay[-1], pol))
    return np.array([np.sqrt(v) / k0 for v in w if v > 0 and np.sqrt(v) / k0 > lim])


# ---- reductions and analytic limits ------------------------------------------------------------

@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("ns, nf, nc, d", [(1.444, 2.12, 1.0, 0.5e-6), (1.444, 2.138, 1.0, 1.2e-6), (1.45, 1.46, 1.45, 4e-6)])
def test_isotropic_stack_matches_slab_waveguide_engine(pol, ns, nf, nc, d):
    got = an.slab_modes(1.55e-6, [ns, nf, nc], [d], pol)["neff"]
    for m, v in enumerate(got):
        assert v == pytest.approx(sw.neff_three_layer(1.55e-6, ns, nf, nc, d, pol, m), abs=1e-9)
    assert len(got) == sw.slab_neff(1.55e-6, ns, nf, nc, d, pol)["n_modes"]


def test_symmetric_tm_matches_hand_derived_dispersion_relation():
    # even TM0 of a symmetric anisotropic slab: (κ/q_f) tan(κ d/2) = γ/q_c, q = n_prop² of each layer
    lam, d = 1.55e-6, 0.4e-6
    core, clad = (2.00, 2.20, 2.10), (1.50, 1.52, 1.51)   # (n_normal, n_trans, n_prop)
    k0 = 2 * np.pi / lam

    def f(ne):
        kappa = k0 * np.sqrt(core[2] ** 2 - ne**2 * core[2] ** 2 / core[0] ** 2)
        gamma = k0 * np.sqrt(ne**2 * clad[2] ** 2 / clad[0] ** 2 - clad[2] ** 2)
        return kappa / core[2] ** 2 * np.tan(kappa * d / 2) - gamma / clad[2] ** 2

    lo, hi = clad[0] + 1e-9, core[0] - 1e-9
    xs = np.linspace(lo, hi, 4000)
    kd2 = k0 * np.sqrt(core[2] ** 2 - xs**2 * core[2] ** 2 / core[0] ** 2) * d / 2
    xs = xs[kd2 < np.pi / 2 - 1e-6]                    # TM0 lives where κd/2 < π/2, so tan has no pole here
    vals = np.array([f(v) for v in xs])
    i = next(j for j in range(len(xs) - 1) if vals[j] * vals[j + 1] < 0)
    ref = brentq(f, xs[i], xs[i + 1], xtol=1e-14)
    got = an.slab_modes(lam, [clad, core, clad], [d], "TM")["neff"][0]
    assert got == pytest.approx(ref, abs=1e-8)


def test_thick_film_limits_are_the_polarisation_limit_index():
    # TE -> n_trans, TM -> n_normal for a very thick film
    film = (2.30, 2.10, 2.00)
    te = an.slab_modes(1.55e-6, [1.444, film, 1.0], [30e-6], "TE")["neff"][0]
    tm = an.slab_modes(1.55e-6, [1.444, film, 1.0], [30e-6], "TM")["neff"][0]
    assert te == pytest.approx(2.10, abs=2e-3) and te < 2.10
    assert tm == pytest.approx(2.30, abs=2e-3) and tm < 2.30


def test_te_ignores_normal_and_propagation_indices_tm_ignores_transverse():
    base = an.slab_modes(1.55e-6, [1.444, (2.0, 2.1, 2.2), 1.0], [0.6e-6], "TE")["neff"]
    other = an.slab_modes(1.55e-6, [1.444, (1.7, 2.1, 2.9), 1.0], [0.6e-6], "TE")["neff"]
    assert np.allclose(base, other, atol=1e-12)
    tm1 = an.slab_modes(1.55e-6, [1.444, (2.0, 2.1, 2.2), 1.0], [0.6e-6], "TM")["neff"]
    tm2 = an.slab_modes(1.55e-6, [1.444, (2.0, 1.5, 2.2), 1.0], [0.6e-6], "TM")["neff"]
    assert np.allclose(tm1, tm2, atol=1e-12)


def test_scale_invariance_and_mode_count_grows_with_thickness():
    film = (2.1, 2.2, 2.15)
    a = an.slab_modes(1.55e-6, [1.444, film, 1.0], [0.8e-6], "TM")["neff"]
    b = an.slab_modes(0.775e-6, [1.444, film, 1.0], [0.4e-6], "TM")["neff"]
    assert np.allclose(a, b, atol=1e-10)
    counts = [an.slab_modes(1.55e-6, [1.444, film, 1.0], [d], "TE")["n_modes"] for d in (0.2e-6, 0.6e-6, 1.2e-6, 3e-6)]
    assert counts == sorted(counts) and counts[-1] > counts[0]


# ---- independent finite-difference cross-check --------------------------------------------------

@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_three_layer_anisotropic_matches_finite_difference(pol):
    layers = [1.444, (2.211, 2.138, 2.211), 1.0]   # x-cut LN, y-propagation
    got = an.slab_modes(1.55e-6, layers, [0.8e-6], pol)["neff"]
    ref = fd_modes(1.55e-6, layers, [0.8e-6], pol)
    n = min(len(got), len(ref), 2)
    assert n >= 1
    assert np.allclose(got[:n], ref[:n], atol=3e-5)


def test_four_layer_biaxial_stack_matches_finite_difference():
    layers = [1.444, (1.80, 1.75, 1.83), (2.211, 2.138, 2.211), 1.0]   # SiO2 | KTP-like | LN | air
    for pol in ("TE", "TM"):
        got = an.slab_modes(1.55e-6, layers, [0.5e-6, 0.4e-6], pol)["neff"]
        ref = fd_modes(1.55e-6, layers, [0.5e-6, 0.4e-6], pol)
        n = min(len(got), len(ref), 2)
        assert n >= 1
        assert np.allclose(got[:n], ref[:n], atol=3e-5), (pol, got, ref)


# ---- orientation helper ---------------------------------------------------------------------------

def test_uniaxial_cut_orientations():
    p = an.uniaxial_principal(LN_O, LN_E)               # optic axis z
    assert an.oriented_indices(p, "z", "x") == (LN_E, LN_O, LN_O)
    assert an.oriented_indices(p, "z", "y") == (LN_E, LN_O, LN_O)
    assert an.oriented_indices(p, "x", "y") == (LN_O, LN_E, LN_O)   # x-cut, y-propagating: transverse is z (n_e)
    assert an.oriented_indices(p, "x", "z") == (LN_O, LN_O, LN_E)   # x-cut, z-propagating
    assert an.oriented_indices(p, "y", "x") == (LN_O, LN_E, LN_O)


def test_biaxial_orientation_permutes_principal_indices():
    nx, ny, nz = 1.74, 1.75, 1.83
    assert an.oriented_indices((nx, ny, nz), "z", "x") == (nz, ny, nx)
    assert an.oriented_indices((nx, ny, nz), "y", "z") == (ny, nx, nz)


def test_orientation_rejects_normal_equal_propagation():
    with pytest.raises(ValueError, match="normal"):
        an.oriented_indices((1.0, 1.0, 1.0), "x", "x")


def test_thin_film_lithium_niobate_orientation_physics():
    kw = dict(wavelength=1.55e-6, n_o=LN_O, n_e=LN_E, thickness=0.6e-6, n_sub=SIO2, n_clad=1.0)
    te_x_y = an.neff_uniaxial_film(cut="x", propagation="y", polarization="TE", **kw)["neff"]
    te_x_z = an.neff_uniaxial_film(cut="x", propagation="z", polarization="TE", **kw)["neff"]
    te_z = an.neff_uniaxial_film(cut="z", propagation="y", polarization="TE", **kw)["neff"]
    assert te_x_z == pytest.approx(te_z, abs=1e-12)    # both see the ordinary index in-plane
    assert te_x_y < te_z                               # x-cut y-propagating TE sees n_e < n_o
    # TM limited by the film index along the normal: z-cut -> n_e, x-cut -> n_o
    tm_z = an.neff_uniaxial_film(cut="z", propagation="y", polarization="TM", **kw)["neff"]
    tm_x = an.neff_uniaxial_film(cut="x", propagation="y", polarization="TM", **kw)["neff"]
    assert SIO2 < tm_z < LN_E and SIO2 < tm_x < LN_O
    assert tm_x > tm_z


def test_unguided_order_is_nan_and_validation():
    r = an.neff_uniaxial_film(1.55e-6, LN_O, LN_E, 0.3e-6, order=5)
    assert np.isnan(r["neff"])
    with pytest.raises(ValueError, match="normal"):
        an.neff_uniaxial_film(1.55e-6, LN_O, LN_E, 0.6e-6, cut="x", propagation="x")
    with pytest.raises(ValueError, match="thicknesses"):
        an.slab_modes(1.55e-6, [1.444, 2.1, 1.0], [0.1e-6, 0.1e-6])
