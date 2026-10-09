import numpy as np
import pytest

from ngrc import rays
from ngrc.field import beamlet_matrix, detector_field, field_correlation
from ngrc.geometry import CircularCell, Port, ports_at, symmetric_ports
from ngrc.index_map import GridIndex, rasterize_shape
from ngrc.perturbative import PhaseScreenModel
from ngrc.rays import Exits, Source, trace
from ngrc.shapes import Perturbation, Shape


def _cell(**kw):
    return CircularCell(ports=symmetric_ports(4, inputs=(0,), width=20e-6, launch=0.35, fan=0.5), **kw)


def _pert(dn=1e-3):
    return Perturbation([Shape(2e-4, 1e-4, 80e-6, dn, 3e-6, a=[0, 0.05, 0.02]),
                         Shape(-3e-4, -2e-4, 60e-6, dn, 3e-6, b=[0, 0.03])])


def test_ports():
    c = _cell()
    assert c.inputs == [0] and c.outputs == [0, 1, 2, 3]
    assert c.port_index(np.array([0.0, np.pi / 2, 0.3]))[:2].tolist() == [0, 1]
    assert c.port_index(0.3)[0] == -1
    a = CircularCell(ports=ports_at([0, 75, 200], inputs=(0, 1)))
    assert a.inputs == [0, 1]
    with pytest.raises(ValueError):
        CircularCell(ports=[Port(0.0), Port(1e-6)]).check()


def test_unperturbed_circle_path_lengths():
    """sin χ is conserved: every chord is 2 Rc cos χ, so L ≈ n0 (nb + 1) 2 Rc cos χ."""
    c = _cell()
    ex = trace(c, Source(0, n_pos=1, n_ang=21))
    cos_chi = np.cos(c.ports[0].launch + np.linspace(-0.5, 0.5, 21) * c.ports[0].fan)[ex.ray]
    ratio = ex.L / (c.n_eff * 2 * c.radius * cos_chi)
    assert np.allclose(ratio, ex.nb + 1, atol=2e-3)


def test_free_beamlet_is_a_gaussian_beam():
    c = _cell()
    w0, z = 4e-6, 300e-6
    n0, k0 = c.n_eff, c.k0
    zR = k0 * n0 * w0**2 / 2
    one = lambda v, dt=float: np.array([v], dt)
    ex = Exits(port=one(1, int), ray=one(0, int), x=one(0.0), y=one(0.0), tx=one(1.0), ty=one(0.0), L=one(0.0),
               Q=one(1.0, complex), P=one(1j * n0 / zR, complex), argQ=one(0.0), amp=one(1.0), phase=one(0.0),
               nb=one(0, int))
    q = np.linspace(-40e-6, 40e-6, 201)
    pts = np.c_[np.full_like(q, z), q]
    G, _ = beamlet_matrix(c, ex, 1, pts)
    I = np.abs(G[:, 0]) ** 2
    w = w0 * np.sqrt(1 + (z / zR) ** 2)
    assert np.allclose(I / I.max(), np.exp(-2 * q**2 / w**2), atol=1e-10)
    assert np.isclose(I.max(), w0 / w, rtol=1e-10)                       # 2D: |E|² ∝ 1/w
    gouy = -0.5 * np.arctan(z / zR)
    assert np.isclose(np.angle(G[100, 0] * np.exp(-1j * k0 * n0 * z)), gouy, atol=1e-9)


def test_diameter_round_trip_matches_abcd():
    """Ray along a diameter: free 2Rc, circular mirror (tangential f = Rc/2), free 2Rc, out through the input port."""
    c = CircularCell(radius=1e-3, ports=[Port(0.0, width=10e-6, role="inout", fan=0.0)])
    src = Source(0, n_pos=1, n_ang=1, waist=6e-6)
    ex = trace(c, src)
    n0, Rc = c.n_eff, c.radius
    zR = c.k0 * n0 * src.waist**2 / 2
    q = -1j * zR + 2 * Rc
    q = 1 / (1 / q - 2 / Rc)
    q = q + 2 * Rc
    assert ex.nb[0] == 1 and ex.port[0] == 0
    assert np.isclose(n0 * ex.Q[0] / ex.P[0], q, rtol=1e-9)
    assert np.isclose(ex.L[0], n0 * 4 * Rc, rtol=1e-12)


def test_quadratic_grin_period():
    """n = n0 (1 − g² y² / 2): paraxial rays oscillate as cos(g s), the beamlet Q too."""
    n0, g = 1.8, 2 * np.pi / 2e-3

    class Grin:
        x0 = y0 = 0.0
        bound = 1.0
        def eval(self, x, y, hess=True):
            z = np.zeros_like(x)
            out = [-n0 * g * g * y * y / 2, z, -n0 * g * g * y]
            return out + ([z, z, np.full_like(x, -n0 * g * g)] if hess else [])

    p = Perturbation([Grin()])
    y0, h, steps = 5e-6, 2e-6, 1000                     # s = 2 mm = one period
    x, y = np.array([0.0]), np.array([y0])
    px, py = np.array([n0]), np.array([0.0])
    L, Q, P = np.zeros(1), np.ones(1, complex), np.zeros(1, complex)
    for _ in range(steps):
        x, y, px, py, L, Q, P = rays._rk4(p, n0, "curved", h, x, y, px, py, L, Q, P)
    assert abs(y[0] - y0) < 1e-3 * y0
    assert abs(Q[0] - 1) < 1e-3
    for _ in range(steps // 4):
        x, y, px, py, L, Q, P = rays._rk4(p, n0, "curved", h, x, y, px, py, L, Q, P)
    assert abs(Q[0]) < 2e-3                              # quarter period: Q = cos(g s) = 0


def test_momentum_tracks_index_in_curved_mode():
    """|p| = n along rays through Δn = 5e-3 shapes; the drift converges at 4th order in the RK4 step."""
    c, p = _cell(), _pert(5e-3)
    err = []
    for h in (1.5e-6, 0.75e-6):
        x, y = np.full(7, 2e-4 - 150e-6), 1e-4 + np.linspace(-60e-6, 60e-6, 7)
        px, py = np.full(7, c.n_eff), np.zeros(7)
        L, Q, P = np.zeros(7), np.ones(7, complex), np.full(7, 1j, complex)
        e = 0.0
        for _ in range(int(round(300e-6 / h))):
            x, y, px, py, L, Q, P = rays._rk4(p, c.n_eff, "curved", h, x, y, px, py, L, Q, P)
            e = max(e, np.max(np.abs(np.hypot(px, py) - c.n_eff - p.value(x, y))))
        err.append(e)
    assert err[0] < 1e-6
    assert err[0] / err[1] > 12


def test_phase_screen_equals_straight_tracer():
    c, p = _cell(), _pert()
    src = Source(0, n_pos=3, n_ang=31)
    F = PhaseScreenModel(c, [src]).fields(p)
    ex = trace(c, src, p, "straight")
    for o in c.outputs:
        assert field_correlation(F[(0, o)], detector_field(c, ex, o)) > 0.99999


def test_curved_reduces_to_phase_screen_for_small_dn():
    """(At Δn ~ 1e-6 single rays already switch between exiting and reflecting at a port edge.)"""
    c, p = _cell(), _pert(1e-8)
    src = Source(0, n_pos=3, n_ang=31)
    F = PhaseScreenModel(c, [src]).fields(p)
    ex = trace(c, src, p, "curved")
    for o in c.outputs:
        assert field_correlation(F[(0, o)], detector_field(c, ex, o)) > 0.99999


def test_grid_image_gives_the_same_speckle_as_the_analytic_shape():
    c = _cell()
    s = Shape(2e-4, 1e-4, 80e-6, 1e-3, 3e-6, a=[0, 0.05, 0.02])
    g = GridIndex(rasterize_shape(s, 1e-6), 1e-6, s.x0, s.y0)
    m = PhaseScreenModel(c, [Source(0, n_pos=3, n_ang=31)])
    Fa, Fg = m.fields(Perturbation([s])), m.fields(Perturbation([g]))
    for o in c.outputs:
        assert field_correlation(Fa[(0, o)], Fg[(0, o)]) > 0.999


def test_unperturbed_speckle_is_developed():
    c = _cell()
    ex = trace(c, Source(0, n_pos=5, n_ang=61))
    I = np.abs(detector_field(c, ex, 1)) ** 2
    assert 0.6 < I.std() / I.mean() < 1.3
    assert ex.port_power(4).sum() > 0.5
