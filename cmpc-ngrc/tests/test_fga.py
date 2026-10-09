"""Frozen-Gaussian (Herman–Kluk) model against exact references."""
import numpy as np

from ngrc.field import _beamlets_at, beamlet_matrix, detector_field, field_correlation, propagation_matrix
from ngrc.geometry import CircularCell, Port
from ngrc.rays import Source, trace
from ngrc.shapes import Perturbation, Shape
from ngrc.wave_ref import bpm


def _exact_on_aperture(c, out_port, fan, nb):
    """One Gaussian beamlet (exact paraxial ABCD, gbs model) sampled over the output opening and propagated."""
    g = CircularCell(**{**c.__dict__, "model": "gbs"})
    w_in = c.wavelength / (np.pi * c.n_eff * np.tan(fan / 2))
    eg = trace(g, Source(0, n_pos=1, n_ang=1, waist=w_in), max_bounces=nb + 1, amp_min=0)
    ap, du = c.aperture_points(out_port)
    B, _ = _beamlets_at(g, eg.select(out_port), ap)
    return propagation_matrix(c, out_port, ap, du) @ B.sum(1)


def test_single_pass_with_aperture_clipping_is_exact():
    for W in (20e-6, 200e-6):
        c = CircularCell(ports=[Port(0.0, width=W, role="in", launch=0.0, fan=0.1),
                                Port(np.pi, width=W, role="out")], reflectance=0.0)
        E = detector_field(c, trace(c, Source(0, n_pos=None, n_ang=801, frozen=10e-6), max_bounces=1), 1)
        Er = _exact_on_aperture(c, 1, 0.1, 0)
        assert field_correlation(E, Er) > 0.9999
        assert abs(np.sum(abs(E) ** 2) / np.sum(abs(Er) ** 2) - 1) < 0.01


def test_polygon_orbits_through_oblique_mirrors():
    for chi, nb in ((np.deg2rad(30), 2), (np.deg2rad(54), 4)):     # triangle and pentagon back to the port
        c = CircularCell(ports=[Port(0.0, width=60e-6, role="inout", launch=chi, fan=0.06)], reflectance=1.0)
        ex = trace(c, Source(0, n_pos=None, n_ang=1601, frozen=8e-6), max_bounces=nb + 1, amp_min=0)
        E = detector_field(c, ex, 0)
        Er = _exact_on_aperture(c, 0, 0.06, nb)
        assert field_correlation(E, Er) > 0.999
        assert abs(np.sum(abs(E) ** 2) / np.sum(abs(Er) ** 2) - 1) < 0.01


def test_frozen_width_does_not_matter():
    c = CircularCell(ports=[Port(0.0, width=200e-6, role="in", launch=0.0, fan=0.1),
                            Port(np.pi, width=200e-6, role="out")], reflectance=0.0)
    E = [detector_field(c, trace(c, Source(0, n_pos=None, n_ang=801, frozen=w), max_bounces=1), 1) for w in (4e-6, 20e-6)]
    assert field_correlation(*E) > 0.9999
    assert abs(np.sum(abs(E[0]) ** 2) / np.sum(abs(E[1]) ** 2) - 1) < 0.01


def test_dot_scattering_matches_bpm():
    Rc = 1e-3
    c = CircularCell(radius=Rc, ports=[Port(0.0, width=600e-6, role="in", launch=0.0, fan=0.06),
                                       Port(np.pi, width=600e-6, role="out")], reflectance=0.0)
    w_in = c.wavelength / (np.pi * c.n_eff * np.tan(0.03))
    yobs = np.linspace(-150e-6, 150e-6, 201)
    pts = np.c_[np.full_like(yobs, -Rc), yobs]
    y = (np.arange(8192) - 4096) * (2e-3 / 8192)
    res = {}
    for dn in (0.0, 3e-3):
        p = Perturbation([Shape(0.4e-3, 8e-6, 50e-6, dn, 3e-6, a=[0, 0.05, 0.03])]) if dn else None
        Eb = bpm(lambda z, yy: (p.value(np.full_like(yy, Rc - z), yy) if p else 0 * yy), c.n_eff, c.wavelength, y,
                 2 * Rc, np.exp(-(y / w_in) ** 2), dz=2e-6)
        Eb = np.interp(yobs, y, Eb.real) + 1j * np.interp(yobs, y, Eb.imag)
        ex = trace(c, Source(0, n_pos=None, n_ang=1201, frozen=6e-6), p, "curved", max_bounces=1)
        res[dn] = (Eb, beamlet_matrix(c, ex, 1, points=pts)[0].sum(1))
    assert field_correlation(res[3e-3][0] - res[0][0], res[3e-3][1] - res[0][1]) > 0.999
