import numpy as np
import pytest

from engines.ase_noise import engine as an

LAM = 1.55e-6
HV = an.H * an.C0 / LAM


def test_quantum_limited_amplifier_noise_figure():
    r = an.amplifier_noise(gain=1000.0, n_sp=1.0)
    assert r["added_photons"] == pytest.approx(999.0)
    assert r["noise_figure"] == pytest.approx(2 - 1 / 1000.0, rel=1e-12)
    assert 10 * np.log10(r["noise_figure"]) == pytest.approx(3.01, abs=0.01)


def test_noise_figure_matches_closed_form():
    for g, nsp in [(10.0, 1.2), (100.0, 1.5), (3162.0, 2.0)]:
        r = an.amplifier_noise(g, nsp)
        assert r["noise_figure"] == pytest.approx(r["noise_figure_analytic"], rel=1e-12)


def test_loss_noise_figure_is_inverse_transmission():
    assert an.noise_figure_of(an.loss_channel(0.25, n_modes=1)) == pytest.approx(4.0)


def test_friis_cascade():
    # independent route: F = F1 + (F2 - 1)/G1 + (F3 - 1)/(G1 G2)
    a1, a2 = an.gain_channel(30.0, 1.3, n_modes=1), an.gain_channel(50.0, 2.0, n_modes=1)
    loss = an.loss_channel(0.1, n_modes=1)
    f1, f2, fl = (an.noise_figure_of(c) for c in (a1, a2, loss))
    f = an.noise_figure_of(an.compose(a1, loss, a2))
    assert f == pytest.approx(f1 + (fl - 1) / 30.0 + (f2 - 1) / (30.0 * 0.1), rel=1e-12)


def test_span_cascade_osnr_drops_10logN_and_hand_value():
    # transparent chain: ASE adds n_sp (G-1) per span, OSNR = P / (2 h ν N n_sp (G-1) B)
    kw = dict(span_transmission=0.01, amp_gain=100.0, n_sp=1.5, signal_power=1e-3, wavelength=LAM, reference_bandwidth=12.5e9)
    r1, r10 = an.span_cascade(1, **kw), an.span_cascade(10, **kw)
    assert r1["net_gain"] == pytest.approx(1.0)
    assert r1["osnr"] == pytest.approx(1e-3 / (2 * HV * 1.5 * 99 * 12.5e9), rel=1e-12)
    assert 10 * np.log10(r1["osnr"] / r10["osnr"]) == pytest.approx(10.0, rel=1e-12)
    # textbook rule: OSNR_dB ≈ 58 + P_dBm - L_dB - NF_dB - 10 log N (0.1 nm at 1550 nm, G >> 1)
    nf_amp = an.amplifier_noise(100.0, 1.5)["noise_figure"]
    rule = 58 + 0 - 20 - 10 * np.log10(nf_amp) - 10
    assert 10 * np.log10(r10["osnr"]) == pytest.approx(rule, abs=0.2)


def test_gain_channel_preserves_commutator_bound():
    # T T† - I must be covered by the noise: D ≥ (T T† - I) for n_sp ≥ 1 (Caves)
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    ch = an.gain_channel([2, 5, 10, 1, 3], 1.0, unitary=U)
    assert np.allclose(ch.D, ch.T @ ch.T.conj().T - np.eye(5))


def test_mode_mixing_does_not_change_total_ase():
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    s = an.propagate(an.coherent_state(4, 1e-6, LAM), an.gain_channel(100.0, 1.5, n_modes=4))
    s2 = an.propagate(s, an.coupling_channel(U))
    assert np.trace(s2.corr).real == pytest.approx(np.trace(s.corr).real, rel=1e-12)
    assert np.linalg.norm(s2.alpha) == pytest.approx(np.linalg.norm(s.alpha), rel=1e-12)


def test_lg_mode_count():
    for n in range(6):
        assert len(an.lg_modes(n)) == (n + 1) * (n + 2) // 2


def test_aperture_hand_values_and_limits():
    modes = an.lg_modes(4)
    A = an.aperture_matrix(modes, 1.0)
    # LG00 power through radius w: 1 - exp(-2)
    assert A[0, 0] == pytest.approx(1 - np.exp(-2), rel=1e-12)
    # LG0,±1: 1 - (1 + u) e^-u at u = 2
    i = modes.index((0, 1))
    assert A[i, i] == pytest.approx(1 - 3 * np.exp(-2), rel=1e-12)
    # huge aperture -> identity (orthonormality), and passivity for any radius
    assert np.allclose(an.aperture_matrix(modes, 8.0), np.eye(len(modes)), atol=1e-10)
    ev = np.linalg.eigvalsh(A)
    assert ev.min() > -1e-12 and ev.max() < 1 + 1e-12
    # no coupling between different l
    j = modes.index((1, 0))
    assert A[i, j] == 0.0 and A[0, j] != 0.0


def test_single_mode_detection_hand_values():
    # one mode, one polarisation: s-sp = 4 R² P_s ρ B_e, sp-sp = R² ρ² (2B_o - B_e) B_e
    g, nsp, p_in, Bo, Be = 100.0, 1.5, 1e-6, 50e9, 10e9
    out = an.propagate(an.coherent_state(1, p_in, LAM), an.gain_channel(g, nsp, n_modes=1))
    r = an.detect(out, LAM, Bo, Be, n_pol=1)
    R = an.QE / HV
    rho = nsp * (g - 1) * HV
    assert r["var_s_sp"] == pytest.approx(4 * R**2 * g * p_in * rho * Be, rel=1e-12)
    assert r["var_sp_sp"] == pytest.approx(R**2 * rho**2 * (2 * Bo - Be) * Be, rel=1e-12)
    assert r["ase_power"] == pytest.approx(rho * Bo, rel=1e-12)
    assert r["effective_modes"] == pytest.approx(1.0)


def test_sp_sp_scales_with_mode_count_and_s_sp_does_not():
    base = dict(signal_power=1e-6, gain=1000.0, n_sp=1.5, spatial_filter="none", pinhole_radius=1.0,
                wavelength=1.064e-6, optical_bandwidth=100e9, electrical_bandwidth=1e9)
    r0 = an.multimode_snr(max_order=0, **base)
    r4 = an.multimode_snr(max_order=4, **base)
    assert r4["amplifier_modes"] == 30 and r4["effective_modes"] == pytest.approx(30.0)
    assert r4["var_s_sp"] == pytest.approx(r0["var_s_sp"], rel=1e-12)
    assert r4["var_sp_sp"] == pytest.approx(15 * r0["var_sp_sp"], rel=1e-12)


def test_single_mode_fibre_recovers_single_mode_amplifier():
    base = dict(signal_power=1e-6, gain=1000.0, n_sp=1.5, pinhole_radius=1.0,
                wavelength=1.064e-6, optical_bandwidth=100e9, electrical_bandwidth=1e9)
    smf = an.multimode_snr(max_order=6, spatial_filter="single_mode", **base)
    one = an.multimode_snr(max_order=0, spatial_filter="none", **base)
    assert smf["snr"] == pytest.approx(one["snr"], rel=1e-12)


def test_pinhole_helps_many_mode_amplifier():
    base = dict(signal_power=1e-6, gain=1000.0, n_sp=1.5, max_order=10,
                wavelength=1.064e-6, optical_bandwidth=100e9, electrical_bandwidth=1e9)
    bare = an.multimode_snr(spatial_filter="none", pinhole_radius=1.0, **base)
    pin = an.multimode_snr(spatial_filter="pinhole", pinhole_radius=1.0, **base)
    assert pin["snr"] > bare["snr"]
    assert pin["effective_modes"] < bare["effective_modes"]
    assert pin["signal_power"] == pytest.approx((1 - np.exp(-2)) * bare["signal_power"], rel=1e-9)


def test_validation():
    with pytest.raises(ValueError, match="gains"):
        an.gain_channel(0.5, 1.0, n_modes=1)
    with pytest.raises(ValueError, match="n_sp"):
        an.gain_channel(10.0, 0.9, n_modes=1)
    with pytest.raises(ValueError, match="passive"):
        an.loss_channel(np.diag([1.2, 0.5]))
    with pytest.raises(ValueError, match="unitary"):
        an.coupling_channel(np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="spatial_filter"):
        an.multimode_snr(1e-6, 100.0, 1.5, 2, "lens", 1.0, 1.064e-6, 1e11, 1e9)


def test_aperture_polar_form_is_exact_for_total_power_and_lg_form_converges_to_it():
    # total power through the pinhole of a field E is <E|χ|E>: 1 - e^-2 for LG00 at a = w
    exact = 1 - np.exp(-2)
    for order in (0, 4, 10):
        modes = an.lg_modes(order)
        s = an.coherent_state(len(modes), HV, LAM)  # one photon/s
        polar = an.propagate(s, an.aperture_channel(modes, 1.0))
        assert np.vdot(polar.alpha, polar.alpha).real == pytest.approx(exact, rel=1e-12)
    lg = [np.sum(an.aperture_matrix(an.lg_modes(n), 1.0)[:, 0] ** 2) for n in (0, 4, 10, 20)]
    assert all(a < b for a, b in zip(lg, lg[1:])) and lg[-1] < exact
    assert lg[-1] == pytest.approx(exact, rel=0.03)
