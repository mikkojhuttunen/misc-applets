import numpy as np
import pytest
from scipy.linalg import expm as scipy_expm

from engines.squeezed_light import engine as sl

WG = dict(d_eff=14e-12, wavelength_signal=1.55e-6, n_signal=2.14, n_pump=2.18, area_eff=2e-12)


def test_coupling_efficiency_hand_value():
    # 8π² (14 pm/V)² / (ε0 c 2.14² 2.18 (1.55 µm)² 2 µm²) = 1.2153e5 1/(W m²) = 1215 %/(W cm²)
    eta = sl.coupling_efficiency(14e-12, 1.55e-6, 1.55e-6, 2.14, 2.14, 2.18, 2e-12)
    assert eta == pytest.approx(1.21534e5, rel=1e-4)


def test_expm_matches_scipy():
    rng = np.random.default_rng(1)
    for scale in (0.01, 1.0, 8.0):
        m = rng.normal(size=(6, 6)) * scale
        assert np.allclose(sl.expm(m), scipy_expm(m), rtol=1e-10, atol=1e-12 * np.abs(scipy_expm(m)).max())


def test_drift_from_hamiltonian_and_symplectic():
    g, dk, L = 300.0, 150.0, 4e-3
    a = sl.drift_matrix(sl.hamiltonian_degenerate(g, dk))
    assert np.allclose(a, [[g, dk / 2], [-dk / 2, -g]])
    s = sl.expm(sl.drift_matrix(sl.hamiltonian_two_mode(g, dk)) * L)
    om = sl.omega(2)
    assert np.allclose(s @ om @ s.T, om, atol=1e-9)        # lossless evolution is symplectic


def test_perfect_matching_gives_e_minus_2r():
    r = sl.degenerate_squeezing(1.0, 5e-3, **WG)
    assert r["variance_min"] == pytest.approx(np.exp(-2 * r["squeezing_parameter"]), rel=1e-9)
    assert r["variance_max"] == pytest.approx(np.exp(2 * r["squeezing_parameter"]), rel=1e-9)
    assert r["mean_photons"] == pytest.approx(np.sinh(r["squeezing_parameter"]) ** 2, rel=1e-9)
    assert r["purity"] == pytest.approx(1.0, rel=1e-9)
    # r = γL = √(η P) L
    assert r["squeezing_parameter"] == pytest.approx(np.sqrt(r["efficiency_norm"] * 1.0) * 5e-3)


@pytest.mark.parametrize("dk", [80.0, 400.0, 900.0, -500.0])
def test_mismatch_matches_closed_form_bogoliubov(dk):
    # independent route: Gaussian covariance from the Hamiltonian vs analytic μ, ν (incl. g imaginary)
    L = 4e-3
    r = sl.degenerate_squeezing(1.0, L, delta_k=dk, **WG)
    mu, nu = sl.bogoliubov(r["gain_rate"], dk, L)
    assert abs(mu) ** 2 - abs(nu) ** 2 == pytest.approx(1.0, rel=1e-9)
    assert r["variance_min"] == pytest.approx((abs(mu) - abs(nu)) ** 2, rel=1e-8)
    assert r["variance_max"] == pytest.approx((abs(mu) + abs(nu)) ** 2, rel=1e-8)


def test_distributed_loss_closed_form():
    # Δk = 0: dV±/dz = (±2γ - α) V± + α  →  V± = V∞ + (1 - V∞) e^{(±2γ-α)L}, V∞ = α/(α ∓ 2γ)
    L, loss = 5e-3, 0.5 * 23.02585093
    r = sl.degenerate_squeezing(0.5, L, loss=loss, **WG)
    g = r["gain_rate"]
    for sign, key in ((-1, "variance_min"), (1, "variance_max")):
        k = sign * 2 * g - loss
        vinf = loss / (loss - sign * 2 * g)
        assert r[key] == pytest.approx(vinf + (1 - vinf) * np.exp(k * L), rel=1e-9)
    assert r["purity"] < 1


def test_detection_loss_and_phase_noise():
    r0 = sl.degenerate_squeezing(1.0, 5e-3, **WG)
    r1 = sl.degenerate_squeezing(1.0, 5e-3, detection_efficiency=0.8, **WG)
    assert r1["variance_min"] == pytest.approx(0.8 * r0["variance_min"] + 0.2, rel=1e-9)
    r2 = sl.degenerate_squeezing(1.0, 5e-3, phase_noise=0.02, **WG)
    c = 0.5 * (1 + np.exp(-2 * 0.02**2))
    assert r2["variance_min"] == pytest.approx(c * r0["variance_min"] + (1 - c) * r0["variance_max"], rel=1e-9)
    # small-angle limit: V ≈ V- + σ² V+
    assert r2["variance_min"] == pytest.approx(r0["variance_min"] + 0.02**2 * r0["variance_max"], rel=2e-3)


def test_loss_only_keeps_vacuum():
    v = sl.gaussian_evolution(-0.5 * 3.0 * np.eye(2), 3.0 * np.eye(2), 1.0)
    assert np.allclose(v, np.eye(2))


def test_two_mode_equals_degenerate_supermode():
    # (a_s ± a_i)/√2 decouple into two single-mode squeezers with the same γ, Δk and loss
    kw = dict(d_eff=14e-12, n_signal=2.14, n_pump=2.18, area_eff=2e-12)
    two = sl.two_mode_squeezing(1.0, 4e-3, wavelength_signal=1.55e-6, wavelength_idler=1.55e-6, n_idler=2.14,
                                delta_k=300.0, loss=5.0, efficiency_signal=0.9, efficiency_idler=0.9, **kw)
    one = sl.degenerate_squeezing(1.0, 4e-3, wavelength_signal=1.55e-6, delta_k=300.0, loss=5.0,
                                  detection_efficiency=0.9, **kw)
    assert two["epr_variance"] == pytest.approx(one["variance_min"], rel=1e-9)
    assert two["epr_antisqueezing_db"] == pytest.approx(one["antisqueezing_db"], rel=1e-9)


def test_two_mode_lossless_tmsv():
    t = sl.two_mode_squeezing(1.0, 5e-3, wavelength_signal=1.55e-6, wavelength_idler=1.6e-6, n_idler=2.137,
                              efficiency_signal=1.0, efficiency_idler=1.0, d_eff=14e-12, n_signal=2.14,
                              n_pump=2.18, area_eff=2e-12)
    r = t["squeezing_parameter"]
    assert t["wavelength_pump"] == pytest.approx(1 / (1 / 1.55e-6 + 1 / 1.6e-6))
    assert t["signal_photons"] == pytest.approx(np.sinh(r) ** 2, rel=1e-9)
    assert t["idler_photons"] == pytest.approx(t["signal_photons"], rel=1e-12)
    assert t["epr_variance"] == pytest.approx(np.exp(-2 * r), rel=1e-9)
    assert t["log_negativity"] == pytest.approx(2 * r / np.log(2), rel=1e-9)
    assert t["signal_noise_db"] == pytest.approx(10 * np.log10(np.cosh(2 * r)), rel=1e-9)
    assert t["purity"] == pytest.approx(1.0, rel=1e-9)


def test_two_mode_one_arm_lost_is_separable():
    t = sl.two_mode_squeezing(1.0, 5e-3, wavelength_signal=1.55e-6, wavelength_idler=1.6e-6, n_idler=2.137,
                              efficiency_signal=1.0, efficiency_idler=0.0, d_eff=14e-12, n_signal=2.14,
                              n_pump=2.18, area_eff=2e-12)
    assert t["log_negativity"] == pytest.approx(0.0, abs=1e-12)


def _opo_langevin(x, w, eta_esc):
    # independent route: Fourier-domain quantum Langevin equations for the squeezed (p) quadrature
    kap = 1.0
    k_out, k_loss, eps = eta_esc * kap, (1 - eta_esc) * kap, x * kap
    den = kap + eps - 1j * w
    c_in = 2 * k_out / den - 1
    c_b = 2 * np.sqrt(k_out * k_loss) / den
    return abs(c_in) ** 2 + abs(c_b) ** 2


@pytest.mark.parametrize("p,w,e_esc", [(0.25, 0.0, 1.0), (0.5, 0.3, 0.9), (0.81, 2.0, 0.97)])
def test_opo_spectrum_matches_langevin(p, w, e_esc):
    r = sl.opo_spectrum(p, w * 5e6, 5e6, escape_efficiency=e_esc, detection_efficiency=0.8)
    s = 0.8 * _opo_langevin(np.sqrt(p), w, e_esc) + 0.2
    assert r["variance_min"] == pytest.approx(s, rel=1e-12)


def test_opo_threshold_limit():
    r = sl.opo_spectrum(0.999999, 0.0, 1e6)
    assert r["variance_min"] == pytest.approx(0.0, abs=1e-6)       # perfect squeezing at threshold, Ω → 0
    assert r["squeezing_db_dc"] == pytest.approx(r["squeezing_db"])


def test_photon_statistics():
    r = 0.9
    p = sl.photon_number_distribution(r, 400)
    n = np.arange(p.size)
    assert p.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(p[1::2] == 0)
    assert (n * p).sum() == pytest.approx(np.sinh(r) ** 2, rel=1e-12)
    s = sl.squeezed_vacuum_statistics(r)
    assert s["photon_variance"] == pytest.approx((n**2 * p).sum() - (n * p).sum() ** 2, rel=1e-10)
    assert s["g2"] == pytest.approx((n * (n - 1) * p).sum() / s["mean_photons"] ** 2, rel=1e-10)
    assert s["p0"] == pytest.approx(1 / np.cosh(r))
    assert s["p2"] == pytest.approx(np.tanh(r) ** 2 / (2 * np.cosh(r)))      # (2!)/(2·1)² = 1/2
    q = sl.photon_number_distribution(r, 400, two_mode=True)
    assert q.sum() == pytest.approx(1.0, rel=1e-12)
    assert (n * q).sum() == pytest.approx(np.sinh(r) ** 2, rel=1e-12)


def test_infer_round_trip():
    r, eta = 1.3, 0.72
    s = sl.squeezed_vacuum_statistics(r, efficiency=eta)
    i = sl.infer_squeezing(s["squeezing_db"], s["antisqueezing_db"])
    assert i["squeezing_parameter"] == pytest.approx(r, rel=1e-12)
    assert i["efficiency"] == pytest.approx(eta, rel=1e-12)


def test_fock_parametric_limit_degenerate():
    # strong coherent pump, small τ: full quantum evolution → Bogoliubov squeezing with r = β τ
    beta, r = 8.0, 0.4
    f = sl.fock_degenerate(0.0, beta, r / beta)
    assert f["norm"] == pytest.approx(1.0, abs=1e-12)
    assert f["signal_photons"] == pytest.approx(np.sinh(r) ** 2, rel=2e-3)
    assert f["signal_squeezing_db"] == pytest.approx(f["parametric_squeezing_db"], abs=0.01)
    assert f["signal_g2"] == pytest.approx(3 + 1 / np.sinh(r) ** 2, rel=2e-3)


def test_fock_parametric_limit_improves_with_pump():
    r = 0.6
    err = [abs(sl.fock_degenerate(0.0, b, r / b)["signal_squeezing_db"] + 2 * r * sl.DB) for b in (3.0, 6.0, 10.0)]
    assert err[0] > err[1] > err[2]


def test_fock_conservation_and_unitarity():
    taus = np.linspace(0, 0.6, 7)
    f = sl.fock_degenerate(2.0, 3.0, taus)
    assert np.allclose(f["conserved_number"], 2 * 9.0 + 4.0, rtol=1e-10)
    assert np.allclose(f["norm"], 1.0, atol=1e-12)
    t = sl.fock_two_mode(5.0, taus)
    assert np.allclose(t["conserved_number"], 25.0, rtol=1e-10)


def test_fock_shg_short_time_and_fundamental_squeezing():
    # SHG from |α⟩: db/dτ = -(1/2) a² → ⟨b†b⟩ ≈ (τ/2)² ⟨a†²a²⟩ = (τ α²/2)² for small τ
    alpha, tau = 4.0, 1e-3
    f = sl.fock_degenerate(alpha, 0.0, tau)
    assert f["pump_photons"] == pytest.approx((tau * alpha**2 / 2) ** 2, rel=1e-3)
    # SHG squeezes the fundamental: linearised Heisenberg expansion to τ² gives V_min ≈ 1 - α²τ²/2
    tau = 0.01
    g = sl.fock_degenerate(alpha, 0.0, tau)
    assert 1 - g["signal_variance_min"] == pytest.approx(alpha**2 * tau**2 / 2, rel=0.02)
    assert sl.fock_degenerate(alpha, 0.0, 0.2)["signal_squeezing_db"] < -1.0


def test_fock_two_mode_parametric_limit():
    beta, r = 9.0, 0.5
    t = sl.fock_two_mode(beta, r / beta)
    assert t["signal_photons"] == pytest.approx(np.sinh(r) ** 2, rel=3e-3)
    assert t["epr_squeezing_db"] == pytest.approx(-2 * r * sl.DB, abs=0.02)
    assert t["signal_g2"] == pytest.approx(2.0, rel=3e-3)
    assert t["cross_g2"] == pytest.approx(2 + 1 / np.sinh(r) ** 2, rel=3e-3)


def test_fock_matches_gaussian_engine_through_r():
    # cross-route: full quantum ↔ Hamiltonian covariance engine, matched by r = γL = βτ
    g = sl.degenerate_squeezing(0.3, 3e-3, detection_efficiency=1.0, phase_noise=0.0, loss=0.0, **WG)
    beta = 10.0
    f = sl.fock_degenerate(0.0, beta, g["squeezing_parameter"] / beta)
    assert f["signal_squeezing_db"] == pytest.approx(g["squeezing_db"], abs=0.02)


def test_validation():
    with pytest.raises(ValueError, match="pump_power"):
        sl.degenerate_squeezing(0.0, 1e-3, **WG)
    with pytest.raises(ValueError, match="detection_efficiency"):
        sl.degenerate_squeezing(1.0, 1e-3, detection_efficiency=1.2, **WG)
    with pytest.raises(ValueError, match="pump_ratio"):
        sl.opo_spectrum(1.0, 0.0, 1e6)
    with pytest.raises(ValueError, match="beta"):
        sl.fock_degenerate(0.0, 13.0, 0.1)
    with pytest.raises(ValueError, match="mean input photon number"):
        sl.fock_degenerate(10.0, 10.0, 0.1)
    with pytest.raises(ValueError):
        sl.infer_squeezing(3.0, 6.0)
