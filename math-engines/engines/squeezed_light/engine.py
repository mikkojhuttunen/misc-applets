"""Squeezed light from χ⁽²⁾ processes in the Hamiltonian (Heisenberg-picture) formalism.

Three routes, from the same interaction Hamiltonian, each usable as a check on the others:

1. Gaussian (quadratic Hamiltonian, classical undepleted pump = parametric approximation).
   The propagation generator in the frame co-rotating with half the mismatch is
       degenerate:  G/ħ = (Δk/2) a†a + i(γ/2)(a†² - a²)
       two-mode:    G/ħ = (Δk/2)(a_s†a_s + a_i†a_i) + iγ(a_s†a_i† - a_s a_i)
   written as G/ħ = ½ Rᵀ M R in quadratures R = (x1, p1, x2, p2, …), x = a + a†, p = -i(a - a†),
   [x, p] = 2i. Heisenberg: dR/dz = A R with drift A = 2 Ω M (Ω = ⊕ [[0, 1], [-1, 0]]).
   Distributed power loss α couples each mode to a vacuum bath (Langevin):
       dV/dz = A' V + V A'ᵀ + α I,  A' = A - (α/2) I,
   V the symmetrised covariance matrix, vacuum V = I (shot noise = 1).
   Without loss V(L) = S V(0) Sᵀ with symplectic S = exp(A L): a Bogoliubov transformation.

2. Exact Fock-space evolution of the fully quantum three-wave Hamiltonian (quantum pump,
   pump depletion, no parametric approximation), in dimensionless time τ = g t:
       degenerate (SHG / DOPA):  H/ħ = i(g/2)(a†² b - a² b†),   conserved 2 b†b + a†a
       non-degenerate (NDOPA):   H/ħ = i g (a_s† a_i† b - a_s a_i b†), conserved a_s†a_s + b†b
   The conserved number splits the Hilbert space into small exactly solvable blocks; the
   only truncation is of the initial coherent states. Normalised so that da/dτ = b a†
   (degenerate) and da_s/dτ = b a_i† (two-mode): with b → β the squeezing parameter is r = β τ.

3. Closed forms: Bogoliubov coefficients with mismatch, the distributed-loss solution,
   the Collett–Gardiner OPO output spectrum, squeezed-vacuum photon statistics.

Coupling in SI: γ = √(η P_p), η = 8π² d_eff² / (ε0 c n_s n_i n_p λ_s λ_i A_eff) (the
normalised SHG efficiency P_SH = η P_FH² L² when degenerate), so r = γ L when Δk = 0.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range

EPS0 = 8.8541878128e-12
C0 = 299792458.0
DB = 10.0 / np.log(10.0)  # dB per neper of power: 10 log10(e^x) = DB x


# ---------------------------------------------------------------- linear algebra helpers

def expm(m) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series (NumPy only)."""
    m = np.asarray(m)
    nrm = np.max(np.sum(np.abs(m), axis=1)) if m.size else 0.0
    s = max(0, int(np.ceil(np.log2(nrm / 0.25))) if nrm > 0.25 else 0)
    x = m / 2.0**s
    out = np.eye(m.shape[0], dtype=np.result_type(m, float))
    term = out.copy()
    for k in range(1, 24):
        term = term @ x / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def omega(n_modes: int) -> np.ndarray:
    """Symplectic form for R = (x1, p1, …, xn, pn): Ω = ⊕ [[0, 1], [-1, 0]]."""
    return np.kron(np.eye(n_modes), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def hamiltonian_degenerate(gamma, delta_k) -> np.ndarray:
    """M of G/ħ = ½ Rᵀ M R = (Δk/2) a†a + i(γ/2)(a†² - a²) (constant dropped)."""
    return np.array([[delta_k / 4, gamma / 2], [gamma / 2, delta_k / 4]], dtype=float)


def hamiltonian_two_mode(gamma, delta_k) -> np.ndarray:
    """M of G/ħ = (Δk/2)(a_s†a_s + a_i†a_i) + iγ(a_s†a_i† - a_s a_i), R = (x_s, p_s, x_i, p_i)."""
    m = np.eye(4) * (delta_k / 4)
    m[0, 3] = m[3, 0] = gamma / 2
    m[1, 2] = m[2, 1] = gamma / 2
    return m


def drift_matrix(hamiltonian) -> np.ndarray:
    """Heisenberg drift A = 2 Ω M for dR/dz = A R (from [x, p] = 2i)."""
    m = np.asarray(hamiltonian, dtype=float)
    return 2.0 * omega(m.shape[0] // 2) @ m


def gaussian_evolution(drift, diffusion, length, cov0=None) -> np.ndarray:
    """Solve dV/dz = A V + V Aᵀ + D over `length` (Van Loan block exponential).

    Returns V(L) = e^{AL} V0 e^{AᵀL} + ∫₀ᴸ e^{As} D e^{Aᵀs} ds; V0 defaults to vacuum (I)."""
    a = np.asarray(drift, dtype=float)
    d = np.asarray(diffusion, dtype=float)
    n = a.shape[0]
    v0 = np.eye(n) if cov0 is None else np.asarray(cov0, dtype=float)
    blk = np.zeros((2 * n, 2 * n))
    blk[:n, :n] = -a
    blk[:n, n:] = d
    blk[n:, n:] = a.T
    e = expm(blk * length)
    f = e[n:, n:]                 # e^{Aᵀ L}
    q = f.T @ e[:n, n:]           # noise integral
    v = f.T @ v0 @ f + q
    return 0.5 * (v + v.T)


def lossy_channel(cov, efficiency) -> np.ndarray:
    """Beam-splitter loss mixing in vacuum: V → η V + (1 - η) I (η per mode, scalar or list)."""
    v = np.asarray(cov, dtype=float)
    eta = np.broadcast_to(np.asarray(efficiency, dtype=float), (v.shape[0] // 2,))
    t = np.sqrt(np.repeat(eta, 2))
    return t[:, None] * v * t[None, :] + np.diag(1.0 - np.repeat(eta, 2))


def quadrature_extremes(cov2) -> tuple[float, float, float]:
    """(V_min, V_max, angle of the squeezed quadrature) of a single-mode 2×2 covariance.
    angle θ is that of X_θ = x cos θ + p sin θ, in (-π/2, π/2]."""
    w, q = np.linalg.eigh(np.asarray(cov2, dtype=float))
    th = float(np.arctan2(q[1, 0], q[0, 0]))
    th = (th + np.pi / 2) % np.pi - np.pi / 2
    return float(w[0]), float(w[1]), th


def with_phase_noise(v_min, v_max, phase_rms):
    """Gaussian phase jitter of rms σ on the measured quadrature:
    V = V_min c + V_max (1 - c), c = ⟨cos² θ⟩ = (1 + e^{-2σ²}) / 2."""
    c = 0.5 * (1.0 + np.exp(-2.0 * np.asarray(phase_rms, dtype=float) ** 2))
    return v_min * c + v_max * (1 - c), v_max * c + v_min * (1 - c)


def log_negativity(cov) -> float:
    """Two-mode logarithmic negativity E_N (ebits) from V (vacuum = I): Σ max(0, -log2 ν̃)
    over the symplectic eigenvalues ν̃ of the partially transposed covariance."""
    v = np.array(cov, dtype=float)
    pt = np.diag([1.0, 1.0, 1.0, -1.0])
    vt = pt @ v @ pt
    nu = np.sort(np.abs(np.linalg.eigvals(1j * omega(2) @ vt)))[::2]
    return float(np.sum(np.maximum(0.0, -np.log2(nu))))


def coupling_efficiency(d_eff, wavelength_signal, wavelength_idler, n_signal, n_idler, n_pump, area_eff):
    """Normalised parametric efficiency η = 8π² d² / (ε0 c n_s n_i n_p λ_s λ_i A_eff) in 1/(W m²);
    γ = √(η P_p). Degenerate: equals the SHG efficiency P_SH = η P² L²."""
    return 8 * np.pi**2 * d_eff**2 / (EPS0 * C0 * n_signal * n_idler * n_pump * wavelength_signal * wavelength_idler * area_eff)


def bogoliubov(gamma, delta_k, length):
    """Closed-form lossless coefficients a(L) = μ a(0) + ν a†(0) (rotating frame), with
    g = √(γ² - Δk²/4): μ = cosh gL - i (Δk/2g) sinh gL, ν = (γ/g) sinh gL. Returns (μ, ν)."""
    g = np.sqrt(complex(gamma**2 - delta_k**2 / 4))
    if abs(g * length) < 1e-12:
        sh_g = length
    else:
        sh_g = np.sinh(g * length) / g
    mu = np.cosh(g * length) - 1j * (delta_k / 2) * sh_g
    nu = gamma * sh_g
    return complex(mu), complex(nu)


def _common_checks(pump_power, length, d_eff, n_signal, n_pump, area_eff, loss, phase_noise):
    require_positive(pump_power=pump_power, length=length, d_eff=d_eff, n_signal=n_signal, n_pump=n_pump, area_eff=area_eff)
    require_nonnegative(loss=loss, phase_noise=phase_noise)


# ---------------------------------------------------------------- Gaussian (parametric) engines

def degenerate_squeezing(pump_power, length, d_eff, wavelength_signal, n_signal, n_pump, area_eff,
                         delta_k=0.0, loss=0.0, detection_efficiency=1.0, phase_noise=0.0) -> Result:
    """Single-pass degenerate parametric amplifier (ω_p = 2ω_s) from vacuum: quadrature squeezing.

    Hamiltonian route: M → drift A = 2ΩM → covariance with distributed loss α (1/m, power),
    then detection loss and Gaussian phase jitter. Lossless limit: V_min = (|μ| - |ν|)²,
    V_max = (|μ| + |ν|)², = e^{∓2r} at Δk = 0."""
    _common_checks(pump_power, length, d_eff, n_signal, n_pump, area_eff, loss, phase_noise)
    require_positive(wavelength_signal=wavelength_signal)
    require_range("detection_efficiency", detection_efficiency, 0.0, 1.0)
    if not np.isfinite(delta_k):
        raise ValueError(f"delta_k must be finite, got {delta_k!r}")
    eta = coupling_efficiency(d_eff, wavelength_signal, wavelength_signal, n_signal, n_signal, n_pump, area_eff)
    gamma = float(np.sqrt(eta * pump_power))
    a = drift_matrix(hamiltonian_degenerate(gamma, delta_k)) - 0.5 * loss * np.eye(2)
    v = gaussian_evolution(a, loss * np.eye(2), length)
    v_lab = _rotate(v, delta_k * length / 2)
    gmin, gmax, ang = quadrature_extremes(v_lab)
    dmin, dmax, _ = quadrature_extremes(lossy_channel(v_lab, detection_efficiency))
    mmin, mmax = with_phase_noise(dmin, dmax, phase_noise)
    mu, nu = bogoliubov(gamma, delta_k, length)
    return Result(
        values={
            "wavelength_pump": wavelength_signal / 2,
            "efficiency_norm": eta,
            "gain_rate": gamma,
            "squeezing_parameter": gamma * length,
            "squeezing_parameter_eff": float(np.arcsinh(abs(nu))),
            "mean_photons": (np.trace(v) - 2) / 4,
            "variance_min_generated": gmin,
            "variance_max_generated": gmax,
            "squeezing_db_generated": DB * np.log(gmin),
            "variance_min": mmin,
            "variance_max": mmax,
            "squeezing_db": DB * np.log(mmin),
            "antisqueezing_db": DB * np.log(mmax),
            "purity": 1.0 / np.sqrt(np.linalg.det(v)),
            "squeezing_angle": ang,
            "covariance": v_lab,
        },
        units={
            "wavelength_pump": "m", "efficiency_norm": "1/(W m^2)", "gain_rate": "1/m", "squeezing_parameter": "",
            "squeezing_parameter_eff": "", "mean_photons": "", "variance_min_generated": "", "variance_max_generated": "",
            "squeezing_db_generated": "dB", "variance_min": "", "variance_max": "", "squeezing_db": "dB",
            "antisqueezing_db": "dB", "purity": "", "squeezing_angle": "rad", "covariance": "",
        },
        assumptions=[
            "Momentum generator G/ħ = (Δk/2) a†a + i(γ/2)(a†² - a²); undepleted classical pump (parametric approximation)",
            "Single spatial mode, quasi-cw (no walk-off or group-velocity mismatch); pump_power is the peak power for pulses",
            "Variances relative to shot noise (vacuum = 1); x = a + a†, p = -i(a - a†), pump phase 0",
            "Distributed signal loss as a continuous beam splitter to vacuum; pump loss neglected",
            "squeezing_parameter_eff = asinh|ν| of the lossless Bogoliubov transformation (includes Δk)",
            "Detection: V → ηV + 1 - η, then Gaussian phase jitter of rms phase_noise",
        ],
    )


def _rotate(v, phi):
    """Covariance in the frame rotated by phase φ (a → a e^{iφ}) for every mode."""
    c, s = np.cos(phi), np.sin(phi)
    r = np.kron(np.eye(v.shape[0] // 2), np.array([[c, -s], [s, c]]))
    return r @ v @ r.T


def two_mode_squeezing(pump_power, length, d_eff, wavelength_signal, wavelength_idler, n_signal, n_idler, n_pump,
                       area_eff, delta_k=0.0, loss=0.0, efficiency_signal=1.0, efficiency_idler=1.0) -> Result:
    """Non-degenerate parametric amplifier (ω_p = ω_s + ω_i) from vacuum: two-mode squeezing.

    EPR variance: smallest variance of (X_s,θ ± X_i,θ)/√2 over θ (vacuum = 1; < 1 is Duan-
    inseparable). Logarithmic negativity from the partially transposed covariance."""
    _common_checks(pump_power, length, d_eff, n_signal, n_pump, area_eff, loss, 0.0)
    require_positive(wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, n_idler=n_idler)
    require_range("efficiency_signal", efficiency_signal, 0.0, 1.0)
    require_range("efficiency_idler", efficiency_idler, 0.0, 1.0)
    eta = coupling_efficiency(d_eff, wavelength_signal, wavelength_idler, n_signal, n_idler, n_pump, area_eff)
    gamma = float(np.sqrt(eta * pump_power))
    a = drift_matrix(hamiltonian_two_mode(gamma, delta_k)) - 0.5 * loss * np.eye(4)
    v = _rotate(gaussian_evolution(a, loss * np.eye(4), length), delta_k * length / 2)
    vd = lossy_channel(v, [efficiency_signal, efficiency_idler])
    epr_min, epr_max = _epr(vd)
    return Result(
        values={
            "wavelength_pump": 1.0 / (1.0 / wavelength_signal + 1.0 / wavelength_idler),
            "efficiency_norm": eta,
            "gain_rate": gamma,
            "squeezing_parameter": gamma * length,
            "signal_photons": (v[0, 0] + v[1, 1] - 2) / 4,
            "idler_photons": (v[2, 2] + v[3, 3] - 2) / 4,
            "epr_variance": epr_min,
            "epr_squeezing_db": DB * np.log(epr_min),
            "epr_antisqueezing_db": DB * np.log(epr_max),
            "signal_noise_db": DB * np.log(quadrature_extremes(vd[:2, :2])[1]),
            "log_negativity": log_negativity(vd),
            "purity": 1.0 / np.sqrt(np.linalg.det(vd)),
            "covariance": vd,
        },
        units={
            "wavelength_pump": "m", "efficiency_norm": "1/(W m^2)", "gain_rate": "1/m", "squeezing_parameter": "",
            "signal_photons": "", "idler_photons": "", "epr_variance": "", "epr_squeezing_db": "dB",
            "epr_antisqueezing_db": "dB", "signal_noise_db": "dB", "log_negativity": "ebit", "purity": "", "covariance": "",
        },
        assumptions=[
            "Momentum generator G/ħ = (Δk/2)(a_s†a_s + a_i†a_i) + iγ(a_s†a_i† - a_s a_i); undepleted classical pump",
            "Signal and idler distinguishable (frequency or polarisation), one spatial mode each, quasi-cw",
            "Same distributed loss for both waves; separate detection efficiencies, applied after generation",
            "R = (x_s, p_s, x_i, p_i), vacuum covariance I; photon numbers before detection loss",
            "signal_noise_db: local (thermal) quadrature noise of the signal alone after detection",
        ],
    )


def _epr(v):
    """Min and max of Var[(X_s,θ ± X_i,θ)/√2] over θ and the sign (4×4 covariance)."""
    s = v[:2, :2] + v[2:, 2:]
    c = v[:2, 2:] + v[2:, :2]
    w_plus = np.linalg.eigvalsh(0.5 * (s + c))
    w_minus = np.linalg.eigvalsh(0.5 * (s - c))
    return float(min(w_plus[0], w_minus[0])), float(max(w_plus[1], w_minus[1]))


def opo_spectrum(pump_ratio, frequency, linewidth, escape_efficiency=1.0, detection_efficiency=1.0) -> Result:
    """Output quadrature noise spectra of a degenerate OPO below threshold (Collett–Gardiner).

    System Hamiltonian H/ħ = i(ε/2)(a†² - a²), input-output with total field decay rate κ
    (= half width at half maximum, rad/s = 2π linewidth); x = √(P/P_th) = ε/κ:
        S_∓(Ω) = 1 ∓ η 4x / ((1 ± x)² + (Ω/κ)²),  η = η_esc η_det."""
    require_positive(linewidth=linewidth)
    require_nonnegative(frequency=frequency, pump_ratio=pump_ratio)
    require_range("pump_ratio", pump_ratio, 0.0, 0.999999)
    require_range("escape_efficiency", escape_efficiency, 0.0, 1.0)
    require_range("detection_efficiency", detection_efficiency, 0.0, 1.0)
    x = np.sqrt(pump_ratio)
    eta = escape_efficiency * detection_efficiency
    w2 = (np.asarray(frequency, dtype=float) / linewidth) ** 2
    s_min = 1 - eta * 4 * x / ((1 + x) ** 2 + w2)
    s_max = 1 + eta * 4 * x / ((1 - x) ** 2 + w2)
    return Result(
        values={
            "variance_min": s_min, "variance_max": s_max,
            "squeezing_db": DB * np.log(s_min), "antisqueezing_db": DB * np.log(s_max),
            "squeezing_db_dc": DB * np.log(1 - eta * 4 * x / (1 + x) ** 2),
            "total_efficiency": eta,
        },
        units={"variance_min": "", "variance_max": "", "squeezing_db": "dB", "antisqueezing_db": "dB",
               "squeezing_db_dc": "dB", "total_efficiency": ""},
        assumptions=[
            "Single-mode degenerate cavity, linearised quantum Langevin equations, classical undepleted pump",
            "linewidth is the cavity HWHM in Hz (field decay rate κ/2π); frequency is the sideband offset Ω/2π",
            "pump_ratio = P/P_th < 1; escape efficiency κ_out/κ; noise relative to shot noise",
        ],
    )


# ---------------------------------------------------------------- statistics and inference

def squeezed_vacuum_statistics(squeezing_parameter, efficiency=1.0) -> Result:
    """Ideal single-mode squeezed vacuum S(r)|0⟩ and two-mode squeezed vacuum with the same r.

    Single mode: ⟨n⟩ = sinh² r, Var n = 2 sinh² r cosh² r, g⁽²⁾ = 3 + 1/sinh² r,
    P(2n) = (2n)! tanh²ⁿ r / (2ⁿ n!)² / cosh r. Two-mode: thermal marginals (g⁽²⁾ = 2),
    g⁽²⁾_si = 2 + 1/sinh² r, P(n, n) = tanh²ⁿ r / cosh² r. Quadratures after loss η."""
    require_nonnegative(squeezing_parameter=squeezing_parameter)
    require_range("efficiency", efficiency, 0.0, 1.0)
    r = float(squeezing_parameter)
    n = np.sinh(r) ** 2
    vmin = efficiency * np.exp(-2 * r) + 1 - efficiency
    vmax = efficiency * np.exp(2 * r) + 1 - efficiency
    with np.errstate(divide="ignore"):
        g2 = 3 + 1 / n if n > 0 else np.inf
        g2x = 2 + 1 / n if n > 0 else np.inf
    p = photon_number_distribution(r, 4)
    return Result(
        values={
            "mean_photons": n, "photon_variance": 2 * n * (n + 1), "g2": g2,
            "p0": p[0], "p2": p[2], "p4": p[4],
            "variance_min": vmin, "variance_max": vmax,
            "squeezing_db": DB * np.log(vmin), "antisqueezing_db": DB * np.log(vmax),
            "g2_cross_two_mode": g2x, "pair_probability_two_mode": np.tanh(r) ** 2 / np.cosh(r) ** 2,
        },
        units={k: ("dB" if k.endswith("_db") else "") for k in (
            "mean_photons", "photon_variance", "g2", "p0", "p2", "p4", "variance_min", "variance_max",
            "squeezing_db", "antisqueezing_db", "g2_cross_two_mode", "pair_probability_two_mode")},
        assumptions=[
            "Pure states S(r)|0⟩ and exp[r(a†b† - ab)]|0,0⟩; photon statistics before loss",
            "Quadrature variances after a beam-splitter loss of transmission efficiency (vacuum = 1)",
        ],
    )


def photon_number_distribution(squeezing_parameter, n_max, two_mode=False) -> np.ndarray:
    """P(n), n = 0…n_max, of S(r)|0⟩ (odd n vanish) or of one arm of the two-mode squeezed vacuum."""
    r = float(squeezing_parameter)
    n = np.arange(int(n_max) + 1)
    t2 = np.tanh(r) ** 2
    if two_mode:
        return t2**n / np.cosh(r) ** 2
    p = np.zeros(n.size)
    m = n[::2] // 2
    # (2m)! / (2^m m!)^2 = C(2m, m) / 4^m, built iteratively for stability
    w = np.ones(m.size)
    for k in range(1, m.size):
        w[k] = w[k - 1] * (2 * k - 1) / (2 * k)
    p[::2] = w * t2**m / np.cosh(r)
    return p


def infer_squeezing(squeezing_db, antisqueezing_db) -> Result:
    """Pure squeezing parameter and total efficiency from a measured (squeezing, anti-squeezing)
    pair, assuming a pure squeezed state followed by loss: V_∓ = η e^{∓2r} + 1 - η."""
    if not (squeezing_db < 0 < antisqueezing_db):
        raise ValueError("need squeezing_db < 0 < antisqueezing_db")
    a = 10 ** (squeezing_db / 10) - 1
    b = 10 ** (antisqueezing_db / 10) - 1
    u = -b / a
    if u <= 1:
        raise ValueError("anti-squeezing must exceed |squeezing| in noise power for a pure state with loss")
    r = 0.5 * np.log(u)
    eta = b / (u - 1)
    return Result(
        values={"squeezing_parameter": r, "efficiency": eta, "squeezing_db_lossless": -2 * r * DB,
                "mean_photons_generated": np.sinh(r) ** 2},
        units={"squeezing_parameter": "", "efficiency": "", "squeezing_db_lossless": "dB", "mean_photons_generated": ""},
        assumptions=["Pure squeezed vacuum + loss only (no phase noise, no excess noise)", "Inputs are noise powers relative to shot noise, in dB"],
    )


# ---------------------------------------------------------------- exact Fock-space evolution

FOCK_MAX_AMPLITUDE = 12.0      # |α|, |β| each; blocks grow as the mean photon numbers
FOCK_MAX_PHOTONS = 150.0       # |α|² + |β|², keeps one call to a few seconds (also in Pyodide)


def _coherent(alpha, n_cut):
    c = np.zeros(n_cut + 1, dtype=complex)
    c[0] = np.exp(-abs(alpha) ** 2 / 2)
    for n in range(1, n_cut + 1):
        c[n] = c[n - 1] * alpha / np.sqrt(n)
    return c


def _cutoff(alpha):
    a = abs(alpha)
    return int(np.ceil(a * a + 9 * a + 14))


def _evolve_block(t_mat, psi0, tau):
    """ψ(τ) = exp(τ T) ψ0 for real antisymmetric T, for every τ (rows)."""
    w, q = np.linalg.eigh(1j * t_mat)              # iT Hermitian: T = -i Q w Q†
    c = q.conj().T @ psi0
    return (np.exp(-1j * np.outer(tau, w)) * c[None, :]) @ q.T


def fock_degenerate_moments(alpha, beta, interaction) -> dict:
    """Exact moments under H/ħ = i(g/2)(a†² b - a² b†) from |α⟩_a |β⟩_b at τ = g t (array allowed).

    Basis blocks K = 2 n_b + n_a, states |n_b = j, n_a = K - 2j⟩. Returns arrays over τ:
    na, nb, a, a2, b, b2, na2 (⟨a†²a²⟩), nb2 (⟨b†²b²⟩), norm, dim."""
    tau = np.atleast_1d(np.asarray(interaction, dtype=float))
    ca, cb = _coherent(alpha, _cutoff(alpha)), _coherent(beta, _cutoff(beta))
    na_c, nb_c = ca.size - 1, cb.size - 1
    psi = {}
    dim = 0
    for k in range(2 * nb_c + na_c + 1):
        j = np.arange(k // 2 + 1)
        n = k - 2 * j
        ok = (j <= nb_c) & (n <= na_c)
        p0 = np.zeros(j.size, dtype=complex)
        p0[ok] = cb[j[ok]] * ca[n[ok]]
        if np.linalg.norm(p0) < 1e-15:
            continue
        t = np.zeros((j.size, j.size))
        jj = j[1:]
        s = np.sqrt(jj * (n[1:] + 1.0) * (n[1:] + 2.0))
        t[jj - 1, jj] = s
        t[jj, jj - 1] = -s
        psi[k] = _evolve_block(t, p0, tau / 2)
        dim += j.size
    m = {x: np.zeros(tau.size, dtype=complex) for x in ("na", "nb", "a", "a2", "b", "b2", "na2", "nb2", "norm")}
    for k, ps in psi.items():
        j = np.arange(ps.shape[1])
        n = k - 2 * j
        pr = np.abs(ps) ** 2
        m["norm"] += pr.sum(1)
        m["na"] += pr @ n
        m["nb"] += pr @ j
        m["na2"] += pr @ (n * (n - 1.0))
        m["nb2"] += pr @ (j * (j - 1.0))
        if k - 1 in psi:                         # a: |j, n⟩ → √n |j, n-1⟩ in block K-1
            q = psi[k - 1]
            jj = j[: q.shape[1]]
            m["a"] += np.sum(q[:, jj].conj() * np.sqrt(n[jj]) * ps[:, jj], 1)
        if k - 2 in psi:
            q = psi[k - 2]
            jj = j[: q.shape[1]]                 # a²: same j, n ≥ 2
            m["a2"] += np.sum(q[:, jj].conj() * np.sqrt(n[jj] * (n[jj] - 1.0)) * ps[:, jj], 1)
            jb = j[1:]                           # b: |j, n⟩ → √j |j-1, n⟩ in block K-2
            m["b"] += np.sum(q[:, jb - 1].conj() * np.sqrt(jb) * ps[:, jb], 1)
        if k - 4 in psi:
            q = psi[k - 4]
            jb = j[2:]
            m["b2"] += np.sum(q[:, jb - 2].conj() * np.sqrt(jb * (jb - 1.0)) * ps[:, jb], 1)
    out = {x: (v.real if x in ("na", "nb", "na2", "nb2", "norm") else v) for x, v in m.items()}
    out["dim"] = dim
    return out


def _vmin(n, a1, a2):
    return 1 + 2 * (n - np.abs(a1) ** 2) - 2 * np.abs(a2 - a1**2)


def _vmax(n, a1, a2):
    return 1 + 2 * (n - np.abs(a1) ** 2) + 2 * np.abs(a2 - a1**2)


def _check_amplitudes(**kw):
    for name, v in kw.items():
        require_range(name, abs(v), 0.0, FOCK_MAX_AMPLITUDE)
    if sum(abs(v) ** 2 for v in kw.values()) > FOCK_MAX_PHOTONS:
        raise ValueError(f"mean input photon number {' + '.join(f'|{k}|²' for k in kw)} must not exceed {FOCK_MAX_PHOTONS:g}")


def fock_degenerate(alpha, beta, interaction) -> Result:
    """Fully quantum degenerate three-wave mixing (DOPA or SHG) beyond the parametric approximation.

    Start |α⟩ (signal/fundamental, ω) ⊗ |β⟩ (pump/second harmonic, 2ω), evolve under
    H/ħ = i(g/2)(a†² b - a² b†) to τ = g t. α = 0, β > 0: degenerate OPA from vacuum, compare
    with r = β τ. β = 0, α > 0: SHG, which squeezes the fundamental."""
    _check_amplitudes(alpha=alpha, beta=beta)
    require_nonnegative(interaction=interaction)
    m = fock_degenerate_moments(alpha, beta, interaction)
    va_min, va_max = _vmin(m["na"], m["a"], m["a2"]), _vmax(m["na"], m["a"], m["a2"])
    vb_min = _vmin(m["nb"], m["b"], m["b2"])
    r_par = abs(beta) * np.asarray(interaction, dtype=float)
    sq = lambda x: x[0] if np.ndim(interaction) == 0 else x  # noqa: E731
    na, nb = m["na"], m["nb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g2a = np.where(na > 0, m["na2"] / na**2, np.nan)
    return Result(
        values={
            "signal_photons": sq(na), "pump_photons": sq(nb),
            "signal_variance_min": sq(va_min), "signal_squeezing_db": sq(DB * np.log(va_min)),
            "signal_antisqueezing_db": sq(DB * np.log(va_max)),
            "pump_squeezing_db": sq(DB * np.log(vb_min)),
            "signal_g2": sq(g2a),
            "pump_depletion": sq(1 - nb / abs(beta) ** 2) if beta != 0 else sq(np.zeros_like(nb)),
            "parametric_squeezing_parameter": r_par,
            "parametric_squeezing_db": -2 * r_par * DB,
            "conserved_number": sq(2 * nb + na),
            "norm": sq(m["norm"]),
            "hilbert_dimension": m["dim"],
        },
        units={"signal_photons": "", "pump_photons": "", "signal_variance_min": "", "signal_squeezing_db": "dB",
               "signal_antisqueezing_db": "dB", "pump_squeezing_db": "dB", "signal_g2": "", "pump_depletion": "",
               "parametric_squeezing_parameter": "", "parametric_squeezing_db": "dB", "conserved_number": "",
               "norm": "", "hilbert_dimension": ""},
        assumptions=[
            "H/ħ = i(g/2)(a†² b - a² b†), two single modes, exact evolution in blocks of fixed 2 b†b + a†a",
            "τ = g t dimensionless; da/dτ = b a†, so the parametric approximation gives r = |β| τ",
            "Coherent inputs truncated at n ≤ |α|² + 9|α| + 14 (tail < 1e-12); 'norm' reports the kept weight",
            "Quadrature variances minimised over phase, vacuum = 1; pump_squeezing_db for the 2ω mode",
        ],
    )


def fock_two_mode_moments(beta, interaction) -> dict:
    """Exact moments under H/ħ = i g (a_s†a_i† b - a_s a_i b†) from |0, 0⟩ |β⟩, τ = g t.
    Blocks m = n_s + n_b, states |k, k, m - k⟩. Returns ns, nb, sa (⟨a_s a_i⟩), b, ns2 (⟨n_s²⟩), norm."""
    tau = np.atleast_1d(np.asarray(interaction, dtype=float))
    cb = _coherent(beta, _cutoff(beta))
    psi = {}
    for mm in range(cb.size):
        if abs(cb[mm]) < 1e-15:
            continue
        k = np.arange(mm + 1)
        t = np.zeros((k.size, k.size))
        kk = k[:-1]
        s = (kk + 1.0) * np.sqrt(mm - kk)
        t[kk + 1, kk] = s
        t[kk, kk + 1] = -s
        p0 = np.zeros(k.size, dtype=complex)
        p0[0] = cb[mm]
        psi[mm] = _evolve_block(t, p0, tau)
    m = {x: np.zeros(tau.size, dtype=complex) for x in ("ns", "nb", "sa", "b", "ns2", "norm")}
    for mm, ps in psi.items():
        k = np.arange(mm + 1)
        pr = np.abs(ps) ** 2
        m["norm"] += pr.sum(1)
        m["ns"] += pr @ k
        m["nb"] += pr @ (mm - k)
        m["ns2"] += pr @ (k * k * 1.0)
        if mm - 1 in psi:
            q = psi[mm - 1]
            m["sa"] += np.sum(q[:, k[:-1]].conj() * k[1:] * ps[:, k[1:]], 1)   # a_s a_i: k → k-1
            m["b"] += np.sum(q[:, k[:-1]].conj() * np.sqrt(mm - k[:-1]) * ps[:, k[:-1]], 1)
    return {x: (v.real if x in ("ns", "nb", "ns2", "norm") else v) for x, v in m.items()}


def fock_two_mode(beta, interaction) -> Result:
    """Fully quantum non-degenerate parametric down-conversion with a depletable quantum pump.

    Start |0⟩_s |0⟩_i |β⟩_p, evolve under H/ħ = i g (a_s†a_i† b - a_s a_i b†) to τ = g t.
    Compare with the two-mode squeezed vacuum of r = |β| τ."""
    _check_amplitudes(beta=beta)
    require_nonnegative(interaction=interaction)
    m = fock_two_mode_moments(beta, interaction)
    ns = m["ns"]
    epr = 1 + 2 * ns - 2 * np.abs(m["sa"])
    with np.errstate(divide="ignore", invalid="ignore"):
        g2s = np.where(ns > 0, (m["ns2"] - ns) / ns**2, np.nan)
        g2x = np.where(ns > 0, m["ns2"] / ns**2, np.nan)
    r_par = abs(beta) * np.asarray(interaction, dtype=float)
    sq = lambda x: x[0] if np.ndim(interaction) == 0 else x  # noqa: E731
    return Result(
        values={
            "signal_photons": sq(ns), "pump_photons": sq(m["nb"]),
            "epr_variance": sq(epr), "epr_squeezing_db": sq(DB * np.log(epr)),
            "signal_g2": sq(g2s), "cross_g2": sq(g2x),
            "parametric_squeezing_parameter": r_par, "parametric_signal_photons": np.sinh(r_par) ** 2,
            "parametric_squeezing_db": -2 * r_par * DB,
            "conserved_number": sq(ns + m["nb"]), "norm": sq(m["norm"]),
        },
        units={"signal_photons": "", "pump_photons": "", "epr_variance": "", "epr_squeezing_db": "dB", "signal_g2": "",
               "cross_g2": "", "parametric_squeezing_parameter": "", "parametric_signal_photons": "",
               "parametric_squeezing_db": "dB", "conserved_number": "", "norm": ""},
        assumptions=[
            "H/ħ = i g (a_s†a_i† b - a_s a_i b†), three single modes, exact evolution in blocks of fixed n_s + n_b",
            "τ = g t dimensionless; parametric approximation r = |β| τ",
            "EPR variance of (X_s ± X_i)/√2 minimised over phase, vacuum = 1",
        ],
    )
