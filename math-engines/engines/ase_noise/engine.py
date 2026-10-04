"""Level-C ASE noise core: linear phase-insensitive multimode channels on Gaussian states.

State per frequency bin and polarisation (one bin of width Δν; ASE assumed flat over it):

    alpha   complex vector, N modes, coherent signal amplitude in sqrt(photons/s):
            P_signal = h ν |alpha|²
    C       N×N Hermitian, normally ordered excess correlation <δa_j† δa_i> in photons
            per mode per second per Hz (dimensionless occupation). Vacuum is C = 0.
            ASE power spectral density in mode i: ρ_i = h ν C_ii  [W/Hz].

Channel a_out = T a + f with noise f independent of the input:

    alpha' = T alpha,   C' = T C T† + D,   D = <f f†> (normally ordered, PSD).

Constructors:
    gain_channel   T = U diag(√G) U†,  D = U diag(n_sp (G - 1)) U†  (Caves minimum for n_sp = 1)
    loss_channel   T = √η (scalar, per mode or matrix with singular values ≤ 1),  D = 0
    coupling       T = U unitary,  D = 0
    aperture       T = overlap matrix of a circular aperture in the LG basis,  D = 0

D = 0 for passive elements is exact at optical frequencies (thermal occupation h ν >> k T).

Detection (square-law, Olsson 1989, generalised to a mode projector W):
    I_s      = R h ν |W T alpha|²
    I_ase    = R h ν n_pol tr(W C W) B_o
    σ²_shot  = 2 e (I_s + I_ase) B_e
    σ²_s-sp  = 4 R² (h ν)² Re(alpha† W C W alpha) B_e          (signal polarisation only)
    σ²_sp-sp = R² (h ν)² n_pol tr((W C W)²) (2 B_o - B_e) B_e
so the sp-sp term counts modes through tr((WCW)²), i.e. M_eff = tr(WCW)² / tr((WCW)²).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

H = 6.62607015e-34
C0 = 299792458.0
QE = 1.602176634e-19


@dataclass(frozen=True)
class ModeState:
    """Signal amplitude vector (sqrt(photons/s)) and ASE correlation matrix (photons/s/Hz)."""

    alpha: np.ndarray
    corr: np.ndarray

    @property
    def n_modes(self) -> int:
        return self.alpha.shape[0]


@dataclass(frozen=True)
class Channel:
    """Linear map a -> T a + f with normally ordered noise PSD matrix D = <f f†>."""

    T: np.ndarray
    D: np.ndarray


def photon_energy(wavelength) -> float:
    require_positive(wavelength=wavelength)
    return H * C0 / wavelength


def coherent_state(n_modes, signal_power, wavelength, signal_mode=None) -> ModeState:
    """Coherent signal of power `signal_power` in mode vector `signal_mode` (default mode 0), no ASE."""
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes!r}")
    require_nonnegative(signal_power=signal_power)
    if signal_mode is None:
        u = np.zeros(n_modes, dtype=complex)
        u[0] = 1.0
    else:
        u = np.asarray(signal_mode, dtype=complex)
        if u.shape != (n_modes,) or np.linalg.norm(u) == 0:
            raise ValueError("signal_mode must be a non-zero vector of length n_modes")
        u = u / np.linalg.norm(u)
    amp = np.sqrt(signal_power / photon_energy(wavelength))
    return ModeState(alpha=amp * u, corr=np.zeros((n_modes, n_modes), dtype=complex))


def _as_modes(x, n_modes, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_modes, float(arr))
    if arr.shape != (n_modes,):
        raise ValueError(f"{name} must be a scalar or have length {n_modes}")
    return arr


def _check_unitary(U):
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or not np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-9):
        raise ValueError("unitary must be a square unitary matrix")
    return U


def gain_channel(gains, n_sp=1.0, n_modes=None, unitary=None) -> Channel:
    """Phase-insensitive amplifier with modal power gains G_k ≥ 1 in the basis given by the
    columns of `unitary` (default: the state basis). Added noise n_sp (G_k - 1) per mode."""
    if n_modes is None:
        n_modes = np.asarray(gains).size
    g = _as_modes(gains, int(n_modes), "gains")
    nsp = _as_modes(n_sp, g.size, "n_sp")
    if np.any(~np.isfinite(g)) or np.any(g < 1):
        raise ValueError(f"gains must be >= 1 (use loss_channel for loss), got {gains!r}")
    if np.any(~np.isfinite(nsp)) or np.any(nsp < 1):
        raise ValueError(f"n_sp must be >= 1 (quantum limit), got {n_sp!r}")
    T = np.diag(np.sqrt(g)).astype(complex)
    D = np.diag(nsp * (g - 1)).astype(complex)
    if unitary is not None:
        U = _check_unitary(unitary)
        T, D = U @ T @ U.conj().T, U @ D @ U.conj().T
    return Channel(T=T, D=D)


def loss_channel(transmission, n_modes=None) -> Channel:
    """Passive element: scalar or per-mode power transmission η, or a full amplitude matrix
    with singular values ≤ 1. Vacuum enters through the lost ports (D = 0)."""
    t = np.asarray(transmission)
    if t.ndim == 2:
        T = t.astype(complex)
        if np.linalg.svd(T, compute_uv=False).max() > 1 + 1e-9:
            raise ValueError("transmission matrix has a singular value > 1 (not passive)")
    else:
        if n_modes is None:
            n_modes = t.size
        eta = _as_modes(t, int(n_modes), "transmission")
        require_range("transmission", eta, 0.0, 1.0)
        T = np.diag(np.sqrt(eta)).astype(complex)
    return Channel(T=T, D=np.zeros_like(T))


def coupling_channel(unitary) -> Channel:
    """Lossless mode mixing (crosstalk, twisted fibre, lens system)."""
    U = _check_unitary(unitary)
    return Channel(T=U, D=np.zeros_like(U))


def compose(*channels: Channel) -> Channel:
    """Channels in propagation order: compose(c1, c2) applies c1 first."""
    if not channels:
        raise ValueError("compose needs at least one channel")
    T, D = channels[0].T, channels[0].D
    for c in channels[1:]:
        if c.T.shape[1] != T.shape[0]:
            raise ValueError("channel dimensions do not match")
        T, D = c.T @ T, c.T @ D @ c.T.conj().T + c.D
    return Channel(T=T, D=D)


def propagate(state: ModeState, *channels: Channel) -> ModeState:
    ch = compose(*channels)
    if ch.T.shape[1] != state.n_modes:
        raise ValueError("channel does not match the number of modes in the state")
    return ModeState(alpha=ch.T @ state.alpha, corr=ch.T @ state.corr @ ch.T.conj().T + ch.D)


# ----------------------------------------------------------------- Laguerre-Gauss modes

def lg_modes(max_order) -> list[tuple[int, int]]:
    """LG_pl modes with mode order 2p + |l| ≤ max_order, ordered by mode order then (p, l).
    Group N holds N + 1 modes, (N + 1)(N + 2)/2 in total (one polarisation)."""
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order!r}")
    out = []
    for n in range(max_order + 1):
        for l in range(-n, n + 1, 2):
            out.append(((n - abs(l)) // 2, l))
    return out


def _genlaguerre(p, a, u):
    """Generalised Laguerre polynomial L_p^a(u) by the three-term recurrence."""
    l0 = np.ones_like(u)
    if p == 0:
        return l0
    l1 = 1 + a - u
    for k in range(1, p):
        l0, l1 = l1, ((2 * k + 1 + a - u) * l1 - (k + a) * l0) / (k + 1)
    return l1


def aperture_matrix(modes, radius) -> np.ndarray:
    """Amplitude transmission matrix of a centred hard circular aperture of radius `radius`
    (in units of the 1/e² intensity radius w of LG_00 in the aperture plane), restricted to
    the given LG basis: A_mn = <m|χ|n>. Diagonal in l; couples radial orders p, p' of equal l.
    Hermitian with eigenvalues in [0, 1]. As a field map it is exact for projections onto these
    LG modes; light diffracted outside the basis is missing from it (see aperture_channel)."""
    require_positive(radius=radius)
    ua = 2.0 * radius**2
    x, wq = np.polynomial.legendre.leggauss(200)
    u = 0.5 * ua * (x + 1)
    wq = 0.5 * ua * wq
    n = len(modes)
    A = np.zeros((n, n))
    for i, (p, l) in enumerate(modes):
        for j, (q, m) in enumerate(modes):
            if m != l or j < i:
                continue
            a = abs(l)
            norm = np.sqrt(factorial(p) * factorial(q) / (factorial(p + a) * factorial(q + a)))
            val = norm * np.sum(wq * u**a * np.exp(-u) * _genlaguerre(p, a, u) * _genlaguerre(q, a, u))
            A[i, j] = A[j, i] = val
    return A


def aperture_channel(modes, radius, output="polar") -> Channel:
    """Pinhole spatial filter: lens, aperture in the focal plane, recollimating lens.
    LG modes Fourier-transform into LG modes with phases (-i)^(2p+|l|) that the second lens
    undoes; `radius` is in units of the focal-plane LG_00 radius w_f = λ f / (π w).

    The true map χ P (P: projector on the input modes) has polar form V A^(1/2), V an isometry.
    output="polar": T = A^(1/2). Output in the basis V (not LG), exact for every quantity that
    does not depend on the output basis: total power, tr(C), tr(C²), the s-sp overlap, so a
    large-area detector after the pinhole is exact with no truncation error.
    output="lg": T = A. Output projected onto the same LG modes, exact when an LG-mode-selective
    element follows (e.g. coupling into a single-mode fibre); light leaving the basis is lost."""
    require_choice("output", output, ("polar", "lg"))
    A = aperture_matrix(modes, radius)
    if output == "lg":
        return loss_channel(A)
    w, V = np.linalg.eigh(A)
    return loss_channel((V * np.sqrt(np.clip(w, 0.0, 1.0))) @ V.T)


def mode_projector(n_modes, indices=(0,)) -> np.ndarray:
    """Detection operator W: orthogonal projector onto the listed modes (single-mode fibre
    pigtail = (0,)); the identity models a large-area detector that collects every mode."""
    W = np.zeros((int(n_modes), int(n_modes)), dtype=complex)
    for i in indices:
        W[i, i] = 1.0
    return W


# ------------------------------------------------------------------------- detection

def noise_figure_of(channel: Channel, signal_mode=None) -> float:
    """Optical noise figure of a channel for a coherent input in `signal_mode` (default mode 0):
    F = (1 + 2 ŝ† D ŝ) / G_s with ŝ the normalised output signal mode and G_s its power gain.
    Signal-shot plus signal-spontaneous definition (sp-sp and ASE shot excluded)."""
    n = channel.T.shape[1]
    u = np.zeros(n, dtype=complex)
    if signal_mode is None:
        u[0] = 1.0
    else:
        u = np.asarray(signal_mode, dtype=complex) / np.linalg.norm(signal_mode)
    v = channel.T @ u
    g = float(np.real(np.vdot(v, v)))
    if g <= 0:
        raise ValueError("signal is fully blocked by the channel")
    s = v / np.sqrt(g)
    return (1 + 2 * float(np.real(np.vdot(s, channel.D @ s)))) / g


def detect(state: ModeState, wavelength, optical_bandwidth, electrical_bandwidth,
           projector=None, n_pol=2, quantum_efficiency=1.0) -> Result:
    """Square-law detection behind an optical filter of bandwidth B_o and a mode projector W.
    Returns photocurrents, the noise variances and the electrical SNR."""
    require_positive(optical_bandwidth=optical_bandwidth, electrical_bandwidth=electrical_bandwidth)
    require_range("quantum_efficiency", quantum_efficiency, 0.0, 1.0)
    if n_pol not in (1, 2):
        raise ValueError(f"n_pol must be 1 or 2, got {n_pol!r}")
    if electrical_bandwidth > 2 * optical_bandwidth:
        raise ValueError("electrical_bandwidth must not exceed 2 × optical_bandwidth")
    hv = photon_energy(wavelength)
    R = quantum_efficiency * QE / hv
    Bo, Be = optical_bandwidth, electrical_bandwidth
    W = np.eye(state.n_modes, dtype=complex) if projector is None else np.asarray(projector, dtype=complex)
    a = W @ state.alpha
    Cw = W @ state.corr @ W.conj().T
    ps = hv * float(np.real(np.vdot(a, a)))
    tr1 = float(np.real(np.trace(Cw)))
    tr2 = float(np.real(np.trace(Cw @ Cw)))
    i_s = R * ps
    i_ase = R * hv * n_pol * tr1 * Bo
    shot = 2 * QE * (i_s + i_ase) * Be
    s_sp = 4 * R**2 * hv**2 * float(np.real(np.vdot(a, Cw @ a))) * Be
    sp_sp = R**2 * hv**2 * n_pol * tr2 * (2 * Bo - Be) * Be
    total = shot + s_sp + sp_sp
    m_eff = n_pol * tr1**2 / tr2 if tr2 > 0 else 0.0
    return Result(
        values={
            "signal_power": ps, "ase_power": hv * n_pol * tr1 * Bo,
            "signal_current": i_s, "ase_current": i_ase,
            "var_shot": shot, "var_s_sp": s_sp, "var_sp_sp": sp_sp, "var_total": total,
            "snr": i_s**2 / total if total > 0 else np.inf, "effective_modes": m_eff,
        },
        units={
            "signal_power": "W", "ase_power": "W", "signal_current": "A", "ase_current": "A",
            "var_shot": "A^2", "var_s_sp": "A^2", "var_sp_sp": "A^2", "var_total": "A^2",
            "snr": "", "effective_modes": "",
        },
        assumptions=[
            "ASE spectrally flat over the optical filter bandwidth B_o (rectangular filter)",
            "Signal in one polarisation; ASE in n_pol polarisations, unpolarised",
            "sp-sp term (2 B_o - B_e) B_e for a rectangular optical filter, B_e ≤ 2 B_o",
            "Thermal (circuit) noise not included",
        ],
    )


def osnr_of(state: ModeState, wavelength, reference_bandwidth, projector=None, n_pol=2) -> float:
    """Signal power over ASE power (both polarisations) in the reference bandwidth, after W."""
    require_positive(reference_bandwidth=reference_bandwidth)
    hv = photon_energy(wavelength)
    W = np.eye(state.n_modes, dtype=complex) if projector is None else np.asarray(projector, dtype=complex)
    a = W @ state.alpha
    ase = hv * n_pol * float(np.real(np.trace(W @ state.corr @ W.conj().T))) * reference_bandwidth
    return hv * float(np.real(np.vdot(a, a))) / ase if ase > 0 else np.inf


# ------------------------------------------------------------- spec (calculator) functions

def amplifier_noise(gain, n_sp) -> Result:
    """Single-mode amplifier: added ASE occupation and noise figure, with the quantum limit."""
    require_positive(gain=gain)
    ch = gain_channel(gain, n_sp, n_modes=1)
    nf = noise_figure_of(ch)
    return Result(
        values={"added_photons": float(np.real(ch.D[0, 0])), "noise_figure": nf,
                "noise_figure_analytic": 1 / gain + 2 * n_sp * (gain - 1) / gain,
                "noise_figure_limit": 2 - 1 / gain},
        units={"added_photons": "", "noise_figure": "", "noise_figure_analytic": "", "noise_figure_limit": ""},
        assumptions=["Phase-insensitive linear amplifier, n_out = G n_in + n_sp (G - 1)",
                     "Noise figure: signal-shot + signal-spontaneous definition"],
    )


def span_cascade(n_spans, span_transmission, amp_gain, n_sp, signal_power, wavelength, reference_bandwidth) -> Result:
    """N × (lossy span followed by an amplifier), single spatial mode."""
    n_spans = int(n_spans)
    if n_spans < 1:
        raise ValueError(f"n_spans must be >= 1, got {n_spans!r}")
    require_positive(signal_power=signal_power)
    hv = photon_energy(wavelength)
    unit = compose(loss_channel(span_transmission, n_modes=1), gain_channel(amp_gain, n_sp, n_modes=1))
    chain = compose(*[unit] * n_spans)
    out = propagate(coherent_state(1, signal_power, wavelength), chain)
    nf = noise_figure_of(chain)
    ase = 2 * hv * float(np.real(out.corr[0, 0])) * reference_bandwidth
    return Result(
        values={"signal_power_out": hv * abs(out.alpha[0]) ** 2, "ase_power": ase,
                "osnr": osnr_of(out, wavelength, reference_bandwidth),
                "noise_figure_total": nf, "net_gain": float(np.real(chain.T[0, 0] * np.conj(chain.T[0, 0])))},
        units={"signal_power_out": "W", "ase_power": "W", "osnr": "", "noise_figure_total": "", "net_gain": ""},
        assumptions=["Lumped stages, single spatial mode, ASE in both polarisations",
                     "OSNR: signal over ASE (both polarisations) in reference_bandwidth",
                     "Noise figure referred to the chain input (Friis cascade is exact here)"],
    )


def multimode_snr(signal_power, gain, n_sp, max_order, spatial_filter, pinhole_radius,
                  wavelength, optical_bandwidth, electrical_bandwidth, n_pol=2) -> Result:
    """Amplifier with equal gain in every LG mode up to `max_order`, signal in LG_00, detected
    on a large-area detector behind no filter, a pinhole spatial filter or a single-mode fibre."""
    require_choice("spatial_filter", spatial_filter, ("none", "pinhole", "single_mode"))
    modes = lg_modes(max_order)
    n = len(modes)
    s0 = coherent_state(n, signal_power, wavelength)
    stages = [gain_channel(gain, n_sp, n_modes=n)]
    W = None
    if spatial_filter == "pinhole":
        stages.append(aperture_channel(modes, pinhole_radius))
    elif spatial_filter == "single_mode":
        W = mode_projector(n, (0,))
    out = propagate(s0, *stages)
    r = detect(out, wavelength, optical_bandwidth, electrical_bandwidth, projector=W, n_pol=n_pol)
    vals = {"amplifier_modes": n * n_pol, **{k: r[k] for k in
            ("signal_power", "ase_power", "var_shot", "var_s_sp", "var_sp_sp", "snr", "effective_modes")}}
    units = {"amplifier_modes": "", **{k: r.units[k] for k in vals if k != "amplifier_modes"}}
    return Result(values=vals, units=units, assumptions=r.assumptions + [
        "Equal gain and n_sp in every guided LG mode (no gain guiding)",
        "Pinhole radius in units of the focal-plane LG_00 1/e² radius; light leaving the LG basis is lost",
    ])
