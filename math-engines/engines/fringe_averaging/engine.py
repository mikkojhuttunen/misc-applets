"""How much of an interference fringe survives angle dithering, laser-frequency dithering, slow drifts and a finite
measurement (averaging) time.

One parasitic interference term between the main light and a stray path with optical path difference ΔL:

    fringe ∝ 2 sqrt(ε) Re{ g · exp(i φ(t)) },   ε = stray / main power ratio
    φ(t) = x_θ w_θ(f_θ t) + x_ν w_ν(f_ν t + ψ) + ω_d t + const

    x_θ = (dφ/dθ) A_θ                 angle-dither depth (dφ/dθ from ray_phase: k0 n dΔL/dθ)
    x_ν = 2π n_g ΔL Δν / c            spectral-dither depth (laser frequency deviation ±Δν)
    ω_d = k0 ΔL (dn/dT) (dT/dt) + 2π n_g ΔL (dν/dt) / c        slow drift (temperature, laser frequency)
    g   = exp(-π Δν_L n_g ΔL / c)      Lorentzian laser linewidth Δν_L (|g(τ)| of the field)
    w   = sine or triangle wave of unit amplitude

Harmonic expansion makes long averages exact without sampling: e^{i x w(f t)} = Σ_m c_m e^{i 2π m f t} with
c_m = J_m(x) for a sine (Miller backward recurrence) and a closed form for a triangle. The product of the two
dithers is a sum of components at Ω = 2π (m f_θ + n f_ν) + ω_d; equal frequencies are summed (dither frequencies are
integers in Hz, so coincidences such as f_ν = 2 f_θ are found exactly). An averaging filter passes each component
with gain H(Ω): boxcar of length T -> |sinc(Ω T / 2)|, first-order RC with time constant T -> 1/sqrt(1 + Ω² T²).
The residual fringe visibility is the rms over the start time of the measurement:

    V_rms(T) = |g| sqrt( Σ_groups |C_g|² H(Ω_g, T)² )

The Ω = 0 group (only without drift) is the static floor that averaging can never remove. It is |g c_0(x_θ) c_0(x_ν)|
(= |g J0 J0| for sines) PLUS every intermodulation product with m f_θ + n f_ν = 0: with f_θ/f_ν = p/q in lowest terms
these are the orders (m, n) = (q k, -p k). Rates in a low-order ratio (2:1, 3:2, 13:10 at deep dithers) can raise the
floor by orders of magnitude; detuning one rate by a few Hz pushes the coinciding orders beyond the harmonic content.
Everything else falls as ~1/(Ω T).
SI units; frequencies in Hz, ω in rad/s.
"""
from __future__ import annotations

import numpy as np

from ..common import Result, require_choice, require_nonnegative, require_positive, require_range

C0 = 299792458.0
_WAVEFORMS = ("sine", "triangle")
_FILTERS = ("boxcar", "rc")
MAX_COMPONENTS = 6_000_000


def bessel_j_all(x, m_max):
    """J_0..J_{m_max}(x) by Miller's backward recurrence, normalised with J0 + 2 Σ J_2k = 1 (array, len m_max + 1)."""
    x = float(x)
    m_max = int(m_max)
    out = np.zeros(m_max + 1)
    if x == 0.0:
        out[0] = 1.0
        return out
    ax = abs(x)
    start = 2 * int(np.ceil((max(m_max, ax) + 20 + 4 * ax ** (1 / 3) + 10 * np.sqrt(ax / 10 + 1)) / 2)) + 20
    jp, j = 0.0, 1e-300
    norm = 0.0
    for k in range(start, 0, -1):
        jm = 2 * k / ax * j - jp                          # J_{k-1}
        jp, j = j, jm
        if abs(j) > 1e250:                                # rescale everything kept so far
            j *= 1e-250
            jp *= 1e-250
            out *= 1e-250
            norm *= 1e-250
        if k - 1 <= m_max:
            out[k - 1] = j
        if (k - 1) % 2 == 0 and k - 1 > 0:
            norm += 2 * j
    norm += out[0]
    out /= norm
    if x < 0:
        out[1::2] *= -1
    return out


def dither_harmonics(x, waveform="sine", tol=1e-14):
    """Harmonic orders m and coefficients c_m of exp(i x w(t)), w(t) = sin(2π t) or a unit triangle wave
    (w(0) = 0, rising, w(1/4) = 1). Coefficients below tol are dropped."""
    require_choice("waveform", waveform, _WAVEFORMS)
    x = float(x)
    if x == 0.0:
        return np.array([0]), np.array([1.0 + 0j])
    ax = abs(x)
    if waveform == "sine":
        M = int(np.ceil(ax + 12 * ax ** (1 / 3) + 40))
        J = bessel_j_all(x, M)
        m = np.arange(-M, M + 1)
        c = np.where(m >= 0, J[np.abs(m)], J[np.abs(m)] * (-1.0) ** np.abs(m)).astype(complex)
    else:
        # rising half t in [-1/4, 1/4]: w = 4t; falling half t in [1/4, 3/4]: w = 2 - 4t; integrate exactly
        # corners in w(t) make |c_m| ~ 8x / (2π m)² for large m: pick M so the neglected power, about
        # (2/3) (8x/4π²)² / M³, stays near 1e-9 (truncated_power() reports it)
        M = max(int(np.ceil(6 * ax / np.pi)) + 200, int(np.ceil((2 / 3 * (8 * ax / (4 * np.pi**2)) ** 2 / 5e-10) ** (1 / 3))))
        m = np.arange(-M, M + 1)
        b1 = 4 * x - 2 * np.pi * m
        b2 = -(4 * x + 2 * np.pi * m)
        seg = lambda b, lo, hi: np.where(np.abs(b) > 1e-12, (np.exp(1j * b * hi) - np.exp(1j * b * lo)) / (1j * np.where(np.abs(b) > 1e-12, b, 1.0)), hi - lo)
        c = seg(b1, -0.25, 0.25) + np.exp(2j * x) * seg(b2, 0.25, 0.75)
    keep = np.abs(c) > tol
    return m[keep], c[keep]


def truncated_power(m, c):
    """Power left out of a harmonic set, 1 - Σ |c_m|² (Parseval)."""
    return float(max(0.0, 1.0 - np.sum(np.abs(c) ** 2)))


def fringe_components(x_angle, f_angle, wave_angle, x_freq, f_freq, wave_freq, phase_freq=0.0):
    """Frequency components of exp(i φ(t)) for the two dithers: (frequency keys in Hz, complex amplitudes), equal
    frequencies summed. Dither frequencies are rounded to integer Hz so coincidences are exact."""
    fa, ff = int(round(f_angle)), int(round(f_freq))
    m1, c1 = dither_harmonics(x_angle, wave_angle)
    m2, c2 = dither_harmonics(x_freq, wave_freq)
    c2 = c2 * np.exp(1j * m2 * phase_freq)
    if m1.size * m2.size > MAX_COMPONENTS:
        raise ValueError(f"too many components ({m1.size * m2.size}); reduce the dither depths")
    keys = (m1[:, None].astype(np.int64) * fa + m2[None, :].astype(np.int64) * ff).ravel()
    amps = (c1[:, None] * c2[None, :]).ravel()
    uk, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    C = np.bincount(inv, weights=amps.real, minlength=uk.size) + 1j * np.bincount(inv, weights=amps.imag, minlength=uk.size)
    return uk.astype(float), C


def filter_gain(omega, T, kind="boxcar", envelope=False):
    """|H(Ω)| of a boxcar average of length T or a first-order RC low-pass with time constant T. envelope=True replaces
    the boxcar's sinc by its lobe-averaged envelope 1/sqrt(1 + 2 z²), z = ΩT/2 (same limits; no exact nulls at
    T = k · 2π/Ω, which real frequency jitter would fill in)."""
    require_choice("filter", kind, _FILTERS)
    om = np.asarray(omega, float)
    if kind == "boxcar" and envelope:
        return 1.0 / np.sqrt(1.0 + 2.0 * (0.5 * om * T) ** 2)
    if kind == "boxcar":
        z = 0.5 * om * T
        return np.where(np.abs(z) < 1e-12, 1.0, np.abs(np.sin(z) / np.where(np.abs(z) < 1e-12, 1.0, z)))
    return 1.0 / np.sqrt(1.0 + (om * T) ** 2)


def residual_visibility(freqs_hz, amps, T, drift_rad_s=0.0, coherence=1.0, kind="boxcar", envelope=False):
    """V_rms(T) = |g| sqrt(Σ |C|² H(2π f + ω_d, T)²) for scalar or array T."""
    P = np.abs(amps) ** 2
    om = 2 * np.pi * np.asarray(freqs_hz, float) + drift_rad_s
    Ts = np.atleast_1d(np.asarray(T, float))
    out = np.array([abs(coherence) * np.sqrt((P * filter_gain(om, t, kind, envelope) ** 2).sum()) for t in Ts])
    return out if np.ndim(T) else float(out[0])


def static_floor(freqs_hz, amps, drift_rad_s=0.0, coherence=1.0):
    """The part that no averaging removes: |g C(Ω = 0)| if there is no drift, else 0."""
    if drift_rad_s != 0.0:
        return 0.0
    k = np.nonzero(np.asarray(freqs_hz) == 0)[0]
    return float(abs(coherence) * abs(amps[k[0]])) if k.size else 0.0


def coincidence_orders(f_angle, f_freq):
    """Lowest harmonic orders (m_θ, m_ν) with m_θ f_θ = m_ν f_ν: intermodulation products of these orders (and their
    multiples) sit at 0 Hz and add to the static floor."""
    fa, ff = int(round(f_angle)), int(round(f_freq))
    g = np.gcd(fa, ff)
    return ff // g, fa // g


def floor_split(x_angle, f_angle, wave_angle, x_freq, f_freq, wave_freq, phase_freq=0.0, coherence=1.0):
    """(|g c_0 c_0|, full static floor including intermodulation products at 0 Hz)."""
    m1, c1 = dither_harmonics(x_angle, wave_angle)
    m2, c2 = dither_harmonics(x_freq, wave_freq)
    f, A = fringe_components(x_angle, f_angle, wave_angle, x_freq, f_freq, wave_freq, phase_freq)
    return float(abs(coherence) * abs(c1[m1 == 0][0] * c2[m2 == 0][0])), static_floor(f, A, 0.0, coherence)


def common_period(f_angle, f_freq):
    """1 / gcd of the two (integer-Hz) dither frequencies: averaging over a whole number of these periods nulls every
    dither harmonic exactly (boxcar), leaving the static floor and the drift."""
    return 1.0 / np.gcd(int(round(f_angle)), int(round(f_freq)))


def depths(wavelength, n_phase, n_group, opd, dphi_dtheta, angle_amp, freq_dev, linewidth, dn_dT, temp_rate, freq_drift):
    """x_θ, x_ν, ω_d and g from physical parameters (module docstring)."""
    k0 = 2 * np.pi / wavelength
    tau = n_group * opd / C0
    return dict(x_angle=dphi_dtheta * angle_amp, x_freq=2 * np.pi * tau * freq_dev,
                drift=k0 * opd * dn_dT * temp_rate + 2 * np.pi * tau * freq_drift,
                coherence=float(np.exp(-np.pi * linewidth * tau)), tau=tau)


def regular_cell_estimate(radius, n_facets, theta_c, delta_pass, wavelength, n_phase=1.0):
    """ΔL and dφ/dθ of a stray path that differs by delta_pass chords in a regular N-gon star orbit launched at θc
    from a facet centre: chord 2 r cos(π/N) cos θc, dchord/dθ = -2 r cos(π/N) sin θc (ray_phase gives the exact
    pass-by-pass values for perturbed cells)."""
    ap = radius * np.cos(np.pi / n_facets)
    opd = abs(delta_pass) * 2 * ap * np.cos(theta_c)
    dL = abs(delta_pass) * 2 * ap * abs(np.sin(theta_c))
    return opd, 2 * np.pi * n_phase / wavelength * dL


def time_trace(t, x_angle, f_angle, wave_angle, x_freq, f_freq, wave_freq, phase_freq=0.0, drift_rad_s=0.0):
    """φ(t) for plotting."""
    t = np.asarray(t, float)

    def w(u, kind):
        return np.sin(2 * np.pi * u) if kind == "sine" else 1 - 4 * np.abs(np.mod(u + 0.25, 1.0) - 0.5)

    return x_angle * w(f_angle * t, wave_angle) + x_freq * w(f_freq * t + phase_freq / (2 * np.pi), wave_freq) + drift_rad_s * t


def fringe_residual(wavelength=1.55e-6, n_phase=1.0, n_group=1.0, opd=0.1, dphi_dtheta=2.45e5, angle_amp=1e-5,
                    angle_freq=1000.0, angle_wave="sine", freq_dev=0.0, freq_freq=1300.0, freq_wave="triangle",
                    freq_phase=0.0, linewidth=1e6, dn_dT=1.5e-4, temp_rate=0.0, freq_drift=0.0, avg_time=1.0,
                    filter_kind="boxcar", stray_db=-40.0) -> Result:
    """Residual fringe visibility and amplitude after angle and spectral dithering, drift and averaging over avg_time."""
    require_positive(wavelength=wavelength, n_phase=n_phase, n_group=n_group, avg_time=avg_time)
    require_nonnegative(opd=opd, angle_amp=angle_amp, freq_dev=freq_dev, linewidth=linewidth)
    require_positive(angle_freq=angle_freq, freq_freq=freq_freq)
    require_choice("angle_wave", angle_wave, _WAVEFORMS)
    require_choice("freq_wave", freq_wave, _WAVEFORMS)
    require_choice("filter_kind", filter_kind, _FILTERS)
    require_range("stray_db", stray_db, -200.0, 0.0)
    d = depths(wavelength, n_phase, n_group, opd, dphi_dtheta, angle_amp, freq_dev, linewidth, dn_dT, temp_rate, freq_drift)
    f, A = fringe_components(d["x_angle"], angle_freq, angle_wave, d["x_freq"], freq_freq, freq_wave, freq_phase)
    V = residual_visibility(f, A, avg_time, d["drift"], d["coherence"], filter_kind)
    floor = static_floor(f, A, d["drift"], d["coherence"])
    amp = 2 * np.sqrt(10 ** (stray_db / 10))
    m0 = dither_harmonics(d["x_angle"], angle_wave), dither_harmonics(d["x_freq"], freq_wave)
    c00 = abs(d["coherence"]) * abs(m0[0][1][m0[0][0] == 0][0] * m0[1][1][m0[1][0] == 0][0])
    om, on = coincidence_orders(angle_freq, freq_freq)
    return Result(
        values={"x_angle": d["x_angle"], "x_freq": d["x_freq"], "drift_hz": d["drift"] / (2 * np.pi), "coherence": d["coherence"],
                "V_static": floor, "V_c0c0": c00, "coincidence_m_angle": int(om), "coincidence_m_freq": int(on), "V_rms": V, "fringe_amplitude": amp * V, "fringe_undithered": amp * d["coherence"],
                "n_components": int(f.size)},
        units={"x_angle": "rad", "x_freq": "rad", "drift_hz": "Hz", "coherence": "", "V_static": "", "V_c0c0": "",
               "coincidence_m_angle": "", "coincidence_m_freq": "", "V_rms": "",
               "fringe_amplitude": "", "fringe_undithered": "", "n_components": ""},
        assumptions=["One stray path with optical path difference ΔL against the main light", "Periodic dithers, frequencies rounded to integer Hz",
                     "V_rms: rms over the start time of the averaging window", "Lorentzian linewidth: |g| = exp(-π Δν_L n_g ΔL / c)",
                     "Linear drifts over the averaging time", "fringe amplitude = 2 sqrt(ε) V in fractional intensity (≈ absorbance)"],
    )
