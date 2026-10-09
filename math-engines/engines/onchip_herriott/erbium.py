"""Er:Al2O3 spectroscopy and the steady-state two-level rate equation, as in er-waveguide-amplifier.html.

Absorption cross-section: a sum of Gaussian bands (peak 1533 nm) scaled to sigma_peak; emission by McCumber,
σe = σa exp[(hc/λ0 − hc/λ)/kT] with λ0 = 1532 nm, T = 295 K. At 980 nm the pump only absorbs (σa = sig980); at 1480
nm it also stimulates emission. Effective two-level system with energy-transfer upconversion C_up N2² and a quenched
fraction f_q of ions that stay in the ground state. In a layer with pump and signal intensities Ip, Is:

    Ra = σa,p Ip/hνp + σa,s Is/hνs,   Re = σe,p Ip/hνp + σe,s Is/hνs,   b = Ra + Re + 1/τ
    N2 = 2 Ra N / (b + √(b² + 4 C_up Ra N)),   N1 = N − N2 + Nq      (N: active ions, Nq: quenched)
    modal gain  g_s = Γ_s (σe,s N2 − σa,s N1),  pump absorption  a_p = Γ_p (σa,p N1 − σe,p N2)

Wavelengths in m outside (nm inside the band model). SI units (cross-sections m², N m⁻³, C_up m³/s).
"""
from __future__ import annotations

import math

H, C = 6.62607015e-34, 2.99792458e8
HCK, TK, LAM0 = 14387.77, 295.0, 1.532            # hc/k in µm·K, temperature, McCumber λ0 in µm
_BANDS = [(1533, 2.9, 11), (1512, 2.35, 58), (1553, 1.15, 30), (1588, 0.4, 50), (1468, 0.7, 40)]


def _raw(l_nm):
    s = 0.0
    for c, a, wd in _BANDS:
        q = (l_nm - c) / wd
        s += a * math.exp(-4 * math.log(2) * q * q)
    return s


_RAWMAX = max(_raw(1400 + 0.25 * i) for i in range(int(300 / 0.25) + 1))


def sigma_a(wavelength, sigma_peak=5.7e-25):
    return sigma_peak * _raw(wavelength * 1e9) / _RAWMAX


def sigma_e(wavelength, sigma_peak=5.7e-25):
    return sigma_a(wavelength, sigma_peak) * math.exp(HCK / TK * (1 / LAM0 - 1e-6 / wavelength))


def cross_sections(lam_s, lam_p, sigma_peak=5.7e-25, sig980=1.7e-25):
    """(σa,s, σe,s, σa,p, σe,p): the 980 nm band only absorbs; other pumps use the 1.5 µm band model."""
    sa_s, se_s = sigma_a(lam_s, sigma_peak), sigma_e(lam_s, sigma_peak)
    if abs(lam_p - 980e-9) < 30e-9:
        return sa_s, se_s, sig980, 0.0
    return sa_s, se_s, sigma_a(lam_p, sigma_peak), sigma_e(lam_p, sigma_peak)


def populations(Ip, Is, xs, lam_s, lam_p, N, Nq, tau, Cup):
    """(N1, N2) for local pump and signal intensities [W/m²]; xs = cross_sections(...)."""
    sa_s, se_s, sa_p, se_p = xs
    hvp, hvs = H * C / lam_p, H * C / lam_s
    Ra = sa_p * Ip / hvp + sa_s * Is / hvs
    Re = se_p * Ip / hvp + se_s * Is / hvs
    b = Ra + Re + 1 / tau
    n2 = 2 * Ra * N / (b + math.sqrt(b * b + 4 * Cup * Ra * N)) if Ra > 0 else 0.0
    return N - n2 + Nq, n2
