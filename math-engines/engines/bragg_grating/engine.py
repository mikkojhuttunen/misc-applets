"""Bragg gratings: coupled-mode estimates and the exact transfer matrix of a periodic stack.

SI units; wavelength is the vacuum wavelength; indices are effective indices of
the guided mode (or bulk indices for a thin-film stack). Lossless unless an
imaginary index is passed to stack_reflectance.

Version 2 adds oblique incidence (s/p, TIR) and the etched-trench membrane DBR used by chip-scale multipass cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Result, require_choice, require_positive, require_range


def coupling_coefficient(delta_n, wavelength, fill_factor=0.5, order=1) -> Result:
    """κ of a rectangular index grating: κ = 2 Δn sin(π m f) / (m λ)."""
    require_positive(delta_n=delta_n, wavelength=wavelength)
    require_range("fill_factor", fill_factor, 0.0, 1.0)
    m = int(order)
    if m < 1:
        raise ValueError("order must be >= 1")
    kappa = 2 * delta_n * abs(np.sin(np.pi * m * fill_factor)) / (m * wavelength)
    return Result(
        values={"kappa": kappa},
        units={"kappa": "1/m"},
        assumptions=["Rectangular profile", "Δn = n_high - n_low of the effective index", "Coupled-mode theory, weak grating"],
    )


def grating_summary(kappa, length, wavelength, n_group) -> Result:
    """Peak reflectance tanh²(κL), null-to-null bandwidth and effective (penetration) length."""
    require_positive(kappa=kappa, length=length, wavelength=wavelength, n_group=n_group)
    kl = kappa * length
    return Result(
        values={
            "R_peak": np.tanh(kl) ** 2,
            "bandwidth_null": wavelength**2 / (np.pi * n_group * length) * np.sqrt(kl**2 + np.pi**2),
            "L_eff": np.tanh(kl) / (2 * kappa),
            "kappa_L": kl,
        },
        units={"R_peak": "", "bandwidth_null": "m", "L_eff": "m", "kappa_L": ""},
        assumptions=["Uniform grating", "Lossless", "Coupled-mode theory"],
    )


def cmt_reflectance(wavelength, bragg_wavelength, n_eff, kappa, length):
    """Coupled-mode reflectance spectrum (helper, vectorised over wavelength).
    Half-detuning δ = 2π n_eff (1/λ - 1/λ_B); material dispersion ignored."""
    require_positive(bragg_wavelength=bragg_wavelength, n_eff=n_eff, length=length)
    lam = np.asarray(wavelength, dtype=float)
    d = 2 * np.pi * n_eff * (1 / lam - 1 / bragg_wavelength)
    s2 = kappa**2 - d**2
    s = np.sqrt(np.abs(s2)) + 1e-300
    L = length
    hyp = kappa**2 * np.sinh(s * L) ** 2 / (d**2 * np.sinh(s * L) ** 2 + s2 * np.cosh(s * L) ** 2)
    osc = kappa**2 * np.sin(s * L) ** 2 / (d**2 - kappa**2 * np.cos(s * L) ** 2)
    return np.where(s2 > 0, hyp, osc)


def stack_reflectance(wavelength, n_high, n_low, d_high, d_low, periods, n_in=None, n_out=None) -> Result:
    """Exact (Abelès) reflectance and transmittance of (H L)^N at normal incidence,
    H facing the input medium. Indices may be complex (loss: positive imaginary part)."""
    require_positive(d_high=d_high, d_low=d_low)
    N = int(periods)
    if N < 1:
        raise ValueError("periods must be >= 1")
    n_in = n_low if n_in is None else n_in
    n_out = n_low if n_out is None else n_out
    lam = np.atleast_1d(np.asarray(wavelength, dtype=float))
    require_positive(wavelength=lam)
    k = 2 * np.pi / lam

    def layer(n, d):
        dl = k * n * d
        c, s = np.cos(dl), np.sin(dl)
        return np.array([[c, -1j * s / n], [-1j * n * s, c]], dtype=complex)  # shape (2, 2, nλ)

    H, Lm = layer(n_high, d_high), layer(n_low, d_low)
    P = np.einsum("ijw,jkw->ikw", H, Lm)
    M = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], P.shape).copy()
    base, e = P, N
    while e:
        if e & 1:
            M = np.einsum("ijw,jkw->ikw", M, base)
        base = np.einsum("ijw,jkw->ikw", base, base)
        e >>= 1
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    x, y = n_in * a + n_in * n_out * b, c + n_out * d
    r = (x - y) / (x + y)
    t = 2 * n_in / (x + y)
    R, T = np.abs(r) ** 2, np.real(n_out) / np.real(n_in) * np.abs(t) ** 2
    if np.ndim(wavelength) == 0:
        R, T = R[0], T[0]
    return Result(
        values={"R": R, "T": T},
        units={"R": "", "T": ""},
        assumptions=["Normal incidence", "Plane waves (or one guided mode with effective indices)", "Abelès characteristic matrices"],
    )


# ---------------------------------------------------------------- oblique incidence (v2)
LATERAL_POL = {"TE": "p", "TM": "s"}
"""Slab-mode label -> polarisation on a vertical (etched-trench) mirror whose plane of incidence is the membrane
plane: slab TE (E in the plane) is p-polarised (Brewster hole), slab TM (E_z dominant) is s-polarised."""


def _norm(M):
    s = np.max(np.abs(M), axis=(0, 1), keepdims=True)
    return M / np.where(s == 0, 1, s)


def stack_R_oblique(wavelength, sin_in, n_in, layers, n_out, polarization="s"):
    """Power reflectance of a layer stack at oblique incidence (helper, vectorised over wavelength and sin_in).

    layers: [(n, d), ...] starting at the input medium; complex n allowed (loss: positive imaginary part).
    sin_in: sine of the angle in the input medium. Tilted admittances η_s = n cos θ, η_p = n / cos θ; evanescent
    layers take the branch Im(cos θ) >= 0, so total internal reflection and frustrated TIR are included.
    The overall scale of the matrix product is renormalised every layer (only r = (x - y)/(x + y) is needed),
    so thick evanescent stacks do not overflow. Reduces to stack_reflectance at normal incidence.
    """
    require_choice("polarization", polarization, ("s", "p"))
    lam = np.asarray(wavelength, dtype=float)
    s = n_in * np.minimum(np.abs(np.asarray(sin_in, dtype=float)), 1 - 1e-9)   # exact grazing excluded
    shape = np.broadcast(lam, s).shape
    k = 2 * np.pi / lam

    def cosine(n):
        return np.sqrt(1 - (s / n) ** 2 + 1e-18j)

    def adm(n):
        c = cosine(n)
        return n * c if polarization == "s" else n / c

    M = np.zeros((2, 2) + shape, dtype=complex)
    M[0, 0] = M[1, 1] = 1
    for n, d in layers:
        q = adm(n)
        dl = k * n * d * cosine(n)
        cs, sn = np.cos(dl), np.sin(dl)
        L = np.zeros_like(M)
        L[0, 0], L[0, 1], L[1, 0], L[1, 1] = cs, -1j * sn / q, -1j * q * sn, cs
        M = _norm(np.einsum("ij...,jk...->ik...", M, L))
    q_in, q_out = adm(n_in), adm(n_out)
    x, y = q_in * (M[0, 0] + M[0, 1] * q_out), M[1, 0] + M[1, 1] * q_out
    return np.abs((x - y) / (x + y)) ** 2


def cascaded_R(wavelength, sin_in, n_in, sections, n_out=1.0, polarization="s"):
    """Several stacks in series (each a list of (n, d)), e.g. a doubly resonant mirror of two DBR sections."""
    return stack_R_oblique(wavelength, sin_in, n_in, [l for sec in sections for l in sec], n_out, polarization)


@dataclass
class TrenchDBR:
    """In-plane DBR of a membrane cell: cavity (membrane, n_tooth) | [gap, tooth] x N | gap medium.

    Gap and tooth are odd multiples m_gap, m_tooth of a quarter wave at lam_design, measured along the layer normal
    at the design angle of incidence (sin_design, in the membrane): d = m λ / (4 n cos θ_layer), Snell's law from the
    membrane. sin_design = 0 is the normal-incidence design; it must stay below n_gap / n_tooth (beyond that the
    first gap totally reflects and there is no quarter-wave condition). n_tooth is the effective index of the slab
    mode; slab_pol the slab-mode label (TE -> p, TM -> s on the trench walls). bounce_loss lumps what a 1D
    effective-index model misses (slot radiation, TE<->TM conversion, roughness)."""

    n_tooth: float
    lam_design: float = 1.55e-6
    N: int = 8
    m_gap: int = 1
    m_tooth: int = 3
    n_gap: float = 1.0
    bounce_loss: float = 3e-4
    slab_pol: str = "TM"
    sin_design: float = 0.0

    def __post_init__(self):
        if not 0 <= self.sin_design < self.n_gap / self.n_tooth:
            raise ValueError(f"sin_design must lie in [0, n_gap/n_tooth = {self.n_gap / self.n_tooth:.4f})")

    @property
    def d_gap(self):
        cos_g = np.sqrt(1 - (self.n_tooth * self.sin_design / self.n_gap) ** 2)
        return self.m_gap * self.lam_design / (4 * self.n_gap * cos_g)

    @property
    def d_tooth(self):
        return self.m_tooth * self.lam_design / (4 * self.n_tooth * np.sqrt(1 - self.sin_design**2))

    def layers(self, n_tooth=None):
        n_t = self.n_tooth if n_tooth is None else n_tooth
        return [(self.n_gap, self.d_gap), (n_t, self.d_tooth)] * self.N

    def R(self, wavelength, sin_chi, n_tooth_at_lam=None):
        """Reflectance vs angle of incidence sin χ (in the membrane) and wavelength; n_tooth_at_lam adds the
        dispersion of the slab mode (thicknesses stay those of the design)."""
        n_t = self.n_tooth if n_tooth_at_lam is None else n_tooth_at_lam
        return stack_R_oblique(wavelength, sin_chi, n_t, self.layers(n_t), self.n_gap, LATERAL_POL[self.slab_pol]) * (1 - self.bounce_loss)

    def angle_average(self, wavelength=None, n=513):
        """⟨R⟩ over sin χ uniform in [0, 1]: the mirror a fully chaotic cell sees (invariant billiard measure)."""
        sg = np.linspace(0, 1, n)
        return float(_trapezoid(self.R(self.lam_design if wavelength is None else wavelength, sg), sg))


def _trapezoid(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def double_resonant_orders(lam1, lam2, n_t1, n_t2, max_order=31, n_gap=1.0, top=5):
    """Odd orders for ONE periodic DBR on a Bragg condition at two wavelengths:
        tooth a1 λ1 / (4 n_t1) = a2 λ2 / (4 n_t2),   gap b1 λ1 / (4 n_gap) = b2 λ2 / (4 n_gap).
    Returns the best `top` candidates with relative thickness mismatches (small is good), ranked by the worse
    mismatch with a weak preference for low orders (wider stop bands)."""
    odds = np.arange(1, max_order + 1, 2)
    o1, o2 = (g.ravel() for g in np.meshgrid(odds, odds, indexing="ij"))     # all (order at λ1, order at λ2) pairs
    e_tooth = np.abs(o1 * lam1 / n_t1 - o2 * lam2 / n_t2) / (o1 * lam1 / n_t1)
    e_gap = np.abs(o1 * lam1 - o2 * lam2) / (o1 * lam1)
    score = np.maximum(e_tooth[:, None], e_gap[None, :]) + 1e-4 * (o1[:, None] + o1[None, :])
    out = []
    for flat in np.argsort(score, axis=None, kind="stable")[:top]:
        i, j = np.unravel_index(flat, score.shape)
        out.append(dict(tooth_orders=(int(o1[i]), int(o2[i])), gap_orders=(int(o1[j]), int(o2[j])),
                        tooth_mismatch=float(e_tooth[i]), gap_mismatch=float(e_gap[j]),
                        d_tooth=float(o1[i] * lam1 / (4 * n_t1)), d_gap=float(o1[j] * lam1 / (4 * n_gap))))
    return out


def oblique_reflectance(wavelength, sin_incidence, n_in, n_a, d_a, n_b, d_b, periods, n_out, polarization="s") -> Result:
    """Reflectance of (A B)^N at oblique incidence, layer A facing the input medium, s or p polarisation.
    For an etched membrane DBR: n_in = n_b = slab n_eff, n_a = n_out = 1 (air gaps)."""
    require_positive(wavelength=wavelength, n_in=n_in, n_a=n_a, d_a=d_a, n_b=n_b, d_b=d_b, n_out=n_out)
    require_range("sin_incidence", sin_incidence, 0.0, 1.0)
    N = int(periods)
    if N < 1:
        raise ValueError("periods must be >= 1")
    R = stack_R_oblique(wavelength, sin_incidence, n_in, [(n_a, d_a), (n_b, d_b)] * N, n_out, polarization)
    s_tir = n_out / n_in if n_out < n_in else np.nan
    s_b = n_out / np.sqrt(n_in**2 + n_out**2)
    return Result(
        values={"R": float(R) if np.ndim(R) == 0 else R, "sin_tir": s_tir, "sin_brewster": s_b},
        units={"R": "", "sin_tir": "", "sin_brewster": ""},
        assumptions=["Plane waves (or one guided mode with effective indices), 1D stack", "Tilted-admittance characteristic matrices",
                     "TIR and frustrated TIR included", "sin_brewster: single input/output interface, p polarisation"],
    )
