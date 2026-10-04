"""Highly multimode waveguide amplifiers: guided-mode sets, mode-dependent gain (MDG) from
overlap integrals, random linear mode coupling, and a saturated cladding-pumped two-level
amplifier with transverse spatial hole burning. Noise propagates with the ase_noise
covariance formalism (alpha' = T alpha, C' = T C T† + D, per polarisation and per Hz).

Mode sets (one polarisation; ×2 for both):
    step-index fibre   LP_lm, weakly guiding, U J_{l+1}(U)/J_l(U) = W K_{l+1}(W)/K_l(W)
    graded-index fibre LG modes of the infinite parabolic profile, groups g = 2p + |l| + 1 ≤ V/2
    planar / rectangular core  hard-wall modes sin(mπx/W) sin(nπy/H), k_x² + k_y² ≤ (k NA)²
A "family" (LP_lm or LG_p|l| or rect_mn) shares one transverse intensity profile; l ≠ 0
families hold two degenerate spatial modes (cos/sin or ±l). Intensities are averaged over
azimuth and normalised to ∫ I dA = 1 on the grid stored with the set.

Per-mode rates (1/m) for a population N2(r), N1(r):
    γ_e,k = σ_e ∫ N2 I_k dA,   γ_a,k = σ_a ∫ N1 I_k dA
A section of length dz with constant rates and loss α is the diagonal channel
    t_k = exp(g_k dz / 2),  d_k = γ_e,k (exp(g_k dz) - 1) / g_k,   g_k = γ_e,k - γ_a,k - α
which is the exact solution of dC/dz = g C + γ_e (spontaneous emission into mode k).
Off-diagonal gain overlaps between non-degenerate modes are dropped (they dephase over
the beat length); linear mode coupling is applied as random unitaries between sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ase_noise.engine import Channel, ModeState, _genlaguerre, photon_energy
from ..common import Result, require_choice, require_nonnegative, require_positive


@dataclass(frozen=True, eq=False)
class ModeSet:
    """Guided mode families of one waveguide at one wavelength (one polarisation)."""

    kind: str
    labels: list
    degeneracy: np.ndarray
    beta: np.ndarray
    group: np.ndarray
    weights: np.ndarray
    intensity: np.ndarray
    coords: dict = field(default_factory=dict)
    wavelength: float = 0.0

    @property
    def n_families(self) -> int:
        return len(self.labels)

    @property
    def n_modes(self) -> int:
        return int(self.degeneracy.sum())

    @property
    def family_of_mode(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_families), self.degeneracy)


def _k(wavelength):
    return 2 * np.pi / wavelength


def _ncore(numerical_aperture, n_clad):
    return np.sqrt(n_clad**2 + numerical_aperture**2)


def _radial_grid(radius_max, points):
    dr = radius_max / points
    r = (np.arange(points) + 0.5) * dr
    return r, 2 * np.pi * r * dr


def _normalise(I, w):
    return I / (I @ w)[:, None]


def _sort(kind, labels, deg, beta, group, w, I, coords, wavelength):
    o = np.argsort(-np.asarray(beta), kind="stable")
    return ModeSet(kind=kind, labels=[labels[i] for i in o], degeneracy=np.asarray(deg, dtype=int)[o],
                   beta=np.asarray(beta)[o], group=np.asarray(group, dtype=int)[o], weights=w,
                   intensity=_normalise(np.asarray(I)[o], w), coords=coords, wavelength=wavelength)


def _lp_roots(l, V):
    from scipy.optimize import brentq
    from scipy.special import jv, kve

    def f(U):
        W = np.sqrt(max(V * V - U * U, 1e-300))
        return U * jv(l + 1, U) * kve(l, W) - W * kve(l + 1, W) * jv(l, U)

    us = np.linspace(1e-9 * V, V * (1 - 1e-10), max(400, int(40 * V)))
    fs = np.array([f(u) for u in us])
    roots = []
    for i in np.nonzero(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)[0]:
        roots.append(brentq(f, us[i], us[i + 1], xtol=1e-13 * V, rtol=1e-14))
    return roots


def step_index_modes(core_radius, numerical_aperture, wavelength, n_clad=1.45, points=800, extent=2.0) -> ModeSet:
    """LP_lm modes of a weakly guiding step-index fibre, ordered by decreasing β."""
    from scipy.special import jv, kve

    require_positive(core_radius=core_radius, numerical_aperture=numerical_aperture, wavelength=wavelength, n_clad=n_clad)
    a, k = core_radius, _k(wavelength)
    V = k * a * numerical_aperture
    nco = _ncore(numerical_aperture, n_clad)
    r, w = _radial_grid(extent * a, int(points))
    x = r / a
    labels, deg, beta, I = [], [], [], []
    l = 0
    while True:
        roots = _lp_roots(l, V)
        if not roots:
            break
        for m, U in enumerate(roots, 1):
            W = np.sqrt(V * V - U * U)
            inside = jv(l, U * x) / jv(l, U)
            with np.errstate(over="ignore", invalid="ignore"):
                outside = kve(l, W * x) / kve(l, W) * np.exp(-W * (x - 1))
            F = np.where(x <= 1, inside, np.nan_to_num(outside))
            labels.append((l, m))
            deg.append(1 if l == 0 else 2)
            beta.append(np.sqrt((k * nco) ** 2 - (U / a) ** 2))
            I.append(F**2)
        l += 1
    return _sort("step_index", labels, deg, beta, np.arange(len(labels)), w, I, {"r": r, "dr": r[1] - r[0]}, wavelength)


def grin_modes(core_radius, numerical_aperture, wavelength, n_clad=1.45, points=800, extent=2.0) -> ModeSet:
    """LG modes of a parabolic-index core (infinite-parabola approximation), groups g ≤ V/2.
    w = sqrt(2a / (k NA)),  β_g² = (k n_co)² - 2 k NA g / a, families (l, m) = (|l|, p + 1)."""
    require_positive(core_radius=core_radius, numerical_aperture=numerical_aperture, wavelength=wavelength, n_clad=n_clad)
    a, k = core_radius, _k(wavelength)
    V = k * a * numerical_aperture
    nco = _ncore(numerical_aperture, n_clad)
    w0 = np.sqrt(2 * a / (k * numerical_aperture))
    r, wts = _radial_grid(extent * a, int(points))
    u = 2 * r**2 / w0**2
    labels, deg, beta, group, I = [], [], [], [], []
    for g in range(1, max(1, int(np.floor(V / 2))) + 1):
        for p in range((g - 1) // 2 + 1):
            l = g - 1 - 2 * p
            labels.append((l, p + 1))
            deg.append(1 if l == 0 else 2)
            beta.append(np.sqrt((k * nco) ** 2 - 2 * k * numerical_aperture * g / a))
            group.append(g)
            I.append(np.exp(l * np.log(np.maximum(u, 1e-300)) - u) * _genlaguerre(p, l, u) ** 2)
    return _sort("grin", labels, deg, beta, group, wts, I, {"r": r, "dr": r[1] - r[0]}, wavelength)


def planar_modes(width, height, numerical_aperture, wavelength, n_clad=1.45, samples_per_lobe=6) -> ModeSet:
    """Hard-wall modes of a rectangular core W × H (planar waveguide when H << W):
    I = (4/WH) sin²(mπx/W) sin²(nπy/H), guided when (mπ/W)² + (nπ/H)² ≤ (k NA)²."""
    require_positive(width=width, height=height, numerical_aperture=numerical_aperture, wavelength=wavelength)
    k = _k(wavelength)
    kmax = k * numerical_aperture
    nco = _ncore(numerical_aperture, n_clad)
    mmax = int(np.floor(kmax * width / np.pi))
    nmax = int(np.floor(kmax * height / np.pi))
    if mmax < 1 or nmax < 1:
        raise ValueError("core too small: no guided mode")
    nx, ny = max(32, samples_per_lobe * mmax), max(8, samples_per_lobe * nmax)
    xs = (np.arange(nx) + 0.5) * width / nx
    ys = (np.arange(ny) + 0.5) * height / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    w = np.full(X.size, width * height / X.size)
    labels, beta, I = [], [], []
    for m in range(1, mmax + 1):
        for n in range(1, nmax + 1):
            kt2 = (m * np.pi / width) ** 2 + (n * np.pi / height) ** 2
            if kt2 <= kmax**2:
                labels.append((m, n))
                beta.append(np.sqrt((k * nco) ** 2 - kt2))
                I.append((np.sin(m * np.pi * X / width) ** 2 * np.sin(n * np.pi * Y / height) ** 2).ravel())
    _, group = np.unique(np.round(np.asarray(beta) / k, 12), return_inverse=True)
    coords = {"x": (X - width / 2).ravel(), "y": (Y - height / 2).ravel(), "dx": width / nx, "dy": height / ny}
    return _sort("planar", labels, np.ones(len(labels), dtype=int), beta, group, w, I, coords, wavelength)


def _cover(c, d, half):
    """Fraction of the cell [c - d/2, c + d/2] inside [-half, half]."""
    return np.clip((np.minimum(c + d / 2, half) - np.maximum(c - d / 2, -half)) / d, 0.0, 1.0)


def disk_profile(modes: ModeSet, radius) -> np.ndarray:
    """Centred disk of `radius` (confined doping of a fibre core): area fraction of each
    grid cell inside the disk (exact cell coverage on radial grids, hard edge on planar ones)."""
    require_positive(radius=radius)
    if "r" in modes.coords:
        r, dr = modes.coords["r"], modes.coords["dr"]
        lo, hi = np.maximum(r - dr / 2, 0.0), r + dr / 2
        return np.clip((radius**2 - lo**2) / (hi**2 - lo**2), 0.0, 1.0)
    return (np.hypot(modes.coords["x"], modes.coords["y"]) <= radius).astype(float)


def box_profile(modes: ModeSet, width, height) -> np.ndarray:
    """Centred W × H rectangle (planar mode sets only), with exact cell coverage."""
    require_positive(width=width, height=height)
    if "x" not in modes.coords:
        raise ValueError("box_profile needs a planar mode set")
    c = modes.coords
    return _cover(c["x"], c["dx"], width / 2) * _cover(c["y"], c["dy"], height / 2)


def overlaps(modes: ModeSet, profile) -> np.ndarray:
    """Γ_k = ∫ profile I_k dA for every family."""
    return modes.intensity @ (np.asarray(profile, dtype=float) * modes.weights)


def small_signal_rates(modes: ModeSet, profile, gain_coefficient, n_sp, loss=0.0):
    """Per-family (γ_e, γ_a, α) for a uniform medium gain g0 = σ_e N2 - σ_a N1 inside
    `profile`, inversion factor n_sp = σ_e N2 / g0: γ_e = n_sp g0 Γ, γ_a = (n_sp - 1) g0 Γ."""
    require_nonnegative(gain_coefficient=gain_coefficient, loss=loss)
    if n_sp < 1:
        raise ValueError(f"n_sp must be >= 1, got {n_sp!r}")
    gam = overlaps(modes, profile)
    return n_sp * gain_coefficient * gam, (n_sp - 1) * gain_coefficient * gam, np.full(modes.n_families, float(loss))


def _section(ge, ga, al, dz):
    g = ge - ga - al
    x = g * dz
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(x) > 1e-12, ge * np.expm1(x) / np.where(g == 0, 1, g), ge * dz)
    return np.exp(x / 2), d


def coupling_unitary(modes: ModeSet, strength, beta_correlation=0.0, rng=None) -> np.ndarray:
    """Random unitary U = exp(i θ H) over the individual modes. H is a Hermitian Gaussian
    matrix with element weights exp(-(Δβ/Δβ_c)²) (Δβ_c = 0: only exactly degenerate modes
    couple), scaled so that Σ_n |H_mn|² averages to 1: θ = `strength` is the rms coupling angle."""
    require_nonnegative(strength=strength, beta_correlation=beta_correlation)
    rng = np.random.default_rng(rng)
    fam = modes.family_of_mode
    b, g = modes.beta[fam], modes.group[fam]
    n = b.size
    X = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    Hm = (X + X.conj().T) / np.sqrt(2)
    if beta_correlation > 0:
        wgt = np.exp(-(((b[:, None] - b[None, :]) / beta_correlation) ** 2))
    else:
        wgt = (g[:, None] == g[None, :]).astype(float)
    Hm *= wgt
    s = np.mean(np.sum(np.abs(Hm) ** 2, axis=1))
    if s == 0 or strength == 0:
        return np.eye(n, dtype=complex)
    lam, V = np.linalg.eigh(Hm / np.sqrt(s))
    return (V * np.exp(1j * strength * lam)) @ V.conj().T


def linear_channel(modes: ModeSet, gamma_e, gamma_a, loss, length, sections=1,
                   coupling_strength=0.0, beta_correlation=0.0, seed=0) -> Channel:
    """Small-signal amplifier of `length` with per-family rates, split into `sections`
    with a random coupling unitary after each section (none if coupling_strength = 0)."""
    require_positive(length=length)
    sections = int(sections)
    if sections < 1:
        raise ValueError(f"sections must be >= 1, got {sections!r}")
    fam = modes.family_of_mode
    t, d = _section(np.asarray(gamma_e)[fam], np.asarray(gamma_a)[fam], np.asarray(loss)[fam], length / sections)
    if coupling_strength == 0:
        tt = t**sections
        dd = d * np.sum(t[:, None] ** (2 * np.arange(sections))[None, :], axis=1)
        return Channel(T=np.diag(tt).astype(complex), D=np.diag(dd).astype(complex))
    rng = np.random.default_rng(seed)
    n = modes.n_modes
    T, D = np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)
    for _ in range(sections):
        U = coupling_unitary(modes, coupling_strength, beta_correlation, rng)
        T = U @ (t[:, None] * T)
        D = U @ (np.outer(t, t) * D + np.diag(d)) @ U.conj().T
    return Channel(T=T, D=D)


def mode_dependent_gain(channel: Channel) -> dict:
    """Eigenmode power gains (squared singular values of T, descending), MDG ratio
    max/min, and the standard deviation of the log gains (natural units)."""
    s2 = np.linalg.svd(channel.T, compute_uv=False) ** 2
    lg = np.log(s2)
    return {"gains": s2, "mdg": s2[0] / s2[-1], "log_gain_std": float(np.std(lg))}


def modal_noise(channel: Channel) -> dict:
    """For a coherent launch into each individual input mode j: power gain G_j = |T e_j|² and
    noise figure F_j = (1 + 2 (T† D T)_jj / G_j) / G_j (signal-shot + signal-spontaneous)."""
    G = np.sum(np.abs(channel.T) ** 2, axis=0)
    q = np.real(np.einsum("ij,ik,kj->j", channel.T.conj(), channel.D, channel.T))
    return {"gain": G, "noise_figure": (1 + 2 * q / G) / G, "ase_occupation": np.real(np.diag(channel.D))}


# ------------------------------------------------------------------- saturated amplifier

def saturated_amplifier(modes: ModeSet, length, pump_power, signal_power, profile, dopant_density,
                        sigma_es, sigma_as, sigma_ep, sigma_ap, lifetime, pump_wavelength, cladding_area,
                        signal_weights=None, ase_bandwidth=3e12, n_pol=2, signal_loss=0.0, pump_loss=0.0,
                        steps=200, coupling_strength=0.0, beta_correlation=0.0, seed=0) -> Result:
    """Co-pumped two-level amplifier (cladding pump, uniform pump intensity P_p / A_clad over
    the doped region) with forward signal and forward ASE resolved per guided mode.

    N2 = N_t (σ_ap φ_p + σ_as φ_s(r)) / ((σ_ap + σ_ep) φ_p + (σ_as + σ_es) φ_s(r) + 1/τ),
    φ_s(r) = Σ_k P_k I_k(r) / (h ν_s), P_k = signal + ASE (n_pol polarisations, one bin Δν).
    Each step is a Heun step with exact exponential sections; a coupling unitary follows each
    step when coupling_strength > 0. Returns z-resolved powers, the output state, and the
    channel linearised about the saturated operating point (for noise figures and MDG)."""
    require_positive(length=length, dopant_density=dopant_density, lifetime=lifetime,
                     pump_wavelength=pump_wavelength, cladding_area=cladding_area)
    require_nonnegative(pump_power=pump_power, signal_power=signal_power, sigma_es=sigma_es, sigma_as=sigma_as,
                        sigma_ep=sigma_ep, sigma_ap=sigma_ap, ase_bandwidth=ase_bandwidth,
                        signal_loss=signal_loss, pump_loss=pump_loss)
    if n_pol not in (1, 2):
        raise ValueError(f"n_pol must be 1 or 2, got {n_pol!r}")
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    hvs, hvp = photon_energy(modes.wavelength), photon_energy(pump_wavelength)
    fam = modes.family_of_mode
    n = modes.n_modes
    if signal_weights is None:
        signal_weights = np.zeros(n)
        signal_weights[0] = 1.0
    sw = np.asarray(signal_weights, dtype=complex)
    if sw.shape != (n,) or np.linalg.norm(sw) == 0:
        raise ValueError("signal_weights must be a non-zero amplitude vector over the individual modes")
    alpha = np.sqrt(signal_power / hvs) * sw / np.linalg.norm(sw)
    coupled = coupling_strength > 0
    C = np.zeros((n, n), dtype=complex) if coupled else np.zeros(n)
    T = np.eye(n, dtype=complex) if coupled else np.ones(n)
    D = np.zeros((n, n), dtype=complex) if coupled else np.zeros(n)
    Nd = dopant_density * np.asarray(profile, dtype=float)
    wts, I = modes.weights, modes.intensity
    rng = np.random.default_rng(seed)
    dz = length / steps

    def mode_powers(alpha, C):
        occ = np.real(np.diag(C)) if coupled else C
        return hvs * (np.abs(alpha) ** 2 + n_pol * ase_bandwidth * occ)

    def rates(alpha, C, pp):
        pk = np.bincount(fam, weights=mode_powers(alpha, C), minlength=modes.n_families)
        phis = (pk / hvs) @ I
        phip = pp / (hvp * cladding_area)
        n2 = Nd * (sigma_ap * phip + sigma_as * phis) / ((sigma_ap + sigma_ep) * phip + (sigma_as + sigma_es) * phis + 1 / lifetime)
        n1 = Nd - n2
        ge = sigma_es * (I @ (n2 * wts))
        ga = sigma_as * (I @ (n1 * wts))
        gp = -(sigma_ap * (n1 @ wts) - sigma_ep * (n2 @ wts)) / cladding_area - pump_loss
        return ge[fam], ga[fam], gp, n2 @ wts

    def advance(alpha, C, t, d):
        if coupled:
            return t * alpha, np.outer(t, t) * C + np.diag(d)
        return t * alpha, t**2 * C + d

    pp = float(pump_power)
    z = np.linspace(0, length, steps + 1)
    out = {"pump": [pp], "signal": [hvs * np.sum(np.abs(alpha) ** 2)],
           "ase": [float(np.sum(mode_powers(np.zeros(n), C)))], "n2_area": []}
    sl = np.full(n, signal_loss)
    for _ in range(steps):
        ge0, ga0, gp0, a0 = rates(alpha, C, pp)
        t0, d0 = _section(ge0, ga0, sl, dz)
        al1, C1 = advance(alpha, C, t0, d0)
        ge1, ga1, gp1, a1 = rates(al1, C1, pp * np.exp(gp0 * dz))
        t, d = _section(0.5 * (ge0 + ge1), 0.5 * (ga0 + ga1), sl, dz)
        alpha, C = advance(alpha, C, t, d)
        pp *= np.exp(0.5 * (gp0 + gp1) * dz)
        if coupled:
            T = t[:, None] * T
            D = np.outer(t, t) * D + np.diag(d)
            U = coupling_unitary(modes, coupling_strength, beta_correlation, rng)
            alpha, C, T, D = U @ alpha, U @ C @ U.conj().T, U @ T, U @ D @ U.conj().T
        else:
            T, D = t * T, t**2 * D + d
        out["pump"].append(pp)
        out["signal"].append(hvs * np.sum(np.abs(alpha) ** 2))
        out["ase"].append(float(np.sum(mode_powers(np.zeros(n), C))))
        out["n2_area"].append(0.5 * (a0 + a1))
    mp = mode_powers(alpha, C)
    if not coupled:
        C, T, D = np.diag(C).astype(complex), np.diag(T).astype(complex), np.diag(D).astype(complex)
    state = ModeState(alpha=alpha, corr=C)
    return Result(
        values={"z": z, "pump_power": np.array(out["pump"]), "signal_power": np.array(out["signal"]),
                "ase_power": np.array(out["ase"]), "n2_area": np.array(out["n2_area"]),
                "mode_power": mp, "state": state, "channel": Channel(T=T, D=D)},
        units={"z": "m", "pump_power": "W", "signal_power": "W", "ase_power": "W", "n2_area": "1/m",
               "mode_power": "W", "state": "", "channel": ""},
        assumptions=[
            "Two-level system, steady state, co-propagating pump, signal and ASE (no backward ASE)",
            "One spectral bin: all ASE at the signal wavelength within ase_bandwidth",
            "Pump uniform over the inner cladding; azimuthally averaged mode intensities",
            "Off-diagonal gain overlaps between non-degenerate modes neglected",
        ],
    )


# ------------------------------------------------------------- spec (calculator) functions

def _fibre_modes(waveguide, core_radius, numerical_aperture, wavelength):
    require_choice("waveguide", waveguide, ("step_index", "grin"))
    fn = step_index_modes if waveguide == "step_index" else grin_modes
    return fn(core_radius, numerical_aperture, wavelength)


def multimode_small_signal(waveguide, core_radius, numerical_aperture, wavelength, doping_ratio,
                           gain_coefficient, n_sp, length) -> Result:
    """Unsaturated, uncoupled multimode fibre amplifier with doping confined to r ≤ ρ a."""
    require_positive(doping_ratio=doping_ratio, length=length)
    modes = _fibre_modes(waveguide, core_radius, numerical_aperture, wavelength)
    ge, ga, al = small_signal_rates(modes, disk_profile(modes, doping_ratio * core_radius), gain_coefficient, n_sp)
    ch = linear_channel(modes, ge, ga, al, length)
    mn = modal_noise(ch)
    G = mn["gain"]
    occ = mn["ase_occupation"]
    return Result(
        values={"modes": modes.n_modes, "families": modes.n_families, "gain_fundamental": float(G[0]),
                "gain_mean": float(np.mean(G)), "mdg": float(G.max() / G.min()),
                "nf_fundamental": float(mn["noise_figure"][0]), "nf_mean": float(np.mean(mn["noise_figure"])),
                "ase_fraction_fundamental": float(occ[0] / occ.sum())},
        units={"modes": "", "families": "", "gain_fundamental": "", "gain_mean": "", "mdg": "",
               "nf_fundamental": "", "nf_mean": "", "ase_fraction_fundamental": ""},
        assumptions=["Uniform inversion inside the doped disk, no saturation, no mode coupling",
                     "modes counts one polarisation; MDG = max/min modal power gain"],
    )


def saturated_fibre_amplifier(waveguide, core_radius, numerical_aperture, doping_ratio, cladding_diameter,
                              length, pump_power, signal_power, wavelength=1.064e-6, pump_wavelength=0.976e-6,
                              dopant_density=6e25, sigma_es=2.5e-25, sigma_as=5e-27, sigma_ep=2.5e-24,
                              sigma_ap=2.6e-24, lifetime=1e-3, ase_bandwidth=3e12, steps=200) -> Result:
    """Cladding-pumped multimode fibre amplifier (Yb-like defaults), signal launched in the
    fundamental mode, no mode coupling. Spatial hole burning by the signal makes the gain
    mode dependent."""
    require_positive(doping_ratio=doping_ratio, cladding_diameter=cladding_diameter)
    modes = _fibre_modes(waveguide, core_radius, numerical_aperture, wavelength)
    r = saturated_amplifier(modes, length, pump_power, signal_power, disk_profile(modes, doping_ratio * core_radius),
                            dopant_density, sigma_es, sigma_as, sigma_ep, sigma_ap, lifetime, pump_wavelength,
                            np.pi * cladding_diameter**2 / 4, ase_bandwidth=ase_bandwidth, steps=steps)
    mn = modal_noise(r["channel"])
    G, mp = mn["gain"], r["mode_power"]
    ase_modes = mp - np.where(np.arange(mp.size) == 0, r["signal_power"][-1], 0.0)
    return Result(
        values={"modes": modes.n_modes, "signal_out": float(r["signal_power"][-1]), "ase_out": float(r["ase_power"][-1]),
                "pump_out": float(r["pump_power"][-1]), "gain_fundamental": float(G[0]),
                "mdg": float(G.max() / G.min()), "nf_fundamental": float(mn["noise_figure"][0]),
                "ase_fraction_fundamental": float(ase_modes[0] / ase_modes.sum()) if ase_modes.sum() > 0 else 0.0},
        units={"modes": "", "signal_out": "W", "ase_out": "W", "pump_out": "W", "gain_fundamental": "",
               "mdg": "", "nf_fundamental": "", "ase_fraction_fundamental": ""},
        assumptions=r.assumptions + ["Signal in LP01 / LG00; gains and noise figures linearised about the saturated state"],
    )
