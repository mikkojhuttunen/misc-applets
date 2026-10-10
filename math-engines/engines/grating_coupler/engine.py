"""Surface-etched slab gratings: local effective index, Fourier coupling, out-of-plane radiation and far field.

Geometry (x up): [handle | BOX] | substrate x < 0 | core 0..t | cladding x > t. The grating is etched from the top of
the core to depth h; within one period u in [0, 1) the local core thickness is t - h + h g(u), g = 1 on a tooth.
Every slice is replaced by the slab mode of its local thickness (effective-index method). SI units, vacuum
wavelength, angles in rad. Indices may be numbers or callables n(λ).

Python reference for profileFn, tableAt, modeParams, planeWave, radiation and farField in
dbr-structures/dbr-engine.js; the discretisation (samples per period, n_eff table points) is a parameter so the
port can be checked to rounding error.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf

import numpy as np

from ..common import Result, require_choice, require_positive, require_range
from ..slab_waveguide.engine import neff_three_layer

PROFILES = ("rect", "trap", "smooth", "sine", "tri", "saw")
_TAU = 2 * np.pi


def _n(v, lam):
    return v(lam) if callable(v) else v


def profile_function(kind, fill, period, etch_depth=0.0, sidewall_angle=np.pi / 2, edge_sigma=0.0):
    """Tooth profile g(u), u in [0, 1) within a period, g = 1 unetched. Returns (g, is_rect).

    rect: tooth of width f centred at u = 1/2; trap: sidewall angle sidewall_angle (rad, π/2 vertical); smooth: rectangular tooth blurred by
    a Gaussian of rms edge_sigma (RIE-like rounding); sine, tri: duty-warped so g > 1/2 over a fraction f; saw: rises
    over a fraction f (blazed)."""
    require_choice("profile", kind, PROFILES)
    f = float(fill)

    def rect(u):
        return 1.0 if abs(u - 0.5) < f / 2 else 0.0

    def warp(av):
        return av / (2 * f) if av <= f / 2 else 0.25 + (av - f / 2) / ((1 - f) / 2) * 0.25

    if kind == "trap":
        a = etch_depth / np.tan(sidewall_angle) / period
        if a < 1e-6:
            return rect, True
        return (lambda u: min(1.0, max(0.0, (f / 2 + a / 2 - abs(u - 0.5)) / a))), False
    if kind == "smooth":
        s = edge_sigma / period
        if s < 1e-6:
            return rect, True
        c = 1 / (np.sqrt(2) * s)

        def g(u):
            v = u - 0.5
            acc = sum(0.5 * (erf((v + k + f / 2) * c) - erf((v + k - f / 2) * c)) for k in (-1, 0, 1))
            return min(1.0, max(0.0, acc))
        return g, False
    if kind == "sine":
        return (lambda u: 0.5 * (1 + np.cos(_TAU * warp(abs(u - 0.5))))), False
    if kind == "tri":
        return (lambda u: 1 - 2 * warp(abs(u - 0.5))), False
    if kind == "saw":
        return (lambda u: u / f if u < f else (1 - u) / (1 - f)), False
    return rect, True


def fourier_coefficient(samples, q):
    """Complex q-th Fourier coefficient (1/n) Σ a_i exp(-2πi q (i + 1/2)/n) of one period sampled at midpoints."""
    a = np.asarray(samples, dtype=float)
    n = a.size
    return complex(np.sum(a * np.exp(-1j * _TAU * q * (np.arange(n) + 0.5) / n)) / n)


def slab_mode(wavelength, n_sub, n_core, n_clad, thickness, polarization="TE", order=0):
    """Transverse field of a three-layer slab mode (E_y for TE, H_y for TM), unnormalised, with its ∫field² dx.

    Substrate x < 0, core 0..d, cladding x > d. Returns None below cut-off, else a dict with neff, kx, gs, gc,
    field and dfield (callables, vectorised; dfield = p dφ/dx, p = 1/ε for TM), norm = ∫ field² dx and
    norm_eps = ∫ field²/ε dx (the TM power integral)."""
    neff = neff_three_layer(wavelength, n_sub, n_core, n_clad, thickness, polarization, order)
    if not np.isfinite(neff):
        return None
    k = _TAU / wavelength
    rs = n_core**2 / n_sub**2 if polarization == "TM" else 1.0
    kx = k * np.sqrt(n_core**2 - neff**2)
    gs = max(k * np.sqrt(neff**2 - n_sub**2), 1e-9 / 1e-6)
    gc = max(k * np.sqrt(neff**2 - n_clad**2), 1e-9 / 1e-6)
    ps = np.arctan2(rs * gs, kx)
    cs, cc = np.cos(ps), np.cos(kx * thickness - ps)
    d = thickness

    def field(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, cs * np.exp(gs * np.minimum(x, 0)),
                        np.where(x <= d, np.cos(kx * x - ps), cc * np.exp(-gc * np.maximum(x - d, 0))))

    tm = polarization == "TM"
    es, ef, ec = (n_sub**2, n_core**2, n_clad**2) if tm else (1.0, 1.0, 1.0)

    def dfield(x):
        """p dφ/dx with p = 1 (TE) or 1/ε (TM): continuous across the interfaces (∝ E_z for TM)."""
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, cs * gs * np.exp(gs * np.minimum(x, 0)) / es,
                        np.where(x <= d, -kx * np.sin(kx * x - ps) / ef, -gc * cc * np.exp(-gc * np.maximum(x - d, 0)) / ec))

    core = d / 2 + (np.sin(2 * (kx * d - ps)) + np.sin(2 * ps)) / (4 * kx)
    norm = cs**2 / (2 * gs) + cc**2 / (2 * gc) + core
    norm_eps = cs**2 / (2 * gs * es) + cc**2 / (2 * gc * ec) + core / ef        # ∫ φ²/ε dx (TM power), = norm for TE
    return {"neff": neff, "kx": kx, "gs": gs, "gc": gc, "field": field, "dfield": dfield, "norm": norm,
            "norm_eps": norm_eps, "thickness": d}


@dataclass
class SurfaceGrating:
    """Surface-etched grating on a three-layer slab, optionally on a BOX (= the substrate layer) over a handle.

    n_handle: complex index of the handle under a BOX of thickness box_thickness (Si: ~3.48, Au: 0.52 + 10.7j at
    1.55 µm), or None for a semi-infinite substrate. The downward radiation channel is open unless the handle
    absorbs (Im n ≥ 0.05). samples: profile points per period; table_points: n_eff table over g in [0, 1]
    (non-rectangular profiles; a rectangular one needs only g = 0 and 1). tm_screening: strength c of the wall
    screening in the TM tooth model (tm_weights), calibrated against the rigorous solver (rcwa)."""

    n_sub: float
    n_core: float
    n_clad: float
    thickness: float
    etch_depth: float
    period: float
    fill: float = 0.5
    profile: str = "rect"
    sidewall_angle: float = np.pi / 2
    edge_sigma: float = 0.0
    polarization: str = "TE"
    n_handle: complex | None = None
    box_thickness: float = 0.0
    samples: int = 1024
    table_points: int = 24
    tm_screening: float = 0.35

    def __post_init__(self):
        require_choice("polarization", self.polarization, ("TE", "TM"))
        require_positive(thickness=self.thickness, period=self.period)
        require_range("etch_depth", self.etch_depth, 0.0, self.thickness)
        require_range("fill", self.fill, 0.0, 1.0)
        self.g_fn, self.rect = profile_function(self.profile, self.fill, self.period, self.etch_depth,
                                                self.sidewall_angle, self.edge_sigma)
        n = self.samples
        if self.rect:          # cell averages: exact fill, Fourier coefficients to O((q/n)²)
            lo, hi, e = 0.5 - self.fill / 2, 0.5 + self.fill / 2, np.arange(n + 1) / n
            self.g = np.maximum(0.0, np.minimum(hi, e[1:]) - np.maximum(lo, e[:-1])) * n
        else:
            self.g = np.array([self.g_fn((i + 0.5) / n) for i in range(n)])

    def indices(self, wavelength):
        return _n(self.n_sub, wavelength), _n(self.n_core, wavelength), _n(self.n_clad, wavelength)

    # ---------------------------------------------------------------- local effective index
    def neff_table(self, wavelength, order=0):
        """n_eff of the local slab vs fill g (thickness t - h + h g); cut-off slices take max(n_sub, n_clad)."""
        ns, nf, nc = self.indices(wavelength)
        K = 1 if self.rect else self.table_points
        floor = max(ns, nc)
        vals, cut = np.empty(K + 1), False
        for i in range(K + 1):
            d = self.thickness - self.etch_depth + self.etch_depth * i / K
            v = neff_three_layer(wavelength, ns, nf, nc, d, self.polarization, order) if d > 1e-12 else np.nan
            if not np.isfinite(v):
                v, cut = floor, True
            vals[i] = v
        hi_cut = not np.isfinite(neff_three_layer(wavelength, ns, nf, nc, self.thickness, self.polarization, order))
        return {"K": K, "vals": vals, "cut": cut, "hi_cut": hi_cut, "n_sub": ns, "n_core": nf, "n_clad": nc}

    @staticmethod
    def _interp(tab, g):
        K = tab["K"]
        x = np.asarray(g, dtype=float) * K
        i = np.clip(np.floor(x), 0, K - 1).astype(int)
        r = x - i
        return tab["vals"][i] * (1 - r) + tab["vals"][i + 1] * r

    def neff_profile(self, wavelength, order=0):
        return self._interp(self.neff_table(wavelength, order), self.g)

    def coupling(self, wavelength, harmonic, order=0):
        """Coupled-mode coefficient of Bragg order m = harmonic: κ_m = 2π |N_m| / λ, N_m the m-th Fourier coefficient
        of n_eff(z) (rectangular profile: 2 Δn sin(π m f) / (m λ)), times the TM factor |ρ| of tm_weights (1 for TE)."""
        kap = _TAU * abs(fourier_coefficient(self.neff_profile(wavelength, order), harmonic)) / wavelength
        return kap * abs(self.tm_weights(wavelength, order)["rho"])

    def tm_weights(self, wavelength, order=0):
        """TM tooth model. A thin tooth layer perturbs the normal field through D_x (continuous across its top face,
        weight ε2/ε1 on the cladding-side E_x) and the longitudinal field E_z directly; walls screen E_z and unscreen
        E_x as the tooth gets taller. Local-field weights w_v = 1/(1 + N_v Δε/ε2), w_l = 1/(1 + N_l Δε/ε2) with
        N_v = w/(w + c h), N_l = 1 - N_v, w = f Λ the tooth width, c = tm_screening (h → 0: exact thin-layer limit).
        In the forward-backward (Bragg) coupling the two terms enter with opposite signs, in the n_eff shift with the
        same sign, so the effective-index κ is scaled by ρ = (C_v - C_l) / (C_v⁰ + C_l⁰) (C⁰: thin-layer weights); the
        coupling itself is -ρ times the effective-index one (sign checked by rigorous band-edge ordering, see crigf)
        and |κ| = |ρ| κ_EIM.
        Fields of the mean-thickness slab mode at the sheet x_g = t - h/2. TE: ρ = 1, weights 1."""
        if self.polarization == "TE" or self.etch_depth == 0:
            return {"rho": 1.0, "w_v": 1.0, "w_l": 1.0}
        ns, nf, nc = self.indices(wavelength)
        e1, e2 = nf * nf, nc * nc
        de = e1 - e2
        h, t = self.etch_depth, self.thickness
        w = max(self.fill, 1e-9) * self.period
        Nv = w / (w + self.tm_screening * h)
        wv, wl = 1 / (1 + Nv * de / e2), 1 / (1 + (1 - Nv) * de / e2)
        m = slab_mode(wavelength, ns, nf, nc, t - h + h * float(np.mean(self.g)), "TM", order)
        if m is None:
            return {"rho": 1.0, "w_v": wv, "w_l": wl}
        xg = t - h / 2
        beta = _TAU / wavelength * m["neff"]
        cv = (beta * float(m["field"](xg))) ** 2 / e2**2            # |D_x|² / ε2² (cladding-side E_x², per ε0² ω²)
        cl = float(m["dfield"](xg)) ** 2                            # |E_z|² (same units)
        rho = (cv * wv - cl * wl) / (cv * e2 / e1 + cl)
        return {"rho": rho, "w_v": wv, "w_l": wl, "c_v": cv, "c_l": cl}

    # ---------------------------------------------------------------- vertical stack, plane waves
    def stack(self, wavelength):
        """Layers bottom to top as (complex n, thickness or None, is_core), and whether the bottom channel is open."""
        ns, nf, nc = self.indices(wavelength)
        if self.n_handle is None:
            return [(complex(ns), None, False), (complex(nf), self.thickness, True), (complex(nc), None, False)], True
        nh = complex(_n(self.n_handle, wavelength))
        return ([(nh, None, False), (complex(ns), self.box_thickness, False), (complex(nf), self.thickness, True),
                 (complex(nc), None, False)], nh.imag < 0.05)

    def bottom_index(self, wavelength):
        layers, _ = self.stack(wavelength)
        return layers[0][0].real

    def channels(self, wavelength):
        """Radiation channels (medium, real index): the cladding, and the bottom medium unless it absorbs."""
        layers, open_ = self.stack(wavelength)
        ch = [("cladding", layers[-1][0].real)]
        if open_:
            ch.append(("substrate", layers[0][0].real))
        return ch

    def plane_wave(self, wavelength, kz, xg, side="top", pol="TE"):
        """Plane wave of in-plane wavevector kz incident from the cladding (side='top') or the bottom medium; TE (E_y)
        or TM (H_y, with (1/ε) ∂H_y/∂x continuous).

        Returns the field F (E_y or H_y) and G = p ∂F/∂x (p = 1 or 1/ε) at height xg inside the core (from the core
        bottom) per unit incident amplitude, the amplitude reflection r and the power-normalised transmission tau, or
        None from an absorbing bottom."""
        layers, open_ = self.stack(wavelength)
        if side == "bottom":
            if not open_:
                return None
            layers = layers[::-1]
        k = _TAU / wavelength

        def kx_of(n):
            v = np.sqrt(complex(k * k * n * n - kz * kz))
            return -v if v.imag < 0 else v

        p = (lambda n: 1.0) if pol == "TE" else (lambda n: 1 / (n * n))
        kb, kt = kx_of(layers[0][0]), kx_of(layers[-1][0])
        pb, pt = p(layers[0][0]), p(layers[-1][0])
        E, D, F, Gx = 1.0 + 0j, -1j * kb * pb, None, None
        for n, d, core in layers[1:-1]:
            kx, pn = kx_of(n), p(n)

            def at(x, E=E, D=D, kx=kx, pn=pn):
                c, s = np.cos(kx * x), np.sin(kx * x)
                return E * c + D / (kx * pn) * s, D * c - E * kx * pn * s

            if core:
                F, Gx = at(self.thickness - xg if side == "bottom" else xg)
                if side == "bottom":
                    Gx = -Gx                                    # derivative along +x (up) in the stack frame
            E, D = at(d)
        tau = 2j * kt * pt / (1j * kt * pt * E - D)
        r = tau * E - 1
        flux = np.sqrt(max(np.real(kb * pb), 0.0) / np.real(kt * pt))
        return {"F": tau * F, "G": tau * Gx, "r": r, "tau": tau * flux, "kt": kt, "kb": kb}

    # ---------------------------------------------------------------- radiation and far field
    def radiation(self, wavelength, order=0, periods=1):
        """Out-of-plane power loss of the guided mode per diffraction order and channel (thin-sheet volume current).

        The etched layer is a current sheet Δε = n_core² - n_clad² of thickness h at depth t - h/2, driven by the mode
        of the mean local thickness; α_q = k⁴ Δε² h² |G_q|² φ²(x_g) / (4 β k_x) |F|², with G_q the Fourier coefficient of
        g and F the local field of a plane wave arriving from that medium (reciprocity: slab and BOX interference).
        TM: the sheet couples the cladding-side normal field through D_x and the longitudinal field E_z with the
        local-field weights of tm_weights, using TM plane waves (formula in the code).
        Also the guided Bragg order nearest 2 N0 Λ / λ: κ, half-detuning and tanh-type reflectance of `periods`."""
        lam = wavelength
        tab = self.neff_table(lam, order)
        ns, nf, nc = tab["n_sub"], tab["n_core"], tab["n_clad"]
        k = _TAU / lam
        nP = self._interp(tab, self.g)
        N0 = float(np.mean(nP))
        out = {"wavelength": lam, "order": order, "N0": N0, "orders": [], "alpha_total": 0.0, "valid": not tab["hi_cut"],
               "table": tab}
        if not out["valid"]:
            return out
        beta = k * N0
        out["beta"] = beta
        h, t = self.etch_depth, self.thickness
        mp = slab_mode(lam, ns, nf, nc, t - h + h * float(np.mean(self.g)), self.polarization, order)
        xg = t - h / 2
        phi2 = float(mp["field"](xg)) ** 2 / mp["norm"] if mp else 0.0
        pref = k**4 * (nf * nf - nc * nc) ** 2 * h**2 * phi2 / (4 * beta)
        tm = self.polarization == "TM"
        if tm:
            tw = self.tm_weights(lam, order)
            e2 = nc * nc
        K = _TAU / self.period
        chans = self.channels(lam)
        nmax = max(ns, nc, *(nj for _, nj in chans))
        for q in range(1, int(np.ceil((beta + k * nmax) / K)) + 1):
            kz = beta - q * K
            G = abs(fourier_coefficient(self.g, q))
            for med, nj in chans:
                if abs(kz) >= k * nj:
                    continue
                kx = np.sqrt(k * k * nj * nj - kz * kz)
                side = "top" if med == "cladding" else "bottom"
                if not tm:
                    lf = self.plane_wave(lam, kz, xg, side)["F"]
                    a = pref * G * G / kx * abs(lf) ** 2
                elif mp:
                    # TM: α = h² |G|² Δε² ε_j |X|² / (4 β k_x ∫φ²/ε), X = -(φ'/ε)(H'/ε) w_l + β k' φ H w_v / ε2², k' = -k_z
                    pw = self.plane_wave(lam, kz, xg, side, "TM")
                    X = (-float(mp["dfield"](xg)) * pw["G"] * tw["w_l"]
                         - beta * kz * float(mp["field"](xg)) * pw["F"] * tw["w_v"] / e2**2)
                    a = h * h * G * G * (nf * nf - nc * nc) ** 2 * nj * nj * abs(X) ** 2 / (4 * beta * kx * mp["norm_eps"])
                else:
                    a = 0.0
                out["orders"].append({"q": q, "medium": med, "n": nj, "kz": kz, "theta": float(np.arcsin(kz / (k * nj))),
                                      "alpha": a, "G": G})
                out["alpha_total"] += a
        qB = 2 * N0 * self.period / lam
        qn = max(1, int(np.floor(qB + 0.5)))
        kap = _TAU * abs(fourier_coefficient(nP, qn)) / lam * (abs(tw["rho"]) if tm else 1.0)
        dh = beta - qn * np.pi / self.period
        L = periods * self.period
        out["bragg"] = {"qB": qB, "q": qn, "kappa": kap, "half_detuning": dh, "R": _cmt_R(kap, dh, L),
                        "lambda_B": 2 * N0 * self.period / qn}
        out["mode"] = mp
        out["n_high"], out["n_low"] = tab["vals"][-1], tab["vals"][0]
        return out

    def far_field(self, rad, theta, n_medium, periods, alpha_power=None):
        """Relative angular intensity emitted into a medium of index n_medium (theta from the normal, rad): one
        period's emission |Σ S_i exp(iDΛ(i + 1/2)/128)|², S the 128 block averages of g - ḡ, times the array factor of `periods` periods with amplitude decay
        exp(-α Λ / 2) per period; α defaults to the total radiation loss of `rad`."""
        alpha = rad["alpha_total"] if alpha_power is None else alpha_power
        k, beta, Lam = _TAU / rad["wavelength"], rad["beta"], self.period
        M = 128
        step = self.g.size / M
        edges = np.floor(np.arange(M + 1) * step).astype(int)
        S = np.array([self.g[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        S = S - S.mean()
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        D = beta - k * n_medium * np.sin(th)
        c = (S[None, :] * np.exp(1j * D[:, None] * Lam * (np.arange(M)[None, :] + 0.5) / M)).sum(axis=1)
        a = np.exp(-alpha * Lam / 2)
        aN = a**periods
        ph = D * Lam
        num = np.abs(1 - aN * np.exp(1j * periods * ph)) ** 2
        den = np.abs(1 - a * np.exp(1j * ph)) ** 2
        af = np.where(den < 1e-20, float(periods) ** 2, num / np.where(den < 1e-20, 1.0, den))
        I = np.abs(c) ** 2 * af
        return I if np.ndim(theta) else float(I[0])


def _cmt_R(kap, dh, L):
    """Coupled-mode peak-region reflectance for half-detuning dh (same branches as bragg_grating.cmt_reflectance)."""
    k2, d2 = kap * kap, dh * dh
    if abs(k2 - d2) < 1e-2:                     # 1e-14 µm⁻² in the JS port
        return k2 * L * L / (1 + k2 * L * L)
    if k2 > d2:
        s = np.sqrt(k2 - d2)
        sh, c = np.sinh(s * L), np.cosh(s * L)
        return float(k2 * sh * sh / (d2 * sh * sh + (k2 - d2) * c * c))
    s = np.sqrt(d2 - k2)
    sn, c = np.sin(s * L), np.cos(s * L)
    return float(k2 * sn * sn / (d2 - k2 * c * c))


# ---------------------------------------------------------------- Result front ends (spec.yaml)
def profile_coupling(wavelength, n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor=0.5, profile="rect",
                     sidewall_angle=np.pi / 2, edge_sigma=0.0, polarization="TE", harmonic=1) -> Result:
    """κ of Bragg order m for an etched profile from the Fourier coefficient of the local n_eff, with the
    rectangular-grating estimate 2 Δn sin(π m f)/(m λ) for comparison."""
    require_positive(wavelength=wavelength)
    m = int(harmonic)
    if m < 1:
        raise ValueError("harmonic must be >= 1")
    gr = SurfaceGrating(n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor, profile, sidewall_angle,
                        edge_sigma, polarization)
    tab = gr.neff_table(wavelength)
    dn = tab["vals"][-1] - tab["vals"][0]
    return Result(
        values={"kappa": gr.coupling(wavelength, m), "kappa_rect": 2 * dn * abs(np.sin(np.pi * m * fill_factor)) / (m * wavelength),
                "n_high": tab["vals"][-1], "n_low": tab["vals"][0], "n_mean": float(np.mean(gr.neff_profile(wavelength))),
                "cut_off": bool(tab["cut"])},
        units={"kappa": "1/m", "kappa_rect": "1/m", "n_high": "", "n_low": "", "n_mean": "", "cut_off": ""},
        assumptions=["Effective-index method, fundamental slab mode of each local thickness", "Weak-grating coupled-mode theory",
                     "Etched slices below cut-off take max(n_sub, n_clad)"],
    )


def grating_radiation(wavelength, n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor=0.5, profile="rect",
                      sidewall_angle=np.pi / 2, edge_sigma=0.0, polarization="TE", periods=100, handle="none",
                      box_thickness=2e-6) -> Result:
    """Out-of-plane radiation loss of a surface grating per channel, directionality, emission angle of the strongest
    order, guided Bragg coupling. handle: 'none' (semi-infinite substrate), 'si' (n = 3.476) or 'au' (0.52 + 10.7i)
    under a BOX of index n_sub."""
    require_choice("handle", handle, ("none", "si", "au"))
    nh = {"none": None, "si": 3.476 + 0j, "au": 0.52 + 10.7j}[handle]
    gr = SurfaceGrating(n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor, profile, sidewall_angle,
                        edge_sigma, polarization, nh, box_thickness)
    r = gr.radiation(wavelength, 0, int(periods))
    if not r["valid"]:
        raise ValueError("the unetched slab is below cut-off")
    up = sum(o["alpha"] for o in r["orders"] if o["medium"] == "cladding")
    down = r["alpha_total"] - up
    best = max(r["orders"], key=lambda o: o["alpha"], default=None)
    tot = r["alpha_total"]
    return Result(
        values={"alpha_total": tot, "alpha_up": up, "alpha_down": down, "up_fraction": up / tot if tot > 0 else np.nan,
                "radiation_length": 1 / tot if tot > 0 else np.inf, "theta_main": best["theta"] if best else np.nan,
                "n_mean": r["N0"], "bragg_order": r["bragg"]["q"], "kappa_bragg": r["bragg"]["kappa"],
                "lambda_bragg": r["bragg"]["lambda_B"], "R_bragg": r["bragg"]["R"]},
        units={"alpha_total": "1/m", "alpha_up": "1/m", "alpha_down": "1/m", "up_fraction": "", "radiation_length": "m",
               "theta_main": "rad", "n_mean": "", "bragg_order": "", "kappa_bragg": "1/m", "lambda_bragg": "m", "R_bragg": ""},
        assumptions=["Effective-index method; first-order (Born) radiation from a thin current sheet at the etch mid-depth",
                     "Local field by reciprocity from a TE plane wave through the unetched stack (BOX interference included)",
                     "TM uses the TE expressions with the H_y profile: rough", "Power loss of the guided mode, not of a coupled cavity",
                     "Expect agreement with rigorous solvers within a factor of a few for shallow gratings"],
    )
