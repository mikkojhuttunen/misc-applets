"""Rigorous coupled-wave analysis (Fourier modal method) of 1D-periodic multilayer gratings, TE and TM.

Geometry: x along the grating (period Λ), z up through the stack; invariant along y. Layers bottom to top between a
semi-infinite substrate and cover. A layer is uniform (index n) or lamellar: segments [(width fraction, n), ...]
filling one period from x = 0. Floquet orders k_x,m = k_x0 + m K, m = -M..M. TE: E_y; TM: H_y with Li's inverse
rule (correct factorisation for lamellar gratings). Complex indices allowed (loss: Im n > 0). Units: SI in and out;
internally lengths are normalised by k0.

Fields in a layer: u = W [e^{iλ(z - z_b)} a + e^{-iλ(z - z_t)} b] (u = E_y or H_y), v = P ∂z u / k0 with P = 1 (TE)
or [[1/ε]] (TM); up-modes referenced at the layer bottom, down-modes at the top, so every exponential is bounded.
The stack is solved by admittance recursion v = Y u (stable for thick and evanescent layers).

- diffraction: plane wave from the cover, reflected and transmitted efficiencies per order.
- leaky_mode: complex k_x0 of a guided (Bloch) mode, from det(I - R_up R_down) = 0 at a plane in the guide
  (transverse resonance against a fixed reference admittance, Muller iteration); outgoing orders take the analytic continuation of the real-axis branch, so radiating (leaky)
  modes are found with Im k_x0 > 0. Power radiation loss α = 2 Im k_x0; up/down split from the outgoing amplitudes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import Result, require_choice, require_positive

_TAU = 2 * np.pi


@dataclass
class Stack:
    """Substrate | layers (bottom to top) | cover. layers: [(thickness, n or [(fraction, n), ...]), ...]."""

    period: float
    n_sub: complex
    n_cover: complex
    layers: list = field(default_factory=list)
    polarization: str = "TE"

    def __post_init__(self):
        require_positive(period=self.period)
        require_choice("polarization", self.polarization, ("TE", "TM"))
        for d, seg in self.layers:
            if d < 0:
                raise ValueError("layer thickness must be >= 0")
            if not np.isscalar(seg):
                fr = sum(w for w, _ in seg)
                if abs(fr - 1) > 1e-9:
                    raise ValueError(f"segment fractions must sum to 1, got {fr}")


# ---------------------------------------------------------------- Fourier matrices and modes
def _fourier(seg, power, M):
    """Toeplitz matrix [[n^power]]_{m-n} of a lamellar profile, orders -M..M (exact for piecewise-constant ε)."""
    q = np.arange(-2 * M, 2 * M + 1)
    c = np.zeros(q.size, dtype=complex)
    x0 = 0.0
    for w, n in seg:
        v = complex(n) ** power
        x1 = x0 + w
        nz = q != 0
        c[nz] += v * (np.exp(-1j * _TAU * q[nz] * x1) - np.exp(-1j * _TAU * q[nz] * x0)) / (-1j * _TAU * q[nz])
        c[~nz] += v * w
        x0 = x1
    i = np.arange(-M, M + 1)
    return c[(i[:, None] - i[None, :]) + 2 * M]


def _sqrt_branch(z, leaky=False):
    """λ = sqrt(z): Im λ ≥ 0 (decaying or outgoing). leaky: analytic continuation from the real axis for orders with
    Re z > 0 (Re λ > 0 kept even if Im λ < 0), used for complex k_x in every layer so the mode function stays
    continuous across Im k_x = 0 (in the semi-infinite media it is also the leaky-wave radiation condition)."""
    lam = np.sqrt(np.asarray(z, dtype=complex))
    if leaky:
        flip = (np.real(z) <= 0) & (np.imag(lam) < 0)
    else:
        flip = (np.imag(lam) < 0) | ((np.imag(lam) == 0) & (np.real(lam) < 0))
    return np.where(flip, -lam, lam)


def _modes(seg, kx, pol, M, leaky=False):
    """(W, λ, V, P-diagonal-or-matrix) of one layer; kx normalised by k0."""
    n_ord = 2 * M + 1
    if np.isscalar(seg):
        eps = complex(seg) ** 2
        lam = _sqrt_branch(eps - kx**2, leaky)
        p = 1.0 if pol == "TE" else 1 / eps
        W = np.eye(n_ord, dtype=complex)
        return W, lam, 1j * p * np.diag(lam), p
    E = _fourier(seg, 2, M)
    Kx = np.diag(kx)
    if pol == "TE":
        Om2 = Kx @ Kx - E
        P = np.eye(n_ord)
    else:
        A = _fourier(seg, -2, M)
        Om2 = np.linalg.solve(A, Kx @ np.linalg.solve(E, Kx) - np.eye(n_ord))
        P = A
    w, W = np.linalg.eig(Om2)
    lam = _sqrt_branch(-w, leaky)
    return W, lam, 1j * P @ W @ np.diag(lam), P


def _down(Y, W, lam, V, d):
    """Admittance at the layer bottom from Y at its top (upper half: only outgoing waves above)."""
    X = np.diag(np.exp(1j * lam * d))
    rho = np.linalg.solve(V + Y @ W, V - Y @ W)
    XrX = X @ rho @ X
    I = np.eye(lam.size)
    return (V @ (I - XrX)) @ np.linalg.inv(W @ (I + XrX)), (W, X, rho)


def _up(Y, W, lam, V, d):
    """Admittance at the layer top from Y at its bottom (lower half: only outgoing waves below)."""
    X = np.diag(np.exp(1j * lam * d))
    sig = np.linalg.solve(V - Y @ W, V + Y @ W)
    XsX = X @ sig @ X
    I = np.eye(lam.size)
    return (V @ (XsX - I)) @ np.linalg.inv(W @ (XsX + I)), (W, X, sig)


class RCWA:
    """Solver for one Stack at one wavelength with 2M + 1 Floquet orders."""

    def __init__(self, stack: Stack, wavelength, orders=20):
        require_positive(wavelength=wavelength)
        self.s, self.lam0, self.M = stack, wavelength, int(orders)
        self.k0 = _TAU / wavelength
        self.K = stack.period * 0 + wavelength / stack.period          # K / k0
        self.m = np.arange(-self.M, self.M + 1)

    def _kx(self, kx0):
        return kx0 + self.m * self.K

    def _media(self, kx, leaky):
        pol, M = self.s.polarization, self.M
        return _modes(self.s.n_sub, kx, pol, M, leaky), _modes(self.s.n_cover, kx, pol, M, leaky)

    def _layer_modes(self, kx, leaky=False):
        return [(self.k0 * d, _modes(seg, kx, self.s.polarization, self.M, leaky)) for d, seg in self.s.layers]

    def _lower(self, kx, upto, leaky):
        """Y at the top of layer upto-1 (bottom-up), with the stored maps."""
        (Ws, ls, Vs, _), _ = self._media(kx, leaky)
        Y, maps = -Vs, []
        for d, (W, lam, V, _) in self._layer_modes(kx, leaky)[:upto]:
            Y, mp = _up(Y, W, lam, V, d)
            maps.append(mp)
        return Y, maps

    def _upper(self, kx, frm, leaky):
        """Y at the bottom of layer frm (top-down), with the stored maps (top layer first)."""
        _, (Wc, lc, Vc, _) = self._media(kx, leaky)
        Y, maps = Vc, []
        for d, (W, lam, V, _) in reversed(self._layer_modes(kx, leaky)[frm:]):
            Y, mp = _down(Y, W, lam, V, d)
            maps.append(mp)
        return Y, maps

    # ---------------------------------------------------------------- plane-wave diffraction
    def diffraction(self, theta=0.0):
        """Plane wave of unit amplitude from the cover at angle theta (rad, in the cover, towards +x); efficiencies of
        the reflected (cover) and transmitted (substrate) orders, m = -M..M."""
        nc = self.s.n_cover
        kx = self._kx(np.real(nc) * np.sin(theta))
        (Ws, ls, Vs, ps), (Wc, lc, Vc, pc) = self._media(kx, False)
        Yd, maps = self._lower(kx, len(self.s.layers), False)
        inc = (self.m == 0).astype(complex)
        r = np.linalg.solve(Vc - Yd, (Vc + Yd) @ inc)
        u = inc + r
        for W, X, sig in reversed(maps):
            I = np.eye(self.m.size)
            b = np.linalg.solve(W @ (X @ sig @ X + I), u)
            u = W @ (sig + I) @ X @ b
        t = u
        flux0 = np.real(lc[self.M] * pc)
        Rm = np.abs(r) ** 2 * np.real(lc * pc) / flux0
        Tm = np.abs(t) ** 2 * np.real(ls * ps) / flux0
        return {"R": Rm, "T": Tm, "r": r, "t": t, "orders": self.m.copy(), "R_total": float(Rm.sum()), "T_total": float(Tm.sum())}

    # ---------------------------------------------------------------- guided and leaky modes
    def mode_function(self, kx0, split):
        """Transverse resonance det(I - R_up R_down) at the bottom of layer `split`, with R_up = (G + Y_up)⁻¹(G - Y_up),
        R_down = (G - Y_down)⁻¹(G + Y_down) for a fixed scalar admittance G = i p n_ref (n_ref: the real part of the
        layer below, the guide core). Zero exactly where det(Y_up - Y_down) = 0, without the poles of that
        determinant, and independent of how modes of the finite layers are labelled up or down (a reference built from
        a layer's own modes jumps when a Floquet order crosses that layer's light line)."""
        kx = self._kx(kx0)
        Yd, _ = self._lower(kx, split, True)
        Yu, _ = self._upper(kx, split, True)
        seg = self.s.layers[split - 1][1] if split > 0 else self.s.n_sub
        nref = float(np.real(seg if np.isscalar(seg) else max(np.real(n) for _, n in seg)))
        G = 1j * nref * (1.0 if self.s.polarization == "TE" else 1 / nref**2) * np.eye(kx.size)
        Ru = np.linalg.solve(G + Yu, G - Yu)
        Rd = np.linalg.solve(G - Yd, G + Yd)
        return np.linalg.det(np.eye(kx.size) - Ru @ Rd)

    def leaky_mode(self, neff_guess, split, alpha_guess=None, tol=1e-13, maxit=80):
        """Complex effective index N = k_x0 / k0 of the mode near neff_guess; α = 2 k0 Im N (power, 1/m), and the
        fractions of the radiated power leaving through the cover and the substrate. alpha_guess (1/m, e.g. the
        thin-sheet estimate) sets the starting Im N: deep gratings can have other zeros near the real axis."""
        x0 = complex(neff_guess) + 1j * (0.0 if alpha_guess is None else alpha_guess / (2 * self.k0))
        if x0.imag <= 0:
            x0 += 1e-6j
        N = _muller(lambda z: self.mode_function(z, split), x0 * (1 - 1e-4), x0 * (1 + 1e-4), x0 * (1 + 3e-5j), tol, maxit)
        up, down = self._outgoing(N, split)
        tot = up + down
        return {"neff": N, "alpha": 2 * self.k0 * N.imag, "up_fraction": up / tot if tot > 0 else np.nan,
                "P_up": up, "P_down": down}

    def _outgoing(self, N, split):
        """Outgoing power (per |u|² at the split plane) through cover and substrate for the mode N."""
        kx = self._kx(N)
        Yd, lmaps = self._lower(kx, split, True)
        Yu, umaps = self._upper(kx, split, True)
        _, sv, vh = np.linalg.svd(Yu - Yd)
        u0 = np.conj(vh[-1])                                       # null vector: u at the split plane
        I = np.eye(self.m.size)
        u = u0
        for W, X, rho in reversed(umaps):                          # upward from the split plane
            a = np.linalg.solve(W @ (I + X @ rho @ X), u)
            u = W @ (I + rho) @ X @ a
        c_up = u
        u = u0
        for W, X, sig in reversed(lmaps):                          # downward from the split plane
            b = np.linalg.solve(W @ (X @ sig @ X + I), u)
            u = W @ (sig + I) @ X @ b
        c_dn = u
        (Ws, ls, Vs, ps), (Wc, lc, Vc, pc) = self._media(kx, True)
        prop_c = np.real(np.complex128(self.s.n_cover) ** 2 - kx**2) > 0
        prop_s = np.real(np.complex128(self.s.n_sub) ** 2 - kx**2) > 0
        up = float(np.sum(np.abs(c_up[prop_c]) ** 2 * np.real(lc[prop_c] * pc)))
        down = float(np.sum(np.abs(c_dn[prop_s]) ** 2 * np.real((ls * ps)[prop_s] if np.ndim(ps) else ls[prop_s] * ps)))
        return up, down

    def bloch_mode(self, neff_guess, split, tol=1e-13, maxit=80):
        """Guided Bloch mode of a lossless grating with no radiating order (e.g. a first-order DBR): N = k_x0 / k0,
        complex inside a stop band (Im N k0 = local coupling strength √(κ² - δ²))."""
        x0 = complex(neff_guess)
        return _muller(lambda z: self.mode_function(z, split), x0 - 1e-4 + 1e-4j, x0 + 1e-4 + 2e-4j, x0 + 3e-4j, tol, maxit)


def _muller(f, x0, x1, x2, tol, maxit):
    """Root of an analytic f by Muller's method; converged when the step is below tol and |f| has dropped below
    1e-6 of its starting size (a small step alone can be a stall near a pole)."""
    f0, f1, f2 = f(x0), f(x1), f(x2)
    fscale = max(abs(f0), abs(f1), abs(f2), 1e-300)
    for _ in range(maxit):
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        disc = np.sqrt(b * b - 4 * f2 * a)
        den = b + disc if abs(b + disc) > abs(b - disc) else b - disc
        dx = -2 * f2 / den if den != 0 else 1e-6
        x0, x1, x2 = x1, x2, x2 + dx
        f0, f1, f2 = f1, f2, f(x2)
        if abs(dx) < tol * max(1.0, abs(x2)) and abs(f2) < 1e-6 * fscale:
            return complex(x2)
    raise RuntimeError(f"Muller iteration did not converge (last step {abs(dx):.2e})")


# ---------------------------------------------------------------- surface gratings from grating_coupler
def surface_grating_stack(gr, wavelength, staircase=16):
    """Stack of a grating_coupler.SurfaceGrating: [handle | BOX] or substrate, unetched core (t - h), etched layer
    (rectangular: one lamellar layer; other profiles: `staircase` sublayers, core where g(u) ≥ level), cover.
    Returns (Stack, split index = the etched layer above the unetched core)."""
    ns, nf, nc = gr.indices(wavelength)
    layers = []
    if gr.n_handle is None:
        sub = ns
    else:
        sub = complex(gr.n_handle(wavelength) if callable(gr.n_handle) else gr.n_handle)
        layers.append((gr.box_thickness, ns))
    layers.append((gr.thickness - gr.etch_depth, nf))
    split = len(layers)
    if gr.etch_depth > 0:
        if gr.rect:
            f = gr.fill
            layers.append((gr.etch_depth, [((1 - f) / 2, nc), (f, nf), ((1 - f) / 2, nc)]))
        else:
            u = (np.arange(4096) + 0.5) / 4096
            g = np.array([gr.g_fn(x) for x in u])
            for j in range(staircase):
                level = (j + 0.5) / staircase
                layers.append((gr.etch_depth / staircase, _segments(g >= level, nf, nc)))
    return Stack(gr.period, sub, nc, layers, gr.polarization), split


def _segments(mask, n_on, n_off):
    seg, n, i = [], mask.size, 0
    while i < n:
        j = i
        while j < n and mask[j] == mask[i]:
            j += 1
        seg.append(((j - i) / n, n_on if mask[i] else n_off))
        i = j
    return seg


def bragg_band(gr, wavelength, orders=10, offsets=(-3.0, -2.0, 2.0, 3.0)):
    """Rigorous coupling coefficient and Bragg wavelength of a first-order (non-radiating) surface-grating DBR.

    Guided Bloch modes q = k_x0 - K/2 are found outside the stop band, at detunings offsets × κ_EIM from the EIM Bragg
    condition (real q, robust), and fitted to the coupled-mode band q² = A (ν - ν_B)² - κ² in ν = 1/λ. Returns κ, the
    Bragg wavelength, the EIM κ and the fit residual."""
    st, split = surface_grating_stack(gr, wavelength)
    kap0 = gr.coupling(wavelength, 1)
    nus, q2 = [], []
    for o in offsets:
        lam = wavelength
        for _ in range(3):                            # wavelength where the EIM half-detuning is o κ_EIM
            N0 = float(np.mean(gr.neff_profile(lam)))
            lam = 2 * np.pi * N0 / (np.pi / gr.period + o * kap0)
        k0 = _TAU / lam
        delta = k0 * N0 - np.pi / gr.period
        qg = np.sign(delta) * np.sqrt(max(delta**2 - kap0**2, 0.0))
        N = RCWA(st, lam, orders).bloch_mode(lam / (2 * gr.period) + qg / k0, split)
        q = k0 * N - np.pi / gr.period
        nus.append(1 / lam)
        q2.append(float(np.real(q * q)))
    nu = np.array(nus)
    x = (nu - nu.mean()) / nu.std()
    a, b, c = np.polyfit(x, q2, 2)
    kappa2 = b * b / (4 * a) - c
    nuB = nu.mean() - b / (2 * a) * nu.std()
    resid = float(np.max(np.abs(np.polyval([a, b, c], x) - q2)) / max(abs(np.array(q2)).max(), 1e-300))
    return {"kappa": float(np.sqrt(max(kappa2, 0.0))), "lambda_B": float(1 / nuB), "kappa_eim": kap0, "fit_residual": resid,
            "q2": q2, "wavelengths": list(1 / nu)}


# ---------------------------------------------------------------- Result front ends (spec.yaml)
def grating_leaky_mode(wavelength, n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor=0.5,
                       polarization="TE", handle="none", box_thickness=2e-6, orders=20) -> Result:
    """Rigorous radiation loss and directionality of the guided mode under a rectangular surface grating, with the
    thin-sheet estimate of grating_coupler for comparison."""
    from ..grating_coupler.engine import SurfaceGrating

    nh = {"none": None, "si": 3.476 + 0j, "au": 0.52 + 10.7j}[handle]
    gr = SurfaceGrating(n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor, "rect",
                        polarization=polarization, n_handle=nh, box_thickness=box_thickness)
    st, split = surface_grating_stack(gr, wavelength)
    est = gr.radiation(wavelength)
    if not est["valid"]:
        raise ValueError("the unetched slab is below cut-off")
    m = RCWA(st, wavelength, orders).leaky_mode(est["N0"], split, est["alpha_total"])
    up_est = sum(o["alpha"] for o in est["orders"] if o["medium"] == "cladding")
    return Result(
        values={"neff_real": m["neff"].real, "alpha": m["alpha"], "up_fraction": m["up_fraction"],
                "alpha_thin_sheet": est["alpha_total"], "up_fraction_thin_sheet": up_est / est["alpha_total"] if est["alpha_total"] > 0 else np.nan,
                "neff_eim": est["N0"]},
        units={"neff_real": "", "alpha": "1/m", "up_fraction": "", "alpha_thin_sheet": "1/m", "up_fraction_thin_sheet": "",
               "neff_eim": ""},
        assumptions=["Fourier modal method, Li's rule for TM, 2·orders + 1 Floquet orders", "Infinite periodic grating",
                     "Mode detuned from the second-order Bragg condition (at the band edge the forward and backward modes mix)",
                     "up_fraction: absorbing handles count as 'down'"],
    )


def dbr_coupling(wavelength, n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor=0.5, polarization="TE",
                 orders=10) -> Result:
    """Rigorous coupling coefficient and Bragg wavelength of a first-order rectangular surface-grating DBR near
    `wavelength`, from guided Bloch modes, with the effective-index κ for comparison."""
    from ..grating_coupler.engine import SurfaceGrating

    gr = SurfaceGrating(n_sub, n_core, n_clad, thickness, etch_depth, period, fill_factor, "rect", polarization=polarization)
    if 2 * period * max(n_sub, n_clad) > wavelength * 0.999:
        raise ValueError("the grating radiates (period too long for a first-order DBR at this wavelength)")
    r = bragg_band(gr, wavelength, orders)
    return Result(
        values={"kappa": r["kappa"], "kappa_eim": r["kappa_eim"], "lambda_B": r["lambda_B"], "fit_residual": r["fit_residual"]},
        units={"kappa": "1/m", "kappa_eim": "1/m", "lambda_B": "m", "fit_residual": ""},
        assumptions=["Fourier modal method, Li's rule for TM", "Infinite periodic grating",
                     "κ and λ_B from the coupled-mode band q² = A(ν - ν_B)² - κ² fitted outside the stop band"],
    )
