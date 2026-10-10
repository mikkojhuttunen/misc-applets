"""CRIGF: cavity-resonant integrated grating filter, DBR | spacer | grating coupler | spacer | DBR, fed by a
free-space Gaussian beam.

Grating coupler (GC) by coupled-mode theory (Kazarinov–Henry): the guided field under the coupler is
U = R(ζ) e^{iβζ} + S(ζ) e^{-iβζ}, with radiative self- and cross-coupling (both waves radiate into the same
near-normal plane waves), second-order guided Bragg coupling from the n_eff modulation, propagation loss, and
excitation by the beam through the local field at the grating (bare-slab plane wave, BOX interference included).
The DBRs and spacers are exact transfer matrices on effective indices, seen from the coupler as boundary
reflections. Out-coupling into the reflected and transmitted beams follows by reciprocity from the same coupling
constant. SI units, vacuum wavelength, angles in rad, losses are power attenuation coefficients.

Python reference for crigfSetup, crigfAt and findResonance in dbr-structures/dbr-engine.js (lengths in µm there).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Result, require_nonnegative, require_positive, require_range
from ..grating_coupler.engine import SurfaceGrating, fourier_coefficient, slab_mode

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12
ETA0 = MU0 * C0
_TAU = 2 * np.pi


def layer_matrix(n, k, d):
    """Abelès characteristic matrix of a layer of complex index n (loss: Im n > 0) and thickness d."""
    dl = k * n * d
    c, s = np.cos(dl), np.sin(dl)
    return np.array([[c, -1j * s / n], [-1j * n * s, c]], dtype=complex)


def r_t(M, n_in, n_out):
    """Amplitude r, t and power R, T of a characteristic matrix between media n_in and n_out (real)."""
    x = n_in * M[0, 0] + n_in * n_out * M[0, 1]
    y = M[1, 0] + n_out * M[1, 1]
    r, t = (x - y) / (x + y), 2 * n_in / (x + y)
    return r, t, abs(r) ** 2, n_out / n_in * abs(t) ** 2


def period_slices(gr: SurfaceGrating):
    """(width fraction, fill g) of the slices of one period: three for a rectangular tooth, else 64 midpoints."""
    if gr.rect:
        f = gr.fill
        return [((1 - f) / 2, 0.0), (f, 1.0), ((1 - f) / 2, 0.0)]
    return [(1 / 64, gr.g_fn((i + 0.5) / 64)) for i in range(64)]


@dataclass
class CRIGF:
    """dbr: the DBR grating (waveguide stack, etch, fill, profile, period); the coupler shares the stack and has its
    own etch gc_etch_depth, fill gc_fill, period gc_period and tooth profile gc_profile (rectangular in the JS port; other
    profiles of grating_coupler.profile_function for band-edge studies). spacer + straight: guide between the coupler
    and each DBR. theta: beam angle in the cladding; w0: 1/e² intensity radius along the guide. overlap_y: lateral
    field overlap of beam and guided mode (amplitude); bounce_eta: power kept per DBR bounce (lateral losses of curved
    or finite DBRs, tapers); alpha_extra: extra loss in the straight sections. steps: RK4 steps over the coupler
    (default max(32, ⌈2.5 N_gc⌉), as in the JS port)."""

    dbr: SurfaceGrating
    dbr_periods: int
    gc_period: float
    gc_periods: int
    gc_etch_depth: float
    gc_fill: float = 0.5
    spacer: float = 0.0
    straight: float = 0.0
    theta: float = 0.0
    w0: float = 10e-6
    alpha_prop: float = 0.0
    alpha_extra: float = 0.0
    overlap_y: float = 1.0
    bounce_eta: float = 1.0
    steps: int | None = None
    gc_profile: str = "rect"

    def __post_init__(self):
        d = self.dbr
        require_positive(gc_period=self.gc_period, w0=self.w0)
        require_nonnegative(spacer=self.spacer, straight=self.straight, alpha_prop=self.alpha_prop, alpha_extra=self.alpha_extra)
        require_range("bounce_eta", self.bounce_eta, 0.0, 1.0)
        if int(self.gc_periods) < 1 or int(self.dbr_periods) < 0:
            raise ValueError("gc_periods must be >= 1 and dbr_periods >= 0")
        self.gc = SurfaceGrating(d.n_sub, d.n_core, d.n_clad, d.thickness, self.gc_etch_depth, self.gc_period, self.gc_fill,
                                 self.gc_profile, polarization=d.polarization, n_handle=d.n_handle, box_thickness=d.box_thickness,
                                 samples=d.samples, table_points=d.table_points)
        self.n_steps = self.steps or max(32, int(np.ceil(self.gc_periods * 2.5)))

    @property
    def gc_length(self):
        return self.gc_periods * self.gc_period

    # ---------------------------------------------------------------- mirrors
    def mirrors(self, wavelength, n_ref):
        """Characteristic matrices of spacer + DBR seen from the coupler, left (traversed outward) and right, and the
        unetched index n_out of the guide beyond them."""
        k = _TAU / wavelength
        tab = self.dbr.neff_table(wavelength)
        sl = [(w * self.dbr.period, self.dbr._interp(tab, g) + 1j * self.alpha_prop / (2 * k)) for w, g in period_slices(self.dbr)]
        P = np.eye(2, dtype=complex)
        for d, n in sl:
            P = P @ layer_matrix(n, k, d)
        Prev = np.eye(2, dtype=complex)
        for d, n in reversed(sl):
            Prev = Prev @ layer_matrix(n, k, d)
        n_sp = self.gc.neff_table(wavelength)["vals"][-1] + 1j * (self.alpha_prop + self.alpha_extra) / (2 * k)
        Msp = layer_matrix(n_sp, k, self.spacer + self.straight)
        N = int(self.dbr_periods)
        return Msp @ np.linalg.matrix_power(Prev, N), Msp @ np.linalg.matrix_power(P, N), float(tab["vals"][-1])

    # ---------------------------------------------------------------- full response
    def response(self, wavelength):
        """Reflection R into the incident beam mode, transmission T into the substrate beam, guided escape through each
        DBR, lateral loss at the DBRs, the bare-slab reflectance Rd, peak guided power |R|² + |S|² (per unit beam
        power), the coupler's radiation loss α_rad, and the DBR reflections r_L, r_R seen from the coupler.
        Also the envelopes (R, S) on the RK4 grid z, the three coupler solutions at ζ = L (Y_end: homogeneous from
        (R, S) = (1, 0) and (0, 1), driven from (0, 0)) and the coupled-mode coefficients (coef) for independent checks."""
        lam = wavelength
        k = _TAU / lam
        om = _TAU * C0 / lam
        dbr, gc = self.dbr, self.gc
        ns, nf, nc = dbr.indices(lam)
        tabG = gc.neff_table(lam)
        nPG = gc._interp(tabG, gc.g)
        N0G = float(np.mean(nPG))
        beta = k * N0G
        KG = _TAU / self.gc_period
        D = KG - beta
        G1 = fourier_coefficient(gc.g, 1)
        Gm1 = np.conj(G1)
        N2 = fourier_coefficient(nPG, 2)
        Nm2 = np.conj(N2)
        h, t = self.gc_etch_depth, dbr.thickness
        m = slab_mode(lam, ns, nf, nc, t - h + h * float(np.mean(gc.g)), dbr.polarization, 0)
        xg = t - h / 2
        eg = np.sqrt(2 * om * MU0 / beta) * float(m["field"](xg)) / np.sqrt(m["norm"]) if m else 0.0
        aC = 0.25 * om * EPS0 * (nf * nf - nc * nc) * h * eg              # coupling a = i aC
        kz_in = k * nc * np.sin(self.theta)
        kz_r = beta - KG
        chi = 0.0
        for med, nj in dbr.channels(lam):
            lf = dbr.plane_wave(lam, kz_r, xg, "top" if med == "cladding" else "bottom")
            chi += 0.5 * abs(lf["F"]) ** 2 / np.sqrt(max(k * k * nj * nj - kz_r * kz_r, 1.0))
        rad = 2 * om * MU0 * chi * aC * aC
        aP = self.alpha_prop
        pw_top = dbr.plane_wave(lam, kz_in, xg, "top")
        pw_bot = dbr.plane_wave(lam, kz_in, xg, "bottom") or {"F": 0j, "tau": 0j}
        nB = dbr.bottom_index(lam)
        Lsp = self.spacer + self.straight
        Lg = self.gc_length
        z0 = self.dbr_periods * self.dbr.period + Lsp
        zc = Lg / 2
        E0 = np.sqrt(2 * ETA0 / (nc * self.w0 * np.sqrt(np.pi / 2)))
        th_s = np.arcsin(min(1.0, nc * np.sin(self.theta) / nB))
        E0s = np.sqrt(2 * ETA0 / (nB * self.w0 * np.sqrt(np.pi / 2)))
        oy = self.overlap_y
        phase0 = kz_in * z0
        cth, cths = np.cos(self.theta), np.cos(th_s)

        def beam(z, c):
            return np.exp(-(((z - zc) * c) / self.w0) ** 2)

        g2 = abs(G1) ** 2

        def rhs(z, Y, with_src):
            eP, e2P = np.exp(1j * D * z), np.exp(2j * D * z)
            eM, e2M = np.conj(eP), np.conj(e2P)
            R, S = Y[:, 0], Y[:, 1]
            dR = R * (-rad * g2 - aP / 2) - rad * G1 * G1 * e2P * S + 1j * k * N2 * e2P * S
            dS = S * (rad * g2 + aP / 2) + rad * Gm1 * Gm1 * e2M * R - 1j * k * Nm2 * e2M * R
            out = np.stack([dR, dS], axis=1)
            if with_src:
                E = pw_top["F"] * np.exp(1j * (kz_in * z + phase0)) * oy * E0 * beam(z, cth)
                out[2] += [1j * aC * G1 * eP * E, -1j * aC * Gm1 * eM * E]
            return out

        n = self.n_steps
        hs = Lg / n
        Y = np.array([[1, 0], [0, 1], [0, 0]], dtype=complex)        # two homogeneous solutions and the driven one
        hist = [Y]
        for i in range(n):
            z = i * hs
            k1 = rhs(z, Y, True)
            k2 = rhs(z + hs / 2, Y + k1 * hs / 2, True)
            k3 = rhs(z + hs / 2, Y + k2 * hs / 2, True)
            k4 = rhs(z + hs, Y + k3 * hs, True)
            Y = Y + (k1 + 2 * k2 + 2 * k3 + k4) * hs / 6
            hist.append(Y)
        MLrev, MR, n_out = self.mirrors(lam, N0G)
        rL0, _, _, TL = r_t(MLrev, N0G, n_out)
        rR0, _, _, TR = r_t(MR, N0G, n_out)
        sb = np.sqrt(self.bounce_eta)
        rL, rR = sb * rL0, sb * rR0
        rho = rR * np.exp(2j * beta * Lg)
        y1, y2, yp = Y
        S0 = (rho * yp[0] - yp[1]) / (y1[1] * rL + y2[1] - rho * (y1[0] * rL + y2[0]))
        R0 = rL * S0
        ys = np.array([H[0] * R0 + H[1] * S0 + H[2] for H in hist])     # (R, S) along the coupler
        PinL, PinR = abs(S0) ** 2, abs(ys[-1, 0]) ** 2
        z = np.arange(n + 1) * hs
        eP = np.exp(1j * D * z)
        Kn = -4j * aC * (Gm1 * np.conj(eP) * ys[:, 0] + G1 * eP * ys[:, 1])
        ph = np.exp(-1j * (kz_in * z + phase0))
        Er = pw_top["F"] * ph * oy * E0 * beam(z, cth)
        Et = pw_bot["F"] * ph * oy * E0s * beam(z, cths)
        w = np.full(n + 1, hs)
        w[0] = w[-1] = hs / 2
        rg, tg = -0.25 * np.sum(Kn * Er * w), -0.25 * np.sum(Kn * Et * w)
        R, T = abs(pw_top["r"] + rg) ** 2, abs(pw_top["tau"] + tg) ** 2
        escL, escR = PinL * TL, PinR * TR
        lat = (PinL * abs(rL0) ** 2 + PinR * abs(rR0) ** 2) * (1 - self.bounce_eta)
        return {"R": R, "T": T, "Rd": abs(pw_top["r"]) ** 2, "escL": escL, "escR": escR, "lat_loss": lat,
                "other": 1 - R - T - escL - escR - lat, "U_max": float(np.max(np.abs(ys[:, 0]) ** 2 + np.abs(ys[:, 1]) ** 2)),
                "alpha_rad": 2 * rad * g2, "rL": rL, "rR": rR, "N0G": N0G, "n_out": n_out, "beta": beta, "D": D,
                "z": z, "RS": ys, "Y_end": Y,
                "coef": {"rad": rad, "aC": aC, "G1": G1, "N2": N2, "k": k, "D": D, "alpha_prop": aP, "F_top": pw_top["F"],
                         "kz_in": kz_in, "phase0": phase0, "E0": E0, "overlap_y": oy, "zc": zc, "w0": self.w0, "cos_theta": cth,
                         "r_top": pw_top["r"], "tau_top": pw_top["tau"], "F_bot": pw_bot["F"], "n_bot": nB, "n_clad": nc,
                         "cos_theta_bot": cths}}

    def infinite_grating(self, wavelength):
        """Plane-wave limit of the coupler model: an infinite coupler grating (no DBRs) under a plane wave at theta.
        Steady state of the coupled-mode equations in the rotating frame, R = ρ e^{i(D + k_z)ζ}, S = σ e^{i(k_z - D)ζ};
        returns the specular reflectance R, the transmittance T into the substrate and R + T (1 without loss, since a
        second-order grating radiates only into the specular orders). For benchmarks against rigorous solvers."""
        co = self.response(wavelength)["coef"]
        rad, aC, G1, N2, k, D, aP, kz = (co[x] for x in ("rad", "aC", "G1", "N2", "k", "D", "alpha_prop", "kz_in"))
        g2 = abs(G1) ** 2
        A = np.array([[-rad * g2 - aP / 2 - 1j * D, -rad * G1 * G1 + 1j * k * N2],
                      [rad * np.conj(G1) ** 2 - 1j * k * np.conj(N2), rad * g2 + aP / 2 + 1j * D]])
        b = np.array([1j * aC * G1 * co["F_top"], -1j * aC * np.conj(G1) * co["F_top"]])
        rho, sig = np.linalg.solve(A - 1j * kz * np.eye(2), -b)
        Kn = -4j * aC * (np.conj(G1) * rho + G1 * sig)
        nc, nb, c, cb = co["n_clad"], co["n_bot"], co["cos_theta"], co["cos_theta_bot"]
        r = co["r_top"] - 0.25 * Kn * co["F_top"] * 2 * ETA0 / (nc * c)
        t = co["tau_top"] - 0.25 * Kn * co["F_bot"] * 2 * ETA0 / np.sqrt(nc * c * nb * cb)
        return {"R": abs(r) ** 2, "T": abs(t) ** 2, "sum": abs(r) ** 2 + abs(t) ** 2, "r": r, "t": t}

    def band_edge_modes(self, wavelength):
        """Coupled-mode band-edge modes of the infinite coupler grating at the Γ point (normal incidence): complex
        wavelengths where the free (undriven) envelope equations have a uniform solution, det A(λ) = 0 with
        A = [[-a - iD, -ρ G₁² + i k N₂], [ρ G₋₁² - i k N₋₂, a + iD]], a = ρ|G₁|² + α/2, D = K - β(λ).
        Closed form in D (quadratic), then λ from the linear dispersion D(λ) about `wavelength`. Returns both modes
        sorted by Q: the bright (radiating) one and the dark one (Q = ∞ for a symmetric tooth: symmetry-protected
        bound state). Coefficients are evaluated at the real `wavelength`."""
        co = self.response(wavelength)["coef"]
        rad, G1, N2, k, D0, aP = (co[x] for x in ("rad", "G1", "N2", "k", "D", "alpha_prop"))
        h = wavelength * 1e-4
        beta = [_TAU / l * float(np.mean(self.gc.neff_profile(l))) for l in (wavelength - h, wavelength + h)]
        dD = -(beta[1] - beta[0]) / (2 * h)                            # dD/dλ = -dβ/dλ = 2π n_g / λ² > 0
        a = rad * abs(G1) ** 2 + aP / 2
        X = G1 * G1 * np.conj(N2) + N2 * np.conj(G1) ** 2
        S = np.sqrt(complex(rad * rad * abs(G1) ** 4 - 1j * k * rad * X - k * k * abs(N2) ** 2))
        modes = []
        for sgn in (1, -1):
            D = 1j * (a - sgn * S)                                     # i D = -a + sgn S
            lam = wavelength + (D - D0) / dD
            modes.append({"wavelength": complex(lam), "Q": lam.real / (2 * lam.imag) if abs(lam.imag) > 0 else np.inf,
                          "D": complex(D)})
        modes.sort(key=lambda m: abs(m["Q"]))
        return {"bright": modes[0], "dark": modes[1], "n_group": dD * wavelength**2 / _TAU,
                "kappa2": float(k * abs(N2)), "alpha_rad": float(2 * rad * abs(G1) ** 2)}

    def find_resonance(self, lam_c, fsr, n=40, width=True):
        """Cavity resonance nearest lam_c: maximise the circulating guided power over one free spectral range
        (grid of n + 1 points, then 30 ternary steps); FWHM by bisection to half the peak on each side."""
        def U(l):
            return self.response(l)["U_max"]

        a, h = lam_c - fsr / 2, fsr / n
        ib, vb = 0, -1.0
        for i in range(n + 1):
            v = U(a + i * h)
            if v > vb:
                ib, vb = i, v
        lo, hi = a + (ib - 1) * h, a + (ib + 1) * h
        for _ in range(30):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if U(m1) > U(m2):
                hi = m2
            else:
                lo = m1
        lam = (lo + hi) / 2
        Um = U(lam)
        if not width:
            return {"wavelength": lam, "U_max": Um, "fwhm": np.nan}

        def edge(sgn):
            l0, l1 = lam, lam + sgn * fsr / 2
            for _ in range(28):
                mid = (l0 + l1) / 2
                if U(mid) > Um / 2:
                    l0 = mid
                else:
                    l1 = mid
            return (l0 + l1) / 2

        return {"wavelength": lam, "U_max": Um, "fwhm": edge(1) - edge(-1)}


# ---------------------------------------------------------------- Result front end (spec.yaml)
def crigf_response(wavelength, n_sub, n_core, n_clad, thickness, dbr_etch_depth, dbr_period, dbr_periods, gc_etch_depth,
                   gc_period, gc_periods, dbr_fill=0.5, gc_fill=0.5, spacer=0.0, theta=0.0, w0=10e-6, alpha_prop=0.0,
                   polarization="TE", handle="none", box_thickness=2e-6) -> Result:
    """Power budget of a CRIGF at one wavelength for a Gaussian beam of unit power."""
    nh = {"none": None, "si": 3.476 + 0j, "au": 0.52 + 10.7j}[handle]
    dbr = SurfaceGrating(n_sub, n_core, n_clad, thickness, dbr_etch_depth, dbr_period, dbr_fill, "rect",
                         polarization=polarization, n_handle=nh, box_thickness=box_thickness)
    r = CRIGF(dbr, int(dbr_periods), gc_period, int(gc_periods), gc_etch_depth, gc_fill, spacer, 0.0, theta, w0,
              alpha_prop).response(wavelength)
    keys = ("R", "T", "Rd", "escL", "escR", "other", "U_max", "alpha_rad")
    vals = {k: float(r[k]) for k in keys}
    vals["R_dbr"] = float(abs(r["rR"]) ** 2)
    return Result(
        values=vals,
        units={**{k: "" for k in keys}, "alpha_rad": "1/m", "R_dbr": ""},
        assumptions=["Kazarinov–Henry coupled-mode coupler, effective-index method, rectangular coupler teeth",
                     "DBRs and spacers by exact transfer matrices on effective indices",
                     "Gaussian beam in one transverse dimension, lateral overlap 1; TE local fields (TM rough)",
                     "other = 1 - R - T - escape: radiation into modes other than the beam, plus loss"],
    )
