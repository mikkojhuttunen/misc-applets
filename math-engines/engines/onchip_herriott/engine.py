"""On-chip (integrated, in-plane) Herriott cell: Gaussian-beam path and volume budget against mirror reflectance.

Geometry and rays come from planar_cell.herriott_planar_cell: two concave in-plane mirrors (radius R, spacing d), a
window on M1 for input and output, spot pattern y_n = A cos(nθ + φ). The light is the cell's own Gaussian mode riding on
the traced ray (a mode-matched beam keeps the resonator's width on every pass):

    λm = λ / n_eff,  zR = √(d (2R − d)) / 2,  w0² = λm zR / π,  w(x) = w0 √(1 + (x/zR)²)    (x along the cell axis)
    w on the mirrors w_m = w(±d/2);  far-field (angular-spectrum) half-angle θ0 = λm / (π w0)

Transverse effective width of a Gaussian with peak-normalised intensity: w_eff = ∫ exp(−2u²/w²) du = w √(π/2); the
vertical (slab) effective height h_eff is an input (slab mode ∫|E|² / max|E|²; multiply by Γ for the evanescent part).

Power bookkeeping along the ray, hit j = 1, 2, ... (I_0 = coupling into the cell through the window):
    I_{j} = I_{j−1} · R_j · c_j             c_j = share of the beam on the mirror and off the window (Gaussian, 1D)
    L_eff = Σ_j I_{j−1} ℓ_j                 absorption-weighted path (a weak absorber α removes α L_eff of the input)
    V_eff = h_eff Σ_j I_{j−1} ∫_chord w_eff ds    absorption-weighted mode volume; A_eff = V_eff / L_eff
    ⟨|χ|⟩ = Σ_j I_{j−1} |χ_j| / Σ_j I_{j−1}  power-weighted angle of incidence (over reflections)
    T = I_{N−1} · η_out                     power leaving through the window after N passes (η_out: window share)

Mirror models: constant R, or an etched-trench DBR (bragg_grating.TrenchDBR, exact transfer matrix at every hit)
evaluated at the ray angle or averaged over the beam's angular spectrum (power ∝ exp(−2α²/θ0²)), plus an extra
scatter loss per bounce. exit_mode 'window': the re-entrant design, light leaves through the window at hit N;
'closed': the window is mirror too and the light circulates until its power drops below 1e-6 (L_eff → ℓ̄/(1 − R)).
SI units; angles in rad unless named *_deg.
"""
from __future__ import annotations

import inspect
import math

import numpy as np

from ..bragg_grating.engine import TrenchDBR
from ..common import Result, require_choice, require_nonnegative, require_positive, require_range
from ..planar_cell.engine import herriott_planar_cell, herriott_planar_trace, in_window, launch, perturb, reflect

N_ANGLE = 31          # angular-spectrum samples over ±3σ (σ = θ0/2)
N_CHORD = 17          # Simpson points per chord for ∫ w ds


def mode_profile(R, d, wavelength, n_eff=1.0):
    """Symmetric two-mirror cell mode in a medium of index n_eff: dict(lam_m, zR, w0, w_mirror, theta0)."""
    lam_m = wavelength / n_eff
    zR = math.sqrt(d * (2 * R - d)) / 2
    w0 = math.sqrt(lam_m * zR / math.pi)
    return dict(lam_m=lam_m, zR=zR, w0=w0, w_mirror=w0 * math.sqrt(1 + (d / 2 / zR) ** 2), theta0=lam_m / (math.pi * w0))


def beam_width(x, mode):
    return mode["w0"] * np.sqrt(1 + (np.asarray(x, float) / mode["zR"]) ** 2)


def chord_width_integral(x0, y0, x1, y1, mode, n=N_CHORD):
    """∫ w(x) ds along straight chords (arrays), Simpson's rule on n (odd) points."""
    x0, y0, x1, y1 = (np.asarray(v, float) for v in (x0, y0, x1, y1))
    f = np.linspace(0.0, 1.0, n)
    wv = beam_width(x0[..., None] + f * (x1 - x0)[..., None], mode)
    c = np.ones(n)
    c[1:-1:2], c[2:-1:2] = 4, 2
    return np.hypot(x1 - x0, y1 - y0) * (wv * c).sum(-1) / (3 * (n - 1))


def gauss_fraction(lo, hi, y, w):
    """Share of a 1D Gaussian beam (intensity ∝ exp(−2(u − y)²/w²)) between lo and hi."""
    s = math.sqrt(2) / w
    return 0.5 * (math.erf(s * (hi - y)) - math.erf(s * (lo - y)))


def angle_weights(theta0, n=N_ANGLE):
    """Offsets α over ±1.5 θ0 (±3σ) and normalised weights ∝ exp(−2α²/θ0²)."""
    a = 1.5 * theta0 * (2 * np.arange(n) / (n - 1) - 1)
    wt = np.exp(-2 * (a / theta0) ** 2)
    return a, wt / wt.sum()


def mirror_function(model="constant", R_mirror=0.999, dbr=None, wavelength=1.55e-6, theta0=0.0, beam_average=False,
                    scatter=0.0):
    """R(|χ|, θ0) for one hit: constant, or a DBR (anything with R(λ, |sin χ|)) at the ray angle or averaged over the
    beam's angular spectrum of half-angle θ0 (the per-hit value when given, else the default theta0); × (1 − scatter)."""
    if model == "constant":
        return lambda chi, t=None: R_mirror * (1 - scatter)
    if beam_average and theta0 > 0:
        def avg(chi, t=None):
            a, wt = angle_weights(t if t else theta0)
            return float((wt * dbr.R(wavelength, np.abs(np.sin(abs(chi) + a)))).sum()) * (1 - scatter)
        return avg
    return lambda chi, t=None: float(dbr.R(wavelength, abs(math.sin(chi)))) * (1 - scatter)


def trace_hits(cell, dtheta=0.0, exit_mode="window", n_max=4000):
    """Hit list of the centre ray: x, y (hit points, start included), element, chi (signed angle of incidence), and
    whether the last hit is the exit through the window."""
    if exit_mode == "window":
        tr = herriott_planar_trace(cell, dtheta, n_max=n_max)
        xs, ys, K, SC = tr["xs"], tr["ys"], tr["element"], tr["sinchi"]
        exit_hit = tr["exit"] == "port"
    else:
        x, y, dx, dy = launch(cell, cell.s_in_default, cell.meta["theta_launch"] + dtheta)
        xs, ys, K, SC = [x[0]], [y[0]], [], []
        for _ in range(n_max):
            t, nx, ny, hs, hk = cell.hit_k(x, y, dx, dy)
            if not np.isfinite(t[0]):
                break
            SC.append(float(dx[0] * ny[0] - dy[0] * nx[0]))
            x, y = x + t * dx, y + t * dy
            xs.append(x[0]); ys.append(y[0]); K.append(int(hk[0]))
            dx, dy = reflect(dx, dy, nx, ny)
        xs, ys, K, SC = np.array(xs), np.array(ys), np.array(K), np.array(SC)
        exit_hit = False
    return dict(x=np.asarray(xs, float), y=np.asarray(ys, float), element=np.asarray(K, int),
                chi=np.arcsin(np.clip(np.asarray(SC, float), -1, 1)), exit=exit_hit)


DB_PER_CM = 100 * math.log(10) / 10      # dB/cm -> power attenuation coefficient [1/m]


def mirror_power(cell, k, chi):
    """Tangential (in-plane) focusing power 2/(R cos χ) of element k at incidence χ: + concave, − convex, 0 flat."""
    kap = cell.kappa[k]
    return 0.0 if kap == 0 else -math.copysign(2.0 / (cell.R[k] * math.cos(chi)), kap)


def _abcd_mul(M, N):
    return (M[0] * N[0] + M[1] * N[2], M[0] * N[1] + M[1] * N[3], M[2] * N[0] + M[3] * N[2], M[2] * N[1] + M[3] * N[3])


def path_matrix(cell, chords, element, chi, j0, j1):
    """ABCD matrix from the start of chord j0 to just after the reflection at hit j1 − 1 (chords j0 .. j1 − 1)."""
    M = (1.0, 0.0, 0.0, 1.0)
    for j in range(j0, j1):
        M = _abcd_mul((1.0, chords[j], 0.0, 1.0), M)
        M = _abcd_mul((1.0, 0.0, -mirror_power(cell, element[j], chi[j]), 1.0), M)
    return M


def eigen_q(M):
    """Self-consistent q of an ABCD round trip, 1/q = (D − A)/(2B) − i √(1 − m²)/|B|, m = (A + D)/2; None if |m| ≥ 1."""
    A, B, C, D = M
    m = 0.5 * (A + D)
    if not abs(m) < 1 or B == 0:
        return None
    return 1 / complex((D - A) / (2 * B), -math.sqrt(1 - m * m) / abs(B))


def q_width(q, lam_m):
    """1/e² radius of a Gaussian with complex beam parameter q (1/q = 1/R − i λm/(π w²))."""
    return math.sqrt(-lam_m / (math.pi * (1 / q).imag))


def coupling(q1, q2, lam_m):
    """Power overlap of two 1D Gaussian beams at the same plane: 2 / (w1 w2 |a|), a = 1/w1² + 1/w2² + i k (1/R1 − 1/R2)/2."""
    u1, u2 = 1 / q1, 1 / q2
    w1, w2 = q_width(q1, lam_m), q_width(q2, lam_m)
    a = complex(-math.pi / lam_m * (u1.imag + u2.imag), math.pi / lam_m * (u1.real - u2.real))
    return 2 / (w1 * w2 * abs(a))


def beam_on_path(cell, hits, lam_m, beam=None, n=N_CHORD):
    """Beam widths at the Simpson points of every chord (array (chords, n)), the angular-spectrum half-angle per chord,
    and the stability data.
    beam None: the paraxial design mode w(x) on every pass (mode_profile).
    beam dict(w_ratio=1, focus_shift=0): Gaussian q propagated along the traced chords with the in-plane mirror powers
    2/(R cos χ); the matched input is the eigenmode of the first round trip (chords 1–2) at the launch point, the
    actual input has its waist scaled by w_ratio and moved by focus_shift [m] along chord 1."""
    x, y, K, chi = hits["x"], hits["y"], hits["element"], hits["chi"]
    chords = np.hypot(np.diff(x), np.diff(y))
    nch = len(chords)
    f = np.linspace(0.0, 1.0, n)
    m = cell.meta
    if beam is None:
        mode = mode_profile(m["R"], m["d"], lam_m, 1.0)
        W = beam_width(x[:-1, None] + f * (x[1:] - x[:-1])[:, None], mode)
        return dict(W=W, theta0=np.full(nch, mode["theta0"]), chords=chords, stability=None)
    stab = {}
    Mrt = path_matrix(cell, chords, K, chi, 0, min(2, nch))
    q_eig = eigen_q(Mrt) if nch >= 2 else None
    if q_eig is None:                                   # unstable round trip: fall back to the design mode at the launch
        mode = mode_profile(m["R"], m["d"], lam_m, 1.0)
        q_eig = complex(-chords[0] / 2, mode["zR"])
    zR0, sw = q_eig.imag, -q_eig.real                   # eigen waist position along chord 1
    q_in = complex(-(sw + beam.get("focus_shift", 0.0)), zR0 * beam.get("w_ratio", 1.0) ** 2)
    W, th = np.empty((nch, n)), np.empty(nch)
    q = q_in
    for j in range(nch):
        W[j] = [q_width(q + fi * chords[j], lam_m) for fi in f]
        th[j] = lam_m / (math.pi * math.sqrt(lam_m * q.imag / math.pi))
        qh = q + chords[j]
        u = 1 / qh - mirror_power(cell, K[j], chi[j])
        q = 1 / u
    mrt = 0.5 * (Mrt[0] + Mrt[3])
    MN = path_matrix(cell, chords, K, chi, 0, nch) if nch else (1.0, 0.0, 0.0, 1.0)
    stab = dict(m_roundtrip=mrt, gouy_roundtrip=math.acos(max(-1.0, min(1.0, mrt))), stable=abs(mrt) < 1,
                m_path=0.5 * (MN[0] + MN[3]), w0_eigen=math.sqrt(lam_m * zR0 / math.pi), waist_pos=sw,
                coupling=coupling(q_in, q_eig, lam_m), q_in=q_in, q_eigen=q_eig)
    return dict(W=W, theta0=th, chords=chords, stability=stab)


def budget(cell, wavelength=1.55e-6, n_eff=2.479, h_eff=0.3e-6, R_of_chi=None, dtheta=0.0, exit_mode="window",
           clip=True, n_max=4000, stop_below=1e-6, alpha=0.0, beam=None):
    """Power, path, volume and angle budget of the centre ray's Gaussian beam (module docstring). R_of_chi(|χ|[, θ0])
    is the mirror power reflectance per hit (default 1); alpha the waveguide power attenuation [1/m] (DB_PER_CM × dB/cm);
    beam None (design mode on every pass) or dict(w_ratio, focus_shift) for q propagation (beam_on_path).
    Returns a dict of totals, per-hit arrays and the beam's stability data."""
    m = cell.meta
    lam_m = wavelength / n_eff
    hits = trace_hits(cell, dtheta, exit_mode, n_max)
    y, K, chi = hits["y"], hits["element"], hits["chi"]
    n = len(K)
    bp = beam_on_path(cell, hits, lam_m, beam)
    W, th0, chords = bp["W"], bp["theta0"], bp["chords"]
    w_hit = W[:, -1]
    w_launch = W[0, 0] if n else math.nan
    a, y0, pw = m["aperture"], m["y0"], m["port_w"]
    win = (y0 - pw / 2, y0 + pw / 2)
    eta = gauss_fraction(*win, y0, w_launch) if exit_mode == "window" else 1.0
    Rf = R_of_chi or (lambda c, t=0.0: 1.0)
    if len(inspect.signature(Rf).parameters) < 2:
        Rf1 = Rf
        Rf = lambda c, t=0.0: Rf1(c)                   # noqa: E731  (R(|χ|) without the beam angle)
    nS = W.shape[1]
    f = np.linspace(0.0, 1.0, nS)
    cS = np.ones(nS)
    cS[1:-1:2], cS[2:-1:2] = 4, 2
    att = np.exp(-alpha * chords[:, None] * f)
    wint = chords * (W * cS).sum(-1) / (3 * (nS - 1)) * math.sqrt(math.pi / 2)              # ∫ w_eff ds
    wint_a = chords * (W * att * cS).sum(-1) / (3 * (nS - 1)) * math.sqrt(math.pi / 2)      # ∫ w_eff e^{-αs} ds
    lch = np.where(alpha > 0, -np.expm1(-alpha * chords) / max(alpha, 1e-300), chords)      # ∫ e^{-αs} ds
    T_ch = np.exp(-alpha * chords)
    I = np.empty(n + 1)
    I[0] = eta
    Rj, cj = np.ones(n), np.ones(n)
    last = n - 1 if hits["exit"] else n               # the exit hit is not a reflection
    for j in range(n):
        if j < last:
            Rj[j] = Rf(abs(chi[j]), th0[j])
            if clip:
                on = gauss_fraction(-a, a, y[j + 1], w_hit[j])
                off_win = gauss_fraction(*win, y[j + 1], w_hit[j]) if (exit_mode == "window" and K[j] == 0) else 0.0
                cj[j] = max(on - off_win, 0.0)
        I[j + 1] = I[j] * T_ch[j] * (Rj[j] * cj[j] if j < last else 1.0)
        if exit_mode == "closed" and I[j + 1] < stop_below * I[0]:
            n = last = j + 1
            break
    I, Rj, cj, chords, chi, K = I[:n + 1], Rj[:n], cj[:n], chords[:n], chi[:n], K[:n]
    wint, wint_a, lch, w_hit, th0 = wint[:n], wint_a[:n], lch[:n], w_hit[:n], th0[:n]
    w_in = I[:-1]                                      # power entering chord j
    L_eff = float((w_in * lch).sum())
    V_eff = float(h_eff * (w_in * wint_a).sum())
    refl = slice(0, last)
    wr = I[:last] * T_ch[:last]                        # power arriving at each reflection
    achi = np.abs(chi[refl])
    T_out = float(I[n] * eta) if hits["exit"] else 0.0
    return dict(mode=mode_profile(m["R"], m["d"], wavelength, n_eff), n_hits=n, reflections=last, exits=bool(hits["exit"]),
                eta_window=eta, T_out=T_out, I_end=float(I[last] if not hits["exit"] else I[n]), L_geom=float(chords.sum()), L_eff=L_eff,
                V_eff=V_eff, A_eff=V_eff / L_eff if L_eff else np.nan, w_eff_mean=V_eff / L_eff / h_eff if L_eff else np.nan,
                V_geom=float(h_eff * wint.sum()),
                chi_mean=float((wr * achi).sum() / wr.sum()) if last else np.nan,
                chi_rms=float(np.sqrt((wr * achi**2).sum() / wr.sum())) if last else np.nan,
                chi_max=float(achi.max()) if last else np.nan, R_mean=float((wr * Rj[refl]).sum() / wr.sum()) if last else np.nan,
                clip_loss=float(1 - np.prod(cj[refl])), I=I, R=Rj, c=cj, chords=chords, w_int=wint, w_hit=w_hit, theta0=th0,
                chi=chi, element=K, x=hits["x"][:n + 1], y=hits["y"][:n + 1], stability=bp["stability"], W=bp["W"][:n])


def footprint_fraction(cell, wavelength=1.55e-6, n_eff=2.479, dtheta=0.0, nx=240, ny=120):
    """Share of the cell area (between the mirrors, |y| < aperture) within one 1/e² radius w(x) of the ray pattern of
    one round (the N chords until the exit)."""
    m = cell.meta
    mode = mode_profile(m["R"], m["d"], wavelength, n_eff)
    h = trace_hits(cell, dtheta, "window")
    x, y = h["x"], h["y"]
    gx = np.linspace(-m["d"] / 2, m["d"] / 2, nx)
    gy = np.linspace(-m["aperture"], m["aperture"], ny)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    inside = cell.inside(X, Y)
    P = np.stack([X[inside], Y[inside]], 1)
    covered = np.zeros(len(P), bool)
    for k in range(len(x) - 1):
        ax, ay, bx, by = x[k], y[k], x[k + 1], y[k + 1]
        vx, vy = bx - ax, by - ay
        t = np.clip(((P[:, 0] - ax) * vx + (P[:, 1] - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        qx, qy = ax + t * vx, ay + t * vy
        covered |= np.hypot(P[:, 0] - qx, P[:, 1] - qy) <= beam_width(qx, mode)
    return float(covered.mean()) if len(P) else np.nan


def make_mirror(model, R_mirror, n_eff, wavelength, polarization="TM", periods=4, sin_design=0.0, beam_average=True,
                scatter=0.0, theta0=0.0, m_gap=1, m_tooth=1):
    dbr = None
    if model == "dbr":
        dbr = TrenchDBR(n_tooth=n_eff, lam_design=wavelength, N=int(periods), m_gap=int(m_gap), m_tooth=int(m_tooth),
                        bounce_loss=0.0, slab_pol=polarization, sin_design=sin_design)
    return mirror_function(model, R_mirror, dbr, wavelength, theta0, beam_average, scatter), dbr


def sweep_R(cell, R_values, wavelength=1.55e-6, n_eff=2.479, h_eff=0.3e-6, exit_mode="window", clip=True, scatter=0.0):
    """budget() for each constant mirror reflectance: arrays of T_out, L_eff, V_eff, A_eff, chi_mean, chi_rms, I_end."""
    keys = ("T_out", "L_eff", "V_eff", "A_eff", "chi_mean", "chi_rms", "I_end", "n_hits")
    out = {k: [] for k in keys}
    for R in R_values:
        b = budget(cell, wavelength, n_eff, h_eff, mirror_function("constant", R, scatter=scatter), exit_mode=exit_mode, clip=clip)
        for k in keys:
            out[k].append(b[k])
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- Result front end
def _cell(R, N, M, A, phase, port_width, dR2, tilt2, spacing_error):
    require_positive(R=R, A=A, port_width=port_width)
    c0 = herriott_planar_cell(R, int(N), int(M), A, port_width, phase=phase)
    c = herriott_planar_cell(R, int(N), int(M), A, port_width, phase=phase, R2=R + dR2, d=c0.meta["d"] + spacing_error)
    c.meta["theta_launch"] = c0.meta["theta_launch"]
    c.meta["d"] = c0.meta["d"]                         # the mode is the design one (mode-matched input)
    return perturb(c, tilts=[0.0, tilt2]) if tilt2 else c


def path_budget(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, port_width=100e-6, wavelength=1.55e-6, n_eff=2.479, h_eff=0.3e-6,
                mirror="dbr", R_mirror=0.999, polarization="TM", periods=4, sin_design=0.0, beam_average=True,
                scatter=0.0, clip=True, exit_mode="window", dR2=0.0, tilt2=0.0, spacing_error=0.0, dtheta=0.0,
                loss_db_cm=0.0, beam_model="propagated", w_ratio=1.0, focus_shift=0.0) -> Result:
    """Effective path, mode volume, angles of incidence and throughput of an on-chip Herriott cell with constant or
    trench-DBR mirrors, Gaussian clipping at the mirror ends and the window."""
    require_positive(wavelength=wavelength, n_eff=n_eff, h_eff=h_eff)
    require_choice("mirror", mirror, ("constant", "dbr"))
    require_choice("exit_mode", exit_mode, ("window", "closed"))
    require_range("R_mirror", R_mirror, 0.0, 1.0)
    require_range("scatter", scatter, 0.0, 1.0)
    require_nonnegative(periods=periods, loss_db_cm=loss_db_cm)
    require_choice("beam_model", beam_model, ("propagated", "design"))
    require_positive(w_ratio=w_ratio)
    cell = _cell(R, N, M, A, phase, port_width, dR2, tilt2, spacing_error)
    beam = dict(w_ratio=w_ratio, focus_shift=focus_shift) if beam_model == "propagated" else None
    mode = mode_profile(cell.meta["R"], cell.meta["d"], wavelength, n_eff)
    Rf, dbr = make_mirror(mirror, R_mirror, n_eff, wavelength, polarization, periods, sin_design, beam_average, scatter, mode["theta0"])
    b = budget(cell, wavelength, n_eff, h_eff, Rf, dtheta, exit_mode, bool(clip), alpha=DB_PER_CM * loss_db_cm, beam=beam)
    st = b["stability"] or {}
    wh = b["w_hit"][:b["reflections"]] if b["reflections"] else np.array([np.nan])
    vals = {"m_roundtrip": st.get("m_roundtrip", np.nan), "coupling": st.get("coupling", np.nan),
            "w_hit_min": float(np.min(wh)), "w_hit_max": float(np.max(wh)),"T_out": b["T_out"], "eta_window": b["eta_window"], "L_eff": b["L_eff"], "L_geom": b["L_geom"], "V_eff": b["V_eff"],
            "A_eff": b["A_eff"], "w_eff_mean": b["w_eff_mean"], "chi_mean_deg": math.degrees(b["chi_mean"]),
            "chi_rms_deg": math.degrees(b["chi_rms"]), "chi_max_deg": math.degrees(b["chi_max"]), "R_mean": b["R_mean"],
            "clip_loss": b["clip_loss"], "reflections": b["reflections"], "exits": b["exits"], "w0": mode["w0"],
            "w_mirror": mode["w_mirror"], "theta0_deg": math.degrees(mode["theta0"]), "d": cell.meta["d"],
            "R_dbr_normal": float(dbr.R(wavelength, 0.0)) if dbr else R_mirror}
    units = {k: "" for k in vals}
    units.update(w_hit_min="m", w_hit_max="m", L_eff="m", L_geom="m", V_eff="m^3", A_eff="m^2", w_eff_mean="m", chi_mean_deg="deg", chi_rms_deg="deg",
                 chi_max_deg="deg", w0="m", w_mirror="m", theta0_deg="deg", d="m")
    return Result(values=vals, units=units, assumptions=[
        "propagated: Gaussian q along the traced passes with in-plane mirror power 2/(R cos χ); input = first round-trip eigenmode × (w_ratio, focus_shift); design: paraxial cell mode on every pass",
        "Waveguide loss loss_db_cm on every chord (path and volume weighted by e^{-αs})",
        "w_eff = w √(π/2) in plane; h_eff = slab effective height (input)",
        "Clipping: 1D Gaussian share on the mirror (|y| < aperture) and, on M1, off the window",
        "DBR: bragg_grating.TrenchDBR, first order, exact R(χ) per hit, optionally averaged over the beam's angular spectrum",
        "exit_mode window: re-entrant design, output through the window at hit N; closed: circulates until 1e-6 of the input"])


# ---------------------------------------------------------------- DBR at another wavelength / mode index
class DBRAt:
    """The trench DBR designed at the signal (its gap and tooth thicknesses) seen by a mode of index n_eff at another
    wavelength (e.g. the pump): exact transfer matrix, R(λ, |sin χ|)."""

    def __init__(self, dbr: TrenchDBR, n_eff):
        self.n, self.dbr = float(n_eff), dbr
        self.layers = [(dbr.n_gap, dbr.d_gap), (self.n, dbr.d_tooth)] * dbr.N
        self.pol = {"TE": "p", "TM": "s"}[dbr.slab_pol]

    def R(self, wavelength, sin_chi):
        from ..bragg_grating.engine import stack_R_oblique
        s = np.atleast_1d(np.asarray(sin_chi, float))
        r = np.array([stack_R_oblique(wavelength, v, self.n, self.layers, self.dbr.n_gap, self.pol) for v in s]) * (1 - self.dbr.bounce_loss)
        return r if np.ndim(sin_chi) else float(r[0])


# ---------------------------------------------------------------- erbium amplifier along the folded path
from . import erbium as _er  # noqa: E402


def amplifier(cell, sig, pump, er, Ps_in=1e-6, Pp_in=0.1, dtheta=0.0, clip=True, n_sub=8, n_max=4000):
    """Signal and co-propagating pump through the cell (in through the window, out after N passes), with Er gain in a
    thin layer. sig / pump: dict(wavelength, n_eff, gamma_er (power share in the Er layer), alpha [1/m], R_of_chi,
    beam (None or dict)); er: dict(N (m⁻³), t (layer thickness, m), tau, Cup, fq (quenched share), sigma_peak, sig980).
    Each pass is integrated with RK4 (n_sub steps); the Er-layer intensity is P Γ_er / (w √(π/2) t), uniform across the
    layer and over the beam's effective width (top-hat). Returns totals and the power and inversion along the path."""
    m = cell.meta
    hits = trace_hits(cell, dtheta, "window", n_max)
    y, K, chi = hits["y"], hits["element"], hits["chi"]
    n = len(K)
    last = n - 1 if hits["exit"] else n
    a, y0, pw = m["aperture"], m["y0"], m["port_w"]
    win = (y0 - pw / 2, y0 + pw / 2)
    xs = _er.cross_sections(sig["wavelength"], pump["wavelength"], er.get("sigma_peak", 5.7e-25), er.get("sig980", 1.7e-25))
    Ntot = er["N"]
    Nq = Ntot * er.get("fq", 0.0)
    Nact = Ntot - Nq
    beams = []
    for b in (sig, pump):
        lam_m = b["wavelength"] / b["n_eff"]
        bp = beam_on_path(cell, hits, lam_m, b.get("beam"), n=2 * n_sub + 1)
        bp["eta"] = gauss_fraction(*win, y0, bp["W"][0, 0]) if n else 0.0
        beams.append(bp)
    bs, bpp = beams
    chords = bs["chords"]
    t_er = er["t"]
    gs, gp = sig["gamma_er"], pump["gamma_er"]
    k = math.sqrt(math.pi / 2)

    def rates(Ps, Pp, ws, wp):
        if t_er <= 0 or Ntot <= 0:
            return -sig["alpha"] * Ps, -pump["alpha"] * Pp, 0.0
        Is = Ps * gs / (ws * k * t_er)
        Ip = Pp * gp / (wp * k * t_er)
        N1, N2 = _er.populations(Ip, Is, xs, sig["wavelength"], pump["wavelength"], Nact, Nq, er["tau"], er["Cup"])
        sa_s, se_s, sa_p, se_p = xs
        return ((gs * (se_s * N2 - sa_s * N1) - sig["alpha"]) * Ps, -(gp * (sa_p * N1 - se_p * N2) + pump["alpha"]) * Pp,
                N2 / Ntot)

    Ps, Pp = Ps_in * bs["eta"], Pp_in * bpp["eta"]
    S, PS, PP, INV = [0.0], [Ps], [Pp], [rates(Ps, Pp, bs["W"][0, 0], bpp["W"][0, 0])[2] if n else 0.0]
    s_acc = 0.0
    for j in range(n):
        L = chords[j]
        h = L / n_sub
        for i in range(n_sub):
            w0s, w1s, w2s = bs["W"][j, 2 * i], bs["W"][j, 2 * i + 1], bs["W"][j, 2 * i + 2]
            w0p, w1p, w2p = bpp["W"][j, 2 * i], bpp["W"][j, 2 * i + 1], bpp["W"][j, 2 * i + 2]
            k1 = rates(Ps, Pp, w0s, w0p)
            k2 = rates(Ps + 0.5 * h * k1[0], Pp + 0.5 * h * k1[1], w1s, w1p)
            k3 = rates(Ps + 0.5 * h * k2[0], Pp + 0.5 * h * k2[1], w1s, w1p)
            k4 = rates(Ps + h * k3[0], Pp + h * k3[1], w2s, w2p)
            Ps = max(Ps + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0.0)
            Pp = max(Pp + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0)
            s_acc += h
            S.append(s_acc); PS.append(Ps); PP.append(Pp); INV.append(rates(Ps, Pp, w2s, w2p)[2])
        if j < last:
            cs = cp = 1.0
            if clip:
                for which, wv in ((0, bs["W"][j, -1]), (1, bpp["W"][j, -1])):
                    on = gauss_fraction(-a, a, y[j + 1], wv)
                    off = gauss_fraction(*win, y[j + 1], wv) if K[j] == 0 else 0.0
                    if which == 0:
                        cs = max(on - off, 0.0)
                    else:
                        cp = max(on - off, 0.0)
            Ps *= sig["R_of_chi"](abs(chi[j]), bs["theta0"][j]) * cs
            Pp *= pump["R_of_chi"](abs(chi[j]), bpp["theta0"][j]) * cp
    out_s = Ps * bs["eta"] if hits["exit"] else 0.0
    return dict(gain_db=10 * math.log10(out_s / Ps_in) if out_s > 0 else -math.inf,
                internal_gain_db=10 * math.log10(Ps / (Ps_in * bs["eta"])) if Ps > 0 else -math.inf,
                Ps_out=out_s, Pp_out=Pp * bpp["eta"] if hits["exit"] else 0.0, pump_left=Pp / (Pp_in * bpp["eta"]) if Pp_in else 0.0,
                eta_s=bs["eta"], eta_p=bpp["eta"], path=float(chords[:n].sum()), exits=bool(hits["exit"]), n_hits=n,
                s=np.array(S), Ps=np.array(PS), Pp=np.array(PP), inversion=np.array(INV), cross_sections=xs)


def scaled_cell(R, N, M, A, phase, port_width, scale):
    """Geometrically similar cell: R, A and the window width multiplied by scale."""
    return herriott_planar_cell(R * scale, int(N), int(M), A * scale, port_width * scale, phase=phase)


def sweep_size(R, N, M, A, phase, port_width, scales, loss_db_cm, wavelength=1.55e-6, n_eff=2.479, h_eff=0.3e-6,
               R_mirror=0.999, clip=True):
    """L_eff and throughput of geometrically similar cells (R, A, window × scale) for each waveguide loss [dB/cm]:
    dict(scales, loss, L_eff (losses × scales), T_out, L_geom (scales))."""
    Lg, Le, T = [], [], []
    for s in scales:
        c = scaled_cell(R, N, M, A, phase, port_width, s)
        row_L, row_T = [], []
        for ldb in loss_db_cm:
            b = budget(c, wavelength, n_eff, h_eff, lambda chi, t=0.0: R_mirror, clip=clip, alpha=DB_PER_CM * ldb)
            row_L.append(b["L_eff"]); row_T.append(b["T_out"])
        Lg.append(b["L_geom"]); Le.append(row_L); T.append(row_T)
    return dict(scales=np.asarray(scales, float), loss=np.asarray(loss_db_cm, float), L_geom=np.array(Lg),
                L_eff=np.array(Le).T, T_out=np.array(T).T)


# ---------------------------------------------------------------- Result front ends (stack, stability, amplifier)
from . import stack as _st  # noqa: E402


def slab_stack(platform="al2o3", film_t=0.4e-6, er_t=0.4e-6, substrate="sio2", cladding="air", wavelength=1.532e-6,
               polarization="TE") -> Result:
    """Fundamental slab mode of substrate | film | Er:Al2O3 layer | cladding: n_eff, effective height and the power
    share in every layer."""
    require_positive(wavelength=wavelength, film_t=film_t)
    require_nonnegative(er_t=er_t)
    require_choice("platform", platform, tuple(_st.PLATFORMS))
    require_choice("substrate", substrate, ("sio2", "air"))
    require_choice("cladding", cladding, ("air", "sio2"))
    require_choice("polarization", polarization, ("TE", "TM"))
    m = _st.stack_mode(platform, film_t, er_t, substrate, cladding, wavelength, polarization)
    v = {"neff": m["neff"], "h_eff": m["h_eff"], "gamma_substrate": m["gamma_substrate"], "gamma_film": m["gamma_film"],
         "gamma_er": m["gamma_er"], "gamma_cladding": m["gamma_cladding"]}
    return Result(values=v, units={"neff": "", "h_eff": "m", "gamma_substrate": "", "gamma_film": "", "gamma_er": "", "gamma_cladding": ""},
                  assumptions=["Planar slab, fundamental TE or TM mode, lossless isotropic layers (film axis per polarisation)",
                               "Γ: share of the power profile E_y² (TE) or H_y²/n² (TM) in each layer", "h_eff = ∫S dz / max S"])


def beam_stability(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, port_width=100e-6, wavelength=1.55e-6, n_eff=2.479,
                   w_ratio=1.0, focus_shift=0.0, dR2=0.0, spacing_error=0.0) -> Result:
    """Gaussian beam along the traced passes: round-trip stability and Gouy phase, eigenmode, coupling of the actual input
    into it, and the breathing of the beam radius on the mirrors over one pattern."""
    require_positive(wavelength=wavelength, n_eff=n_eff, w_ratio=w_ratio)
    cell = _cell(R, N, M, A, phase, port_width, dR2, 0.0, spacing_error)
    hits = trace_hits(cell, 0.0, "window")
    bp = beam_on_path(cell, hits, wavelength / n_eff, dict(w_ratio=w_ratio, focus_shift=focus_shift))
    st = bp["stability"]
    wh = bp["W"][:-1, -1] if len(bp["W"]) > 1 else bp["W"][:, -1]
    d0 = cell.meta["d"]
    v = {"m_roundtrip": st["m_roundtrip"], "stable": st["stable"], "gouy_roundtrip": st["gouy_roundtrip"],
         "gouy_design": 2 * cell.meta["theta"], "m_path": st["m_path"], "w0_eigen": st["w0_eigen"], "coupling": st["coupling"],
         "w_mirror_min": float(wh.min()), "w_mirror_max": float(wh.max()), "breathing": float(wh.max() / wh.min()),
         "w_mirror_design": mode_profile(R, d0, wavelength, n_eff)["w_mirror"]}
    u = {k: "" for k in v}
    u.update(gouy_roundtrip="rad", gouy_design="rad", w0_eigen="m", w_mirror_min="m", w_mirror_max="m", w_mirror_design="m")
    return Result(values=v, units=u, assumptions=[
        "Paraxial Gaussian q along the exact traced chords; in-plane (tangential) mirror power 2/(R cos χ)",
        "Eigenmode of the first round trip (passes 1–2) at the launch point; stable when |m| = |A + D|/2 < 1",
        "Input: waist w_ratio × eigen waist, moved focus_shift along the first pass; coupling = 1D Gaussian overlap"])


def er_amplifier(R=10e-3, N=20, M=3, A=1e-3, phase=0.0, port_width=100e-6, platform="al2o3", film_t=0.4e-6, er_t=0.4e-6,
                 substrate="sio2", cladding="air", polarization="TE", lam_s=1.532e-6, lam_p=980e-9, P_signal=1e-6,
                 P_pump=0.1, N_er=1.5e26, tau=7.5e-3, Cup=4e-24, fq=0.0, loss_db_cm=0.25, periods=6, R_pump=0.99,
                 pump_mirror="constant", w_ratio=1.0, focus_shift=0.0, scale=1.0) -> Result:
    """Er:Al2O3 gain in an on-chip Herriott cell: signal and co-propagating pump enter through the window, travel the N
    passes in the slab (stack modes at each wavelength), are amplified/absorbed in the Er layer and reflected by a trench
    DBR (signal) and a constant mirror or the same DBR (pump); scale multiplies R, A and the window."""
    require_positive(P_signal=P_signal, N_er=N_er, tau=tau, scale=scale)
    require_nonnegative(P_pump=P_pump, Cup=Cup, er_t=er_t, loss_db_cm=loss_db_cm)
    require_range("fq", fq, 0.0, 0.95)
    require_choice("pump_mirror", pump_mirror, ("constant", "dbr"))
    ms = _st.stack_mode(platform, film_t, er_t, substrate, cladding, lam_s, polarization)
    mp = _st.stack_mode(platform, film_t, er_t, substrate, cladding, lam_p, polarization)
    if not (math.isfinite(ms["neff"]) and math.isfinite(mp["neff"])):
        raise ValueError("the stack guides no mode at the signal or the pump wavelength")
    cell = scaled_cell(R, N, M, A, phase, port_width, scale)
    dbr = TrenchDBR(n_tooth=ms["neff"], lam_design=lam_s, N=int(periods), m_gap=1, m_tooth=1, bounce_loss=0.0, slab_pol=polarization)
    th_s = mode_profile(cell.meta["R"], cell.meta["d"], lam_s, ms["neff"])["theta0"]
    Rs = mirror_function("dbr", dbr=dbr, wavelength=lam_s, theta0=th_s, beam_average=True)
    Rp = (mirror_function("dbr", dbr=DBRAt(dbr, mp["neff"]), wavelength=lam_p) if pump_mirror == "dbr"
          else mirror_function("constant", R_pump))
    beam = dict(w_ratio=w_ratio, focus_shift=focus_shift)
    a = DB_PER_CM * loss_db_cm
    sig = dict(wavelength=lam_s, n_eff=ms["neff"], gamma_er=ms["gamma_er"], alpha=a, R_of_chi=Rs, beam=beam)
    pmp = dict(wavelength=lam_p, n_eff=mp["neff"], gamma_er=mp["gamma_er"], alpha=a, R_of_chi=Rp, beam=beam)
    r = amplifier(cell, sig, pmp, dict(N=N_er, t=er_t, tau=tau, Cup=Cup, fq=fq), P_signal, P_pump)
    xs = r["cross_sections"]
    v = {"gain_db": r["gain_db"], "internal_gain_db": r["internal_gain_db"], "Ps_out": r["Ps_out"], "pump_left": r["pump_left"],
         "path": r["path"], "gain_per_cm": r["internal_gain_db"] / (100 * r["path"]) if r["path"] else np.nan,
         "inversion_mean": float(np.mean(r["inversion"])), "neff_s": ms["neff"], "neff_p": mp["neff"], "gamma_er_s": ms["gamma_er"],
         "gamma_er_p": mp["gamma_er"], "R_dbr_signal": float(dbr.R(lam_s, 0.0)), "R_dbr_pump": float(DBRAt(dbr, mp["neff"]).R(lam_p, 0.0)),
         "small_signal_absorption_db_cm": 10 * math.log10(math.e) * ms["gamma_er"] * xs[0] * N_er / 100,
         "exits": r["exits"], "footprint_area": (cell.meta["d"] + 2 * cell.meta["aperture"] ** 2 / cell.meta["R"]) * 2 * cell.meta["aperture"]}
    u = {k: "" for k in v}
    u.update(gain_db="dB", internal_gain_db="dB", Ps_out="W", path="m", gain_per_cm="dB/cm", small_signal_absorption_db_cm="dB/cm",
             footprint_area="m^2")
    return Result(values=v, units=u, assumptions=[
        "Effective two-level Er³⁺ with upconversion and quenched ions (er-waveguide-amplifier model), no ASE",
        "Er-layer intensity = P Γ_er/(w√(π/2) t_er): uniform across the thin layer and the beam's effective width",
        "Signal mirror: first-order trench DBR at λs (beam-averaged); pump mirror constant R_pump or the same DBR at λp",
        "Mode-matched or mismatched input as for beam_stability; same waveguide loss at both wavelengths",
        "Crossing passes do not share inversion (no transverse overlap of the passes)"])
