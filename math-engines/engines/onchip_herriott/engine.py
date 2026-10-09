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
    """R(|χ|) for one hit: constant, or a TrenchDBR at the ray angle or averaged over the beam's angular spectrum;
    times (1 − scatter)."""
    if model == "constant":
        return lambda chi: R_mirror * (1 - scatter)
    if beam_average and theta0 > 0:
        a, wt = angle_weights(theta0)
        return lambda chi: float((wt * dbr.R(wavelength, np.abs(np.sin(abs(chi) + a)))).sum()) * (1 - scatter)
    return lambda chi: float(dbr.R(wavelength, abs(math.sin(chi)))) * (1 - scatter)


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


def budget(cell, wavelength=1.55e-6, n_eff=2.479, h_eff=0.3e-6, R_of_chi=None, dtheta=0.0, exit_mode="window",
           clip=True, n_max=4000, stop_below=1e-6):
    """Power, path, volume and angle budget of the centre ray's Gaussian beam (module docstring). R_of_chi(|χ|) is the
    mirror power reflectance per hit (default 1). Returns a dict of totals and per-hit arrays."""
    m = cell.meta
    mode = mode_profile(m["R"], m["d"], wavelength, n_eff)
    wm = mode["w_mirror"]
    hits = trace_hits(cell, dtheta, exit_mode, n_max)
    x, y, K, chi = hits["x"], hits["y"], hits["element"], hits["chi"]
    n = len(K)
    a, y0, pw = m["aperture"], m["y0"], m["port_w"]
    win = (y0 - pw / 2, y0 + pw / 2)
    eta = gauss_fraction(*win, y0, wm) if exit_mode == "window" else 1.0
    Rf = R_of_chi or (lambda c: 1.0)
    chords = np.hypot(np.diff(x), np.diff(y))
    wint = chord_width_integral(x[:-1], y[:-1], x[1:], y[1:], mode) * math.sqrt(math.pi / 2)
    I = np.empty(n + 1)
    I[0] = eta
    Rj, cj = np.ones(n), np.ones(n)
    last = n - 1 if hits["exit"] else n               # the exit hit is not a reflection
    for j in range(n):
        if j < last:
            Rj[j] = Rf(abs(chi[j]))
            if clip:
                on = gauss_fraction(-a, a, y[j + 1], wm)
                off_win = gauss_fraction(*win, y[j + 1], wm) if (exit_mode == "window" and K[j] == 0) else 0.0
                cj[j] = max(on - off_win, 0.0)
        I[j + 1] = I[j] * (Rj[j] * cj[j] if j < last else 1.0)
        if exit_mode == "closed" and I[j + 1] < stop_below * I[0]:
            n = last = j + 1
            break
    I, Rj, cj, chords, wint, chi, K = I[:n + 1], Rj[:n], cj[:n], chords[:n], wint[:n], chi[:n], K[:n]
    w_in = I[:-1]                                      # power on chord j (arriving at hit j)
    L_eff = float((w_in * chords).sum())
    V_eff = float(h_eff * (w_in * wint).sum())
    refl = slice(0, last)
    wr = I[:last]
    achi = np.abs(chi[refl])
    T_out = float(I[n - 1] * eta) if hits["exit"] else 0.0
    return dict(mode=mode, n_hits=n, reflections=last, exits=bool(hits["exit"]), eta_window=eta, T_out=T_out,
                I_end=float(I[last]), L_geom=float(chords.sum()), L_eff=L_eff, V_eff=V_eff, A_eff=V_eff / L_eff if L_eff else np.nan,
                w_eff_mean=V_eff / L_eff / h_eff if L_eff else np.nan, V_geom=float(h_eff * wint.sum()),
                chi_mean=float((wr * achi).sum() / wr.sum()) if last else np.nan,
                chi_rms=float(np.sqrt((wr * achi**2).sum() / wr.sum())) if last else np.nan,
                chi_max=float(achi.max()) if last else np.nan, R_mean=float((wr * Rj[refl]).sum() / wr.sum()) if last else np.nan,
                clip_loss=float(1 - np.prod(cj[refl])), I=I, R=Rj, c=cj, chords=chords, w_int=wint, chi=chi, element=K,
                x=hits["x"][:n + 1], y=hits["y"][:n + 1])


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
                scatter=0.0, clip=True, exit_mode="window", dR2=0.0, tilt2=0.0, spacing_error=0.0, dtheta=0.0) -> Result:
    """Effective path, mode volume, angles of incidence and throughput of an on-chip Herriott cell with constant or
    trench-DBR mirrors, Gaussian clipping at the mirror ends and the window."""
    require_positive(wavelength=wavelength, n_eff=n_eff, h_eff=h_eff)
    require_choice("mirror", mirror, ("constant", "dbr"))
    require_choice("exit_mode", exit_mode, ("window", "closed"))
    require_range("R_mirror", R_mirror, 0.0, 1.0)
    require_range("scatter", scatter, 0.0, 1.0)
    require_nonnegative(periods=periods)
    cell = _cell(R, N, M, A, phase, port_width, dR2, tilt2, spacing_error)
    mode = mode_profile(cell.meta["R"], cell.meta["d"], wavelength, n_eff)
    Rf, dbr = make_mirror(mirror, R_mirror, n_eff, wavelength, polarization, periods, sin_design, beam_average, scatter, mode["theta0"])
    b = budget(cell, wavelength, n_eff, h_eff, Rf, dtheta, exit_mode, bool(clip))
    vals = {"T_out": b["T_out"], "eta_window": b["eta_window"], "L_eff": b["L_eff"], "L_geom": b["L_geom"], "V_eff": b["V_eff"],
            "A_eff": b["A_eff"], "w_eff_mean": b["w_eff_mean"], "chi_mean_deg": math.degrees(b["chi_mean"]),
            "chi_rms_deg": math.degrees(b["chi_rms"]), "chi_max_deg": math.degrees(b["chi_max"]), "R_mean": b["R_mean"],
            "clip_loss": b["clip_loss"], "reflections": b["reflections"], "exits": b["exits"], "w0": mode["w0"],
            "w_mirror": mode["w_mirror"], "theta0_deg": math.degrees(mode["theta0"]), "d": cell.meta["d"],
            "R_dbr_normal": float(dbr.R(wavelength, 0.0)) if dbr else R_mirror}
    units = {k: "" for k in vals}
    units.update(L_eff="m", L_geom="m", V_eff="m^3", A_eff="m^2", w_eff_mean="m", chi_mean_deg="deg", chi_rms_deg="deg",
                 chi_max_deg="deg", w0="m", w_mirror="m", theta0_deg="deg", d="m")
    return Result(values=vals, units=units, assumptions=[
        "Mode-matched Gaussian beam with the symmetric cell's eigenmode width on every pass (paraxial)",
        "w_eff = w √(π/2) in plane; h_eff = slab effective height (input)",
        "Clipping: 1D Gaussian share on the mirror (|y| < aperture) and, on M1, off the window",
        "DBR: bragg_grating.TrenchDBR, first order, exact R(χ) per hit, optionally averaged over the beam's angular spectrum",
        "exit_mode window: re-entrant design, output through the window at hit N; closed: circulates until 1e-6 of the input"])
