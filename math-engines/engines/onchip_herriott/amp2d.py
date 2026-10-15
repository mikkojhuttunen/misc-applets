"""Coupled Er amplifier of the on-chip Herriott cell: shared inversion where passes overlap, and ASE.

Every pass j of the folded path carries signal, pump and amplified spontaneous emission (ASE, forward and backward, in
spectral bins) as Gaussian beams of their own 1/e² radius w(s) around the traced ray. The Er:Al2O3 layer is resolved
in depth (NZ sublayers, slab power profile S(z) at each wavelength). At a point X of the layer, at depth z, the
photon flux of beam type b is the incoherent sum over ALL passes k:

    φ_b(X, z) = Σ_k P_b,k(t_k) · √(2/π)/w_k exp(−2 d_k²/w_k²) · S_b(z)/h_eff,b / hν_b

(t_k, d_k: position along and distance from pass k; crossing passes add, interference fringes of crossing beams are
averaged). N1, N2 follow the steady-state two-level rate equation of erbium.py from the total pump + signal + ASE
fluxes. The modal coefficient of a beam at a point of its own path is its overlap with the inversion:

    g_b(λ) = Σ_q ω_q Σ_m Γ_m,b [σe(λ) N2 − σa(λ) N1](X_q, z_m)     (ω_q: lateral Gauss–Hermite weights of its own
                                                                    profile, Γ_m,b: power share of sublayer m)
and the ASE source per bin is Σ ω Γ σe(λk) N2 · m hνk Δνk (m = number of guided polarisation/transverse modes
counted). Signal and pump propagate forward from the window, forward ASE from zero at the window, backward ASE from
zero at the exit back to the window; every mirror hit multiplies by R(λ, χ) and the Gaussian clipping share. The
inversion and the fields are iterated (under-relaxed) to self-consistency. coupled=False keeps only each pass's own
beams (the independent-pass model, for comparison). SI units.
"""
from __future__ import annotations

import math

import numpy as np

from . import erbium as _er

GH_X = np.array([-2.6519613568352334, -1.6735516287674714, -0.8162878828589646, 0.0, 0.8162878828589646, 1.6735516287674714,
                 2.6519613568352334])
GH_W = np.array([0.0005482688559722182, 0.03075712396758651, 0.24012317860501273, 0.45714285714285713, 0.24012317860501273,
                 0.03075712396758651, 0.0005482688559722182])
H, C = _er.H, _er.C


def _gauss_frac(lo, hi, y, w):
    s = math.sqrt(2) / w
    return 0.5 * (math.erf(s * (hi - y)) - math.erf(s * (lo - y)))


def vertical(mode, z0, z1, nz):
    """Sublayer midpoints of the Er layer, power share of each (Σ = Γ_er) and the vertical density S(z)/∫S [1/m]."""
    dz = (z1 - z0) / nz
    zm = z0 + (np.arange(nz) + 0.5) * dz
    Sm = np.interp(zm, mode["z"], mode["S"])
    dens = Sm / mode["h_eff"]
    return zm, dens * dz, dens


def _points(x, y, W, nq=7):
    """Lateral quadrature points of every sample of every pass: array (n, ns, nq, 2)."""
    n, ns = W.shape
    f = np.linspace(0.0, 1.0, ns)
    dx, dy = np.diff(x), np.diff(y)
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    px = x[:-1, None] + f[None, :] * dx[:, None]
    py = y[:-1, None] + f[None, :] * dy[:, None]
    u = W[:, :, None] * GH_X[None, None, :] / math.sqrt(2)
    return np.stack([px[:, :, None] + u * nx[:, None, None], py[:, :, None] + u * ny[:, None, None]], -1)


def kernel(P, x, y, W, own=None, cut=4.0):
    """Sparse lateral kernel from points P (m, 2) to the beams of all passes (widths W (n, ns)): entries (point, column
    i0 of pass k flattened, frac, density). own: array (m,) of the pass each point belongs to → keep only that pass."""
    n, ns = W.shape
    rows, cols, fr, dens = [], [], [], []
    m = len(P)
    idx = np.arange(m)
    for k in range(n):
        ax, ay, bx, by = x[k], y[k], x[k + 1], y[k + 1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = np.clip(((P[:, 0] - ax) * vx + (P[:, 1] - ay) * vy) / L2, 0.0, 1.0)
        d = np.hypot(P[:, 0] - (ax + t * vx), P[:, 1] - (ay + t * vy))
        u = t * (ns - 1)
        i0 = np.minimum(np.floor(u).astype(int), ns - 2)
        a = u - i0
        w = W[k, i0] * (1 - a) + W[k, i0 + 1] * a
        keep = d < cut * w
        if own is not None:
            keep &= own == k
        if not keep.any():
            continue
        rows.append(idx[keep]); cols.append(k * ns + i0[keep]); fr.append(a[keep])
        dens.append(math.sqrt(2 / math.pi) / w[keep] * np.exp(-2 * (d[keep] / w[keep]) ** 2))
    r, c, a, de = (np.concatenate(v) for v in (rows, cols, fr, dens))
    o = np.argsort(r, kind="stable")
    r, c, a, de = r[o], c[o], a[o], de[o]
    starts = np.r_[0, np.nonzero(np.diff(r))[0] + 1]
    return dict(rows=r, cols=c, frac=a, dens=de, starts=starts, uniq=r[starts], m=m)


def apply_kernel(K, F):
    """Lateral line density Σ P λ at every point for fields F (..., n*ns): array (..., m)."""
    v = (F[..., K["cols"]] * (1 - K["frac"]) + F[..., K["cols"] + 1] * K["frac"]) * K["dens"]
    out = np.zeros(F.shape[:-1] + (K["m"],))
    out[..., K["uniq"]] = np.add.reduceat(v, K["starts"], axis=-1)
    return out


def _propagate(G, S, alpha, L, P0, refl, out_last, reverse=False):
    """Fields along the passes for coefficient G (…, n, ns), source S (same shape or 0), loss alpha, chord lengths L,
    start value P0 (…), reflection factors refl (…, n) at the end of each pass (the last pass: none). Forward: from the
    start of pass 0; reverse: from the end of the last pass back to the start of pass 0."""
    G = np.asarray(G, float)
    shape = G.shape
    n, ns = shape[-2], shape[-1]
    F = np.zeros(shape)
    P = np.array(P0, float) * np.ones(shape[:-2])
    order = range(n - 1, -1, -1) if reverse else range(n)
    for j in order:
        h = L[j] / (ns - 1)
        rng = range(ns - 1, 0, -1) if reverse else range(ns - 1)
        i_first = ns - 1 if reverse else 0
        F[..., j, i_first] = P
        for i in rng:
            i2 = i - 1 if reverse else i + 1
            g = 0.5 * (G[..., j, i] + G[..., j, i2]) - alpha
            s = 0.5 * (S[..., j, i] + S[..., j, i2]) if np.ndim(S) else 0.0
            e = np.exp(g * h)
            with np.errstate(divide="ignore", invalid="ignore"):
                srcterm = np.where(np.abs(g) > 1e-12, s * (e - 1) / np.where(np.abs(g) > 1e-12, g, 1.0), s * h)
            P = P * e + srcterm
            F[..., j, i2] = P
        if reverse:
            if j > 0:
                P = P * refl[..., j - 1]
        elif j < n - 1 or not out_last:
            P = P * refl[..., j]
    return F, P


def solve(cell, hits, sig, pump, er, Ps_in, Pp_in, bs, bp, ase=None, coupled=True, nz=4, max_iter=300, tol=1e-6,
          relax=0.5):
    """Self-consistent coupled amplifier. sig / pump: dict(wavelength, n_eff, mode (stack_mode dict), z_er (z0, z1),
    alpha, R (per-hit reflectance array incl. clipping)); bs / bp: beam_on_path outputs (widths W (n, ns)) for the
    signal (also used by the ASE) and the pump; er: dict(N, tau, Cup, fq, sigma_peak, sig980); ase: None or
    dict(lams (bin centres), dlam (bin widths), R (bins × n per-hit reflectance incl. clipping), modes). Returns totals,
    spectra, the inversion along the path and the convergence record."""
    x, y = hits["x"], hits["y"]
    Ws, Wp = bs["W"], bp["W"]
    n, ns = Ws.shape
    L = bs["chords"][:n]
    Ntot = er["N"]
    Nq = Ntot * er.get("fq", 0.0)
    N = Ntot - Nq
    tau, Cup = er["tau"], er["Cup"]
    ls, lp = sig["wavelength"], pump["wavelength"]
    pk, s980 = er.get("sigma_peak", 5.7e-25), er.get("sig980", 1.7e-25)
    sa_s, se_s, sa_p, se_p = _er.cross_sections(ls, lp, pk, s980)
    hvs, hvp = H * C / ls, H * C / lp
    z0, z1 = sig["z_er"]
    _, sh_s, de_s = vertical(sig["mode"], z0, z1, nz)
    _, sh_p, de_p = vertical(pump["mode"], z0, z1, nz)
    Pts = _points(x, y, Ws).reshape(-1, 2)
    Ptp = _points(x, y, Wp).reshape(-1, 2)
    own = np.repeat(np.arange(n), ns * len(GH_X))
    o = None if coupled else own
    K_ss, K_sp = kernel(Pts, x, y, Ws, o), kernel(Pts, x, y, Wp, o)
    K_ps, K_pp = kernel(Ptp, x, y, Ws, o), kernel(Ptp, x, y, Wp, o)
    if ase:
        lk = np.asarray(ase["lams"], float)
        dnu = C / lk**2 * np.asarray(ase["dlam"], float)
        sa_k, se_k = _er.sigma_a_arr(lk, pk), _er.sigma_e_arr(lk, pk)
        hvk = H * C / lk
        Rk = np.asarray(ase["R"], float)
        msp = ase.get("modes", 2)
        nk = len(lk)
    else:
        nk = 0
    Rs, Rp = np.asarray(sig["R"], float), np.asarray(pump["R"], float)
    eta_s, eta_p = sig["eta"], pump["eta"]
    # initial fields: no Er (loss and mirrors only)
    Fs, _ = _propagate(np.zeros((n, ns)), 0.0, sig["alpha"], L, Ps_in * eta_s, Rs, True)
    Fp, _ = _propagate(np.zeros((n, ns)), 0.0, pump["alpha"], L, Pp_in * eta_p, Rp, True)
    Af = np.zeros((nk, n, ns)); Ab = np.zeros((nk, n, ns))
    A_s = A_p = None
    hist = []
    for it in range(max_iter):
        # fluxes at the signal-type and pump-type points (lateral line density × vertical density / hν)
        ls_s, lp_s = apply_kernel(K_ss, Fs.reshape(-1)), apply_kernel(K_sp, Fp.reshape(-1))
        ls_p, lp_p = apply_kernel(K_ps, Fs.reshape(-1)), apply_kernel(K_pp, Fp.reshape(-1))
        if nk:
            Atot = (Af + Ab).reshape(nk, -1)
            la_s, la_p = apply_kernel(K_ss, Atot), apply_kernel(K_ps, Atot)
        A_new = []
        for (lsig, lpu, lase) in ((ls_s, lp_s, la_s if nk else None), (ls_p, lp_p, la_p if nk else None)):
            fs = lsig[:, None] * de_s[None, :] / hvs                       # (points, nz)
            fp = lpu[:, None] * de_p[None, :] / hvp
            Ra = sa_p * fp + sa_s * fs
            Re = se_p * fp + se_s * fs
            if nk:
                fa = lase[:, :, None] * de_s[None, None, :] / hvk[:, None, None]     # (bins, points, nz)
                Ra = Ra + (sa_k[:, None, None] * fa).sum(0)
                Re = Re + (se_k[:, None, None] * fa).sum(0)
            b = Ra + Re + 1 / tau
            with np.errstate(invalid="ignore", divide="ignore"):
                N2 = np.where(Ra > 0, 2 * Ra * N / (b + np.sqrt(b * b + 4 * Cup * Ra * N)), 0.0)
            N1 = N - N2 + Nq
            A_new.append((N1, N2))
        # overlap with each beam's own profile and depth: A1, A2 per sample
        (N1s, N2s), (N1p, N2p) = A_new
        q = len(GH_X)
        a1s = (N1s.reshape(n, ns, q, nz) * GH_W[None, None, :, None] * sh_s[None, None, None, :]).sum((2, 3))
        a2s = (N2s.reshape(n, ns, q, nz) * GH_W[None, None, :, None] * sh_s[None, None, None, :]).sum((2, 3))
        a1p = (N1p.reshape(n, ns, q, nz) * GH_W[None, None, :, None] * sh_p[None, None, None, :]).sum((2, 3))
        a2p = (N2p.reshape(n, ns, q, nz) * GH_W[None, None, :, None] * sh_p[None, None, None, :]).sum((2, 3))
        if A_s is None:
            A_s, A_p = (a1s, a2s), (a1p, a2p)
        else:
            A_s = (A_s[0] + relax * (a1s - A_s[0]), A_s[1] + relax * (a2s - A_s[1]))
            A_p = (A_p[0] + relax * (a1p - A_p[0]), A_p[1] + relax * (a2p - A_p[1]))
        Gs = se_s * A_s[1] - sa_s * A_s[0]
        Gp = se_p * A_p[1] - sa_p * A_p[0]
        Fs, Ps_end = _propagate(Gs, 0.0, sig["alpha"], L, Ps_in * eta_s, Rs, True)
        Fp, Pp_end = _propagate(Gp, 0.0, pump["alpha"], L, Pp_in * eta_p, Rp, True)
        if nk:
            Gk = se_k[:, None, None] * A_s[1][None] - sa_k[:, None, None] * A_s[0][None]
            Sk = (msp * hvk * dnu)[:, None, None] * se_k[:, None, None] * A_s[1][None]
            Af, Af_end = _propagate(Gk, Sk, sig["alpha"], L, np.zeros(nk), Rk, True)
            Ab, Ab_end = _propagate(Gk, Sk, sig["alpha"], L, np.zeros(nk), Rk, True, reverse=True)
            ase_tot = float(Af_end.sum() + Ab_end.sum())
        else:
            ase_tot = 0.0
        hist.append((float(Ps_end), ase_tot, float(Pp_end)))
        if it > 2:
            (a, b2, c), (a0, b0, c0) = hist[-1], hist[-2]
            if abs(a - a0) <= tol * abs(a) and abs(b2 - b0) <= tol * max(abs(b2), 1e-30) and abs(c - c0) <= tol * max(abs(c), 1e-30):
                break
    out_s = Ps_end * eta_s if hits["exit"] else 0.0
    res = dict(gain_db=10 * math.log10(out_s / Ps_in) if out_s > 0 else -math.inf,
               internal_gain_db=10 * math.log10(Ps_end / (Ps_in * eta_s)) if Ps_end > 0 else -math.inf,
               Ps_out=float(out_s), Pp_out=float(Pp_end * eta_p), pump_left=float(Pp_end / (Pp_in * eta_p)) if Pp_in else 0.0,
               iterations=it + 1, converged=it + 1 < max_iter, Fs=Fs, Fp=Fp, A_s=A_s, A_p=A_p, chords=L,
               inversion=A_s[1] / (N + Nq) / max(sh_s.sum(), 1e-300), history=hist, kernels=(K_ss, K_sp, K_ps, K_pp),
               verticals=(sh_s, de_s, sh_p, de_p), points=(Pts, Ptp), N_act=N, Nq=Nq, cross=(sa_s, se_s, sa_p, se_p),
               hv=(hvs, hvp), tau=tau, Cup=Cup)
    if nk:
        out_f, out_b = Af_end * eta_s, Ab[:, 0, 0] * eta_s
        res.update(ase_sa=sa_k, ase_se=se_k, ase_dnu=dnu, ase_lams=lk, ase_fwd=out_f, ase_bwd=out_b, ase_fwd_total=float(out_f.sum()), ase_bwd_total=float(out_b.sum()), Af=Af, Ab=Ab)
        G = out_s / Ps_in if out_s > 0 else 0.0
        rho = float(np.interp(ls, lk, out_f / dnu)) / msp           # forward ASE density at λs in the signal's mode, W/Hz
        res["nf_db"] = 10 * math.log10(1 / G + 2 * rho / (G * hvs)) if G > 0 else math.inf     # 1/G + 2 S_ASE/(G hν)
    return res


def inversion_at(res, hits, bs, bp, P, per_depth=False):
    """Depth-averaged N2/N_total of the converged solution at arbitrary points P (m, 2) of the Er layer (per_depth: N2
    [m⁻³] at every sublayer, array (m, nz))."""
    x, y = hits["x"], hits["y"]
    sh_s, de_s, sh_p, de_p = res["verticals"]
    sa_s, se_s, sa_p, se_p = res["cross"]
    Ks, Kp = kernel(P, x, y, bs["W"]), kernel(P, x, y, bp["W"])
    hvs, hvp = res["hv"]
    fs = apply_kernel(Ks, res["Fs"].reshape(-1))[:, None] * de_s[None, :] / hvs
    fp = apply_kernel(Kp, res["Fp"].reshape(-1))[:, None] * de_p[None, :] / hvp
    Ra, Re = sa_p * fp + sa_s * fs, se_p * fp + se_s * fs
    if "Af" in res:
        lk = res["ase_lams"]
        fa = apply_kernel(Ks, (res["Af"] + res["Ab"]).reshape(len(lk), -1))[:, :, None] * de_s[None, None, :] / (H * C / lk)[:, None, None]
        Ra = Ra + (res["ase_sa"][:, None, None] * fa).sum(0)
        Re = Re + (res["ase_se"][:, None, None] * fa).sum(0)
    N, Nq, tau, Cup = res["N_act"], res["Nq"], res["tau"], res["Cup"]
    b = Ra + Re + 1 / tau
    with np.errstate(invalid="ignore", divide="ignore"):
        N2 = np.where(Ra > 0, 2 * Ra * N / (b + np.sqrt(b * b + 4 * Cup * Ra * N)), 0.0)
    return N2 if per_depth else N2.mean(1) / (N + Nq)


def photon_balance(res, cell, er_t, h=6e-6):
    """Steady state: photons the Er layer takes from all beams per second (Σ_beams ∫ −g P/hν ds) against the decay
    ∫ (N2/τ + C_up N2²) dV on a grid of pitch h over the cell. Returns (from_beams, decay)."""
    hits, bs, bp = res["hits"], res["bs"], res["bp"]
    m = cell.meta
    L = res["chords"]
    n, ns = res["Fs"].shape
    f = np.linspace(0, 1, ns)
    sa_s, se_s, sa_p, se_p = res["cross"]
    hvs, hvp = res["hv"]
    trap = lambda Y: (0.5 * (Y[..., 1:] + Y[..., :-1]) * (L[:, None] / (ns - 1))).sum((-1, -2))
    Gs = se_s * res["A_s"][1] - sa_s * res["A_s"][0]
    Gp = se_p * res["A_p"][1] - sa_p * res["A_p"][0]
    tot = trap(-Gs * res["Fs"]) / hvs + trap(-Gp * res["Fp"]) / hvp
    if "Af" in res:
        Gk = res["ase_se"][:, None, None] * res["A_s"][1][None] - res["ase_sa"][:, None, None] * res["A_s"][0][None]
        hvk = H * C / res["ase_lams"]
        tot += float((trap(-Gk * (res["Af"] + res["Ab"])) / hvk).sum())
    xg = np.arange(-m["d"] / 2, m["d"] / 2, h) + h / 2
    yg = np.arange(-m["aperture"], m["aperture"], h) + h / 2
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    ins = cell.inside(X, Y)
    Pg = np.stack([X[ins], Y[ins]], 1)
    N2 = inversion_at(res, hits, bs, bp, Pg, per_depth=True)
    nz = N2.shape[1]
    dec = ((N2 / res["tau"] + res["Cup"] * N2**2).sum(1) * (er_t / nz)).sum() * h * h
    return float(tot), float(dec)
