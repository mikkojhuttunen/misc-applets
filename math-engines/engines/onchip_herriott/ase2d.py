"""In-plane ASE field of the on-chip Herriott cell: spontaneous emission into ALL guided directions of the slab.

amp2d follows the ASE that stays in the cell's own Gaussian mode along the folded path (one lateral mode). Spontaneous
emission also goes into every other in-plane direction (all other lateral modes of the slab); that light crosses the
pumped beams elsewhere, is amplified or reabsorbed far from where it was born, may be trapped by the two concave
mirrors (the cell is a stable resonator) and leaves through the open sides, through the mirrors or the window. This
module transports that field by rays and returns its photon flux to the rate equations.

Phase space. The cell is closed by two side lines between the mirror ends (open: light crossing them is lost), giving
a boundary M1 | bottom | M2 | top with arclength s. Every straight chord of the cell starts on the boundary with an
inward direction whose tangential component is p = sin θ (θ from the inward normal). (s, p) are Birkhoff coordinates:
ds dp = dy⊥ dθ is conserved along chords and by reflection (the tangential component is kept), and one lateral slab
mode per polarisation occupies ds dp = λ/n_eff. Chord bins (Δs × Δp) carry the power P in their étendue e = Δs Δp.

Along a chord (grid cell c, length ℓ):  dP/dl = g_c P + e (S_sp + S_sc)
    g_c   = Σ_m Γ_m [σe(λ) N2 − σa(λ) N1](c, z_m) − α            (depth-weighted, as for the beams)
    S_sp  = m (n_eff/λ) hν Δν σe(λ) A2(c)                        spontaneous emission per (dy⊥ dθ) per length
    S_sc  = f_sc α Φ_ASE(c) / 2π                                 in-plane scattering of the ASE (fraction f_sc of α)
Chords that end on a mirror outside the window continue as the chord bin(s) of the reflected direction (same s, same
p, bilinear weights) times R(λ, χ = asin|p|); chords ending on a side or in the window leave. The steady state of the
start powers B = M R (T B + Q) is the sum over bounces (Neumann series, geometric tail when the bounce-to-bounce ratio
has settled). A ratio ≥ 1 means the trapped ASE grows from bounce to bounce at the present inversion: parasitic lasing;
the series is then cut at max_bounce and the self-consistent loop pulls the inversion down.

The field's lateral line density on a grid of pitch h is Φ = Σ_chords ∫P dl / A_cell [W/m], the same unit as the
beams' Σ P √(2/π)/w e^(−2d²/w²); the rate equations add Φ to the fluxes of all beams.

mode_noise back-traces the output signal mode (Gauss–Hermite samples of its Wigner function at the window, 1/e²
radius w and divergence λm/(π w)) through the cell and integrates the sources it collects: spontaneous emission
(the mode's own ASE, a check on amp2d's on-path ASE) and scattering of the in-plane field or of the on-path ASE into
the mode (the part of the off-axis ASE that can reach the signal's output mode). SI units.
"""
from __future__ import annotations

import math

import numpy as np

from ..planar_cell.engine import Cell2D, Element
from . import erbium as _er

H, C = _er.H, _er.C
GH_X = np.array([-2.6519613568352334, -1.6735516287674714, -0.8162878828589646, 0.0, 0.8162878828589646, 1.6735516287674714,
                 2.6519613568352334])
GH_W = np.array([0.0005482688559722182, 0.03075712396758651, 0.24012317860501273, 0.45714285714285713, 0.24012317860501273,
                 0.03075712396758651, 0.0005482688559722182])
MIRROR, SIDE, WINDOW = 0, 1, 2


def boundary(cell):
    """Closed boundary M1 | bottom side | M2 | top side of a two-mirror planar cell (elements 0 and 2 are the mirrors)."""
    A, B, k = cell.A, cell.B, cell.kappa
    els = [Element(tuple(A[0]), tuple(B[0]), float(k[0]), 0.0, "M1"), Element(tuple(B[0]), tuple(A[1]), 0.0, 0.0, "side-"),
           Element(tuple(A[1]), tuple(B[1]), float(k[1]), 0.0, "M2"), Element(tuple(B[1]), tuple(A[0]), 0.0, 0.0, "side+")]
    return Cell2D(els, "ASE boundary")


def _e1(u):
    """(e^u − 1)/u: Taylor series to u⁵ when every |u| < 0.05 (relative error < 2e-11), else exact."""
    if u.size and np.abs(u).max() < 0.05:
        return 1 + u * (1 / 2 + u * (1 / 6 + u * (1 / 24 + u * (1 / 120 + u / 720))))
    small = np.abs(u) < 1e-5
    us = np.where(small, 1.0, u)
    return np.where(small, 1 + u / 2 + u * u / 6, np.expm1(us) / us)


def _e2(u):
    """(e^u − 1 − u)/u²."""
    if u.size and np.abs(u).max() < 0.05:
        return 1 / 2 + u * (1 / 6 + u * (1 / 24 + u * (1 / 120 + u * (1 / 720 + u / 5040))))
    small = np.abs(u) < 1e-4
    us = np.where(small, 1.0, u)
    return np.where(small, 0.5 + u / 6 + u * u / 24, (np.expm1(us) - us) / (us * us))


def setup(cell, h=None, ds=None, n_p=64, sub=4):
    """Geometry of the in-plane field: grid (pitch h), chord bins (Δs ≈ ds on every boundary element, n_p bins in p),
    every chord's grid segments, where it ends and into which bins it reflects."""
    m = cell.meta
    b = boundary(cell)
    h = float(h if h else 2 * m["aperture"] / 48)
    ds = float(ds if ds else 2 * h)
    ox, oy = b.outline(48)
    ok = np.isfinite(ox)
    x0, x1, y0, y1 = ox[ok].min(), ox[ok].max(), oy[ok].min(), oy[ok].max()
    nx, ny = int(math.ceil((x1 - x0) / h)), int(math.ceil((y1 - y0) / h))
    x0 -= (nx * h - (x1 - x0)) / 2
    y0 -= (ny * h - (y1 - y0)) / 2
    xc = x0 + (np.arange(nx) + 0.5) * h
    yc = y0 + (np.arange(ny) + 0.5) * h
    # inside share of every grid cell (sub × sub samples)
    f = (np.arange(sub) + 0.5) / sub - 0.5
    SX = (xc[:, None, None, None] + f[None, None, :, None] * h) + 0 * f[None, None, None, :]
    SY = (yc[None, :, None, None] + f[None, None, None, :] * h) + 0 * f[None, None, :, None]
    SX, SY = np.broadcast_arrays(SX, SY)
    frac = b.inside(SX.reshape(-1), SY.reshape(-1)).reshape(nx, ny, sub * sub).mean(-1).reshape(-1)
    # chord bins
    dp = 2.0 / n_p
    pc = -1 + (np.arange(n_p) + 0.5) * dp
    nk = [max(2, int(round(b.L[k] / ds))) for k in range(4)]
    dsk = [b.L[k] / nk[k] for k in range(4)]
    off = np.r_[0, np.cumsum([n * n_p for n in nk])]
    S, Pp, Ek, Kc = [], [], [], []
    for k in range(4):
        sc = b.s0[k] + (np.arange(nk[k]) + 0.5) * dsk[k]
        S.append(np.repeat(sc, n_p)); Pp.append(np.tile(pc, nk[k])); Ek.append(np.full(nk[k] * n_p, dsk[k] * dp))
        Kc.append(np.full(nk[k] * n_p, k))
    s_c, p_c, e_c, k_c = (np.concatenate(v) for v in (S, Pp, Ek, Kc))
    C_n = len(s_c)
    x, y, nxo, nyo = b.point_at(s_c)
    tx, ty = -nyo, nxo
    q = np.sqrt(1 - p_c**2)
    dx, dy = p_c * tx - q * nxo, p_c * ty - q * nyo
    eps = 1e-9 * b.perimeter
    px, py = x - eps * nxo, y - eps * nyo
    t, nxe, nye, se, ke = b.hit_k(px, py, dx, dy)
    t = np.where(np.isfinite(t), t, 0.0)
    pe = dx * (-nye) + dy * nxe
    # end classes and reflection weights
    win = (cell.s_in_default - m["port_w"] / 2, cell.s_in_default + m["port_w"] / 2)
    end = np.full(C_n, SIDE)
    mir = (ke == 0) | (ke == 2)
    end[mir] = MIRROR
    end[(ke == 0) & (se >= win[0]) & (se <= win[1])] = WINDOW
    rows, cols, wts = [], [], []
    for k in (0, 2):
        sel = np.nonzero((end == MIRROR) & (ke == k))[0]
        u = (se[sel] - b.s0[k]) / dsk[k] - 0.5
        i0 = np.clip(np.floor(u).astype(int), 0, nk[k] - 2)
        a = np.clip(u - i0, 0.0, 1.0)
        v = (pe[sel] + 1) / dp - 0.5
        j0 = np.clip(np.floor(v).astype(int), 0, n_p - 2)
        bb = np.clip(v - j0, 0.0, 1.0)
        for di, dj, w in ((0, 0, (1 - a) * (1 - bb)), (1, 0, a * (1 - bb)), (0, 1, (1 - a) * bb), (1, 1, a * bb)):
            rows.append(off[k] + (i0 + di) * n_p + j0 + dj); cols.append(sel); wts.append(w)
    rows, cols, wts = (np.concatenate(v) for v in (rows, cols, wts))
    # grid segments of every chord: samples every h/2, consecutive samples in one cell merged
    nst = np.maximum(1, np.ceil(t / (0.5 * h)).astype(int))
    cid = np.repeat(np.arange(C_n), nst)
    first = np.r_[0, np.cumsum(nst)[:-1]]
    loc = np.arange(cid.size) - first[cid]
    fr = (loc + 0.5) / nst[cid]
    sx = px[cid] + fr * t[cid] * dx[cid]
    sy = py[cid] + fr * t[cid] * dy[cid]
    ix = np.clip(((sx - x0) / h).astype(int), 0, nx - 1)
    iy = np.clip(((sy - y0) / h).astype(int), 0, ny - 1)
    cell_id = ix * ny + iy
    step = t[cid] / nst[cid]
    brk = np.r_[True, (cell_id[1:] != cell_id[:-1]) | (cid[1:] != cid[:-1])]
    starts = np.nonzero(brk)[0]
    seg_cell = cell_id[starts]
    seg_len = np.add.reduceat(step, starts)
    seg_chord = cid[starts]
    cfirst = np.searchsorted(seg_chord, np.arange(C_n))
    seg_e = e_c[seg_chord]
    clast = np.r_[cfirst[1:], len(seg_chord)] - 1
    return dict(b=b, h=h, nx=nx, ny=ny, x0=x0, y0=y0, xc=xc, yc=yc, area=np.maximum(frac, 1.0 / (sub * sub)) * h * h, frac=frac,
                n_p=n_p, nk=nk, dsk=dsk, off=off, s=s_c, p=p_c, e=e_c, k=k_c, len=t, end=end, s_end=se, p_end=pe,
                chi_end=np.arcsin(np.clip(np.abs(pe), 0, 1)), rows=rows, cols=cols, wts=wts, seg_cell=seg_cell, seg_len=seg_len,
                seg_chord=seg_chord, seg_e=seg_e, cfirst=cfirst, clast=clast, window=win, n_chords=C_n)


def grid_points(F):
    """Centres of the grid cells (ix * ny + iy order)."""
    X, Y = np.meshgrid(F["xc"], F["yc"], indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1)], 1)


def bilinear(F, P):
    """Indices (m, 4) and weights (m, 4) of points P (m, 2) on the grid of cell centres."""
    nx, ny, h = F["nx"], F["ny"], F["h"]
    u = (P[:, 0] - F["x0"]) / h - 0.5
    v = (P[:, 1] - F["y0"]) / h - 0.5
    i0 = np.clip(np.floor(u).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, ny - 2)
    a = np.clip(u - i0, 0, 1)
    b = np.clip(v - j0, 0, 1)
    idx = np.stack([i0 * ny + j0, (i0 + 1) * ny + j0, i0 * ny + j0 + 1, (i0 + 1) * ny + j0 + 1], 1)
    w = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1)
    return idx, w


def mirror_R(F, lams, R_of):
    """Reflectance at every chord end (bins × chords): R(λ, χ) on the mirrors, 0 at the sides and the window."""
    Rm = np.zeros((len(lams), F["n_chords"]))
    mir = F["end"] == MIRROR
    for k, lam in enumerate(lams):
        Rm[k, mir] = R_of(float(lam), F["chi_end"][mir]) if R_of else 1.0
    return Rm


def _chord_cumsum(F, v):
    """Cumulative sum of segment values v (…, segments) restarted at every chord."""
    cs = np.cumsum(v, axis=-1)
    base = np.where(F["cfirst"] > 0, cs[..., np.maximum(F["cfirst"] - 1, 0)], 0.0)
    return cs - base[..., F["seg_chord"]]


def transport(F, G, S, Rm, tol=1e-8, max_bounce=4000, settle_tol=1e-5, cap=1e8):
    """Steady in-plane field for gain G (bins × cells, loss included) and source S (bins × cells, per étendue per
    length). Returns Φ (bins × cells, W/m), the end powers, start powers, bounce count and ratio per bin, and the
    per-bin totals generated / leaving."""
    sc, sl, ch = F["seg_cell"], F["seg_len"], F["seg_chord"]
    u = G[:, sc] * sl
    e1 = _e1(u)
    src = S[:, sc] * F["seg_e"]
    q = src * sl * e1                                   # source added in the segment, at its end
    Gc = _chord_cumsum(F, u)                            # cumulative gain to the end of the segment
    qe = q * np.exp(-Gc)
    Wc = _chord_cumsum(F, qe)
    T = np.exp(Gc[:, F["clast"]])
    Q = T * Wc[:, F["clast"]]
    nb, Cn = G.shape[0], F["n_chords"]
    rows, cols, wts = F["rows"], F["cols"], F["wts"]
    B = np.zeros((nb, Cn))
    d_prev = np.full(nb, np.nan)
    r_prev = np.full(nb, np.nan)
    done = np.zeros(nb, bool)
    ratio = np.zeros(nb)
    n_used = np.zeros(nb, int)
    stop = np.zeros(nb, bool)
    for it in range(max_bounce):
        RE = Rm * (T * B + Q)
        Bn = np.stack([np.bincount(rows, wts * RE[k, cols], minlength=Cn) for k in range(nb)])
        D = Bn - B
        d = np.abs(D).sum(1)
        tot = np.abs(Bn).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d_prev > 0, d / d_prev, 0.0)
        act = ~done & ~stop
        conv = act & (d <= tol * np.maximum(tot, 1e-300))
        settle = act & ~conv & (it > 4) & (r < 1) & (np.abs(r - r_prev) < settle_tol * r)
        if it == 0:
            first = np.maximum(tot, 1e-300)
        runaway = act & ((tot > cap * first) |           # trapped ASE growing bounce after bounce: parasitic lasing
                         ((it > 4) & (r >= 1) & (np.abs(r - r_prev) < 1e-4 * r)))
        B[act] = Bn[act]
        if settle.any():
            B[settle] += (r[settle] / (1 - r[settle]))[:, None] * D[settle]
        ratio[act] = r[act]
        n_used[act] = it + 1
        done |= conv | settle
        stop = stop | runaway
        if (done | stop).all():
            break
        d_prev, r_prev = d, r
    lasing = ~done & (ratio >= 1)
    E = T * B + Q
    # deposition: ∫P dl in every segment
    P0 = np.exp(Gc - u) * (B[:, ch] + Wc - qe)
    intP = P0 * sl * e1 + src * sl * sl * _e2(u)
    ncell = F["nx"] * F["ny"]
    Phi = np.stack([np.bincount(sc, intP[k], minlength=ncell) for k in range(nb)]) / F["area"][None, :]
    end = F["end"]
    out = dict(Phi=Phi, E=E, B=B, bounces=n_used, ratio=ratio, lasing=lasing, converged=done,
               window=(E[:, end == WINDOW]).sum(1), side=(E[:, end == SIDE]).sum(1),
               mirror_loss=((1 - Rm) * E)[:, end == MIRROR].sum(1),
               source=np.stack([np.bincount(sc, (src * sl)[k], minlength=ncell) for k in range(nb)]),
               intP=np.stack([np.bincount(sc, intP[k], minlength=ncell) for k in range(nb)]))
    return out


def mode_noise(fld, hits, w_out, lam_s, n_eff, R_of=None, lams=None, max_bounce=2000, t_min=1e-7):
    """ASE that reaches the output signal mode, by back-tracing the mode through the cell. The mode at the window is
    sampled by 7 × 7 Gauss–Hermite rays of its Wigner function (1/e² radius w_out along the mirror, divergence
    λs/(n_eff π w_out)); each ray runs backwards through the converged cell (reflections × R(λ, χ), gain of the grid)
    and collects the sources along it. Returns, at the wavelengths lams (default λs), the spectral density in the mode
    arriving at the window [W/Hz, all counted polarisations]: spont (spontaneous emission along the mode: its own ASE),
    scat_field (in-plane field scattered into it) and scat_beam (on-path ASE scattered into it); the bounces of the
    longest ray."""
    F = fld["F"]
    b = F["b"]
    lams = np.atleast_1d(np.asarray(lam_s if lams is None else lams, float))
    nb = len(lams)
    sa, se = _er.sigma_a_arr(lams, fld["sigma_peak"]), _er.sigma_e_arr(lams, fld["sigma_peak"])
    A1, A2 = fld["A"]
    G = se[:, None] * A2[None] - sa[:, None] * A1[None] - fld["alpha"]
    Ssp = (fld["modes"] * fld["n_eff"] / lams * H * C / lams * se)[:, None] * A2[None]
    lb = fld["lams"]
    Wl = np.array([np.interp(lams, lb, np.eye(len(lb))[i]) for i in range(len(lb))]).T      # (eval, bins) linear weights
    comps = (Ssp, Wl @ (fld["S_sc"] / fld["dnu"][:, None]), Wl @ (fld["S_cb"] / fld["dnu"][:, None]))
    x, y = hits["x"], hits["y"]
    de = np.array([x[-1] - x[-2], y[-1] - y[-2]])
    de /= np.hypot(*de)
    se = F["b"].hit_k(x[-2], y[-2], de[0], de[1])[3][0]          # boundary coordinate of the exit point
    _, _, nxo, nyo = b.point_at(np.array([se]))
    cos_chi = abs(de[0] * nxo[0] + de[1] * nyo[0])
    th_w = lam_s / (n_eff * math.pi * w_out)
    yo = w_out / math.sqrt(2) * GH_X
    to = th_w / math.sqrt(2) * GH_X
    S0 = np.repeat(se + yo / max(cos_chi, 1e-6), len(GH_X))
    TH = np.tile(to, len(GH_X))
    Wt = np.outer(GH_W, GH_W).reshape(-1)
    px, py, nx_, ny_ = b.point_at(S0)
    c, s_ = np.cos(TH), np.sin(TH)
    vx, vy = -(de[0] * c - de[1] * s_), -(de[0] * s_ + de[1] * c)
    eps = 1e-9 * b.perimeter
    px, py = px - eps * nx_, py - eps * ny_
    nr = len(S0)
    T = np.ones((nr, nb))
    acc = np.zeros((3, nr, nb))
    alive = np.ones(nr, bool)
    h, nxg, nyg = F["h"], F["nx"], F["ny"]
    nbounce = 0
    for it in range(max_bounce):
        idx = np.nonzero(alive)[0]
        if not idx.size:
            break
        nbounce = it + 1
        t, hx, hy, hs, hk = b.hit_k(px[idx], py[idx], vx[idx], vy[idx])
        t = np.where(np.isfinite(t), t, 0.0)
        nst = np.maximum(1, np.ceil(t / (0.5 * h)).astype(int))
        rid = np.repeat(np.arange(idx.size), nst)
        first = np.r_[0, np.cumsum(nst)[:-1]]
        fr = (np.arange(rid.size) - first[rid] + 0.5) / nst[rid]
        sx = px[idx][rid] + fr * t[rid] * vx[idx][rid]
        sy = py[idx][rid] + fr * t[rid] * vy[idx][rid]
        cid = np.clip(((sx - F["x0"]) / h).astype(int), 0, nxg - 1) * nyg + np.clip(((sy - F["y0"]) / h).astype(int), 0, nyg - 1)
        ln = (t / nst)[rid]
        u = G[:, cid] * ln                                                  # (bins, samples)
        e1 = _e1(u)
        cs = np.cumsum(u, axis=1)
        base = np.where(first > 0, cs[:, np.maximum(first - 1, 0)], 0.0)[:, rid]
        before = np.exp(cs - u - base)                                      # gain from the chord start to the sample
        Tr = T[idx][rid].T                                                  # (bins, samples)
        for q_, Sq in enumerate(comps):
            contrib = Tr * before * Sq[:, cid] * ln * e1
            for k in range(nb):
                acc[q_, idx, k] += np.bincount(rid, contrib[k], minlength=idx.size)
        tot_u = np.add.reduceat(u, first, axis=1)                           # (bins, rays)
        T[idx] *= np.exp(tot_u.T)
        # where each chord ends
        ex, ey = px[idx] + t * vx[idx], py[idx] + t * vy[idx]
        win = (hk == 0) & (hs >= F["window"][0]) & (hs <= F["window"][1])
        mir = ((hk == 0) | (hk == 2)) & ~win
        dot = vx[idx] * hx + vy[idx] * hy
        chi = np.arcsin(np.clip(np.abs(vx[idx] * (-hy) + vy[idx] * hx), 0, 1))
        for k, lam in enumerate(lams):
            T[idx, k] *= np.where(mir, (R_of(float(lam), chi) if R_of else 1.0), 0.0)
        nvx, nvy = vx[idx] - 2 * dot * hx, vy[idx] - 2 * dot * hy
        px[idx], py[idx] = ex - eps * hx, ey - eps * hy
        vx[idx], vy[idx] = nvx, nvy
        alive[idx] = mir & (T[idx].max(1) > t_min)
    lm = (lams / n_eff)[None, :]
    P = (Wt[None, :, None] * acc).sum(1) * lm
    return dict(spont=P[0], scat_field=P[1], scat_beam=P[2], bounces=nbounce, exit_s=float(se), w_out=w_out, theta_w=th_w)
