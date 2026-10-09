"""Ray + Gaussian-beamlet ("complex ray") tracer for the circular CMPC.

Between perturbations the slab is uniform (n0): rays are straight and the beamlet propagates exactly. Inside the
bounding circles of the perturbations the ray equation and the 2D dynamic-ray-tracing (Červený) system are
integrated with RK4 in arclength s:

    dr/ds = t,  dp/ds = ∇n  (p = n t),       dL/ds = n                  (optical path)
    dQ/ds = P / n,  dP/ds = (n_nn − 2 n_n² / n) Q                        (paraxial beam about the ray)

n_n, n_nn: first and second derivative of n normal to the ray. The beamlet field near the ray is
A sqrt(Q0/Q) exp(i k0 [L + ½ (P/Q) q²]), q the normal offset; P0/Q0 = i n0 / z_R at the launch waist.
At the wall (radius Rc) a ray reflects specularly; the circular mirror acts on the beamlet as the tangential
oblique-incidence lens P → P − 2 n0 Q / (Rc cos χ), amplitude × sqrt(R(χ)), phase + φ_R.
Field models (cell.model):
  "fga" (default): frozen Gaussian approximation (Herman–Kluk). The input beam at a port (waist w_in set by the
     fan: divergence half-angle fan/2) is projected onto coherent states g_{q,p} (fixed width w_f = Source.frozen)
     on the launch grid of positions q × directions p = n0 sin φ; each ray carries its weight W = ⟨g|E0⟩ ΔqΔp k0/2π,
     the optical path and the stability matrix [[A, B], [C, D]] (from Q = A + iβB, P = C + iβD), and contributes
     W · R · e^{i k0 L} · g at the end, with the prefactor R = sqrt(½ (A + D − i (γ/k0) B + i (k0/γ) C)), γ = 2/w_f²,
     its branch followed continuously. Frozen Gaussians stay narrow, so a dot only affects the rays that cross it.
     Ports: every wall hit within a few w_f of a port opening is an exit record (field.beamlet_matrix samples it
     across the opening and propagates the aperture field to the detector); the ray reflects with amplitude ×
     sqrt(1 − T), T the overlap of its footprint w_f / cos χ with the opening. Smooth in the ray geometry.
  "gbs": Gaussian beam summation with evolving beamlets (P/Q), hard ports: a ray leaves when its centre hits a
     port and its beamlet is continued straight to the detector. In the circle (a degenerate mirror system) these
     beamlets grow to millimetres, so this model is kept only for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Source:
    port: int
    n_pos: int = 5               # launch positions (gbs: across the aperture; fga: across the coherent-state
                                 # extent ±2.5 sqrt(w_in² + w_f²); None → automatic, spacing w_f/2)
    n_ang: int = 41              # launch angles across the fan
    waist: float = 5e-6          # beamlet waist at the port (m)
    mode_width: float = None     # gbs: 1/e field half-width of the input mode across the aperture (None → width/2);
                                 # fga: input waist (None → from the fan, divergence half-angle fan/2)
    ang_width: float = None      # gbs: 1/e half-width of an angular weight; None → flat over the fan
    frozen: float = 20e-6        # fga: frozen-Gaussian 1/e amplitude half-width w_f (m)


@dataclass
class Exits:
    port: np.ndarray
    ray: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    L: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    argQ: np.ndarray
    amp: np.ndarray
    phase: np.ndarray
    nb: np.ndarray
    n_launched: int = 0
    lost: dict = field(default_factory=dict)
    paths: list = None
    chords: dict = None
    nch: np.ndarray = None       # chords travelled by the ray before this exit (for per-exit path sums)
    model: str = "gbs"           # "fga": frozen-Gaussian records at the wall, sampled over the port opening
    W: np.ndarray = None         # fga: complex launch weight of the ray
    argZ: np.ndarray = None      # fga: continuous argument of the prefactor argument z
    beta: float = None
    gamma: float = None

    def select(self, port):
        m = self.port == port
        out = {k: getattr(self, k)[m] for k in ("ray", "x", "y", "tx", "ty", "L", "Q", "P", "argQ", "amp", "phase", "nb")}
        if self.model == "fga":
            out["W"], out["argZ"] = self.W[m], self.argZ[m]
        return out

    def index(self, port):
        return np.nonzero(self.port == port)[0]

    def subset(self, mask):
        import copy
        e = copy.copy(self)
        for k in ("port", "ray", "x", "y", "tx", "ty", "L", "Q", "P", "argQ", "amp", "phase", "nb", "nch", "argZ", "W"):
            v = getattr(self, k)
            if v is not None:
                setattr(e, k, v[mask])
        return e

    def weights(self, k0):
        """|amp · W · R| of every fga record (its amplitude at the port)."""
        z = zfun(self.Q, self.P, self.beta, self.gamma, k0)
        return self.amp * np.abs(self.W) * np.sqrt(np.abs(z))

    def port_power(self, n_ports):
        return np.array([np.sum(self.amp[self.port == i] ** 2) for i in range(n_ports)])


def launch(cell, src: Source):
    p = cell.ports[src.port]
    c, nrm, tan = cell.port_frame(src.port)
    u = (np.linspace(-0.5, 0.5, src.n_pos) if src.n_pos > 1 else np.zeros(1)) * 0.9 * p.width
    v = (np.linspace(-0.5, 0.5, src.n_ang) if src.n_ang > 1 else np.zeros(1)) * p.fan
    U, V = np.meshgrid(u, v, indexing="ij")
    U, V = U.ravel(), V.ravel()
    x = c[0] + U * tan[0]
    y = c[1] + U * tan[1]
    ang = np.arctan2(-nrm[1], -nrm[0]) + p.launch + V
    mw = p.width / 2 if src.mode_width is None else src.mode_width
    amp = np.exp(-(U / mw) ** 2)
    if src.ang_width is not None:
        amp = amp * np.exp(-(V / src.ang_width) ** 2)
    amp = amp / np.sqrt(np.sum(amp**2))
    return x, y, np.cos(ang), np.sin(ang), amp


def fga_params(cell, src: Source):
    """β (Q, P encoding) and γ = 2/w_f² of the frozen Gaussians."""
    zR = cell.k0 * cell.n_eff * src.frozen**2 / 2
    return cell.n_eff / zR, 2 / src.frozen**2


def launch_fga(cell, src: Source):
    """Phase-space grid of frozen Gaussians for the beam injected at src.port: positions q across the port line,
    directions φ (from the inward normal); returns x, y, tx, ty, the complex weights W = ⟨g_{q,p}|E0⟩ ΔqΔp k0/2π,
    and cos φ (for the line → ray-frame stability matrix)."""
    p = cell.ports[src.port]
    c, nrm, tan = cell.port_frame(src.port)
    n0, k0, lam = cell.n_eff, cell.k0, cell.wavelength
    _, gam = fga_params(cell, src)
    th_d = max(p.fan / 2, 1e-4)
    w_in = src.mode_width if src.mode_width is not None else lam / (np.pi * n0 * np.tan(th_d))
    qmax = 2.5 * np.sqrt((w_in / max(np.cos(p.launch), 1e-3)) ** 2 + src.frozen**2)   # extent of ⟨g_q|E0⟩
    n_pos = src.n_pos if src.n_pos else int(np.ceil(2 * qmax / (0.5 * src.frozen))) + 1
    q = np.linspace(-qmax, qmax, n_pos) if n_pos > 1 else np.zeros(1)
    phi_span = min(2.5 * th_d + 3 / (k0 * n0 * src.frozen), 1.4)
    phi = p.launch + (np.linspace(-1, 1, src.n_ang) * phi_span if src.n_ang > 1 else np.zeros(1))
    dq = q[1] - q[0] if len(q) > 1 else np.sqrt(2 * np.pi / gam)
    dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
    Qg, Pg = np.meshgrid(q, phi, indexing="ij")
    Qg, Pg = Qg.ravel(), Pg.ravel()
    pp = -n0 * np.sin(Pg)                      # momentum along the port line (tangent), dir·tan = −sin φ
    p0 = -n0 * np.sin(p.launch)
    w_line = w_in / max(np.cos(p.launch), 1e-3)       # a beam of waist w_in launched at angle φ0, seen on the line
    a = gam / 2 + 1 / w_line**2
    b = gam * Qg - 1j * k0 * pp + 1j * k0 * p0
    cc = -gam / 2 * Qg**2 + 1j * k0 * pp * Qg
    ov = (gam / np.pi) ** 0.25 * np.sqrt(np.pi / a) * np.exp(b * b / (4 * a) + cc)
    W = ov * dq * n0 * np.cos(Pg) * dphi * k0 / (2 * np.pi)
    ang = np.arctan2(-nrm[1], -nrm[0]) + Pg
    x = c[0] + Qg * tan[0]
    y = c[1] + Qg * tan[1]
    return x, y, np.cos(ang), np.sin(ang), W, np.cos(Pg)


def zfun(Q, P, beta, gam, k0):
    """Herman–Kluk prefactor argument z = ½ (A + D − i(γ/k0) B + i(k0/γ) C) with Q = A + iβB, P = C + iβD."""
    A, B, C, D = Q.real, Q.imag / beta, P.real, P.imag / beta
    return 0.5 * (A + D - 1j * (gam / k0) * B + 1j * (k0 / gam) * C)


def _derivs(pert, n0, mode, x, y, px, py, Q, P):
    if mode == "curved":
        dn, gx, gy, hxx, hxy, hyy = pert.eval(x, y, hess=True)
    else:
        dn = pert.eval(x, y, hess=False)[0]
    n = n0 + dn
    pn = np.hypot(px, py)
    tx, ty = px / pn, py / pn
    if mode == "curved":
        ex, ey = -ty, tx
        nn = gx * ex + gy * ey
        nnn = ex * ex * hxx + 2 * ex * ey * hxy + ey * ey * hyy
        return tx, ty, gx, gy, n, P / n, (nnn - 2 * nn * nn / n) * Q
    z = np.zeros_like(x)
    return tx, ty, z, z, n, P / n0, 0 * Q


def _rk4(pert, n0, mode, h, x, y, px, py, L, Q, P):
    s = (x, y, px, py, L, Q, P)
    k1 = _derivs(pert, n0, mode, x, y, px, py, Q, P)
    def add(st, k, f):
        return [a + f * b for a, b in zip(st, k)]
    s2 = add(s, k1, h / 2)
    k2 = _derivs(pert, n0, mode, s2[0], s2[1], s2[2], s2[3], s2[5], s2[6])
    s3 = add(s, k2, h / 2)
    k3 = _derivs(pert, n0, mode, s3[0], s3[1], s3[2], s3[3], s3[5], s3[6])
    s4 = add(s, k3, h)
    k4 = _derivs(pert, n0, mode, s4[0], s4[1], s4[2], s4[3], s4[5], s4[6])
    return [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4)]


def trace(cell, src: Source, pert=None, mode="curved", ds=None, max_bounces=400, amp_min=1e-3,
          record_paths=0, record_chords=False, max_iter=200000, n_sub=8, adaptive=True, rec_tol=1e-3, zmax=1e4):
    """Trace the fan of `src` through the cell and perturbation. Returns Exits (one record per ray that leaves
    through a port). ds: RK4 step in the edge zones of the perturbations (default half the edge width of the sharpest
    shape, ≤ 2 µm; ds = edge gives field correlations ≈ 0.996 with the converged result); with `adaptive`, flat parts (Shape interiors, gaps) are crossed in larger steps that stop short of the
    next edge zone. Rays inside perturbations advance n_sub steps per event round, batched over rays.
    zmax (fga): a ray is dropped once its Herman–Kluk prefactor argument |z| exceeds zmax. In chaotic cells
    |z| grows like e^(λ·bounces) and those contributions only cancel with impractically dense sampling (beyond the
    Ehrenfest time ray methods lose the field anyway); in the circle |z| grows linearly and stays far below.
    rec_tol: aperture-model exit records are kept when the beamlet's amplitude at the port (amp |Q|^-1/2, Gaussian
    fall-off beyond the opening) exceeds rec_tol × the largest launch amplitude."""
    from .shapes import Perturbation
    pert = Perturbation() if pert is None else pert
    if mode == "none":
        pert = Perturbation()
    Rc, n0, k0 = cell.radius, cell.n_eff, cell.k0
    cell.check_regions(pert.regions)
    fga = getattr(cell, "model", "gbs") == "fga"
    reg = pert.regions
    if ds is None:
        edges = [getattr(c, "edge", getattr(c, "smooth", 2e-6) or 2e-6) for c in pert.components]
        ds = min([2e-6] + [e / 2 for e in edges])

    if fga:
        x, y, tx, ty, Wl, cos0 = launch_fga(cell, src)
        beta, gam = fga_params(cell, src)
        amp = np.ones(len(x))
        Q = cos0.astype(complex)
        P = 1j * beta / cos0
        argZ = np.zeros(len(x))
        wmag = np.abs(Wl)
        live = wmag > rec_tol * wmag.max()            # drop launch cells with negligible projection weight
        x, y, tx, ty, Wl, Q, P, argZ, amp = (v[live] for v in (x, y, tx, ty, Wl, Q, P, argZ, amp))
    else:
        x, y, tx, ty, amp = launch(cell, src)
        zR = k0 * n0 * src.waist**2 / 2
        Q = np.ones(len(x), complex)
        P = np.full(len(x), 1j * n0 / zR)
        argZ = np.zeros(len(x))
    N = len(x)
    px, py = n0 * tx, n0 * ty
    argQ = np.zeros(N)
    L = np.zeros(N)
    phase = np.zeros(N)
    nb = np.zeros(N, int)
    nch = np.zeros(N, int)
    alive = np.ones(N, bool)
    inside = np.zeros(N, bool)
    amp0max = amp.max()
    rec = {k: [] for k in ("port", "ray", "x", "y", "tx", "ty", "L", "Q", "P", "argQ", "amp", "phase", "nb", "nch",
                           "argZ")}
    lost = dict(bounces=0, amplitude=0, power_bounces=0.0, power_amplitude=0.0)
    paths = [[(x[i], y[i])] for i in range(min(record_paths, N))]
    chords = {k: [] for k in ("ray", "x", "y", "tx", "ty", "len")} if record_chords else None
    ids = np.arange(N)

    def zarg_update(m, Qn, Pn):
        if fga:
            argZ[m] += np.angle(zfun(Qn, Pn, beta, gam, k0) / zfun(Q[m], P[m], beta, gam, k0))

    def free(m, s):
        nonlocal Q, argQ
        x[m] += s * tx[m]
        y[m] += s * ty[m]
        L[m] += n0 * s
        Qn = Q[m] + P[m] * s / n0
        zarg_update(m, Qn, P[m])
        argQ[m] += np.angle(Qn / Q[m])
        Q[m] = Qn

    it = 0
    while alive.any():
        it += 1
        if it > max_iter:
            raise RuntimeError("trace: max_iter exceeded")
        out = np.nonzero(alive & ~inside)[0]
        if len(out):
            xo, yo, txo, tyo = x[out], y[out], tx[out], ty[out]
            s_wall, wnx, wny, wpow, wsb = cell.wall_hit(xo, yo, txo, tyo)
            leak = ~np.isfinite(s_wall)
            if leak.any():
                lost["leaked"] = lost.get("leaked", 0) + int(leak.sum())
                alive[out[leak]] = False
                keep = ~leak
                out, xo, yo, txo, tyo = out[keep], xo[keep], yo[keep], txo[keep], tyo[keep]
                s_wall, wnx, wny, wpow, wsb = s_wall[keep], wnx[keep], wny[keep], wpow[keep], wsb[keep]
            s_reg = np.full(len(out), np.inf)
            for cx, cy, rr in reg:
                dx, dy = xo - cx, yo - cy
                bb = dx * txo + dy * tyo
                cc = dx * dx + dy * dy - rr * rr
                disc = bb * bb - cc
                ok = (cc > 0) & (bb < 0) & (disc > 0)
                s = np.where(ok, -bb - np.sqrt(np.where(ok, disc, 0)), np.inf)
                s_reg = np.minimum(s_reg, s)
            to_reg = s_reg < s_wall
            s_go = np.where(to_reg, s_reg, s_wall)
            if record_chords:
                chords["ray"].append(out); chords["x"].append(xo.copy()); chords["y"].append(yo.copy())
                chords["tx"].append(txo.copy()); chords["ty"].append(tyo.copy()); chords["len"].append(s_go.copy())
            free(out, s_go)
            nch[out] += 1
            inside[out[to_reg]] = True
            px[out[to_reg]] = n0 * tx[out[to_reg]]
            py[out[to_reg]] = n0 * ty[out[to_reg]]
            hit = out[~to_reg]
            if len(hit):
                hsel = ~to_reg
                hn_x, hn_y = wnx[hsel], wny[hsel]
                widths = np.array([p.width for p in cell.ports])
                D = cell.port_offsets(wsb[hsel])                          # (ports, hits)
                if fga:
                    from scipy.special import erf
                    foot = np.full(len(hit), src.frozen)                   # 1/e² intensity half-width on the line
                    near = np.abs(D) < widths[:, None] / 2 + 3 * foot
                    ww = foot / np.sqrt(2)
                    T = 0.5 * (erf((widths[:, None] / 2 - D) / ww) + erf((widths[:, None] / 2 + D) / ww))
                    T = np.where(near, T, 0.0)
                    Ttot = np.minimum(T.sum(0), 1.0)
                    Ttot = np.where(Ttot > 1 - 1e-9, 1.0, Ttot)
                else:
                    near = np.abs(D) <= widths[:, None] / 2
                    Ttot = near.any(0).astype(float)
                for pi_ in range(len(cell.ports)):
                    sel = near[pi_]
                    if not sel.any():
                        continue
                    ex = hit[sel]
                    for k, arr in (("ray", ids), ("x", x), ("y", y), ("tx", tx), ("ty", ty), ("L", L), ("Q", Q),
                                   ("P", P), ("argQ", argQ), ("amp", amp), ("phase", phase), ("nb", nb), ("nch", nch),
                                   ("argZ", argZ)):
                        rec[k].append(arr[ex].copy())
                    rec["port"].append(np.full(len(ex), pi_))
                cont = Ttot < 1
                alive[hit[~cont]] = False
                rsel = cont
                rf = hit[rsel]
                if len(rf):
                    nx, ny = hn_x[rsel], hn_y[rsel]
                    cosc = tx[rf] * nx + ty[rf] * ny
                    tx[rf] -= 2 * cosc * nx
                    ty[rf] -= 2 * cosc * ny
                    tn = np.hypot(tx[rf], ty[rf])
                    tx[rf] /= tn
                    ty[rf] /= tn
                    Pn = P[rf] - 2 * n0 * Q[rf] * wpow[hsel][rsel] / np.maximum(cosc, 1e-9)
                    zarg_update(rf, Q[rf], Pn)
                    P[rf] = Pn
                    amp[rf] *= np.sqrt(1 - Ttot[rsel])
                    amp[rf] *= np.sqrt(cell.R(cosc))
                    phase[rf] += cell.reflection_phase
                    nb[rf] += 1
                    if fga:
                        kz = rf[np.abs(zfun(Q[rf], P[rf], beta, gam, k0)) > zmax]
                        lost["prefactor"] = lost.get("prefactor", 0) + len(kz)
                        alive[kz] = False
                    kb = rf[nb[rf] >= max_bounces]
                    ka = rf[(nb[rf] < max_bounces) & (amp[rf] < amp_min * amp0max)]
                    lost["bounces"] += len(kb); lost["power_bounces"] += float(np.sum(amp[kb] ** 2))
                    lost["amplitude"] += len(ka); lost["power_amplitude"] += float(np.sum(amp[ka] ** 2))
                    alive[kb] = False
                    alive[ka] = False
            if paths:
                so = set(out.tolist())
                for i in range(len(paths)):
                    if i in so:
                        paths[i].append((x[i], y[i]))
        ins = np.nonzero(alive & inside)[0]
        for _ in range(n_sub if len(ins) else 0):
            h = pert.safe_step(x[ins], y[ins], ds) if adaptive else ds
            s = _rk4(pert, n0, mode, h, x[ins], y[ins], px[ins], py[ins], L[ins], Q[ins], P[ins])
            x[ins], y[ins], px[ins], py[ins], L[ins] = s[0], s[1], s[2], s[3], s[4]
            zarg_update(ins, s[5], s[6])
            argQ[ins] += np.angle(s[5] / Q[ins])
            Q[ins], P[ins] = s[5], s[6]
            if not np.all(cell.contains(x[ins], y[ins])):
                raise RuntimeError("ray reached the wall inside a perturbation region")
            inany = np.zeros(len(ins), bool)
            for cx, cy, rr in reg:
                inany |= (x[ins] - cx) ** 2 + (y[ins] - cy) ** 2 < rr * rr
            done = ins[~inany]
            if len(done):
                pn = np.hypot(px[done], py[done])
                tx[done], ty[done] = px[done] / pn, py[done] / pn
                inside[done] = False
            for i in range(len(paths)):
                if i in ins:
                    paths[i].append((x[i], y[i]))
            ins = ins[inany]
            if not len(ins):
                break

    cat = lambda k, dt: np.concatenate(rec[k]) if rec[k] else np.zeros(0, dt)
    ex = Exits(port=cat("port", int), ray=cat("ray", int), x=cat("x", float), y=cat("y", float), tx=cat("tx", float),
               ty=cat("ty", float), L=cat("L", float), Q=cat("Q", complex), P=cat("P", complex),
               argQ=cat("argQ", float), amp=cat("amp", float), phase=cat("phase", float), nb=cat("nb", int),
               nch=cat("nch", int), argZ=cat("argZ", float), model="fga" if fga else "gbs", n_launched=N, lost=lost, paths=[np.array(p) for p in paths] if record_paths else None)
    if record_chords:
        ex.chords = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chords.items()}
    if fga:
        ex.W = Wl[ex.ray]
        ex.beta, ex.gamma = beta, gam
    return ex
