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
A wall hit inside a port aperture ends the ray there (an exit record).

mode = "curved" (full), "straight" (rays and beamlets as in the unperturbed cell, only the optical path feels Δn:
the first-order / phase-screen model), "none" (perturbation ignored).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Source:
    port: int
    n_pos: int = 5               # launch positions across the aperture
    n_ang: int = 41              # launch angles across the fan
    waist: float = 5e-6          # beamlet waist at the port (m)
    mode_width: float = None     # 1/e field half-width of the input mode across the aperture; None → width/2
    ang_width: float = None      # 1/e half-width of an angular weight; None → flat over the fan


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

    def select(self, port):
        m = self.port == port
        return {k: getattr(self, k)[m] for k in ("ray", "x", "y", "tx", "ty", "L", "Q", "P", "argQ", "amp", "phase", "nb")}

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
          record_paths=0, record_chords=False, max_iter=200000, n_sub=8, adaptive=True):
    """Trace the fan of `src` through the cell and perturbation. Returns Exits (one record per ray that leaves
    through a port). ds: RK4 step in the edge zones of the perturbations (default half the edge width of the sharpest
    shape, ≤ 2 µm; ds = edge gives field correlations ≈ 0.996 with the converged result); with `adaptive`, flat parts (Shape interiors, gaps) are crossed in larger steps that stop short of the
    next edge zone. Rays inside perturbations advance n_sub steps per event round, batched over rays."""
    from .shapes import Perturbation
    pert = Perturbation() if pert is None else pert
    if mode == "none":
        pert = Perturbation()
    Rc, n0, k0 = cell.radius, cell.n_eff, cell.k0
    pert.check_inside(Rc)
    reg = pert.regions
    if ds is None:
        edges = [getattr(c, "edge", getattr(c, "smooth", 2e-6) or 2e-6) for c in pert.components]
        ds = min([2e-6] + [e / 2 for e in edges])

    x, y, tx, ty, amp = launch(cell, src)
    N = len(x)
    px, py = n0 * tx, n0 * ty
    zR = k0 * n0 * src.waist**2 / 2
    Q = np.ones(N, complex)
    P = np.full(N, 1j * n0 / zR)
    argQ = np.zeros(N)
    L = np.zeros(N)
    phase = np.zeros(N)
    nb = np.zeros(N, int)
    alive = np.ones(N, bool)
    inside = np.zeros(N, bool)
    amp0max = amp.max()
    rec = {k: [] for k in ("port", "ray", "x", "y", "tx", "ty", "L", "Q", "P", "argQ", "amp", "phase", "nb")}
    lost = dict(bounces=0, amplitude=0, power_bounces=0.0, power_amplitude=0.0)
    paths = [[(x[i], y[i])] for i in range(min(record_paths, N))]
    chords = {k: [] for k in ("ray", "x", "y", "tx", "ty", "len")} if record_chords else None
    ids = np.arange(N)

    def free(m, s):
        nonlocal Q, argQ
        x[m] += s * tx[m]
        y[m] += s * ty[m]
        L[m] += n0 * s
        Qn = Q[m] + P[m] * s / n0
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
            b = xo * txo + yo * tyo
            s_wall = -b + np.sqrt(np.maximum(b * b - (xo * xo + yo * yo) + Rc * Rc, 0))
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
            inside[out[to_reg]] = True
            px[out[to_reg]] = n0 * tx[out[to_reg]]
            py[out[to_reg]] = n0 * ty[out[to_reg]]
            hit = out[~to_reg]
            if len(hit):
                xh, yh = x[hit], y[hit]
                rr = np.hypot(xh, yh)
                x[hit], y[hit] = xh * Rc / rr, yh * Rc / rr
                pid = cell.port_index(np.arctan2(y[hit], x[hit]))
                ex = hit[pid >= 0]
                if len(ex):
                    for k, arr in (("ray", ids), ("x", x), ("y", y), ("tx", tx), ("ty", ty), ("L", L), ("Q", Q),
                                   ("P", P), ("argQ", argQ), ("amp", amp), ("phase", phase), ("nb", nb)):
                        rec[k].append(arr[ex].copy())
                    rec["port"].append(pid[pid >= 0])
                    alive[ex] = False
                rf = hit[pid < 0]
                if len(rf):
                    nx, ny = x[rf] / Rc, y[rf] / Rc
                    cosc = tx[rf] * nx + ty[rf] * ny
                    tx[rf] -= 2 * cosc * nx
                    ty[rf] -= 2 * cosc * ny
                    P[rf] = P[rf] - 2 * n0 * Q[rf] / (Rc * np.maximum(cosc, 1e-9))
                    amp[rf] *= np.sqrt(cell.R(cosc))
                    phase[rf] += cell.reflection_phase
                    nb[rf] += 1
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
            argQ[ins] += np.angle(s[5] / Q[ins])
            Q[ins], P[ins] = s[5], s[6]
            if np.any(x[ins] ** 2 + y[ins] ** 2 >= Rc * Rc):
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
               n_launched=N, lost=lost, paths=[np.array(p) for p in paths] if record_paths else None)
    if record_chords:
        ex.chords = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chords.items()}
    return ex
