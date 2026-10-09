"""First-order (phase-screen) model: the unperturbed rays and beamlets are traced once; a perturbation only adds
the optical path ΔL_j = ∫ Δn ds along each ray's unperturbed chords, so E = G @ exp(i k0 ΔL).

Exact for the "straight" tracer mode (same physics, quadrature instead of RK4), and the Δn → 0 limit of the full
curved-ray result. It ignores ray bending; the bending matters once the deflection × remaining path exceeds the
beamlet width (see analysis.validity_scan)."""
from __future__ import annotations

import numpy as np

from .field import beamlet_matrix
from .rays import Source, trace


class PhaseScreenModel:
    def __init__(self, cell, sources, max_bounces=400, amp_min=1e-3):
        self.cell = cell
        self.sources = [s if isinstance(s, Source) else Source(int(s)) for s in sources]
        self.outputs = cell.outputs
        self.tables = []
        for src in self.sources:
            ex = trace(cell, src, None, mode="none", max_bounces=max_bounces, amp_min=amp_min, record_chords=True)
            G = {}
            for p in self.outputs:
                G[p] = beamlet_matrix(cell, ex, p)
            self.tables.append(dict(exits=ex, chords=ex.chords, G=G, n_rays=ex.n_launched))

    def delta_L(self, k, pert, step=None, chunk=400000):
        """ΔL per launched ray of source k (m): midpoint quadrature of each component along the chords that cross
        its bounding circle. The integrand is smooth and vanishes at both ends, so the midpoint rule converges
        spectrally: step = edge gives |k0 ΔL| errors ≈ 1e-4 rad, edge/2 ≈ 1e-6 rad."""
        ch = self.tables[k]["chords"]
        dL = np.zeros(self.tables[k]["n_rays"])
        if pert is None or len(pert) == 0:
            return dL
        x, y, tx, ty, ln, ray = ch["x"], ch["y"], ch["tx"], ch["ty"], ch["len"], ch["ray"].astype(int)
        for comp in pert.components:
            cx, cy, rr = comp.x0, comp.y0, comp.bound
            dx, dy = x - cx, y - cy
            b = dx * tx + dy * ty
            disc = b * b - (dx * dx + dy * dy - rr * rr)
            ok = disc > 0
            sq = np.sqrt(np.where(ok, disc, 0))
            s1 = np.maximum(-b - sq, 0)
            s2 = np.minimum(-b + sq, ln)
            idx = np.nonzero(ok & (s2 > s1))[0]
            if not len(idx):
                continue
            st = min(3e-6, getattr(comp, "edge", getattr(comp, "smooth", 2e-6) or 2e-6)) if step is None else step
            nq = max(8, int(np.ceil(np.max(s2[idx] - s1[idx]) / st)) + 1)
            g = (np.arange(nq) + 0.5) / nq
            for c0 in range(0, len(idx), max(1, chunk // nq)):
                j = idx[c0:c0 + max(1, chunk // nq)]
                seg = s2[j] - s1[j]
                s = s1[j][:, None] + g[None, :] * seg[:, None]
                v = comp.eval(x[j][:, None] + s * tx[j][:, None], y[j][:, None] + s * ty[j][:, None], hess=False)[0]
                np.add.at(dL, ray[j], v.mean(axis=1) * seg)
        return dL

    def fields(self, pert):
        """{(input port, output port): complex detector field}."""
        out = {}
        for k, src in enumerate(self.sources):
            ph = np.exp(1j * self.cell.k0 * self.delta_L(k, pert))
            for p, (G, rid) in self.tables[k]["G"].items():
                out[(src.port, p)] = G @ ph[rid]
        return out
