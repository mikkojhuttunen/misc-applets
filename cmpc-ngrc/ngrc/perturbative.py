"""First-order (phase-screen) model: the unperturbed rays and beamlets are traced once; a perturbation only adds
the optical path ΔL = ∫ Δn ds along the unperturbed chords travelled before each exit, so E = G @ exp(i k0 ΔL).

Exact for the "straight" tracer mode (same physics, quadrature instead of RK4), and the Δn → 0 limit of the full
curved-ray result. It ignores ray bending; the bending matters once the deflection × remaining path exceeds the
beamlet width (see analysis.validity_scan). With soft ports a ray can leave several times; every exit record gets
the path sum over the chords before it."""
from __future__ import annotations

import numpy as np

from .field import _frozen_at, beamlet_matrix, propagation_matrix
from .rays import Source, trace


class PhaseScreenModel:
    def __init__(self, cell, sources, max_bounces=400, amp_min=1e-3, prune=0.02):
        """prune (fga): drop exit records whose amplitude at the port is below prune × the largest one."""
        self.cell = cell
        self.sources = [s if isinstance(s, Source) else Source(int(s)) for s in sources]
        self.outputs = cell.outputs
        self.tables = []
        for src in self.sources:
            ex = trace(cell, src, None, mode="none", max_bounces=max_bounces, amp_min=amp_min, record_chords=True)
            if ex.model == "fga" and prune:
                w = ex.weights(cell.k0)
                ex = ex.subset(w > prune * w.max())
            ch = ex.chords
            ray = ch["ray"].astype(int)
            order = np.argsort(ray, kind="stable")              # chords of each ray, in travel order
            start = np.zeros(ex.n_launched + 1, int)
            np.add.at(start, ray + 1, 1)
            start = np.cumsum(start)                              # first sorted chord of each ray
            pick = order[start[ex.ray] + ex.nch - 1]              # last chord before each exit (sorted → original)
            if ex.model == "fga":                                 # E = K (B · phase): aperture field, then detector
                G = {}
                for p in self.outputs:
                    idx = ex.index(p)
                    ap, du = cell.aperture_points(p)
                    B = _frozen_at(cell, ex, ex.subset(ex.port == p).select(p), ap, p).astype(np.complex64)
                    G[p] = (("KB", propagation_matrix(cell, p, ap, du), B), idx)
            else:
                G = {p: (beamlet_matrix(cell, ex, p)[0], ex.index(p)) for p in self.outputs}
            t = dict(exits=ex, chords=ch, G=G, n_rays=ex.n_launched, order=order, start=start, ray=ray,
                     exit_ray=ex.ray, exit_nch=ex.nch)
            t["path"] = self._per_exit(t, ch["len"])
            self.tables.append(t)

    @staticmethod
    def _per_exit(t, per_chord):
        """Sum of a per-chord quantity over the chords each exit record travelled."""
        cs = np.cumsum(per_chord[t["order"]])
        cs = np.concatenate([[0.0], cs])
        first = t["start"][t["exit_ray"]]
        return cs[first + t["exit_nch"]] - cs[first]

    def delta_L_chords(self, k, pert, step=None, chunk=400000):
        """ΔL on every chord of source k (m). Dots (Shape): every chord crossing the bounding circle reads the dot's
        Radon table (Shape.radon: line integral vs direction and offset, |k0 ΔL| errors ≲ 2e-3 rad). Images: midpoint
        quadrature with step ≤ the blur (the integrand is smooth and vanishes at both ends: spectral convergence)."""
        ch = self.tables[k]["chords"]
        dc = np.zeros(len(ch["x"]))
        if pert is None or len(pert) == 0:
            return dc
        x, y, tx, ty, ln = ch["x"], ch["y"], ch["tx"], ch["ty"], ch["len"]
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
            if hasattr(comp, "radon"):                            # every crossing from the dot's Radon table
                tab = comp.radon()
                dc[idx] += comp.radon_lookup(tab, cx, cy, x[idx], y[idx], tx[idx], ty[idx])
                continue
            if hasattr(comp, "chord_integral"):                   # grazing chords: smooth along the chord
                g, wg = np.polynomial.legendre.leggauss(32)
                g, wg = (g + 1) / 2, wg / 2
            else:                                                 # images: midpoint rule, step ≤ the blur
                st = min(3e-6, getattr(comp, "smooth", 2e-6) or 2e-6) if step is None else step
                nq = max(8, int(np.ceil(np.max(s2[idx] - s1[idx]) / st)) + 1)
                g, wg = (np.arange(nq) + 0.5) / nq, np.full(nq, 1.0 / nq)
            nq = len(g)
            for c0 in range(0, len(idx), max(1, chunk // nq)):
                j = idx[c0:c0 + max(1, chunk // nq)]
                seg = s2[j] - s1[j]
                s = s1[j][:, None] + g[None, :] * seg[:, None]
                v = comp.eval(x[j][:, None] + s * tx[j][:, None], y[j][:, None] + s * ty[j][:, None], hess=False)[0]
                dc[j] += (v * wg[None, :]).sum(axis=1) * seg
        return dc

    def delta_L(self, k, pert, **kw):
        """ΔL per exit record of source k (m)."""
        return self._per_exit(self.tables[k], self.delta_L_chords(k, pert, **kw))

    def fields(self, pert, dn_global=0.0):
        """{(input port, output port): complex detector field}. dn_global: a uniform change of the slab index
        (temperature drift, dn/dT ≈ 2e-5 /K for SiN), which adds dn_global × geometric path to every exit."""
        out = {}
        for k, src in enumerate(self.sources):
            t = self.tables[k]
            ph = np.exp(1j * self.cell.k0 * (self.delta_L(k, pert) + dn_global * t["path"]))
            for p, (G, idx) in t["G"].items():
                if isinstance(G, tuple):
                    _, K, B = G
                    out[(src.port, p)] = K @ (B @ ph[idx].astype(np.complex64)).astype(complex)
                else:
                    out[(src.port, p)] = G @ ph[idx]
        return out
