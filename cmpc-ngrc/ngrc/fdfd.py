"""Full-wave reference: 2D scalar Helmholtz (finite differences, frequency domain) for a small cell.

    ∇²E + k0² n(x, y)² E = −J

The slab has index n0 plus the dot Δn(x, y). The wall is a thin lossy metal layer (ε_m, chosen for the wall
reflectance) with gaps at the ports; outside the wall the slab continues into a stretched-coordinate PML. Each
input port is driven by a Gaussian line current across its opening with a linear phase for the launch angle. The
detector is the far field of the field across each opening (the same kernel as the ray engine).

The empty cell is factorised once (MKL PARDISO through pypardiso, real 2N form). A dot only changes the operator
on its own grid cells, so the field with a dot is found by GMRES on those cells:
    z = x0|d − [A0⁻¹ (ΔA z)]|d,   x = x0 − A0⁻¹ (ΔA z),
one sparse solve per iteration. Second-order stencil: about 12 points per wavelength in the slab; numerical
dispersion makes the speckle differ from the continuum one but leaves a consistent wave system (diffraction at
corners and port edges, ray bending and multiple scattering in the dot are all included).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

def _find_mkl():
    import glob
    import sys
    for d in [os.path.join(sys.prefix, "lib"), "/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu"]:
        hits = sorted(glob.glob(os.path.join(d, "libmkl_rt.so*")))
        if hits:
            return hits[0]
    return None


if "PYPARDISO_MKL_RT" not in os.environ and _find_mkl():
    os.environ["PYPARDISO_MKL_RT"] = _find_mkl()


def metal_eps(n0, R, eps_re=-40.0):
    """ε_m = eps_re + i b with normal-incidence reflectance R from the slab (bisection on b)."""
    lo, hi = 0.0, 200.0
    for _ in range(80):
        b = 0.5 * (lo + hi)
        r = (n0 - np.sqrt(eps_re + 1j * b)) / (n0 + np.sqrt(eps_re + 1j * b))
        lo, hi = (b, hi) if abs(r) ** 2 > R else (lo, b)
    return eps_re + 1j * 0.5 * (lo + hi)


@dataclass
class WaveCell:
    radius: float = 25e-6               # circle radius or polygon circumradius
    facets: int = 0                     # 0: smooth circle; n: regular n-gon (facet 0 centred at angle 0)
    n_eff: float = 1.8
    wavelength: float = 1.55e-6
    reflectance: float = 0.97
    port_angles: tuple = (0, 67, 151, 238)   # degrees, polar angle of the port centre
    inputs: tuple = (0, 1)
    port_width: float = 1.5e-6
    launch: float = 0.0                 # rad, from the inward normal
    source_width: float = None          # Gaussian 1/e half-width of the line current (default 0.35 port width)
    ppw: float = 12.0                   # grid points per wavelength in the slab
    wall: float = 0.4e-6
    gap: float = 1.0e-6                 # slab between the wall and the PML
    pml: float = 2.0e-6
    n_pix: int = 64
    max_sin: float = 0.9
    _ready: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ geometry
    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    def boundary_distance(self, x, y):
        """Signed distance to the wall line (negative inside)."""
        if self.facets == 0:
            return np.hypot(x, y) - self.radius
        n = self.facets
        a = self.radius * np.cos(np.pi / n)
        th = 2 * np.pi * np.arange(n) / n
        return np.max(x[..., None] * np.cos(th) + y[..., None] * np.sin(th), axis=-1) - a

    def port_frame(self, i):
        """Port centre on the wall line, outward normal, tangent."""
        ph = np.deg2rad(self.port_angles[i])
        if self.facets == 0:
            nrm = np.array([np.cos(ph), np.sin(ph)])
            return self.radius * nrm, nrm, np.array([-nrm[1], nrm[0]])
        n = self.facets
        k = int(np.round(ph / (2 * np.pi / n))) % n
        nrm = np.array([np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)])
        a = self.radius * np.cos(np.pi / n)
        d = np.array([np.cos(ph), np.sin(ph)])
        c = d * a / (d @ nrm)
        return c, nrm, np.array([-nrm[1], nrm[0]])

    # ------------------------------------------------------------------ assembly
    def setup(self):
        import pypardiso
        n0, k0 = self.n_eff, self.k0
        h = self.wavelength / (n0 * self.ppw)
        L = self.radius + self.wall + self.gap + self.pml
        N = int(np.ceil(2 * L / h))
        self.h, self.N, self.L = h, N, N * h / 2
        x = -self.L + (np.arange(N) + 0.5) * h
        self.x = x
        X, Y = np.meshgrid(x, x, indexing="xy")          # Y rows, X columns; index = iy * N + ix
        d = self.boundary_distance(X, Y)
        eps = np.full(X.shape, n0 ** 2, complex)
        wall = (d >= 0) & (d < self.wall)
        for i in range(len(self.port_angles)):
            c, nrm, t = self.port_frame(i)
            u = (X - c[0]) * t[0] + (Y - c[1]) * t[1]
            v = (X - c[0]) * nrm[0] + (Y - c[1]) * nrm[1]
            wall &= ~((np.abs(u) < self.port_width / 2) & (v > -h) & (v < self.wall + h))
        self.eps_m = metal_eps(n0, self.reflectance)
        eps[wall] = self.eps_m
        self.eps, self.wall_mask = eps, wall
        # stretched-coordinate PML
        def stretch(xx):
            dpml = np.clip(np.abs(xx) - (self.L - self.pml), 0, None) / self.pml
            return 1 + 1j * 6.0 * dpml ** 3
        xc, xf = x, x + h / 2
        sc, sf = stretch(xc), stretch(xf)
        e = np.ones(N)
        Df = sp.diags([-e, e[:-1]], [0, 1], shape=(N, N)) / h
        D1 = sp.diags(1 / sc) @ (-Df.T) @ sp.diags(1 / sf) @ Df
        I = sp.identity(N)
        self.A0 = (sp.kron(I, D1) + sp.kron(D1, I) + sp.diags(k0 ** 2 * eps.ravel())).tocsr()
        self._solver = pypardiso.PyPardisoSolver()
        self._Ar = sp.bmat([[self.A0.real, -self.A0.imag], [self.A0.imag, self.A0.real]]).tocsr() * h ** 2
        self._solver.factorize(self._Ar)
        self._sources = {i: self._source(i) for i in self.inputs}
        self._x0 = {i: self.solve(self._sources[i]) for i in self.inputs}
        self._ready = True
        return self

    def solve(self, b):
        """A0 x = b (complex; several right-hand sides as columns)."""
        Nn = self.N * self.N
        br = np.concatenate([b.real, b.imag]) * self.h ** 2
        xr = self._solver.solve(self._Ar, br)
        return xr[:Nn] + 1j * xr[Nn:]

    def _line(self, i, n_samp=None):
        c, nrm, t = self.port_frame(i)
        n_samp = n_samp or max(8, int(np.ceil(self.port_width / (self.wavelength / self.n_eff / 3))) + 1)
        du = self.port_width / n_samp
        u = (np.arange(n_samp) + 0.5) * du - self.port_width / 2
        pts = c[None, :] + u[:, None] * t[None, :]
        return pts, u, du, nrm, t

    def _interp(self, pts):
        """Sparse bilinear interpolation matrix (n_pts × N²)."""
        fx = (pts[:, 0] + self.L) / self.h - 0.5
        fy = (pts[:, 1] + self.L) / self.h - 0.5
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - ix, fy - iy
        rows, cols, vals = [], [], []
        for dx, dy, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
            rows.append(np.arange(len(pts)))
            cols.append((iy + dy) * self.N + ix + dx)
            vals.append(w)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(pts), self.N * self.N))

    def _source(self, i):
        pts, u, du, nrm, t = self._line(i, n_samp=4 * int(np.ceil(self.port_width / self.h)))
        w = self.source_width or 0.35 * self.port_width
        # the beam travels along -nrm rotated by `launch`: phase k0 n0 sin(launch) along the tangent
        J = np.exp(-(u / w) ** 2) * np.exp(-1j * self.k0 * self.n_eff * np.sin(self.launch) * u) * du
        return np.asarray(self._interp(pts).T @ J).ravel() / self.h ** 2

    def detector_operators(self):
        """Per output port: far-field operator (n_pix × N²) = K · interpolation onto the opening."""
        if hasattr(self, "_det"):
            return self._det
        k = self.k0 * self.n_eff
        sns = np.linspace(-self.max_sin, self.max_sin, self.n_pix)
        self._det = {}
        for o in range(len(self.port_angles)):
            pts, u, du, nrm, t = self._line(o)
            cs = np.sqrt(1 - sns ** 2)
            dirs = cs[:, None] * nrm[None, :] + sns[:, None] * t[None, :]
            K = np.sqrt(k / (2 * np.pi)) * du * cs[:, None] * np.exp(-1j * (k * dirs @ pts.T + np.pi / 4))
            self._det[o] = (K, self._interp(pts))
        return self._det

    def fields(self, pert=None, dn_global=0.0, tol=1e-6, maxiter=200, born=False):
        """{(input, output): far field (n_pix,)} for a Perturbation (or None). born: first-order Born field."""
        if not self._ready:
            self.setup()
        det = self.detector_operators()
        k02 = self.k0 ** 2
        out, info = {}, {}
        if pert is None and dn_global == 0:
            xs = self._x0
        else:
            X, Y = np.meshgrid(self.x, self.x, indexing="xy")
            dn = np.zeros(X.shape)
            if pert is not None:
                for c in pert.components:
                    r = c.bound if hasattr(c, "bound") else None
                    m = (np.hypot(X - c.x0, Y - c.y0) < r) if r is not None else np.ones(X.shape, bool)
                    dn[m] += c.eval(X[m], Y[m], hess=False)[0] if hasattr(c, "eval") else c.value(X[m], Y[m])
            inside = self.boundary_distance(X, Y) < 0
            dn = np.where(inside, dn + dn_global, 0.0)
            dA = (k02 * ((self.n_eff + dn) ** 2 - self.n_eff ** 2)).ravel()
            idx = np.nonzero(dA)[0]
            dAd = dA[idx]
            xs = {}
            for i in self.inputs:
                x0 = self._x0[i]
                if born:
                    v = np.zeros(self.N * self.N, complex)
                    v[idx] = dAd * x0[idx]
                    xs[i] = x0 - self.solve(v)
                    continue

                def mv(z):
                    v = np.zeros(self.N * self.N, complex)
                    v[idx] = dAd * z
                    return z + self.solve(v)[idx]
                Aop = sla.LinearOperator((len(idx), len(idx)), matvec=mv, dtype=complex)
                it = [0]
                z, flag = sla.gmres(Aop, x0[idx], rtol=tol, restart=40, maxiter=maxiter,
                                    callback=lambda r: it.__setitem__(0, it[0] + 1), callback_type="pr_norm")
                v = np.zeros(self.N * self.N, complex)
                v[idx] = dAd * z
                xs[i] = x0 - self.solve(v)
                info[i] = (it[0], flag)
        for i in self.inputs:
            for o, (K, P) in det.items():
                out[(i, o)] = K @ (P @ xs[i])
        self.last_info = info
        return out
