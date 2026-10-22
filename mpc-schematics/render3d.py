"""Minimal z-buffered renderer for schematic 3D figures (numpy + Pillow + shapely + matplotlib only).

Orthographic camera, Lambert shading, planar polygons with holes, prisms extruded from shapely polygons, face outlines
from the face-id buffer, depth-tested polylines, and 2x supersampling. Enough for clean block diagrams of layered
devices (membranes, trenches, cutaways) where matplotlib's mplot3d painter's algorithm draws faces in the wrong order.

    sc = Scene(azim=-50, elev=32)
    sc.prism(shapely_polygon, z0, z1, color="#7d93b8", section=lambda p, q: ...)   # section(p, q): edge on a cut plane?
    sc.line(points_xyz, color="#d03a2a", width=3)
    img, to_px = sc.render(width=1600)                     # RGB float image, function xyz -> (px, py) in that image
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient


class Scene:
    def __init__(self, azim=-50.0, elev=32.0, light=(-0.45, -0.25, 1.0), ambient=0.42, edge_color="#1b2328", edge_alpha=0.75):
        a, e = np.radians(azim), np.radians(elev)
        self.d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])      # toward the camera
        self.right = np.cross([0.0, 0.0, 1.0], self.d)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.d, self.right)
        L = np.asarray(light, float)
        self.light = L / np.linalg.norm(L)
        self.ambient = ambient
        self.edge_color = np.array(to_rgb(edge_color))
        self.edge_alpha = edge_alpha
        self.faces = []        # (rings [exterior, holes...] as (n, 3) arrays, normal, rgb, group)
        self.lines = []        # (points (n, 3), rgb, width_px, depth_tested)
        self.glows = []        # (origin, U, V, alpha(s, t) on [0, 1]^2, rgb): translucent overlays, depth tested

    # ------------------------------------------------------------------ geometry
    def face(self, rings, color, normal=None, group=None):
        rings = [np.asarray(r, float) for r in rings]
        if normal is None:
            P = rings[0]
            n = np.zeros(3)
            for i in range(len(P)):                          # Newell's method
                p, q = P[i], P[(i + 1) % len(P)]
                n += np.array([(p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]), (p[0] - q[0]) * (p[1] + q[1])])
            normal = n
        normal = np.asarray(normal, float)
        nn = np.linalg.norm(normal)
        if nn == 0:
            return
        self.faces.append((rings, normal / nn, np.array(to_rgb(color)), group))

    def prism(self, poly, z0, z1, color, section=None, section_color=None, top_color=None, group=None):
        """Extrude a shapely (Multi)Polygon from z0 to z1. Walls whose edge satisfies section(p, q) get section_color
        (cut faces of a cutaway)."""
        geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for g in geoms:
            if g.is_empty or g.area == 0:
                continue
            g = orient(g, 1.0)                               # exterior CCW, holes CW: (dy, -dx) is the outward normal
            rings2 = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in g.interiors]
            self.face([np.c_[r, np.full(len(r), z1)] for r in rings2], top_color or color, (0, 0, 1), group)
            self.face([np.c_[r, np.full(len(r), z0)] for r in rings2], color, (0, 0, -1), group)
            for r in rings2:
                for i in range(len(r)):
                    p, q = r[i], r[(i + 1) % len(r)]
                    if np.allclose(p, q):
                        continue
                    sec = section is not None and section(p, q)
                    quad = [[p[0], p[1], z0], [q[0], q[1], z0], [q[0], q[1], z1], [p[0], p[1], z1]]
                    self.face([quad], (section_color or color) if sec else color, (q[1] - p[1], -(q[0] - p[0]), 0), group)

    def glow(self, origin, U, V, alpha, color="#ff2a1a"):
        """Translucent parallelogram origin + s U + t V (s, t in [0, 1]) with opacity alpha(s, t) (vectorised), drawn
        over the faces where it is not hidden behind them; e.g. a guided-mode intensity on a cut face."""
        self.glows.append((np.asarray(origin, float), np.asarray(U, float), np.asarray(V, float), alpha, np.array(to_rgb(color))))

    def line(self, pts, color, width=2.0, depth_test=True):
        self.lines.append((np.asarray(pts, float), np.array(to_rgb(color)), width, depth_test))

    # ------------------------------------------------------------------ projection
    def project(self, P):
        P = np.asarray(P, float)
        return P @ self.right, P @ self.up, P @ self.d

    def render(self, width=1600, margin=0.04, ss=2, background="#ffffff", extra_points=()):
        allp = np.vstack([np.vstack(f[0]) for f in self.faces] + [l[0] for l in self.lines] + [np.zeros((0, 3))] +
                         [np.asarray(extra_points, float).reshape(-1, 3)])
        u, v, _ = self.project(allp)
        span = max(u.max() - u.min(), 1e-12)
        W = int(width * ss)
        s = W * (1 - 2 * margin) / span
        H = int(np.ceil((v.max() - v.min()) * s + 2 * margin * W))
        u0, v1 = u.min() - margin * W / s, v.max() + margin * W / s

        def to_px(P, scale=1.0):
            uu, vv, ww = self.project(P)
            return (uu - u0) * s * scale, (v1 - vv) * s * scale, ww

        img = np.empty((H, W, 3))
        img[:] = to_rgb(background)
        zb = np.full((H, W), -np.inf)
        ib = np.full((H, W), -1, int)
        for fid, (rings, n, col, _) in enumerate(self.faces):
            nd = n @ self.d
            if nd <= 1e-9:                                   # back face or edge-on
                continue
            pr = [np.c_[to_px(r)[:2]] for r in rings]
            x0, y0 = np.floor(pr[0].min(0)).astype(int)
            x1, y1 = np.ceil(pr[0].max(0)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            m = Image.new("L", (x1 - x0, y1 - y0), 0)
            dr = ImageDraw.Draw(m)
            dr.polygon([tuple(p) for p in pr[0] - [x0, y0]], fill=1)
            for h in pr[1:]:
                dr.polygon([tuple(p) for p in h - [x0, y0]], fill=0)
            mask = np.asarray(m, bool)
            if not mask.any():
                continue
            c = n @ rings[0][0]                              # plane n·p = c, depth along d at pixel (x, y)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            uu, vv = xx / s + u0, v1 - yy / s
            w = (c - uu * (n @ self.right) - vv * (n @ self.up)) / nd
            sub = zb[y0:y1, x0:x1]
            upd = mask & (w > sub + 1e-9 * span)
            sub[upd] = w[upd]
            shade = self.ambient + (1 - self.ambient) * max(n @ self.light, 0.0)
            img[y0:y1, x0:x1][upd] = np.clip(col * shade, 0, 1)
            ib[y0:y1, x0:x1][upd] = fid
        # translucent overlays (after the faces, before the outlines)
        for O, U, V, alpha, col in self.glows:
            corners = np.array([O, O + U, O + U + V, O + V])
            pr = np.c_[to_px(corners)[:2]]
            x0, y0 = np.maximum(np.floor(pr.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(pr.max(0)).astype(int) + 1, [W, H])
            if x1 <= x0 or y1 <= y0:
                continue
            m = Image.new("L", (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(m).polygon([tuple(p) for p in pr - [x0, y0]], fill=1)
            mask = np.asarray(m, bool)
            ox, oy, ow = to_px(O[None])
            ux, uy, uw = (np.ravel(c) - np.ravel(o) for c, o in zip(to_px((O + U)[None]), (ox, oy, ow)))
            vx, vy, vw = (np.ravel(c) - np.ravel(o) for c, o in zip(to_px((O + V)[None]), (ox, oy, ow)))
            det = ux * vy - uy * vx
            if abs(det) < 1e-12:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx - ox[0], yy - oy[0]
            gs, gt = (dx * vy - dy * vx) / det, (ux * dy - uy * dx) / det
            w = ow[0] + gs * uw + gt * vw
            vis = mask & (w >= zb[y0:y1, x0:x1] - 2e-3 * span)
            a = np.clip(alpha(np.clip(gs, 0, 1), np.clip(gt, 0, 1)), 0, 1) * vis
            sub = img[y0:y1, x0:x1]
            sub[:] = sub * (1 - a[..., None]) + col * a[..., None]
        # outlines: boundaries of the face-id buffer
        e = np.zeros((H, W), bool)
        e[:, 1:] |= ib[:, 1:] != ib[:, :-1]
        e[1:, :] |= ib[1:, :] != ib[:-1, :]
        e = e & ((ib >= 0) | np.roll(ib >= 0, 1, 0) | np.roll(ib >= 0, 1, 1))
        if ss > 1:
            e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
        img[e] = img[e] * (1 - self.edge_alpha) + self.edge_color * self.edge_alpha
        # polylines, depth tested against the faces
        for pts, col, width, dt in self.lines:
            for a, b in zip(pts[:-1], pts[1:]):
                n_s = int(max(np.linalg.norm(np.subtract(to_px(b)[:2], to_px(a)[:2])), 1) * 1.5) + 1
                P = a + np.linspace(0, 1, n_s)[:, None] * (b - a)
                px, py, pw = to_px(P)
                xi, yi = np.round(px).astype(int), np.round(py).astype(int)
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                xi, yi, pw = xi[ok], yi[ok], pw[ok]
                if dt:
                    vis = pw >= zb[yi, xi] - 2e-3 * span
                    xi, yi = xi[vis], yi[vis]
                r = int(round(width * ss / 2))
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r + r:
                            X, Y = np.clip(xi + dx, 0, W - 1), np.clip(yi + dy, 0, H - 1)
                            img[Y, X] = col
        if ss > 1:
            H2, W2 = H // ss, W // ss
            img = img[: H2 * ss, : W2 * ss].reshape(H2, ss, W2, ss, 3).mean((1, 3))
        return img, (lambda P: to_px(P, 1 / ss)[:2])


def ngon(apothem, n, rot=0.0):
    """Regular n-gon with the given apothem, facet k centred at angle rot + 2πk/n (cmpc_sim.SegmentedCell convention)."""
    r = apothem / np.cos(np.pi / n)
    a = rot + 2 * np.pi * np.arange(n) / n - np.pi / n
    return Polygon(np.c_[r * np.cos(a), r * np.sin(a)])
