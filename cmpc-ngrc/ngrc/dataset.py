"""Random shape ensembles → (labels, speckle features), with a JSON config for exact regeneration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .features import stack_intensities
from .harmonics import power, shape_chd
from .rays import Source, trace
from .shapes import Perturbation, random_shape


@dataclass
class Ensemble:
    n: int = 500
    m_max: int = 6
    n_shapes: int = 1
    R_range: tuple = (50e-6, 120e-6)
    dn_range: tuple = (1e-3, 1e-3)
    edge: float = 3e-6
    sigma: float = 0.08
    decay: float = 0.5
    centre: tuple = None          # fixed centre (x, y) or None = random
    place_radius: float = None    # max centre distance from the cell centre
    rotation: bool = False
    jitter: float = 0.0           # with a fixed centre: uniform random offset within this radius (m)
    seed: int = 0

    def key(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:12]


def sample_shapes(cell, ens: Ensemble):
    rng = np.random.default_rng(ens.seed)
    out = []
    for _ in range(ens.n):
        shapes = [random_shape(rng, cell.radius, ens.m_max, ens.R_range, ens.dn_range, ens.edge, ens.sigma,
                               ens.decay, ens.centre, ens.place_radius, rotation=ens.rotation)
                  for _ in range(ens.n_shapes)]
        if ens.jitter > 0:
            for s in shapes:
                r, ph = ens.jitter * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
                s.x0 += r * np.cos(ph)
                s.y0 += r * np.sin(ph)
        out.append(Perturbation(shapes))
    return out


def labels(perts, m_max):
    """Per sample (first shape): R, a_2..a_M, b_2..b_M, p_2..p_M and the dominant order."""
    rows = []
    for p in perts:
        R, a, b = shape_chd(p.components[0], m_max)
        pw = power(a, b)
        rows.append(np.concatenate([[R], a[1:], b[1:], pw[1:], [np.argmax(pw[1:]) + 2]]))
    rows = np.array(rows)
    m = np.arange(2, m_max + 1)
    names = ["R"] + [f"a{k}" for k in m] + [f"b{k}" for k in m] + [f"p{k}" for k in m] + ["m_dom"]
    return rows, names


def build_multiplexed(cell, ens: Ensemble, variants, engine="phase", feature_mode="relative", **kw):
    """Concatenate the features of several cell variants (dicts for CircularCell.variant: wavelength,
    launch_offset, launch), e.g. launch-angle or wavelength steps used as extra virtual nodes."""
    parts, D = [], None
    for v in variants:
        D = build(cell.variant(**v), ens, engine=engine, feature_mode=feature_mode, **kw)
        parts.append(D["X"])
    D = dict(D)
    D["X"] = np.hstack(parts)
    D["variants"] = list(variants)
    D.pop("fields", None)
    return D


def build(cell, ens: Ensemble, engine="phase", model=None, sources=None, feature_mode="relative", progress=None,
          **trace_kw):
    """Features for every sample. engine: "phase" (PhaseScreenModel, fast), "curved" (Python tracer) or
    "node-curved" / "node-phase" (the JS engine through node on all cores; trace_kw: n_pos, n_ang, waist, workers)."""
    perts = sample_shapes(cell, ens)
    if engine.startswith("node-"):
        from .jsengine import node_fields
        F = node_fields(cell, [None] + perts, mode=engine[5:], **trace_kw)
        ref = F[0]
        X = np.array([stack_intensities(f, reference=ref, mode=feature_mode) for f in F[1:]])
        Y, names = labels(perts, ens.m_max)
        return dict(X=X, Y=Y, names=names, perts=perts, reference=ref, ensemble=asdict(ens), key=ens.key(),
                    fields=F[1:])
    Y, names = labels(perts, ens.m_max)
    sources = [Source(i) for i in cell.inputs] if sources is None else sources
    if engine == "phase":
        from .perturbative import PhaseScreenModel
        model = PhaseScreenModel(cell, sources) if model is None else model
        ref = model.fields(None)
        fieldfn = model.fields
    else:
        from .field import detector_field
        def fieldfn(p):
            out = {}
            for s in sources:
                ex = trace(cell, s, p, mode=engine, **trace_kw)
                for o in cell.outputs:
                    out[(s.port, o)] = detector_field(cell, ex, o)
            return out
        ref = fieldfn(None)
    X = []
    for i, p in enumerate(perts):
        X.append(stack_intensities(fieldfn(p), reference=ref, mode=feature_mode))
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{len(perts)}", flush=True)
    return dict(X=np.array(X), Y=Y, names=names, perts=perts, reference=ref, ensemble=asdict(ens), key=ens.key())
