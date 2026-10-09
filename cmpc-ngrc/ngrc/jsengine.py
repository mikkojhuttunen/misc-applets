"""Run the JS port of the engine (web/ngrc-engine.js) from Python through node, on all CPU cores.

The JS tracer is the same algorithm as rays.py (checked by tools/check_js.mjs) and ≈ 20× faster per ray, so
curved-ray datasets of thousands of samples are practical. Analytic Shape dots only (no GridIndex images) and a
constant wall reflectance."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "node_fields.mjs"


def available():
    return shutil.which("node") is not None


def node_fields(cell, perts, n_pos=None, n_ang=2000, waist=5e-6, mode="curved", workers=None, ds=None, chunk=None,
                frozen=20e-6, prune=0.02, amp_min=None, max_bounces=None, dn_global=None):
    """Detector fields for every perturbation: list of {(input, output): complex array (n_pix,)}.
    mode: "curved", "straight", "none" or "phase" (JS PhaseScreen: one unperturbed trace per worker, then
    ΔL from the dots' Radon tables). dn_global: optional per-sample uniform index change (phase mode)."""
    if cell.reflectance_fn is not None:
        raise ValueError("the JS engine takes a constant wall reflectance only")
    samples = []
    for i, p in enumerate(perts):
        comps = [] if p is None else p.components
        for c in comps:
            if not hasattr(c, "to_dict") or c.to_dict().get("kind") != "shape":
                raise ValueError("the JS engine takes analytic Shape dots only")
        sh = [c.to_dict() for c in comps]
        samples.append(dict(shapes=sh, dn=float(dn_global[i])) if dn_global is not None else sh)
    opt = dict(mode=mode, nPos=n_pos, nAng=n_ang, waist=waist, frozen=frozen, prune=prune)
    for k, v in (("ds", ds), ("ampMin", amp_min), ("maxBounces", max_bounces)):
        if v is not None:
            opt[k] = v
    out = []
    chunk = len(samples) if chunk is None else chunk
    with tempfile.TemporaryDirectory() as td:
        for c0 in range(0, len(samples), max(1, chunk)):
            job = dict(cell=cell.to_dict(), opt=opt, samples=samples[c0:c0 + chunk])
            if workers:
                job["workers"] = int(workers)
            jp, op = Path(td) / "job.json", Path(td) / "out.bin"
            jp.write_text(json.dumps(job))
            subprocess.run(["node", str(SCRIPT), str(jp), str(op)], check=True)
            meta = json.loads(Path(str(op) + ".json").read_text())
            raw = np.fromfile(op, dtype="<f8")
            keys = [tuple(k) for k in meta["keys"]]
            arr = raw.reshape(meta["n_samples"], len(keys), meta["n_pix"], 2)
            E = arr[..., 0] + 1j * arr[..., 1]
            out += [{k: E[s, j] for j, k in enumerate(keys)} for s in range(meta["n_samples"])]
    return out
