"""Cascaded THG design: double matching of ω+ω→2ω (Δk_a) and ω+2ω→3ω (Δk_b) with one poling mask.

The cascade amplitude is a nested integral (math-engines cwe_cascade), not one Fourier transform, so a mask has
to supply grating content near both K_a = Δk_a(λ0) and K_b = Δk_b(λ0) *and* order them sensibly in z. Designs:

  two_section    SHG grating on [0, sL], SFG grating on the rest; undepleted optimum s = 1/2,
                 P3 = (2/π)⁴ κ3² κa² P³ L⁴/16
  superlattice   g = sign(w cos K_a z + (1 - w) cos(K_b z + ψ)): both processes all along the device; the
                 cascade then grows as ∫ z dz (L²/2 instead of (L/2)² per amplitude), at reduced Fourier weights
  commensurate   one period with orders (m_a, m_b) when Δk_b/Δk_a is close to a ratio of small integers
  blend          superlattice weight w(z) sweeping from SHG-rich to SFG-rich along z (generalises both)

thg_spectrum gives η_THG(λ) = P3/P³ (undepleted, κ per λ); thg_devices applies one mask to many devices of a
ThicknessFamily of THGModel (both Δk move with the film thickness).
"""
from __future__ import annotations

import math

import numpy as np

from engines.cwe_cascade import engine as casc

from . import poling
from .poling import DEFAULT_FAB, FAB_PRESETS, Poling


def _k(model, lam0):
    return abs(float(model.dk_a(lam0))), abs(float(model.dk_b(lam0)))


def _fab(p: Poling, fab):
    if fab is None:
        return p
    f = FAB_PRESETS[fab]
    return poling.fabricate(p, f["l_min"], f["grid"])


def two_section(model, lam0, length, split=0.5, fab=DEFAULT_FAB) -> Poling:
    ka, kb = _k(model, lam0)
    b, g = casc.two_section_domains(length, 2 * np.pi / ka, 2 * np.pi / kb, split)
    return _fab(Poling(b, g), fab)


def superlattice(model, lam0, length, weight=0.5, phase=0.0, fab=DEFAULT_FAB, points_per_domain=40) -> Poling:
    """sign(w(z) cos K_a z + (1 - w(z)) cos(K_b z + ψ)); weight may be a callable w(z) (blend)."""
    ka, kb = _k(model, lam0)
    z = np.linspace(0.0, length, int(points_per_domain * length * max(ka, kb) / np.pi) + 2)
    w = weight(z) if callable(weight) else weight
    return _fab(poling.from_sign(z, w * np.cos(ka * z) + (1 - w) * np.cos(kb * z + phase)), fab)


def blend(model, lam0, length, w_start=0.8, w_end=0.2, phase=0.0, fab=DEFAULT_FAB) -> Poling:
    """Superlattice whose SHG weight falls linearly along z: 2ω is made early and converted later."""
    return superlattice(model, lam0, length, lambda z: w_start + (w_end - w_start) * z / length, phase, fab)


def commensurate(model, lam0, length, max_order=7, fab=DEFAULT_FAB):
    """Best single-period higher-order design; returns (Poling, row of cwe_cascade.commensurate_orders)."""
    ka, kb = _k(model, lam0)
    rows = casc.commensurate_orders(ka, kb, max_order)
    # rank by the cascade figure of merit including the residual mismatch of the SFG step
    def score(r):
        x = r["residual"] * length / 2
        return r["strength"] ** 2 * (np.sinc(x / np.pi) ** 2)
    best = max(rows, key=score)
    p = poling.ordered(length, best["period"], best["duty_cycle"])
    return _fab(p, fab), best


def thg_spectrum(model, p: Poling, lam, power=1.0, delta_k=None, alpha=(0.0, 0.0, 0.0)):
    """Undepleted η_THG(λ) = P3/P³ (1/W²) and η_SHG(λ) = P2/P² (1/W); delta_k: optional (2, M) override."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    dka, dkb = (model.dk_a(lam), model.dk_b(lam)) if delta_k is None else delta_k
    c = [model.coupling(x) for x in lam]
    ka = np.array([float(x["kappa_a"]) for x in c])
    k3 = np.array([float(x["kappa_3"]) for x in c])
    r = casc.cascade_undepleted(p.boundaries, p.g, np.abs(dka), np.abs(dkb), ka, k3, power, alpha)
    return r["P_th"] / power**3, r["P_sh"] / power**2


def optimise_superlattice(model, lam0, length, weights=np.linspace(0.2, 0.8, 13), phases=np.linspace(0, np.pi, 7),
                          fab=DEFAULT_FAB):
    """Grid search of (w, ψ) for the superlattice at λ0; returns (best Poling, w, ψ, η_THG)."""
    best = (None, None, None, -1.0)
    for w in weights:
        for ph in phases:
            p = superlattice(model, lam0, length, w, ph, fab)
            eta = float(thg_spectrum(model, p, [lam0])[0][0])
            if eta > best[3]:
                best = (p, float(w), float(ph), eta)
    return best


def reference_efficiencies(model, lam0, length) -> dict:
    """Ideal references at λ0 (1/W²): both processes matched at full strength in a uniform medium (L⁴/4), and
    the two-section closed form with first-order factors (2/π)⁴ L⁴/16."""
    c = model.coupling(lam0)
    ka, k3 = float(c["kappa_a"]), float(c["kappa_3"])
    return {"matched_uniform": (k3 * ka) ** 2 * length**4 / 4,
            "two_section_closed": (2 / np.pi) ** 4 * (k3 * ka) ** 2 * length**4 / 16}


def thg_devices(family, p: Poling, lam, lam0, n_devices=100, seed=0):
    """η_THG(λ) of one mask on n_devices thickness samples (κ of the nominal model; Δk_a, Δk_b per device)."""
    lam = np.asarray(lam, dtype=float)
    df, dl = family.spread.sample(n_devices, seed, with_load=family.has_load)
    dk = family.evaluate(lambda m: m(lam), df, dl)                      # (D, 2, M)
    nom = family.nominal
    eta = np.array([thg_spectrum(nom, p, lam, delta_k=(dk[d, 0], dk[d, 1]))[0] for d in range(n_devices)])
    nom_eta = thg_spectrum(nom, p, lam)[0]
    i0 = int(np.argmin(np.abs(lam - lam0)))
    return {"d_film": df, "d_load": dl, "lam": lam, "eta": eta, "nominal": nom_eta, "at_target": eta[:, i0],
            "nominal_at_target": float(nom_eta[i0]), "peak": lam[np.argmax(eta, axis=1)],
            "peak_eta": eta.max(axis=1)}


def double_matching_shift(family, lam0, h=0.5e-9) -> dict:
    """How a film-thickness change detunes the two steps: ∂Δk_a/∂t, ∂Δk_b/∂t and the fundamental-wavelength
    shifts that would re-match each step alone (they differ, so tuning λ cannot restore both)."""
    nom = family.nominal
    dx = lam0 * 1e-5
    out = {}
    for name, fn in (("a", nom.dk_a), ("b", nom.dk_b)):
        idx = 0 if name == "a" else 1
        dkdt = float((family.evaluate(lambda m: m(np.array([lam0]))[idx], [h])[0][0]
                      - family.evaluate(lambda m: m(np.array([lam0]))[idx], [-h])[0][0]) / (2 * h))
        dkdl = float((fn(lam0 + dx) - fn(lam0 - dx)) / (2 * dx))
        out[f"ddk_{name}_dt"] = dkdt
        out[f"dlam_{name}_dt"] = -dkdt / dkdl
    return out


def optimise_blend(model, lam0, length, starts=(0.7, 0.8, 0.9, 1.0), ends=(0.0, 0.1, 0.2, 0.3),
                   phases=np.linspace(0, np.pi, 5), fab=DEFAULT_FAB):
    """Grid search of the blend (w_start, w_end, ψ) at λ0; returns (best Poling, params dict, η_THG)."""
    best = (None, None, -1.0)
    for ws in starts:
        for we in ends:
            for ph in phases:
                p = blend(model, lam0, length, ws, we, ph, fab)
                eta = float(thg_spectrum(model, p, [lam0])[0][0])
                if eta > best[2]:
                    best = (p, {"w_start": ws, "w_end": we, "phase": float(ph)}, eta)
    return best
