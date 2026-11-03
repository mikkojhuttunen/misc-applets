"""Material models: lossless dispersive media for the scattering runs.

Every model gives
* principal permittivities (eps_a, eps_b, eps_c) in the crystal frame vs wavelength (numpy),
* Tidy3D media for the ordinary and extraordinary components (dispersive, lossless).

Isotropic materials have eps_a = eps_b = eps_c. Uniaxial materials have the optic axis
along crystal c. Absorption is never included here; intrinsic absorption is a separate
budget item (README). `provenance()` lists the source used for every material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from .config import Cfg, resolve_path

C0_UM_PER_S = 299792458.0e6


def freq(lam_um):
    return C0_UM_PER_S / np.asarray(lam_um, dtype=float)


def sellmeier_eps(lam_um, coeffs):
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    return 1.0 + sum(b * lam2 / (lam2 - c) for b, c in coeffs)


def fit_sellmeier(lam_um, n, terms: int):
    """Lossless Sellmeier fit (B_i >= 0, C_i >= 0) to tabulated real n; returns coeffs and max |dn|."""
    lam_um = np.asarray(lam_um, dtype=float)
    n = np.asarray(n, dtype=float)
    lam_min = lam_um.min()

    def unpack(p):
        return [(p[2 * i], p[2 * i + 1]) for i in range(terms)]

    def resid(p):
        return np.sqrt(sellmeier_eps(lam_um, unpack(p))) - n

    best = None
    uv_starts = (0.1 * lam_min) ** 2, (0.3 * lam_min) ** 2, (0.5 * lam_min) ** 2
    for c_uv in uv_starts:
        p0, lo, hi = [], [], []
        for i in range(terms):
            c0 = c_uv if i == 0 else (10.0 * i) ** 2
            p0 += [n.mean() ** 2 - 1.0 if i == 0 else 0.5, c0]
            lo += [0.0, 0.0 if i == 0 else (2.0 * lam_um.max()) ** 2]
            hi += [np.inf, (0.95 * lam_min) ** 2 if i == 0 else np.inf]
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
        except ValueError:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    coeffs = unpack(best.x)
    return coeffs, float(np.max(np.abs(resid(best.x))))


def _read_csv(path, fit_range=None):
    rows = []
    with open(resolve_path(path)) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("wavelength"):
                rows.append([float(v) for v in line.split(",")[:2]])
    data = np.array(rows)
    if fit_range is not None:
        data = data[(data[:, 0] >= fit_range[0]) & (data[:, 0] <= fit_range[1])]
    return data[:, 0], data[:, 1]


@dataclass
class MaterialModel:
    name: str
    uniaxial: bool
    source: str
    valid_range_um: tuple[float, float] | None
    td_o: object = field(repr=False)
    td_e: object = field(repr=False)
    notes: list[str] = field(default_factory=list)

    def eps_principal(self, lam_um) -> np.ndarray:
        """Real principal permittivities, shape (..., 3): (eps_a, eps_b, eps_c)."""
        f = freq(lam_um)
        eo = np.real(np.asarray(self.td_o.eps_model(f)))
        ee = np.real(np.asarray(self.td_e.eps_model(f)))
        return np.stack([eo, eo, ee], axis=-1)

    def n_principal(self, lam_um) -> np.ndarray:
        return np.sqrt(self.eps_principal(lam_um))

    def td_medium(self):
        if self.uniaxial:
            raise ValueError(f"{self.name} is uniaxial: build it with orientation.oriented_medium")
        return self.td_o

    def out_of_range(self, lam_um) -> list[float]:
        if self.valid_range_um is None:
            return []
        lo, hi = self.valid_range_um
        return [float(x) for x in np.atleast_1d(lam_um) if x < lo * (1 - 1e-9) or x > hi * (1 + 1e-9)]


def _td():
    import tidy3d as td
    return td


def _library_medium(name, variant):
    td = _td()
    if name not in td.material_library:
        raise KeyError(f"tidy3d material_library has no {name!r} (version {td.__version__})")
    entry = td.material_library[name]
    if variant not in entry.variants:
        raise KeyError(f"{name!r} has no variant {variant!r}; available: {list(entry.variants)}")
    return entry.variants[variant]


def _range_um(medium):
    fr = getattr(medium, "frequency_range", None)
    if not fr:
        return None
    return (C0_UM_PER_S / fr[1], C0_UM_PER_S / fr[0])


def _check_lossless(medium, lam_um, tol, label):
    imag = np.max(np.abs(np.imag(np.asarray(medium.eps_model(freq(lam_um))))))
    if imag > tol:
        raise ValueError(f"{label} has Im(eps) = {imag:.2e} > {tol:.1e}; use kind: sellmeier_fit")


def _sellmeier_medium(coeffs, rng):
    td = _td()
    kwargs = {}
    if rng is not None:
        kwargs["frequency_range"] = (C0_UM_PER_S / rng[1], C0_UM_PER_S / rng[0])
    return td.Sellmeier(coeffs=[(float(b), float(c)) for b, c in coeffs], **kwargs)


def build(name: str, cfg: Cfg) -> MaterialModel:
    m = cfg.materials[name]
    kind = m.kind
    tol_dn = m["max_abs_dn"] if "max_abs_dn" in m else cfg.material_fit.max_abs_dn
    lossless_tol = cfg.material_fit.lossless_imag_eps_tol
    all_lams = np.array(sorted(set(cfg.wavelengths_um.vis) | set(cfg.wavelengths_um.telecom)))
    td = _td()

    if kind == "constant":
        med = td.Medium(permittivity=float(m.n) ** 2, name=name)
        return MaterialModel(name, False, f"constant n = {m.n} (non-dispersive)", None, med, med)

    if kind == "library":
        var = _library_medium(m.name, m.variant)
        med = var.medium
        rng = _range_um(med)
        lams = all_lams if rng is None else all_lams[(all_lams >= rng[0]) & (all_lams <= rng[1])]
        _check_lossless(med, lams, lossless_tol, f"{m.name}/{m.variant}")
        ref = _doi(var)
        return MaterialModel(name, False, f"tidy3d material_library {m.name}/{m.variant} {ref}", rng, med, med)

    if kind == "uniaxial_library":
        var = _library_medium(m.name, m.variant)
        aniso = var.medium(2)
        rng = _range_um(aniso.xx)
        for comp in (aniso.xx, aniso.zz):
            _check_lossless(comp, all_lams, lossless_tol, f"{m.name}/{m.variant}")
        ref = _doi(var)
        return MaterialModel(name, True, f"tidy3d material_library {m.name}/{m.variant} (uniaxial) {ref}",
                             rng, aniso.xx, aniso.zz)

    if kind in ("sellmeier", "uniaxial_sellmeier"):
        sq =bool(m["c_are_wavelengths"]) if "c_are_wavelengths" in m else False
        conv = (lambda raw: [(b, c**2 if sq else c) for b, c in raw])
        if kind == "sellmeier":
            med = _sellmeier_medium(conv(m.coeffs), None)
            return MaterialModel(name, False, f"Sellmeier: {m.citation}", None, med, med)
        med_o = _sellmeier_medium(conv(m.o), None)
        med_e = _sellmeier_medium(conv(m.e), None)
        return MaterialModel(name, True, f"uniaxial Sellmeier: {m.citation}", None, med_o, med_e)

    if kind == "sellmeier_fit":
        if "data" in m:
            lam, n = _read_csv(m.data, m["fit_range_um"] if "fit_range_um" in m else None)
            src = f"lossless Sellmeier fit ({m.terms} terms) to {m.data}: {m.citation}"
        else:
            var = _library_medium(m.library_name, m.library_variant)
            lam = np.linspace(*m.fit_range_um, 200)
            n = np.real(np.sqrt(np.asarray(var.medium.eps_model(freq(lam)))))
            src = (f"lossless Sellmeier fit ({m.terms} terms) to Re(n) of tidy3d "
                   f"{m.library_name}/{m.library_variant} {_doi(var)} over {list(m.fit_range_um)} um")
        coeffs, dn = _fit_cached(tuple(lam), tuple(n), int(m.terms))
        if dn > tol_dn:
            raise ValueError(f"{name}: Sellmeier fit max |dn| = {dn:.1e} > {tol_dn:.1e}")
        rng = (float(lam.min()), float(lam.max()))
        med = _sellmeier_medium(coeffs, rng)
        return MaterialModel(name, False, f"{src} (max |dn| {dn:.1e})", rng, med, med)

    if kind == "uniaxial_sellmeier_fit":
        meds, dns, rngs = [], [], []
        for key in ("data_o", "data_e"):
            lam, n = _read_csv(m[key], m["fit_range_um"] if "fit_range_um" in m else None)
            coeffs, dn = _fit_cached(tuple(lam), tuple(n), int(m.terms))
            if dn > tol_dn:
                raise ValueError(f"{name} {key}: Sellmeier fit max |dn| = {dn:.1e} > {tol_dn:.1e}")
            rng = (float(lam.min()), float(lam.max()))
            meds.append(_sellmeier_medium(coeffs, rng))
            dns.append(dn)
            rngs.append(rng)
        rng = (max(r[0] for r in rngs), min(r[1] for r in rngs))
        src = (f"uniaxial lossless Sellmeier fits ({m.terms} terms) to {m.data_o}, {m.data_e}: "
               f"{m.citation} (max |dn| {max(dns):.1e})")
        return MaterialModel(name, True, src, rng, meds[0], meds[1])

    raise ValueError(f"material {name}: unknown kind {kind!r}")


@lru_cache(maxsize=None)
def _fit_cached(lam, n, terms):
    return fit_sellmeier(np.array(lam), np.array(n), terms)


def _doi(variant):
    refs = getattr(variant, "reference", None) or []
    dois = [r.doi for r in refs if getattr(r, "doi", None)]
    return f"[{', '.join(dois)}]" if dois else "[no reference in library]"


def provenance(cfg: Cfg) -> list[dict]:
    """One row per material: source, validity range and configured wavelengths outside it."""
    lams = sorted(set(cfg.wavelengths_um.vis) | set(cfg.wavelengths_um.telecom))
    rows = []
    for name in cfg.materials.keys():
        model = build(name, cfg)
        rows.append({"material": name, "uniaxial": model.uniaxial, "source": model.source,
                     "valid_range_um": model.valid_range_um,
                     "outside_range_um": model.out_of_range(lams)})
    return rows
