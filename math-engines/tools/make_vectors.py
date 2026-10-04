"""Shared JSON test vectors from the Python reference engines.

JavaScript ports (dbr-structures/dbr-engine.js and
amplifiers/parametric-amplifiers/opa_engine.js) are checked against these by
tools/check_js_ports.mjs.
Run from math-engines/:  python tools/make_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines.bragg_grating import engine as bg  # noqa: E402
from engines.materials import engine as mat  # noqa: E402
from engines.parametric_amplifier import engine as pa  # noqa: E402
from engines.slab_waveguide import engine as sw  # noqa: E402
from engines.step_index_fiber import engine as sif  # noqa: E402
from engines.stimulated_scattering import engine as ss  # noqa: E402

OUT = ROOT / "test_vectors" / "vectors.json"


def build() -> dict:
    lams = [0.532e-6, 0.775e-6, 1.064e-6, 1.55e-6, 2.0e-6]
    v = {"units": "SI (m, 1/m); indices dimensionless", "materials": [], "slab": [], "cmt": [], "stack": [], "lp01": []}
    for m in ("sio2", "si3n4", "ln_e", "ln_o", "lt_e"):
        for l in lams:
            v["materials"].append({"material": m, "wavelength": l, "n": float(mat.index(m, l))})
    for (l, ns, nf, nc, d, pol, o) in [
        (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TE", 0), (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TM", 0),
        (1.55e-6, 1.444, 2.130, 1.0, 0.6e-6, "TE", 1), (0.775e-6, 1.454, 2.17, 1.0, 0.4e-6, "TE", 0),
        (1.55e-6, 1.444, 1.996, 1.444, 0.4e-6, "TE", 0), (1.55e-6, 1.444, 3.476, 1.444, 0.22e-6, "TE", 0),
    ]:
        v["slab"].append({"wavelength": l, "n_sub": ns, "n_core": nf, "n_clad": nc, "thickness": d, "polarization": pol, "order": o,
                          "neff": float(sw.neff_three_layer(l, ns, nf, nc, d, pol, o))})
    for (kap, L, dl) in [(2e4, 3e-4, 0.0), (2e4, 3e-4, 1.5e4), (2e4, 3e-4, 5e4), (5e3, 1e-3, 3e3)]:
        lamB, neff = 1.55e-6, 2.0
        lam = 1 / (dl / (2 * np.pi * neff) + 1 / lamB)
        v["cmt"].append({"kappa": kap, "length": L, "half_detuning": dl, "R": float(bg.cmt_reflectance(lam, lamB, neff, kap, L))})
    for (l, nH, nL, dH, dL, N, nin, nout) in [
        (1.55e-6, 2.0, 1.5, 1.55e-6 / 8, 1.55e-6 / 6, 5, 1.5, 1.5),
        (1.551e-6, 2.0272, 2.0095, 0.1918e-6, 0.1918e-6, 800, 2.0272, 2.0272),
        (1.548e-6, 2.0272, 2.0095, 0.1918e-6, 0.1918e-6, 800, 2.0272, 2.0272),
    ]:
        r = bg.stack_reflectance(l, nH, nL, dH, dL, N, n_in=nin, n_out=nout)
        v["stack"].append({"wavelength": l, "n_high": nH, "n_low": nL, "d_high": dH, "d_low": dL, "periods": N, "n_in": nin, "n_out": nout, "R": float(r["R"]), "T": float(r["T"])})
    for (l, a, dn) in [(1.55e-6, 4.1e-6, 0.005), (0.532e-6, 2.0e-6, 0.005), (1.064e-6, 3.0e-6, 0.008)]:
        r = sif.lp01(l, a, dn)
        v["lp01"].append({"wavelength": l, "core_radius": a, "delta_n": dn, "neff": float(r["neff"]), "V": float(r["V"]), "mode_radius": float(r["mode_radius"])})
    _amplifier_vectors(v)
    return v


def _amplifier_vectors(v: dict) -> None:
    """Parametric amplifier, fiber dispersion and scattering cases for opa_engine.js."""
    thz = 2 * np.pi * 1e12
    det = [x * thz for x in (-8, -3.5, -2, -0.5, 0.3, 1.5, 4, 8)]
    v["pa_dispersion"] = []
    for proc, lp, a, dn in (("chi2", 0.532e-6, 4.1e-6, 0.005), ("chi3", 1.064e-6, 1.8e-6, 0.03), ("chi3", 1.56e-6, 1.8e-6, 0.03)):
        r = sif.phase_mismatch(1.55e-6, lp, a, dn, proc, detuning=det)
        v["pa_dispersion"].append({"process": proc, "wavelength_pump": lp, "wavelength_signal": 1.55e-6, "core_radius": a, "delta_n": dn,
                                   "dk0": float(r["dk0"]), "gvm": float(r["gvm"]), "beta2_sum": float(r["beta2_sum"]),
                                   "detuning": det, "dk_rel": [float(x) for x in r["dk_rel"]]})
    v["pa_coupling"] = []
    for proc, lp, radii, P, extra in (("chi2", 0.532e-6, (2.5e-6, 5e-6, 3.18e-6), 1.0, {"d_eff": 0.08e-12}),
                                      ("chi3", 1.064e-6, (1.7e-6, 2.14e-6, 1.52e-6), 10.0, {"n2": 2.6e-20}),
                                      ("chi3", 1.56e-6, (2.2e-6, 2.2e-6, 2.25e-6), 5.0, {"n2": 2.6e-20})):
        r = pa.coupling(proc, lp, 1.55e-6, *radii, P, **extra)
        v["pa_coupling"].append({"process": proc, "lambda_pump": lp, "lambda_signal": 1.55e-6, "w_pump": radii[0], "w_signal": radii[1],
                                 "w_idler": radii[2], "pump_power": P, "d_eff": extra.get("d_eff", 0.0), "n2": extra.get("n2", 0.0),
                                 "gain_coefficient": float(r["gain_coefficient"]), "A_eff": float(r["A_eff"]),
                                 "lambda_idler": float(r["lambda_idler"]), "nonlinear_phase": float(r["nonlinear_phase"])})
    v["pa_small_signal_gain"] = []
    for G, dk, L, ai, as_, nd, ti, ts in ((0.95, 0.0, 5, 0, 0, 0, 1, 1), (0.95, 1.5, 5, 0, 0, 0, 1, 1), (0.5, 3.0, 7, 0, 0, 0, 1, 1),
                                          (0.95, 0.4, 5, 4.6, 0, 0, 1, 1), (2.0, -1.0, 5, 20, 0, 0, 1, 1), (1.0, 0.0, 6, 0, 0, 3, 0.0, 1),
                                          (1.0, 0.7, 6, 0, 0, 9, 1e-3, 1), (0.8, 0.2, 5, 1.0, 0.3, 4, 0.5, 0.9), (0.0, 0.0, 3, 0, 0.7, 0, 1, 1)):
        r = pa.small_signal_gain(G, dk, L, idler_loss=ai, signal_loss=as_, n_dumps=nd, idler_dump_transmission=ti, signal_dump_transmission=ts)
        v["pa_small_signal_gain"].append({"gain_coefficient": G, "phase_mismatch": dk, "length": L, "idler_loss": ai, "signal_loss": as_,
                                          "n_dumps": nd, "idler_dump_transmission": ti, "signal_dump_transmission": ts,
                                          "gain": float(r["gain"]), "idler_conversion": float(r["idler_conversion"])})
    radii2, radii3, radii15 = [2.5e-6, 5e-6, 3.18e-6], [1.7e-6, 2.14e-6, 1.52e-6], [2.2e-6, 2.2e-6, 2.25e-6]
    cases = [
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.0),
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-3, length=8.0, gain_coefficient=1.2, phase_mismatch=0.3),
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.5, idler_loss=4.6),
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=1.5, phase_mismatch=0.0,
             n_dumps=4, idler_dump_transmission=1e-3, signal_dump_transmission=0.9, pump_dump_transmission=0.95),
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.2,
             idler_loss=2.0, signal_loss=0.1, pump_loss=0.05),
        dict(process="chi3", lambda_pump=1.064e-6, pump_power=10.0, signal_power=1e-5, length=30.0, gain_coefficient=0.147, phase_mismatch=0.0),
        dict(process="chi3", lambda_pump=1.064e-6, pump_power=10.0, signal_power=1e-2, length=40.0, gain_coefficient=0.2, phase_mismatch=0.1),
        dict(process="chi3", lambda_pump=1.56e-6, pump_power=5.0, signal_power=1e-5, length=30.0, gain_coefficient=0.1, phase_mismatch=-0.05, idler_loss=0.3),
        dict(process="chi2", lambda_pump=0.532e-6, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.0,
             sbs=True, srs=True, pump_linewidth=1e6, mode_radii=radii2),
        dict(process="chi3", lambda_pump=1.064e-6, pump_power=60.0, signal_power=1e-5, length=30.0, gain_coefficient=0.88, phase_mismatch=0.0,
             sbs=True, srs=True, pump_linewidth=3e9, mode_radii=radii3),
        dict(process="chi3", lambda_pump=1.56e-6, pump_power=25.0, signal_power=1e-5, length=30.0, gain_coefficient=0.3, phase_mismatch=0.0,
             srs=True, mode_radii=radii15),
    ]
    v["pa_amplify"] = []
    for c in cases:
        r = pa.amplify(lambda_signal=1.55e-6, **c)
        v["pa_amplify"].append({"inputs": {**c, "lambda_signal": 1.55e-6}, "pump_fraction": float(r["pump_fraction"]),
                                "signal_fraction": float(r["signal_fraction"]), "idler_fraction": float(r["idler_fraction"]),
                                "P_raman_out": float(r["P_raman_out"]), "steps": int(r["steps"])})
    v["brillouin"] = []
    for lam, lw in ((0.532e-6, 1e6), (1.064e-6, 0.0), (1.55e-6, 1e9)):
        b = ss.brillouin_parameters(lam, lw)
        v["brillouin"].append({"lambda_pump": lam, "pump_linewidth": lw, "shift": float(b["shift"]), "linewidth": float(b["linewidth"]),
                               "g_B": float(b["g_B"]), "seed_power": float(b["seed_power"])})
    v["sbs"] = []
    for P0, gA, L, seed in ((10.0, 5e-11 / 10e-12, 30.0, 1e-9), (1.0, 2e-11 / 20e-12, 5.0, 5e-9), (60.0, 1e-13 / 10e-12, 30.0, 1e-8)):
        s = ss.SbsSolution(P0, gA, L, seed)
        v["sbs"].append({"P0": P0, "gA": gA, "length": L, "seed": seed, "reflected": s.D, "transmitted": s.PL, "PB_mid": s.PB(L / 2)})
    v["raman"] = [{"shift": nu * thz, "shape": float(ss.raman_shape(nu * thz))} for nu in (-30, -13.1, -1.2, 0.5, 1.2, 5, 13.1, 20, 35, 88)]
    v["raman_peak_shift"] = float(ss.RAMAN_PEAK_SHIFT)


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
