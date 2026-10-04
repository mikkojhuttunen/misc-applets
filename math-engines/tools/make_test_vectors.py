"""Generate shared JSON test vectors for the JavaScript port of the amplifier engines.

The applet amplifiers/parametric-amplifiers/opa_engine.js is a hot-path port of
fiber_mode, parametric_amplifier and stimulated_scattering. These vectors are
computed by the Python reference implementation; the JavaScript test
(amplifiers/parametric-amplifiers/test/opa_engine.test.mjs) must reproduce them.

    python tools/make_test_vectors.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.common import C0, json_safe  # noqa: E402
from engines.fiber_mode import engine as fm  # noqa: E402
from engines.parametric_amplifier import engine as pa  # noqa: E402
from engines.stimulated_scattering import engine as ss  # noqa: E402

OUT = ROOT / "engines" / "parametric_amplifier" / "test_vectors.json"
THZ = 2 * math.pi * 1e12


def build() -> dict:
    v = {"generated_by": "math-engines/tools/make_test_vectors.py", "units": "SI", "silica_index": [], "lp01": [],
         "dispersion": [], "coupling": [], "small_signal_gain": [], "amplify": [], "brillouin": [], "sbs": [], "raman": []}
    for lam in (532e-9, 810e-9, 1064e-9, 1550e-9, 2000e-9):
        v["silica_index"].append({"wavelength": lam, "n": fm.silica_index(lam)["n"]})
    for lam, a, dn in ((1550e-9, 4.1e-6, 0.005), (532e-9, 4.1e-6, 0.005), (1064e-9, 1.8e-6, 0.03), (810e-9, 2e-6, 0.02)):
        r = fm.lp01_mode(lam, a, dn)
        v["lp01"].append({"wavelength": lam, "core_radius": a, "delta_n": dn, "n_eff": r["n_eff"], "V": r["V"], "w_mode": r["w_mode"]})
    det = [x * THZ for x in (-8, -3.5, -2, -0.5, 0.3, 1.5, 4, 8)]
    for proc, lp, a, dn in (("chi2", 532e-9, 4.1e-6, 0.005), ("chi3", 1064e-9, 1.8e-6, 0.03), ("chi3", 1560e-9, 1.8e-6, 0.03)):
        r = fm.phase_mismatch(1550e-9, lp, a, dn, proc, detuning=det)
        v["dispersion"].append({"process": proc, "lambda_pump": lp, "lambda_signal": 1550e-9, "core_radius": a, "delta_n": dn,
                                "dk0": r["dk0"], "gvm": r["gvm"], "beta2_sum": r["beta2_sum"], "detuning": det, "dk_rel": list(r["dk_rel"])})
    for proc, lp, radii, P, extra in (("chi2", 532e-9, (2.5e-6, 5e-6, 3.18e-6), 1.0, {"d_eff": 0.08e-12}),
                                      ("chi3", 1064e-9, (1.7e-6, 2.14e-6, 1.52e-6), 10.0, {"n2": 2.6e-20}),
                                      ("chi3", 1560e-9, (2.2e-6, 2.2e-6, 2.25e-6), 5.0, {"n2": 2.6e-20})):
        r = pa.coupling(proc, lp, 1550e-9, *radii, P, **extra)
        v["coupling"].append({"process": proc, "lambda_pump": lp, "lambda_signal": 1550e-9, "w_pump": radii[0], "w_signal": radii[1],
                              "w_idler": radii[2], "pump_power": P, "d_eff": extra.get("d_eff", 0.0), "n2": extra.get("n2", 0.0),
                              "gain_coefficient": r["gain_coefficient"], "A_eff": r["A_eff"], "lambda_idler": r["lambda_idler"],
                              "nonlinear_phase": r["nonlinear_phase"]})
    for G, dk, L, ai, as_, nd, ti, ts in ((0.95, 0.0, 5, 0, 0, 0, 1, 1), (0.95, 1.5, 5, 0, 0, 0, 1, 1), (0.5, 3.0, 7, 0, 0, 0, 1, 1),
                                          (0.95, 0.4, 5, 4.6, 0, 0, 1, 1), (2.0, -1.0, 5, 20, 0, 0, 1, 1), (1.0, 0.0, 6, 0, 0, 3, 0.0, 1),
                                          (1.0, 0.7, 6, 0, 0, 9, 1e-3, 1), (0.8, 0.2, 5, 1.0, 0.3, 4, 0.5, 0.9), (0.0, 0.0, 3, 0, 0.7, 0, 1, 1)):
        r = pa.small_signal_gain(G, dk, L, idler_loss=ai, signal_loss=as_, n_dumps=nd, idler_dump_transmission=ti, signal_dump_transmission=ts)
        v["small_signal_gain"].append({"gain_coefficient": G, "phase_mismatch": dk, "length": L, "idler_loss": ai, "signal_loss": as_,
                                       "n_dumps": nd, "idler_dump_transmission": ti, "signal_dump_transmission": ts,
                                       "gain": r["gain"], "idler_conversion": r["idler_conversion"]})
    cases = [
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.0),
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-3, length=8.0, gain_coefficient=1.2, phase_mismatch=0.3),
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.5,
             idler_loss=4.6, n_dumps=0),
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=1.5, phase_mismatch=0.0,
             n_dumps=4, idler_dump_transmission=1e-3, signal_dump_transmission=0.9, pump_dump_transmission=0.95),
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.2,
             idler_loss=2.0, signal_loss=0.1, pump_loss=0.05),
        dict(process="chi3", lambda_pump=1064e-9, pump_power=10.0, signal_power=1e-5, length=30.0, gain_coefficient=0.147, phase_mismatch=0.0),
        dict(process="chi3", lambda_pump=1064e-9, pump_power=10.0, signal_power=1e-2, length=40.0, gain_coefficient=0.2, phase_mismatch=0.1),
        dict(process="chi3", lambda_pump=1560e-9, pump_power=5.0, signal_power=1e-5, length=30.0, gain_coefficient=0.1, phase_mismatch=-0.05,
             idler_loss=0.3),
        dict(process="chi2", lambda_pump=532e-9, pump_power=1.0, signal_power=1e-5, length=5.0, gain_coefficient=0.95, phase_mismatch=0.0,
             sbs=True, srs=True, pump_linewidth=1e6, mode_radii=(2.5e-6, 5e-6, 3.18e-6)),
        dict(process="chi3", lambda_pump=1064e-9, pump_power=60.0, signal_power=1e-5, length=30.0, gain_coefficient=0.88, phase_mismatch=0.0,
             sbs=True, srs=True, pump_linewidth=3e9, mode_radii=(1.7e-6, 2.14e-6, 1.52e-6)),
        dict(process="chi3", lambda_pump=1560e-9, pump_power=25.0, signal_power=1e-5, length=30.0, gain_coefficient=0.3, phase_mismatch=0.0,
             srs=True, mode_radii=(2.2e-6, 2.2e-6, 2.25e-6)),
    ]
    for c in cases:
        r = pa.amplify(lambda_signal=1550e-9, **c)
        v["amplify"].append({"inputs": {**c, "lambda_signal": 1550e-9, "mode_radii": list(c.get("mode_radii", (1, 1, 1)))},
                             "pump_fraction": r["pump_fraction"], "signal_fraction": r["signal_fraction"],
                             "idler_fraction": r["idler_fraction"], "P_raman_out": r["P_raman_out"], "steps": r["steps"]})
    for lam, lw in ((532e-9, 1e6), (1064e-9, 0.0), (1550e-9, 1e9)):
        b = ss.brillouin_parameters(lam, lw)
        v["brillouin"].append({"lambda_pump": lam, "pump_linewidth": lw, "shift": b["shift"], "linewidth": b["linewidth"],
                               "g_B": b["g_B"], "seed_power": b["seed_power"]})
    for P0, gA, L, seed in ((10.0, 5e-11 / 10e-12, 30.0, 1e-9), (1.0, 2e-11 / 20e-12, 5.0, 5e-9), (60.0, 1e-13 / 10e-12, 30.0, 1e-8)):
        s = ss.SbsSolution(P0, gA, L, seed)
        v["sbs"].append({"P0": P0, "gA": gA, "length": L, "seed": seed, "reflected": s.D, "transmitted": s.PL, "PB_mid": s.PB(L / 2)})
    for nu in (-30, -13.1, -1.2, 0.5, 1.2, 5, 13.1, 20, 35, 88):
        v["raman"].append({"shift": nu * THZ, "shape": float(ss.raman_shape(nu * THZ))})
    v["raman_peak_shift"] = ss.RAMAN_PEAK_SHIFT
    return json_safe(v)


def dumps(v: dict) -> str:
    return json.dumps(v, indent=1, sort_keys=True) + "\n"


def main() -> int:
    OUT.write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
