"""Task 5b: wave-optics limit of the input angle dither, against the engine's single-ray value.

For a linear, time-invariant cell the exit field is E(theta) = U E_in(theta) with E_in(theta) = E_in(x) exp(i k n theta x).
The static (time-averaged) fringe of a sine angle dither of amplitude A is therefore the fringe of the averaged input field
E_in(x) J0(k n A x): its overlap with the undithered input is rho = <J0(k n A x)>_I, and the rms static fringe factor is
V_wave ~ sqrt(<J0(k n A x)^2>_I) (coherent part rho plus a random part from the orthogonal remainder), whatever the cell
does (chaos included). I = Gaussian intensity of 1/e^2 radius w. Compared with the engine's V_theta, which follows a single
centre ray per dither angle (exit path length changes by many wavelengths per mrad) and so ignores that tilts smaller than
the beam divergence lambda / (pi n w) do not change the field.
-> results/t5/wave_bound.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import j0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qv import analysis as A  # noqa: E402

LAM, N = 1.5317e-6, 1.5391
K = 2 * np.pi * N / LAM
out = {"model": __doc__.strip().splitlines()[0], "lambda_m": LAM, "n_eff": N, "beams": {}, "engine": {}}
for w in (40e-6, 75e-6, 150e-6):
    x = np.linspace(-3 * w, 3 * w, 4001)
    I = np.exp(-2 * x ** 2 / w ** 2)
    I /= I.sum()
    row = {"divergence_mrad": 1e3 * LAM / (np.pi * N * w)}
    for a in (2e-3, 5e-3, 10e-3, 20e-3, 50e-3):
        J = j0(K * a * x)
        row[f"{a * 1e3:g}mrad"] = {"rho": float((I * J).sum()), "V_wave": float(np.sqrt((I * J ** 2).sum()))}
    out["beams"][f"w{w * 1e6:.0f}um"] = row
for f in ("circle_nir_R0.95_0.3dB", "circle_nir_R0.97_0.3dB", "circle_nir_R0.995_0.3dB", "herriott_nir_R0.97_0.3dB"):
    d = A.load(ROOT / "results" / "t2" / f"{f}.json")
    ch = A.chaotic(d)
    med = lambda k: float(np.median([x[k] for x in ch]))
    amp = [2e-3 / np.sqrt(x["grains"]) for x in ch]
    out["engine"][f] = {"V_theta_engine": med("V_theta"), "V_lambda": med("V_lambda"), "lam_ftle_per_bounce": med("lam_ftle"),
                        "mean_bounces": med("mean_bounces"), "lam_ftle_regular": d["regular"]["lam_ftle"],
                        "fringe_both_engine": float(np.median([a * x["V_joint"] for a, x in zip(amp, ch)])),
                        "fringe_spectral_only": float(np.median([a * x["V_lambda"] for a, x in zip(amp, ch)])),
                        "fringe_both_wave_bound_2mrad_w40": float(np.median([a * x["V_lambda"] for a, x in zip(amp, ch)])) * out["beams"]["w40um"]["2mrad"]["V_wave"]}
(ROOT / "results" / "t5").mkdir(parents=True, exist_ok=True)
A.dump(ROOT / "results" / "t5" / "wave_bound.json", out)
for k, v in out["engine"].items():
    print(k, {kk: (f"{vv:.3g}" if isinstance(vv, float) else vv) for kk, vv in v.items()})
