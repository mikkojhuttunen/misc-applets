"""Step 6: CW DFG source design in MgO:PPLN (qs/dfg.py; literature parameters and their status listed there).
Pump 1064 nm, signal 1.50-1.63 um, idler 3.06-3.66 um: QPM period vs idler and temperature (25-200 C), tuning with a
multi-period / fan-out grating, temperature and spectral acceptance, walk-off, single-pass CW idler power (L 20-50 mm,
pump 1 W, signal 1-5 W, Boyd-Kleinman), signal wavelengths for the CH4 and C2H6 lines of step 4, and the mapping of
the +-20 pm idler dither to the signal laser at 1-100 kHz.  -> results/step6_dfg.json, results/step6_period_grid.csv
python scripts/step6_dfg.py"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qs import dfg  # noqa: E402
from qs.metrics import RESULTS, dump  # noqa: E402

LP = 1.064
C = dfg.C


def main():
    out = {"assumptions": [
        "MgO:CLN 5 %, extraordinary Sellmeier of Gayer et al. 2008 (coefficients VERIFY; QPM check: 1064 nm SHG period "
        "6.95 um at 40 C, as in practice); its absolute dn_e/dT (2.7e-4 /K) is high against the 3-4e-5 /K often quoted for "
        "LN: VERIFY before relying on absolute temperatures (relative tuning is what the fits reproduce)",
        "thermal expansion of the period: Jundt 1997 (undoped CLN, VERIFY for MgO)",
        "d_eff = (2/pi) d33 = 14 pm/V (d33 25 pm/V at 1064 nm, LIT; Miller scaling and duty-cycle errors lower it)",
        "Boyd-Kleinman focusing factor h = 0.8-1.0 for DFG with lambda_i/lambda_p ~ 3 (1.07 is the SHG optimum); "
        "undepleted pump and signal, no idler absorption, no thermal lensing; measured CW DFG reaches 30-60 % of this",
        "type-0 (eee) QPM propagating along x: no birefringent (Poynting) walk-off; group-velocity walk-off is irrelevant CW",
        "pump wavelength exactly 1064.0 nm (a real Nd or Yb fibre laser sits at 1063.8-1064.6 nm; the signal shifts by "
        "dls/dlp = (ls/lp)^2 = 2.2 nm per nm of pump)"]}
    # 1. period grid (period to write at 25 C so that QPM is met at T)
    lis = np.round(np.arange(3.06, 3.6601, 0.02), 3)
    Ts = np.arange(25, 201, 25)
    grid = [[float(dfg.period_at_25(li, T, LP)) for T in Ts] for li in lis]
    with open(RESULTS / "step6_period_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idler_um", "signal_um"] + [f"period_um_T{int(T)}C" for T in Ts])
        for li, row in zip(lis, grid):
            w.writerow([f"{li:.3f}", f"{dfg.signal_for_idler(LP, li):.5f}"] + [f"{v:.4f}" for v in row])
    out["period_um"] = {"idler_um": lis.tolist(), "T_C": Ts.tolist(), "period_at_25C_um": grid,
                        "range_um": [float(np.min(grid)), float(np.max(grid))]}
    # 2. tuning with a multi-period grating: idler vs T for periods spanning the range (0.2 um steps)
    periods = np.round(np.arange(math.floor(np.min(grid) * 5) / 5, math.ceil(np.max(grid) * 5) / 5 + 1e-9, 0.2), 2)
    Tf = np.linspace(25, 200, 36)
    tune = {}
    for Lam in periods:
        li = []
        for T in Tf:
            f = lambda x: float(dfg.period_at_25(x, T, LP)) - Lam
            xs = np.linspace(3.0, 3.8, 81)
            v = [f(x) for x in xs]
            r = math.nan
            for i in range(80):
                if v[i] * v[i + 1] < 0:
                    from scipy.optimize import brentq
                    r = brentq(f, xs[i], xs[i + 1]); break
            li.append(r)
        tune[f"{Lam:.2f}"] = li
    out["tuning"] = {"T_C": Tf.tolist(), "idler_um_by_period": tune, "periods_um": periods.tolist(),
                     "note": "multi-period chip: periods every 0.2 um cover 3.06-3.66 um with 25-200 C; a fan-out "
                             f"grating from {periods[0]:.1f} to {periods[-1]:.1f} um tunes continuously at fixed T"}
    # 3./4. acceptances at the two gas lines, L = 20-50 mm
    lines = {"CH4_3.31 (3016.49 /cm)": 1e4 / 3016.492, "C2H6_3.35 (2986.68 /cm)": 1e4 / 2986.682,
             "CH4_3.31 nominal (brief)": 3.31, "C2H6_3.34 Q-branch (2996.85 /cm)": 1e4 / 2996.852}
    acc = {}
    for name, li in lines.items():
        ls = dfg.signal_for_idler(LP, li)
        T0 = 60.0
        Lam = float(dfg.period_at_25(li, T0, LP))
        rows = {}
        for Lmm in (20, 30, 40, 50):
            L = Lmm * 1e-3
            dT, _, _ = dfg.fwhm_of(lambda T: dfg.dk(ls, T, Lam, LP), T0, L, 0.5)
            dls, _, _ = dfg.fwhm_of(lambda x: dfg.dk(x, T0, Lam, LP), ls, L, 2e-4)
            dnu = C * dls * 1e-6 / (ls * 1e-6) ** 2          # signal = idler frequency acceptance (fixed pump), Hz
            pw = {f"Ps_{Ps}W_h{h}": dfg.power(1.0, Ps, L, li, ls, T0, LP, 14e-12, h) for Ps in (1, 2, 5) for h in (0.8, 1.0)}
            rows[Lmm] = {"T_fwhm_C": dT, "signal_fwhm_nm": dls * 1e3, "idler_freq_fwhm_GHz": dnu / 1e9,
                         "idler_fwhm_nm": (li * 1e-6) ** 2 * dnu / C * 1e9,
                         "P_idler_mW": {k: v * 1e3 for k, v in pw.items()},
                         "efficiency_mW_per_W2_cm_h1": dfg.power(1.0, 1.0, L, li, ls, T0, LP, 14e-12, 1.0) * 1e3 / (Lmm / 10)}
        acc[name] = {"idler_um": li, "signal_um": ls, "period_um_for_60C": Lam, "by_length_mm": rows,
                     "walk_off_mrad": 0.0}
    out["lines"] = acc
    # 8. dither mapping: +-20 pm idler -> signal, fixed pump
    dl_i = 20e-12
    d = {}
    for name, li in lines.items():
        dnu = C * dl_i / (li * 1e-6) ** 2
        ls = dfg.signal_for_idler(LP, li)
        d[name] = {"idler_dnu_GHz": dnu / 1e9, "signal_dlambda_pm": (ls * 1e-6) ** 2 * dnu / C * 1e12,
                   "or_pump_dlambda_pm": (LP * 1e-6) ** 2 * dnu / C * 1e12}
    out["dither"] = {"idler_amplitude_pm": 20, "by_line": d, "rates_kHz": [1, 10, 100], "notes": [
        "fixed pump: the signal carries the whole idler frequency excursion (+-0.55 GHz = +-4.5 pm at 1.567 um)",
        "DFB seed current tuning ~ 0.5-1 GHz/mA in the thermal regime (ASSUMPTION, laser specific): +-0.5-1 mA at 1-100 kHz; "
        "above ~ 100 kHz - 1 MHz the thermal response rolls off and the carrier effect (opposite sign) takes over, so "
        "100 kHz is the practical upper end without a phase modulator; residual amplitude modulation ~ 0.5-1 %/mA",
        "the excursion is < 1 % of the QPM frequency acceptance (> 100 GHz for L <= 50 mm): the idler power stays constant",
        "an EDFA (L-band for 1.567 um) passes the modulation unchanged at these rates"]}
    dump(RESULTS / "step6_dfg.json", out)
    print("QPM period range (25 C write, 25-200 C):", [round(x, 2) for x in out["period_um"]["range_um"]], "um")
    for name, a in acc.items():
        r = a["by_length_mm"]
        print(f"{name}: idler {a['idler_um']:.4f} um, signal {a['signal_um']*1e3:.2f} nm, period {a['period_um_for_60C']:.3f} um at 60 C; "
              + "; ".join(f"L {Lmm} mm: dT {v['T_fwhm_C']:.2f} C, dls {v['signal_fwhm_nm']:.2f} nm ({v['idler_freq_fwhm_GHz']:.0f} GHz), "
                          f"P_i (1 W, 5 W, h 0.8-1) {v['P_idler_mW']['Ps_5W_h0.8']:.1f}-{v['P_idler_mW']['Ps_5W_h1.0']:.1f} mW" for Lmm, v in r.items() if Lmm in (20, 50))
              + f"; eta {r[50]['efficiency_mW_per_W2_cm_h1']:.2f} mW/(W^2 cm)")
    for name, v in d.items():
        print(f"dither {name}: +-{v['idler_dnu_GHz']:.3f} GHz = signal +-{v['signal_dlambda_pm']:.2f} pm (or pump +-{v['or_pump_dlambda_pm']:.2f} pm)")


if __name__ == "__main__":
    main()
