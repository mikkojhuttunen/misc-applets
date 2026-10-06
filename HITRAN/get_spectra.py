#!/usr/bin/env python3
"""
get_spectra.py -- HITRAN (classic HAPI) absorption spectra for the CMPC proposal.

For each (analyte, spectral window) it writes, into ./spectra_out/:
    spec_<tag>.csv      absorption coefficient alpha [1/cm] of the analyte at 1 ppm
    peaks_<tag>.csv     the strongest absorption peaks (nu, lambda, alpha at 1 ppm)
    interf_<gas>_<tag>.csv   H2O / CO2 background at breath-like mixing ratios
    overview_<tag>.png  quick-look plot (if matplotlib is installed)
and one summary file  peaks_all.csv  with the strongest peaks of every window.

Requirements:   pip install hitran-api numpy scipy        (matplotlib optional)
Run:            python get_spectra.py
Needs internet access to hitran.org the first time (data is cached in ./hitran_data/).
If the fetch asks for credentials, register at hitran.org, or download the lines
from the HITRANonline web interface and put the files in ./hitran_data/.

Everything you may want to change is in the CONFIG block below.
Alpha at other mixing ratios: alpha scales linearly with x for trace gases
(air-broadened regime), so alpha(1 ppb) = alpha(1 ppm) / 1000.
"""
import csv
import os
import sys

import numpy as np
from scipy.signal import find_peaks

try:
    from hapi import db_begin, fetch, absorptionCoefficient_Voigt
except ImportError:
    sys.exit("hapi not found. Install with:  pip install hitran-api")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
T_K = 310.0          # gas temperature [K] (breath-like; set to your cell temperature)
P_ATM = 1.0          # total pressure [atm]
X_ANALYTE = 1e-6     # analyte mixing ratio used for the saved spectra (1 ppm)
STEP = 0.002         # wavenumber step [cm^-1]
N_PEAKS = 15         # peaks saved per window
OUTDIR = "spectra_out"
DBDIR = "hitran_data"

# Starting guesses -- check the windows/molecules against the lines you really want.
# (label, HITRAN molecule id, isotopologue id, (nu_min, nu_max) in cm^-1, stage)
TARGETS = [
    ("CH4",  6, 1, (6040, 6180), "1.6um"),
    ("NH3", 11, 1, (6480, 6620), "1.5um"),
    ("CH4",  6, 1, (2950, 3100), "3.3um"),
    ("C2H6", 27, 1, (2880, 3000), "3.3um"),
    ("NO",   8, 1, (1860, 1930), "5.3um"),
]

# Interferents at breath-like mixing ratios (fraction of total gas):
# (label, HITRAN molecule id, isotopologue id, mixing ratio)
INTERFERENTS = [
    ("H2O", 1, 1, 0.06),
    ("CO2", 2, 1, 0.045),
]
# ----------------------------------------------------------------------------


def alpha(label, mol_id, iso_id, window, x):
    """Absorption coefficient [1/cm] for mixing ratio x in air at (T_K, P_ATM)."""
    table = f"{label}_{int(window[0])}_{int(window[1])}"
    fetch(table, mol_id, iso_id, window[0], window[1])
    nu, a = absorptionCoefficient_Voigt(
        SourceTables=table,
        Environment={"T": T_K, "p": P_ATM},
        Diluent={"air": 1.0 - x, "self": x},
        HITRAN_units=False,            # -> cm^-1, includes the partial pressure
        WavenumberRange=window,
        WavenumberStep=STEP,
    )
    return np.asarray(nu), np.asarray(a)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    db_begin(DBDIR)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    summary = []
    for label, mol, iso, win, stage in TARGETS:
        tag = f"{label}_{stage}"
        print(f"[{tag}] fetching {label} {win[0]}-{win[1]} cm^-1 ...")
        try:
            nu, a = alpha(label, mol, iso, win, X_ANALYTE)
        except Exception as e:                      # keep going with other windows
            print(f"  FAILED: {e}")
            continue

        write_csv(os.path.join(OUTDIR, f"spec_{tag}.csv"),
                  ["nu_cm-1", "alpha_cm-1_at_1ppm"], zip(nu[::5], a[::5]))

        # strongest peaks (min separation 0.1 cm^-1)
        sep = max(1, int(0.1 / STEP))
        pk, _ = find_peaks(a, height=0.02 * a.max(), distance=sep)
        top = pk[np.argsort(a[pk])[::-1][:N_PEAKS]]
        rows = [(nu[i], 1e4 / nu[i], a[i]) for i in top]
        write_csv(os.path.join(OUTDIR, f"peaks_{tag}.csv"),
                  ["nu_cm-1", "lambda_um", "alpha_cm-1_at_1ppm"], rows)
        summary += [(label, stage, *r) for r in rows]

        interf = {}
        for lab2, m2, i2, x2 in INTERFERENTS:
            try:
                nu2, a2 = alpha(lab2, m2, i2, win, x2)
            except Exception as e:
                print(f"  interferent {lab2} FAILED: {e}")
                continue
            interf[lab2] = (nu2, a2, x2)
            write_csv(os.path.join(OUTDIR, f"interf_{lab2}_{tag}.csv"),
                      ["nu_cm-1", f"alpha_cm-1_at_x={x2:g}"], zip(nu2[::5], a2[::5]))

        if plt is not None:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            for lab2, (nu2, a2, x2) in interf.items():
                ax.semilogy(nu2, a2, lw=0.6, label=f"{lab2} ({x2*100:g} %)")
            ax.semilogy(nu, a, lw=0.8, color="k", label=f"{label} (1 ppm)")
            ax.set_xlabel("wavenumber (cm$^{-1}$)")
            ax.set_ylabel("absorption coefficient (cm$^{-1}$)")
            ax.set_title(f"{label}, {stage}, T={T_K:g} K, p={P_ATM:g} atm")
            ax.set_ylim(max(a.max() * 1e-5, 1e-12), None)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, f"overview_{tag}.png"), dpi=150)
            plt.close(fig)

    write_csv(os.path.join(OUTDIR, "peaks_all.csv"),
              ["molecule", "stage", "nu_cm-1", "lambda_um", "alpha_cm-1_at_1ppm"],
              summary)
    print(f"Done. Results in ./{OUTDIR}/  (send peaks_all.csv and the spec_*/interf_* files).")


if __name__ == "__main__":
    main()
