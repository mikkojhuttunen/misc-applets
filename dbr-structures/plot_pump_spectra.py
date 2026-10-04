"""Plot pump spectra exported from the Waveguide DBR Calculator ("Copy CSV" buttons).

Recognised CSV headers (first line):
  CRIGF pump response : wavelength_nm,R_beam,T_beam,guided_escape,R_bare_slab
  In-plane DBR        : wavelength_nm,R,T,R_coupled_mode
  QPM transfer fn.    : pump_wavelength_nm,eta_designed_percent_per_W,eta_uniform_percent_per_W

Usage:
  python plot_pump_spectra.py crigf.csv
  python plot_pump_spectra.py crigf.csv dbr.csv --db --out spectra.png
  python plot_pump_spectra.py a.csv b.csv --labels "sp=100 nm" "sp=200 nm" --xlim 1549 1551
"""
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
    return header, data


def kind_of(header):
    if "R_beam" in header:
        return "crigf"
    if "eta_designed_percent_per_W" in header:
        return "qpm"
    if "R" in header and "T" in header:
        return "dbr"
    raise ValueError("unrecognised CSV header: " + ",".join(header))


def to_db(y, floor=1e-6):
    return 10 * np.log10(np.maximum(y, floor))


def resonance_metrics(lam, r):
    i = int(np.argmax(r))
    base = float(np.median(r))
    half = 0.5 * (r[i] + base)
    lo = i
    while lo > 0 and r[lo] > half:
        lo -= 1
    hi = i
    while hi < len(r) - 1 and r[hi] > half:
        hi += 1
    fwhm = lam[hi] - lam[lo] if 0 < lo and hi < len(r) - 1 else float("nan")
    return lam[i], r[i], fwhm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSV files copied from the calculator")
    ap.add_argument("--labels", nargs="*", help="legend label per file")
    ap.add_argument("--db", action="store_true", help="reflectance / transmittance in dB")
    ap.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), help="wavelength range in nm")
    ap.add_argument("--no-t", action="store_true", help="hide transmittance curves")
    ap.add_argument("--out", help="save the figure to this file instead of showing it")
    args = ap.parse_args()

    labels = args.labels or [p.rsplit("/", 1)[-1] for p in args.csv]
    if len(labels) != len(args.csv):
        sys.exit("give one label per CSV file")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ylabel = "reflectance / transmittance"
    for path, label in zip(args.csv, labels):
        header, d = read_csv(path)
        kind = kind_of(header)
        lam = d[:, 0]
        if kind == "qpm":
            ax.plot(lam, d[:, 1], lw=2, label=f"{label}: designed")
            ax.plot(lam, d[:, 2], lw=1, ls="--", label=f"{label}: uniform")
            ylabel = "SHG efficiency (%/W)"
            continue
        r, t = d[:, 1], d[:, 2]
        f = to_db if args.db else (lambda y: y)
        (line,) = ax.plot(lam, f(r), lw=2, label=f"{label}: R")
        if not args.no_t:
            ax.plot(lam, f(t), lw=1.2, color=line.get_color(), alpha=0.6, label=f"{label}: T")
        if kind == "crigf":
            ax.plot(lam, f(d[:, 4]), lw=0.8, ls=":", color="0.5", label=f"{label}: bare slab R")
            lp, rp, fw = resonance_metrics(lam, r)
            q = lp / fw if fw == fw and fw > 0 else float("nan")
            print(f"{label}: peak R = {rp:.4f} at {lp:.4f} nm, FWHM = {fw * 1000:.1f} pm, Q = {q:.3g}")
        else:
            i = int(np.argmax(r))
            print(f"{label}: peak R = {r[i]:.5f} at {lam[i]:.4f} nm")
        if args.db:
            ylabel = "reflectance / transmittance (dB)"

    ax.set_xlabel("pump wavelength (nm)")
    ax.set_ylabel(ylabel)
    if args.xlim:
        ax.set_xlim(*args.xlim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    if args.out:
        fig.savefig(args.out, dpi=200)
        print("saved", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
