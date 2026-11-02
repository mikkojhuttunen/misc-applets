"""Markdown tables of the step results (python scripts/report.py step1|step2|step3|step4|step5)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def step1():
    d = json.loads((R / "step1_mirrors.json").read_text())
    films = {f"{f['n_eff_nominal']:.2f}": f for f in d["films"]}
    Ns = ["1", "2", "4", "6", "8", "12", "20"]
    for cell, c in d["cells"].items():
        print(f"\n{cell}: bounce-averaged R vs N (optimised design per N)\n")
        print("| n_eff (film) | above crit. | " + " | ".join(f"N={n}" for n in Ns) + " | best design (N=20) | σ=5 nm | σ=10 nm | diffraction bound |")
        print("|---" * (len(Ns) + 6) + "|")
        for ne, f in c["films"].items():
            fl = films[ne]
            b = f["best"]
            des = (f"m {b['m_gap']}/{b['m_tooth']}, gap {sum(b['d_gap_nm'])/len(b['d_gap_nm']):.0f} nm, tooth {sum(b['d_tooth_nm'])/len(b['d_tooth_nm']):.0f} nm"
                   + (f", 2-band" if b["s2"] is not None else "") + (f", chirp {b['chirp']:+.1f}" if b["chirp"] else ""))
            print(f"| {ne} ({fl['thickness_nm']:.0f} nm @ {fl['lambda_um']:.2f} µm) | {100*f['share_above_critical']:.1f} % | "
                  + " | ".join(f"{f['by_N'][n]['R_avg']:.4f}" for n in Ns)
                  + f" | {des} | {f['roughness_R_avg']['5.0']:.4f} | {f['roughness_R_avg']['10.0']:.4f} | {f['diffraction_bound']['R_avg_reoptimised_N8_first_order']:.2f} |")



def step2():
    d = json.loads((R / "step2_summary.json").read_text())
    b = d["baseline_Si_R0.99"]
    print("\n| cell | film (n_eff, Γ) | R(χ) | static fringe: none / spectral / angle / both (chaotic median) | both, regular | layouts < 5e-7 | Γ⟨L⟩ | T | ⟨R⟩ bounce | NH3 1 s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    fb = b["fringe"]["modes"]
    print(f"| circle (baseline) | Si 250 nm (2.0, 0.20) | R 0.99 | {fb['none']['chaotic_median']:.1e} / {fb['spectral']['chaotic_median']:.1e} / {fb['angle']['chaotic_median']:.1e} / "
          f"{fb['both']['chaotic_median']:.1e} | {fb['both']['regular']:.1e} | {100*fb['both']['share_below']:.0f} % | {b['fringe']['gammaL_cm']:.2f} cm | "
          f"{b['fringe']['T_median']:.2f} | 0.99 | {1e3*b['nh3']['ppm']:.0f} ppb |")
    for k, c in sorted(d["cases"].items(), key=lambda kv: (kv[1]["cell"], kv[1]["n_eff"], "rough" in kv[0])):
        m = c["modes"]
        film = f"SiNx {c['thickness_nm']} nm ({c['n_eff']:.2f}, {c['gamma']:.2f})"
        rt = "R(χ), σ 5 nm" if "rough" in k else "R(χ)"
        print(f"| {c['cell']} | {film} | {rt} | {m['none']['chaotic_median']:.1e} / {m['spectral']['chaotic_median']:.1e} / {m['angle']['chaotic_median']:.1e} / "
              f"{m['both']['chaotic_median']:.1e} | {m['both']['regular']:.1e} | {100*m['both']['share_below']:.0f} % | {c['gammaL_cm']:.2f} cm | "
              f"{c['T_median']:.2f} | {c['R_bounce_median']:.4f} | {1e3*c['detection_limit']['ppm']:.0f} ppb |")



def step3():
    d = json.loads((R / "step3_summary.json").read_text())
    print("\n| n_eff | m | χ | passes to exit (regular) | bounces above critical | ⟨R⟩ bounce (N=20 mirror) | Γ⟨L⟩ | T | static fringe both (chaotic / regular) | NH3 1 s [68 % CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(d["cases"].items(), key=lambda kv: (-kv[1]["n_eff"], -kv[1]["m"])):
        dl = v["detection_limit"]
        print(f"| {v['n_eff']:.2f} | {v['m']} | {v['chi_deg']:.1f}° | {v['passes_to_exit_regular']} | {100*v['above_critical_median']:.0f} % | {v['R_bounce_median']:.4f} | "
              f"{v['gammaL_cm']:.2f} cm | {v['T_median']:.2f} | {v['modes']['both']['chaotic_median']:.1e} / {v['modes']['both']['regular']:.1e} | "
              f"{1e3*dl['ppm']:.0f} ppb [{1e3*dl['ci68_ppm'][0]:.0f}–{1e3*dl['ci68_ppm'][1]:.0f}] |")



def step4():
    g = json.loads((R / "step4_gas.json").read_text())
    print("\n| analyte | strongest line near target | α peak, 1 ppm (/cm) | H2O / CO2 background there (/cm) | Fig. 3 line: α (HAPI / Fig. 3 data) |")
    print("|---|---|---|---|---|")
    for k, v in g["lines"].items():
        f = g["fig3_lines"][k]
        print(f"| {k} | {v['nu0_cm']:.3f} /cm ({v['lambda0_um']:.4f} µm) | {v['alpha_peak_per_ppm_cm']:.3e} | {v['bg_at_peak_per_cm']['H2O']:.1e} / {v['bg_at_peak_per_cm']['CO2']:.1e} | "
              f"{f['nu0_cm']:.3f} /cm: {f['alpha_peak_per_ppm_cm']:.3e} / {f['fig3_alpha_peak_per_ppm_cm']:.3e} |")
    r = g["ratio_CH4_3p31_over_1p65"]
    print(f"\nCH4 3.31 / 1.65 µm peak ratio: {r['strongest_lines']:.0f}× (strongest lines), {r['fig3_lines']:.0f}× (Fig. 3 lines).")
    d = json.loads((R / "step4_summary.json").read_text())
    fmt = lambda t: "not within 100 s" if t is None else (f"{1e3*t:.2g} ms" if t < 1 else f"{t:.0f} s")
    print("\n| cell | film (n_eff, Γ) | loss | Γ⟨L⟩ | T | ⟨R⟩ | CH4 1 s | CH4 τ(10 ppb) / τ(1 ppb) | C2H6 1 s | C2H6 τ(10 ppb) / τ(1 ppb) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(d["cases"].items(), key=lambda kv: (kv[1]["cell"], kv[1]["film_nm"], kv[1]["dBcm"])):
        a, b = v["limits"]["CH4"], v["limits"]["C2H6"]
        print(f"| {v['cell']} | {v['film_nm']/1e3:.1f} µm ({v['n_eff']:.2f}, {v['gamma']:.2f}) | {v['dBcm']} dB/cm | {v['gammaL_cm']:.2f} cm | {v['T_median']:.2f} | {v['R_bounce_median']:.4f} | "
              f"{a['ppb_1s']:.1f} ppb | {fmt(a['tau_10ppb_s'])} / {fmt(a['tau_1ppb_s'])} | {b['ppb_1s']:.1f} ppb | {fmt(b['tau_10ppb_s'])} / {fmt(b['tau_1ppb_s'])} |")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
