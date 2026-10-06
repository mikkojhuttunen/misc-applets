import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cmpc"))  # core modules
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import coherence_model as cm
for f in Path("/usr/share/fonts/truetype/crosextra").glob("Carlito-*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C = 299792458.0; dnu = np.concatenate([np.linspace(0, 2e9, 800), np.linspace(2e9, 30e9, 800)])
SI = pickle.load(open("results/single_port_scan.pkl", "rb")); GE = pickle.load(open("results/single_port_scan_ge.pkl", "rb"))
KS = (1, 2, 4, 8)


def metrics(D, cell, kind, nu0, dndT, line):
    r = []
    for K in KS:
        pl = D[(cell, kind, K)]; p = cm.merge_incoherent(pl) if K > 1 else pl[0]
        g = cm.autocovariance(p, dnu)
        r.append((np.mean([q.L_mean for q in pl]), cm.contrast(p, 1e6, True), cm.speckle_noise_A(p, dnu, g, 1e-3, nu0, dndT, 1e6, line, True, 0.0) / np.mean([q.L_mean for q in pl])))
    r = np.array(r); return r / r[0]
GEM = {k: metrics(GE, "curved", k, 1900.08 * 29979245800.0, 3e-4, 2 * 0.058 * 29979245800.0) for k in ("ports", "angles", "positions")}
SIM = {k: metrics(SI, "curved", k, C / 1.5317e-6, 1.5e-4, 5.7e9) for k in ("ports", "angles", "positions")}
fig, ax = plt.subplots(1, 2, figsize=(6.69, 3.1)); col = {"ports": "C3", "angles": "C0", "positions": "C2"}
lab = {"ports": "K ports", "angles": "1 port, K angles", "positions": "1 port, K sub-apertures"}
for k in col:
    ax[0].plot(KS, GEM[k][:, 0], "o-", color=col[k], label=lab[k]); ax[1].plot(KS, GEM[k][:, 2], "o-", color=col[k])
ax[0].plot(KS, SIM["ports"][:, 0], ":", color="C3", lw=1.2); ax[0].text(1.1, 0.80, "loss-limited\n(Si, dotted)", fontsize=10, color="C3", va="top")
ax[1].plot(KS, 1 / np.sqrt(KS), "k:", lw=1); ax[1].text(5, 0.43, "1/√K", fontsize=10)
for a in ax: a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS); a.set_xlabel("number of incoherent beams K")
ax[0].set(ylabel=r"path $\langle L\rangle$ per beam (relative)", ylim=(0.3, 1.1)); ax[0].set_title("(a) light lost through ports", loc="left"); ax[0].legend(loc="lower left", frameon=False, borderaxespad=0.1)
ax[1].set(ylabel="detection limit (relative)", ylim=(0.15, 1.1)); ax[1].set_title("(b) speckle-limited noise", loc="left"); ax[1].set_yscale("log")
pass
fig.tight_layout(pad=0.4); Path("results/figs_prelim").mkdir(parents=True, exist_ok=True); fig.savefig("results/figs_prelim/fig_single_port.pdf")
print("GE (port-limited) rows K=1,2,4,8 as [L, contrast, detection limit] relative:")
for k in col: print(k, np.round(GEM[k], 3).tolist())
print("SI (loss-limited):")
for k in col: print(k, np.round(SIM[k], 3).tolist())
print("regular cell, Si, contrast K=8: ports", round(cm.contrast(cm.merge_incoherent(SI[("regular","ports",8)]),1e6,True),3), "angles", round(cm.contrast(cm.merge_incoherent(SI[("regular","angles",8)]),1e6,True),3), " K=1", round(cm.contrast(SI[("regular","ports",1)][0],1e6,True),3))
