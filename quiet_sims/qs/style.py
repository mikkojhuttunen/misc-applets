"""Matplotlib style of the proposal figures (Fig. 1-3 PDFs): Liberation Sans 10 pt, width <= 17 cm, the house
colours read from the existing PDFs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CM = 1 / 2.54
MAXW = 16.8 * CM                       # the existing figures are 475.9 pt = 16.8 cm wide
COL = {"si": "#D62728", "sin": "#154E7A", "sin_light": "#7CB2DC", "r_herriott": "#7CB2DC", "c_herriott": "#154E7A",
       "r_cmpc": "#B892DC", "c_cmpc": "#5B2A86", "target": "#D62728", "angle": "#2CA02C", "both": "#000000",
       "lambda": "#E377C2", "none": "#ABABAB", "shot": "#595959", "blue": "#1F77B4", "orange": "#FF7F0E", "inclusion": "#E85A0C"}


def setup():
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42,
                         "axes.linewidth": 0.8, "lines.linewidth": 1.4, "mathtext.fontset": "custom",
                         "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                         "mathtext.bf": "Liberation Sans:bold", "legend.frameon": False})
    return plt
