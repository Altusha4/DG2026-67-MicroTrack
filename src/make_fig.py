r"""
Build Fig. 1 for the IEEE DG 2026 paper: total actuator energy per period,
mean +/- SD over the multi-seed run.

OUTPUTS
  generalization.pdf  <- use this one in the manuscript (vector, fonts embedded)
  generalization.svg  <- vector source for the repository / web
  generalization.png  <- raster preview only, 600 dpi

WHY PDF AND NOT SVG IN THE PAPER
  pdflatex cannot \includegraphics an SVG. The PDF is the same vector artwork
  and is what IEEE expects; the SVG is kept for the repository and for anyone
  who wants to edit the figure.

TYPOGRAPHY
  IEEE asks that lettering in figures be legible at final printed size and no
  smaller than the caption. At single-column width (3.5 in) that means ~8 pt,
  set in the same serif family as the body text. Matplotlib ships STIXGeneral,
  which is metrically close to the Times face IEEEtran uses, so the figure does
  not look pasted in from another document. Fonts are embedded as Type 42;
  IEEE PDF eXpress rejects Type 3.

USAGE
  python make_fig.py [--csv results/multiseed_raw.csv] [--out-dir .]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC = "energy_kwh_month"
PERIODS = ["P1", "P2"]
XLABELS = ["Period 1\n(in-period)", "Period 2\n(held-out)"]
ALGOS = ["Threshold", "Q-Learning"]
COLORS = {"Threshold": "#C0392B", "Q-Learning": "#27AE60"}

def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    })

def arrow_axes(ax):
    """Show only the left and bottom spines, each ending in an arrowhead."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xy, marker in [((1.0, 0.0), ">"), ((0.0, 1.0), "^")]:
        ax.plot(*xy, marker, transform=ax.transAxes, color="black",
                markersize=3.5, clip_on=False, zorder=5)

def build(csv, out_dir):
    ms = pd.read_csv(csv)
    style()

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(PERIODS))
    w = 0.34
    top = 0.0
    means, errs = {}, {}

    for k, algo in enumerate(ALGOS):
        m = [ms[(ms.period == p) & (ms.algorithm == algo)][METRIC].mean() for p in PERIODS]
        s = [ms[(ms.period == p) & (ms.algorithm == algo)][METRIC].std(ddof=1) for p in PERIODS]
        s = [0.0 if np.isnan(v) else v for v in s]
        means[algo] = m
        errs[algo] = s
        top = max(top, max(np.array(m) + np.array(s)))
        ax.bar(x + k * w, m, w, yerr=s, capsize=2.5, label=algo,
               color=COLORS[algo], edgecolor="white", linewidth=0.6,
               error_kw=dict(lw=0.8, capthick=0.8), zorder=3)
        for xi, (mi, si) in enumerate(zip(m, s)):
            ax.text(xi + k * w, mi + si + top * 0.025, f"{mi:.1f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, top * 1.34)
    ax.set_xlim(-0.42, len(PERIODS) - 1 + 2 * w + 0.30)

    thr_m, thr_s = means["Threshold"], errs["Threshold"]
    for xi, label in enumerate(["-26%", "+96%"]):
        y = thr_m[xi] + thr_s[xi] + top * 0.10
        ax.annotate("", xy=(xi + w, y), xytext=(xi, y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
        ax.text(xi + w / 2, y + top * 0.02, label, ha="center", fontsize=7.5,
                fontweight="bold", color="#333333")

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(XLABELS)
    ax.set_ylabel("Total actuator energy (kWh/month)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    arrow_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
              handlelength=1.1, handletextpad=0.5, borderpad=0.1)

    fig.tight_layout(pad=0.25)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext, kw in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        path = os.path.join(out_dir, f"generalization.{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01, **kw)
        written.append(path)
    plt.close(fig)
    return written

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Fig. 1 (PDF, SVG, PNG).")
    ap.add_argument("--csv", default="results/multiseed_raw.csv")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    for p in build(a.csv, a.out_dir):
        print("wrote", p)
