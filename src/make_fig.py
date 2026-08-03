r"""
Regenerate generalization.png as a COMPACT SINGLE-COLUMN figure for IEEEtran.

The wide 9x3.4in two-panel version from the analysis run is unreadable at
\columnwidth (~3.5in): the labels shrink roughly threefold. This makes one
panel, sized for a single column, showing total actuator energy -- the
headline reversal. Fan activation is already in Table III, so the figure does
not need to repeat it.

Input : results/multiseed_raw.csv   (written by microtrack_revision.py)
Output: results/generalization.png (300 dpi, 3.4 x 2.5 in)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--multiseed-csv", type=Path,
                    default=ROOT / "results" / "multiseed_raw.csv",
                    help="per-seed metrics written by microtrack_revision.py")
parser.add_argument("--out", type=Path,
                    default=ROOT / "results" / "generalization.png",
                    help="output path for the figure")
args = parser.parse_args()

CSV = args.multiseed_csv
OUT = args.out
METRIC = "energy_kwh_month"

ms = pd.read_csv(CSV)
periods = ["P1", "P2"]
labels = ["Period 1\n(in-period)", "Period 2\n(held-out)"]
algos = ["Threshold", "Q-Learning"]
colors = {"Threshold": "#C0392B", "Q-Learning": "#27AE60"}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(len(periods))
w = 0.36
top = 0.0

for k, algo in enumerate(algos):
    m = [ms[(ms.period == p) & (ms.algorithm == algo)][METRIC].mean() for p in periods]
    s = [ms[(ms.period == p) & (ms.algorithm == algo)][METRIC].std(ddof=1) for p in periods]
    s = [0.0 if np.isnan(v) else v for v in s]
    top = max(top, max(np.array(m) + np.array(s)))
    ax.bar(x + k * w, m, w, yerr=s, capsize=3, label=algo,
           color=colors[algo], edgecolor="white", linewidth=0.8, zorder=3)
    for xi, (mi, si) in enumerate(zip(m, s)):
        ax.text(xi + k * w, mi + si + top * 0.02, f"{mi:.1f}",
                ha="center", va="bottom", fontsize=7)

# annotate the reversal
ax.annotate("", xy=(1 + w, top * 0.60), xytext=(1, top * 0.60),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"))
ax.text(1 + w / 2, top * 0.63, "+96%", ha="center", fontsize=7.5,
        fontweight="bold", color="#333333")
ax.annotate("", xy=(0 + w, top * 0.45), xytext=(0, top * 0.45),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"))
ax.text(0 + w / 2, top * 0.48, "-26%", ha="center", fontsize=7.5,
        fontweight="bold", color="#333333")

ax.set_ylim(0, top * 1.30)
ax.set_xticks(x + w / 2)
ax.set_xticklabels(labels)
ax.set_ylabel("Total actuator energy\n(kWh/month)")
ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
          handlelength=1.2, borderpad=0.2)

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
