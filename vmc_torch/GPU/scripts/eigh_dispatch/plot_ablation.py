"""
Plot eigh dispatch ablation results: timing comparison and speedup.

Reads data from data/eigh_dispatch/eigh_ablation.npy and produces:
  - eigh_ablation_timing.{png,pdf}   (stock vs xsyev, per B)
  - eigh_ablation_speedup.{png,pdf}  (speedup ratio)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eigh_dispatch"
OUT_DIR = DATA_DIR  # save figures next to data

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
results = np.load(DATA_DIR / "eigh_ablation.npy", allow_pickle=True).item()

DTYPE_LIST = ["f32", "f64"]
B_LIST = [1, 64, 1024]
N_LIST = [8, 16, 32, 33, 48, 64, 128, 256, 512, 513]
METHODS = ["stock", "xsyev"]
DISPATCH_THRESHOLD = 32

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)

METHOD_COLORS = {"stock": "#d62728", "xsyev": "#1f77b4"}
METHOD_LABELS = {"stock": "stock eigh", "xsyev": "XsyevBatched"}
DTYPE_LINESTYLE = {"f32": "-", "f64": "--"}
DTYPE_MARKER = {"f32": "o", "f64": "s"}

# ===================================================================
# Plot 1: Timing comparison  (3 columns, one per B)
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

for col, B in enumerate(B_LIST):
    ax = axes[col]
    for method in METHODS:
        for dtype in DTYPE_LIST:
            ns, means, stds = [], [], []
            for n in N_LIST:
                key = (dtype, B, n, method)
                if key in results:
                    m, s = results[key]
                    ns.append(n)
                    means.append(m)
                    stds.append(s)
            if not ns:
                continue
            ns = np.array(ns)
            means = np.array(means)
            stds = np.array(stds)

            label = f"{METHOD_LABELS[method]} ({dtype})"
            ax.errorbar(
                ns,
                means,
                yerr=stds,
                fmt=DTYPE_MARKER[dtype] + DTYPE_LINESTYLE[dtype],
                color=METHOD_COLORS[method],
                label=label,
                capsize=2,
                capthick=1,
                alpha=0.85,
            )

    # Dispatch threshold line
    ax.axvline(
        DISPATCH_THRESHOLD,
        color="gray",
        ls=":",
        lw=1.2,
        label="dispatch threshold" if col == 0 else None,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size $n$")
    ax.set_title(f"$B = {B}$")
    ax.set_xticks(N_LIST)
    ax.set_xticklabels(
        [str(n) for n in N_LIST], rotation=45, ha="right"
    )
    ax.minorticks_off()
    ax.grid(True, which="major", ls=":", alpha=0.4)

axes[0].set_ylabel("Time (ms)")
# Single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 1.08),
    frameon=False,
)
fig.suptitle(
    "Eigendecomposition timing: stock eigh vs XsyevBatched",
    y=1.14,
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(
        OUT_DIR / f"eigh_ablation_timing.{ext}",
        bbox_inches="tight",
    )
print(f"Saved timing plots to {OUT_DIR}/eigh_ablation_timing.{{png,pdf}}")
plt.close(fig)

# ===================================================================
# Plot 2: Speedup  (stock / xsyev)
# ===================================================================
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

B_MARKERS = {1: "o", 64: "s", 1024: "D"}
DTYPE_COLORS = {"f32": "#1f77b4", "f64": "#ff7f0e"}

for B in B_LIST:
    for dtype in DTYPE_LIST:
        ns, speedups = [], []
        for n in N_LIST:
            k_stock = (dtype, B, n, "stock")
            k_xsyev = (dtype, B, n, "xsyev")
            if k_stock in results and k_xsyev in results:
                m_stock = results[k_stock][0]
                m_xsyev = results[k_xsyev][0]
                if m_xsyev > 0:
                    ns.append(n)
                    speedups.append(m_stock / m_xsyev)
        if not ns:
            continue
        ax2.plot(
            ns,
            speedups,
            marker=B_MARKERS[B],
            ls=DTYPE_LINESTYLE[dtype],
            color=DTYPE_COLORS[dtype],
            label=f"B={B}, {dtype}",
            alpha=0.85,
        )

ax2.axhline(1.0, color="gray", ls="-", lw=1, alpha=0.6)
ax2.axvline(
    DISPATCH_THRESHOLD, color="gray", ls=":", lw=1.2, label="dispatch threshold"
)

ax2.set_xscale("log")
ax2.set_xlabel("Matrix size $n$")
ax2.set_ylabel("Speedup (stock / XsyevBatched)")
ax2.set_title(
    "Speedup from XsyevBatched dispatch",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(N_LIST)
ax2.set_xticklabels([str(n) for n in N_LIST], rotation=45, ha="right")
ax2.minorticks_off()
ax2.grid(True, which="major", ls=":", alpha=0.4)
ax2.legend(loc="best", framealpha=0.9)
fig2.tight_layout()

for ext in ("png", "pdf"):
    fig2.savefig(
        OUT_DIR / f"eigh_ablation_speedup.{ext}",
        bbox_inches="tight",
    )
print(f"Saved speedup plots to {OUT_DIR}/eigh_ablation_speedup.{{png,pdf}}")
plt.close(fig2)
