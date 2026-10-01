#!/usr/bin/env python3
"""Manuscript Figures 5-7 (matplotlib, 220 dpi), drawn from the tables written by analyze_external.py.
Run after analyze_external.py: python external_validation/code/make_figures.py
Palette: validated categorical slots (dataviz reference): blue, orange, aqua, yellow, magenta, green.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_paths as RP
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5, "legend.fontsize": 11,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8, "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.labelcolor": C["ink"], "text.color": C["ink"]})
OUT = str(RP.FIGURES); import os; os.makedirs(OUT, exist_ok=True)
T = RP.TABLES
WINDOWS = [1, 2, 3, 4, 5, 7, 10]; PASSES = [1, 2, 3, 5]
seq = LinearSegmentedColormap.from_list("blueseq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

def style(ax):
    ax.grid(True, color=C["grid"], linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------------- Figure 6: development vs external grids
dev = pd.read_csv(RP.DEV_TEST_SUMMARY).pivot(index="passes", columns="window", values="mean_nrmse").loc[PASSES, WINDOWS]
ext = pd.read_csv(T / "S1_sweep_test_grid.csv").set_index("Passes p")
ext.columns = WINDOWS; ext = ext.loc[PASSES, WINDOWS]
fig, axes = plt.subplots(1, 2, figsize=(12.25, 5.6))
for ax, grid, title, fmt, norm in ((axes[0], dev, "(a) Development panel: BIST 30 and S&P 500, 14 features", "{:.3f}", LogNorm(vmin=0.19, vmax=2.7)),
                                   (axes[1], ext, "(b) External panel: 34 country ETFs, OHLCV (asset-level mean)", "{:.4f}", None)):
    im = ax.imshow(grid.values, cmap=seq, aspect="auto", norm=norm)
    ax.set_xticks(range(len(WINDOWS))); ax.set_xticklabels(WINDOWS); ax.set_yticks(range(len(PASSES))); ax.set_yticklabels(PASSES)
    ax.set_xlabel("Temporal window width w"); ax.set_ylabel("Refinement passes p"); ax.set_title(title, loc="left", fontsize=12.5)
    vmax = grid.values.max(); vmin = grid.values.min()
    for i in range(len(PASSES)):
        for j in range(len(WINDOWS)):
            v = grid.values[i, j]
            frac = (np.log(v) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)) if norm is not None else (v - vmin) / (vmax - vmin)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5, color="white" if frac > 0.55 else C["ink"])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Test mean NRMSE", fontsize=10.5); cb.outline.set_visible(False)
    if norm is not None:
        cb.set_ticks([0.2, 0.3, 0.5, 1.0, 2.0]); cb.set_ticklabels(["0.2", "0.3", "0.5", "1.0", "2.0"])
# mark the prespecified cell
for ax in axes:
    ax.add_patch(plt.Rectangle((-0.5, 3 - 0.5), 1, 1, fill=False, edgecolor=C["orange"], linewidth=2.2))
fig.suptitle("Window and pass interaction: development study versus external validation", fontsize=14, y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/figure_6_grid_dev_vs_external.png", dpi=220); plt.close(fig)

# ---------------- Figure 5: block-length curves on the external panel
E2 = pd.read_csv(T / "E2_external_by_block.csv"); S2 = pd.read_csv(T / "S2_sweep_by_block.csv")
A2 = pd.read_csv(T / "A2_amended_by_block.csv") if (T / "A2_amended_by_block.csv").exists() else None
blocks = E2["Block length"].tolist(); x = np.arange(len(blocks))
fig, axes = plt.subplots(1, 2, figsize=(12.25, 7.0))
ax = axes[0]; style(ax)
series = [("Linear interpolation", E2["Linear interpolation"], C["orange"], "^"),
          ("ITRI w = 1, p = 5 (prespecified)", E2["ITRI w=1, p=5 (prespecified)"], C["blue"], "o"),
          ("ITRI w = 7, p = 5 (external-validation choice)", S2["ITRI w7_p5"], C["aqua"], "s"),
          ("PCHIP", E2["PCHIP"], C["yellow"], "D"),
          ("Chained ridge", E2["Chained ridge"], C["magenta"], "v")]
for name, y, col, mk in series:
    ax.plot(x, y, color=col, marker=mk, markersize=7, linewidth=2, label=name)
ax.set_xticks(x); ax.set_xticklabels(blocks); ax.set_xlabel("Contiguous block length in trading days"); ax.set_ylabel("Test mean NRMSE (asset-level)")
ax.set_title("(a) Temporal and cross-sectional comparators", loc="left", fontsize=12.5)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
ax = axes[1]; style(ax)
series2 = [("Linear interpolation", E2["Linear interpolation"], C["orange"], "^"),
           ("ITRI w = 1, p = 5", E2["ITRI w=1, p=5 (prespecified)"], C["blue"], "o"),
           ("SAITS, frozen protocol v1", E2["SAITS (v1)"], C["yellow"], "D"),
           ("BRITS, frozen protocol v1", E2["BRITS (v1)"], C["magenta"], "v")]
if A2 is not None:
    for col_name, label, col, mk in (("SAITS + window norm. (30 ep.)", "SAITS + window norm., 30 epochs (exploratory)", C["aqua"], "s"),
                                     ("SAITS + window norm. (200 ep.)", "SAITS + window norm., 200 epochs (exploratory)", C["green"], "P"),
                                     ("BRITS + window norm. (30 ep.)", "BRITS + window normalization, 30 epochs (exploratory)", C["violet"], "X")):
        if col_name in A2.columns:
            series2.append((label, A2[col_name], col, mk))
for name, y, col, mk in series2:
    ax.plot(x, y, color=col, marker=mk, markersize=7, linewidth=2, label=name)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(blocks); ax.set_xlabel("Contiguous block length in trading days")
ax.set_ylabel("Test mean NRMSE (asset-level, log scale)"); ax.set_title("(b) Learned comparators, frozen protocol and amended", loc="left", fontsize=12.5)
ax.set_yticks([0.2, 0.3, 0.5, 1.0, 2.0]); ax.set_yticklabels(["0.2", "0.3", "0.5", "1.0", "2.0"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
fig.suptitle("External validation: reconstruction error by contiguous gap length", fontsize=14, y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/figure_5_external_block_length.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 7: level-shift scatter
m = pd.read_csv(T / "level_shift_joined.csv")
fig, ax = plt.subplots(figsize=(12.25, 6.2)); style(ax)
pts = [("Linear interpolation", "linear", C["orange"], "^"), ("ITRI w = 1, p = 5", "itri_w1_p5", C["blue"], "o"),
       ("Chained ridge", "chained_ridge", C["magenta"], "v"), ("KNN (k = 5)", "knn5", C["yellow"], "D"),
       ("SAITS, frozen protocol v1", "saits", C["red"], "s")]
if "saits_winnorm_V1" in m.columns and "saits_winnorm_V2" not in m.columns:
    pts.append(("SAITS + window normalization, 30 epochs (exploratory)", "saits_winnorm_V1", C["aqua"], "P"))
if "saits_winnorm_V2" in m.columns:
    pts.append(("SAITS + window normalization, 200 epochs (exploratory)", "saits_winnorm_V2", C["green"], "X"))
from scipy.stats import spearmanr
for name, col, color, mk in pts:
    rho = spearmanr(m["test_mean_abs_z_price"], m[col]).statistic
    ax.scatter(m["test_mean_abs_z_price"], m[col], s=34, color=color, marker=mk, alpha=0.85, edgecolor="white", linewidth=0.6,
               label=f"{name} (Spearman rho = {rho:.2f})")
ax.set_yscale("log"); ax.set_xlabel("Mean |z| of test-period prices under training-period scaling (asset-fold)")
ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]); ax.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "1.0", "2.0", "5.0"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel("Test NRMSE (asset-fold, log scale)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10)
ax.set_title("Comparator error rises with the test-period displacement from the training distribution", fontsize=13.5, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figure_7_level_shift.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("figures written")
