#!/usr/bin/env python3
"""Run the complete ITRI window and refinement ablation experiment."""


# Notebook cell 3
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, wilcoxon
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")


@dataclass
class Config:
    start: str = "2015-01-01"
    end: str = "2026-01-01"
    windows: tuple = (1, 2, 3, 4, 5, 7, 10)
    passes: tuple = (1, 2, 3, 5)
    block_lengths: tuple = (1, 2, 3, 5, 10, 20)
    rates: tuple = (0.10, 0.20, 0.30)
    seeds: tuple = (11, 23, 37)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    bootstrap_reps: int = 5000
    full_test_grid: bool = True


FRESH = True
CFG = Config()

BASE = Path(__file__).resolve().parents[1] / "reproduction_run"
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "results"
FIG_DIR = OUT_DIR / "figures"
for directory in (DATA_DIR, OUT_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VAL_PATH = OUT_DIR / "validation_grid.csv"
TEST_PATH = OUT_DIR / "test_grid.csv"
SELECTED_PATH = OUT_DIR / "selected_configuration.json"

if FRESH:
    for path in (VAL_PATH, TEST_PATH, SELECTED_PATH):
        if path.exists():
            path.unlink()

MARKETS = {"BIST30": "XU030.IS", "SP500": "^GSPC"}
FEATURES = [
    "open", "high", "low", "close", "volume", "log_return", "hl_range",
    "volume_change", "sma_5_gap", "sma_10_gap", "sma_20_gap",
    "volatility_10", "rsi_14", "macd",
]

print("The full manuscript experiment will run automatically.")
print("Config:", asdict(CFG))
print("Versions:", {"numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                    "yfinance": yf.__version__, "python": platform.python_version()})

# Notebook cell 5
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_market(name, ticker, cfg):
    path = DATA_DIR / f"{name}_{cfg.start}_{cfg.end}.csv"
    if path.exists():
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raw = yf.download(ticker, start=cfg.start, end=cfg.end,
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError(f"No data were downloaded for {name} ({ticker})")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.to_csv(path)
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{name}: missing OHLCV columns: {missing}")
    return raw[needed].astype(float)


def make_features(raw):
    x = raw.copy()
    x["log_return"] = np.log(x["close"]).diff()
    x["hl_range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["volume_change"] = np.log1p(x["volume"]).diff()
    for w in (5, 10, 20):
        x[f"sma_{w}_gap"] = x["close"] / x["close"].rolling(w).mean() - 1.0
    x["volatility_10"] = x["log_return"].rolling(10).std()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = (ema12 - ema26) / x["close"].replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan).dropna()[FEATURES]


def chronological_split(df, cfg):
    n = len(df)
    a = int(n * cfg.train_ratio)
    b = int(n * (cfg.train_ratio + cfg.val_ratio))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def prepare_market(name, ticker, cfg):
    full = make_features(download_market(name, ticker, cfg))
    train, val, test = chronological_split(full, cfg)
    scaler = StandardScaler().fit(train)
    return {
        "dates": (train.index, val.index, test.index),
        "train": scaler.transform(train).astype(float),
        "val": scaler.transform(val).astype(float),
        "test": scaler.transform(test).astype(float),
        "scaler": scaler,
        "counts": (len(train), len(val), len(test)),
    }


market_data = {name: prepare_market(name, ticker, CFG)
               for name, ticker in MARKETS.items()}
pd.DataFrame({name: data["counts"] for name, data in market_data.items()},
             index=["train", "validation", "test"])

# Notebook cell 7
def make_block_mask(shape, rate, block_length, seed, edge_guard):
    # A contiguous-block mask with an exact target number of eligible cells.
    n, d = shape
    if n <= 2 * edge_guard + block_length:
        raise ValueError("The partition is too short for the selected window and block length.")
    rng = np.random.default_rng(seed)
    eligible = np.zeros((n, d), dtype=bool)
    eligible[edge_guard:n - edge_guard, :] = True
    target = max(1, int(round(rate * eligible.sum())))
    mask = np.zeros((n, d), dtype=bool)
    attempts = 0
    max_attempts = target * 100

    while mask.sum() < target and attempts < max_attempts:
        need = target - int(mask.sum())
        length = min(block_length, need)
        col = int(rng.integers(0, d))
        start_low = edge_guard
        start_high = n - edge_guard - length + 1
        start = int(rng.integers(start_low, start_high))
        segment = mask[start:start + length, col]
        if not segment.any():
            mask[start:start + length, col] = True
        attempts += 1

    if mask.sum() != target:
        raise RuntimeError(f"Could not complete the block mask: {mask.sum()} / {target}")
    return mask


def corrupt(x, mask):
    out = x.copy()
    out[mask] = np.nan
    return out


def mask_seed(base_seed, market_index, split_index, block_length, rate):
    return int(base_seed + 1009 * market_index + 101 * split_index
               + 17 * block_length + round(rate * 1000))

# Notebook cell 9
class ITRIImputer:
    def __init__(self, window):
        self.window = int(window)
        self.models_ = []

    @staticmethod
    def _initial_fill(x):
        df = pd.DataFrame(x).interpolate(method="linear", limit_direction="both")
        return df.ffill().bfill().to_numpy(float)

    def _temporal_matrix(self, col):
        w = self.window
        columns = []
        for lag in range(w, 0, -1):
            columns.append(np.roll(col, lag))
        for lead in range(1, w + 1):
            columns.append(np.roll(col, -lead))
        z = np.column_stack(columns)
        z[:w, :] = np.nan
        z[-w:, :] = np.nan
        return z

    def fit(self, x_corrupt):
        missing = np.isnan(x_corrupt)
        completed = self._initial_fill(x_corrupt)
        self.models_ = []
        for d in range(x_corrupt.shape[1]):
            z = self._temporal_matrix(completed[:, d])
            good = (~missing[:, d]) & np.isfinite(z).all(axis=1)
            if good.sum() < 2 * self.window + 2:
                self.models_.append(None)
                continue
            model = LinearRegression()
            model.fit(z[good], completed[good, d])
            self.models_.append(model)
        return self

    def transform(self, x_corrupt, passes):
        missing = np.isnan(x_corrupt)
        observed = ~missing
        completed = self._initial_fill(x_corrupt)
        deltas = []

        for _ in range(int(passes)):
            previous = completed.copy()
            for d, model in enumerate(self.models_):
                if model is None:
                    continue
                z = self._temporal_matrix(completed[:, d])
                rows = missing[:, d] & np.isfinite(z).all(axis=1)
                if rows.any():
                    completed[rows, d] = model.predict(z[rows])
            completed[observed] = x_corrupt[observed]
            if missing.any():
                deltas.append(float(np.mean(np.abs(completed[missing] - previous[missing]))))
            else:
                deltas.append(0.0)
        return completed, deltas


def reconstruction_metrics(truth, estimate, mask):
    error = estimate[mask] - truth[mask]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = max(float(np.std(truth[mask])), 1e-12)
    return {"mae": mae, "rmse": rmse, "nrmse": rmse / denom}

# Notebook cell 11
def run_grid(split_name, output_path, cfg):
    edge_guard = max(cfg.windows)
    started = time.perf_counter()
    result_rows = []
    total = (len(MARKETS) * len(cfg.block_lengths) * len(cfg.rates)
             * len(cfg.seeds) * len(cfg.windows) * len(cfg.passes))
    done = 0

    for market_index, (market, data) in enumerate(market_data.items()):
        truth = data[split_name]
        train_truth = data["train"]
        for block_length in cfg.block_lengths:
            for rate in cfg.rates:
                for seed in cfg.seeds:
                    train_mask = make_block_mask(
                        train_truth.shape, rate, block_length,
                        mask_seed(seed, market_index, 0, block_length, rate), edge_guard,
                    )
                    eval_mask = make_block_mask(
                        truth.shape, rate, block_length,
                        mask_seed(seed, market_index, 1 if split_name == "val" else 2,
                                  block_length, rate), edge_guard,
                    )
                    train_corrupt = corrupt(train_truth, train_mask)
                    eval_corrupt = corrupt(truth, eval_mask)

                    for window in cfg.windows:
                        fit_start = time.perf_counter()
                        imputer = ITRIImputer(window).fit(train_corrupt)
                        fit_seconds = time.perf_counter() - fit_start
                        for passes in cfg.passes:
                            transform_start = time.perf_counter()
                            estimate, deltas = imputer.transform(eval_corrupt, passes)
                            transform_seconds = time.perf_counter() - transform_start
                            metrics = reconstruction_metrics(truth, estimate, eval_mask)
                            result_rows.append({
                                "split": split_name,
                                "market": market,
                                "block_length": block_length,
                                "rate": rate,
                                "seed": seed,
                                "window": window,
                                "passes": passes,
                                **metrics,
                                "fit_seconds": fit_seconds,
                                "transform_seconds": transform_seconds,
                                "last_update_delta": deltas[-1],
                                "masked_cells": int(eval_mask.sum()),
                            })
                            done += 1
                            if done % 100 == 0 or done == total:
                                elapsed = time.perf_counter() - started
                                print(f"{split_name}: {done}/{total} | {elapsed:.1f} s")

    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(output_path, index=False)
    print(f"{split_name}: wrote {len(result_frame)} rows to {output_path}")
    return result_frame


validation = run_grid("val", VAL_PATH, CFG)
validation.head()

# Notebook cell 12
# Select the configuration from validation results only
validation_summary = (
    validation.groupby(["window", "passes"], as_index=False)
    .agg(mean_nrmse=("nrmse", "mean"), median_nrmse=("nrmse", "median"),
         mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
         mean_runtime=("transform_seconds", "mean"),
         mean_last_delta=("last_update_delta", "mean"))
    .sort_values(["mean_nrmse", "mean_mae", "mean_runtime"])
)

winner = validation_summary.iloc[0]
selected = {
    "window": int(winner.window),
    "passes": int(winner.passes),
    "selection_metric": "mean validation NRMSE",
    "mean_validation_nrmse": float(winner.mean_nrmse),
    "candidate_windows": list(CFG.windows),
    "candidate_passes": list(CFG.passes),
}
SELECTED_PATH.write_text(json.dumps(selected, indent=2), encoding="utf-8")
validation_summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
print(json.dumps(selected, indent=2))
validation_summary.head(12)

# Notebook cell 13
# Validation heatmap
pivot = validation_summary.pivot(index="passes", columns="window", values="mean_nrmse")
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_xlabel("Temporal window width w")
ax.set_ylabel("Refinement passes p")
ax.set_title("Validation mean NRMSE")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.iloc[i, j]:.4f}", ha="center", va="center",
                color="white" if pivot.iloc[i, j] > np.nanmedian(pivot.values) else "black",
                fontsize=9)
fig.colorbar(im, ax=ax, label="NRMSE")
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_window_pass_heatmap.png", dpi=220)
plt.show()

# Window sensitivity by block length, using each window's best pass count
best_pass_by_window = validation_summary.loc[
    validation_summary.groupby("window")["mean_nrmse"].idxmin(), ["window", "passes"]
]
filtered = validation.merge(best_pass_by_window, on=["window", "passes"])
curve = filtered.groupby(["block_length", "window"], as_index=False).nrmse.mean()
fig, ax = plt.subplots(figsize=(9, 5))
for block_length, group in curve.groupby("block_length"):
    ax.plot(group.window, group.nrmse, marker="o", label=f"block={block_length}")
ax.set_xlabel("Temporal window width w")
ax.set_ylabel("Mean validation NRMSE")
ax.set_title("Window sensitivity by contiguous-gap length")
ax.grid(alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_window_by_block_length.png", dpi=220)
plt.show()

# Notebook cell 15
if CFG.full_test_grid:
    test = run_grid("test", TEST_PATH, CFG)
else:
    original_windows, original_passes = CFG.windows, CFG.passes
    CFG.windows = (selected["window"],)
    CFG.passes = (selected["passes"],)
    test = run_grid("test", TEST_PATH, CFG)
    CFG.windows, CFG.passes = original_windows, original_passes
test_summary = (
    test.groupby(["window", "passes"], as_index=False)
    .agg(mean_nrmse=("nrmse", "mean"), median_nrmse=("nrmse", "median"),
         mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
         mean_runtime=("transform_seconds", "mean"),
         mean_last_delta=("last_update_delta", "mean"))
    .sort_values(["mean_nrmse", "mean_mae", "mean_runtime"])
)
test_summary.to_csv(OUT_DIR / "test_summary.csv", index=False)
test_summary.head(12)

# Notebook cell 16
def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted_sorted[idx] = running
    return adjusted_sorted


def bootstrap_mean_ci(values, reps, seed=2026):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return np.quantile(means, [0.025, 0.975])


stats_rows = []
if CFG.full_test_grid:
    keys = ["market", "block_length", "rate", "seed"]
    wide = test.pivot_table(index=keys, columns=["window", "passes"], values="nrmse")
    wide = wide.dropna(axis=0)
    friedman = friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
    print("Friedman:", {"statistic": float(friedman.statistic), "p": float(friedman.pvalue),
                        "paired_conditions": int(len(wide))})

    selected_col = (selected["window"], selected["passes"])
    selected_values = wide[selected_col].to_numpy()
    raw_p = []
    comparisons = []
    for col in wide.columns:
        if col == selected_col:
            continue
        competitor = wide[col].to_numpy()
        difference = selected_values - competitor
        try:
            result = wilcoxon(selected_values, competitor, alternative="two-sided",
                              zero_method="wilcox")
            p_value = float(result.pvalue)
            statistic = float(result.statistic)
        except ValueError:
            p_value, statistic = 1.0, 0.0
        ci_low, ci_high = bootstrap_mean_ci(difference, CFG.bootstrap_reps)
        comparisons.append({
            "selected_window": selected_col[0],
            "selected_passes": selected_col[1],
            "competitor_window": int(col[0]),
            "competitor_passes": int(col[1]),
            "mean_difference_selected_minus_competitor": float(difference.mean()),
            "median_difference": float(np.median(difference)),
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
            "selected_better_conditions": int(np.sum(difference < 0)),
            "paired_conditions": int(len(difference)),
            "wilcoxon_statistic": statistic,
            "p_raw": p_value,
        })
        raw_p.append(p_value)

    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(comparisons, adjusted):
        row["p_holm"] = float(p_adj)
        row["significant_0_05"] = bool(p_adj < 0.05)
    stats_df = pd.DataFrame(comparisons).sort_values("p_holm")
    stats_df.to_csv(OUT_DIR / "selected_vs_competitors_tests.csv", index=False)
    pd.DataFrame([{
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "paired_conditions": int(len(wide)),
    }]).to_csv(OUT_DIR / "friedman_test.csv", index=False)
    print(stats_df.head(15).to_string(index=False))

# Notebook cell 17
# Manuscript-ready summary tables
by_block = (
    test.groupby(["block_length", "window", "passes"], as_index=False)
    .agg(mean_nrmse=("nrmse", "mean"), mean_mae=("mae", "mean"),
         mean_rmse=("rmse", "mean"))
)
by_block.to_csv(OUT_DIR / "test_by_block_length.csv", index=False)

selected_test = test[
    (test.window == selected["window"]) & (test.passes == selected["passes"])
]
selected_by_market_block = (
    selected_test.groupby(["market", "block_length"], as_index=False)
    .agg(mean_nrmse=("nrmse", "mean"), sd_nrmse=("nrmse", "std"),
         mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
         mean_runtime=("transform_seconds", "mean"))
)
selected_by_market_block.to_csv(OUT_DIR / "selected_test_by_market_block.csv", index=False)
print(selected_by_market_block.to_string(index=False))

metadata = {
    "config": asdict(CFG),
    "selected": selected,
    "markets": MARKETS,
    "features": FEATURES,
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "yfinance": yf.__version__,
    },
    "note": "MCAR and MAR were not used; only controlled contiguous-block masks were evaluated.",
}
(OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

readme = OUT_DIR / "README.txt"
readme.write_text(
    "ITRI window and pass ablation results\n"
    "MCAR and MAR were not used. Only controlled contiguous-block masks were evaluated.\n"
    f"Selected window: {selected['window']}\n"
    f"Selected passes: {selected['passes']}\n"
    "See selected_configuration.json, validation_summary.csv, test_summary.csv, "
    "selected_vs_competitors_tests.csv and figures/.\n",
    encoding="utf-8",
)

archive = shutil.make_archive(str(BASE.parent / "itri_window_ablation_results"), "zip", OUT_DIR)
print("Result archive:", archive)

try:
    from google.colab import files
    print("Downloading the result archive...")
    files.download(archive)
except ImportError:
    print("Automatic browser download was skipped outside Colab.")
