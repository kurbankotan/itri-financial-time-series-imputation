#!/usr/bin/env python3
"""Reproduce matched-mask ITRI baselines from the exact archived study files.

The script uses the same feature engineering, chronological split, StandardScaler,
block-mask generator, seed mapping, and masked-cell metrics as the published
ITRI experiment. It neither downloads data nor changes the archived results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.interpolate import PchipInterpolator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def discover_directory(env_name: str, candidates: list[Path]) -> Path:
    """Support the local workspace and a direct upload into the public repo."""
    if env_name in os.environ:
        return Path(os.environ[env_name]).expanduser().resolve()
    return next((path for path in candidates if path.is_dir()), candidates[0])


DATA_DIR = discover_directory("ITRI_DATA_DIR", [
    ROOT / "work" / "data", ROOT / "reproduction_run" / "data", ROOT / "data",
])
ARCHIVE = discover_directory("ITRI_RESULTS_DIR", [
    ROOT / "github_itri_repo" / "results", ROOT / "results",
])
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

MARKETS = {
    "BIST30": ("BIST30_2015-01-01_2026-01-01.csv", "1dc840d48f440b6138187a5835b5b713d0288254364a304c5047e4a5f292a642"),
    "SP500": ("SP500_2015-01-01_2026-01-01.csv", "28b34e5363852b718c6ae61db56d981de26673e5d36c6a8b3415e8fe6e688c4b"),
}
FEATURES = [
    "open", "high", "low", "close", "volume", "log_return", "hl_range",
    "volume_change", "sma_5_gap", "sma_10_gap", "sma_20_gap",
    "volatility_10", "rsi_14", "macd",
]
BLOCKS = (1, 2, 3, 5, 10, 20)
RATES = (0.1, 0.2, 0.3)
SEEDS = (11, 23, 37)
EDGE_GUARD = 10


def features(raw: pd.DataFrame) -> pd.DataFrame:
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


def load_market(name: str) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    filename, expected_sha = MARKETS[name]
    path = DATA_DIR / filename
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Study-file hash mismatch for {name}: {actual_sha}")
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    x = features(raw)
    n = len(x)
    a, b = int(0.70 * n), int(0.85 * n)
    train, val, test = x.iloc[:a], x.iloc[a:b], x.iloc[b:]
    scaler = StandardScaler().fit(train)
    return {
        "train": scaler.transform(train).astype(float),
        "val": scaler.transform(val).astype(float),
        "test": scaler.transform(test).astype(float),
        "val_dates": val.index,
        "test_dates": test.index,
    }


def make_block_mask(shape: tuple[int, int], rate: float, block_length: int, seed: int) -> np.ndarray:
    n, d = shape
    if n <= 2 * EDGE_GUARD + block_length:
        raise ValueError("Partition is too short for the block and guard.")
    rng = np.random.default_rng(seed)
    eligible = np.zeros((n, d), dtype=bool)
    eligible[EDGE_GUARD:n - EDGE_GUARD, :] = True
    target = max(1, int(round(rate * eligible.sum())))
    mask = np.zeros((n, d), dtype=bool)
    attempts = 0
    while mask.sum() < target and attempts < target * 100:
        need = target - int(mask.sum())
        length = min(block_length, need)
        col = int(rng.integers(0, d))
        start = int(rng.integers(EDGE_GUARD, n - EDGE_GUARD - length + 1))
        if not mask[start:start + length, col].any():
            mask[start:start + length, col] = True
        attempts += 1
    if mask.sum() != target:
        raise RuntimeError(f"Incomplete mask: {mask.sum()} versus {target}")
    return mask


def mask_seed(base_seed: int, market_index: int, split_index: int, block_length: int, rate: float) -> int:
    return int(base_seed + 1009 * market_index + 101 * split_index
               + 17 * block_length + round(rate * 1000))


def corrupt(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[mask] = np.nan
    return y


def pchip_impute(x: np.ndarray) -> np.ndarray:
    completed = x.copy()
    t = np.arange(len(x))
    for d in range(x.shape[1]):
        observed = np.isfinite(x[:, d])
        if observed.sum() < 2:
            raise ValueError("PCHIP needs at least two observed points per feature.")
        interpolator = PchipInterpolator(t[observed], x[observed, d], extrapolate=True)
        completed[~observed, d] = interpolator(t[~observed])
    return completed


def estimate(method: str, parameter: int, train: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    if method == "PCHIP":
        return pchip_impute(evaluation)
    if method == "KNN":
        model = KNNImputer(n_neighbors=parameter, weights="distance", metric="nan_euclidean")
    elif method == "MICE":
        model = IterativeImputer(
            estimator=BayesianRidge(), max_iter=parameter,
            initial_strategy="mean", sample_posterior=False,
            random_state=0, tol=0.0, skip_complete=False,
        )
    else:
        raise ValueError(method)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
        model.fit(train)
        output = model.transform(evaluation)
    return output


def metrics(truth: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.isfinite(filled).all():
        raise ValueError("Imputer returned nonfinite values")
    error = filled[mask] - truth[mask]
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = max(float(np.std(truth[mask])), 1e-12)
    return {
        "mae": float(np.mean(np.abs(error))), "rmse": rmse,
        "nrmse": rmse / denominator, "masked_truth_sd": denominator,
    }


def conditions(split: str, pilot: bool = False):
    cached = {name: load_market(name) for name in MARKETS}
    for market_index, (market, data) in enumerate(cached.items()):
        for block_length in BLOCKS:
            for rate in RATES:
                for seed in SEEDS:
                    if pilot and (market, block_length, rate, seed) != ("BIST30", 1, 0.1, 11):
                        continue
                    train_truth = data["train"]
                    truth = data[split]
                    train_mask = make_block_mask(
                        train_truth.shape, rate, block_length,
                        mask_seed(seed, market_index, 0, block_length, rate),
                    )
                    eval_mask = make_block_mask(
                        truth.shape, rate, block_length,
                        mask_seed(seed, market_index, 1 if split == "val" else 2, block_length, rate),
                    )
                    yield (market, block_length, rate, seed,
                           corrupt(train_truth, train_mask), corrupt(truth, eval_mask), truth, eval_mask)


def run_split(split: str, methods: list[tuple[str, int]], pilot: bool = False) -> pd.DataFrame:
    rows = []
    started = time.monotonic()
    for index, (market, block_length, rate, seed, train, evaluation, truth, mask) in enumerate(conditions(split, pilot), 1):
        for method, parameter in methods:
            t0 = time.monotonic()
            filled = estimate(method, parameter, train, evaluation)
            row = {
                "split": split, "market": market, "block_length": block_length,
                "rate": rate, "seed": seed, "method": method,
                "parameter": parameter, "seconds": time.monotonic() - t0,
                "masked_cells": int(mask.sum()),
                **metrics(truth, filled, mask),
            }
            rows.append(row)
        if index % 9 == 0 or pilot:
            print(f"{split}: {index} conditions; {time.monotonic() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["split", "method", "parameter"], as_index=False).agg(
        conditions=("nrmse", "size"), mean_nrmse=("nrmse", "mean"),
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"),
        mean_seconds=("seconds", "mean"),
    ).sort_values(["split", "mean_nrmse"])


def check_masks(frame: pd.DataFrame) -> None:
    archive = pd.read_csv(ARCHIVE / ("validation_grid.csv" if frame.split.iloc[0] == "val" else "test_grid.csv"))
    archived = archive[(archive.window == 1) & (archive.passes == 5)][
        ["market", "block_length", "rate", "seed", "masked_cells"]
    ]
    current = frame[frame.method == "PCHIP"][
        ["market", "block_length", "rate", "seed", "masked_cells"]
    ]
    comparison = current.merge(archived, on=["market", "block_length", "rate", "seed"], suffixes=("_new", "_archived"))
    if len(comparison) != len(current) or not (comparison.masked_cells_new == comparison.masked_cells_archived).all():
        raise AssertionError("New mask counts differ from the archive.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    candidates = [("PCHIP", 0), ("KNN", 3), ("KNN", 5), ("KNN", 10), ("MICE", 3), ("MICE", 5)]
    validation = run_split("val", candidates, pilot=args.pilot)
    print(summarize(validation).to_string(index=False), flush=True)
    if args.pilot:
        return
    check_masks(validation)
    validation.to_csv(OUT / "additional_baselines_validation.csv", index=False)
    summary = summarize(validation)
    selected = [
        ("PCHIP", 0),
        ("KNN", int(summary[summary.method == "KNN"].iloc[0].parameter)),
        ("MICE", int(summary[summary.method == "MICE"].iloc[0].parameter)),
    ]
    test = run_split("test", selected)
    check_masks(test)
    test.to_csv(OUT / "additional_baselines_test.csv", index=False)
    combined_summary = summarize(pd.concat([validation, test], ignore_index=True))
    combined_summary.to_csv(OUT / "additional_baselines_summary.csv", index=False)
    metadata = {
        "raw_data_sha256": {name: digest for name, (_, digest) in MARKETS.items()},
        "methods": {"PCHIP": "SciPy PchipInterpolator, feature-wise along time",
                    "KNN": "scikit-learn KNNImputer, nan_euclidean and distance weights",
                    "MICE": "scikit-learn IterativeImputer with BayesianRidge, deterministic"},
        "candidate_parameters": {"KNN_neighbors": [3, 5, 10], "MICE_max_iter": [3, 5]},
        "selected_on_validation": {m: int(p) for m, p in selected},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "note": "Same archive study files, feature engineering, chronological split, scale fitting, masks and metrics as original ITRI run. Model settings selected from validation only.",
    }
    (OUT / "additional_baselines_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(combined_summary.to_string(index=False), flush=True)
    print("Selected baselines:", selected, flush=True)


if __name__ == "__main__":
    main()
