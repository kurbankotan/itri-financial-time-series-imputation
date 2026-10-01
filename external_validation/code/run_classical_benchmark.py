#!/usr/bin/env python3
"""Frozen OHLCV external-validation benchmark, classical comparators.

This program reads the PRIVATE vendor snapshot ZIP without extracting it. It
never writes OHLCV values to the public result directory. All methods receive
the exact same mask per asset/fold/condition. Outputs are checkpointed by
asset/fold. Use --pilot for a small smoke test; never report pilot results as
the full prespecified experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.interpolate import PchipInterpolator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.exceptions import ConvergenceWarning

from masking import run_lengths, strict_block_mask


FEATURES = ("open", "high", "low", "close", "volume")
METHODS = ("linear", "pchip", "knn5", "chained_ridge", "itri_w1_p5")
FOLDS = {
    "A": {"train": (2015, 2018), "val": (2019, 2019), "test": (2020, 2021)},
    "B": {"train": (2015, 2020), "val": (2021, 2021), "test": (2022, 2023)},
    "C": {"train": (2015, 2022), "val": (2023, 2023), "test": (2024, 2025)},
}
BLOCKS = (1, 2, 3, 5, 10, 20)
RATES = (0.10, 0.20, 0.30)
SEEDS = (11, 23, 37)
EDGE_GUARD = 20


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def stable_mask_seed(manifest_sha: str, ticker: str, fold: str,
                     partition: str, block: int, rate: float,
                     repetition_seed: int) -> int:
    key = (f"{manifest_sha}|{ticker}|{fold}|{partition}|"
           f"{block}|{rate:.2f}|{repetition_seed}")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def split_frame(frame: pd.DataFrame, fold: str) -> dict[str, pd.DataFrame]:
    parts = {}
    for name, (start_year, end_year) in FOLDS[fold].items():
        subset = frame.loc[(frame.index.year >= start_year) &
                           (frame.index.year <= end_year), list(FEATURES)]
        if subset.empty:
            raise ValueError(f"empty {fold}/{name} partition")
        parts[name] = subset.copy()
    return parts


def train_observed_scaler(train: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.shape != hidden.shape:
        raise ValueError("training array/mask shape mismatch")
    means, scales = [], []
    for feature in range(train.shape[1]):
        observed = train[~hidden[:, feature], feature]
        if observed.size < 2 or not np.isfinite(observed).all():
            raise ValueError(f"insufficient finite training data for feature {feature}")
        mean = float(observed.mean())
        scale = float(observed.std(ddof=0))
        if scale <= 1e-12:
            raise ValueError(f"near-zero training-observed std for feature {feature}")
        means.append(mean)
        scales.append(scale)
    return np.asarray(means), np.asarray(scales)


def corrupt(values: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    output = values.copy()
    output[hidden] = np.nan
    return output


def linear_fill(corrupt_values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(corrupt_values).interpolate(
        method="linear", limit_direction="both").ffill().bfill().to_numpy(float)


def pchip_fill(corrupt_values: np.ndarray) -> np.ndarray:
    out = corrupt_values.copy()
    x = np.arange(len(out))
    for feature in range(out.shape[1]):
        seen = np.isfinite(out[:, feature])
        if seen.sum() < 2:
            raise ValueError("PCHIP needs at least two observed cells")
        curve = PchipInterpolator(x[seen], out[seen, feature], extrapolate=True)
        out[~seen, feature] = curve(x[~seen])
    return out


class ITRIImputer:
    """Exact w=1, p=5 temporal-regression logic from the archived implementation.

    Only the panel and observed-only scaling differ from the original study.
    The base model has not been modified to chase external test results.
    """

    def __init__(self) -> None:
        self.window = 1
        self.models_: list[LinearRegression | None] = []

    def temporal_matrix(self, col: np.ndarray) -> np.ndarray:
        z = np.column_stack((np.roll(col, 1), np.roll(col, -1)))
        z[0, :] = np.nan
        z[-1, :] = np.nan
        return z

    def fit(self, train_corrupt: np.ndarray) -> "ITRIImputer":
        missing = np.isnan(train_corrupt)
        completed = linear_fill(train_corrupt)
        self.models_ = []
        for feature in range(train_corrupt.shape[1]):
            z = self.temporal_matrix(completed[:, feature])
            good = (~missing[:, feature]) & np.isfinite(z).all(axis=1)
            if int(good.sum()) < 4:
                self.models_.append(None)
                continue
            model = LinearRegression()
            model.fit(z[good], completed[good, feature])
            self.models_.append(model)
        return self

    def transform(self, test_corrupt: np.ndarray) -> np.ndarray:
        missing = np.isnan(test_corrupt)
        observed = ~missing
        completed = linear_fill(test_corrupt)
        for _ in range(5):
            for feature, model in enumerate(self.models_):
                if model is None:
                    continue
                z = self.temporal_matrix(completed[:, feature])
                rows = missing[:, feature] & np.isfinite(z).all(axis=1)
                if rows.any():
                    completed[rows, feature] = model.predict(z[rows])
            completed[observed] = test_corrupt[observed]
        return completed


def run_methods(train_corrupt: np.ndarray, test_corrupt: np.ndarray,
                random_seed: int) -> dict[str, tuple[np.ndarray, float, int]]:
    outputs = {}
    start = time.perf_counter()
    outputs["linear"] = (linear_fill(test_corrupt), time.perf_counter() - start, 0)
    start = time.perf_counter()
    outputs["pchip"] = (pchip_fill(test_corrupt), time.perf_counter() - start, 0)
    start = time.perf_counter()
    knn = KNNImputer(n_neighbors=5, weights="distance")
    knn.fit(train_corrupt)
    outputs["knn5"] = (knn.transform(test_corrupt), time.perf_counter() - start, 0)
    start = time.perf_counter()
    chained = IterativeImputer(
        estimator=Ridge(alpha=1.0), max_iter=10, tol=1e-3,
        initial_strategy="mean", imputation_order="ascending",
        sample_posterior=False, random_state=random_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        chained.fit(train_corrupt)
        chained_estimate = chained.transform(test_corrupt)
    nonconvergence = sum(issubclass(note.category, ConvergenceWarning)
                         for note in caught)
    outputs["chained_ridge"] = (chained_estimate,
                                 time.perf_counter() - start, nonconvergence)
    start = time.perf_counter()
    temporal = ITRIImputer().fit(train_corrupt)
    outputs["itri_w1_p5"] = (temporal.transform(test_corrupt),
                               time.perf_counter() - start, 0)
    if tuple(outputs) != METHODS:
        raise AssertionError("method registry/order mismatch")
    for method, (estimate, _, _) in outputs.items():
        if estimate.shape != test_corrupt.shape or not np.isfinite(estimate).all():
            raise ValueError(f"{method} produced invalid estimates")
        observed = np.isfinite(test_corrupt)
        if not np.allclose(estimate[observed], test_corrupt[observed],
                           atol=1e-9, rtol=1e-9):
            raise ValueError(f"{method} changed originally observed cells")
    return outputs


def score_features(truth_raw: np.ndarray, estimate_scaled: np.ndarray,
                   hidden: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                   common: dict, elapsed: float,
                   convergence_warning_count: int) -> list[dict]:
    estimate_raw = estimate_scaled * scale + mean
    results = []
    for feature_index, feature in enumerate(FEATURES):
        selected = hidden[:, feature_index]
        count = int(selected.sum())
        if count < 1:
            raise ValueError(f"no masked cells for {feature}")
        error = estimate_raw[selected, feature_index] - truth_raw[selected, feature_index]
        mae = float(np.abs(error).mean())
        rmse = float(np.sqrt(np.square(error).mean()))
        results.append({
            **common,
            "feature": feature,
            "masked_cells": count,
            "mae_raw": mae,
            "rmse_raw": rmse,
            "mae_trainstd": mae / scale[feature_index],
            "rmse_trainstd": rmse / scale[feature_index],
            "fit_and_impute_seconds": elapsed,
            "convergence_warning_count": convergence_warning_count,
        })
    return results


def load_verified_asset(archive: zipfile.ZipFile, record: dict) -> pd.DataFrame:
    name = f"clean_vendor_cache_do_not_publish/{record['ticker']}_2015-01-01_2026-01-01.csv"
    blob = archive.read(name)
    if sha256(blob) != record["clean_sha256"]:
        raise ValueError(f"private data hash mismatch: {record['ticker']}")
    frame = pd.read_csv(io.BytesIO(blob), index_col=0,
                        parse_dates=True, float_precision="round_trip")
    if frame.columns.tolist() != list(FEATURES) or len(frame) != record["row_count"]:
        raise ValueError(f"private data shape mismatch: {record['ticker']}")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError(f"nonfinite data: {record['ticker']}")
    return frame


def run(snapshot: Path, output_dir: Path, pilot: bool = False) -> None:
    with zipfile.ZipFile(snapshot) as archive:
        manifest_blob = archive.read("asset_universe_v1.json")
        receipt_blob = archive.read("acquisition_qc_receipt.json")
        receipt = json.loads(receipt_blob)
        manifest = json.loads(manifest_blob)
        manifest_sha = sha256(manifest_blob)
        if receipt["manifest_sha256"] != manifest_sha:
            raise ValueError("snapshot manifest/receipt SHA mismatch")
        if receipt["eligible_count"] != 34 or receipt["excluded_count"]:
            raise ValueError("unexpected eligibility count; revise only by protocol amendment")
        if [x["ticker"] for x in manifest["assets"]] != [
                x["ticker"] for x in receipt["assets"]]:
            raise ValueError("manifest/receipt asset order mismatch")

        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__}
        run_fingerprint = sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("masking.py").read_bytes()
            + manifest_blob + receipt_blob + str(pilot).encode()
            + json.dumps(versions, sort_keys=True).encode())
        run_meta = {
            "run_fingerprint": run_fingerprint,
            "protocol": manifest["protocol_version"],
            "manifest_sha256": manifest_sha,
            "private_zip_sha256": sha256(snapshot.read_bytes()),
            "pilot_not_publishable": pilot,
            "methods": METHODS,
            "folds": FOLDS,
            "blocks": BLOCKS,
            "rates": RATES,
            "seeds": SEEDS,
            "edge_guard": EDGE_GUARD,
            "primary_panel": FEATURES,
            "primary_endpoint": "macro mean across 5 feature-wise RMSE/training-observed-SD",
            "versions": versions,
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8")
        records = receipt["assets"][:1] if pilot else receipt["assets"]
        folds = ("A",) if pilot else tuple(FOLDS)
        conditions = [(b, r, s) for b in BLOCKS for r in RATES for s in SEEDS]
        if pilot:
            conditions = conditions[:2]
        for asset_index, record in enumerate(records, start=1):
            ticker = record["ticker"]
            frame = load_verified_asset(archive, record)
            for fold in folds:
                shard_path = shard_dir / f"{ticker}_{fold}.csv"
                meta_path = shard_dir / f"{ticker}_{fold}.meta.json"
                expected_rows = len(conditions) * len(METHODS) * len(FEATURES)
                if shard_path.exists() and meta_path.exists():
                    sidecar = json.loads(meta_path.read_text())
                    if (sidecar.get("run_fingerprint") == run_fingerprint and
                            sidecar.get("expected_rows") == expected_rows and
                            sidecar.get("csv_sha256") == sha256(shard_path.read_bytes())):
                        print(f"[{asset_index}/{len(records)}] {ticker}/{fold}: verified checkpoint")
                        continue
                partitions = split_frame(frame, fold)
                train = partitions["train"].to_numpy(float)
                test = partitions["test"].to_numpy(float)
                rows = []
                for condition_index, (block, rate, repetition_seed) in enumerate(conditions, 1):
                    train_seed = stable_mask_seed(manifest_sha, ticker, fold, "train",
                                                  block, rate, repetition_seed)
                    test_seed = stable_mask_seed(manifest_sha, ticker, fold, "test",
                                                 block, rate, repetition_seed)
                    train_hidden = strict_block_mask(train.shape, rate, block,
                                                     train_seed, EDGE_GUARD)
                    test_hidden = strict_block_mask(test.shape, rate, block,
                                                    test_seed, EDGE_GUARD)
                    mean, scale = train_observed_scaler(train, train_hidden)
                    train_scaled = (train - mean) / scale
                    test_scaled = (test - mean) / scale
                    estimates = run_methods(corrupt(train_scaled, train_hidden),
                                            corrupt(test_scaled, test_hidden),
                                            repetition_seed)
                    lengths = run_lengths(test_hidden)
                    common = {
                        "ticker": ticker, "country": record["country"],
                        "region": record["region"], "fold": fold,
                        "test_start_year": FOLDS[fold]["test"][0],
                        "test_end_year": FOLDS[fold]["test"][1],
                        "block_length": block, "rate": rate,
                        "seed": repetition_seed,
                        "train_mask_seed": train_seed,
                        "test_mask_seed": test_seed,
                        "actual_max_test_run": int(lengths.max()),
                    }
                    if common["actual_max_test_run"] > block:
                        raise AssertionError("mask-length invariant violated")
                    for method, (estimate, elapsed, warning_count) in estimates.items():
                        rows.extend(score_features(test, estimate, test_hidden,
                                                   mean, scale,
                                                   {**common, "method": method},
                                                   elapsed, warning_count))
                    if condition_index % 10 == 0 or condition_index == len(conditions):
                        print(f"[{asset_index}/{len(records)}] {ticker}/{fold}: "
                              f"{condition_index}/{len(conditions)} conditions")
                if len(rows) != expected_rows:
                    raise AssertionError("missing or duplicate method-feature rows")
                temporary = shard_path.with_suffix(".csv.tmp")
                pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.17g")
                os.replace(temporary, shard_path)
                sidecar = {
                    "run_fingerprint": run_fingerprint,
                    "expected_rows": expected_rows,
                    "csv_sha256": sha256(shard_path.read_bytes()),
                    "pilot_not_publishable": pilot,
                }
                meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        expected_shards = len(records) * len(folds)
        verified_shards = list(shard_dir.glob("*.meta.json"))
        if len(verified_shards) < expected_shards:
            raise AssertionError("run ended with missing checkpoints")
        (output_dir / "COMPLETE.marker").write_text(
            json.dumps({"expected_shards": expected_shards,
                        "expected_rows": expected_shards * len(conditions)
                        * len(METHODS) * len(FEATURES),
                        "pilot_not_publishable": pilot,
                        "run_fingerprint": run_fingerprint}, indent=2),
            encoding="utf-8")
        print("COMPLETE", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Private data snapshot ZIP from acquisition notebook")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test only: first asset/fold/two masks; not publishable")
    args = parser.parse_args()
    run(args.snapshot, args.out, pilot=args.pilot)
