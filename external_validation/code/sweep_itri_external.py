#!/usr/bin/env python3
"""EXPLORATORY sensitivity sweep of ITRI (w, p) on the frozen 34-ETF external benchmark.

Reuses the frozen classical code path (masks, folds, scaling, scoring) verbatim
by importing run_classical_benchmark. Adds a general-window ITRI (identical to
the archived two-index implementation) and scores every (w, p) on the same
train/validation/test masks. Labeled exploratory: it must not replace the
prespecified w=1, p=5 primary result.
"""
from __future__ import annotations
import io, json, sys, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "results/code_and_protocol")
from masking import run_lengths, strict_block_mask  # noqa: E402
from run_classical_benchmark import (  # noqa: E402
    BLOCKS, EDGE_GUARD, FEATURES, FOLDS, RATES, SEEDS, corrupt, linear_fill,
    load_verified_asset, score_features, sha256, split_frame, stable_mask_seed,
    train_observed_scaler)

WINDOWS = (1, 2, 3, 4, 5, 7, 10)
PASSES = (1, 2, 3, 5)
SNAPSHOT = Path("private/snapshot.zip")
OUT = Path("sweep_out")


class ITRIGeneral:
    """Archived two-index ITRI (general w), see repo/scripts/run_full_experiment.py."""

    def __init__(self, window: int):
        self.window = int(window)
        self.models_ = []

    def temporal_matrix(self, col):
        w = self.window
        cols = [np.roll(col, lag) for lag in range(w, 0, -1)]
        cols += [np.roll(col, -lead) for lead in range(1, w + 1)]
        z = np.column_stack(cols)
        z[:w, :] = np.nan
        z[-w:, :] = np.nan
        return z

    def fit(self, x_corrupt):
        missing = np.isnan(x_corrupt)
        completed = np.array(linear_fill(x_corrupt), dtype=float, copy=True)
        self.models_ = []
        for d in range(x_corrupt.shape[1]):
            z = self.temporal_matrix(completed[:, d])
            good = (~missing[:, d]) & np.isfinite(z).all(axis=1)
            if good.sum() < 2 * self.window + 2:
                self.models_.append(None)
                continue
            m = LinearRegression().fit(z[good], completed[good, d])
            self.models_.append(m)
        return self

    def transform_all_passes(self, x_corrupt, passes_wanted):
        """Return {p: completed} for every p in passes_wanted from one sweep."""
        missing = np.isnan(x_corrupt)
        observed = ~missing
        completed = np.array(linear_fill(x_corrupt), dtype=float, copy=True)
        out = {}
        for p in range(1, max(passes_wanted) + 1):
            for d, model in enumerate(self.models_):
                if model is None:
                    continue
                z = self.temporal_matrix(completed[:, d])
                rows = missing[:, d] & np.isfinite(z).all(axis=1)
                if rows.any():
                    completed[rows, d] = model.predict(z[rows])
            completed[observed] = x_corrupt[observed]
            if p in passes_wanted:
                out[p] = completed.copy()
        return out


def process_asset(args):
    record, manifest_sha = args
    ticker = record["ticker"]
    with zipfile.ZipFile(SNAPSHOT) as archive:
        frame = load_verified_asset(archive, record)
    rows, val_rows, coef_rows = [], [], []
    conditions = [(b, r, s) for b in BLOCKS for r in RATES for s in SEEDS]
    for fold in FOLDS:
        parts = split_frame(frame, fold)
        train = parts["train"].to_numpy(float)
        val = parts["val"].to_numpy(float)
        test = parts["test"].to_numpy(float)
        for block, rate, rep in conditions:
            train_seed = stable_mask_seed(manifest_sha, ticker, fold, "train", block, rate, rep)
            val_seed = stable_mask_seed(manifest_sha, ticker, fold, "val", block, rate, rep)
            test_seed = stable_mask_seed(manifest_sha, ticker, fold, "test", block, rate, rep)
            train_hidden = strict_block_mask(train.shape, rate, block, train_seed, EDGE_GUARD)
            val_hidden = strict_block_mask(val.shape, rate, block, val_seed, EDGE_GUARD)
            test_hidden = strict_block_mask(test.shape, rate, block, test_seed, EDGE_GUARD)
            mean, scale = train_observed_scaler(train, train_hidden)
            train_in = corrupt((train - mean) / scale, train_hidden)
            val_in = corrupt((val - mean) / scale, val_hidden)
            test_in = corrupt((test - mean) / scale, test_hidden)
            lengths = run_lengths(test_hidden)
            common = {"ticker": ticker, "country": record["country"], "region": record["region"],
                      "fold": fold, "block_length": block, "rate": rate, "seed": rep,
                      "train_mask_seed": train_seed, "test_mask_seed": test_seed,
                      "actual_max_test_run": int(lengths.max())}
            for w in WINDOWS:
                t0 = time.perf_counter()
                model = ITRIGeneral(w).fit(train_in)
                fit_s = time.perf_counter() - t0
                if w == 1:
                    for d, m in enumerate(model.models_):
                        if m is not None:
                            coef_rows.append({"ticker": ticker, "fold": fold, "block_length": block,
                                              "rate": rate, "seed": rep, "feature": FEATURES[d],
                                              "coef_lag1": float(m.coef_[0]), "coef_lead1": float(m.coef_[1]),
                                              "intercept": float(m.intercept_)})
                t0 = time.perf_counter()
                test_out = model.transform_all_passes(test_in, PASSES)
                tr_s = time.perf_counter() - t0
                val_out = model.transform_all_passes(val_in, PASSES)
                for p in PASSES:
                    name = f"itri_w{w}_p{p}"
                    rows.extend(score_features(test, test_out[p], test_hidden, mean, scale,
                                               {**common, "method": name, "window": w, "passes": p},
                                               fit_s + tr_s, 0))
                    val_rows.extend(score_features(val, val_out[p], val_hidden, mean, scale,
                                                   {**common, "method": name, "window": w, "passes": p,
                                                    "partition": "validation"}, fit_s + tr_s, 0))
    return ticker, pd.DataFrame(rows), pd.DataFrame(val_rows), pd.DataFrame(coef_rows)


def main():
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(SNAPSHOT) as archive:
        manifest_blob = archive.read("asset_universe_v1.json")
        receipt = json.loads(archive.read("acquisition_qc_receipt.json"))
    manifest_sha = sha256(manifest_blob)
    assert receipt["manifest_sha256"] == manifest_sha
    records = receipt["assets"]
    print("assets:", len(records), "manifest:", manifest_sha[:12], flush=True)
    t0 = time.time()
    tests, vals, coefs = [], [], []
    with ProcessPoolExecutor() as pool:
        for i, (ticker, t, v, c) in enumerate(pool.map(process_asset, [(r, manifest_sha) for r in records]), 1):
            tests.append(t); vals.append(v); coefs.append(c)
            print(f"[{i}/{len(records)}] {ticker} done, {time.time()-t0:.0f}s", flush=True)
    test = pd.concat(tests, ignore_index=True)
    val = pd.concat(vals, ignore_index=True)
    coef = pd.concat(coefs, ignore_index=True)
    test.to_csv(OUT / "sweep_test_feature_rows.csv", index=False, float_format="%.17g")
    val.to_csv(OUT / "sweep_validation_feature_rows.csv", index=False, float_format="%.17g")
    coef.to_csv(OUT / "itri_w1_coefficients.csv", index=False, float_format="%.17g")
    json.dump({"label": "EXPLORATORY sensitivity sweep, not the prespecified primary result",
               "windows": WINDOWS, "passes": PASSES, "manifest_sha256": manifest_sha,
               "private_zip_sha256": sha256(SNAPSHOT.read_bytes()),
               "test_rows": len(test), "validation_rows": len(val)},
              open(OUT / "sweep_metadata.json", "w"), indent=2)
    print("rows:", len(test), len(val), "elapsed:", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
