#!/usr/bin/env python3
"""Re-execute the archived ITRI implementation without its notebook side effects.

This loads the original experiment's function/class AST nodes rather than the
retyped functions in the supplementary comparator script. The study CSV hashes,
every constructed mask, and every archived numerical metric are checked.
Runtime columns are intentionally excluded because elapsed time is not stable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_additional_baselines as comparator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = next((p for p in (
    ROOT / "github_itri_repo" / "scripts" / "run_full_experiment.py",
    ROOT / "scripts" / "run_full_experiment.py",
) if p.is_file()), ROOT / "scripts" / "run_full_experiment.py")
ARCHIVE = comparator.ARCHIVE
DATA = comparator.DATA_DIR
OUT = Path(__file__).resolve().parent / "results" / "full_audit_report.json"

EXPECTED_HASHES = {
    "BIST30": "1dc840d48f440b6138187a5835b5b713d0288254364a304c5047e4a5f292a642",
    "SP500": "28b34e5363852b718c6ae61db56d981de26673e5d36c6a8b3415e8fe6e688c4b",
}
FEATURES = [
    "open", "high", "low", "close", "volume", "log_return", "hl_range",
    "volume_change", "sma_5_gap", "sma_10_gap", "sma_20_gap",
    "volatility_10", "rsi_14", "macd",
]
METHOD_NAMES = {
    "make_features", "chronological_split", "make_block_mask", "corrupt",
    "mask_seed", "ITRIImputer", "reconstruction_metrics",
}
KEYS = ["split", "market", "block_length", "rate", "seed", "window", "passes"]
METRICS = ["mae", "rmse", "nrmse", "last_update_delta", "masked_cells"]


def load_original_implementation() -> dict:
    original_ast = ast.parse(ORIGINAL.read_text())
    nodes = [node for node in original_ast.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in METHOD_NAMES]
    if {node.name for node in nodes} != METHOD_NAMES:
        raise RuntimeError("Original implementation symbols changed.")
    scope = {"np": np, "pd": pd, "LinearRegression": LinearRegression,
             "FEATURES": FEATURES}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ORIGINAL), "exec"), scope)
    return scope


def prepare_markets(scope: dict) -> dict:
    prepared = {}
    for market in EXPECTED_HASHES:
        path = DATA / f"{market}_2015-01-01_2026-01-01.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_HASHES[market]:
            raise AssertionError(f"{market}: raw CSV SHA-256 mismatch: {digest}")
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        feature_frame = scope["make_features"](raw[["open", "high", "low", "close", "volume"]])
        if list(feature_frame.columns) != FEATURES:
            raise AssertionError(f"{market}: feature order changed")
        class SplitConfig:
            train_ratio = 0.70
            val_ratio = 0.15
        train, val, test = scope["chronological_split"](feature_frame, SplitConfig)
        scaler = StandardScaler().fit(train)
        prepared[market] = {
            "train": scaler.transform(train).astype(float),
            "val": scaler.transform(val).astype(float),
            "test": scaler.transform(test).astype(float),
            "dates": {"val": val.index, "test": test.index},
        }
    return prepared


def main() -> None:
    scope = load_original_implementation()
    prepared = prepare_markets(scope)
    for market, panel in prepared.items():
        extra = comparator.load_market(market)
        for split in ("train", "val", "test"):
            if not np.array_equal(panel[split], extra[split]):
                raise AssertionError(f"{market} {split}: comparator feature panel differs")
    audited_rows = 0
    worst = {metric: 0.0 for metric in METRICS}
    mismatches = []
    mask_count_checks = 0
    comparator_mask_checks = 0
    for split in ("val", "test"):
        archive = pd.read_csv(ARCHIVE / ("validation_grid.csv" if split == "val" else "test_grid.csv"))
        if archive.duplicated(KEYS).any():
            raise AssertionError(f"{split}: duplicate condition/configuration keys")
        lookup = archive.set_index(KEYS)
        expected_rows = 2 * 6 * 3 * 3 * 7 * 4
        if len(lookup) != expected_rows:
            raise AssertionError(f"{split}: {len(lookup)} rows instead of {expected_rows}")
        for market_index, (market, panel) in enumerate(prepared.items()):
            train_truth = panel["train"]
            truth = panel[split]
            for block in (1, 2, 3, 5, 10, 20):
                for rate in (0.10, 0.20, 0.30):
                    for seed in (11, 23, 37):
                        train_mask = scope["make_block_mask"](
                            train_truth.shape, rate, block,
                            scope["mask_seed"](seed, market_index, 0, block, rate), 10)
                        eval_mask = scope["make_block_mask"](
                            truth.shape, rate, block,
                            scope["mask_seed"](seed, market_index, 1 if split == "val" else 2, block, rate), 10)
                        comparator_train_mask = comparator.make_block_mask(
                            train_truth.shape, rate, block,
                            comparator.mask_seed(seed, market_index, 0, block, rate))
                        comparator_eval_mask = comparator.make_block_mask(
                            truth.shape, rate, block,
                            comparator.mask_seed(seed, market_index,
                                                 1 if split == "val" else 2, block, rate))
                        if not np.array_equal(train_mask, comparator_train_mask) or \
                           not np.array_equal(eval_mask, comparator_eval_mask):
                            raise AssertionError(f"{split} {market} {block} {rate} {seed}: mask mismatch")
                        comparator_mask_checks += 2
                        train_corrupt = scope["corrupt"](train_truth, train_mask)
                        eval_corrupt = scope["corrupt"](truth, eval_mask)
                        for window in (1, 2, 3, 4, 5, 7, 10):
                            imputer = scope["ITRIImputer"](window).fit(train_corrupt)
                            for passes in (1, 2, 3, 5):
                                filled, deltas = imputer.transform(eval_corrupt, passes)
                                actual = scope["reconstruction_metrics"](truth, filled, eval_mask)
                                actual["last_update_delta"] = deltas[-1]
                                actual["masked_cells"] = int(eval_mask.sum())
                                key = (split, market, block, rate, seed, window, passes)
                                expected = lookup.loc[key]
                                for metric in METRICS:
                                    difference = abs(float(actual[metric]) - float(expected[metric]))
                                    worst[metric] = max(worst[metric], difference)
                                    limit = 0 if metric == "masked_cells" else 1e-8
                                    if difference > limit and len(mismatches) < 20:
                                        mismatches.append({"key": key, "metric": metric,
                                                           "expected": float(expected[metric]),
                                                           "actual": float(actual[metric]),
                                                           "difference": difference})
                                audited_rows += 1
                        mask_count_checks += 1
        print(f"{split}: verified {len(lookup)} archived metric rows", flush=True)
    report = {
        "original_source": str(ORIGINAL),
        "data_sha256": EXPECTED_HASHES,
        "audited_metric_rows": audited_rows,
        "audited_condition_masks": mask_count_checks,
        "comparator_train_and_evaluation_masks_checked": comparator_mask_checks,
        "max_absolute_difference": worst,
        "mismatches_first_20": mismatches,
        "all_rows_match_within_tolerance": not mismatches,
        "excluded_columns": ["fit_seconds", "transform_seconds"],
        "tolerance": "1e-8 for numerical error/update metrics; exact masked-cell count",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
