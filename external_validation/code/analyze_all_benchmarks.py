#!/usr/bin/env python3
"""Audit and combine the frozen classical and BRITS/SAITS external results.

The primary endpoint is a feature-macro normalized RMSE for each asset/fold/
condition/method. Asset summaries then average folds and mask conditions;
the 5,508 conditions are not treated as independent market replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


CLASSICAL = ("linear", "pchip", "knn5", "chained_ridge", "itri_w1_p5")
NEURAL = ("brits", "saits")
ALL_METHODS = CLASSICAL + NEURAL
KEY = ["ticker", "fold", "block_length", "rate", "seed", "feature", "method"]
CONDITION = ["ticker", "country", "region", "fold", "block_length",
             "rate", "seed", "method"]
NUMERIC = ["mae_raw", "rmse_raw", "mae_trainstd", "rmse_trainstd",
           "fit_and_impute_seconds"]


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def load_checked(source: Path, expected_methods: tuple[str, ...]) -> tuple[pd.DataFrame, dict, dict]:
    if source.is_dir():
        names = sorted(p.name for p in (source / "shards").glob("*.csv"))

        def read(name: str) -> bytes:
            return (source / name).read_bytes()
    elif zipfile.is_zipfile(source):
        archive = zipfile.ZipFile(source)
        if archive.testzip() is not None:
            raise ValueError(f"corrupt ZIP: {source}")
        names = sorted(Path(n).name for n in archive.namelist()
                       if n.startswith("results/shards/") and n.endswith(".csv"))

        def read(name: str) -> bytes:
            return archive.read("results/" + name)
    else:
        raise ValueError(f"expected result directory or ZIP: {source}")
    marker = json.loads(read("COMPLETE.marker"))
    metadata = json.loads(read("run_metadata.json"))
    if marker["pilot_not_publishable"] or metadata["pilot_not_publishable"]:
        raise ValueError("pilot output is not a full scientific result")
    if (len(names) != marker["expected_shards"] or
            marker["run_fingerprint"] != metadata["run_fingerprint"]):
        raise ValueError("incomplete or mixed result checkpoints")
    frames = []
    for name in names:
        blob = read("shards/" + name)
        sidecar = json.loads(read("shards/" + name.replace(".csv", ".meta.json")))
        if (sha256(blob) != sidecar["csv_sha256"] or
                sidecar["run_fingerprint"] != marker["run_fingerprint"]):
            raise ValueError(f"checksum/fingerprint mismatch: {name}")
        frame = pd.read_csv(io.BytesIO(blob), float_precision="round_trip")
        if len(frame) != sidecar["expected_rows"]:
            raise ValueError(f"row count mismatch: {name}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != marker["expected_rows"] or set(data.method) != set(expected_methods):
        raise ValueError("result row count or method registry mismatch")
    if data.duplicated(KEY).any() or data[NUMERIC].isna().any().any():
        raise ValueError("duplicate key or missing numeric result")
    if not np.isfinite(data[NUMERIC].to_numpy(float)).all():
        raise ValueError("non-finite numeric result")
    if (data.masked_cells <= 0).any() or (data[NUMERIC] < 0).any().any():
        raise ValueError("invalid masked count, error or runtime")
    if source.is_file():
        archive.close()
    return data, marker, metadata


def load_validation_checked(source: Path, marker: dict, metadata: dict) -> pd.DataFrame:
    if source.is_dir():
        names = sorted(p.name for p in (source / "validation_shards").glob("*.csv"))

        def read(name: str) -> bytes:
            return (source / "validation_shards" / name).read_bytes()
    else:
        archive = zipfile.ZipFile(source)
        names = sorted(Path(n).name for n in archive.namelist()
                       if n.startswith("results/validation_shards/") and n.endswith(".csv"))

        def read(name: str) -> bytes:
            return archive.read("results/validation_shards/" + name)
    if len(names) != marker["expected_shards"]:
        raise ValueError("validation shard grid is incomplete")
    frames = []
    for name in names:
        blob = read(name)
        sidecar = json.loads(read(name.replace(".csv", ".meta.json")))
        if (sidecar["run_fingerprint"] != metadata["run_fingerprint"] or
                sidecar["csv_sha256"] != sha256(blob) or
                sidecar["expected_rows"] != 35):
            raise ValueError(f"invalid validation checkpoint: {name}")
        frame = pd.read_csv(io.BytesIO(blob), float_precision="round_trip")
        if len(frame) != 35:
            raise ValueError(f"validation row count mismatch: {name}")
        frames.append(frame)
    if source.is_file():
        archive.close()
    data = pd.concat(frames, ignore_index=True)
    if (len(data) != marker["expected_validation_rows"] or
            set(data.method) != set(ALL_METHODS) or
            data.duplicated(KEY).any() or
            (data.masked_cells <= 0).any() or
            not np.isfinite(data[NUMERIC].to_numpy(float)).all()):
        raise ValueError("invalid complete validation grid")
    return data


def reconcile(classical: pd.DataFrame, neural: pd.DataFrame,
              classical_meta: dict, neural_meta: dict) -> pd.DataFrame:
    if (classical_meta["run_fingerprint"] != neural_meta["classical_fingerprint"] or
            classical_meta["manifest_sha256"] != neural_meta["manifest_sha256"] or
            classical_meta["private_zip_sha256"] != neural_meta["private_zip_sha256"]):
        raise ValueError("classical/neural input provenance does not match")
    data = pd.concat([classical, neural], ignore_index=True, sort=False)
    if len(data) != 34 * 3 * 6 * 3 * 3 * len(ALL_METHODS) * 5:
        raise ValueError("combined prespecified grid is incomplete")
    if data.duplicated(KEY).any():
        raise ValueError("duplicate combined result key")
    condition_keys = ["ticker", "fold", "block_length", "rate", "seed"]
    methods = data.groupby(condition_keys)["method"].nunique()
    if len(methods) != 34 * 3 * 6 * 3 * 3 or not methods.eq(len(ALL_METHODS)).all():
        raise ValueError("at least one condition lacks a comparator")
    feature_counts = data.groupby(condition_keys + ["method"])["feature"].nunique()
    if not feature_counts.eq(5).all():
        raise ValueError("at least one method lacks a feature")
    for field in ("train_mask_seed", "test_mask_seed", "masked_cells",
                  "actual_max_test_run"):
        varying = data.groupby(condition_keys + ["feature"])[field].nunique()
        if not varying.eq(1).all():
            raise ValueError(f"methods did not share {field}")
    return data


def summarize(data: pd.DataFrame, validation: pd.DataFrame, output_dir: Path,
              classical_meta: dict, neural_meta: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.sort_values(KEY).to_csv(output_dir / "feature_metrics_combined.csv",
                                 index=False, float_format="%.17g")
    condition = data.groupby(CONDITION, as_index=False).agg(
        nrmse=("rmse_trainstd", "mean"),
        nmae=("mae_trainstd", "mean"),
        masked_cells_total=("masked_cells", "sum"),
        fit_and_impute_seconds=("fit_and_impute_seconds", "first"),
    )
    condition.to_csv(output_dir / "condition_macro.csv", index=False,
                     float_format="%.17g")
    validation.to_csv(output_dir / "validation_feature_metrics.csv", index=False,
                      float_format="%.17g")
    val_condition = validation.groupby(CONDITION, as_index=False).agg(
        validation_nrmse=("rmse_trainstd", "mean"),
        validation_nmae=("mae_trainstd", "mean"))
    paired_conditions = condition.merge(
        val_condition,
        on=["ticker", "country", "region", "fold", "block_length", "rate",
            "seed", "method"], validate="one_to_one")
    if len(paired_conditions) != len(condition):
        raise ValueError("validation/test condition keys differ")
    paired_conditions["test_minus_validation_nrmse"] = (
        paired_conditions.nrmse - paired_conditions.validation_nrmse)
    paired_conditions.to_csv(output_dir / "validation_test_conditions.csv",
                             index=False, float_format="%.17g")
    gap_asset = paired_conditions.groupby(["ticker", "method"],
                                          as_index=False).agg(
        mean_test_nrmse=("nrmse", "mean"),
        mean_validation_nrmse=("validation_nrmse", "mean"),
        mean_test_minus_validation_nrmse=("test_minus_validation_nrmse", "mean"))
    gap_asset.to_csv(output_dir / "validation_test_asset.csv", index=False,
                     float_format="%.17g")
    gap_summary = gap_asset.groupby("method", as_index=False).agg(
        mean_test_nrmse=("mean_test_nrmse", "mean"),
        mean_validation_nrmse=("mean_validation_nrmse", "mean"),
        mean_test_minus_validation_nrmse=("mean_test_minus_validation_nrmse", "mean"))
    gap_summary.to_csv(output_dir / "validation_test_method.csv", index=False,
                       float_format="%.17g")
    asset = condition.groupby(["ticker", "country", "region", "method"],
                              as_index=False).agg(
        nrmse=("nrmse", "mean"), nmae=("nmae", "mean"))
    asset.to_csv(output_dir / "asset_method.csv", index=False,
                 float_format="%.17g")
    block_asset = condition.groupby(["ticker", "region", "block_length", "method"],
                                    as_index=False).agg(nrmse=("nrmse", "mean"),
                                                        nmae=("nmae", "mean"))
    block_summary = block_asset.groupby(["block_length", "method"],
                                        as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"),
        median_asset_nrmse=("nrmse", "median"),
        mean_asset_nmae=("nmae", "mean"))
    block_summary.to_csv(output_dir / "block_method.csv", index=False,
                         float_format="%.17g")
    block_rate_asset = condition.groupby(
        ["ticker", "block_length", "rate", "method"], as_index=False).agg(
            nrmse=("nrmse", "mean"), nmae=("nmae", "mean"))
    block_rate_summary = block_rate_asset.groupby(
        ["block_length", "rate", "method"], as_index=False).agg(
            mean_asset_nrmse=("nrmse", "mean"),
            median_asset_nrmse=("nrmse", "median"),
            mean_asset_nmae=("nmae", "mean"))
    block_rate_summary.to_csv(output_dir / "block_rate_method.csv", index=False,
                              float_format="%.17g")
    feature_asset = data.groupby(["ticker", "feature", "method"],
                                 as_index=False).agg(
        nrmse=("rmse_trainstd", "mean"), nmae=("mae_trainstd", "mean"))
    feature_asset.groupby(["feature", "method"], as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"),
        mean_asset_nmae=("nmae", "mean")).to_csv(
            output_dir / "feature_method.csv", index=False, float_format="%.17g")
    fold_asset = condition.groupby(["ticker", "fold", "method"],
                                   as_index=False).agg(nrmse=("nrmse", "mean"))
    fold_asset.groupby(["fold", "method"], as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"),
        median_asset_nrmse=("nrmse", "median")).to_csv(
            output_dir / "fold_method.csv", index=False, float_format="%.17g")
    short = condition.loc[condition.block_length.isin((1, 2))]
    short_asset = short.groupby(["ticker", "method"], as_index=False).agg(
        nrmse=("nrmse", "mean"), nmae=("nmae", "mean"))
    short_summary = short_asset.groupby("method", as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"),
        median_asset_nrmse=("nrmse", "median"),
        mean_asset_nmae=("nmae", "mean"))
    short_summary.to_csv(output_dir / "short_gap_method.csv", index=False,
                         float_format="%.17g")
    all_wide = asset.pivot(index="ticker", columns="method", values="nrmse")
    short_wide = short_asset.pivot(index="ticker", columns="method", values="nrmse")
    paired = []
    for scope, wide in (("all_blocks", all_wide), ("blocks_1_2", short_wide)):
        for ticker, row in wide.iterrows():
            for method in ALL_METHODS:
                if method != "itri_w1_p5":
                    paired.append({"scope": scope, "ticker": ticker,
                                   "comparator": method,
                                   "itri_nrmse": row["itri_w1_p5"],
                                   "comparator_nrmse": row[method],
                                   "itri_minus_comparator": row["itri_w1_p5"] - row[method]})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output_dir / "paired_asset_differences.csv", index=False,
                        float_format="%.17g")
    win_summary = paired_frame.groupby(["scope", "comparator"], as_index=False).agg(
        itri_wins=("itri_minus_comparator", lambda x: int((x < 0).sum())),
        itri_losses=("itri_minus_comparator", lambda x: int((x > 0).sum())),
        mean_difference=("itri_minus_comparator", "mean"),
        median_difference=("itri_minus_comparator", "median"))
    win_summary.to_csv(output_dir / "paired_win_counts.csv", index=False,
                       float_format="%.17g")
    region_asset = asset.groupby(["region", "method"], as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"), asset_count=("ticker", "nunique"))
    region_asset.to_csv(output_dir / "region_method.csv", index=False,
                        float_format="%.17g")
    runtime = condition.groupby("method", as_index=False).agg(
        total_fit_and_impute_seconds=("fit_and_impute_seconds", "sum"),
        median_condition_seconds=("fit_and_impute_seconds", "median"))
    runtime.to_csv(output_dir / "runtime_method.csv", index=False,
                   float_format="%.17g")
    overview = asset.groupby("method", as_index=False).agg(
        mean_asset_nrmse=("nrmse", "mean"),
        median_asset_nrmse=("nrmse", "median"),
        mean_asset_nmae=("nmae", "mean"))
    overview.to_csv(output_dir / "overall_method.csv", index=False,
                    float_format="%.17g")
    audit = {"status": "PASS", "assets": data.ticker.nunique(),
             "folds": sorted(data.fold.unique().tolist()),
             "methods": list(ALL_METHODS),
             "feature_rows": len(data), "condition_method_rows": len(condition),
             "asset_method_rows": len(asset),
             "validation_feature_rows": len(validation),
             "classical_fingerprint": classical_meta["run_fingerprint"],
             "neural_fingerprint": neural_meta["run_fingerprint"],
             "unit_of_analysis": "asset (conditions and folds averaged within asset)",
             "inferential_tests": "none; ETF dependence not assumed away"}
    (output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# External-validation results (descriptive)", "",
             "The primary endpoint is feature-macro NRMSE. Results below first "
             "average conditions and folds within each asset, then summarize "
             "the 34 assets. They do not treat masks or ETFs as independent markets.",
             "", "## All block lengths", "",
             markdown_table(overview.sort_values("mean_asset_nrmse")),
             "", "## One- and two-day blocks", "",
             markdown_table(short_summary.sort_values("mean_asset_nrmse")),
             "", "## Paired asset win counts", "",
             markdown_table(win_summary),
             "", "## Validation–test diagnostic", "",
             "Validation selected neural epochs and is not a second held-out test.",
             "", markdown_table(gap_summary),
             "", "No causal or population-level significance claim follows "
             "from these descriptive comparisons."]
    (output_dir / "paper_tables.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    return audit


def run(classical_source: Path, neural_source: Path, output_dir: Path) -> dict:
    if not classical_source.is_file():
        raise ValueError("classical results must be the exact original ZIP")
    classical, _, classical_meta = load_checked(classical_source, CLASSICAL)
    neural, neural_marker, neural_meta = load_checked(neural_source, NEURAL)
    validation = load_validation_checked(neural_source, neural_marker, neural_meta)
    if neural_meta["classical_zip_sha256"] != sha256(classical_source.read_bytes()):
        raise ValueError("neural experiment used a different classical result ZIP")
    data = reconcile(classical, neural, classical_meta, neural_meta)
    return summarize(data, validation, output_dir, classical_meta, neural_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.classical, args.neural, args.out), indent=2))
