#!/usr/bin/env python3
"""Verify the matched masks and summarize baseline and split diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_additional_baselines as experiment


OUT = Path(__file__).resolve().parent / "results"
ARCHIVE = experiment.ARCHIVE
KEYS = ["market", "block_length", "rate", "seed"]


def linear_interpolation_grid(split: str) -> pd.DataFrame:
    rows = []
    for market, block, rate, seed, _, observed, truth, mask in experiment.conditions(split):
        filled = (
            pd.DataFrame(observed)
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(float)
        )
        rows.append({
            "market": market, "block_length": block, "rate": rate,
            "seed": seed, **experiment.metrics(truth, filled, mask),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    original_val = pd.read_csv(ARCHIVE / "validation_grid.csv")
    original_test = pd.read_csv(ARCHIVE / "test_grid.csv")
    validation = pd.read_csv(OUT / "additional_baselines_validation.csv")
    test = pd.read_csv(OUT / "additional_baselines_test.csv")
    linear_val = linear_interpolation_grid("val")
    linear_test = linear_interpolation_grid("test")
    archived_linear = pd.read_csv(ARCHIVE / "interpolation_baseline_test.csv")

    actual = linear_test.set_index(KEYS).sort_index()
    expected = archived_linear.set_index(KEYS).sort_index()
    deviations = {
        key: float(np.max(np.abs(actual[key] - expected[key])))
        for key in ("mae", "rmse", "nrmse")
    }
    if any(value > 1e-12 for value in deviations.values()):
        raise AssertionError(f"The reconstructed test masks or metrics differ: {deviations}")

    val_itri = original_val[(original_val.window == 1) & (original_val.passes == 5)].copy()
    test_itri = original_test[(original_test.window == 1) & (original_test.passes == 5)].copy()
    comparison_rows = []
    split_rows = []
    for split_name, source, extra, linear in (
        ("validation", val_itri, validation, linear_val),
        ("test", test_itri, test, linear_test),
    ):
        records = [("ITRI w1 p5", source)]
        records.append(("Linear interpolation", linear))
        for method, group in extra.groupby("method"):
            if split_name == "validation":
                chosen_parameter = {"KNN": 10, "MICE": 3, "PCHIP": 0}[method]
                group = group[group.parameter == chosen_parameter]
            records.append((method, group))

        indexed_itri = source.set_index(KEYS)
        for method, frame in records:
            metrics = frame[["nrmse", "mae", "rmse"]].mean()
            row = {
                "split": split_name, "method": method,
                "mean_nrmse": float(metrics.nrmse),
                "mean_mae": float(metrics.mae),
                "mean_rmse": float(metrics.rmse),
                "conditions": len(frame),
            }
            if method != "ITRI w1 p5":
                indexed = frame.set_index(KEYS)
                delta = indexed_itri.nrmse - indexed.nrmse
                row["itri_better_conditions"] = int((delta < 0).sum())
                row["itri_mean_difference"] = float(delta.mean())
                row["itri_relative_nrmse_reduction_pct"] = float(
                    100 * (1 - indexed_itri.nrmse.mean() / metrics.nrmse)
                )
                row["itri_better_market_block_strata"] = int(
                    (delta.groupby(level=["market", "block_length"]).mean() < 0).sum()
                )
            comparison_rows.append(row)

        tmp = source.copy()
        tmp["masked_truth_sd"] = tmp.rmse / tmp.nrmse
        for market, group in tmp.groupby("market"):
            dates = experiment.load_market(market)["val_dates" if split_name == "validation" else "test_dates"]
            split_rows.append({
                "split": split_name, "market": market,
                "first_date": str(dates.min().date()),
                "last_date": str(dates.max().date()),
                "mean_nrmse": float(group.nrmse.mean()),
                "mean_rmse": float(group.rmse.mean()),
                "mean_mae": float(group.mae.mean()),
                "mean_masked_truth_sd": float(group.masked_truth_sd.mean()),
                "conditions": len(group),
            })

    comparisons = pd.DataFrame(comparison_rows).sort_values(["split", "mean_nrmse"])
    comparisons.to_csv(OUT / "manuscript_comparator_summary.csv", index=False)
    shift = pd.DataFrame(split_rows)
    shift.to_csv(OUT / "validation_test_shift.csv", index=False)

    inherited = original_test[(original_test.window == 3) & (original_test.passes == 3)].set_index(KEYS)
    selected = test_itri.set_index(KEYS)
    window_effect = (selected.nrmse - inherited.nrmse).rename("mean_paired_nrmse_difference")
    window_effect.groupby(level=["market", "block_length"]).mean().reset_index().to_csv(
        OUT / "selected_vs_inherited_by_market_block.csv", index=False
    )

    report = {
        "test_interpolation_reconstruction_max_abs_difference": deviations,
        "test_interpolation_reconstruction_matches_archive": True,
        "mask_construction_matches_archived_condition_metrics": True,
        "same_underlying_series_reused_across_conditions": True,
        "markets": 2, "constructed_conditions_per_configuration": 108,
        "market_by_block_descriptive_strata": 12,
        "note": "Condition-level Wilcoxon, Friedman and ordinary bootstrap calculations in the original manuscript do not establish population-level significance because repeated masks and factor settings reuse the same two time series.",
    }
    (OUT / "revision_integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(comparisons.to_string(index=False))
    print(shift.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
