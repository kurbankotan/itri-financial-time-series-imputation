#!/usr/bin/env python3
"""Verify the integrity and headline statistics of the published ITRI results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED = json.loads((ROOT / "metadata" / "expected_metrics.json").read_text())


def close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: expected {expected:.15g}, got {actual:.15g}")


def load_grid(name: str, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / name)
    if len(frame) != expected_rows:
        raise AssertionError(f"{name}: expected {expected_rows} rows, got {len(frame)}")
    metric_columns = ["mae", "rmse", "nrmse"]
    if not np.isfinite(frame[metric_columns].to_numpy(float)).all():
        raise AssertionError(f"{name}: nonfinite reconstruction metric")
    keys = ["market", "block_length", "rate", "seed", "window", "passes"]
    if frame.duplicated(keys).any():
        raise AssertionError(f"{name}: duplicate experimental condition")
    counts = frame.groupby(["window", "passes"]).size()
    expected_conditions = EXPECTED["expected_rows"]["conditions_per_configuration"]
    if not (counts == expected_conditions).all():
        raise AssertionError(f"{name}: not every configuration has {expected_conditions} conditions")
    return frame


validation = load_grid(
    "validation_grid.csv", EXPECTED["expected_rows"]["validation_grid"]
)
test = load_grid("test_grid.csv", EXPECTED["expected_rows"]["test_grid"])
interpolation = pd.read_csv(RESULTS / "interpolation_baseline_test.csv")
if len(interpolation) != EXPECTED["expected_rows"]["interpolation_baseline_test"]:
    raise AssertionError("interpolation_baseline_test.csv: unexpected row count")
if not np.isfinite(interpolation[["mae", "rmse", "nrmse"]].to_numpy(float)).all():
    raise AssertionError("interpolation_baseline_test.csv: nonfinite metric")

selected = json.loads((RESULTS / "selected_configuration.json").read_text())
expected_selected = EXPECTED["selected_configuration"]
if (selected["window"], selected["passes"]) != (
    expected_selected["window"], expected_selected["passes"]
):
    raise AssertionError("selected_configuration.json: unexpected window or pass count")
close(
    float(selected["mean_validation_nrmse"]),
    expected_selected["mean_validation_nrmse"],
    "selected mean validation NRMSE",
)


def means(frame: pd.DataFrame, window: int, passes: int) -> dict[str, float]:
    rows = frame[(frame.window == window) & (frame.passes == passes)]
    if len(rows) != EXPECTED["expected_rows"]["conditions_per_configuration"]:
        raise AssertionError(f"w={window}, p={passes}: unexpected condition count")
    return rows[["nrmse", "mae", "rmse"]].mean().to_dict()


comparisons = {
    "selected_w1_p5": means(test, 1, 5),
    "inherited_w3_p3": means(test, 3, 3),
    "interpolation_only": interpolation[["nrmse", "mae", "rmse"]].mean().to_dict(),
}

for label, actual_metrics in comparisons.items():
    for metric, expected_value in EXPECTED["published_test_means"][label].items():
        close(float(actual_metrics[metric]), float(expected_value), f"{label} {metric}")

print("Selected configuration: w=1, p=5")
for label, metric_values in comparisons.items():
    print(
        f"{label}: NRMSE={metric_values['nrmse']:.6f}, "
        f"MAE={metric_values['mae']:.6f}, RMSE={metric_values['rmse']:.6f}"
    )
print("All published-result checks passed.")
