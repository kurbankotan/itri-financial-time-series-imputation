#!/usr/bin/env python3
"""Cross-hardware reproducibility check for amendment v2 (Section 5.10 of the manuscript).

Four amended pieces (EWC and EWZ, fold A, mask seed 11; BRITS V1 and SAITS V2) were computed
both on a CPU and on the A100 used for the reported runs. This script compares the two on the
test partition and writes tables/R1_cross_hardware_check.csv.
Run from any directory: python external_validation/code/cross_hardware_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_paths as RP

KEY = ["ticker", "fold", "block_length", "rate", "seed"]


def compute() -> pd.DataFrame:
    cpu_all = pd.read_csv(RP.CPU_PIECES, float_precision="round_trip")
    gpu_all = pd.concat([pd.read_csv(RP.WINDOWNORM_BRITS_SUBSET, float_precision="round_trip"),
                         pd.read_csv(RP.WINDOWNORM_SAITS, float_precision="round_trip")], ignore_index=True)
    rows = []
    for (ticker, fold, method), cpu in cpu_all.groupby(["ticker", "fold", "method"], sort=False):
        variant = method.rsplit("_", 1)[-1]
        piece = f"{ticker}_{fold}_{method.split('_')[0]}_{variant}.csv"
        gpu = gpu_all[(gpu_all.ticker == ticker) & (gpu_all.fold == fold) & (gpu_all.method == method)]
        c, g = cpu[cpu.partition == "test"], gpu[gpu.partition == "test"]
        both = c.merge(g, on=KEY + ["feature"], suffixes=("_cpu", "_gpu"))
        cm = c.groupby(KEY).rmse_trainstd.mean()
        gm = g.groupby(KEY).rmse_trainstd.mean().reindex(cm.index)
        rows.append({"piece": piece, "conditions": len(cm),
                     "max_abs_feature_diff": (both.rmse_trainstd_cpu - both.rmse_trainstd_gpu).abs().max(),
                     "max_abs_condition_macro_diff": (cm - gm).abs().max(),
                     "mean_cpu": cm.mean(), "mean_gpu": gm.mean()})
    R = pd.DataFrame(rows)
    order = R.piece.str.split("_").str[2]  # BRITS pieces first, then SAITS; tickers alphabetical
    return R.assign(_m=order).sort_values(["_m", "piece"]).drop(columns="_m").reset_index(drop=True)


if __name__ == "__main__":
    R = compute()
    RP.TABLES.mkdir(exist_ok=True)
    R.to_csv(RP.TABLES / "R1_cross_hardware_check.csv", index=False)
    print(R.to_string(index=False))
