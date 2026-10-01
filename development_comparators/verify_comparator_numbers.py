#!/usr/bin/env python3
"""Check the matched-mask comparator numbers of the development study (Sections 4.4, 5.6, 5.7;
Tables 9 and 10 of the manuscript) against the published result files. No market data needed.
Run from any directory: python development_comparators/verify_comparator_numbers.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DEV = HERE.parent / "results"
KEY = ["market", "block_length", "rate", "seed"]
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <- {detail}"))
    if not cond:
        fails.append(name)


r6 = lambda v: round(float(v), 6)
test = pd.read_csv(RES / "additional_baselines_test.csv")
val = pd.read_csv(RES / "additional_baselines_validation.csv")
grid_t = pd.read_csv(DEV / "test_grid.csv")
grid_v = pd.read_csv(DEV / "validation_grid.csv")
interp = pd.read_csv(DEV / "interpolation_baseline_test.csv")
itri_t = grid_t[(grid_t.window == 1) & (grid_t.passes == 5)]
itri_v = grid_v[(grid_v.window == 1) & (grid_v.passes == 5)]
meta = json.loads((RES / "additional_baselines_metadata.json").read_text())

# locked settings chosen on validation only
check("validation selects KNN k=10 and chained equations max_iter=3", meta["selected_on_validation"] == {"PCHIP": 0, "KNN": 10, "MICE": 3})
vmean = val.groupby(["method", "parameter"]).nrmse.mean()
check("KNN k=10 and MICE 3 have the lowest validation NRMSE in their grids",
      vmean["KNN"].idxmin() == 10 and vmean["MICE"].idxmin() == 3)
lock = {"PCHIP": 0, "KNN": 10, "MICE": 3}
check("test rows use only the locked settings, 108 conditions each",
      all((test[test.method == m].parameter == p).all() and (test.method == m).sum() == 108 for m, p in lock.items()))

# Table 9
expected = {  # method: (validation NRMSE, test NRMSE, test MAE, test RMSE, ITRI better of 108)
    "PCHIP": (0.742948, 0.240863, 0.411878, 0.726701, 108),
    "MICE": (0.679285, 0.441629, 1.237505, 2.579421, 90),
    "KNN": (0.934847, 1.075055, 3.670148, 6.130940, 108)}
for m, (v, t, mae, rmse, wins) in expected.items():
    tv = test[test.method == m]
    vv = val[(val.method == m) & (val.parameter == lock[m])]
    d = itri_t[KEY + ["nrmse"]].merge(tv[KEY + ["nrmse"]], on=KEY, suffixes=("_itri", "_cmp"))
    check(f"Table 9 {m}: {v} / {t} / {mae} / {rmse}, ITRI better in {wins} of 108",
          (r6(vv.nrmse.mean()), r6(tv.nrmse.mean()), r6(tv.mae.mean()), r6(tv.rmse.mean())) == (v, t, mae, rmse)
          and len(d) == 108 and int((d.nrmse_itri < d.nrmse_cmp).sum()) == wins)
check("Table 9 ITRI w=1, p=5: 0.464934 / 0.198094 / 0.362097 / 0.609216",
      (r6(itri_v.nrmse.mean()), r6(itri_t.nrmse.mean()), r6(itri_t.mae.mean()), r6(itri_t.rmse.mean())) == (0.464934, 0.198094, 0.362097, 0.609216))
di = itri_t[KEY + ["nrmse"]].merge(interp, on=KEY, suffixes=("_itri", "_lin"))
check("Table 9 linear interpolation: test 0.232413 / 0.401498 / 0.701797, ITRI better in 108 of 108",
      (r6(interp.nrmse.mean()), r6(interp.mae.mean()), r6(interp.rmse.mean())) == (0.232413, 0.401498, 0.701797)
      and int((di.nrmse_itri < di.nrmse_lin).sum()) == 108)
summ = pd.read_csv(RES / "manuscript_comparator_summary.csv").set_index(["split", "method"])
check("Table 9 linear interpolation: validation 0.715296", r6(summ.loc[("validation", "Linear interpolation"), "mean_nrmse"]) == 0.715296)
red = {m: round(100 * (test[test.method == m].nrmse.mean() - itri_t.nrmse.mean()) / test[test.method == m].nrmse.mean(), 1) for m in expected}
red["linear"] = round(100 * (interp.nrmse.mean() - itri_t.nrmse.mean()) / interp.nrmse.mean(), 1)
check("Section 5.6 reductions 14.8%, 17.8%, 55.1%, 81.6%", (red["linear"], red["PCHIP"], red["MICE"], red["KNN"]) == (14.8, 17.8, 55.1, 81.6))

# Section 5.6 text: market and gap-length pattern
dd = itri_t[KEY + ["nrmse"]].merge(test[KEY + ["method", "nrmse"]], on=KEY, suffixes=("_itri", "_cmp"))
dd["diff"] = dd.nrmse_itri - dd.nrmse_cmp
mk = dd[dd.method == "MICE"].groupby("market")["diff"].mean()
check("advantage over chained equations smaller for S&P 500 than for BIST 30", abs(mk["SP500"]) < abs(mk["BIST30"]) and (mk < 0).all())
check("advantage over PCHIP positive in both markets and every gap length",
      bool((dd[dd.method == "PCHIP"].groupby(["market", "block_length"])["diff"].mean() < 0).all()))

# Section 4.4: exact-mask reconstruction of the archived interpolation baseline
rep = json.loads((RES / "revision_integrity_report.json").read_text())
mx = rep["test_interpolation_reconstruction_max_abs_difference"]
check("reconstructed interpolation matches the archive within 1.2e-16", rep["test_interpolation_reconstruction_matches_archive"] and max(mx.values()) < 1.2e-16)
audit = json.loads((RES / "full_audit_report.json").read_text())
check("all 6,048 archived ITRI rows re-executed within tolerance", audit["audited_metric_rows"] == 6048 and audit["all_rows_match_within_tolerance"])

# Table 10: partition diagnostics
shift = pd.read_csv(RES / "validation_test_shift.csv").set_index(["market", "split"])
exp10 = {("BIST30", "validation"): (0.163187, 0.904344, 0.457876, 5.559024), ("BIST30", "test"): (0.068942, 0.613120, 0.398966, 8.924044),
         ("SP500", "validation"): (0.766682, 1.969572, 0.377937, 2.316600), ("SP500", "test"): (0.327245, 0.605311, 0.325228, 1.851111)}
check("Table 10 partition diagnostics", all(tuple(r6(shift.loc[k, c]) for c in ("mean_nrmse", "mean_rmse", "mean_mae", "mean_masked_truth_sd")) == v
                                            for k, v in exp10.items()))
check("Section 5.7 partition dates (validation Sep 2022 to May 2024, test May 2024 to Dec 2025)",
      all(shift.loc[(m, "validation"), "first_date"].startswith("2022-09") and shift.loc[(m, "test"), "last_date"] == "2025-12-31" for m in ("BIST30", "SP500")))

print("\nFAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
