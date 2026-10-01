#!/usr/bin/env python3
"""Exploratory amendment v2, BRITS subset (fold A, mask seed 11; 34 assets x 18 conditions = 612).
All methods are compared on exactly the same subset of conditions (asset-level means): manuscript Table 21.
Run from any directory: python external_validation/code/analyze_brits_subset.py
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_paths as RP
from scipy.stats import spearmanr
AN = RP.TABLES; AN.mkdir(exist_ok=True)
KEY = ["ticker", "fold", "block_length", "rate", "seed"]
LABEL = {"linear": "Linear interpolation", "pchip": "PCHIP", "knn5": "KNN (k=5)", "chained_ridge": "Chained ridge",
         "itri_w1_p5": "ITRI w=1, p=5 (prespecified)", "brits": "BRITS (v1)", "saits": "SAITS (v1)",
         "saits_winnorm_V1": "SAITS + window norm. (30 ep.)", "saits_winnorm_V2": "SAITS + window norm. (200 ep.)",
         "brits_winnorm_V1": "BRITS + window norm. (30 ep.)"}
def load(path):
    return pd.read_csv(path, float_precision="round_trip")
v1 = pd.read_csv(RP.V1_TEST, float_precision="round_trip")
am = load(RP.WINDOWNORM_SAITS); am = am[am.partition == "test"]
br = load(RP.WINDOWNORM_BRITS_SUBSET); br = br[br.partition == "test"]
cols = KEY + ["feature", "method", "rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]
T = pd.concat([v1[cols], am[cols], br[cols]], ignore_index=True)
T["rate"] = T.rate.round(2)
sub_keys = br[KEY].drop_duplicates().assign(rate=lambda d: d.rate.round(2))
assert len(sub_keys) == 612
T = T.merge(sub_keys, on=KEY)
cond = T.groupby(KEY + ["method"])[["rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]].mean().reset_index()
n = cond.groupby("method").size()
assert (n == 612).all(), n
at = cond.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack()
am_ = cond.groupby(["ticker", "method"]).mae_trainstd.mean().unstack()
order = ["linear", "pchip", "chained_ridge", "knn5", "itri_w1_p5", "brits", "saits", "brits_winnorm_V1", "saits_winnorm_V1", "saits_winnorm_V2"]
rows = []
for m in order:
    rows.append({"Method": LABEL[m], "Mean asset NRMSE": at[m].mean(), "Median": at[m].median(), "Mean asset NMAE": am_[m].mean(),
                 "Better than ITRI (of 34)": "" if m == "itri_w1_p5" else int((at[m] < at["itri_w1_p5"]).sum()),
                 "Better than BRITS + norm. (of 34)": "" if m == "brits_winnorm_V1" else int((at[m] < at["brits_winnorm_V1"]).sum()),
                 "Median seconds": float(cond[cond.method == m].fit_and_impute_seconds.median())})
B1 = pd.DataFrame(rows); B1.to_csv(AN / "B1_brits_subset.csv", index=False)
blk = cond.groupby(["block_length", "method"]).rmse_trainstd.mean().unstack()  # equal weights: 34 assets x 3 rates per block
rows = []
for b in sorted(cond.block_length.unique()):
    g = cond[cond.block_length == b]; a = g.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack()
    r = {"Block length": b}
    for m in ["linear", "itri_w1_p5", "chained_ridge", "brits", "brits_winnorm_V1", "saits_winnorm_V1", "saits_winnorm_V2"]:
        r[LABEL[m]] = a[m].mean()
    r["BRITS + norm. better than ITRI (of 34)"] = int((a["brits_winnorm_V1"] < a["itri_w1_p5"]).sum())
    rows.append(r)
B2 = pd.DataFrame(rows); B2.to_csv(AN / "B2_brits_subset_by_block.csv", index=False)
fe = T.groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
B3 = fe[["linear", "itri_w1_p5", "brits", "brits_winnorm_V1", "saits_winnorm_V1", "saits_winnorm_V2"]].rename(columns=LABEL).reset_index().rename(columns={"feature": "Feature"})
B3.to_csv(AN / "B3_brits_subset_by_feature.csv", index=False)
# displacement correlation on fold A (34 asset-folds)
z = pd.read_csv(RP.LEVEL_SHIFT); zA = z[z.fold == "A"].set_index("ticker")
rho = {m: spearmanr(zA.loc[at.index, "test_mean_abs_z_price"], at[m]).statistic for m in order}
pd.DataFrame([{"Method": LABEL[m], "Spearman rho (34 fold-A assets)": rho[m]} for m in order]).to_csv(AN / "B4_brits_subset_spearman.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(B1.to_string(index=False)); print(); print(B2.to_string(index=False)); print(); print(B3.to_string(index=False))
print(); print({LABEL[m]: round(v, 3) for m, v in rho.items()})
