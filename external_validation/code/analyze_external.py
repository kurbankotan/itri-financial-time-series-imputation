#!/usr/bin/env python3
"""Every external-validation table of the manuscript, recomputed from the public feature-level rows.

Unit of analysis: asset (conditions and folds averaged within each ETF).
Outputs: external_validation/tables/*.csv and tables/TABLES.md.
Run from any directory: python external_validation/code/analyze_external.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_paths as RP
import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = RP.TABLES; OUT.mkdir(exist_ok=True)
KEY = ["ticker", "fold", "block_length", "rate", "seed"]
V1_METHODS = ["linear", "pchip", "knn5", "chained_ridge", "itri_w1_p5", "brits", "saits"]
LABEL = {"linear": "Linear interpolation", "pchip": "PCHIP", "knn5": "KNN (k=5)", "chained_ridge": "Chained ridge",
         "itri_w1_p5": "ITRI w=1, p=5 (prespecified)", "brits": "BRITS (v1)", "saits": "SAITS (v1)",
         "saits_winnorm_V1": "SAITS + window norm. (30 ep.)", "saits_winnorm_V2": "SAITS + window norm. (200 ep.)",
         "brits_winnorm_V1": "BRITS + window norm. (30 ep.)"}
md = []

def table(df: pd.DataFrame, title: str, fname: str, floatfmt="{:.4f}"):
    df.to_csv(OUT / fname, index=False)
    md.append(f"\n### {title}\n")
    cols = list(df.columns)
    md.append("| " + " | ".join(cols) + " |"); md.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in df.itertuples(index=False):
        md.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")

# ---------------- load v1 test/validation feature rows
test = pd.read_csv(RP.V1_TEST)
val = pd.read_csv(RP.V1_VALIDATION)
assert len(test) == 192780 and len(val) == 192780
# amended neural rows (if present)
amended = pd.read_csv(RP.WINDOWNORM_SAITS) if RP.WINDOWNORM_SAITS.exists() else pd.DataFrame()
if len(amended):
    am_test = amended[amended.partition == "test"]; am_val = amended[amended.partition == "validation"]
    print("amended rows:", len(amended), "methods:", sorted(amended.method.unique()),
          "asset-folds:", am_test.groupby("method").apply(lambda g: g[["ticker", "fold"]].drop_duplicates().shape[0]).to_dict())
    test = pd.concat([test, am_test[test.columns.intersection(am_test.columns)]], ignore_index=True)
    val = pd.concat([val, am_val[val.columns.intersection(am_val.columns)]], ignore_index=True)
# sweep rows
sw_t = RP.read(RP.SWEEP_TEST); sw_v = RP.read(RP.SWEEP_VALIDATION)

def cond_level(df):
    return df.groupby(KEY + ["method"])[["rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]].mean().reset_index()

def asset_level(c):
    return c.groupby(["ticker", "method"])[["rmse_trainstd", "mae_trainstd"]].mean().reset_index()

def complete_methods(c):
    """Only methods with the full 5,508-condition grid (asset unit needs completeness)."""
    n = c.groupby("method").size()
    return [m for m in n.index if n[m] == 5508]

ct, cv = cond_level(test), cond_level(val)
methods = [m for m in V1_METHODS + ["saits_winnorm_V1", "saits_winnorm_V2", "brits_winnorm_V1"] if m in complete_methods(ct)]
methods_v1 = [m for m in V1_METHODS if m in methods]
partial = {m: int(ct[ct.method == m].shape[0]) for m in ct.method.unique() if m not in methods}
print("complete methods:", methods, "| partial:", partial)
at = asset_level(ct).pivot(index="ticker", columns="method", values="rmse_trainstd")
at_mae = asset_level(ct).pivot(index="ticker", columns="method", values="mae_trainstd")
av = asset_level(cv).pivot(index="ticker", columns="method", values="rmse_trainstd")

# ---------------- Table E1: primary external results
rows = []
for m in methods_v1:
    d = at["itri_w1_p5"] - at[m]
    rows.append({"Method": LABEL[m], "Mean asset NRMSE": at[m].mean(), "Median asset NRMSE": at[m].median(),
                 "Mean asset NMAE": at_mae[m].mean(), "Validation NRMSE": av[m].mean(),
                 "ITRI better (of 34)": "" if m == "itri_w1_p5" else int((d < 0).sum()),
                 "Median seconds per condition": float(ct[ct.method == m].fit_and_impute_seconds.median())})
E1 = pd.DataFrame(rows); table(E1, "E1 External validation: all block lengths (asset-level)", "E1_external_primary.csv")

# ---------------- Table E2: by block length
rows = []
for b, g in ct.groupby("block_length"):
    a = asset_level(g).pivot(index="ticker", columns="method", values="rmse_trainstd")
    r = {"Block length": b}
    for m in methods_v1:
        r[LABEL[m]] = a[m].mean()
    r["ITRI better than linear (of 34)"] = int(((a["itri_w1_p5"] - a["linear"]) < 0).sum())
    rows.append(r)
E2 = pd.DataFrame(rows); table(E2, "E2 External validation by block length (asset-level mean NRMSE)", "E2_external_by_block.csv")

# ---------------- Table E3: per feature
fe = test[test.method.isin(methods)].groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
E3 = fe[methods_v1].rename(columns=LABEL).reset_index().rename(columns={"feature": "Feature"})
table(E3, "E3 External validation by feature (asset-level mean NRMSE)", "E3_external_by_feature.csv")

# ---------------- Table E4: fold / validation-test diagnostic
rows = []
for m in methods_v1:
    r = {"Method": LABEL[m], "Validation NRMSE": av[m].mean(), "Test NRMSE": at[m].mean(), "Test minus validation": at[m].mean() - av[m].mean()}
    for f, g in ct.groupby("fold"):
        a = asset_level(g).pivot(index="ticker", columns="method", values="rmse_trainstd")
        r[f"Test fold {f}"] = a[m].mean()
    rows.append(r)
E4 = pd.DataFrame(rows); table(E4, "E4 Validation-test and fold diagnostic", "E4_fold_diagnostic.csv")

# ---------------- Table S1: sweep grid (test, validation, wins)
WINDOWS = (1, 2, 3, 4, 5, 7, 10); PASSES = (1, 2, 3, 5)
cst, csv_ = cond_level(sw_t), cond_level(sw_v)
ast_ = asset_level(cst).pivot(index="ticker", columns="method", values="rmse_trainstd")
asv = asset_level(csv_).pivot(index="ticker", columns="method", values="rmse_trainstd")
assert np.allclose(ast_["itri_w1_p5"], at["itri_w1_p5"], atol=1e-12)
rows = []
for p in PASSES:
    r = {"Passes p": p}
    for w in WINDOWS:
        r[f"w={w} test"] = ast_[f"itri_w{w}_p{p}"].mean()
    rows.append(r)
S1 = pd.DataFrame(rows); table(S1, "S1 Exploratory sweep: test asset-level mean NRMSE (linear = %.4f)" % at["linear"].mean(), "S1_sweep_test_grid.csv")
rows = []
for p in PASSES:
    r = {"Passes p": p}
    for w in WINDOWS:
        r[f"w={w} val"] = asv[f"itri_w{w}_p{p}"].mean()
    rows.append(r)
S1v = pd.DataFrame(rows); table(S1v, "S1v Exploratory sweep: validation asset-level mean NRMSE (linear = %.4f)" % av["linear"].mean(), "S1v_sweep_validation_grid.csv")
rows = []
for p in PASSES:
    r = {"Passes p": p}
    for w in WINDOWS:
        r[f"w={w} wins"] = int(((ast_[f"itri_w{w}_p{p}"] - at["linear"]) < 0).sum())
    rows.append(r)
S1w = pd.DataFrame(rows); table(S1w, "S1w Exploratory sweep: assets (of 34) where ITRI beats linear interpolation on test", "S1w_sweep_wins.csv", "{:d}")
vr = {f"itri_w{w}_p{p}": asv[f"itri_w{w}_p{p}"].mean() for w in WINDOWS for p in PASSES}
tr = {f"itri_w{w}_p{p}": ast_[f"itri_w{w}_p{p}"].mean() for w in WINDOWS for p in PASSES}
vsel = min(vr, key=vr.get); tbest = min(tr, key=tr.get)
rank_v = sorted(vr, key=vr.get).index("itri_w1_p5") + 1; rank_t = sorted(tr, key=tr.get).index("itri_w1_p5") + 1
rho_vt = spearmanr(list(vr.values()), [tr[k] for k in vr]).statistic
sweep_summary = {"external_validation_selected": vsel, "its_test": tr[vsel], "its_val": vr[vsel],
                 "post_hoc_best_test": tbest, "its_test_value": tr[tbest],
                 "w1p5_rank_validation": rank_v, "w1p5_rank_test": rank_t, "spearman_val_test_28": rho_vt,
                 "w1p1_test": tr["itri_w1_p1"], "w3p3_test": tr["itri_w3_p3"], "linear_test": at["linear"].mean(),
                 "max_grid_test": max(tr.values()), "argmax": max(tr, key=tr.get)}
json.dump(sweep_summary, open(OUT / "sweep_summary.json", "w"), indent=2)
md.append("\nSweep summary: " + json.dumps(sweep_summary, indent=1))

# ---------------- Table S2: sweep by block for selected configs + wins
rows = []
for b, g in cst.groupby("block_length"):
    a = asset_level(g).pivot(index="ticker", columns="method", values="rmse_trainstd")
    lin = asset_level(ct[(ct.block_length == b) & (ct.method == "linear")]).pivot(index="ticker", columns="method", values="rmse_trainstd")["linear"]
    r = {"Block length": b, "Linear": lin.mean()}
    for m in ["itri_w1_p1", "itri_w1_p5", "itri_w3_p3", vsel]:
        r[m.replace("itri_", "ITRI ")] = a[m].mean()
    r["w1p1 wins"] = int(((a["itri_w1_p1"] - lin) < 0).sum()); r["w1p5 wins"] = int(((a["itri_w1_p5"] - lin) < 0).sum())
    r[f"{vsel.replace('itri_', '')} wins"] = int(((a[vsel] - lin) < 0).sum())
    rows.append(r)
S2 = pd.DataFrame(rows); table(S2, "S2 Exploratory sweep by block length", "S2_sweep_by_block.csv")

# ---------------- Table S3: per-feature for sweep configs
fe2 = sw_t[sw_t.method.isin(["itri_w1_p1", "itri_w1_p5", "itri_w3_p3", vsel, "itri_w10_p5"])].groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
lin_fe = test[test.method == "linear"].groupby(["ticker", "feature"]).rmse_trainstd.mean().groupby("feature").mean()
S3 = fe2.copy(); S3.insert(0, "linear", lin_fe)
S3 = S3[["linear", "itri_w1_p1", "itri_w1_p5", "itri_w3_p3", vsel, "itri_w10_p5"]].reset_index().rename(columns={"feature": "Feature"})
table(S3, "S3 Exploratory sweep: per-feature test NRMSE", "S3_sweep_by_feature.csv")

# ---------------- Table S4: w=1 coefficients
coef = pd.read_csv(RP.W1_COEFFICIENTS); coef["sum"] = coef.coef_lag1 + coef.coef_lead1
S4 = coef.groupby("feature")[["coef_lag1", "coef_lead1", "sum", "intercept"]].agg(["mean", "std"])
S4.columns = [f"{a} {b}" for a, b in S4.columns]; S4 = S4.reset_index().rename(columns={"feature": "Feature"})
table(S4, "S4 Fitted w=1 coefficients over 5,508 training fits", "S4_w1_coefficients.csv")

# ---------------- Table Z1: level shift
z = pd.read_csv(RP.LEVEL_SHIFT)
af = ct.groupby(["ticker", "fold", "method"]).rmse_trainstd.mean().unstack("method")
m = z.set_index(["ticker", "fold"]).join(af)
rows = []
for meth in methods:
    rho = spearmanr(m["test_mean_abs_z_price"], m[meth]).statistic
    rows.append({"Method": LABEL[meth], "Spearman rho (102 asset-folds)": rho})
Z1 = pd.DataFrame(rows); table(Z1, "Z1 Spearman correlation between test-period displacement (mean |z| of prices under training scaling) and test NRMSE", "Z1_level_shift_spearman.csv")
m["zbin"] = pd.cut(m["test_mean_abs_z_price"], [0, 1, 2, 3, 4, 100], labels=["< 1", "1 to 2", "2 to 3", "3 to 4", "> 4"])
rows = []
for b, g in m.groupby("zbin", observed=True):
    r = {"Mean |z| of test prices": b, "Asset-folds": len(g)}
    for meth in methods:
        r[LABEL[meth]] = g[meth].mean()
    rows.append(r)
Z2 = pd.DataFrame(rows); table(Z2, "Z2 Test NRMSE by displacement bin (asset-fold units)", "Z2_level_shift_bins.csv")
rows = []
for f, g in m.groupby("fold"):
    r = {"Fold": f, "Mean |z| test prices": g["test_mean_abs_z_price"].mean(), "Share |z|>2": g["test_frac_out2_price"].mean(),
         "Mean |z| validation prices": g["val_mean_abs_z_price"].mean()}
    rows.append(r)
Z3 = pd.DataFrame(rows); table(Z3, "Z3 Displacement by fold", "Z3_level_shift_by_fold.csv")
m.reset_index().to_csv(OUT / "level_shift_joined.csv", index=False)

# ---------------- Amended neural results (if complete)
am_methods = [x for x in ["saits_winnorm_V1", "saits_winnorm_V2", "brits_winnorm_V1"] if x in methods]
if am_methods:
    rows = []
    for mth in ["linear", "itri_w1_p5", vsel if False else "itri_w1_p5", "chained_ridge", "saits", "brits"] + am_methods:
        if mth not in at: continue
        d = at["itri_w1_p5"] - at[mth]
        rows.append({"Method": LABEL[mth], "Mean asset NRMSE": at[mth].mean(), "Median": at[mth].median(),
                     "Mean asset NMAE": at_mae[mth].mean(), "Validation NRMSE": av[mth].mean(),
                     "ITRI better (of 34)": "" if mth == "itri_w1_p5" else int((d < 0).sum()),
                     "Linear better (of 34)": int(((at["linear"] - at[mth]) < 0).sum()),
                     "Median seconds": float(ct[ct.method == mth].fit_and_impute_seconds.median())})
    A1 = pd.DataFrame(rows).drop_duplicates("Method"); table(A1, "A1 Amended (exploratory) neural results", "A1_amended_neural.csv")
    rows = []
    for b, g in ct.groupby("block_length"):
        a = asset_level(g).pivot(index="ticker", columns="method", values="rmse_trainstd")
        r = {"Block length": b}
        for mth in ["linear", "itri_w1_p5", "chained_ridge", "saits", "brits"] + am_methods:
            r[LABEL[mth]] = a[mth].mean()
        for mth in am_methods:
            r[f"{LABEL[mth]} better than ITRI (of 34)"] = int(((a[mth] - a["itri_w1_p5"]) < 0).sum())
        rows.append(r)
    A2 = pd.DataFrame(rows); table(A2, "A2 Amended neural results by block length", "A2_amended_by_block.csv")
    # per-feature for amended
    fe_am = test[test.method.isin(["linear", "itri_w1_p5", "chained_ridge", "saits"] + am_methods)].groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
    A3 = fe_am[["linear", "itri_w1_p5", "chained_ridge", "saits"] + am_methods].rename(columns=LABEL).reset_index().rename(columns={"feature": "Feature"})
    table(A3, "A3 Amended neural results by feature", "A3_amended_by_feature.csv")
(OUT / "TABLES.md").write_text("# External validation tables (generated by analyze_external.py)\n" + "\n".join(md) + "\n")
print("\n".join(md))
